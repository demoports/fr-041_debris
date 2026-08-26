// Standalone oracle for the enabled (#if 1) Bitmap_Perlin path in the pinned
// werkkzeug3/genbitmap.cpp. It deliberately has no project dependencies.
// Float helpers force the single-precision, round-to-nearest-even arithmetic
// selected by the released x86 player's sFloatFix control word.

#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

static float fadd(float a, float b) { volatile float result = a + b; return result; }
static float fsub(float a, float b) { volatile float result = a - b; return result; }
static float fmul(float a, float b) { volatile float result = a * b; return result; }
static float fdiv(float a, float b) { volatile float result = a / b; return result; }
static int32_t sFtol(float value) { return static_cast<int32_t>(std::nearbyintf(value)); }

static uint32_t randomSeed = 0x74382381u;

static uint32_t getRnd() {
  const uint32_t first = randomSeed * 0x343fdu + 0x269ec3u;
  const uint32_t second = first * 0x343fdu + 0x269ec3u;
  randomSeed = second;
  return ((second >> 10) & 0xffffu) | ((first << 6) & 0xffff0000u);
}

static void setRndSeed(int32_t seed) {
  const int32_t mixed = seed + seed * 17 + seed * 121 + seed * 121 / 17;
  randomSeed = static_cast<uint32_t>(mixed);
  getRnd(); randomSeed ^= static_cast<uint32_t>(mixed);
  getRnd(); randomSeed ^= static_cast<uint32_t>(mixed);
  getRnd(); randomSeed ^= static_cast<uint32_t>(mixed);
  getRnd();
}

struct PerlinTables {
  std::array<std::array<float, 2>, 256> gradients{};
  std::array<uint8_t, 512> permutation{};
};

static PerlinTables makePerlinTables() {
  PerlinTables tables;
  setRndSeed(1);
  std::array<float, 256> keys{};
  for (int index = 0; index < 256; ++index) {
    keys[index] = static_cast<float>(getRnd() % 0x10000u);
    tables.permutation[index] = static_cast<uint8_t>(index);
  }
  // The source uses a stable exchange sort: equal random keys do not swap.
  for (int left = 0; left < 255; ++left) for (int right = left + 1; right < 256; ++right) {
    if (keys[left] > keys[right]) {
      std::swap(keys[left], keys[right]);
      std::swap(tables.permutation[left], tables.permutation[right]);
    }
  }
  std::copy_n(tables.permutation.begin(), 256, tables.permutation.begin() + 256);
  for (int index = 0; index < 256;) {
    const int x = static_cast<int>(getRnd() % 0x10000u) - 0x8000;
    const int y = static_cast<int>(getRnd() % 0x10000u) - 0x8000;
    if (x * x + y * y < 0x8000 * 0x8000) {
      tables.gradients[index][0] = fdiv(static_cast<float>(x), 32768.0f);
      tables.gradients[index][1] = fdiv(static_cast<float>(y), 32768.0f);
      ++index;
    }
  }
  return tables;
}

static int powerOfTwo(int value) {
  int exponent = 0;
  while ((1 << exponent) < value) ++exponent;
  return exponent;
}

static int32_t range7fff(int32_t value) {
  if (static_cast<uint32_t>(value) < 0x7fffu) return value;
  return value < 0 ? 0 : 0x7fff;
}

static int32_t mulShift(int32_t a, int32_t b) {
  return static_cast<int32_t>((static_cast<int64_t>(a) * b) >> 16);
}

static int16_t signed16(uint16_t value) { return static_cast<int16_t>(value); }
static int16_t saturate16(int32_t value) {
  return static_cast<int16_t>(std::max(-0x8000, std::min(0x7fff, value)));
}

static uint16_t fadeChannel(uint16_t a, uint16_t b, int32_t fade) {
  const int32_t halfFade = fade >> 1;
  const int16_t weightB = saturate16(halfFade);
  const int16_t weightA = saturate16(0x8000 - halfFade);
  const int32_t productA = static_cast<int32_t>(signed16(a)) * weightA;
  const int32_t productB = static_cast<int32_t>(signed16(b)) * weightB;
  const uint16_t sum = static_cast<uint16_t>((productA >> 16) + (productB >> 16));
  return static_cast<uint16_t>(sum << 1);
}

static std::array<uint16_t, 4> packedColor(uint32_t color) {
  return {
    static_cast<uint16_t>((((color >> 16) & 255u) * 257u) >> 1),
    static_cast<uint16_t>((((color >> 8) & 255u) * 257u) >> 1),
    static_cast<uint16_t>(((color & 255u) * 257u) >> 1),
    static_cast<uint16_t>(((color >> 24) * 257u) >> 1),
  };
}

struct PerlinCase {
  const char *name;
  int width, height, frequency, octaves;
  float fadeoff;
  int seed, mode;
  float amplitude, gamma;
  uint32_t color0, color1;
};

static std::vector<uint16_t> bitmapPerlin(const PerlinCase &test, const PerlinTables &tables) {
  const auto color0 = packedColor(test.color0), color1 = packedColor(test.color1);
  const int shiftX = 16 - powerOfTwo(test.width);
  const int shiftY = 16 - powerOfTwo(test.height);
  const int seed = test.seed & 255, mode = test.mode & 3;

  std::array<int32_t, 1025> gammaTable{};
  for (int index = 0; index < 1025; ++index) {
    const double value = std::pow(static_cast<float>(index) / 1024.0f, test.gamma) * 0x8000;
    gammaTable[index] = range7fff(static_cast<int32_t>(value)) * 2;
  }

  float scaledAmplitude;
  int offset;
  if (mode & 1) {
    scaledAmplitude = fmul(test.amplitude, static_cast<float>(0x8000));
    offset = 0;
  } else {
    scaledAmplitude = fmul(test.amplitude, static_cast<float>(0x4000));
    offset = 0x4000;
  }
  const int32_t amplitudeFixed = sFtol(scaledAmplitude);

  std::array<int32_t, 257> sine{};
  if (mode & 2) {
    constexpr float tau = 6.28318530717958647692528676655901f;
    for (int index = 0; index < 257; ++index) {
      const float angle = fdiv(fmul(tau, static_cast<float>(index)), 256.0f);
      sine[index] = sFtol(static_cast<float>(std::sin(static_cast<double>(angle)) * 0.5 * 65536.0));
    }
  }

  std::vector<int32_t> row(test.width);
  std::vector<int32_t> polynomial(test.width >> test.frequency);
  for (int x = 0; x < static_cast<int>(polynomial.size()); ++x) {
    const float f = fdiv(static_cast<float>(x), static_cast<float>(polynomial.size()));
    float value = fmul(fmul(f, f), f);
    const float inner = fadd(10.0f, fmul(f, fsub(fmul(6.0f, f), 15.0f)));
    value = fmul(fmul(value, inner), 16384.0f);
    polynomial[x] = sFtol(value);
  }

  std::vector<uint16_t> output(static_cast<size_t>(test.width) * test.height * 4);
  size_t destination = 0;
  for (int y = 0; y < test.height; ++y) {
    std::fill(row.begin(), row.end(), 0);
    float scale = 1.0f;
    for (int octave = test.frequency; octave < test.frequency + test.octaves; ++octave) {
      const bool grouped = shiftX + octave < 16;
      const int groupSize = grouped
        ? std::min(test.width, 1 << (16 - shiftX - octave)) : 1;
      const int groups = grouped
        ? test.width >> (16 - shiftX - octave) : test.width;
      const int mask = ((1 << octave) - 1) & 255;
      const int32_t py = static_cast<int32_t>(static_cast<uint32_t>(y) << (shiftY + octave));
      const int vy = (py >> 16) & mask;
      const int dtx = 1 << (shiftX + octave);
      const float ty = fdiv(static_cast<float>(py & 0xffff), 65536.0f);
      float tyFade = fmul(fmul(ty, ty), ty);
      tyFade = fmul(tyFade, fadd(10.0f, fmul(ty, fsub(fmul(6.0f, ty), 15.0f))));
      const float ty0 = fmul(ty, fsub(1.0f, tyFade));
      const float ty1 = fmul(fsub(ty, 1.0f), tyFade);
      const int vy0 = tables.permutation[vy ^ seed];
      const int vy1 = tables.permutation[((vy + 1) & mask) ^ seed];
      const int polynomialShift = octave - test.frequency;
      const int scaleFixed = sFtol(fmul(scale, 16384.0f));

      if (shiftX + octave < 16 || (py & 0xffff)) {
        int rowIndex = 0;
        for (int vx = 0; vx < groups; ++vx) {
          const int v00 = tables.permutation[((vx + 0) & mask) + vy0];
          const int v01 = tables.permutation[((vx + 1) & mask) + vy0];
          const int v10 = tables.permutation[((vx + 0) & mask) + vy1];
          const int v11 = tables.permutation[((vx + 1) & mask) + vy1];

          const float horizontal0 = fadd(tables.gradients[v00][0],
            fmul(fsub(tables.gradients[v10][0], tables.gradients[v00][0]), tyFade));
          const float horizontal1 = fadd(tables.gradients[v01][0],
            fmul(fsub(tables.gradients[v11][0], tables.gradients[v01][0]), tyFade));
          const float vertical0 = fadd(fmul(tables.gradients[v00][1], ty0),
            fmul(tables.gradients[v10][1], ty1));
          const float vertical1 = fadd(fmul(tables.gradients[v01][1], ty0),
            fmul(tables.gradients[v11][1], ty1));

          int32_t fa = sFtol(fmul(vertical0, 65536.0f));
          int32_t fb = sFtol(fmul(fsub(vertical1, horizontal1), 65536.0f));
          const int32_t faDelta = sFtol(fmul(horizontal0, static_cast<float>(dtx)));
          const int32_t fbDelta = sFtol(fmul(horizontal1, static_cast<float>(dtx)));

          for (int x = 0; x < groupSize; ++x) {
            const int32_t delta = fb - fa;
            int32_t noise = fa + ((delta * polynomial[x << polynomialShift]) >> 14);
            switch (mode) {
              case 0: break;
              case 1: noise = std::abs(noise); break;
              case 3: noise &= 0x7fff; [[fallthrough]];
              case 2: {
                const int index = (noise >> 8) & 255;
                noise = sine[index] + ((sine[index + 1] - sine[index]) * (noise & 255) >> 8);
                break;
              }
            }
            row[rowIndex] += (noise * scaleFixed) >> 14;
            ++rowIndex;
            fa += faDelta;
            fb += fbDelta;
          }
        }
      }
      scale = fmul(scale, test.fadeoff);
    }

    for (int x = 0; x < test.width; ++x) {
      const int value = range7fff(mulShift(row[x], amplitudeFixed) + offset);
      const int gammaIndex = value >> 5;
      const int fade = gammaTable[gammaIndex] +
        (((gammaTable[gammaIndex + 1] - gammaTable[gammaIndex]) * (value & 31)) >> 5);
      for (int channel = 0; channel < 4; ++channel) {
        output[destination++] = fadeChannel(color0[channel], color1[channel], fade);
      }
    }
  }
  return output;
}

static uint32_t hashWords(const std::vector<uint16_t> &words) {
  uint32_t hash = 0x811c9dc5u;
  for (uint16_t word : words) {
    hash = (hash ^ (word & 255u)) * 0x01000193u;
    hash = (hash ^ (word >> 8)) * 0x01000193u;
  }
  return hash;
}

int main() {
  std::fesetround(FE_TONEAREST);
  const PerlinTables tables = makePerlinTables();
  const PerlinCase cases[] = {
    {"small0", 8, 8, 1, 3, 0.5f, 17, 0, 1.0f, 1.0f, 0xff000000u, 0xffffffffu},
    {"small1", 16, 8, 1, 4, 0.625f, 93, 1, 0.75f, 1.25f, 0xff102030u, 0xffe0c080u},
    {"small2", 16, 8, 2, 3, 0.75f, 7, 2, 1.125f, 0.8f, 0x80402010u, 0xffd0e0f0u},
    {"small3", 16, 8, 2, 3, 0.75f, 7, 3, 1.125f, 0.8f, 0x80402010u, 0xffd0e0f0u},
    // Released Debris KX operators 11 and 49 at their native 512x512 size.
    {"production11", 512, 512, 2, 7, 0.999755859375f, 1, 0,
      0.849609375f, 0.7998046875f, 0xff000000u, 0xffffffffu},
    {"production49", 512, 512, 4, 5, 0.639892578125f, 0, 0,
      1.599609375f, 1.0f, 0xff202020u, 0xffffffffu},
  };
  for (const PerlinCase &test : cases) {
    const auto bitmap = bitmapPerlin(test, tables);
    std::printf("%s %08x\n", test.name, hashWords(bitmap));
  }
}
