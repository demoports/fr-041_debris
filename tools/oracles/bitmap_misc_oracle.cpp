// Standalone source oracle for bitmap paths whose released implementation
// mixes MMX word arithmetic with single-precision scalar intermediates.
// It has no werkkzeug dependencies and stays deliberately small enough to run
// as part of a bounded fidelity audit.

#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

static float fadd(float a, float b) { volatile float value = a + b; return value; }
static float fsub(float a, float b) { volatile float value = a - b; return value; }
static float fmul(float a, float b) { volatile float value = a * b; return value; }
static float fdiv(float a, float b) { volatile float value = a / b; return value; }
static int32_t sFtol(float value) { return static_cast<int32_t>(std::nearbyintf(value)); }
static int32_t truncInt(float value) { return static_cast<int32_t>(value); }

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
  const int16_t weightB = saturate16(fade >> 1);
  const int16_t weightA = saturate16(0x8000 - (fade >> 1));
  const uint16_t sum = static_cast<uint16_t>(
    (signed16(a) * weightA >> 16) + (signed16(b) * weightB >> 16));
  return static_cast<uint16_t>(sum << 1);
}

static int32_t mulHighSigned16(uint16_t a, int16_t b) {
  return (static_cast<int32_t>(signed16(a)) * b) >> 16;
}

static void addScalePixel(std::vector<uint16_t> &output, size_t offset,
                          const std::array<uint16_t, 4> &base,
                          const std::array<uint16_t, 4> &scale, int fade) {
  const int16_t weight = saturate16(fade >> 1);
  for (int channel = 0; channel < 4; ++channel) {
    const int value = signed16(base[channel]) +
      (mulHighSigned16(scale[channel], weight) << 1);
    output[offset + channel] = static_cast<uint16_t>(saturate16(value));
  }
}

static std::array<uint16_t, 4> packedColor(uint32_t color) {
  return {
    static_cast<uint16_t>((((color >> 16) & 255u) * 257u) >> 1),
    static_cast<uint16_t>((((color >> 8) & 255u) * 257u) >> 1),
    static_cast<uint16_t>(((color & 255u) * 257u) >> 1),
    static_cast<uint16_t>(((color >> 24) * 257u) >> 1),
  };
}

static uint16_t grayMMX(uint16_t r, uint16_t g, uint16_t b) {
  const uint16_t rg = static_cast<uint16_t>(r + g) >> 1;
  const uint16_t gb = static_cast<uint16_t>(g + b) >> 1;
  return static_cast<uint16_t>(rg + gb) >> 1;
}

static uint32_t hashWords(const std::vector<uint16_t> &words) {
  uint32_t hash = 0x811c9dc5u;
  for (uint16_t word : words) {
    hash = (hash ^ (word & 255u)) * 0x01000193u;
    hash = (hash ^ (word >> 8)) * 0x01000193u;
  }
  return hash;
}

struct SourceRandom {
  uint32_t state = 0x74382381u;

  uint32_t next() {
    const uint32_t first = state * 0x343fdu + 0x269ec3u;
    const uint32_t second = first * 0x343fdu + 0x269ec3u;
    state = second;
    return ((second >> 10) & 0xffffu) | ((first << 6) & 0xffff0000u);
  }

  void seed(int32_t value) {
    const int32_t value121 = static_cast<int32_t>(
      static_cast<uint32_t>(value) * 121u);
    const int32_t mixed = static_cast<int32_t>(
      static_cast<uint32_t>(value) + static_cast<uint32_t>(value) * 17u +
      static_cast<uint32_t>(value121) +
      static_cast<uint32_t>(value121 / 17));
    state = static_cast<uint32_t>(mixed);
    next(); state ^= static_cast<uint32_t>(mixed);
    next(); state ^= static_cast<uint32_t>(mixed);
    next(); state ^= static_cast<uint32_t>(mixed);
    next();
  }

  float unitFloat() {
    return fdiv(static_cast<float>(next() & 0x3fffffffu), 1073741824.0f);
  }
};

// A 1x1-brick layout with zero mortar widths paints one source-random colour
// over the complete bitmap. It retains all relevant Bricks conversions while
// making the native oracle independent of the large layout loops.
static std::vector<uint16_t> bricksColorRounding() {
  SourceRandom random;
  random.seed(138);
  (void) random.unitFloat(); // initial cb, replaced by the forced first head
  const float powered = static_cast<float>(std::pow(random.unitFloat(), 0.25f));
  const int fade = truncInt(fmul(powered, 65536.0f));
  const auto black = packedColor(0xff000000u);
  const auto white = packedColor(0xffffffffu);
  std::array<uint16_t, 4> cell{};
  for (int channel = 0; channel < 4; ++channel) {
    cell[channel] = fadeChannel(black[channel], white[channel], fade);
  }
  std::vector<uint16_t> output(8 * 8 * 4);
  for (size_t index = 0; index < output.size(); index += 4) {
    for (int channel = 0; channel < 4; ++channel) {
      output[index + channel] = fadeChannel(
        black[channel], cell[channel], 0x10000);
    }
  }
  return output;
}

static std::vector<uint16_t> pattern(int width, int height) {
  std::vector<uint16_t> result(static_cast<size_t>(width) * height * 4);
  for (size_t index = 0; index < result.size(); ++index) {
    result[index] = static_cast<uint16_t>(
      (((static_cast<uint32_t>(index) + 17u) * 7919u) ^
       (static_cast<uint32_t>(index) * 313u)) & 0x7fffu);
  }
  return result;
}

static std::vector<uint16_t> gradient(int width, int height, uint32_t color0,
                                      uint32_t color1, float position,
                                      float angle, float length) {
  constexpr float tau = 6.28318530717958647692528676655901f;
  const auto c0 = packedColor(color0), c1 = packedColor(color1);
  const float l = fdiv(32768.0f, length);
  const float radians = fmul(angle, tau);
  const int dx = sFtol(static_cast<float>(std::cos(radians) * l));
  const int dy = sFtol(static_cast<float>(std::sin(radians) * l));
  const int cdx = mulShift(dx, 0x10000 / width);
  const int cdy = mulShift(dy, 0x10000 / height) - width * cdx;
  int coordinate = truncInt(fsub(16384.0f,
    fmul(static_cast<float>(dx / 2 + dy / 2), fadd(position, 1.0f))));
  std::vector<uint16_t> output(static_cast<size_t>(width) * height * 4);
  size_t destination = 0;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const int fade = range7fff(coordinate) * 2;
      for (int channel = 0; channel < 4; ++channel) {
        output[destination++] = fadeChannel(c0[channel], c1[channel], fade);
      }
      coordinate += cdx;
    }
    coordinate += cdy;
  }
  return output;
}

static int cbLookup(const std::array<int, 257> &table, uint16_t value) {
  const int index = value >> 7;
  return table[index] +
    (((table[index + 1] - table[index]) * (value & 127)) >> 7);
}

static std::vector<uint16_t> colorBalance(
    const std::vector<uint16_t> &source,
    const std::array<float, 9> &parameters) {
  const float scale = static_cast<float>(100.0f / 255.0f);
  std::array<std::array<int, 257>, 3> tables{};
  for (int channel = 0; channel < 3; ++channel) {
    const float shadow = parameters[channel];
    const float midtone = parameters[channel + 3];
    const float highlight = parameters[channel + 6];
    const float argument = fadd(fadd(fmul(shadow, 0.5f), midtone),
                                fmul(highlight, 0.5f));
    const float exponent = static_cast<float>(std::pow(0.5, argument));
    const float minimum = fmul(-std::min(shadow, 0.0f), scale);
    const float maximum = fsub(1.0f, fmul(std::max(highlight, 0.0f), scale));
    const float multiplier = fdiv(1.0f, fsub(maximum, minimum));
    for (int index = 0; index <= 256; ++index) {
      const float coordinate = fdiv(static_cast<float>(index), 256.0f);
      const float bounded = std::max(minimum, std::min(maximum, coordinate));
      const float normalized = fmul(fsub(bounded, minimum), multiplier);
      const float mapped = static_cast<float>(std::pow(normalized, exponent));
      tables[channel][index] = range7fff(truncInt(fmul(mapped, 32768.0f)));
    }
  }
  std::vector<uint16_t> output(source.size());
  for (size_t index = 0; index < source.size(); index += 4) {
    output[index] = static_cast<uint16_t>(cbLookup(tables[0], source[index]));
    output[index + 1] = static_cast<uint16_t>(cbLookup(tables[1], source[index + 1]));
    output[index + 2] = static_cast<uint16_t>(cbLookup(tables[2], source[index + 2]));
    output[index + 3] = source[index + 3];
  }
  return output;
}

static int hscbLookup(const std::array<int, 1026> &table, uint16_t value) {
  const int index = value >> 5;
  return table[index] +
    (((table[index + 1] - table[index]) * (value & 31)) >> 5);
}

// Complete HSCB path for the authored hue=0/saturation=1 mode. This is the
// production mode at ops 23 and 46 and isolates the x87-single gamma table
// from the (integer-only) HSV adjustment below it.
static std::vector<uint16_t> hscbNeutral(
    const std::vector<uint16_t> &source, float contrast, float brightness) {
  const float exponent = fmul(contrast, contrast);
  std::array<int, 1026> table{};
  for (int index = 0; index <= 1025; ++index) {
    const float coordinate = fdiv(
      fadd(static_cast<float>(index * 32), 0.01f), 32768.0f);
    const float curved = static_cast<float>(std::pow(coordinate, exponent));
    table[index] = truncInt(fmul(fmul(curved, 32768.0f), brightness));
  }
  std::vector<uint16_t> output(source);
  for (size_t index = 0; index < source.size(); index += 4) {
    for (int channel = 0; channel < 3; ++channel) {
      output[index + channel] = static_cast<uint16_t>(range7fff(
        hscbLookup(table, source[index + channel])));
    }
  }
  return output;
}

// Square-distance GlowRect with a 1024x1 radius. Each output x addresses one
// exact GammaTable entry, so the fixture observes the single-precision
// subtraction/multiply around sFPow without depending on geometric rounding.
static std::vector<uint16_t> glowLinear() {
  constexpr float curvePower = 0.4f; // 0.25f / authored power 0.625f
  constexpr float alpha = 32768.0f;
  const float exponent = fmul(curvePower, 2.0f);
  std::array<int, 1025> table{};
  for (int index = 0; index <= 1024; ++index) {
    const float coordinate = fdiv(static_cast<float>(index), 1024.0f);
    const float falloff = static_cast<float>(
      1.0 - std::pow(coordinate, exponent));
    table[index] = range7fff(truncInt(fmul(falloff, alpha))) * 2;
  }
  const auto black = packedColor(0xff000000u);
  const auto white = packedColor(0xffffffffu);
  std::vector<uint16_t> output(1024 * 4);
  for (int x = 0; x < 1024; ++x) {
    const int fade = table[x];
    for (int channel = 0; channel < 4; ++channel) {
      output[x * 4 + channel] = fadeChannel(
        black[channel], white[channel], fade);
    }
  }
  return output;
}

struct BilinearContext {
  const std::vector<uint16_t> &source;
  int width, height, xMask, yMask, xMax, yMax;

  void sample(int32_t u, int32_t v, std::vector<uint16_t> &output,
              size_t destination) const {
    u &= xMax;
    v &= yMax;
    const int x0 = static_cast<uint32_t>(u) >> 16;
    const int y0 = static_cast<uint32_t>(v) >> 16;
    const int x1 = (x0 + 1) & xMask, y1 = (y0 + 1) & yMask;
    const int fu = (u & 0xffff) >> 1, fv = (v & 0xffff) >> 1;
    const size_t p00 = static_cast<size_t>(y0 * width + x0) * 4;
    const size_t p10 = static_cast<size_t>(y0 * width + x1) * 4;
    const size_t p01 = static_cast<size_t>(y1 * width + x0) * 4;
    const size_t p11 = static_cast<size_t>(y1 * width + x1) * 4;
    for (int channel = 0; channel < 4; ++channel) {
      const uint16_t top = source[p00 + channel], bottom = source[p01 + channel];
      const int16_t topDelta = static_cast<int16_t>(source[p10 + channel] - top);
      const int16_t bottomDelta = static_cast<int16_t>(source[p11 + channel] - bottom);
      const uint16_t h0 = static_cast<uint16_t>(top + ((topDelta * fu >> 16) << 1));
      const uint16_t h1 = static_cast<uint16_t>(bottom + ((bottomDelta * fu >> 16) << 1));
      const int16_t verticalDelta = static_cast<int16_t>(h1 - h0);
      const uint16_t biased = static_cast<uint16_t>(
        h0 + 0x8000 + ((verticalDelta * fv >> 16) << 1));
      output[destination + channel] = biased < 0x8000 ? 0 : biased - 0x8000;
    }
  }
};

static std::vector<uint16_t> bulge(const std::vector<uint16_t> &source,
                                   int width, int height, float warp) {
  std::vector<uint16_t> output(source.size());
  const BilinearContext context{source, width, height, width - 1, height - 1,
                                width * 0x10000 - 1,
                                height * 0x10000 - 1};
  const float invX = fdiv(1.0f, static_cast<float>(width));
  const float invY = fdiv(1.0f, static_cast<float>(height));
  for (int y = 0; y < height; ++y) {
    const float fy = fmul(static_cast<float>(y), invY);
    const float fyCentered = fmul(fsub(0.5f, fy), 2.0f);
    for (int x = 0; x < width; ++x) {
      const float fx = fmul(static_cast<float>(x), invX);
      float u = fmul(fsub(fx, 0.5f), 2.0f);
      float v = fyCentered;
      const float radius2 = fadd(fmul(u, u), fmul(v, v));
      if (radius2 <= 1.0f) {
        const float root = static_cast<float>(std::sqrt(
          static_cast<double>(fsub(1.0f, radius2))));
        const float denominator = fadd(1.0f, fmul(warp, root));
        const float inverse = fdiv(1.0f, denominator);
        u = fmul(u, inverse);
        v = fmul(v, inverse);
      }
      u = fmul(fadd(1.0f, u), 0.5f);
      v = fmul(fsub(1.0f, v), 0.5f);
      context.sample(truncInt(fmul(u, static_cast<float>(width * 0x10000))),
                     truncInt(fmul(v, static_cast<float>(height * 0x10000))),
                     output, static_cast<size_t>(y * width + x) * 4);
    }
  }
  return output;
}

static std::vector<uint16_t> directionalBump(
    const std::vector<uint16_t> &source,
    const std::vector<uint16_t> &normals, float directionA, float directionB,
    uint32_t diffuseColor, uint32_t ambientColor, float amplitude,
    uint32_t specularColor, float specularPower, float specularAmplitude) {
  constexpr float pi = 3.1415926535897932384626433832795f;
  constexpr float tau = 6.28318530717958647692528676655901f;
  const auto diffuse = packedColor(diffuseColor);
  const auto ambient = packedColor(ambientColor);
  const auto specular = packedColor(specularColor);
  directionA = fmul(directionA, tau);
  directionB = fmul(directionB, pi);
  const float cosineB = static_cast<float>(std::cos(directionB));
  const float sineA = static_cast<float>(std::sin(directionA));
  const float cosineA = static_cast<float>(std::cos(directionA));
  const float dx = fmul(cosineB, sineA);
  const float dy = fmul(cosineB, cosineA);
  const float dz = static_cast<float>(std::sin(directionB));
  const float hz = fadd(dz, 1.0f);
  const float halfLength2 = fadd(fadd(fmul(dx, dx), fmul(dy, dy)), fmul(hz, hz));
  const float halfInverse = static_cast<float>(1.0 / std::sqrt(halfLength2));
  const float scaledSpecular = fmul(specularAmplitude, 65536.0f);
  std::vector<uint16_t> output(source.size());
  std::array<uint16_t, 4> lit{};
  for (size_t index = 0; index < source.size(); index += 4) {
    const float nx = fdiv(static_cast<float>(normals[index] - 0x4000), 16384.0f);
    const float ny = fdiv(static_cast<float>(normals[index + 1] - 0x4000), 16384.0f);
    const float nz = fdiv(static_cast<float>(normals[index + 2] - 0x4000), 16384.0f);
    const float light = fadd(fadd(fmul(dx, nx), fmul(dy, ny)), fmul(dz, nz));
    float specularFactor = fadd(fadd(fmul(dx, nx), fmul(dy, ny)), fmul(hz, nz));
    specularFactor = std::max(0.0f, specularFactor);
    specularFactor = static_cast<float>(std::pow(fmul(specularFactor, halfInverse),
                                                 specularPower));
    const float strength = fmul(light, amplitude);
    for (int channel = 0; channel < 4; ++channel) {
      const float lighting = fadd(static_cast<float>(ambient[channel]),
                                  fmul(static_cast<float>(diffuse[channel]), strength));
      const float value = fdiv(fmul(static_cast<float>(source[index + channel]), lighting),
                               32768.0f);
      lit[channel] = static_cast<uint16_t>(range7fff(sFtol(value)));
    }
    addScalePixel(output, index, lit, specular,
                  sFtol(fmul(specularFactor, scaledSpecular)));
  }
  return output;
}

int main() {
  std::fesetround(FE_TONEAREST);
  std::printf("gray=%u alpha=%u\n", grayMMX(3, 0, 1), grayMMX(7, 2, 5));

  const auto gradientWords = gradient(16, 8, 0xff102030u, 0xffe0c080u,
                                      0.3125f, 0.173828125f, 0.73193359375f);
  std::printf("gradient=%u\n", hashWords(gradientWords));

  const auto source = pattern(16, 8);
  const std::array<float, 9> balanceParameters{
    -0.25f, 0.0f, 0.5f, 0.5498046875f, -0.125f, 0.25f,
    0.875f, 0.125f, -0.0478515625f,
  };
  std::printf("color_balance=%u\n",
              hashWords(colorBalance(source, balanceParameters)));
  auto hscbSource = source;
  hscbSource[0] = hscbSource[1] = hscbSource[2] = 234 * 32;
  std::printf("hscb=%u\n", hashWords(hscbNeutral(
    hscbSource, 0.8447265625f, 0.82275390625f)));
  std::printf("glow=%u\n", hashWords(glowLinear()));
  std::printf("bricks=%u\n", hashWords(bricksColorRounding()));
  std::printf("bulge=%u\n", hashWords(bulge(source, 16, 8, 1.0f)));

  auto normals = pattern(16, 8);
  for (size_t index = 3; index < normals.size(); index += 4) normals[index] = 0xffff;
  std::printf("directional_bump=%u\n", hashWords(directionalBump(
    source, normals, 0.125f, 1.0f, 0xffffffffu, 0xffffffffu, 1.0f,
    0xffffffffu, 16.0f, 1.0f)));
  std::printf("directional_bump_trig=%u\n", hashWords(directionalBump(
    source, normals, 0.25f, -0.125f, 0xffffffffu, 0xffffffffu, 1.0f,
    0xffffffffu, 16.0f, 1.0f)));
  return 0;
}
