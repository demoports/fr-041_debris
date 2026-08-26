// Standalone scalar oracle for Bitmap_Blur/BlurCore in the pinned
// werkkzeug3/genbitmap.cpp.  The released implementation is x86 MMX; this
// transcription spells out each lane operation so it can run on non-x86
// development machines without making the JavaScript implementation its own
// oracle.

#include <algorithm>
#include <array>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static float fmul(float left, float right) {
  volatile float result = left * right;
  return result;
}

// sFloatFix selects round-to-nearest-even and single precision before the
// bitmap graph runs.  sFtol is an x87 fistp and therefore observes that mode.
static int32_t sFtol(float value) {
  return static_cast<int32_t>(std::nearbyintf(value));
}

static int16_t signed16(uint16_t value) {
  int16_t result;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

static int32_t signed32(uint32_t value) {
  int32_t result;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

static int32_t addWrap32(int32_t left, int32_t right) {
  return signed32(static_cast<uint32_t>(left) + static_cast<uint32_t>(right));
}

static int32_t subWrap32(int32_t left, int32_t right) {
  return signed32(static_cast<uint32_t>(left) - static_cast<uint32_t>(right));
}

static int32_t maddSignedWords(uint16_t pixel0, int coefficient0,
                               uint16_t pixel1, int coefficient1) {
  const int32_t product0 = static_cast<int32_t>(signed16(pixel0)) *
    static_cast<int32_t>(signed16(static_cast<uint16_t>(coefficient0)));
  const int32_t product1 = static_cast<int32_t>(signed16(pixel1)) *
    static_cast<int32_t>(signed16(static_cast<uint16_t>(coefficient1)));
  return signed32(static_cast<uint32_t>(product0) + static_cast<uint32_t>(product1));
}

static uint16_t addSaturateUnsigned16(uint16_t left, uint16_t right) {
  const uint32_t sum = static_cast<uint32_t>(left) + right;
  return static_cast<uint16_t>(std::min(sum, 0xffffu));
}

// Literal scalar form of BlurCore's psrad/unpack/multiply/saturating-add
// sequence.  This is intentionally not simplified to a division: the three
// paddusw instructions and signed accumulator comparison are observable.
static uint16_t scaleAccumulator(int32_t accumulator, int32_t amplitude,
                                 int32_t amplitudeClip) {
  const int32_t shifted = accumulator >> 6;
  const uint16_t low = static_cast<uint16_t>(shifted);
  const uint16_t high = static_cast<uint16_t>(static_cast<uint32_t>(shifted) >> 16);
  const uint16_t amplitudeLow = static_cast<uint16_t>(amplitude);
  const uint16_t amplitudeHigh = static_cast<uint16_t>(static_cast<uint32_t>(amplitude) >> 16);

  uint16_t output = accumulator > amplitudeClip ? 0xffffu : 0u;
  output = addSaturateUnsigned16(output,
    static_cast<uint16_t>(static_cast<uint32_t>(high) * amplitudeLow));
  output = addSaturateUnsigned16(output,
    static_cast<uint16_t>(static_cast<uint32_t>(low) * amplitudeHigh));
  output = addSaturateUnsigned16(output,
    static_cast<uint16_t>((static_cast<uint32_t>(low) * amplitudeLow) >> 16));

  // paddw 0x8000 followed by psubsw 0x8000 clamps an unsigned word to
  // 0x7fff, without clamping negative accumulators before the fixed multiply.
  return std::min<uint16_t>(output, 0x7fffu);
}

static void blurCore(const uint16_t *source, uint16_t *destination,
                     int kernelSize, int resolution, int f0, int f1,
                     int amplitude, int amplitudeClip) {
  const int coordinateMask = resolution - 1;
  int entering = (-((kernelSize + 1) >> 1)) & coordinateMask;
  int leaving = entering;
  std::array<int32_t, 4> accumulator{};
  const int edgeDifference = signed16(static_cast<uint16_t>(f1 - f0));

  for (int channel = 0; channel < 4; ++channel) {
    accumulator[channel] = maddSignedWords(
      source[leaving * 4 + channel], edgeDifference,
      source[leaving * 4 + channel], 0);
  }

  for (int index = 0; index < kernelSize; ++index) {
    const int next = (entering + 1) & coordinateMask;
    for (int channel = 0; channel < 4; ++channel) {
      accumulator[channel] = addWrap32(accumulator[channel], maddSignedWords(
        source[entering * 4 + channel], f0,
        source[next * 4 + channel], f1));
    }
    entering = next;
  }

  for (int coordinate = 0; coordinate < resolution; ++coordinate) {
    const int enteringNext = (entering + 1) & coordinateMask;
    for (int channel = 0; channel < 4; ++channel) {
      accumulator[channel] = addWrap32(accumulator[channel], maddSignedWords(
        source[entering * 4 + channel], f0,
        source[enteringNext * 4 + channel], f1));
      destination[coordinate * 4 + channel] = scaleAccumulator(
        accumulator[channel], amplitude, amplitudeClip);
    }
    entering = enteringNext;

    const int leavingNext = (leaving + 1) & coordinateMask;
    for (int channel = 0; channel < 4; ++channel) {
      accumulator[channel] = subWrap32(accumulator[channel], maddSignedWords(
        source[leavingNext * 4 + channel], f0,
        source[leaving * 4 + channel], f1));
    }
    leaving = leavingNext;
  }
}

static void transpose(const uint16_t *source, uint16_t *destination,
                      int width, int height) {
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int channel = 0; channel < 4; ++channel) {
        destination[(x * height + y) * 4 + channel] =
          source[(y * width + x) * 4 + channel];
      }
    }
  }
}

static std::vector<uint16_t> bitmapBlur(const std::vector<uint16_t> &input,
                                        int width, int height, int flags,
                                        float sx, float sy, float requestedAmplitude) {
  std::vector<uint16_t> original = input;
  const int order = flags & 15;
  if (order == 0) return original;

  std::vector<uint16_t> scratch(input.size());
  uint16_t *current = original.data();
  uint16_t *other = scratch.data();
  int currentWidth = width;
  int currentHeight = height;

  int size = sFtol(fmul(fmul(128.0f, sx), static_cast<float>(width)));
  const int size2 = sFtol(fmul(fmul(128.0f, sy), static_cast<float>(height)));
  // Unlike sFtol above, this is the ordinary C++ float-to-int conversion used
  // by `famp = ...`: it truncates toward zero after single-precision products.
  const int32_t fixedAmplitude = static_cast<int32_t>(
    fmul(fmul(requestedAmplitude, 65536.0f), 64.0f));

  for (int axis = 0; axis < 2; ++axis) {
    const int f1 = (size & 127) / 2;
    const int f0 = 64 - f1;
    int kernelSize = (size / 128) * 2;
    if (flags & 0x10) ++kernelSize;

    const int divisor = std::max(1, kernelSize * 64 + f1 * 2);
    const int32_t amplitude = fixedAmplitude / divisor;
    const int32_t amplitudeClip = amplitude > 128
      ? static_cast<int32_t>((static_cast<int64_t>(65536 * 64) << 16) / amplitude) - 1
      : 0x7fffffff;

    uint16_t *const rowSourceBuffer = current;
    uint16_t *const rowScratchBuffer = other;
    for (int y = 0; y < currentHeight; ++y) {
      uint16_t *passSource = rowSourceBuffer + y * currentWidth * 4;
      uint16_t *passDestination = rowScratchBuffer + y * currentWidth * 4;
      for (int pass = 0; pass < order; ++pass) {
        blurCore(passSource, passDestination, kernelSize, currentWidth,
          f0, f1, amplitude, amplitudeClip);
        std::swap(passSource, passDestination);
      }
    }

    uint16_t *const blurred = (order & 1) ? rowScratchBuffer : rowSourceBuffer;
    uint16_t *const transposed = (blurred == original.data()) ? scratch.data() : original.data();
    transpose(blurred, transposed, currentWidth, currentHeight);
    current = transposed;
    other = blurred;
    std::swap(currentWidth, currentHeight);
    size = size2;
  }

  // Two transposes and identical order parity on both axes put the released
  // result back into the bitmap's original allocation.
  if (current != original.data()) {
    std::fprintf(stderr, "internal buffer parity mismatch\n");
    std::abort();
  }
  return original;
}

static uint32_t hashWords(const std::vector<uint16_t> &words) {
  uint32_t hash = 0x811c9dc5u;
  for (uint16_t value : words) {
    hash = (hash ^ (value & 255u)) * 0x01000193u;
    hash = (hash ^ (value >> 8)) * 0x01000193u;
  }
  return hash;
}

static std::vector<uint16_t> makeInput(int width, int height, int variant) {
  std::vector<uint16_t> input(static_cast<size_t>(width) * height * 4);
  static constexpr std::array<uint16_t, 12> signedPattern = {
    0x0000, 0x0001, 0x7fff, 0x8000, 0xffff, 0x4000,
    0xc000, 0x1234, 0xfedc, 0x7ffe, 0x8001, 0x5555,
  };
  for (size_t index = 0; index < input.size(); ++index) {
    const uint32_t mixed = (static_cast<uint32_t>(index + 17) * 7919u) ^
      (static_cast<uint32_t>(index) * 313u);
    if (variant == 0) input[index] = static_cast<uint16_t>(mixed & 0x7fffu);
    else if (variant == 1) input[index] = static_cast<uint16_t>(mixed);
    else input[index] = signedPattern[(index * 5 + index / 4) % signedPattern.size()];
  }
  return input;
}

struct BlurCase {
  const char *name;
  int width, height, flags;
  float sx, sy, amplitude;
  int inputVariant;
};

static void printWords(const std::vector<uint16_t> &words, size_t begin, size_t count) {
  std::putchar('[');
  for (size_t index = 0; index < count; ++index) {
    if (index) std::putchar(',');
    std::printf("%u", static_cast<unsigned>(words[begin + index]));
  }
  std::putchar(']');
}

static void printFixtures() {
  static constexpr std::array<BlurCase, 6> cases = {{
    {"production_normal_order2", 16, 16, 0x02,
      0.06298828125f, 0.06298828125f, 1.0f, 0},
    {"production_sharpen_order2", 16, 16, 0x12,
      0.0999755859375f, 0.0999755859375f, 1.0f, 0},
    {"production_sharpen_odd_one_axis", 16, 8, 0x11,
      0.024993896484375f, 0.0f, 1.0f, 0},
    {"multiwrap_order3", 8, 8, 0x03,
      1.75f, 0.125f, 0.875f, 0},
    {"signed_saturating_sharpen", 8, 8, 0x12,
      0.029998779296875f, 0.029998779296875f, 1.25f, 2},
    {"round_even_and_fractional_amplitude", 8, 8, 0x01,
      0.00244140625f, 0.00341796875f, 0.10000000149011612f, 1},
  }};

  std::puts("[");
  for (size_t index = 0; index < cases.size(); ++index) {
    const BlurCase &test = cases[index];
    const auto output = bitmapBlur(makeInput(test.width, test.height, test.inputVariant),
      test.width, test.height, test.flags, test.sx, test.sy, test.amplitude);
    std::printf("  {\"name\":\"%s\",\"width\":%d,\"height\":%d,\"flags\":%d,"
                "\"sx\":%.17g,\"sy\":%.17g,\"amplitude\":%.17g,"
                "\"inputVariant\":%d,\"hash\":%u,\"head\":",
      test.name, test.width, test.height, test.flags,
      static_cast<double>(test.sx), static_cast<double>(test.sy),
      static_cast<double>(test.amplitude), test.inputVariant, hashWords(output));
    printWords(output, 0, 16);
    std::printf(",\"tail\":");
    printWords(output, output.size() - 16, 16);
    std::printf("}%s\n", index + 1 == cases.size() ? "" : ",");
  }
  std::puts("]");
}

static void runBenchmark() {
  constexpr int width = 128, height = 128, iterations = 12;
  const auto input = makeInput(width, height, 0);
  uint32_t checksum = 0;
  const auto start = std::chrono::steady_clock::now();
  for (int iteration = 0; iteration < iterations; ++iteration) {
    const auto output = bitmapBlur(input, width, height, 0x12,
      0.0999755859375f, 0.0999755859375f, 1.0f);
    checksum ^= hashWords(output) + static_cast<uint32_t>(iteration);
  }
  const auto end = std::chrono::steady_clock::now();
  const double milliseconds = std::chrono::duration<double, std::milli>(end - start).count();
  std::printf("blur oracle benchmark: %d iterations of %dx%d in %.3f ms; checksum=%u\n",
    iterations, width, height, milliseconds, checksum);
}

int main(int argc, char **argv) {
  std::fesetround(FE_TONEAREST);
  if (argc == 2 && std::strcmp(argv[1], "--benchmark") == 0) runBenchmark();
  else printFixtures();
  return 0;
}
