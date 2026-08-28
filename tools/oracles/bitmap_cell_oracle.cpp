// Standalone transcription of werkkzeug3 Bitmap_Cell for differential tests.
//
// Source: vendor/wz3/genbitmap.cpp:2694 (Bitmap_Cell, the !sINTRO branch),
// with sRange7fff (genbitmap.cpp:125), GetColor64(sU32) (genbitmap.cpp:146),
// sSetRndSeed/sGetRnd (vendor/wz3/_types.cpp:43-118) and the scalar form of
// the MMX Fade64 (genbitmap.cpp:170) that bitmap_misc_oracle.cpp already uses.
//
// Bitmap_Cell itself is asm-free; only Fade64 and the RNG needed transcribing.
//
// Build: c++ -std=c++17 -O2 -o /tmp/cell tools/oracles/bitmap_cell_oracle.cpp
// Run:   /tmp/cell            (prints one FNV-1a per shipped fixture)
//        /tmp/cell <path>     (also writes the raw little-endian sU64 words)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static uint32_t randomSeed = 0x74382381u;

static uint32_t getRnd() {
  const uint32_t first = randomSeed * 0x343fdu + 0x269ec3u;
  const uint32_t second = first * 0x343fdu + 0x269ec3u;
  randomSeed = second;
  return ((second >> 10) & 0xffffu) | ((first << 6) & 0xffff0000u);
}

static uint32_t getRnd(uint32_t max) { return getRnd() % max; }

static void setRndSeed(int32_t seed) {
  const int32_t mixed = seed + seed * 17 + seed * 121 + seed * 121 / 17;
  randomSeed = static_cast<uint32_t>(mixed);
  getRnd(); randomSeed ^= static_cast<uint32_t>(mixed);
  getRnd(); randomSeed ^= static_cast<uint32_t>(mixed);
  getRnd(); randomSeed ^= static_cast<uint32_t>(mixed);
  getRnd();
}

static int32_t sFtol(float value) { return static_cast<int32_t>(std::nearbyintf(value)); }

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

// Fade64 lane: r = (a * (0x8000 - fade/2) >> 16) + (b * (fade/2) >> 16), << 1.
static uint16_t fadeChannel(uint16_t a, uint16_t b, int32_t fade) {
  const int16_t weightB = saturate16(fade >> 1);
  const int16_t weightA = saturate16(0x8000 - (fade >> 1));
  const uint16_t sum = static_cast<uint16_t>(
    (signed16(a) * weightA >> 16) + (signed16(b) * weightB >> 16));
  return static_cast<uint16_t>(sum << 1);
}

// GetColor64 (genbitmap.cpp:146) spreads each byte to 15 bits as byte*257/2.
// Its sU64 lane order is b,g,r,a; the port and the other oracles here store
// the same four values as r,g,b,a, so use that layout for a direct compare.
struct Color64 { uint16_t lane[4]; };  // r, g, b, a

static Color64 getColor64(uint32_t c) {
  Color64 result;
  result.lane[0] = static_cast<uint16_t>((((c >> 16) & 255u) * 257u) >> 1);
  result.lane[1] = static_cast<uint16_t>((((c >> 8) & 255u) * 257u) >> 1);
  result.lane[2] = static_cast<uint16_t>(((c & 255u) * 257u) >> 1);
  result.lane[3] = static_cast<uint16_t>(((c >> 24) * 257u) >> 1);
  return result;
}

static void fade64(Color64 &r, const Color64 &c0, const Color64 &c1, int32_t fade) {
  for (int i = 0; i < 4; ++i) r.lane[i] = fadeChannel(c0.lane[i], c1.lane[i], fade);
}

static int getPower2(int value) {
  int power = 0;
  while ((1 << power) < value) ++power;
  return power;
}

static std::vector<Color64> cell(int xsize, int ysize, uint32_t col0, uint32_t col1,
                                 uint32_t col2, int max, int seed, float amp, float gamma,
                                 int mode, float mindistf, int percent, float aspect) {
  static int cells[256][3];
  static int cellt[256];
  std::vector<Color64> data(static_cast<size_t>(xsize) * ysize);

  setRndSeed(seed);
  for (int i = 0; i < max * 3; ++i) cells[0][i] = static_cast<int>(getRnd(0x4000));

  int mdist = sFtol(mindistf * 0x4000);
  mdist = mdist * mdist;
  for (int i = 1; i < max;) {
    if ((mode & 2) && static_cast<int>(getRnd(255)) < percent) cells[i][2] = 0xffff;
    const int px = ((cells[i][0]) & 0x3fff) - 0x2000;
    const int py = ((cells[i][1]) & 0x3fff) - 0x2000;
    bool cut = false;
    for (int j = 0; j < i && !cut; ++j) {
      const int dx = ((cells[j][0] - px) & 0x3fff) - 0x2000;
      const int dy = ((cells[j][1] - py) & 0x3fff) - 0x2000;
      if (dx * dx + dy * dy < mdist) cut = true;
    }
    if (cut) {
      --max;
      cells[i][0] = cells[max][0];
      cells[i][1] = cells[max][1];
      cells[i][2] = cells[max][2];
    } else {
      ++i;
    }
  }

  const int shiftx = 14 - getPower2(xsize);
  const int shifty = 14 - getPower2(ysize);
  const Color64 c0 = getColor64(col1);
  const Color64 c1 = getColor64(col0);
  const Color64 cb = getColor64(col2);

  aspect = std::pow(2.0f, aspect);
  float aspdiv;
  int aspf;
  bool flipxy;
  if (aspect >= 1.0f) {
    aspf = static_cast<int>(65536 / (aspect * aspect));
    aspdiv = aspect / 16384.0f;
    flipxy = false;
  } else {
    aspf = static_cast<int>(aspect * aspect * 65536);
    aspdiv = 1.0f / (16384.0f * aspect);
    flipxy = true;
  }
  if (flipxy) for (int i = 0; i < max; ++i) std::swap(cells[i][0], cells[i][1]);

  const int tileSize = 16;
  for (int by = 0; by < ysize; by += tileSize) {
    for (int bx = 0; bx < xsize; bx += tileSize) {
      int px0 = bx << shiftx, px1 = (bx + tileSize - 1) << shiftx;
      int py0 = by << shifty, py1 = (by + tileSize - 1) << shifty;
      if (flipxy) { std::swap(px0, py0); std::swap(px1, py1); }

      for (int i = 0; i < max; ++i) {
        int dx = ((cells[i][0] - px0) & 0x3fff) - 0x2000;
        int dy = ((cells[i][0] - px1) & 0x3fff) - 0x2000;
        if ((dx ^ dy) <= 0) {
          cellt[i] = 0;
        } else {
          dx = std::min(std::abs(dx), std::abs(dy));
          cellt[i] = mulShift(dx * dx, aspf);
        }
        dx = ((cells[i][1] - py0) & 0x3fff) - 0x2000;
        dy = ((cells[i][1] - py1) & 0x3fff) - 0x2000;
        if ((dx ^ dy) > 0) {
          dy = std::min(std::abs(dx), std::abs(dy));
          cellt[i] += dy * dy;
        }
      }

      for (int i = 1; i < max; ++i) {
        const int x = cells[i][0], y = cells[i][1], c = cells[i][2];
        const int dy = cellt[i];
        int j = i;
        while (j && cellt[j - 1] > dy) {
          cells[j][0] = cells[j - 1][0];
          cells[j][1] = cells[j - 1][1];
          cells[j][2] = cells[j - 1][2];
          cellt[j] = cellt[j - 1];
          --j;
        }
        cells[j][0] = x; cells[j][1] = y; cells[j][2] = c; cellt[j] = dy;
      }

      size_t tile = static_cast<size_t>(by) * xsize + bx;
      for (int ty = 0; ty < tileSize; ++ty) {
        const int py = (by + ty) << shifty;
        for (int tx = 0; tx < tileSize; ++tx) {
          const int px = (bx + tx) << shiftx;
          const int x = flipxy ? py : px;
          const int y = flipxy ? px : py;

          int best = 0x8000 * 0x8000, best2 = 0x8000 * 0x8000;
          int besti = -1, best2i = -1;
          for (int i = 0; i < max && best2 > cellt[i]; ++i) {
            const int dx = ((cells[i][0] - x) & 0x3fff) - 0x2000;
            const int dy = ((cells[i][1] - y) & 0x3fff) - 0x2000;
            const int dist = mulShift(dx * dx, aspf) + dy * dy;
            if (dist < best) {
              best2 = best; best2i = besti; best = dist; besti = i;
            } else if (dist > best && dist < best2) {
              best2 = dist; best2i = i;
            }
          }
          (void)best2i;

          float v0 = std::sqrt(static_cast<float>(best)) * aspdiv;
          if (mode & 1) {
            const float v1 = std::sqrt(static_cast<float>(best2)) * aspdiv;
            v0 = (v0 + v1 > 0.00001f) ? (v1 - v0) / (v1 + v0) : 0.0f;
          }
          int val = range7fff(static_cast<int>(std::pow(v0 * amp, gamma) * 0x8000)) * 2;
          if (mode & 4) val = 0x10000 - val;

          if (mode & 2) {
            Color64 cc;
            if (cells[besti][2] == 0xffff) cc = cb;
            else fade64(cc, c0, c1, cells[besti][2] * 4);
            fade64(data[tile], cc, cb, val);
          } else {
            fade64(data[tile], c0, c1, val);
          }
          ++tile;
        }
        tile += static_cast<size_t>(xsize) - tileSize;
      }
    }
  }
  return data;
}

static uint32_t fnv1a(const uint8_t *bytes, size_t size) {
  uint32_t hash = 0x811c9dc5u;
  for (size_t i = 0; i < size; ++i) {
    hash ^= bytes[i];
    hash *= 0x01000193u;
  }
  return hash;
}

struct Fixture {
  const char *name; int size;
  uint32_t col0, col1, col2; int max, seed;
  float amp, gamma; int mode; float mindistf; int percent; float aspect;
};

int main(int argc, char **argv) {
  // The three shipped Bitmap_Cell operators of assets/debris_party.kx, at the
  // authored sizes tools/test_bitmap.mjs evaluates them at.
  const Fixture fixtures[] = {
    // Parameter words copied verbatim from notes/debris_party_kx.json order
    // (xs, ys, col0, col1, col2, max, seed, amp, gamma, mode, mindistf,
    // percent, aspect) as decimals, so no hand hex conversion can slip in.
    {"op452", 128, 4294967295u, 4278190080u, 4278190080u, 128, 1,
     1.499755859375f, 0.5f, 1, 0.0f, 0, -2.49993896484375f},
    {"op475", 256, 4280624421u, 4278190080u, 4278190080u, 90, 5,
     0.0f, 0.0f, 7, 0.125f, 0, 0.0f},
    {"op481", 512, 4280624421u, 4278453252u, 4278190080u, 255, 11,
     0.0f, 0.0f, 7, 0.0625f, 0, 0.0f},
  };
  for (const Fixture &f : fixtures) {
    const std::vector<Color64> data = cell(f.size, f.size, f.col0, f.col1, f.col2,
      f.max, f.seed, f.amp, f.gamma, f.mode, f.mindistf, f.percent, f.aspect);
    std::vector<uint8_t> bytes(data.size() * 8);
    for (size_t i = 0; i < data.size(); ++i) {
      for (int lane = 0; lane < 4; ++lane) {
        bytes[i * 8 + lane * 2] = static_cast<uint8_t>(data[i].lane[lane] & 0xff);
        bytes[i * 8 + lane * 2 + 1] = static_cast<uint8_t>(data[i].lane[lane] >> 8);
      }
    }
    std::printf("%s=%u\n", f.name, fnv1a(bytes.data(), bytes.size()));
    if (argc > 1) {
      char path[512];
      std::snprintf(path, sizeof(path), "%s/%s.bin", argv[1], f.name);
      if (FILE *out = std::fopen(path, "wb")) {
        std::fwrite(bytes.data(), 1, bytes.size(), out);
        std::fclose(out);
      }
    }
  }
  return 0;
}
