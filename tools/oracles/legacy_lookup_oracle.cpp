// Standalone oracle for tools/test_legacy_lookup.mjs. This mirrors only the
// two released _start.cpp generators, using forced float stores and the
// round-to-nearest-even mode used by sFtol. It has no project dependencies.

#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

static float f32add(float a, float b) { volatile float r = a + b; return r; }
static float f32mul(float a, float b) { volatile float r = a * b; return r; }
static int sFtol(float value) { return static_cast<int>(std::nearbyintf(value)); }

static uint32_t fnv1a(const uint8_t *bytes, size_t size) {
  uint32_t hash = 0x811c9dc5u;
  for (size_t i = 0; i < size; ++i) hash = (hash ^ bytes[i]) * 0x01000193u;
  return hash;
}

static std::vector<uint8_t> makeCubeNormalizer() {
  constexpr int size = 64;
  static const float faces[6][2][3] = {
    {{ 0, 0,-1},{ 0, 1, 0}},
    {{ 0, 0, 1},{ 0, 1, 0}},
    {{ 1, 0, 0},{ 0, 0,-1}},
    {{ 1, 0, 0},{ 0, 0, 1}},
    {{ 1, 0, 0},{ 0, 1, 0}},
    {{-1, 0, 0},{ 0, 1, 0}},
  };
  std::vector<uint8_t> output(6 * size * size * 4);
  size_t offset = 0;
  const float half = f32mul(size - 1.0f, 0.5f);
  for (int face = 0; face < 6; ++face) {
    const float *u = faces[face][0];
    const float *v = faces[face][1];
    const float n[3] = {
      f32add(f32mul(u[1], v[2]), -f32mul(u[2], v[1])),
      f32add(f32mul(u[2], v[0]), -f32mul(u[0], v[2])),
      f32add(f32mul(u[0], v[1]), -f32mul(u[1], v[0])),
    };
    for (int y = 0; y < size; ++y) for (int x = 0; x < size; ++x) {
      const float us = static_cast<float>(x) - half;
      const float vs = -static_cast<float>(y) + half;
      float p[3];
      for (int c = 0; c < 3; ++c) {
        p[c] = f32add(f32mul(n[c], half), f32mul(u[c], us));
        p[c] = f32add(p[c], f32mul(v[c], vs));
      }
      float dot = f32add(f32add(f32mul(p[0], p[0]), f32mul(p[1], p[1])),
        f32mul(p[2], p[2]));
      volatile float inverse = static_cast<float>(1.0 / std::sqrt(dot));
      for (float &component : p) component = f32mul(component, inverse);
      // Logical RGBA corresponding to native memory bytes z, y, x, 0.
      output[offset++] = sFtol(f32add(128.0f, f32mul(p[0], 127.0f)));
      output[offset++] = sFtol(f32add(128.0f, f32mul(p[1], 127.0f)));
      output[offset++] = sFtol(f32add(128.0f, f32mul(p[2], 127.0f)));
      output[offset++] = 0;
    }
  }
  return output;
}

static std::vector<uint8_t> makeAttenuationVolume() {
  constexpr int size = 32;
  const float scale = static_cast<float>(2.0f / (size - 2.0f));
  const float middle = size / 2.0f;
  std::vector<uint8_t> output(size * size * size * 4);
  size_t offset = 0;
  for (int z = 0; z < size; ++z) {
    const float vz = f32mul(z - middle, scale);
    for (int y = 0; y < size; ++y) {
      const float vy = f32mul(y - middle, scale);
      for (int x = 0; x < size; ++x) {
        const float vx = f32mul(x - middle, scale);
        float dot = f32add(f32add(f32mul(vx, vx), f32mul(vy, vy)),
          f32mul(vz, vz));
        const float attenuation = std::fmax(f32add(1.0f, -dot), 0.0f);
        const uint8_t byte = sFtol(f32mul(attenuation, 255.0f));
        for (int channel = 0; channel < 4; ++channel) output[offset++] = byte;
      }
    }
  }
  return output;
}

int main() {
  std::fesetround(FE_TONEAREST);
  const auto cube = makeCubeNormalizer();
  std::printf("cube %08x\n", fnv1a(cube.data(), cube.size()));
  for (int face = 0; face < 6; ++face) {
    const size_t offset = face * 64 * 64 * 4;
    std::printf("face%d %08x\n", face, fnv1a(cube.data() + offset, 64 * 64 * 4));
  }
  const auto volume = makeAttenuationVolume();
  std::printf("volume %08x\n", fnv1a(volume.data(), volume.size()));
}
