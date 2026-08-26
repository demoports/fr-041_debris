// Offline renderer for auditing src/v2.js against the pinned v2redux source.
// This tool is not part of the browser build. Build, for example, with:
//   c++ -std=c++17 -O2 -ffp-contract=off -DV2_RONAN=0 \
//     tools/oracles/v2_render_oracle.cpp vendor/v2redux/src/{v2core,v2seq,v2load,v2player}.cpp \
//     -Ivendor/v2redux/src -o /tmp/v2_render_oracle

#include "v2redux.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

bool readFile(const char *path, std::vector<unsigned char> &bytes)
{
  FILE *file = std::fopen(path, "rb");
  if (!file) return false;
  if (std::fseek(file, 0, SEEK_END) || std::ftell(file) < 0) {
    std::fclose(file);
    return false;
  }
  const long size = std::ftell(file);
  std::rewind(file);
  bytes.resize(static_cast<size_t>(size));
  const bool ok = bytes.empty() || std::fread(bytes.data(), 1, bytes.size(), file) == bytes.size();
  std::fclose(file);
  return ok;
}

bool writeFile(const char *path, const void *data, size_t size)
{
  FILE *file = std::fopen(path, "wb");
  if (!file) return false;
  const bool ok = !size || std::fwrite(data, 1, size, file) == size;
  const bool closed = std::fclose(file) == 0;
  return ok && closed;
}

bool parseUnsigned(const char *text, uint32_t &value)
{
  errno = 0;
  char *end = nullptr;
  const unsigned long long parsed = std::strtoull(text, &end, 0);
  if (errno || !end || *end || parsed > UINT32_MAX) return false;
  value = static_cast<uint32_t>(parsed);
  return true;
}

} // namespace

int main(int argc, char **argv)
{
  if (argc < 4 || argc > 5) {
    std::fprintf(stderr, "usage: %s INPUT.v2m OUTPUT.f32 FRAMES [CHANNEL_MASK]\n", argv[0]);
    return 2;
  }
  uint32_t frames = 0, mask = 0xffffu;
  if (!parseUnsigned(argv[3], frames) || (argc == 5 && !parseUnsigned(argv[4], mask))) {
    std::fprintf(stderr, "invalid frame count or channel mask\n");
    return 2;
  }
  std::vector<unsigned char> song;
  if (!readFile(argv[1], song)) {
    std::perror(argv[1]);
    return 1;
  }
  v2redux::Player player;
  if (player.open(song.data(), song.size()) != v2redux::Result::OK) {
    std::fprintf(stderr, "v2redux rejected the input\n");
    return 1;
  }
  player.setChannelMask(mask);
  player.play();
  std::vector<float> pcm(static_cast<size_t>(frames) * 2);
  player.render(pcm.data(), frames);
  if (!writeFile(argv[2], pcm.data(), pcm.size() * sizeof(float))) {
    std::perror(argv[2]);
    return 1;
  }
  return 0;
}
