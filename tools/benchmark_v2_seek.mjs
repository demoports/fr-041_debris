#!/usr/bin/env node
// Exact production-song checkpoint benchmark.  This intentionally includes a
// cold linear oracle so the reported backward-seek PCM is byte-verified.

import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { performance } from 'node:perf_hooks';

import { parseV2M, V2Player } from '../src/v2.js';
const options = new Map();
for (const argument of process.argv.slice(2)) {
  const match = /^--([^=]+)=(.+)$/.exec(argument);
  if (match) options.set(match[1], match[2]);
}
const numberOption = (name, fallback) => {
  const value = Number(options.get(name));
  return Number.isFinite(value) && value >= 0 ? value : fallback;
};
const sampleRate = Math.floor(numberOption('sample-rate', 44100));
const fromSeconds = numberOption('from', 400);
const toSeconds = Math.min(fromSeconds, numberOption('to', 350));
const verifyFrames = Math.max(1, Math.floor(numberOption('verify-frames', 2048)));
const memoryMiB = numberOption('memory-mib', 64);

const kx = await readFile(new URL('../assets/debris_party.kx', import.meta.url));
const view = new DataView(kx.buffer, kx.byteOffset, kx.byteLength);
const tuneLength = view.getUint32(36, true);
const song = parseV2M(kx.subarray(40, 40 + tuneLength));
const fromSample = Math.floor(fromSeconds * sampleRate);
const toSample = Math.floor(toSeconds * sampleRate);

const player = new V2Player(song, {
  sampleRate,
  checkpointMemoryBytes: Math.floor(memoryMiB * 1024 * 1024),
});
let started = performance.now();
player.seekSamples(fromSample);
const forwardMilliseconds = performance.now() - started;
const before = player.checkpointStats();
let maximumGapSamples = 0, previousCheckpoint = 0;
for (const sample of before.samples) {
  maximumGapSamples = Math.max(maximumGapSamples, sample - previousCheckpoint);
  previousCheckpoint = sample;
}

started = performance.now();
player.seekSamples(toSample);
const backwardMilliseconds = performance.now() - started;
const checkpointPCM = player.renderFrames(verifyFrames);

const cold = new V2Player(song, { sampleRate, checkpointMemoryBytes: 0 });
started = performance.now();
cold.seekSamples(toSample);
const coldMilliseconds = performance.now() - started;
const coldPCM = cold.renderFrames(verifyFrames);
assert.deepEqual(checkpointPCM, coldPCM);
globalThis.gc?.();
const processMemory = process.memoryUsage();

console.log(JSON.stringify({
  ok: true,
  sampleRate,
  fromSeconds,
  toSeconds,
  verifyFrames,
  forwardMilliseconds: Number(forwardMilliseconds.toFixed(3)),
  backwardMilliseconds: Number(backwardMilliseconds.toFixed(3)),
  coldMilliseconds: Number(coldMilliseconds.toFixed(3)),
  speedup: Number((coldMilliseconds / Math.max(backwardMilliseconds, 0.001)).toFixed(2)),
  checkpoints: {
    count: before.count,
    memoryMiB: Number((before.bytes / 1048576).toFixed(3)),
    capMiB: Number((before.maxBytes / 1048576).toFixed(3)),
    firstSeconds: before.samples.length ? Number((before.samples[0] / sampleRate).toFixed(3)) : null,
    lastSeconds: before.samples.length ? Number((before.samples.at(-1) / sampleRate).toFixed(3)) : null,
    maximumReplaySeconds: Number((maximumGapSamples / sampleRate).toFixed(3)),
  },
  processMemoryMiB: {
    rss: Number((processMemory.rss / 1048576).toFixed(3)),
    heapUsed: Number((processMemory.heapUsed / 1048576).toFixed(3)),
    external: Number((processMemory.external / 1048576).toFixed(3)),
  },
}, null, 2));
