import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';

import * as v2 from '../src/v2.js';
import { renderSynthBlock } from '../src/audio.js';

const D = { ...v2, renderSynthBlock };
// The party executable payload stores its V2M directly after the 40-byte KX
// header; the little-endian byte length is the final header word.
const partyKX = await readFile(new URL('../assets/debris_party.kx', import.meta.url));
const partyLength = new DataView(partyKX.buffer, partyKX.byteOffset, partyKX.byteLength).getUint32(36, true);
const partyBytes = partyKX.subarray(40, 40 + partyLength);
const loaderBytes = await readFile(new URL('../assets/debris_loader.v2m', import.meta.url));
const NATIVE_FIRST_SECOND_SHA256 =
  '02546c706545c60b079b71e7701b0e87b9921e574bac3f299cff3221bc5cd337';

const party = D.parseV2M(partyBytes);
assert.equal(party.version, 6);
assert.equal(party.timediv, 128);
assert.equal(party.maxtime, 148132);
assert.equal(party.globalCount, 1);
assert.equal(party.globalEvents[0].tempo, 371519);
assert.equal(party.patchCount, 116);
assert.equal(party.speechBytes.length, 23);
assert.equal(party.lyrics[0], 'aoaoaooaoaoaoa');
assert.equal(party.events.length, 37307);
assert.deepEqual(Array.from(party.globals), [37, 42, 49, 63, 127, 106, 55, 55, 18, 32, 92, 0, 127, 1, 1, 1, 0, 90, 14, 30, 10, 64, 0]);
assert.deepEqual(
  party.events.slice(1, 5).map(event => [event.kind, event.channel, event.values[0]]),
  [['program', 0, 0], ['program', 1, 3], ['program', 2, 39], ['program', 3, 59]],
);
assert.deepEqual(
  party.events.find(event => event.kind === 'note'),
  { time: 1, kind: 'note', channel: 14, controller: 0, values: [48, 126], order: 0 },
);
// v2seq builds same-tick MIDI in channel order, then uses program/CC/bend/note
// priority within each channel. At this production boundary ch3 must allocate
// a voice before the later ch4/ch5 program changes free theirs.
assert.deepEqual(party.events.filter(event => event.time === 28267).map(event => [
  event.kind, event.channel, event.controller, ...event.values,
]), [
  ['note', 3, 0, 48, 76],
  ['program', 4, 0, 47],
  ['program', 5, 0, 46],
  ['note', 14, 0, 48, 0],
  ['note', 14, 0, 48, 126],
]);

const loader = D.parseV2M(loaderBytes);
assert.equal(loader.version, 6);
assert.equal(loader.timediv, 128);
assert.equal(loader.maxtime, 19997);
assert.equal(loader.patchCount, 16);
assert.equal(loader.speechBytes.length, 4);
assert.equal(loader.events.length, 140);

// A zero delta in a non-note stream makes v2seq run another zero-sample Tick:
// global, then every channel's program/CC/bend/note pass, then the repeated
// rows. Notes are the exception because the source consumes all of them in a
// while-loop during the first pass.
{
  const u32 = value => {
    const bytes = Buffer.alloc(4);
    bytes.writeUInt32LE(value >>> 0);
    return bytes;
  };
  const stream = columns => Buffer.concat(columns.map(column => Buffer.from(column)));
  const channel = ({ notes, programs = [], cc1 = [] }) => {
    const parts = [u32(notes[0].length), stream(notes)];
    parts.push(u32(programs.length ? programs[0].length : 0));
    if (programs.length) parts.push(stream(programs));
    parts.push(u32(0)); // pitch bend
    for (let controller = 1; controller <= 7; controller++) {
      const rows = controller === 1 ? cc1 : [];
      parts.push(u32(rows.length ? rows[0].length : 0));
      if (rows.length) parts.push(stream(rows));
    }
    return Buffer.concat(parts);
  };
  const globalRows = Buffer.concat([
    stream([[0, 0], [0, 0], [0, 0]]),
    u32(1000), u32(2000),
    Buffer.from([4, 4, 4, 4, 8, 8]),
  ]);
  const channels = [
    channel({
      notes: [[0, 0], [0, 0], [0, 0], [60, 1], [100, 0]],
      programs: [[0, 0], [0, 0], [0, 0], [1, 1]],
      cc1: [[0, 0], [0, 0], [0, 0], [10, 10]],
    }),
    channel({
      notes: [[0], [0], [0], [64], [100]],
      programs: [[0], [0], [0], [3]],
    }),
    ...Array.from({ length: 14 }, () => u32(0)),
  ];
  const duplicateRows = Buffer.concat([
    u32(128), u32(0), u32(2), globalRows, ...channels,
    u32(loader.globals.length), Buffer.from(loader.globals),
    u32(loader.patchBlock.length), Buffer.from(loader.patchBlock),
    u32(0),
  ]);
  const ordered = D.parseV2M(duplicateRows).events.map(event => [
    event.kind, event.channel ?? -1, event.values?.[0] ?? event.tempo,
    event.sameTimePass || 0,
  ]);
  assert.deepEqual(ordered, [
    ['global', -1, 1000, 0],
    ['program', 0, 1, 0], ['controller', 0, 10, 0],
    ['note', 0, 60, 0], ['note', 0, 61, 0],
    ['program', 1, 3, 0], ['note', 1, 64, 0],
    ['global', -1, 2000, 1],
    ['program', 0, 2, 1], ['controller', 0, 20, 1],
  ]);
}

// These are the exact v2seq event-clock lengths (also returned by the pinned
// native reference), before any optional release-tail playback.
assert.equal(new D.V2Player(party).calcSongSamples(), 18934066);
assert.equal(new D.V2Player(loader).calcSongSamples(), 2559552);
assert.ok(D.V2Player.prototype.render.length >= 2);
assert.ok(D.V2Synth.prototype.render.length >= 2);

const referenceRandom = new D.V2Rand(1);
assert.deepEqual(Array.from({ length: 6 }, () => referenceRandom.next()), [
  1804289383, 846930886, 1681692777, 1714636915, 1957747793, 424238335,
]);

// Streaming is invariant to host callback block size.
const frames = 8192;
const whole = new Float32Array(frames * 2);
new D.V2Player(party).render(whole, frames);
const chunked = new Float32Array(frames * 2);
const chunkPlayer = D.createV2Player(party);
let position = 0;
for (const count of [1, 127, 128, 3, 511, 64, 1000, 2048, 4310]) {
  const amount = Math.min(count, frames - position);
  if (!amount) break;
  chunkPlayer.render(chunked, amount, position);
  position += amount;
}
if (position < frames) chunkPlayer.render(chunked, frames - position, position);
assert.deepEqual(chunked, whole);

// The browser seam selects render(target, frames) through Function.length;
// cover odd/non-power-of-two callback sizes as well as the returned target.
const seamFrames = 10007;
const seamWhole = new Float32Array(seamFrames * 2);
const seamWholePlayer = new D.V2Player(party);
assert.equal(seamWholePlayer.render(seamWhole), seamWhole);
const seamChunked = new Float32Array(seamFrames * 2);
const seamPlayer = new D.V2Player(party);
let seamPosition = 0;
for (const count of [17, 129, 257, 1001, 2049, 4093, 2461]) {
  const amount = Math.min(count, seamFrames - seamPosition);
  if (!amount) break;
  const block = D.renderSynthBlock(seamPlayer, amount);
  seamChunked.set(block, seamPosition * 2);
  seamPosition += amount;
}
if (seamPosition < seamFrames) seamChunked.set(D.renderSynthBlock(seamPlayer, seamFrames - seamPosition), seamPosition * 2);
assert.deepEqual(seamChunked, seamWhole);

// Reset, seek and replay-coordinate snapshots are deterministic.
chunkPlayer.reset();
const reset = new Float32Array(frames * 2);
chunkPlayer.render(reset, frames);
assert.deepEqual(reset, whole);
const snapshotPlayer = new D.V2Player(party);
snapshotPlayer.seekSamples(4096);
const snapshot = snapshotPlayer.snapshot();
const tailA = snapshotPlayer.renderFrames(2048);
snapshotPlayer.restore(snapshot);
const tailB = snapshotPlayer.renderFrames(2048);
assert.deepEqual(tailB, tailA);
assert.equal(snapshotPlayer.samplePosition, 6144);
snapshotPlayer.reset();
snapshotPlayer.restore(snapshot);
const tailC = snapshotPlayer.renderFrames(2048);
assert.deepEqual(tailC, tailA);
assert.ok(snapshot.stateBytes > 0 && snapshot.stateBytes < 2 * 1024 * 1024);
assert.throws(
  () => new D.V2Player(party, { sampleRate: 48000 }).restore(snapshot),
  /Incompatible V2Player snapshot/,
);

const seekReference = new D.V2Player(party).renderFrames(7000);
const seekPlayer = new D.V2Player(party);
assert.equal(seekPlayer.seekSamples(3333), 3333);
assert.equal(seekPlayer.samplePosition, 3333);
assert.deepEqual(seekPlayer.renderFrames(2049), seekReference.slice(3333 * 2, (3333 + 2049) * 2));
assert.throws(() => seekPlayer.seekSamples(Infinity), /finite sample position/);

// Periodic seek checkpoints preserve both PCM and every sequencer boundary
// field across a deliberately scrambled forward/backward seek sequence.  The
// reference is one checkpoint-free linear render, not another seek path.
{
  const maximum = 180000, probeFrames = 257;
  const boundaryPlayer = new D.V2Player(party, { checkpointMemoryBytes: 0 });
  const eventBoundaries = [];
  const processTick = boundaryPlayer.processTick;
  boundaryPlayer.processTick = function () {
    const result = processTick.call(this);
    eventBoundaries.push(this.samplePosition);
    return result;
  };
  const linearPCM = boundaryPlayer.renderFrames(maximum + probeFrames);
  const uniqueBoundaries = [...new Set(eventBoundaries)].filter(sample => sample > 0 && sample < maximum);

  const positions = new Set([0, 1, 127, 128, 129, 4095, 4096, 4097, maximum - probeFrames]);
  for (const boundary of uniqueBoundaries) {
    for (const delta of [-1, 0, 1]) if (boundary + delta >= 0 && boundary + delta + probeFrames <= maximum) positions.add(boundary + delta);
  }
  let random = 0x041deb15;
  for (let i = 0; i < 12; i++) {
    random = (Math.imul(random, 1664525) + 1013904223) >>> 0;
    positions.add(random % (maximum - probeFrames));
  }

  const timingFields = player => ({
    samplePosition: player.samplePosition,
    eventIndex: player.eventIndex,
    tickTime: player.tickTime,
    nextTick: player.nextTick,
    tempoSamples: player.tempoSamples,
    tempoNumerator: player.tempoNumerator,
    tempoDenominator: player.tempoDenominator,
    ticksPerQuarter: player.ticksPerQuarter,
    sampleRemainder: player.sampleRemainder,
    samplesToEvent: player.samplesToEvent,
    isPlaying: player.isPlaying,
    synthFrameRead: player.synth.frameRead,
  });
  const expectedTiming = new Map();
  const timingOracle = new D.V2Player(party, { checkpointMemoryBytes: 0 });
  for (const at of [...positions].sort((a, b) => a - b)) {
    timingOracle.seekSamples(at);
    expectedTiming.set(at, timingFields(timingOracle));
  }

  const checkpointPlayer = new D.V2Player(party, {
    checkpointIntervalSamples: 4096,
    checkpointMemoryBytes: 8 * 1024 * 1024,
  });
  checkpointPlayer.seekSamples(maximum);
  const scrambled = [...positions].sort((a, b) => ((a * 2654435761) >>> 0) - ((b * 2654435761) >>> 0));
  for (const at of scrambled) {
    checkpointPlayer.seekSamples(at);
    assert.deepEqual(timingFields(checkpointPlayer), expectedTiming.get(at), `V2 timing state at sample ${at}`);
    const actual = checkpointPlayer.renderFrames(probeFrames);
    const expected = linearPCM.slice(at * 2, (at + probeFrames) * 2);
    assert.deepEqual(actual, expected, `V2 checkpoint PCM at sample ${at}`);
  }
  const stats = checkpointPlayer.checkpointStats();
  assert.ok(stats.count > 1);
  assert.ok(stats.bytes <= stats.maxBytes);
  assert.ok(stats.samples.every((sample, index) => index === 0 || sample > stats.samples[index - 1]));

  checkpointPlayer.seekSamples(12000);
  const branchSnapshot = checkpointPlayer.snapshot();
  checkpointPlayer.seekSamples(maximum);
  assert.ok(checkpointPlayer.checkpointStats().samples.some(sample => sample > branchSnapshot.samplePosition));
  checkpointPlayer.restore(branchSnapshot);
  assert.ok(checkpointPlayer.checkpointStats().samples.every(sample => sample < branchSnapshot.samplePosition));

  // Hardware-rate recreation uses a different synth frame size and tempo
  // integer, so checkpoint state is independently verified at 48 kHz.
  const rate = 48000, rateMaximum = 60000, rateProbe = 193;
  const rateReference = new D.V2Player(loader, { sampleRate: rate, checkpointMemoryBytes: 0 })
    .renderFrames(rateMaximum + rateProbe);
  const ratePlayer = new D.V2Player(loader, {
    sampleRate: rate, checkpointIntervalSamples: 3000, checkpointMemoryBytes: 8 * 1024 * 1024,
  });
  ratePlayer.seekSamples(rateMaximum);
  for (const at of [59900, 0, 279, 280, 281, 2999, 3000, 3001, 17321, 48000, 11111]) {
    ratePlayer.seekSamples(at);
    assert.deepEqual(
      ratePlayer.renderFrames(rateProbe),
      rateReference.slice(at * 2, (at + rateProbe) * 2),
      `48 kHz V2 checkpoint PCM at sample ${at}`,
    );
  }
}

// Reaching calcSongSamples exactly applies trailing-edge events and marks the
// sequencer stopped without needing a one-sample probe render.
const tinySong = { ...loader, events: loader.events.filter(event => event.time <= 1) };
const tinyPlayer = new D.V2Player(tinySong);
const tinyLength = tinyPlayer.calcSongSamples();
assert.ok(tinyLength > 0);
const tinyTarget = new Float32Array(tinyLength * 2);
assert.equal(tinyPlayer.render(tinyTarget, tinyLength), tinyTarget);
assert.equal(tinyPlayer.samplePosition, tinyLength);
assert.equal(tinyPlayer.isPlaying, false);

// The digest was generated by the pinned native v2redux source and reproduced
// by a temporary Wasm build of that same C++ core. The raw audit capture lives
// under ignored work/, so a clean checkout pins its byte digest without adding
// 352,800 bytes of generated PCM. tools/oracles/v2_render_oracle.cpp can
// independently regenerate the capture when a word-by-word audit is wanted.
const second = new Float32Array(44100 * 2);
const player = new D.V2Player(party);
player.render(second, 44100);
assert.ok(second.every(Number.isFinite));
assert.equal(second.findIndex(value => value !== 0), 436);
let peak = 0;
for (const value of second) peak = Math.max(peak, Math.abs(value));
assert.equal(peak, 0.09395445883274078);
const hash = createHash('sha256').update(new Uint8Array(second.buffer)).digest('hex');
assert.equal(hash, NATIVE_FIRST_SECOND_SHA256);

// The synth can also be driven without the file sequencer.
const synth = new D.V2Synth(loader);
synth.processMIDI(new Uint8Array([0xc0, 0, 0x90, 48, 127, 0xfd]));
const direct = new Float32Array(512 * 2);
synth.render(direct, 512);
assert.ok(direct.every(Number.isFinite));
assert.ok(direct.some(value => value !== 0));
assert.equal(synth.getPolyphony()[16], 1);

// The source's stereo distortion dispatcher intentionally runs Moog modes as
// one mono filter over interleaved L/R words (unlike modes 5..9, which own a
// filter per channel). Lock that unusual mode-10 state sharing explicitly.
const moogDistortion = new D.V2Synth(loader).channelStrips[0].distortion;
moogDistortion.set(new Float32Array([10, 32, 100, 64]), 0);
const moogLeft = new Float32Array([1, 0, 0, 0]);
const moogRight = new Float32Array(4);
moogDistortion.renderStereo(moogLeft, moogRight, 4);
assert.deepEqual(Array.from(moogLeft), [
  0.0007031011045910418, 0.0241440087556839, 0.0657389909029007, 0.07678461074829102,
]);
assert.deepEqual(Array.from(moogRight), [
  0.007351545616984367, 0.04617859050631523, 0.07694916427135468, 0.0648886114358902,
]);

assert.throws(() => D.parseV2M(new Uint8Array(8)), /Truncated/);

console.log(JSON.stringify({
  party: { patches: party.patchCount, events: party.events.length, samples: 18934066 },
  loader: { patches: loader.patchCount, events: loader.events.length, samples: 2559552 },
  jsFirstSecondSHA256: hash,
  pinnedNativeFirstSecondSHA256: NATIVE_FIRST_SECOND_SHA256,
  jsPeak: peak,
}, null, 2));
