import assert from 'node:assert/strict';
import {
  AudioBlockProducer,
  AudioStream,
  renderSynthBlock,
  seekSynth,
} from '../src/audio.js';
import {
  DEBRIS_OUTPUT_PROCESSOR_NAME,
  StereoPcmResampler,
  createDebrisOutputProcessor,
} from '../src/audio_worklet_core.js';

const D = { AudioBlockProducer, AudioStream, renderSynthBlock, seekSynth };

let hardwareSampleRate = 12;
const nodes = [];
const workletModules = [];
class FakePort {
  constructor() { this.messages = []; this.onmessage = null; this.started = false; this.closed = false; }
  postMessage(message) { this.messages.push(message); }
  emit(message) { this.onmessage?.({ data: message }); }
  start() { this.started = true; }
  close() { this.closed = true; }
}
class FakeAudioWorkletNode {
  constructor(audioContext, name, options) {
    this.context = audioContext; this.name = name; this.options = options;
    this.port = new FakePort(); this.connected = false; nodes.push(this);
  }
  connect() { this.connected = true; }
  disconnect() { this.connected = false; }
}
class FakeAudioContext {
  constructor() {
    this.sampleRate = hardwareSampleRate; this.currentTime = 0; this.state = 'suspended';
    this.destination = {};
    this.audioWorklet = { addModule: async url => { workletModules.push(url); } };
  }
  async resume() { this.state = 'running'; }
  async close() { this.state = 'closed'; }
}

class MockSynth {
  constructor() { this.sample = 0; }
  reset() { this.sample = 0; }
  render(target, frames) {
    for (let index = 0; index < frames; index++, this.sample++) {
      target[index * 2] = this.sample;
      target[index * 2 + 1] = -this.sample;
    }
    return target;
  }
}

const synth = new MockSynth();
let block = D.renderSynthBlock(synth, 4);
assert.deepEqual(Array.from(block), [0, -0, 1, -1, 2, -2, 3, -3]);
D.seekSynth(synth, 7, 3);
assert.equal(synth.sample, 7);
block = D.renderSynthBlock(synth, 2);
assert.deepEqual(Array.from(block), [7, -7, 8, -8]);

const planar = { render: frames => [new Float32Array(frames).fill(0.25), new Float32Array(frames).fill(-0.5)] };
assert.deepEqual(Array.from(D.renderSynthBlock(planar, 2)), [0.25, -0.5, 0.25, -0.5]);

assert.throws(() => D.renderSynthBlock({}, 1), /neither render\(\) nor process/);

class FiniteSynth {
  constructor(sampleRate, song, alwaysLoud = false) {
    this.sampleRate = sampleRate; this.song = song; this.samplePosition = 0; this.alwaysLoud = alwaysLoud;
  }
  calcSongSamples() { return 8; }
  reset() { this.samplePosition = 0; return this; }
  seekSamples(sample) { this.reset(); this.samplePosition = sample; return sample; }
  render(target, frames) {
    for (let index = 0; index < frames; index++, this.samplePosition++) {
      // Two audible release samples follow the sequencer boundary at sample 8.
      const value = this.alwaysLoud || this.samplePosition < 10 ? 0.5 : 0;
      target[index * 2] = value; target[index * 2 + 1] = -value;
    }
    return target;
  }
}

const zeroTailProducer = new D.AudioBlockProducer(
  new FiniteSynth(10, { name: 'zero-tail' }),
  { sampleRate: 10, blockFrames: 4, tailSeconds: 0 },
);
assert.equal(zeroTailProducer.produceBlock().end, 4);
const zeroTailFinal = zeroTailProducer.produceBlock();
assert.equal(zeroTailFinal.end, 8);
assert.equal(zeroTailProducer.ended, true,
  'a zero-tail producer ends on the exact final block without another refill request');
assert.equal(zeroTailProducer.endSample, 8);

const factoryCalls = [];
const playerFactory = (song, options) => {
  factoryCalls.push({ song, sampleRate: options.sampleRate });
  return new FiniteSynth(options.sampleRate, song);
};

let endedCalls = 0;
const song = { name: 'rate-test' };
const stream = new D.AudioStream({
  sampleRate: 10, blockFrames: 4, queueBlocks: 3,
  tailSeconds: 1, tailSilenceSeconds: 0.4, tailSilenceThreshold: 0.01,
  onEnded: () => endedCalls++,
  playerFactory,
  AudioContextClass: FakeAudioContext,
  AudioWorkletNodeClass: FakeAudioWorkletNode,
});
await stream.init(new FiniteSynth(44100, song));
const node = nodes.at(-1);
assert.match(String(workletModules.at(-1)), /\/src\/audio_worklet\.js$/);
assert.deepEqual(factoryCalls, [{ song, sampleRate: 10 }],
  'a mismatched caller synth is recreated at the production rate, not the device rate');
assert.equal(stream.sampleRate, 10);
assert.equal(stream.synth.sampleRate, 10);
assert.equal(stream.outputSampleRate, 12);
assert.equal(stream.sequenceSamples, 8);
assert.equal(stream.durationSamples, 18);
assert.equal(node.options.processorOptions.queueBlocks, 3);
assert.equal(node.options.processorOptions.sourceSampleRate, 10);
assert.equal(node.options.processorOptions.outputSampleRate, 12);
assert.equal(node.port.messages.filter(message => message.type === 'block').length, 3);
assert.equal(node.port.messages.some(message => message.type === 'end'), false);

// Execute the static processor core itself: the explicit initial fill is counted
// as incoming (no duplicate request), and each consumed block is replenished
// immediately instead of waiting for a low-water burst.
class WorkletBase {
  constructor() { this.port = new FakePort(); }
}
assert.equal(DEBRIS_OUTPUT_PROCESSOR_NAME, 'debris-output');
const Processor = createDebrisOutputProcessor(WorkletBase);
const processor = new Processor({ processorOptions: { queueBlocks: 3 } });
processor.port.onmessage({ data: { type: 'pause', value: false } });
processor.process([], [[new Float32Array(4), new Float32Array(4)]]);
assert.equal(processor.port.messages.some(message => message.type === 'need'), false);
for (let index = 0; index < 3; index++) {
  processor.port.onmessage({ data: { type: 'block', data: new Float32Array(8).buffer } });
}
processor.process([], [[new Float32Array(4), new Float32Array(4)]]);
processor.process([], [[new Float32Array(4), new Float32Array(4)]]);
const needMessage = processor.port.messages.find(message => message.type === 'need');
assert.equal(needMessage.type, 'need'); assert.equal(needMessage.blocks, 1);

processor.port.messages.length = 0;
processor.port.onmessage({ data: { type: 'reset', sample: 100, incoming: 1 } });
processor.port.onmessage({ data: { type: 'block', data: new Float32Array(8).buffer } });
processor.port.onmessage({ data: { type: 'end', sample: 104 } });
processor.process([], [[new Float32Array(4), new Float32Array(4)]]);
const drainedMessage = processor.port.messages.find(message => message.type === 'drained');
assert.equal(drainedMessage.type, 'drained'); assert.equal(drainedMessage.sample, 104);

// Device-rate conversion happens only at the output processor. Across block
// boundaries, eight 8-Hz sink frames consume exactly four 4-Hz production
// frames and report the clock in the source coordinate.
const rateProcessor = new Processor({ processorOptions: {
  queueBlocks: 2, reportClock: true, sourceSampleRate: 4, outputSampleRate: 8,
} });
rateProcessor.port.onmessage({ data: { type: 'reset', sample: 0, incoming: 2 } });
rateProcessor.port.onmessage({ data: {
  type: 'block', start: 0, end: 2, frames: 2,
  data: new Float32Array([0, 0, 1, -1]).buffer,
} });
rateProcessor.port.onmessage({ data: {
  type: 'block', start: 2, end: 4, frames: 2,
  data: new Float32Array([2, -2, 3, -3]).buffer,
} });
rateProcessor.port.onmessage({ data: { type: 'end', sample: 4 } });
rateProcessor.port.onmessage({ data: { type: 'pause', value: false } });
const rateLeft = new Float32Array(8), rateRight = new Float32Array(8);
rateProcessor.process([], [[rateLeft, rateRight]]);
assert.deepEqual(Array.from(rateLeft), [0, 0.5, 1, 1.5, 2, 2.5, 3, 3]);
assert.deepEqual(Array.from(rateRight), [0, -0.5, -1, -1.5, -2, -2.5, -3, -3]);
assert.equal(rateProcessor.played, 4);
const rateClock = rateProcessor.port.messages.find(message => message.type === 'clock');
assert.equal(rateClock.sample, 4);
assert.equal(rateClock.receivedThrough, 4);
assert.equal(rateProcessor.port.messages.find(message => message.type === 'drained').sample, 4);

const realRateResampler = new StereoPcmResampler(44100, 48000);
realRateResampler.push(new Float32Array(44100 * 2));
realRateResampler.end();
assert.equal(realRateResampler.pull(new Float32Array(48000), new Float32Array(48000)), 48000);
assert.equal(realRateResampler.played, 44100,
  'one device second consumes exactly one production second');
assert.equal(realRateResampler.drained, true);

// Rational phase tracking is invariant to both producer and sink callback
// partitioning, including downsampling and non-integer ratios. Ending extends
// only the final source sample and emits ceil(N * sink/source) frames.
function renderPartitioned(sourceRate, outputRate, source, sourceParts, outputParts) {
  const resampler = new StereoPcmResampler(sourceRate, outputRate);
  const output = [];
  let sourceAt = 0, sourcePart = 0, outputPart = 0;
  const drainAvailable = () => {
    for (;;) {
      const frames = outputParts[outputPart++ % outputParts.length];
      const left = new Float32Array(frames), right = new Float32Array(frames);
      const written = resampler.pull(left, right);
      for (let frame = 0; frame < written; frame++) output.push(left[frame], right[frame]);
      if (written < frames) break;
    }
  };
  while (sourceAt < source.length / 2) {
    const frames = Math.min(sourceParts[sourcePart++ % sourceParts.length], source.length / 2 - sourceAt);
    resampler.push(source.slice(sourceAt * 2, (sourceAt + frames) * 2));
    sourceAt += frames;
    drainAvailable();
  }
  resampler.end();
  while (!resampler.drained) drainAvailable();
  return { pcm: new Float32Array(output), played: resampler.played };
}

const partitionSource = new Float32Array(37 * 2);
for (let frame = 0; frame < 37; frame++) {
  partitionSource[frame * 2] = Math.fround(Math.sin(frame * 0.31));
  partitionSource[frame * 2 + 1] = Math.fround(Math.cos(frame * 0.17));
}
for (const [sourceRate, outputRate] of [[44100, 48000], [48000, 44100], [7, 11], [11, 7]]) {
  const wholeRate = renderPartitioned(sourceRate, outputRate, partitionSource, [37], [128]);
  const splitRate = renderPartitioned(sourceRate, outputRate, partitionSource, [1, 5, 2, 11, 3], [1, 7, 2, 5]);
  assert.deepEqual(splitRate.pcm, wholeRate.pcm,
    `${sourceRate}->${outputRate} PCM is independent of queue/callback partitions`);
  assert.equal(splitRate.played, 37);
  assert.equal(splitRate.pcm.length / 2, Math.ceil(37 * outputRate / sourceRate));
}

// The legacy ScriptProcessor fallback uses the identical output-only
// conversion and never asks the synth to adopt the device rate.
let scriptNode = null;
class ScriptAudioContext {
  constructor() {
    this.sampleRate = 8; this.currentTime = 0; this.state = 'suspended';
    this.destination = {};
  }
  createScriptProcessor(frames) {
    const channels = [new Float32Array(frames), new Float32Array(frames)];
    scriptNode = {
      onaudioprocess: null,
      channels,
      connect() {}, disconnect() {},
      process() {
        this.onaudioprocess({ outputBuffer: { getChannelData: index => channels[index] } });
      },
    };
    return scriptNode;
  }
  async resume() { this.state = 'running'; }
  async close() { this.state = 'closed'; }
}
class RampSynth {
  constructor() { this.sampleRate = 4; this.song = {}; this.samplePosition = 0; }
  calcSongSamples() { return 4; }
  reset() { this.samplePosition = 0; }
  render(target, frames) {
    for (let frame = 0; frame < frames; frame++, this.samplePosition++) {
      target[frame * 2] = this.samplePosition;
      target[frame * 2 + 1] = -this.samplePosition;
    }
    return target;
  }
}
const scriptStream = new D.AudioStream({
  sampleRate: 4, blockFrames: 4, tailSeconds: 0,
  AudioContextClass: ScriptAudioContext,
  AudioWorkletNodeClass: null,
});
await scriptStream.init(new RampSynth());
await scriptStream.start();
scriptNode.process();
assert.equal(scriptStream.producerBackend, 'script-processor');
assert.equal(scriptStream.sampleRate, 4);
assert.equal(scriptStream.outputSampleRate, 8);
assert.deepEqual(Array.from(scriptNode.channels[0].subarray(0, 8)),
  [0, 0.5, 1, 1.5, 2, 2.5, 3, 3]);
assert.deepEqual(Array.from(scriptNode.channels[1].subarray(0, 8)),
  [-0, -0.5, -1, -1.5, -2, -2.5, -3, -3]);
assert.equal(scriptStream.sample, 4);
assert.equal(scriptStream.drained, true);
await scriptStream.close();

// A Worker producer uses its transferred port exclusively for need/block/end
// traffic. Priming and lifecycle reports continue over the built-in worklet
// port, and generation tags reject stale blocks.
const directProcessor = new Processor({ processorOptions: { queueBlocks: 3, reportClock: false } });
const directPort = new FakePort();
directProcessor.port.onmessage({ data: {
  type: 'producer-port', port: directPort, generation: 7, sample: 0,
} });
assert.equal(directPort.messages[0].type, 'need');
assert.equal(directPort.messages[0].generation, 7);
assert.equal(directPort.messages[0].blocks, 3);
assert.equal(directProcessor.port.messages.some(message => message.type === 'need'), false);
directPort.onmessage({ data: {
  type: 'block', generation: 6, start: 0, end: 4, frames: 4,
  data: new Float32Array(8).fill(99).buffer,
} });
for (let blockIndex = 0; blockIndex < 3; blockIndex++) {
  const data = new Float32Array(8).fill(blockIndex + 1);
  directPort.onmessage({ data: {
    type: 'block', generation: 7,
    start: blockIndex * 4, end: (blockIndex + 1) * 4, frames: 4,
    data: data.buffer,
  } });
}
const primedMessage = directProcessor.port.messages.find(message => message.type === 'primed');
assert.equal(primedMessage.generation, 7);
assert.equal(primedMessage.receivedThrough, 12);
const directLeft = new Float32Array(4), directRight = new Float32Array(4);
directProcessor.port.onmessage({ data: { type: 'pause', value: false } });
directProcessor.process([], [[directLeft, directRight]]);
assert.deepEqual(Array.from(directLeft), [1, 1, 1, 1]);
assert.deepEqual(Array.from(directRight), [1, 1, 1, 1]);
assert.equal(directPort.messages.at(-1).type, 'need');
assert.equal(directPort.messages.at(-1).generation, 7);
assert.equal(directPort.messages.at(-1).blocks, 1);
assert.equal(directProcessor.port.messages.some(message => message.type === 'clock'), false);
directProcessor.port.onmessage({ data: { type: 'detach-producer' } });
assert.equal(directPort.closed, true);

// A direct-port render error is FIFO-ordered after already generated PCM.
// The processor drains that valid queue exactly once and rejects any block
// arriving after the terminal error.
const errorProcessor = new Processor({ processorOptions: { queueBlocks: 2, reportClock: false } });
const errorPort = new FakePort();
errorProcessor.port.onmessage({ data: {
  type: 'producer-port', port: errorPort, generation: 3, sample: 0,
} });
errorPort.onmessage({ data: {
  type: 'block', generation: 3, start: 0, end: 4, frames: 4,
  data: new Float32Array(8).fill(0.5).buffer,
} });
errorPort.onmessage({ data: {
  type: 'error', generation: 3, name: 'Error', message: 'synthetic render failure',
} });
assert.equal(errorProcessor.port.messages.some(message => message.type === 'producer-error'), true);
assert.equal(errorProcessor.port.messages.some(message => message.type === 'drained'), false);
errorPort.onmessage({ data: {
  type: 'block', generation: 3, start: 4, end: 8, frames: 4,
  data: new Float32Array(8).fill(0.75).buffer,
} });
errorProcessor.port.onmessage({ data: { type: 'pause', value: false } });
errorProcessor.process([], [[new Float32Array(4), new Float32Array(4)]]);
const orderedDrain = errorProcessor.port.messages.filter(message => message.type === 'drained');
assert.equal(orderedDrain.length, 1);
assert.equal(orderedDrain[0].sample, 4);

// Pause captures the interpolated sample before changing the paused flag.
stream.paused = false; stream.clockSample = 0; stream.clockTime = 0;
stream.context.currentTime = 1; stream.renderedSample = 12;
stream.pause(true);
assert.equal(stream.clockSample, 10);

// One more quiet block satisfies the 4-frame silence window. The two audible
// post-sequence samples were queued, and completion waits for worklet drain.
await stream.pump(2);
const blocks = node.port.messages.filter(message => message.type === 'block');
assert.equal(blocks.length, 4);
assert.deepEqual(Array.from(new Float32Array(blocks[2].data)), [0.5, -0.5, 0.5, -0.5, 0, -0, 0, -0]);
assert.equal(stream.endSample, 16);
assert.equal(stream.durationSamples, 16);
assert.equal(stream.ended, true);
assert.equal(stream.drained, false);
assert.equal(endedCalls, 0);
assert.equal(node.port.messages.filter(message => message.type === 'end').length, 1);
node.port.emit({ type: 'drained', sample: 16 });
assert.equal(stream.drained, true);
assert.equal(stream.sample, 16);
assert.equal(endedCalls, 1);
node.port.emit({ type: 'drained', sample: 16 });
assert.equal(endedCalls, 1);

// A non-decaying custom synth still terminates at the documented hard tail
// ceiling, using a partial final block rather than overshooting it.
const hardStream = new D.AudioStream({
  sampleRate: 10, blockFrames: 4, queueBlocks: 1,
  tailSeconds: 1, tailSilenceSeconds: 0.4, tailSilenceThreshold: 0.01,
  playerFactory,
  AudioContextClass: FakeAudioContext,
  AudioWorkletNodeClass: FakeAudioWorkletNode,
});
await hardStream.init(new FiniteSynth(10, { name: 'hard-tail' }, true));
while (!hardStream.ended) await hardStream.pump(1);
const hardNode = nodes.at(-1);
const hardBlocks = hardNode.port.messages.filter(message => message.type === 'block');
assert.equal(hardStream.endSample, 18);
assert.equal(new Float32Array(hardBlocks.at(-1).data).length, 4);
assert.equal(hardNode.port.messages.filter(message => message.type === 'end').length, 1);

// A browser may reject an explicitly requested context rate. Retry only that
// standards-defined negotiation failure at the default sink rate; the synth
// and public sample clock remain in the production coordinate.
const retryContextOptions = [];
class UnsupportedRateAudioContext extends FakeAudioContext {
  constructor(options = {}) {
    retryContextOptions.push({ ...options });
    if ('sampleRate' in options) {
      const error = new Error('requested context rate is unavailable');
      error.name = 'NotSupportedError';
      throw error;
    }
    super();
  }
}
const retryStream = new D.AudioStream({
  sampleRate: 10, blockFrames: 4, queueBlocks: 1, tailSeconds: 0,
  AudioContextClass: UnsupportedRateAudioContext,
  AudioWorkletNodeClass: FakeAudioWorkletNode,
});
await retryStream.init(new FiniteSynth(10, { name: 'context-rate-retry' }));
assert.deepEqual(retryContextOptions, [
  { sampleRate: 10, latencyHint: 'playback' },
  { latencyHint: 'playback' },
]);
assert.equal(retryStream.sampleRate, 10);
assert.equal(retryStream.outputSampleRate, 12);
const retryNode = nodes.at(-1);
assert.equal(retryNode.options.processorOptions.sourceSampleRate, 10);
assert.equal(retryNode.options.processorOptions.outputSampleRate, 12);
await retryStream.close();

let securityContextAttempts = 0;
class SecurityFailureAudioContext {
  constructor() {
    securityContextAttempts++;
    const error = new Error('audio permission denied');
    error.name = 'SecurityError';
    throw error;
  }
}
const securityFailureStream = new D.AudioStream({
  sampleRate: 10,
  AudioContextClass: SecurityFailureAudioContext,
  AudioWorkletNodeClass: FakeAudioWorkletNode,
});
await assert.rejects(
  securityFailureStream.init(new FiniteSynth(10, { name: 'no-context-retry' })),
  error => error?.name === 'SecurityError',
);
assert.equal(securityContextAttempts, 1,
  'non-rate AudioContext construction failures are never retried');
assert.equal(securityFailureStream.context, null,
  'a failed constructor leaves no WebAudio object requiring cleanup');

console.log('audio stream adapter and lifecycle tests passed');
