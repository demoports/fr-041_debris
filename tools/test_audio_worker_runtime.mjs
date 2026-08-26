import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createAudioWorkerController } from '../src/audio_worker_core.js';

class FakePort {
  constructor() {
    this.messages = [];
    this.transfers = [];
    this.onmessage = null;
    this.onmessageerror = null;
    this.started = false;
    this.closed = false;
  }
  postMessage(message, transfer = []) {
    this.messages.push(message);
    this.transfers.push(transfer);
  }
  start() { this.started = true; }
  close() { this.closed = true; }
  emit(message) { this.onmessage?.({ data: message }); }
}

function createWorkerHarness() {
  const controlMessages = [];
  let closed = false;
  const scope = {
    postMessage: message => controlMessages.push(message),
    close: () => { closed = true; },
  };
  const controller = createAudioWorkerController(scope);
  return { controller, controlMessages, isClosed: () => closed };
}

const loaderBytes = readFileSync(new URL('../assets/debris_loader.v2m', import.meta.url));
const worker = createWorkerHarness();
const loaderBuffer = new Uint8Array(loaderBytes).buffer;

worker.controller.handleMessage({ data: {
  type: 'prepare', requestId: 1, song: loaderBuffer,
  playerOptions: { sampleRate: 48000, checkpointMemoryBytes: 1234 },
  producerOptions: {
    sampleRate: 48000, blockFrames: 64, tailSeconds: 0,
    tailSilenceSeconds: 0.25, tailSilenceThreshold: 1e-5,
  },
} });
const ready = worker.controlMessages.find(message => message.type === 'ready');
assert.ok(ready, JSON.stringify(worker.controlMessages));
assert.equal(ready.replyTo, 1);
assert.equal(ready.renderedSample, 0);
assert.ok(ready.sequenceSamples > 0);

const port = new FakePort();
worker.controller.handleMessage({ data: {
  type: 'connect', requestId: 2, generation: 7, port,
} });
assert.equal(port.started, true);
assert.equal(worker.controlMessages.find(message => message.type === 'connected').replyTo, 2);

port.emit({ type: 'need', generation: 6, blocks: 2 });
assert.equal(port.messages.length, 0, 'stale generations must not render PCM');
port.emit({ type: 'need', generation: 7, blocks: 2 });
const blocks = port.messages.filter(message => message.type === 'block');
assert.equal(blocks.length, 2);
assert.deepEqual(blocks.map(block => [block.start, block.end, block.frames]), [
  [0, 64, 64], [64, 128, 64],
]);
assert.ok(blocks.every(block => block.data?.byteLength === 64 * 2 * 4));
assert.ok(port.transfers.every((transfer, index) =>
  port.messages[index].type !== 'block' || transfer[0] === port.messages[index].data));
assert.equal(worker.controlMessages.some(message => message.type === 'block'), false,
  'PCM must remain on the direct Worker-to-worklet port');

worker.controller.handleMessage({ data: { type: 'finish', generation: 7 } });
const end = port.messages.find(message => message.type === 'end');
assert.equal(end.sample, 128);
assert.equal(end.generation, 7);
worker.controller.handleMessage({ data: { type: 'close' } });
assert.equal(port.closed, true);
assert.equal(worker.isClosed(), true);

const renderFailureWorker = createWorkerHarness();
renderFailureWorker.controller.handleMessage({ data: {
  type: 'prepare', requestId: 5, song: new Uint8Array(loaderBytes).buffer,
  playerOptions: { sampleRate: 48000 },
  producerOptions: { sampleRate: 48000, blockFrames: 64, tailSeconds: 0 },
} });
class RejectingBlockPort extends FakePort {
  postMessage(message, transfer = []) {
    if (message.type === 'block') throw new Error('synthetic PCM transfer failure');
    super.postMessage(message, transfer);
  }
}
const rejectingPort = new RejectingBlockPort();
renderFailureWorker.controller.handleMessage({ data: {
  type: 'connect', requestId: 6, generation: 2, port: rejectingPort,
} });
rejectingPort.emit({ type: 'need', generation: 2, blocks: 1 });
const firstRenderErrors = renderFailureWorker.controlMessages.filter(message =>
  message.type === 'error' && message.stage === 'render');
assert.equal(firstRenderErrors.length, 1);
rejectingPort.emit({ type: 'need', generation: 2, blocks: 1 });
assert.equal(renderFailureWorker.controlMessages.filter(message =>
  message.type === 'error' && message.stage === 'render').length, 1,
'a failed Worker producer must latch terminally instead of rerunning the failing render');

const invalidWorker = createWorkerHarness();
invalidWorker.controller.handleMessage({ data: {
  type: 'prepare', requestId: 9, song: new Uint8Array([0]).buffer,
  playerOptions: { sampleRate: 44100 },
  producerOptions: { sampleRate: 44100, blockFrames: 64, tailSeconds: 0 },
} });
const failure = invalidWorker.controlMessages.find(message => message.type === 'error');
assert.equal(failure.replyTo, 9);
assert.equal(failure.kind, 'content');
assert.equal(failure.stage, 'prepare');

console.log('audio Worker runtime and direct PCM protocol tests passed');
