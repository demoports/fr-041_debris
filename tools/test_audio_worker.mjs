import assert from 'node:assert/strict';
import { AudioStream } from '../src/audio.js';

async function flushMicrotasks(rounds = 12) {
  for (let index = 0; index < rounds; index++) await Promise.resolve();
}

function createHarness(options = {}) {
  const state = {
    workers: [], channels: [], nodes: [], contexts: [],
    directRequests: [], events: [],
  };

  class FakeMessagePort {
    constructor(label) {
      this.label = label;
      this.peer = null;
      this.onmessage = null;
      this.onmessageerror = null;
      this.sent = [];
      this.received = [];
      this.startCalls = 0;
      this.closeCalls = 0;
      this.closed = false;
    }

    postMessage(message, transfer = []) {
      if (this.closed) throw new Error(`${this.label} is closed`);
      const record = { message, transfer: Array.from(transfer || []) };
      this.sent.push(record);
      const peer = this.peer;
      queueMicrotask(() => {
        if (!peer || peer.closed) return;
        peer.received.push(record);
        peer.onmessage?.({ data: message, ports: record.transfer });
      });
    }

    start() { this.startCalls++; }

    close() {
      this.closeCalls++;
      this.closed = true;
    }
  }

  class FakeMessageChannel {
    constructor() {
      const index = state.channels.length;
      this.port1 = new FakeMessagePort(`channel-${index}:port1`);
      this.port2 = new FakeMessagePort(`channel-${index}:port2`);
      this.port1.peer = this.port2;
      this.port2.peer = this.port1;
      state.channels.push(this);
    }
  }

  class FakeControlPort {
    constructor() {
      this.onmessage = null;
      this.sent = [];
    }

    postMessage(message, transfer = []) {
      this.sent.push({ message, transfer: Array.from(transfer || []) });
    }

    emit(message) {
      queueMicrotask(() => this.onmessage?.({ data: message }));
    }
  }

  class FakeAudioWorkletNode {
    constructor(audioContext, name, nodeOptions) {
      this.context = audioContext;
      this.name = name;
      this.options = nodeOptions;
      this.port = new FakeControlPort();
      this.connectCalls = 0;
      this.disconnectCalls = 0;
      state.nodes.push(this);
    }

    connect(destination) {
      assert.strictEqual(destination, this.context.destination);
      this.connectCalls++;
    }

    disconnect() { this.disconnectCalls++; }
  }

  class FakeAudioContext {
    constructor(contextOptions = {}) {
      this.requestedOptions = contextOptions;
      this.sampleRate = options.sampleRate || 10;
      this.currentTime = 0;
      this.state = 'suspended';
      this.destination = {};
      this.resumeCalls = 0;
      this.closeCalls = 0;
      this.audioWorklet = {
        modules: [],
        addModule: async url => { this.audioWorklet.modules.push(url); },
      };
      state.contexts.push(this);
    }

    async resume() {
      this.resumeCalls++;
      this.state = 'running';
      if (options.processorErrorDuringResume) {
        const node = state.nodes.find(candidate => candidate.context === this);
        node?.onprocessorerror?.({ message: 'synthetic processor failure during resume' });
      }
    }

    async close() {
      this.closeCalls++;
      this.state = 'closed';
    }
  }

  class FakeWorker {
    constructor(url, workerOptions = {}) {
      if (options.failConstructor) throw new Error('synthetic Worker constructor failure');
      this.url = url;
      this.options = workerOptions;
      this.onmessage = null;
      this.onerror = null;
      this.sent = [];
      this.terminateCalls = 0;
      this.producerPort = null;
      this.generation = 0;
      this.renderedSample = 0;
      state.workers.push(this);
      state.events.push('worker:construct');
    }

    emit(message) {
      queueMicrotask(() => this.onmessage?.({ data: message }));
    }

    postMessage(message, transfer = []) {
      const record = { message, transfer: Array.from(transfer || []) };
      this.sent.push(record);
      state.events.push(`worker:${message.type}`);
      if (message.type === 'prepare') {
        if (options.failPrepare) throw new Error('synthetic Worker prepare failure');
        this.prepareMessage = message;
        if (options.deferPrepare) return;
        this.emit({
          type: 'ready', replyTo: message.requestId,
          sequenceSamples: 100, durationSamples: 100,
          renderedSample: 0, endSample: null, ended: false,
        });
      } else if (message.type === 'connect') {
        if (options.failConnect) {
          const error = new Error('synthetic MessagePort setup failure');
          error.name = 'DataCloneError';
          throw error;
        }
        this.producerPort = message.port;
        this.generation = message.generation;
        this.producerPort.onmessage = event => this.handleProducerMessage(event.data || {});
        this.producerPort.start?.();
        this.emit({
          type: 'connected', replyTo: message.requestId,
          sequenceSamples: 100, durationSamples: 100,
          renderedSample: this.renderedSample, endSample: null, ended: false,
        });
      }
    }

    handleProducerMessage(message) {
      state.directRequests.push(message);
      if (message.type !== 'need' || message.generation !== this.generation) return;
      const count = Math.max(0, Math.floor(Number(message.blocks) || 0));
      const blockFrames = Math.max(1,
        Math.floor(Number(this.prepareMessage?.producerOptions?.blockFrames) || 1));
      for (let index = 0; index < count; index++) {
        const start = this.renderedSample;
        const end = start + blockFrames;
        const block = new Float32Array(blockFrames * 2);
        for (let frame = 0; frame < blockFrames; frame++) {
          block[frame * 2] = start + frame;
          block[frame * 2 + 1] = -(start + frame);
        }
        this.renderedSample = end;
        this.producerPort.postMessage({
          type: 'block', generation: this.generation,
          start, end, frames: blockFrames, data: block.buffer,
        }, [block.buffer]);
      }
    }

    terminate() {
      this.terminateCalls++;
      state.events.push('worker:terminate');
    }
  }

  class HarnessAudioStream extends AudioStream {
    constructor(streamOptions = {}) {
      super({
        ...streamOptions,
        AudioContextClass: FakeAudioContext,
        AudioWorkletNodeClass: FakeAudioWorkletNode,
        WorkerClass: FakeWorker,
        MessageChannelClass: FakeMessageChannel,
      });
    }
  }

  return { D: { AudioStream: HarnessAudioStream }, state };
}

// Successful loader setup uses copied song bytes and a direct Worker/worklet
// MessageChannel. init() establishes the producer but deliberately does not
// wait for the worklet's bounded queue to become primed.
{
  const { D, state } = createHarness();
  const backing = new Uint8Array([99, 1, 2, 3, 88]);
  const workerSong = backing.subarray(1, 4);
  let mainRenderCalls = 0, mainTimelineCalls = 0, mainPumpCalls = 0;
  const mainSynth = {
    sampleRate: 10,
    calcSongSamples() { mainTimelineCalls++; return 100; },
    reset() { mainTimelineCalls++; },
    render() { mainRenderCalls++; throw new Error('main synth must not render in Worker mode'); },
  };
  const stream = new D.AudioStream({
    sampleRate: 10, blockFrames: 4, queueBlocks: 2,
    tailSeconds: 0, workerSong,
    workerInitTimeoutMilliseconds: 1000,
  });
  const inheritedPump = stream.pump.bind(stream);
  stream.pump = (...args) => { mainPumpCalls++; return inheritedPump(...args); };

  const initialized = await stream.init(mainSynth);
  assert.strictEqual(initialized, stream);
  assert.equal(state.workers.length, 1);
  assert.equal(state.channels.length, 1);
  assert.equal(state.nodes.length, 1);
  const worker = state.workers[0], channel = state.channels[0], node = state.nodes[0];
  assert.match(String(worker.url), /\/src\/audio_worker\.js$/);
  assert.deepEqual(worker.options, { name: 'debris-v2-loader', type: 'module' });
  assert.match(String(stream.context.audioWorklet.modules[0]), /\/src\/audio_worklet\.js$/);
  const prepare = worker.sent.find(record => record.message.type === 'prepare');
  const connect = worker.sent.find(record => record.message.type === 'connect');
  const workletConnection = node.port.sent.find(record => record.message.type === 'producer-port');

  assert.ok(prepare, 'Worker receives a prepare command');
  assert.equal('synth' in prepare.message, false);
  assert.notStrictEqual(prepare.message.song, workerSong.buffer,
    'the caller-owned song allocation is not transferred');
  assert.deepEqual(Array.from(new Uint8Array(prepare.message.song)), [1, 2, 3]);
  assert.deepEqual(prepare.transfer, [prepare.message.song],
    'only the copied song allocation is transferred to the Worker');
  backing[1] = 77;
  assert.deepEqual(Array.from(new Uint8Array(prepare.message.song)), [1, 2, 3],
    'mutating the caller-owned song cannot alter the Worker copy');

  assert.strictEqual(workletConnection.message.port, channel.port1);
  assert.deepEqual(workletConnection.transfer, [channel.port1]);
  assert.strictEqual(connect.message.port, channel.port2);
  assert.deepEqual(connect.transfer, [channel.port2]);
  assert.equal(stream.producerBackend, 'worker');
  assert.equal(stream.workerPrepared, true);
  assert.equal(stream.workerPrimed, false,
    'init resolves after setup without waiting for PCM priming');
  assert.equal(stream.synth, null);
  assert.equal(mainTimelineCalls, 0);
  assert.equal(mainRenderCalls, 0);
  assert.equal(mainPumpCalls, 0);
  assert.equal(node.port.sent.some(record => record.message.type === 'block'), false,
    'PCM never routes through the main-thread AudioWorkletNode port');

  let startSettled = false;
  const starting = stream.start().then(value => { startSettled = true; return value; });
  await flushMicrotasks();
  assert.equal(stream.context.state, 'running');
  assert.equal(startSettled, false, 'start waits for the worklet to report a primed queue');

  channel.port1.postMessage({ type: 'need', generation: stream.generation, blocks: 2 });
  await flushMicrotasks();
  assert.deepEqual(state.directRequests, [
    { type: 'need', generation: stream.generation, blocks: 2 },
  ]);
  const directBlocks = channel.port1.received.filter(record => record.message.type === 'block');
  assert.equal(directBlocks.length, 2);
  assert.deepEqual(directBlocks.map(record => [
    record.message.start, record.message.end, record.message.frames,
  ]), [[0, 4, 4], [4, 8, 4]]);
  assert.ok(directBlocks.every(record =>
    record.transfer.length === 1 && record.transfer[0] === record.message.data));
  assert.equal(worker.sent.some(record => record.message.type === 'need' || record.message.type === 'block'), false,
    'refills stay on the direct MessageChannel rather than Worker control messages');

  // A stray legacy refill request on the built-in port is ignored in Worker
  // mode and therefore cannot restart main-thread synthesis.
  node.port.emit({ type: 'need', blocks: 1 });
  await flushMicrotasks();
  assert.equal(mainPumpCalls, 0);
  assert.equal(mainRenderCalls, 0);
  assert.equal(startSettled, false);

  node.port.emit({
    type: 'primed', generation: stream.generation,
    receivedThrough: 8, queued: 2,
  });
  await starting;
  assert.equal(startSettled, true);
  assert.equal(stream.workerPrimed, true);
  assert.equal(stream.renderedSample, 8);
  assert.equal(stream.paused, false);
  assert.equal(node.port.sent.some(record =>
    record.message.type === 'pause' && record.message.value === false), true);
  assert.equal(mainRenderCalls, 0);
  assert.equal(mainPumpCalls, 0);

  // Once direct Worker refills pass the initially primed queue, clock reports
  // carry the worklet's received watermark. Main's public production clock can
  // therefore interpolate forward instead of freezing at sample 8 until end.
  stream.context.currentTime = 0.4;
  node.port.emit({
    type: 'clock', generation: stream.generation,
    sample: 4, receivedThrough: 12,
  });
  await flushMicrotasks();
  stream.context.currentTime = 1;
  assert.equal(stream.renderedSample, 12);
  assert.equal(stream.sample, 10);

  await assert.rejects(stream.seek(1), /forward-only and cannot seek/);
  assert.equal(worker.sent.some(record => record.message.type === 'seek'), false);
  assert.equal(mainRenderCalls, 0);

  const audioContext = stream.context;
  const firstClose = stream.close();
  const secondClose = stream.close();
  assert.strictEqual(firstClose, secondClose, 'close is idempotent while cleanup is pending');
  await firstClose;
  assert.equal(worker.terminateCalls, 1);
  assert.equal(worker.sent.filter(record => record.message.type === 'close').length, 1);
  assert.equal(channel.port1.closeCalls, 1);
  assert.equal(channel.port2.closeCalls, 1);
  assert.equal(node.disconnectCalls, 1);
  assert.equal(node.port.onmessage, null);
  assert.equal(audioContext.closeCalls, 1);
  assert.equal(stream.producerWorker, null);
  assert.equal(stream.producerChannel, null);
  assert.equal(stream.node, null);
  assert.equal(stream.context, null);
  assert.equal(stream.workerSong, null);
  await stream.close();
  assert.equal(worker.terminateCalls, 1);
  assert.equal(node.disconnectCalls, 1);
  assert.equal(audioContext.closeCalls, 1);
}

// A setup-only failure after Worker preparation falls back to the retained
// main-thread worklet producer, cleans the partially transferred channel, and
// reconstructs a local player from the original song bytes.
{
  const { D, state } = createHarness({ failConnect: true });
  const workerSong = new Uint8Array([4, 5, 6]);
  let factoryCalls = 0, fallbackRenderCalls = 0, fallbackResetCalls = 0;
  const fallbackSynth = {
    sampleRate: 10,
    calcSongSamples() { return 100; },
    reset() { fallbackResetCalls++; return this; },
    render(target, frames) {
      fallbackRenderCalls++;
      target.fill(0.25, 0, frames * 2);
      return target;
    },
  };
  const playerFactory = (song, playerOptions) => {
    factoryCalls++;
    state.events.push('fallback:factory');
    assert.strictEqual(song, workerSong);
    assert.equal(playerOptions.sampleRate, 10);
    return fallbackSynth;
  };
  const stream = new D.AudioStream({
    sampleRate: 10, blockFrames: 4, queueBlocks: 2,
    tailSeconds: 0, workerSong, workerInitTimeoutMilliseconds: 1000,
    playerFactory,
  });

  await stream.init();
  const worker = state.workers[0], channel = state.channels[0], node = state.nodes[0];
  assert.equal(stream.producerBackend, 'main-worklet');
  assert.match(stream.workerFallbackReason?.message || '', /MessagePort setup failure/);
  assert.equal(factoryCalls, 1);
  assert.equal(fallbackResetCalls, 1);
  assert.equal(fallbackRenderCalls, 2,
    'fallback primes the ordinary worklet queue on the main thread');
  assert.strictEqual(stream.synth, fallbackSynth);
  assert.equal(worker.terminateCalls, 1);
  assert.ok(state.events.indexOf('worker:terminate') < state.events.indexOf('fallback:factory'));
  assert.equal(channel.port1.closeCalls, 1);
  assert.equal(channel.port2.closeCalls, 1);
  assert.equal(node.port.sent.some(record => record.message.type === 'detach-producer'), true);
  const fallbackReset = node.port.sent.find(record => record.message.type === 'reset');
  assert.ok(fallbackReset);
  assert.equal(fallbackReset.message.generation, stream.generation);
  assert.ok(fallbackReset.message.generation >
    worker.sent.find(record => record.message.type === 'connect').message.generation,
  'fallback invalidates every late direct-port block generation');
  assert.equal(node.port.sent.filter(record => record.message.type === 'block').length, 2);
  assert.equal(stream.producerWorker, null);
  assert.equal(stream.producerChannel, null);

  await stream.close();
  assert.equal(worker.terminateCalls, 1,
    'closing the fallback stream does not terminate an already-cleaned Worker twice');
}

// Closing while start waits for priming rejects start instead of reporting a
// successful unpause on a torn-down stream.
{
  const { D } = createHarness();
  const stream = new D.AudioStream({
    sampleRate: 10, blockFrames: 4, queueBlocks: 2, tailSeconds: 0,
    workerSong: new Uint8Array([1, 2, 3]),
  });
  await stream.init();
  const starting = stream.start();
  await flushMicrotasks();
  await stream.close();
  await assert.rejects(starting, /closed/);
  assert.equal(stream.paused, true);
}

// close() also cancels a Worker prepare command that has not replied yet;
// init cannot continue into a fallback synth or worklet node afterward.
{
  const { D, state } = createHarness({ deferPrepare: true });
  const stream = new D.AudioStream({
    sampleRate: 10, blockFrames: 4, queueBlocks: 2, tailSeconds: 0,
    workerSong: new Uint8Array([1, 2, 3]),
  });
  const initializing = stream.init();
  await flushMicrotasks();
  await stream.close();
  await assert.rejects(initializing, /closed/);
  assert.equal(state.nodes.length, 0);
  assert.equal(stream.context, null);
}

// Ordered render errors are already sequenced behind valid PCM on the direct
// port, so main must not race them with a graceful control-port end. A
// catastrophic Worker error instead uses a generation-scoped abort.
{
  const { D, state } = createHarness();
  let errors = 0;
  const stream = new D.AudioStream({
    sampleRate: 10, blockFrames: 4, queueBlocks: 2, tailSeconds: 0,
    workerSong: new Uint8Array([1, 2, 3]), onError: () => errors++,
  });
  await stream.init();
  const node = state.nodes[0];
  node.port.emit({
    type: 'producer-error', generation: stream.generation,
    name: 'Error', message: 'ordered render failure',
  });
  await flushMicrotasks();
  assert.equal(errors, 1);
  assert.equal(node.port.sent.some(record => record.message.type === 'end'), false,
    'ordered direct-port failure must not race a graceful end over the control port');
  await stream.close();
}

{
  const { D, state } = createHarness();
  let errors = 0;
  const stream = new D.AudioStream({
    sampleRate: 10, blockFrames: 4, queueBlocks: 2, tailSeconds: 0,
    workerSong: new Uint8Array([1, 2, 3]), onError: () => errors++,
  });
  await stream.init();
  const worker = state.workers[0], node = state.nodes[0];
  worker.onerror?.({ message: 'catastrophic worker failure', preventDefault() {} });
  assert.equal(errors, 1);
  const abort = node.port.sent.find(record => record.message.type === 'abort');
  assert.equal(abort.message.generation, stream.generation);
  await stream.close();
}

{
  const { D, state } = createHarness();
  let errors = 0;
  const stream = new D.AudioStream({
    sampleRate: 10, blockFrames: 4, queueBlocks: 2, tailSeconds: 0,
    workerSong: new Uint8Array([1, 2, 3]), onError: () => errors++,
  });
  await stream.init();
  const node = state.nodes[0];
  node.onprocessorerror?.({ message: 'synthetic processor failure' });
  assert.equal(errors, 1);
  assert.equal(stream.workerError?.name, 'AudioWorkletError');
  await stream.close();
}

// A main-thread worklet can fail while resume() is yielding control back to the
// browser. start() must observe that lifecycle error before it unpauses audio.
{
  const { D, state } = createHarness({ processorErrorDuringResume: true });
  let errors = 0;
  const synth = {
    sampleRate: 10,
    calcSongSamples() { return 100; },
    reset() { return this; },
    render(target, frames) { target.fill(0, 0, frames * 2); return target; },
  };
  const stream = new D.AudioStream({
    sampleRate: 10, blockFrames: 4, queueBlocks: 2, tailSeconds: 0,
    onError: () => errors++,
  });
  await stream.init(synth);
  const context = state.contexts[0], node = state.nodes[0];
  assert.equal(state.workers.length, 0);
  assert.equal(stream.producerBackend, 'main-worklet');
  assert.equal(stream.paused, true);
  assert.equal(stream.workerError, null);

  await assert.rejects(stream.start(), error => {
    assert.equal(error.name, 'AudioWorkletError');
    assert.match(error.message, /processor failure during resume/);
    return true;
  });
  assert.equal(context.resumeCalls, 1);
  assert.equal(errors, 1);
  assert.equal(stream.paused, true,
    'a processor failure during resume must not leave the stream playing');
  assert.equal(node.port.sent.some(record =>
    record.message.type === 'pause' && record.message.value === false), false,
  'start must not send an unpause command after the processor failure');
  await stream.close();
}

console.log('worker-backed audio stream protocol tests passed');
