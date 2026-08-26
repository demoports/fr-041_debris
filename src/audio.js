// WebAudio bridge for the plain-JavaScript V2 synth. Normal playback keeps its
// seekable producer on main; the precalc loader can instead put V2 production
// in a dedicated module Worker connected directly to the real-time AudioWorklet.
import { createV2Player } from './v2.js';
import {
  AudioBlockProducer,
  copyAudioBytes,
  renderSynthBlock,
  resetSynth,
  seekSynth,
} from './audio_core.js';
import {
  DEBRIS_OUTPUT_PROCESSOR_NAME,
  StereoPcmResampler,
} from './audio_worklet_core.js';

const DEFAULT_AUDIO_WORKER_URL = new URL('./audio_worker.js', import.meta.url);
const DEFAULT_AUDIO_WORKLET_URL = new URL('./audio_worklet.js', import.meta.url);

class AudioStream extends AudioBlockProducer {
  constructor(options = {}) {
    super(null, options);
    const environment = options.environment || globalThis;
    this.playerFactory = options.playerFactory || options.createV2Player || createV2Player;
    this.AudioContextClass = options.AudioContextClass || options.audioContextClass ||
      options.AudioContext || environment.AudioContext || environment.webkitAudioContext || null;
    this.AudioWorkletNodeClass = options.AudioWorkletNodeClass || options.audioWorkletNodeClass ||
      options.AudioWorkletNode || environment.AudioWorkletNode || null;
    this.WorkerClass = options.WorkerClass || options.workerClass ||
      options.Worker || environment.Worker || null;
    this.MessageChannelClass = options.MessageChannelClass || options.messageChannelClass ||
      options.MessageChannel || environment.MessageChannel || null;
    this.setTimer = options.setTimeout || environment.setTimeout?.bind(environment) ||
      globalThis.setTimeout.bind(globalThis);
    this.clearTimer = options.clearTimeout || environment.clearTimeout?.bind(environment) ||
      globalThis.clearTimeout.bind(globalThis);
    this.queueBlocks = Math.max(1, Math.floor(options.queueBlocks || 8));
    this.reportClock = options.reportClock !== false;
    this.context = null;
    this.node = null;
    this.outputSampleRate = this.sampleRate;
    this.scriptResampler = null;
    this.playedSample = 0;
    this.clockSample = 0;
    this.clockTime = 0;
    this.paused = true;
    this.drained = false;
    this.endPosted = false;
    this.endedNotified = false;
    this.pumping = false;
    this.pendingBlocks = 0;
    this.pumpPromise = null;
    this.generation = 0;
    this.onEnded = options.onEnded || null;
    this.onUnderrun = options.onUnderrun || null;
    this.onError = options.onError || null;
    this.workerSong = options.workerSong || null;
    this.workerUrl = options.workerUrl || DEFAULT_AUDIO_WORKER_URL;
    this.workletUrl = options.workletUrl || DEFAULT_AUDIO_WORKLET_URL;
    this.workerPlayerOptions = { ...(options.workerPlayerOptions || {}) };
    this.workerInitTimeoutMilliseconds = Math.max(1000,
      Number(options.workerInitTimeoutMilliseconds) || 10000);
    this.producerWorker = null;
    this.producerChannel = null;
    this.workerPrepared = false;
    this.workerPrimed = false;
    this.workerPrimePromise = null;
    this.workerPrimeResolve = null;
    this.workerPrimeReject = null;
    this.workerPrimeTimer = null;
    this.workerError = null;
    this.workerMessageWaiters = new Map();
    this.workerMessageSerial = 0;
    this.workerErrorNotified = false;
    this.workerFallbackReason = null;
    this.producerBackend = 'uninitialized';
    this.closed = false;
    this.lifecycleGeneration = 0;
    this.closePromise = null;
  }

  assertOpen(generation = this.lifecycleGeneration) {
    if (this.closed || generation !== this.lifecycleGeneration || !this.context) {
      throw new Error('audio stream was closed during an asynchronous operation');
    }
  }

  workerCapabilityAvailable() {
    return Boolean(this.workerSong && this.context?.audioWorklet &&
      typeof this.AudioWorkletNodeClass === 'function' &&
      typeof this.WorkerClass === 'function' &&
      typeof this.MessageChannelClass === 'function');
  }

  workerPlayerConfiguration() {
    const output = {};
    for (const [key, value] of Object.entries(this.workerPlayerOptions)) {
      if (value === null || ['number', 'string', 'boolean'].includes(typeof value)) output[key] = value;
    }
    output.sampleRate = this.sampleRate;
    return output;
  }

  workerCommand(type, payload = {}, transfer = []) {
    if (!this.producerWorker) return Promise.reject(new Error('audio producer Worker is missing'));
    const requestId = ++this.workerMessageSerial;
    return new Promise((resolve, reject) => {
      const timer = this.setTimer(() => {
        this.workerMessageWaiters.delete(requestId);
        const error = new Error(`audio producer Worker timed out during ${type}`);
        error.workerErrorKind = 'setup';
        reject(error);
      }, this.workerInitTimeoutMilliseconds);
      this.workerMessageWaiters.set(requestId, { resolve, reject, timer, type });
      try {
        this.producerWorker.postMessage({ type, requestId, ...payload }, transfer);
      } catch (error) {
        this.clearTimer(timer);
        this.workerMessageWaiters.delete(requestId);
        error.workerErrorKind ||= 'setup';
        reject(error);
      }
    });
  }

  updateWorkerState(message = {}) {
    if (Number.isFinite(message.sequenceSamples)) this.sequenceSamples = Math.max(0, Math.floor(message.sequenceSamples));
    if (Number.isFinite(message.renderedSample)) this.renderedSample = Math.max(0, Math.floor(message.renderedSample));
    if (Number.isFinite(message.durationSamples)) this.durationSamples = Math.max(0, Math.floor(message.durationSamples));
    if (Number.isFinite(message.endSample)) this.endSample = Math.max(0, Math.floor(message.endSample));
    if (message.ended) {
      this.ended = true;
      if (this.endSample === null) this.endSample = this.renderedSample;
      this.durationSamples = this.endSample;
    }
  }

  workerMessageError(message = {}) {
    const error = new Error(message.message || 'audio producer Worker failed');
    error.name = message.name || 'Error';
    error.workerErrorKind = message.kind || 'content';
    error.workerStage = message.stage || '';
    return error;
  }

  handleWorkerMessage(message = {}) {
    if (this.closed) return;
    this.updateWorkerState(message);
    const waiter = message.replyTo && this.workerMessageWaiters.get(message.replyTo);
    if (waiter) {
      this.clearTimer(waiter.timer);
      this.workerMessageWaiters.delete(message.replyTo);
      if (message.type === 'error') waiter.reject(this.workerMessageError(message));
      else waiter.resolve(message);
      return;
    }
    if (message.type === 'error') {
      // The matching direct-port error is ordered after every valid PCM block.
      // Never race it with a graceful end sent over the unrelated control port.
      this.handleWorkerFailure(this.workerMessageError(message), { ordered: true });
    }
  }

  handleWorkerFailure(error, options = {}) {
    if (this.closed || this.workerError) return;
    this.workerError = error instanceof Error ? error : new Error(String(error));
    for (const [requestId, waiter] of this.workerMessageWaiters) {
      this.clearTimer(waiter.timer); waiter.reject(this.workerError);
      this.workerMessageWaiters.delete(requestId);
    }
    if (!this.workerPrepared && !options.force) return;
    this.rejectWorkerPrime(this.workerError);
    this.markProducerEnded(this.renderedSample);
    if (options.abort && this.node?.port) {
      this.node.port.postMessage({ type: 'abort', generation: this.generation });
    }
    if (!this.workerErrorNotified) {
      this.workerErrorNotified = true;
      this.onError?.(this.workerError, this);
    }
  }

  terminateProducerWorker(reason = null) {
    const error = reason instanceof Error ? reason : new Error(reason || 'audio producer Worker closed');
    this.rejectWorkerPrime(error);
    for (const [requestId, waiter] of this.workerMessageWaiters) {
      this.clearTimer(waiter.timer); waiter.reject(error);
      this.workerMessageWaiters.delete(requestId);
    }
    if (this.producerWorker) {
      this.producerWorker.onmessage = null;
      this.producerWorker.onerror = null;
      this.producerWorker.onmessageerror = null;
      try { this.producerWorker.postMessage({ type: 'close' }); } catch (_) {}
      try { this.producerWorker.terminate(); } catch (_) {}
    }
    for (const port of [this.producerChannel?.port1, this.producerChannel?.port2]) {
      try { port?.close?.(); } catch (_) {}
    }
    this.producerWorker = null;
    this.producerChannel = null;
    this.workerPrepared = false;
    this.workerPrimed = false;
  }

  async prepareWorkerProducer() {
    if (!this.workerCapabilityAvailable()) return false;
    try {
      this.producerWorker = new this.WorkerClass(this.workerUrl, {
        name: 'debris-v2-loader',
        type: 'module',
      });
      this.producerWorker.onmessage = event => this.handleWorkerMessage(event.data || {});
      this.producerWorker.onerror = event => {
        event?.preventDefault?.();
        const error = new Error(event?.message || 'audio producer Worker script failed');
        error.workerErrorKind = this.workerPrepared ? 'content' : 'setup';
        this.handleWorkerFailure(error, { abort: this.workerPrepared });
      };
      this.producerWorker.onmessageerror = () => {
        const error = new Error('audio producer Worker control message failed');
        error.workerErrorKind = this.workerPrepared ? 'content' : 'setup';
        this.handleWorkerFailure(error, { abort: this.workerPrepared });
      };
      const song = copyAudioBytes(this.workerSong);
      const ready = await this.workerCommand('prepare', {
        song,
        playerOptions: this.workerPlayerConfiguration(),
        producerOptions: {
          sampleRate: this.sampleRate,
          blockFrames: this.blockFrames,
          tailSeconds: this.tailSeconds,
          tailSilenceSeconds: this.tailSilenceSeconds,
          tailSilenceThreshold: this.tailSilenceThreshold,
        },
      }, [song]);
      this.workerPrepared = true;
      this.synth = null;
      this.sequenceSamples = Number.isFinite(ready.sequenceSamples)
        ? Math.max(0, Math.floor(ready.sequenceSamples)) : null;
      this.resetEndState();
      this.updateWorkerState(ready);
      return true;
    } catch (error) {
      this.terminateProducerWorker(error);
      if (error?.workerErrorKind === 'content') throw error;
      this.workerFallbackReason = error;
      this.workerError = null;
      return false;
    }
  }

  createWorkerFallbackSynth(existingSynth = null) {
    if (existingSynth) return existingSynth;
    if (!this.workerSong || typeof this.playerFactory !== 'function') {
      throw new TypeError('audio synth is missing');
    }
    return this.playerFactory(this.workerSong, this.workerPlayerConfiguration());
  }

  async connectWorkerProducer() {
    const channel = new this.MessageChannelClass();
    this.producerChannel = channel;
    this.generation++;
    this.workerPrimed = false;
    try {
      this.node.port.postMessage({
        type: 'producer-port', port: channel.port1,
        generation: this.generation, sample: 0,
      }, [channel.port1]);
      const connected = await this.workerCommand('connect', {
        port: channel.port2,
        generation: this.generation,
      }, [channel.port2]);
      this.updateWorkerState(connected);
      return true;
    } catch (error) {
      this.generation++;
      this.node?.port?.postMessage({ type: 'detach-producer' });
      this.terminateProducerWorker(error);
      if (error?.workerErrorKind === 'content') throw error;
      this.workerFallbackReason = error;
      this.workerError = null;
      return false;
    }
  }

  async init(synth) {
    if (this.context) return this;
    if (this.closed) throw new Error('AudioStream is closed');
    const lifecycleGeneration = this.lifecycleGeneration;
    this.synth = synth || null;
    if (!this.AudioContextClass) throw new Error('This browser does not support WebAudio.');
    try {
      this.context = new this.AudioContextClass({
        sampleRate: this.sampleRate,
        latencyHint: 'playback',
      });
    } catch (error) {
      // The Web Audio specification permits NotSupportedError when a requested
      // context rate is unavailable. Retry at the browser's default sink rate;
      // V2 still runs at the fixed production rate and the output resampler
      // below bridges the two coordinates. Security/quota failures are not
      // rate negotiation and must remain visible to the caller.
      if (error?.name !== 'NotSupportedError') throw error;
      this.context = new this.AudioContextClass({ latencyHint: 'playback' });
    }
    try {
      // sampleRate is always the V2/visual production coordinate. A browser
      // may ignore the requested context rate; only the final sink adopts that
      // hardware rate and AudioWorklet performs the conversion.
      this.outputSampleRate = Math.max(1,
        Number(this.context.sampleRate) || this.sampleRate);
      let workerProducer = await this.prepareWorkerProducer();
      this.assertOpen(lifecycleGeneration);
      if (!workerProducer) {
        this.synth = this.createWorkerFallbackSynth(this.synth);
        const synthRate = Number(this.synth?.sampleRate);
        if (Number.isFinite(synthRate) && synthRate !== this.sampleRate) {
          if (!this.synth.song || typeof this.playerFactory !== 'function') {
            throw new Error(`Production runs at ${this.sampleRate} Hz, but the V2 player runs at ${synthRate} Hz and cannot be recreated`);
          }
          this.synth = this.playerFactory(this.synth.song, { sampleRate: this.sampleRate });
        }
        this.configureTimeline();
      }
      if (!this.context.audioWorklet || typeof this.AudioWorkletNodeClass !== 'function') {
        this.assertOpen(lifecycleGeneration);
        this.initScriptProcessor();
        this.producerBackend = 'script-processor';
        return this;
      }
      await this.context.audioWorklet.addModule(this.workletUrl);
      this.assertOpen(lifecycleGeneration);
      this.node = new this.AudioWorkletNodeClass(this.context, DEBRIS_OUTPUT_PROCESSOR_NAME, {
        numberOfInputs: 0, numberOfOutputs: 1, outputChannelCount: [2],
        processorOptions: {
          queueBlocks: this.queueBlocks,
          reportClock: this.reportClock,
          sourceSampleRate: this.sampleRate,
          outputSampleRate: this.outputSampleRate,
        },
      });
      this.node.port.onmessage = event => this.handleMessage(event.data || {});
      this.node.port.onmessageerror = () => {
        this.handleWorkerFailure(new Error('AudioWorklet control message failed'), {
          abort: true, force: true,
        });
      };
      this.node.onprocessorerror = event => {
        const error = new Error(event?.message || 'AudioWorklet processor failed');
        error.name = 'AudioWorkletError';
        this.handleWorkerFailure(error, { force: true });
      };
      this.node.connect(this.context.destination);
      this.clockTime = this.context.currentTime;
      if (workerProducer) {
        workerProducer = await this.connectWorkerProducer();
        this.assertOpen(lifecycleGeneration);
        if (!workerProducer) {
          this.synth = this.createWorkerFallbackSynth();
          this.configureTimeline();
          this.node.port.postMessage({
            type: 'reset', generation: this.generation,
            sample: 0, incoming: this.queueBlocks,
          });
        }
      }
      if (workerProducer) this.producerBackend = 'worker';
      else {
        this.producerBackend = 'main-worklet';
        await this.pump(this.queueBlocks);
        this.assertOpen(lifecycleGeneration);
      }
      return this;
    } catch (error) {
      try { await this.close(); }
      catch (closeError) { console.warn('audio initialization cleanup failed', closeError); }
      throw error;
    }
  }

  resetEndState() {
    super.resetEndState();
    this.drained = false;
    this.endPosted = false;
    this.endedNotified = false;
  }

  markProducerEnded(sample = this.renderedSample) {
    if (this.ended) return;
    super.markProducerEnded(sample);
    this.pendingBlocks = 0;
  }

  postEnd() {
    if (this.endPosted || !this.ended) return;
    this.endPosted = true;
    if (this.node?.port) this.node.port.postMessage({
      type: 'end', generation: this.generation, sample: this.endSample,
    });
    else this.notifyDrained(this.endSample);
  }

  notifyDrained(sample = this.endSample ?? this.playedSample) {
    sample = Math.max(0, Math.floor(Number(sample) || 0));
    this.playedSample = this.clockSample = sample;
    this.clockTime = this.context?.currentTime || 0;
    this.drained = true;
    if (!this.endedNotified) {
      this.endedNotified = true;
      this.onEnded?.(this);
    }
  }

  initScriptProcessor() {
    const frames = Math.max(1024, this.blockFrames);
    const node = this.context.createScriptProcessor(frames, 0, 2);
    const resampler = new StereoPcmResampler(this.sampleRate, this.outputSampleRate);
    this.scriptResampler = resampler;
    node.onaudioprocess = event => {
      const left = event.outputBuffer.getChannelData(0), right = event.outputBuffer.getChannelData(1);
      left.fill(0); right.fill(0);
      if (this.paused || this.drained) return;
      // Keep enough source PCM for this device callback plus the interpolation
      // look-ahead. Production remains bounded to roughly one synth block.
      const required = Math.ceil(left.length * resampler.ratio) + 2;
      while (!this.ended && resampler.queuedFrames < required) {
        const produced = this.produceBlock(Math.max(
          this.blockFrames, required - resampler.queuedFrames,
        ));
        if (!produced) break;
        resampler.push(produced.block);
      }
      if (this.ended) resampler.end();
      resampler.pull(left, right);
      this.playedSample = resampler.played;
      this.clockSample = this.playedSample;
      this.clockTime = this.context.currentTime;
      if (resampler.drained) this.notifyDrained(this.playedSample);
    };
    node.connect(this.context.destination);
    this.node = node;
  }

  handleMessage(message) {
    if (this.closed) return;
    if (message.type === 'need') {
      if (!this.producerWorker) this.pump(message.blocks || 1);
    }
    else if (message.type === 'clock') {
      if (message.generation !== undefined && message.generation !== this.generation) return;
      this.playedSample = message.sample | 0;
      // Direct Worker PCM bypasses main, so its production watermark is known
      // only by the worklet. Keep interpolation capped by that live watermark
      // instead of the initial primed queue forever.
      this.renderedSample = Math.max(this.renderedSample, Number.isFinite(message.receivedThrough)
        ? Math.floor(message.receivedThrough) : this.playedSample);
      this.clockSample = this.playedSample;
      this.clockTime = this.context.currentTime;
    } else if (message.type === 'primed') {
      if (message.generation !== this.generation) return;
      if (this.workerError) return;
      this.workerPrimed = true;
      if (Number.isFinite(message.receivedThrough)) {
        this.renderedSample = Math.max(this.renderedSample, Math.floor(message.receivedThrough));
      }
      this.resolveWorkerPrime();
    } else if (message.type === 'producer-error') {
      if (message.generation !== undefined && message.generation !== this.generation) return;
      const error = this.workerMessageError({ ...message, kind: 'content' });
      this.handleWorkerFailure(error, { ordered: true });
    } else if (message.type === 'underrun') {
      if (message.generation !== undefined && message.generation !== this.generation) return;
      this.onUnderrun?.(this);
    } else if (message.type === 'drained') {
      if (message.generation !== undefined && message.generation !== this.generation) return;
      this.notifyDrained(message.sample);
    }
  }

  resolveWorkerPrime() {
    if (this.workerPrimeTimer) this.clearTimer(this.workerPrimeTimer);
    this.workerPrimeTimer = null;
    this.workerPrimeResolve?.(this);
    this.workerPrimeResolve = this.workerPrimeReject = null;
    this.workerPrimePromise = null;
  }

  rejectWorkerPrime(error) {
    if (this.workerPrimeTimer) this.clearTimer(this.workerPrimeTimer);
    this.workerPrimeTimer = null;
    this.workerPrimeReject?.(error);
    this.workerPrimeResolve = this.workerPrimeReject = null;
    this.workerPrimePromise = null;
  }

  waitForWorkerPrime() {
    if (this.workerError) return Promise.reject(this.workerError);
    if (!this.producerWorker || this.workerPrimed) return Promise.resolve(this);
    if (this.workerPrimePromise) return this.workerPrimePromise;
    this.workerPrimePromise = new Promise((resolve, reject) => {
      this.workerPrimeResolve = resolve;
      this.workerPrimeReject = reject;
      this.workerPrimeTimer = this.setTimer(() => {
        const error = new Error('audio producer Worker timed out while priming');
        error.workerErrorKind = 'setup';
        this.workerPrimeTimer = null;
        this.workerPrimeResolve = this.workerPrimeReject = null;
        this.workerPrimePromise = null;
        reject(error);
      }, this.workerInitTimeoutMilliseconds);
    });
    return this.workerPrimePromise;
  }

  pump(count = 1) {
    count = Math.max(0, Math.floor(Number(count) || 0));
    if (this.producerWorker || this.ended || !this.node?.port) return Promise.resolve();
    this.pendingBlocks += count;
    if (this.pumpPromise) return this.pumpPromise;
    const generation = this.generation;
    this.pumpPromise = (async () => {
      // Always yield once so pumpPromise is installed before cleanup can run.
      await Promise.resolve();
      this.pumping = true;
      let producedCount = 0;
      try {
        while (this.pendingBlocks && generation === this.generation && !this.ended) {
          this.pendingBlocks--;
          const produced = this.produceBlock(this.blockFrames);
          if (!produced) { if (this.ended) this.postEnd(); break; }
          this.node.port.postMessage({ type: 'block', data: produced.block.buffer }, [produced.block.buffer]);
          if (this.ended) { this.postEnd(); break; }
          if ((++producedCount & 1) === 0) await Promise.resolve();
        }
      } finally {
        this.pumping = false;
        this.pumpPromise = null;
      }
    })();
    return this.pumpPromise;
  }

  async start() {
    if (this.closed) throw new Error('AudioStream is closed');
    if (!this.context) throw new Error('AudioStream.init() has not completed');
    const context = this.context;
    const lifecycleGeneration = this.lifecycleGeneration;
    const playbackGeneration = this.generation;
    if (context.state !== 'running') await context.resume();
    this.assertOpen(lifecycleGeneration);
    if (this.workerError) throw this.workerError;
    if (playbackGeneration !== this.generation) throw new Error('audio stream changed while starting');
    if (this.producerWorker) await this.waitForWorkerPrime();
    this.assertOpen(lifecycleGeneration);
    if (this.workerError) throw this.workerError;
    if (playbackGeneration !== this.generation) throw new Error('audio stream changed while starting');
    this.paused = false;
    this.clockTime = context.currentTime;
    if (this.node?.port) this.node.port.postMessage({ type: 'pause', value: false });
    return this;
  }

  pause(value = !this.paused) {
    const heldSample = this.sample;
    this.paused = Boolean(value);
    if (this.node?.port) this.node.port.postMessage({ type: 'pause', value: this.paused });
    this.clockSample = heldSample;
    this.clockTime = this.context?.currentTime || 0;
    return this.paused;
  }

  get sample() {
    if (!this.context || this.paused) return this.clockSample | 0;
    const interpolated = this.clockSample + Math.max(0, this.context.currentTime - this.clockTime) * this.sampleRate;
    return Math.max(0, Math.min(this.renderedSample, Math.floor(interpolated))) | 0;
  }

  get time() { return this.sample / this.sampleRate; }

  async seek(seconds) {
    if (this.closed || !this.context) throw new Error('AudioStream is closed');
    if (this.producerWorker) {
      throw new Error('Worker-backed loader audio is forward-only and cannot seek');
    }
    let target = Math.max(0, Math.floor(seconds * this.sampleRate));
    if (Number.isFinite(this.durationSamples)) target = Math.min(target, this.durationSamples);
    const wasPaused = this.paused;
    this.pause(true);
    this.generation++;
    this.pendingBlocks = 0;
    if (this.pumpPromise) await this.pumpPromise;
    target = seekSynth(this.synth, target);
    this.resetEndState();
    this.scriptResampler?.reset(target);
    this.renderedSample = this.playedSample = this.clockSample = target;
    this.clockTime = this.context.currentTime;
    if (this.node?.port) this.node.port.postMessage({
      type: 'reset', generation: this.generation,
      sample: target, incoming: this.queueBlocks,
    });
    await this.pump(this.queueBlocks);
    if (!wasPaused) this.pause(false);
    return target / this.sampleRate;
  }

  finish() {
    if (this.producerWorker) {
      try { this.producerWorker.postMessage({ type: 'finish', generation: this.generation }); }
      catch (_) {}
      return this.renderedSample;
    }
    this.markProducerEnded(this.renderedSample);
    if (this.scriptResampler) {
      this.scriptResampler.end();
      if (this.scriptResampler.drained) this.notifyDrained(this.renderedSample);
      return this.endSample;
    }
    this.postEnd();
    return this.endSample;
  }

  close() {
    if (this.closePromise) return this.closePromise;
    this.closed = true;
    this.lifecycleGeneration++;
    this.ended = true; this.generation++;
    this.pendingBlocks = 0;
    this.closePromise = (async () => {
      this.rejectWorkerPrime(new Error('audio stream closed'));
      if (this.node?.port) {
        try { this.node.port.postMessage({ type: 'detach-producer' }); } catch (_) {}
        this.node.port.onmessage = null;
        this.node.port.onmessageerror = null;
      }
      this.terminateProducerWorker();
      if (this.node) {
        try { this.node.disconnect(); } catch (_) {}
        this.node.onprocessorerror = null;
        if ('onaudioprocess' in this.node) this.node.onaudioprocess = null;
      }
      try {
        if (this.context) await this.context.close();
      } finally {
        this.node = this.context = null;
        this.synth = null;
        this.scriptResampler = null;
        this.workerSong = null;
      }
    })();
    return this.closePromise;
  }
}

export { AudioBlockProducer, AudioStream, renderSynthBlock, resetSynth, seekSynth };
