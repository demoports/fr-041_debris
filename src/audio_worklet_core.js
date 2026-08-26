// Queueing and clock logic for the output AudioWorklet. Keeping the base class
// injectable makes the processor deterministic and dependency-free in Node.

const DEBRIS_OUTPUT_PROCESSOR_NAME = 'debris-output';

// V2 and the visual runtime share the production's fixed sample coordinate.
// WebAudio is allowed to run at a different device rate, so conversion belongs
// at this final PCM boundary rather than in the synth. The queue retains at
// most the producer's bounded block budget and linear interpolation keeps the
// converter allocation-free on the real-time rendering thread.
class StereoPcmResampler {
  constructor(sourceSampleRate = 44100, outputSampleRate = sourceSampleRate) {
    this.sourceSampleRate = Math.max(1, Math.round(Number(sourceSampleRate) || 44100));
    this.outputSampleRate = Math.max(1,
      Math.round(Number(outputSampleRate) || this.sourceSampleRate));
    this.ratio = this.sourceSampleRate / this.outputSampleRate;
    this.queue = [];
    this.offset = 0;
    this.phaseNumerator = 0;
    this.played = 0;
    this.queuedFrames = 0;
    this.ended = false;
  }

  reset(sample = 0) {
    this.queue.length = 0;
    this.offset = 0;
    this.phaseNumerator = 0;
    this.played = Math.max(0, Math.floor(Number(sample) || 0));
    this.queuedFrames = 0;
    this.ended = false;
  }

  push(block) {
    if (!(block instanceof Float32Array) || (block.length & 1)) {
      throw new TypeError('resampler PCM block must contain interleaved stereo Float32 samples');
    }
    if (!block.length) return;
    this.queue.push(block);
    this.queuedFrames += block.length / 2;
  }

  end() { this.ended = true; }

  get drained() { return this.ended && this.queuedFrames === 0; }

  discardConsumed() {
    while (this.queue.length && this.offset >= this.queue[0].length) {
      this.queue.shift();
      this.offset = 0;
    }
  }

  current(channel) {
    this.discardConsumed();
    return this.queue.length ? this.queue[0][this.offset + channel] : 0;
  }

  next(channel) {
    this.discardConsumed();
    if (!this.queue.length) return 0;
    const block = this.queue[0], index = this.offset + 2 + channel;
    if (index < block.length) return block[index];
    return this.queue.length > 1 ? this.queue[1][channel] : block[this.offset + channel];
  }

  consume() {
    if (!this.queuedFrames) return false;
    this.offset += 2;
    this.queuedFrames--;
    this.played++;
    this.discardConsumed();
    return true;
  }

  pull(left, right = left) {
    let written = 0;
    for (let frame = 0; frame < left.length; frame++) {
      // A downsampler can owe more than one source-frame advance. If a block
      // boundary temporarily starves it, preserve that debt until refill.
      while (this.phaseNumerator >= this.outputSampleRate && this.queuedFrames) {
        this.consume();
        this.phaseNumerator -= this.outputSampleRate;
      }
      if (this.phaseNumerator >= this.outputSampleRate || !this.queuedFrames) break;
      // Interpolation needs the following source frame. At the actual end,
      // extending the final sample gives exactly ceil(N * out / source)
      // output frames without inventing an extra production-clock sample.
      if (this.phaseNumerator > 0 && this.queuedFrames < 2 && !this.ended) break;
      const mix = this.phaseNumerator / this.outputSampleRate;
      const left0 = this.current(0), right0 = this.current(1);
      left[frame] = left0 + (this.next(0) - left0) * mix;
      right[frame] = right0 + (this.next(1) - right0) * mix;
      written++;
      this.phaseNumerator += this.sourceSampleRate;
      while (this.phaseNumerator >= this.outputSampleRate && this.queuedFrames) {
        this.consume();
        this.phaseNumerator -= this.outputSampleRate;
      }
    }
    if (this.drained) this.phaseNumerator = 0;
    return written;
  }
}

function createDebrisOutputProcessor(AudioWorkletProcessorBase) {
  if (typeof AudioWorkletProcessorBase !== 'function') {
    throw new TypeError('AudioWorkletProcessor base class is required');
  }

  return class DebrisOutputProcessor extends AudioWorkletProcessorBase {
    constructor(options = {}) {
      super();
      const configured = Number(options.processorOptions && options.processorOptions.queueBlocks) || 8;
      this.targetBlocks = Math.max(1, Math.floor(configured));
      this.reportClock = options.processorOptions?.reportClock !== false;
      const sourceSampleRate = options.processorOptions?.sourceSampleRate || 44100;
      const outputSampleRate = options.processorOptions?.outputSampleRate || sourceSampleRate;
      this.resampler = new StereoPcmResampler(sourceSampleRate, outputSampleRate);
      this.queue = this.resampler.queue;
      this.played = this.resampler.played;
      this.paused = true;
      this.ended = false;
      this.producerPort = null;
      this.generation = 0;
      this.receivedThrough = 0;
      this.primedReported = false;
      this.underrunReported = false;
      // AudioStream performs the initial fill explicitly. Counting those
      // blocks as incoming prevents the first empty quantum from requesting a
      // duplicate fill before the messages arrive.
      this.incoming = this.targetBlocks;
      this.drainReported = false;
      this.reportAt = 0;
      this.handleMessage = event => {
        const message = event.data || {};
        if (message.type === 'block') {
          if (message.generation !== undefined && message.generation !== this.generation) return;
          if (this.ended) return;
          const block = new Float32Array(message.data);
          const frames = Math.max(0, Math.floor(Number(message.frames) || block.length / 2));
          if (block.length !== frames * 2 ||
              (Number.isFinite(message.start) && message.start !== this.receivedThrough)) {
            this.failProducer('invalid or discontinuous PCM block');
            return;
          }
          if (Number.isFinite(message.end)) this.receivedThrough = Math.floor(message.end);
          else this.receivedThrough += frames;
          this.resampler.push(block);
          this.incoming = Math.max(0, this.incoming - 1);
          this.underrunReported = false;
          this.reportPrimed();
        } else if (message.type === 'reset') {
          if (Number.isFinite(message.generation)) {
            this.generation = Math.max(0, Math.floor(message.generation));
          }
          this.resampler.reset(message.sample || 0);
          this.played = this.resampler.played;
          this.incoming = Math.max(0, Math.floor(message.incoming || 0));
          this.ended = false;
          this.drainReported = false;
          this.underrunReported = false;
          this.receivedThrough = this.played;
          this.primedReported = false;
          this.reportAt = this.played;
        } else if (message.type === 'pause') {
          this.paused = Boolean(message.value);
        } else if (message.type === 'end') {
          if (message.generation !== undefined && message.generation !== this.generation) return;
          if (Number.isFinite(message.sample) && message.sample !== this.receivedThrough) {
            this.failProducer('audio producer ended at a discontinuous sample');
            return;
          }
          this.ended = true;
          this.resampler.end();
          this.incoming = 0;
          this.reportPrimed();
        } else if (message.type === 'error') {
          if (message.generation !== undefined && message.generation !== this.generation) return;
          this.failProducer(message.message || 'audio producer failed', message);
        } else if (message.type === 'abort') {
          if (message.generation !== undefined && message.generation !== this.generation) return;
          this.resampler.reset(this.played);
          this.resampler.end();
          this.incoming = 0;
          this.ended = true;
          this.drainReported = false;
          this.reportPrimed();
        } else if (message.type === 'producer-port') {
          this.producerPort?.close?.();
          this.producerPort = message.port || null;
          this.generation = Math.max(0, Math.floor(Number(message.generation) || 0));
          this.resampler.reset(message.sample);
          this.incoming = 0;
          this.played = this.resampler.played;
          this.receivedThrough = this.played;
          this.ended = false;
          this.drainReported = false;
          this.primedReported = false;
          if (this.producerPort) {
            this.producerPort.onmessage = this.handleMessage;
            this.producerPort.onmessageerror = () => {
              this.failProducer('audio producer port message failed');
            };
            this.producerPort.start?.();
            this.requestFill();
          }
        } else if (message.type === 'detach-producer') {
          this.producerPort?.close?.();
          this.producerPort = null;
        }
      };
      this.port.onmessage = this.handleMessage;
    }

    reportPrimed() {
      if (this.producerPort && !this.primedReported &&
          (this.ended || this.queue.length >= this.targetBlocks)) {
        this.primedReported = true;
        this.port.postMessage({
          type: 'primed', generation: this.generation,
          receivedThrough: this.receivedThrough, queued: this.queue.length,
        });
      }
    }

    failProducer(message, detail = {}) {
      if (this.ended) return;
      this.ended = true;
      this.resampler.end();
      this.incoming = 0;
      this.port.postMessage({
        type: 'producer-error', generation: this.generation,
        name: detail.name || 'Error', message,
      });
      this.reportPrimed();
    }

    requestFill() {
      const available = this.queue.length + this.incoming;
      if (!this.ended && available < this.targetBlocks) {
        const blocks = this.targetBlocks - available;
        this.incoming += blocks;
        (this.producerPort || this.port).postMessage({
          type: 'need', generation: this.generation, blocks,
        });
      }
    }

    process(_inputs, outputs) {
      const output = outputs[0];
      const left = output[0];
      const right = output[1] || output[0];
      left.fill(0);
      right.fill(0);
      if (!this.paused) {
        const written = this.resampler.pull(left, right);
        this.played = this.resampler.played;
        this.requestFill();
        if (written < left.length && !this.ended && !this.underrunReported) {
          this.underrunReported = true;
          this.port.postMessage({
            type: 'underrun', generation: this.generation, sample: this.played,
          });
        }
        if (this.reportClock && this.played >= this.reportAt) {
          this.reportAt = this.played + 1024;
          this.port.postMessage({
            type: 'clock', generation: this.generation,
            sample: this.played, receivedThrough: this.receivedThrough,
            queued: this.queue.length,
          });
        }
        if (this.resampler.drained && !this.drainReported) {
          this.drainReported = true;
          this.port.postMessage({
            type: 'drained', generation: this.generation, sample: this.played,
          });
        }
      }
      return true;
    }
  };
}

export { DEBRIS_OUTPUT_PROCESSOR_NAME, StereoPcmResampler, createDebrisOutputProcessor };
