// Pure V2 block production shared by the browser stream and loader Worker.
// This module intentionally owns no WebAudio objects, ports, timers, or DOM state.

function normalizeRendered(rendered, target, frames) {
  if (rendered == null || rendered === target || typeof rendered === 'number') return target;
  if (rendered instanceof Float32Array && rendered.length >= frames * 2) {
    if (rendered !== target) target.set(rendered.subarray(0, target.length));
    return target;
  }
  if (Array.isArray(rendered) && rendered.length === 2 && rendered[0]?.length >= frames) {
    for (let index = 0; index < frames; index++) {
      target[index * 2] = rendered[0][index];
      target[index * 2 + 1] = rendered[1][index];
    }
    return target;
  }
  throw new TypeError('V2 synth returned an unsupported audio block');
}

function renderSynthBlock(synth, frames, target = new Float32Array(frames * 2)) {
  if (!synth) throw new TypeError('audio synth is missing');
  let rendered;
  if (typeof synth.render === 'function') {
    // Direct V2 API: render(interleavedTarget, frameCount). A few test/oracle
    // adapters use render(frameCount), which is accepted as a fallback.
    rendered = synth.render.length >= 2 ? synth.render(target, frames) : synth.render(frames);
  } else if (typeof synth.process === 'function') {
    rendered = synth.process.length >= 2 ? synth.process(target, frames) : synth.process(frames);
  } else {
    throw new TypeError('V2 synth has neither render() nor process()');
  }
  return normalizeRendered(rendered, target, frames);
}

function resetSynth(synth) {
  if (typeof synth.reset === 'function') synth.reset();
  else if (typeof synth.rewind === 'function') synth.rewind();
  else throw new TypeError('V2 synth cannot be reset');
}

function seekSynth(synth, targetSample, blockFrames = 8192) {
  targetSample = Math.max(0, Math.floor(targetSample));
  if (typeof synth.seekSamples === 'function') {
    const result = synth.seekSamples(targetSample);
    return Number.isFinite(result) ? Math.floor(result) : targetSample;
  }
  resetSynth(synth);
  let cursor = 0;
  while (cursor < targetSample) {
    const frames = Math.min(blockFrames, targetSample - cursor);
    renderSynthBlock(synth, frames);
    cursor += frames;
  }
  return cursor;
}

function copyAudioBytes(input) {
  if (input instanceof ArrayBuffer) return input.slice(0);
  if (ArrayBuffer.isView(input)) {
    return new Uint8Array(input.buffer, input.byteOffset, input.byteLength).slice().buffer;
  }
  throw new TypeError('Worker V2 input must be an ArrayBuffer or typed array');
}

class AudioBlockProducer {
  constructor(synth = null, options = {}) {
    this.sampleRate = options.sampleRate || 44100;
    this.blockFrames = Math.max(1, Math.floor(options.blockFrames || 2048));
    this.tailSeconds = Math.max(0, Number(options.tailSeconds ?? 12));
    this.tailSilenceSeconds = Math.max(0, Number(options.tailSilenceSeconds ?? 0.25));
    this.tailSilenceThreshold = Math.max(0, Number(options.tailSilenceThreshold ?? 1e-5));
    this.synth = synth;
    this.sequenceSamples = null;
    this.maxTailFrames = 0;
    this.silentTailFrames = 0;
    this.quietTailFrames = 0;
    this.durationSamples = Infinity;
    this.endSample = null;
    this.renderedSample = 0;
    this.ended = false;
    if (synth) this.configureTimeline();
  }

  configureTimeline() {
    this.maxTailFrames = Math.max(0, Math.round(this.tailSeconds * this.sampleRate));
    this.silentTailFrames = Math.max(1, Math.round(this.tailSilenceSeconds * this.sampleRate));
    this.sequenceSamples = null;
    if (typeof this.synth?.calcSongSamples === 'function') {
      const samples = Number(this.synth.calcSongSamples());
      if (Number.isFinite(samples) && samples >= 0) this.sequenceSamples = Math.floor(samples);
      // Timing oracles may mutate play state. Reset after probing so both the
      // Worker and local producer begin from the identical state.
      if (typeof this.synth.reset === 'function' || typeof this.synth.rewind === 'function') {
        resetSynth(this.synth);
      }
    }
    this.renderedSample = 0;
    this.resetEndState();
  }

  resetEndState() {
    this.quietTailFrames = 0;
    this.endSample = null;
    this.ended = false;
    this.durationSamples = this.sequenceSamples === null
      ? Infinity
      : this.sequenceSamples + this.maxTailFrames;
  }

  markProducerEnded(sample = this.renderedSample) {
    if (this.ended) return;
    this.ended = true;
    this.endSample = Math.max(0, Math.floor(sample));
    this.durationSamples = this.endSample;
  }

  produceBlock(requestedFrames = this.blockFrames) {
    if (this.ended) return null;
    let frames = Math.max(0, Math.floor(requestedFrames));
    if (this.sequenceSamples !== null) {
      const hardEnd = this.sequenceSamples + this.maxTailFrames;
      frames = Math.min(frames, Math.max(0, hardEnd - this.renderedSample));
      if (!frames) {
        this.markProducerEnded(this.renderedSample);
        return null;
      }
    }
    const start = this.renderedSample;
    const block = renderSynthBlock(this.synth, frames);
    this.renderedSample += frames;
    if (this.sequenceSamples !== null && this.renderedSample > this.sequenceSamples) {
      const tailStart = Math.max(start, this.sequenceSamples);
      const first = (tailStart - start) * 2;
      let peak = 0;
      for (let index = first; index < block.length; index++) {
        peak = Math.max(peak, Math.abs(block[index]));
      }
      const tailFrames = this.renderedSample - tailStart;
      this.quietTailFrames = peak <= this.tailSilenceThreshold
        ? this.quietTailFrames + tailFrames
        : 0;
      const hardEnd = this.sequenceSamples + this.maxTailFrames;
      if (this.quietTailFrames >= this.silentTailFrames || this.renderedSample >= hardEnd) {
        this.markProducerEnded(this.renderedSample);
      }
    }
    if (this.sequenceSamples !== null &&
        this.renderedSample >= this.sequenceSamples + this.maxTailFrames) {
      this.markProducerEnded(this.renderedSample);
    }
    return { block, frames, start, end: this.renderedSample };
  }
}

export { AudioBlockProducer, copyAudioBytes, renderSynthBlock, resetSynth, seekSynth };
