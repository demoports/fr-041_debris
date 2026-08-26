class DebrisWorkerSmokeCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.frames = 0;
    this.inputFrames = 0;
    this.nonzeroFrames = 0;
    this.invalid = 0;
    this.peak = 0;
    this.port.onmessage = event => {
      if (event.data?.type !== 'snapshot') return;
      this.port.postMessage({
        type: 'snapshot', requestId: event.data.requestId,
        frames: this.frames, inputFrames: this.inputFrames,
        nonzeroFrames: this.nonzeroFrames, invalid: this.invalid,
        peak: this.peak,
      });
    };
  }

  process(inputs, outputs) {
    const input = inputs[0] || [];
    const output = outputs[0] || [];
    const left = input[0];
    const right = input[1] || left;
    const outLeft = output[0];
    const outRight = output[1] || outLeft;
    if (!outLeft || !outRight) return true;
    if (left) this.inputFrames += outLeft.length;
    for (let index = 0; index < outLeft.length; index++) {
      const l = left ? left[index] : 0;
      const r = right ? right[index] : l;
      if (!Number.isFinite(l) || !Number.isFinite(r)) this.invalid++;
      const safeLeft = Number.isFinite(l) ? l : 0;
      const safeRight = Number.isFinite(r) ? r : 0;
      outLeft[index] = safeLeft;
      outRight[index] = safeRight;
      if (safeLeft !== 0 || safeRight !== 0) this.nonzeroFrames++;
      this.peak = Math.max(this.peak, Math.abs(safeLeft), Math.abs(safeRight));
    }
    this.frames += outLeft.length;
    return true;
  }
}

registerProcessor('debris-worker-smoke-capture', DebrisWorkerSmokeCapture);
