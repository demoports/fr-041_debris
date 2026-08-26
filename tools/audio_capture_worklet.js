class DebrisCaptureProcessor extends AudioWorkletProcessor {
  constructor(options = {}) {
    super();
    this.targetFrames = Math.max(128, Number(options.processorOptions?.targetFrames) || 8192);
    this.started = false;
    this.reported = false;
    this.frames = 0;
    this.nonzeroLeft = 0;
    this.nonzeroRight = 0;
    this.stereoDifference = 0;
    this.invalid = 0;
    this.peakLeft = 0;
    this.peakRight = 0;
    this.energyLeft = 0;
    this.energyRight = 0;
  }

  process(inputs, outputs) {
    const input = inputs[0] || [];
    const output = outputs[0] || [];
    const left = input[0];
    const right = input[1] || left;
    const outLeft = output[0];
    const outRight = output[1] || outLeft;
    if (!left || !right || !outLeft || !outRight) return true;
    for (let index = 0; index < outLeft.length; index++) {
      const l = left[index];
      const r = right[index];
      outLeft[index] = l;
      outRight[index] = r;
      if (!this.started && (l !== 0 || r !== 0)) this.started = true;
      if (!this.started || this.frames >= this.targetFrames) continue;
      if (!Number.isFinite(l) || !Number.isFinite(r)) this.invalid++;
      if (l !== 0) this.nonzeroLeft++;
      if (r !== 0) this.nonzeroRight++;
      if (Math.abs(l - r) > 1e-12) this.stereoDifference++;
      this.peakLeft = Math.max(this.peakLeft, Math.abs(l));
      this.peakRight = Math.max(this.peakRight, Math.abs(r));
      this.energyLeft += l * l;
      this.energyRight += r * r;
      this.frames++;
    }
    if (!this.reported && this.frames >= this.targetFrames) {
      this.reported = true;
      this.port.postMessage({
        type: 'capture', frames: this.frames,
        nonzeroLeft: this.nonzeroLeft, nonzeroRight: this.nonzeroRight,
        stereoDifference: this.stereoDifference, invalid: this.invalid,
        peakLeft: this.peakLeft, peakRight: this.peakRight,
        rmsLeft: Math.sqrt(this.energyLeft / this.frames),
        rmsRight: Math.sqrt(this.energyRight / this.frames),
      });
    }
    return true;
  }
}

registerProcessor('debris-capture', DebrisCaptureProcessor);
