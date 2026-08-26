import {
  DEBRIS_OUTPUT_PROCESSOR_NAME,
  createDebrisOutputProcessor,
} from './audio_worklet_core.js';

const DebrisOutputProcessor = createDebrisOutputProcessor(AudioWorkletProcessor);
registerProcessor(DEBRIS_OUTPUT_PROCESSOR_NAME, DebrisOutputProcessor);

export { DebrisOutputProcessor };
