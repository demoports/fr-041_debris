// Forward-only V2 PCM producer used by the loader soundtrack. The visual
// runtime never enters this module Worker; its only owned inputs are copied
// song bytes and its only large outputs are transferable, bounded PCM blocks.
import { installAudioWorker } from './audio_worker_core.js';

installAudioWorker(globalThis);
