import { AudioBlockProducer } from './audio_core.js';
import { createV2Player } from './v2.js';

// Stateful protocol controller for the forward-only loader producer. The
// Worker global is injected so the real entry stays tiny and Node tests can
// exercise multiple isolated controllers without source evaluation.
function createAudioWorkerController(scope, options = {}) {
  if (!scope || typeof scope.postMessage !== 'function') {
    throw new TypeError('audio Worker scope with postMessage() is required');
  }
  const playerFactory = options.playerFactory || options.createV2Player || createV2Player;
  const ProducerClass = options.ProducerClass || options.AudioBlockProducer || AudioBlockProducer;

  let producer = null;
  let producerPort = null;
  let generation = 0;
  let pendingBlocks = 0;
  let pumping = false;
  let endPosted = false;
  let failed = false;
  let closed = false;

  function state() {
    return {
      sequenceSamples: producer?.sequenceSamples ?? null,
      durationSamples: producer?.durationSamples ?? null,
      renderedSample: producer?.renderedSample || 0,
      endSample: producer?.endSample ?? null,
      ended: Boolean(producer?.ended),
    };
  }

  function reply(type, replyTo, values = {}) {
    scope.postMessage({ type, replyTo, ...state(), ...values });
  }

  function errorRecord(error, stage, replyTo = 0) {
    return {
      type: 'error', kind: 'content', stage, replyTo,
      generation, name: error?.name || 'Error',
      message: error?.message || String(error),
    };
  }

  function fail(error, stage, replyTo = 0) {
    if (closed || failed) return;
    failed = true;
    pendingBlocks = 0;
    const record = errorRecord(error, stage, replyTo);
    try { producerPort?.postMessage(record); } catch (_) {}
    try { scope.postMessage({ ...record, ...state() }); } catch (_) {}
  }

  function postEnd() {
    if (failed || endPosted || !producerPort || !producer) return;
    endPosted = true;
    producerPort.postMessage({
      type: 'end', generation,
      sample: producer.endSample ?? producer.renderedSample,
    });
    scope.postMessage({ type: 'state', generation, ...state() });
  }

  function pump(count) {
    pendingBlocks += Math.max(0, Math.floor(Number(count) || 0));
    if (pumping || failed || closed || !producer || !producerPort) return;
    pumping = true;
    try {
      while (pendingBlocks && !closed && !endPosted) {
        pendingBlocks--;
        const produced = producer.produceBlock(producer.blockFrames);
        if (!produced) {
          if (producer.ended) postEnd();
          break;
        }
        producerPort.postMessage({
          type: 'block', generation,
          start: produced.start, end: produced.end, frames: produced.frames,
          data: produced.block.buffer,
        }, [produced.block.buffer]);
        if (producer.ended) postEnd();
      }
    } catch (error) {
      fail(error, 'render');
    } finally {
      pumping = false;
    }
  }

  function handleProducerMessage(event) {
    const message = event.data || {};
    if (failed) return;
    if (message.generation !== undefined && message.generation !== generation) return;
    if (message.type === 'need') pump(message.blocks);
    else if (message.type === 'close') close();
  }

  function close() {
    if (closed) return;
    closed = true;
    pendingBlocks = 0;
    try { producerPort?.close?.(); } catch (_) {}
    producerPort = null;
    if (producer) producer.synth = null;
    producer = null;
    scope.close?.();
  }

  function handleMessage(event) {
    const message = event.data || {};
    if (closed && message.type !== 'close') return;
    if (message.type === 'prepare') {
      try {
        const producerOptions = { ...(message.producerOptions || {}) };
        const playerOptions = {
          ...(message.playerOptions || {}),
          sampleRate: producerOptions.sampleRate || 44100,
          checkpointMemoryBytes: 0,
          checkpointIntervalSamples: 0,
        };
        const song = new Uint8Array(message.song);
        const player = playerFactory(song, playerOptions);
        producer = new ProducerClass(player, producerOptions);
        endPosted = false;
        failed = false;
        reply('ready', message.requestId);
      } catch (error) {
        fail(error, 'prepare', message.requestId);
      }
    } else if (message.type === 'connect') {
      try {
        if (!producer) throw new Error('audio producer Worker was not prepared');
        if (!message.port) throw new TypeError('audio producer MessagePort is missing');
        try { producerPort?.close?.(); } catch (_) {}
        producerPort = message.port;
        generation = Math.max(0, Math.floor(Number(message.generation) || 0));
        producerPort.onmessage = handleProducerMessage;
        producerPort.onmessageerror = () => {
          fail(new Error('audio producer port message failed'), 'port');
        };
        producerPort.start?.();
        reply('connected', message.requestId);
      } catch (error) {
        fail(error, 'connect', message.requestId);
      }
    } else if (message.type === 'finish') {
      if (!failed && (message.generation === undefined || message.generation === generation)) {
        producer?.markProducerEnded(producer.renderedSample);
        postEnd();
      }
    } else if (message.type === 'close') {
      close();
    }
  }

  function handleMessageError() {
    fail(new Error('audio Worker control message failed'), 'control');
  }

  return { close, handleMessage, handleMessageError, state };
}

function installAudioWorker(scope, options = {}) {
  const controller = createAudioWorkerController(scope, options);
  scope.onmessage = controller.handleMessage;
  scope.onmessageerror = controller.handleMessageError;
  return controller;
}

export { createAudioWorkerController, installAudioWorker };
