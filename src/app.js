// Browser lifecycle for fr-041: production data -> KX graph -> V2 sample clock
// -> semantic frame jobs -> WebGL2. No native or Wasm component is involved.
import { bitmapFromRGBA, newBitmap, setBitmapTextureSizeOffset } from './bitmap.js';
import { loadProductionData } from './data.js';
import { advanceEffectFrame } from './effects.js';
import { parseKX } from './kx.js';
import { meshStorageStats } from './mesh.js';
import { minMeshStorageStats } from './minmesh.js';
import { createOperatorHandlers } from './operators.js';
import { Renderer } from './renderer.js';
import { Environment, KC_MESH, KC_MINMESH, KC_SCENE, Runtime } from './runtime.js';
import { AudioStream } from './audio.js';
import { createV2Player, V2Player } from './v2.js';

const DEFAULT_RATE = 44100;
const PRODUCTION_ASPECT = 2;
const REFERENCE_REPLAY_FPS = 30;
const MAX_DEBUG_SECONDS = 3600;
const DEFAULT_PLAYBACK_WARMUP_BYTES = 256 * 1024 * 1024;
const DEFAULT_PLAYBACK_WARMUP_RESOURCE_BYTES = 64 * 1024 * 1024;
const DEFAULT_SNAPSHOT_BUDGET_BYTES = 32 * 1024 * 1024;
const MAX_SNAPSHOT_BUDGET_BYTES = 128 * 1024 * 1024;
const MAX_SNAPSHOT_COUNT = 64;
const DEFAULT_DEPENDENCIES = Object.freeze({
  AudioStream,
  Environment,
  Renderer,
  Runtime,
  advanceEffectFrame,
  bitmapFromRGBA,
  createOperatorHandlers,
  createV2Player,
  loadProductionData,
  meshStorageStats,
  minMeshStorageStats,
  newBitmap,
  parseKX,
  setBitmapTextureSizeOffset,
  V2Player,
});

function lifecycleAbortError(reason = 'Debris initialization was cancelled') {
  if (reason?.name === 'AbortError') return reason;
  let error;
  if (typeof DOMException === 'function') {
    error = new DOMException(String(reason || 'Debris initialization was cancelled'), 'AbortError');
  } else {
    error = new Error(String(reason || 'Debris initialization was cancelled'));
    error.name = 'AbortError';
  }
  if (reason instanceof Error && reason !== error) {
    try { error.cause = reason; } catch (_) {}
  }
  return error;
}

function isAbortError(error) { return error?.name === 'AbortError'; }

function throwIfPreparationAborted(options = {}) {
  if (options.signal?.aborted) throw lifecycleAbortError(options.signal.reason);
  if (options.shouldAbort?.()) throw lifecycleAbortError();
}

function waitForLifecycle(value, signal) {
  if (!signal) return Promise.resolve(value);
  if (signal.aborted) return Promise.reject(lifecycleAbortError(signal.reason));
  return new Promise((resolve, reject) => {
    let settled = false;
    const finish = (callback, result) => {
      if (settled) return;
      settled = true;
      signal.removeEventListener('abort', onAbort);
      callback(result);
    };
    const onAbort = () => finish(reject, lifecycleAbortError(signal.reason));
    signal.addEventListener('abort', onAbort, { once: true });
    // Attach both reactions even if cancellation wins. A late failure from an
    // uncooperative dependency is then consumed instead of becoming an
    // unhandled rejection after the app has already returned to the launcher.
    Promise.resolve(value).then(
      result => finish(resolve, result),
      error => finish(reject, error),
    );
  });
}

function documentDuration(document, sampleRate = DEFAULT_RATE, player = null) {
  const rate = Number(sampleRate);
  sampleRate = Number.isFinite(rate) && rate > 0 ? rate : DEFAULT_RATE;
  if (player && typeof player.calcSongSamples === 'function') {
    const samples = Number(player.calcSongSamples());
    if (Number.isFinite(samples) && samples > 0) return samples / sampleRate;
  }
  // KX SongLength is a 16.16 beat value.
  const songLength = Number(document?.songLength);
  const songBPM = Number(document?.songBPM);
  if (!Number.isFinite(songLength) || songLength <= 0 ||
      !Number.isFinite(songBPM) || songBPM <= 0) return 0;
  return (songLength / 65536) * 60 / songBPM;
}

function sampleAtSeconds(seconds, sampleRate = DEFAULT_RATE) {
  seconds = Number(seconds);
  sampleRate = Number(sampleRate);
  if (!Number.isFinite(seconds) || seconds <= 0) return 0;
  if (!Number.isFinite(sampleRate) || sampleRate <= 0) sampleRate = DEFAULT_RATE;
  const sample = seconds * sampleRate;
  return Number.isFinite(sample)
    ? Math.min(Number.MAX_SAFE_INTEGER, Math.floor(sample))
    : Number.MAX_SAFE_INTEGER;
}

function normalizedSample(value, fallback = 0) {
  value = Number(value);
  if (!Number.isFinite(value)) value = Number(fallback);
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(Number.MAX_SAFE_INTEGER, Math.floor(value)));
}

function positiveInteger(value, fallback, maximum = Number.MAX_SAFE_INTEGER) {
  value = Number(value);
  if (!Number.isFinite(value) || value <= 0) value = fallback;
  return Math.max(1, Math.min(maximum, Math.floor(value)));
}

function* replaySampleSequence(startSample, targetSample, sampleRate = DEFAULT_RATE,
    frameRate = REFERENCE_REPLAY_FPS) {
  let sample = normalizedSample(startSample);
  const target = Math.max(sample, normalizedSample(targetSample, sample));
  const rate = positiveInteger(sampleRate, DEFAULT_RATE);
  const fps = positiveInteger(frameRate, REFERENCE_REPLAY_FPS);
  // Derive every boundary from the global frame index. Adding a rounded frame
  // size would accumulate drift whenever the sample rate is not divisible by
  // the reference video's 30 fps cadence.
  let frame = Math.floor(sample * fps / rate) + 1;
  while (sample < target) {
    let boundary = Math.floor(frame * rate / fps);
    while (boundary <= sample) boundary = Math.floor(++frame * rate / fps);
    sample = Math.min(target, boundary);
    yield sample;
    if (sample === boundary) frame++;
  }
}

function textureSizeOffset(quality) {
  if (Number.isFinite(Number(quality))) return Math.max(-3, Math.min(0, Number(quality) | 0));
  switch (String(quality || '').toLowerCase()) {
    case 'high': case 'ultra': case 'full': return 0;
    case 'normal': case 'reduced': case 'medium': return -1;
    case 'low': return -2;
    default: return 0;
  }
}

function decodeRenderPixels(gl, width, height) {
  const bottomUp = new Uint8Array(width * height * 4);
  const topDown = new Uint8Array(bottomUp.length);
  gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, bottomUp);
  const rowBytes = width * 4;
  for (let y = 0; y < height; y++) {
    topDown.set(bottomUp.subarray(y * rowBytes, (y + 1) * rowBytes), (height - y - 1) * rowBytes);
  }
  return topDown;
}

function makeBitmapRenderer(runtime, renderer, dependencies = DEFAULT_DEPENDENCIES) {
  return record => {
    const bitmap = dependencies.newBitmap(record.widthExponent | 0, record.heightExponent | 0);
    const width = bitmap.width, height = bitmap.height;
    const environment = new dependencies.Environment(runtime);
    environment.initView();
    environment.aspect = width / height;
    environment.initFrame(0, 0);
    try {
      const input = record.op.inputs[0];
      if (!input) return bitmap;
      input.exec(environment);
      const output = environment.lastOutput;
      if (!output) return bitmap;
      renderer.render(output, environment, { width, height, pixelRatio: 1 });
      const rgba = decodeRenderPixels(renderer.gl, width, height);
      return dependencies.bitmapFromRGBA(width, height, rgba);
    } finally {
      environment.exitFrame();
    }
  };
}

function geometryStorageBuffers(geometry, output = new Set()) {
  if (!geometry || typeof geometry !== 'object') return output;
  // Prepared MinMesh exposes animation through an enumerable lazy getter.
  // Reading that getter after immutable topology release would try to expand
  // data which was deliberately discarded. Byte accounting only needs owned
  // data properties, so inspect descriptors without invoking accessors.
  for (const descriptor of Object.values(Object.getOwnPropertyDescriptors(geometry))) {
    if (!Object.prototype.hasOwnProperty.call(descriptor, 'value')) continue;
    const value = descriptor.value;
    if (ArrayBuffer.isView(value) && value.buffer) output.add(value.buffer);
  }
  return output;
}

function preparedGeometryBytes(geometry, buffers = new Set()) {
  geometryStorageBuffers(geometry, buffers);
  let bytes = 0;
  for (const buffer of buffers) bytes += buffer.byteLength;
  return bytes;
}

function compactTopologyBuffers(mesh, output = new Set()) {
  const storage = mesh?._compact;
  if (!storage) return output;
  const count = value => {
    if (ArrayBuffer.isView(value) && value.buffer) output.add(value.buffer);
  };
  for (const value of Object.values(storage)) count(value);
  for (const value of Object.values(storage.animation || {})) count(value);
  return output;
}

function legacyTerminalStaticGeometry(runtime) {
  const operations = runtime?.operations || [];
  const references = new Map();
  const consumers = new Map();
  const addConsumer = (producer, consumer) => {
    if (!producer) return;
    let list = consumers.get(producer);
    if (!list) consumers.set(producer, list = []);
    list.push(consumer);
  };
  for (const operation of operations) {
    const cache = operation?.cache;
    if (cache?.kind === 'mesh' || cache?.kind === 'minmesh') {
      let list = references.get(cache);
      if (!list) references.set(cache, list = []);
      list.push(operation);
    }
    for (const input of operation?.inputs || []) addConsumer(input, operation);
    for (const link of operation?.links || []) addConsumer(link, operation);
  }
  const result = [];
  for (const [mesh, owners] of references) {
    // Ownership aliases deliberately make upstream operator caches observable.
    // Releasing their topology would make a later incremental recalculation
    // invalid, so playback compaction is restricted to one terminal cache.
    if (owners.length !== 1 || mesh.released || mesh.topologyReleasedForPlayback ||
      mesh._sharedVertices || !mesh._compact || typeof mesh.releaseTopologyForPlayback !== 'function') continue;
    const operation = owners[0];
    if (operation.changed || operation._calcState !== 2 || operation.classInfo?.dynamic !== false) continue;
    if (mesh.hasAnimation?.()) continue;
    const downstream = consumers.get(operation) || [];
    if (!downstream.length || !downstream.some(consumer => consumer.outputClassId === KC_SCENE)) continue;
    if (downstream.some(consumer =>
      consumer.outputClassId === KC_MESH || consumer.outputClassId === KC_MINMESH)) continue;
    result.push({ mesh, operation });
  }
  return result;
}

// Immutable precalc captures terminal meshes inside Scene objects. Those
// captured identities are the renderer's playback inputs; the procedural
// operator topology behind them is never evaluated again. Identity aliases
// are safe here because preparation/release happens once per object, not once
// per owning operation. An uncaptured execution boundary remains excluded:
// arbitrary dynamic handlers may inspect its procedural representation.
function immutableStaticGeometryPlan(runtime) {
  const operations = runtime?.operations || [];
  const owners = new Map();
  for (const operation of operations) {
    const mesh = operation?.cache;
    if (mesh?.kind !== 'mesh' && mesh?.kind !== 'minmesh') continue;
    let list = owners.get(mesh);
    if (!list) owners.set(mesh, list = []);
    list.push(operation);
  }
  const identities = new Set(owners.keys());
  const captured = new Set();
  const visited = new Set();
  const operationSet = new Set(operations);
  const skipKeys = new Set(['op', 'runtime', 'environment']);
  const visit = value => {
    if (!value || typeof value !== 'object' || ArrayBuffer.isView(value) ||
        operationSet.has(value) || value === runtime || value === runtime?.environment) return;
    if (identities.has(value)) {
      captured.add(value);
      return;
    }
    if (visited.has(value)) return;
    visited.add(value);
    if (Array.isArray(value)) {
      for (const item of value) visit(item);
      return;
    }
    for (const [key, descriptor] of Object.entries(Object.getOwnPropertyDescriptors(value))) {
      if (skipKeys.has(key) || !Object.prototype.hasOwnProperty.call(descriptor, 'value')) continue;
      visit(descriptor.value);
    }
  };
  for (const operation of operations) {
    if (operation?.cache && !identities.has(operation.cache)) visit(operation.cache);
  }

  const candidates = [];
  const excluded = {
    uncapturedExecutionIdentities: 0,
    incompleteOwnerIdentities: 0,
    animatedIdentities: 0,
    unavailableTopologyIdentities: 0,
    sharedExpandedIdentities: 0,
  };
  for (const [mesh, meshOwners] of owners) {
    if (!captured.has(mesh)) {
      excluded.uncapturedExecutionIdentities++;
      continue;
    }
    if (meshOwners.some(operation => !operationIsStaticAndComplete(operation))) {
      excluded.incompleteOwnerIdentities++;
      continue;
    }
    const animated = typeof mesh.hasAnimation === 'function'
      ? mesh.hasAnimation()
      : (mesh.storageSummary?.()?.bones || 0) > 0;
    if (animated) {
      excluded.animatedIdentities++;
      continue;
    }
    if (mesh._sharedVertices) {
      excluded.sharedExpandedIdentities++;
      continue;
    }
    if (mesh.released || mesh.topologyReleasedForPlayback || !mesh._compact ||
        typeof mesh.releaseTopologyForPlayback !== 'function' || typeof mesh.prepare !== 'function') {
      excluded.unavailableTopologyIdentities++;
      continue;
    }
    candidates.push({ mesh, operation: meshOwners[0], operations: meshOwners });
  }
  return {
    immutable: true,
    candidates,
    capturedIdentities: captured.size,
    candidateReferences: candidates.reduce((sum, item) => sum + item.operations.length, 0),
    sharedCandidateIdentities: candidates.filter(item => item.operations.length > 1).length,
    ...excluded,
  };
}

function staticGeometryPlan(runtime) {
  if (runtime?.immutablePlayback && runtime?.playbackCachePruning?.immutable) {
    return immutableStaticGeometryPlan(runtime);
  }
  const candidates = legacyTerminalStaticGeometry(runtime);
  return {
    immutable: false,
    candidates,
    capturedIdentities: candidates.length,
    candidateReferences: candidates.length,
    sharedCandidateIdentities: 0,
    uncapturedExecutionIdentities: 0,
    incompleteOwnerIdentities: 0,
    animatedIdentities: 0,
    unavailableTopologyIdentities: 0,
    sharedExpandedIdentities: 0,
  };
}

function terminalStaticGeometry(runtime) {
  return staticGeometryPlan(runtime).candidates;
}

async function prepareStaticTerminalGeometry(runtime, options = {}) {
  throwIfPreparationAborted(options);
  if (runtime?.immutablePlayback && runtime.playbackGeometryCompaction) {
    return runtime.playbackGeometryCompaction;
  }
  const plan = staticGeometryPlan(runtime);
  throwIfPreparationAborted(options);
  const candidates = plan.candidates;
  const budgetMilliseconds = Math.max(0, Number(options.budgetMilliseconds ?? 12));
  const now = () => globalThis.performance?.now?.() ?? Date.now();
  const yieldThread = typeof options.yield === 'function' ? options.yield : nextFramePromise;
  let deadline = now() + budgetMilliseconds;
  const yieldIfNeeded = () => {
    throwIfPreparationAborted(options);
    if (now() < deadline) return null;
    return (async () => {
      await waitForLifecycle(yieldThread(), options.signal);
      throwIfPreparationAborted(options);
      deadline = now() + budgetMilliseconds;
    })();
  };
  const stats = {
    candidates: candidates.length,
    immutable: plan.immutable,
    capturedIdentities: plan.capturedIdentities,
    candidateReferences: plan.candidateReferences,
    sharedCandidateIdentities: plan.sharedCandidateIdentities,
    uncapturedExecutionIdentities: plan.uncapturedExecutionIdentities,
    incompleteOwnerIdentities: plan.incompleteOwnerIdentities,
    animatedIdentities: plan.animatedIdentities,
    unavailableTopologyIdentities: plan.unavailableTopologyIdentities,
    sharedExpandedIdentities: plan.sharedExpandedIdentities,
    prepared: 0,
    marked: 0,
    queued: 0,
    pendingIdentities: candidates.length,
    releasedIdentities: 0,
    releasedAliasReferences: 0,
    releasedTopologyBytes: 0,
    candidateTopologyBytes: 0,
    potentiallyReclaimableTopologyBytes: 0,
    pendingReclaimableTopologyBytes: 0,
    retainedSharedTopologyBytes: 0,
    preparedGeometryBytes: 0,
    createdPreparedGeometryBytes: 0,
    deferredPreparedGeometryBytes: 0,
    existingPreparedGeometryBytes: 0,
    conversionNetBytes: 0,
    estimatedNetBytes: 0,
    aborted: false,
  };

  // A compact buffer can be shared by distinct shells. Count it as reclaimed
  // only when every Runtime identity that references it is in this release.
  const candidateMeshes = new Set(candidates.map(item => item.mesh));
  const topologyUse = new Map();
  const candidateBufferIds = new Map();
  const bufferStates = [];
  const seenMeshes = new Set();
  for (const operation of runtime?.operations || []) {
    throwIfPreparationAborted(options);
    const mesh = operation?.cache;
    if (mesh && (mesh.kind === 'mesh' || mesh.kind === 'minmesh') && !seenMeshes.has(mesh)) {
      seenMeshes.add(mesh);
      for (const buffer of compactTopologyBuffers(mesh)) {
        throwIfPreparationAborted(options);
        let use = topologyUse.get(buffer);
        if (!use) {
          use = { id: bufferStates.length, bytes: buffer.byteLength, remaining: 0, retained: false };
          topologyUse.set(buffer, use);
          bufferStates.push(use);
        }
        if (candidateMeshes.has(mesh)) {
          use.remaining++;
          let ids = candidateBufferIds.get(mesh);
          if (!ids) candidateBufferIds.set(mesh, ids = []);
          ids.push(use.id);
        } else use.retained = true;
      }
    }
    const yielding = yieldIfNeeded();
    if (yielding) await yielding;
  }
  for (const use of bufferStates) {
    throwIfPreparationAborted(options);
    if (use.remaining) {
      stats.candidateTopologyBytes += use.bytes;
      if (use.retained) stats.retainedSharedTopologyBytes += use.bytes;
      else stats.potentiallyReclaimableTopologyBytes += use.bytes;
    }
    const yielding = yieldIfNeeded();
    if (yielding) await yielding;
  }
  stats.pendingReclaimableTopologyBytes = stats.potentiallyReclaimableTopologyBytes;
  // The Map was needed only for physical-byte classification. Do not let its
  // keys pin every topology buffer while candidates are later drawn. Numeric
  // ids are sufficient to update exact shared-buffer telemetry on release.
  topologyUse.clear();
  seenMeshes.clear();

  // The legacy helper remains eager for its narrow, non-immutable callers.
  // Production immutable playback is lazy below: creating every timeline
  // geometry here was measured to add more backing storage than it released.
  if (!plan.immutable) {
    const preparedBuffers = new Set();
    const createdPreparedBuffers = new Set();
    for (const { mesh, operation, operations = [operation] } of candidates) {
      throwIfPreparationAborted(options);
      const hadPrepared = Boolean(mesh._prepared);
      const prepared = mesh.prepare({ releaseTopology: true });
      stats.prepared++;
      stats.marked++;
      stats.releasedIdentities++;
      stats.releasedAliasReferences += operations.length;
      stats.pendingIdentities--;
      geometryStorageBuffers(prepared, preparedBuffers);
      if (!hadPrepared) geometryStorageBuffers(prepared, createdPreparedBuffers);
      options.onProgress?.({ ...stats, opId: operation.id });
      const yielding = yieldIfNeeded();
      if (yielding) await yielding;
    }
    stats.releasedTopologyBytes = stats.potentiallyReclaimableTopologyBytes;
    stats.pendingReclaimableTopologyBytes = 0;
    stats.preparedGeometryBytes = Array.from(preparedBuffers)
      .reduce((sum, buffer) => sum + buffer.byteLength, 0);
    stats.createdPreparedGeometryBytes = Array.from(createdPreparedBuffers)
      .reduce((sum, buffer) => sum + buffer.byteLength, 0);
    stats.conversionNetBytes = stats.preparedGeometryBytes - stats.releasedTopologyBytes;
    stats.estimatedNetBytes = stats.createdPreparedGeometryBytes - stats.releasedTopologyBytes;
    return stats;
  }

  const preparedBuffers = new Set();
  const recordRelease = (geometry, ids, aliasReferences, deferred) => {
    let preparedBytes = 0;
    for (const buffer of geometryStorageBuffers(geometry)) {
      if (preparedBuffers.has(buffer)) continue;
      preparedBuffers.add(buffer);
      preparedBytes += buffer.byteLength;
    }
    stats.preparedGeometryBytes += preparedBytes;
    if (deferred) stats.deferredPreparedGeometryBytes += preparedBytes;
    else stats.existingPreparedGeometryBytes += preparedBytes;
    stats.prepared++;
    stats.releasedIdentities++;
    stats.releasedAliasReferences += aliasReferences;
    stats.pendingIdentities = Math.max(0, stats.pendingIdentities - 1);
    for (const id of ids) {
      const use = bufferStates[id];
      use.remaining = Math.max(0, use.remaining - 1);
      if (!use.remaining && !use.retained) stats.releasedTopologyBytes += use.bytes;
    }
    stats.pendingReclaimableTopologyBytes = Math.max(0,
      stats.potentiallyReclaimableTopologyBytes - stats.releasedTopologyBytes);
    // Deferred prepared buffers were required by the renderer independently
    // of compaction. The incremental memory effect of this policy is only the
    // topology it drops; conversionNetBytes remains useful shape telemetry.
    stats.conversionNetBytes = stats.preparedGeometryBytes - stats.releasedTopologyBytes;
    stats.estimatedNetBytes = -stats.releasedTopologyBytes;
    if (!stats.pendingIdentities) preparedBuffers.clear();
  };

  const installedMarkers = [];
  try {
    runtime.playbackGeometryCompaction = stats;
    for (const { mesh, operation, operations = [operation] } of candidates) {
      throwIfPreparationAborted(options);
      const ids = candidateBufferIds.get(mesh) || [];
      const deferred = !mesh._prepared;
      const marker = {
        release: geometry => recordRelease(geometry, ids, operations.length, deferred),
      };
      Object.defineProperty(mesh, '_playbackTopologyRelease', {
        configurable: true, writable: true, value: marker,
      });
      installedMarkers.push({ mesh, marker });
      stats.marked++;
      if (deferred) stats.queued++;
      else mesh.releaseTopologyForPlayback();
      options.onProgress?.({ ...stats, opId: operation.id });
      const yielding = yieldIfNeeded();
      if (yielding) await yielding;
    }
    candidateBufferIds.clear();
    candidateMeshes.clear();
    return stats;
  } catch (error) {
    stats.aborted = isAbortError(error);
    // Deferred release markers close over the complete topology accounting
    // graph. Remove every marker that has not already fired so cancelling a
    // discarded Runtime cannot retain those buffers through a mesh identity.
    for (const { mesh, marker } of installedMarkers) {
      if (mesh._playbackTopologyRelease !== marker) continue;
      delete mesh._playbackTopologyRelease;
    }
    if (runtime.playbackGeometryCompaction === stats) {
      runtime.playbackGeometryCompaction = null;
    }
    candidateBufferIds.clear();
    candidateMeshes.clear();
    preparedBuffers.clear();
    throw error;
  }
}

// Mesh.prepare() and MinMesh.prepare() are derivable caches. Precalc-time
// Bitmap_Render jobs must not leave those CPU arrays attached to Runtime's
// long-lived operator caches after their temporary renderer has been deleted.
function discardPreparedGeometry(runtime) {
  const seen = new Set();
  let geometries = 0, bytes = 0;
  for (const operation of runtime?.operations || []) {
    const mesh = operation?.cache;
    if (!mesh || seen.has(mesh) || (mesh.kind !== 'mesh' && mesh.kind !== 'minmesh')) continue;
    seen.add(mesh);
    if (mesh._prepared) {
      geometries++;
      bytes += preparedGeometryBytes(mesh._prepared);
      mesh._prepared = null;
    }
    if (!mesh.released && !mesh.topologyReleasedForPlayback) mesh.compact?.();
  }
  return { geometries, bytes };
}

function playbackCacheKind(cache, operation = null) {
  if (!cache || typeof cache !== 'object') return null;
  const output = String(operation?.classInfo?.outputClass || cache.outputClass || '').toLowerCase();
  const name = String(cache.kind || cache.type || cache.constructor?.name || '').toLowerCase();
  if (output.includes('minmesh') || name.includes('minmesh')) return 'minmesh';
  if (output === 'kc_mesh' || name === 'mesh') return 'mesh';
  if (output === 'kc_bitmap' || name === 'bitmap' ||
      (ArrayBuffer.isView(cache.data) && cache.data.BYTES_PER_ELEMENT === 2 &&
       Number.isFinite(cache.width) && Number.isFinite(cache.height))) return 'bitmap';
  return null;
}

function operationIsStaticAndComplete(operation) {
  return Boolean(operation && operation.classInfo?.dynamic === false &&
    operation.changed === false && operation._calcState === 2);
}

function cacheStorageBuffers(cache, candidateIdentities, output = new Set()) {
  const seen = new Set();
  const semanticKeys = new Set([
    'children', 'drawMesh', 'effect', 'material', 'materials', 'texture', 'textures',
  ]);
  const isBuffer = value => value && typeof value === 'object' &&
    typeof value.byteLength === 'number' &&
    Object.prototype.toString.call(value).includes('ArrayBuffer');
  const visit = (value, root = false) => {
    if (!value || typeof value !== 'object' || seen.has(value)) return;
    if (ArrayBuffer.isView(value)) {
      if (value.buffer) output.add(value.buffer);
      return;
    }
    if (isBuffer(value)) { output.add(value); return; }
    if (!root && candidateIdentities.has(value)) return;
    seen.add(value);
    if (Array.isArray(value)) {
      for (const item of value) visit(item);
      return;
    }
    for (const [key, descriptor] of Object.entries(Object.getOwnPropertyDescriptors(value))) {
      if (!Object.prototype.hasOwnProperty.call(descriptor, 'value') || semanticKeys.has(key)) continue;
      visit(descriptor.value);
    }
  };
  // These are the only large, cache-owned stores in Bitmap, Mesh and MinMesh.
  // Semantic references such as materials/textures are deliberately excluded:
  // their buffers have their own candidate identity and lifetime decision.
  visit(cache.data);
  visit(cache.Data);
  visit(cache._compact, true);
  visit(cache._prepared, true);
  return output;
}

// Precalc builds self-contained Scene/Material/Effect objects for playback,
// but Runtime.operations otherwise keeps every procedural intermediate alive.
// This pass is intentionally identity-conservative: any candidate referenced
// by dynamic execution, by a root/event edge, by a captured playback object,
// or by an incomplete/unknown owner is retained through every alias. Only an
// identity whose complete set of owners is proven static and dead is cleared.
function pruneImmutablePlaybackCaches(runtime) {
  if (!runtime || !Array.isArray(runtime.operations)) {
    throw new TypeError('immutable playback cache pruning needs a Runtime');
  }
  if (!runtime.precalculated) {
    throw new Error('immutable playback cache pruning requires completed precalc');
  }
  if (runtime.immutablePlayback && runtime.playbackCachePruning) {
    return runtime.playbackCachePruning;
  }

  const operations = runtime.operations;
  const operationSet = new Set(operations);
  const owners = new Map();
  const kinds = new Map();
  for (const operation of operations) {
    const cache = operation?.cache;
    const kind = playbackCacheKind(cache, operation);
    if (!kind) continue;
    let list = owners.get(cache);
    if (!list) owners.set(cache, list = []);
    list.push(operation);
    kinds.set(cache, kind);
  }
  const candidateIdentities = new Set(owners.keys());
  const retained = new Set();
  const executionVisited = new Set();

  // Follow the executable graph until the first procedural asset cache. Exec
  // handlers consume that boundary cache, never its static construction DAG.
  const retainExecutionGraph = operation => {
    if (!operation || executionVisited.has(operation)) return;
    executionVisited.add(operation);
    if (candidateIdentities.has(operation.cache)) {
      retained.add(operation.cache);
      return;
    }
    for (const input of operation.inputs || []) retainExecutionGraph(input);
    for (const link of operation.links || []) retainExecutionGraph(link);
  };
  for (const operation of operations) {
    if (operation?.classInfo?.dynamic !== false) retainExecutionGraph(operation);
  }
  for (const root of runtime.roots || []) retainExecutionGraph(root);
  for (const event of runtime.events || []) retainExecutionGraph(event?.op);

  // Unknown, changed or incomplete owners cannot participate in pruning. One
  // such alias retains the complete identity, including all static aliases.
  for (const [cache, cacheOwners] of owners) {
    if (cacheOwners.some(operation => !operationIsStaticAndComplete(operation))) retained.add(cache);
  }

  // Find assets captured inside terminal Scene/Material/Effect-style caches.
  // Data descriptors avoid triggering lazy geometry getters, and operation /
  // runtime backreferences are barriers so the scan cannot accidentally turn
  // into a traversal of the whole procedural graph.
  const capturedSeen = new Set();
  const capturedCandidateSeen = new Set();
  const skipKeys = new Set(['op', 'runtime', 'environment']);
  const visitCaptured = value => {
    if (!value || typeof value !== 'object' || ArrayBuffer.isView(value) ||
        operationSet.has(value) || value === runtime || value === runtime.environment) return;
    if (candidateIdentities.has(value)) {
      retained.add(value);
      if (capturedCandidateSeen.has(value)) return;
      capturedCandidateSeen.add(value);
      // A retained mesh can itself own semantic material/texture references.
      for (const key of ['children', 'drawMesh', 'effect', 'material', 'materials', 'texture', 'textures']) {
        const descriptor = Object.getOwnPropertyDescriptor(value, key);
        if (descriptor && Object.prototype.hasOwnProperty.call(descriptor, 'value')) {
          visitCaptured(descriptor.value);
        }
      }
      return;
    }
    if (capturedSeen.has(value)) return;
    capturedSeen.add(value);
    if (Array.isArray(value)) {
      for (const item of value) visitCaptured(item);
      return;
    }
    for (const [key, descriptor] of Object.entries(Object.getOwnPropertyDescriptors(value))) {
      if (skipKeys.has(key) || !Object.prototype.hasOwnProperty.call(descriptor, 'value')) continue;
      visitCaptured(descriptor.value);
    }
  };
  for (const operation of operations) {
    if (operation?.cache && !candidateIdentities.has(operation.cache)) visitCaptured(operation.cache);
  }

  const released = new Set();
  const releasedKinds = {};
  let clearedReferences = 0;
  for (const [cache, cacheOwners] of owners) {
    if (retained.has(cache)) continue;
    // All owners were checked above as static and complete. Clear references
    // rather than mutating the cache object: this cannot corrupt a shared
    // backing store if an embedding we do not own exists outside Runtime.
    released.add(cache);
    const kind = kinds.get(cache);
    releasedKinds[kind] = (releasedKinds[kind] || 0) + 1;
    for (const operation of cacheOwners) {
      operation.cache = null;
      operation._playbackCachePruned = true;
      clearedReferences++;
    }
  }

  // Physical-byte telemetry is deliberately a lower bound: it counts only
  // known ArrayBuffers in data/_compact/_prepared and deduplicates storage
  // shared by aliases or distinct cache shells.
  const bufferUse = new Map();
  for (const cache of candidateIdentities) {
    for (const buffer of cacheStorageBuffers(cache, candidateIdentities)) {
      let use = bufferUse.get(buffer);
      if (!use) bufferUse.set(buffer, use = { retained: false, released: false });
      if (released.has(cache)) use.released = true;
      else use.retained = true;
    }
  }
  let candidateKnownBytes = 0, estimatedReclaimableBytes = 0, retainedKnownBytes = 0;
  for (const [buffer, use] of bufferUse) {
    const bytes = Math.max(0, Number(buffer.byteLength) || 0);
    candidateKnownBytes += bytes;
    if (use.released && !use.retained) estimatedReclaimableBytes += bytes;
    else retainedKnownBytes += bytes;
  }

  const stats = Object.freeze({
    immutable: true,
    candidateReferences: Array.from(owners.values()).reduce((sum, list) => sum + list.length, 0),
    candidateIdentities: candidateIdentities.size,
    retainedIdentities: candidateIdentities.size - released.size,
    releasedIdentities: released.size,
    clearedReferences,
    sharedIdentities: Array.from(owners.values()).filter(list => list.length > 1).length,
    releasedKinds: Object.freeze({ ...releasedKinds }),
    candidateKnownBytes,
    retainedKnownBytes,
    estimatedReclaimableBytes,
  });
  runtime.immutablePlayback = true;
  runtime.playbackCachePruning = stats;
  return stats;
}

// Build a chronological, descriptor-safe queue of immutable playback
// resources. Event caches are visited before the operation-table fallback so
// a bounded warm-up spends its budget on the earliest scenes first. Accessors
// and Runtime/operation backreferences are barriers: collecting resources
// must never expand a released topology or execute the timeline.
function collectPlaybackResourcePlan(runtime) {
  const operations = runtime?.operations || [];
  const operationSet = new Set(operations);
  const ownerIds = new Map();
  for (const operation of operations) {
    if (operation?.cache && !ownerIds.has(operation.cache)) {
      ownerIds.set(operation.cache, operation.id);
    }
  }
  const tasks = [];
  const meshes = new Set();
  const materials = new Set();
  const visited = new Set();
  const visitedOperations = new Set();
  const semanticKeys = new Set([
    'children', 'drawMesh', 'effect', 'material', 'materials',
    'texture', 'textures', 'geometry', 'mesh', 'clusters', '_clusters',
  ]);
  const visit = value => {
    if (!value || typeof value !== 'object' || ArrayBuffer.isView(value) ||
        value instanceof ArrayBuffer || operationSet.has(value) || visited.has(value)) return;
    visited.add(value);
    if (Array.isArray(value)) {
      for (const item of value) visit(item);
      return;
    }
    const kind = String(value.kind || '').toLowerCase();
    if (kind === 'mesh' || kind === 'minmesh') {
      if (!meshes.has(value)) {
        meshes.add(value);
        tasks.push({ kind: 'mesh', value, sourceId: ownerIds.get(value) });
      }
      // Compacted MinMesh stores its material-bearing clusters behind the
      // ordinary `clusters` accessor. Read only the owned compact descriptor
      // so planning cannot expand vertex/face topology.
      const compact = Object.getOwnPropertyDescriptor(value, '_compact')?.value;
      const compactClusters = compact &&
        Object.getOwnPropertyDescriptor(compact, 'clusters')?.value;
      if (Array.isArray(compactClusters)) {
        for (const cluster of compactClusters) {
          visit(Object.getOwnPropertyDescriptor(cluster, 'material')?.value);
        }
      }
    } else if (kind === 'material') {
      if (!materials.has(value)) {
        materials.add(value);
        tasks.push({ kind: 'material', value, sourceId: ownerIds.get(value) });
      }
    } else if (kind === 'bitmap') {
      // Texture tasks are derived later from each compiled material pass. A
      // linked but unused bitmap must not consume the residency budget.
      return;
    }
    for (const [key, descriptor] of Object.entries(Object.getOwnPropertyDescriptors(value))) {
      if (!semanticKeys.has(key) || !Object.prototype.hasOwnProperty.call(descriptor, 'value')) {
        continue;
      }
      visit(descriptor.value);
    }
  };

  const visitOperation = operation => {
    if (!operation || visitedOperations.has(operation)) return;
    visitedOperations.add(operation);
    const cache = operation.cache;
    visit(cache);
    const kind = String(cache?.kind || '').toLowerCase();
    // Semantic caches already capture everything their playback handler uses;
    // crossing them would warm the procedural construction DAG as well.
    if (kind === 'mesh' || kind === 'minmesh' || kind === 'material' ||
        kind === 'scene' || kind === 'effect' || kind === 'bitmap') return;
    for (const input of operation.inputs || []) visitOperation(input);
    for (const link of operation.links || []) visitOperation(link);
  };
  const events = Array.from(runtime?.events || []).sort((a, b) =>
    ((a?.start ?? 0) - (b?.start ?? 0)) || ((a?.op?.id ?? 0) - (b?.op?.id ?? 0)));
  for (const event of events) visitOperation(event?.op);
  for (const root of runtime?.roots || []) visitOperation(root);
  for (const operation of operations) visitOperation(operation);
  return {
    tasks,
    meshes: meshes.size,
    materials: materials.size,
  };
}

// The production contains three scene-to-bitmap operators. Sharing their
// renderer with playback pins every uploaded mesh and texture through the rest
// of graph precalc. A renderer is therefore owned by exactly one readback and
// disposed at the synchronous readPixels boundary.
function makeDisposableBitmapRenderer(runtime, canvas, rendererOptions = {}, onDispose = null,
    dependencies = DEFAULT_DEPENDENCIES) {
  return record => {
    let renderer = null;
    let resources = null;
    try {
      renderer = new dependencies.Renderer(canvas, rendererOptions);
      const result = makeBitmapRenderer(runtime, renderer, dependencies)(record);
      resources = renderer.resourceStats?.() || null;
      return result;
    } finally {
      renderer?.dispose?.();
      const discarded = discardPreparedGeometry(runtime);
      onDispose?.({ resources, discarded });
    }
  };
}

function makeV2Player(song, options, dependencies = DEFAULT_DEPENDENCIES) {
  if (typeof dependencies.createV2Player === 'function') {
    return dependencies.createV2Player(song, options);
  }
  if (typeof dependencies.V2Player === 'function') return new dependencies.V2Player(song, options);
  throw new Error('The plain-JavaScript V2 player has not been loaded.');
}

function nextFramePromise() {
  return new Promise(resolve => {
    if (typeof requestAnimationFrame === 'function') requestAnimationFrame(() => resolve());
    else setTimeout(resolve, 0);
  });
}

function telemetryPublishDue(lastPublishedAt, now, visible = false, force = false,
    intervalMilliseconds = 250) {
  return Boolean(force || visible || !Number.isFinite(lastPublishedAt) ||
    now - lastPublishedAt >= Math.max(1, Number(intervalMilliseconds) || 250));
}

function deferMaintenance(callback, timeoutMilliseconds = 1000) {
  let active = true;
  const invoke = deadline => {
    if (!active) return;
    active = false;
    callback(deadline);
  };
  if (typeof globalThis.requestIdleCallback === 'function') {
    const handle = globalThis.requestIdleCallback(invoke, {
      timeout: Math.max(1, Number(timeoutMilliseconds) || 1000),
    });
    return () => {
      if (!active) return;
      active = false;
      globalThis.cancelIdleCallback?.(handle);
    };
  }
  const handle = setTimeout(() => invoke(null), 0);
  return () => {
    if (!active) return;
    active = false;
    clearTimeout(handle);
  };
}

function estimateObjectBytes(value, seen = new Set()) {
  if (value === null || value === undefined) return 0;
  if (typeof value === 'number') return 8;
  if (typeof value === 'boolean') return 4;
  if (typeof value === 'string') return value.length * 2;
  if (typeof value !== 'object' || seen.has(value)) return 0;
  seen.add(value);
  if (ArrayBuffer.isView(value)) {
    const buffer = value.buffer;
    if (!buffer || seen.has(buffer)) return 0;
    seen.add(buffer);
    // A small view still retains its complete backing allocation.
    return Math.max(0, Number(buffer.byteLength) || 0);
  }
  if (value instanceof ArrayBuffer) return value.byteLength;
  if (Array.isArray(value)) {
    let bytes = 16 + value.length * 8;
    for (const item of value) bytes += estimateObjectBytes(item, seen);
    return bytes;
  }
  let bytes = 32;
  for (const [key, descriptor] of Object.entries(Object.getOwnPropertyDescriptors(value))) {
    bytes += key.length * 2 + 8;
    if (Object.prototype.hasOwnProperty.call(descriptor, 'value')) {
      bytes += estimateObjectBytes(descriptor.value, seen);
    }
  }
  return bytes;
}

function percentile(values, fraction) {
  if (!values.length) return 0;
  const index = Math.max(0, Math.min(values.length - 1, Math.ceil(values.length * fraction) - 1));
  return values[index];
}

function runtimeCacheStats(runtime, dependencies = DEFAULT_DEPENDENCIES) {
  const seen = new Set(), bitmapBuffers = new Set();
  const kinds = {}, referenceKinds = {};
  let references = 0, bitmapIdentities = 0, bitmapPixels = 0;
  let bitmapBytes = 0, releasedBitmaps = 0;
  const kindOf = cache => String(
    cache?.kind || cache?.type || cache?.outputClass || cache?.constructor?.name || 'object',
  ).toLowerCase();
  for (const operation of runtime?.operations || []) {
    const cache = operation?.cache;
    if (!cache || typeof cache !== 'object') continue;
    const kind = kindOf(cache);
    references++;
    referenceKinds[kind] = (referenceKinds[kind] || 0) + 1;
    if (seen.has(cache)) continue;
    seen.add(cache);
    kinds[kind] = (kinds[kind] || 0) + 1;
    const bitmapLike = ArrayBuffer.isView(cache.data) && cache.data.BYTES_PER_ELEMENT === 2 &&
      Number.isFinite(cache.width) && Number.isFinite(cache.height);
    if (!bitmapLike) continue;
    bitmapIdentities++;
    bitmapPixels += Math.max(0, cache.width * cache.height) || cache.data.length / 4;
    if (cache.released || !cache.data.byteLength) releasedBitmaps++;
    const buffer = cache.data.buffer;
    if (buffer && !bitmapBuffers.has(buffer)) {
      bitmapBuffers.add(buffer);
      // Count the physical allocation once, not merely this view's slice.
      bitmapBytes += buffer.byteLength;
    }
  }
  const mesh = dependencies.meshStorageStats?.(runtime) || null;
  const minmesh = dependencies.minMeshStorageStats?.(runtime) || null;
  const meshBytes = (mesh?.compactBytes || 0) + (minmesh?.compactBytes || 0);
  const preparedMeshBytes = (mesh?.preparedBytes || 0) + (minmesh?.preparedBytes || 0);
  return {
    references, identities: seen.size, kinds, referenceKinds,
    bitmapIdentities, releasedBitmaps, bitmapPixels, bitmapBytes,
    mesh, minmesh, meshBytes, preparedMeshBytes,
    pruning: runtime?.playbackCachePruning || null,
    playbackGeometry: runtime?.playbackGeometryCompaction || null,
    // preparedMeshBytes is reported separately because an uploaded geometry
    // cache references these same physical buffers. Adding it here would
    // double-count them in Renderer.resourceStats after their first draw.
    estimatedBytes: bitmapBytes + meshBytes,
  };
}

class DebrisApp {
  constructor(canvas, status = null, options = {}) {
    if (!canvas) throw new TypeError('DebrisApp needs a canvas');
    this.canvas = canvas;
    this.setStatus = typeof status === 'function' ? status : () => {};
    this.options = options;
    this.dependencies = { ...DEFAULT_DEPENDENCIES, ...(options.dependencies || {}) };
    this.sampleRate = positiveInteger(options.sampleRate, DEFAULT_RATE, 384000);
    this.renderer = null;
    this.runtime = null;
    this.document = null;
    this.player = null;
    this.loaderAudio = null;
    this.audio = null;
    this.running = false;
    this.stopped = false;
    this.paused = false;
    this.pauseRequested = false;
    this.ready = false;
    this.raf = 0;
    this.debugSample = null;
    this.currentSample = 0;
    this.duration = 0;
    this.initialSnapshot = null;
    this.snapshots = [];
    this.snapshotBytes = 0;
    const requestedSnapshotBudget = Number(options.snapshotBudgetBytes);
    this.snapshotBudgetBytes = Math.floor(Math.min(
      MAX_SNAPSHOT_BUDGET_BYTES,
      Math.max(1024 * 1024, Number.isFinite(requestedSnapshotBudget) &&
        requestedSnapshotBudget > 0
        ? requestedSnapshotBudget : DEFAULT_SNAPSHOT_BUDGET_BYTES),
    ));
    const requestedSnapshotLimit = Number(options.snapshotLimit);
    this.snapshotLimit = Math.floor(Math.min(
      MAX_SNAPSHOT_COUNT,
      Math.max(2, Number.isFinite(requestedSnapshotLimit) && requestedSnapshotLimit > 0
        ? requestedSnapshotLimit : 18),
    ));
    this.lastSnapshotSecond = 0;
    this.snapshotIntervalSeconds = Math.max(1,
      Number(options.snapshotIntervalSeconds) || 30);
    this.frameTimes = [];
    this.disposed = false;
    this.lifecycleGeneration = 0;
    this.lifecycleController = new AbortController();
    this.initializationPromise = null;
    this.startPromise = null;
    this.stopPromise = null;
    this.seekQueue = Promise.resolve();
    this.activeSeekController = null;
    this.activeStartController = null;
    this.audioClosePromises = new WeakMap();
    this.disposedRenderers = new WeakSet();
    this.scheduleMaintenance = typeof options.scheduleMaintenance === 'function'
      ? options.scheduleMaintenance : deferMaintenance;
    this.snapshotTaskCancel = null;
    this.resourceStatsTaskCancel = null;
    this.resourceStatsPublishPending = false;
    this.resourceTelemetryEnabled = Boolean(options.resourceTelemetry || options.diagnostics);
    this.resourceStatsIntervalMilliseconds = Math.max(250,
      Number(options.resourceStatsIntervalMilliseconds) || 1000);
    this.lastResourceStatsAt = -Infinity;
    this.stats = {
      frames: 0, drawCalls: 0, triangles: 0, startedAt: 0,
      precalcMilliseconds: 0, frameMilliseconds: 0,
      runtimeMilliseconds: 0, renderMilliseconds: 0,
      snapshotBytes: 0, snapshotCount: 0,
      loaderAudioBackend: 'disabled', loaderAudioQueueBytes: 0,
      loaderAudioUnderruns: 0, loaderAudioErrors: 0,
    };
    this.renderWidth = options.width || 0;
    this.renderHeight = options.height || 0;
  }

  async init() {
    if (this.ready) return this;
    const token = this.lifecycleToken();
    this.assertLifecycle(token);
    if (this.initializationPromise) return this.initializationPromise;
    const initialization = this.initializeLifecycle(token);
    this.initializationPromise = initialization;
    try {
      return await initialization;
    } finally {
      if (this.initializationPromise === initialization) this.initializationPromise = null;
    }
  }

  lifecycleToken() {
    return {
      generation: this.lifecycleGeneration,
      controller: this.lifecycleController,
      signal: this.lifecycleController.signal,
    };
  }

  lifecycleIsActive(token) {
    return Boolean(token && !this.disposed && token.generation === this.lifecycleGeneration &&
      token.controller === this.lifecycleController && !token.signal.aborted);
  }

  lifecycleError(token, error = null) {
    if (this.lifecycleIsActive(token)) return error;
    return lifecycleAbortError(token?.signal?.reason || error);
  }

  assertLifecycle(token) {
    if (!this.lifecycleIsActive(token)) throw this.lifecycleError(token);
  }

  closeAudioStream(stream) {
    if (!stream || (typeof stream !== 'object' && typeof stream !== 'function')) {
      return Promise.resolve();
    }
    const existing = this.audioClosePromises.get(stream);
    if (existing) return existing;
    let closing;
    try { closing = Promise.resolve(stream.close?.()); }
    catch (error) { closing = Promise.reject(error); }
    this.audioClosePromises.set(stream, closing);
    return closing;
  }

  disposeRenderer(renderer) {
    if (!renderer || (typeof renderer !== 'object' && typeof renderer !== 'function') ||
        this.disposedRenderers.has(renderer)) return;
    this.disposedRenderers.add(renderer);
    renderer.dispose?.();
  }

  async cleanupFailedInitialization(primaryError = null) {
    const audio = this.audio;
    const loaderAudio = this.loaderAudio;
    const renderer = this.renderer;
    const runtime = this.runtime;
    this.audio = null;
    this.loaderAudio = null;
    this.renderer = null;
    this.runtime = null;
    this.document = null;
    this.player = null;
    this.ready = false;
    if (runtime) runtime.bitmapRendererHook = null;
    try { this.disposeRenderer(renderer); }
    catch (error) {
      if (primaryError) console.warn('renderer shutdown failed', error);
      else throw error;
    }
    for (const stream of new Set([audio, loaderAudio].filter(Boolean))) {
      try { await this.closeAudioStream(stream); }
      catch (error) {
        if (primaryError) console.warn('audio shutdown failed', error);
        else throw error;
      }
    }
    this.initialSnapshot = null;
    this.snapshots.length = 0;
    this.snapshotBytes = 0;
  }

  async initializeLifecycle(token) {
    try {
      return await this.initializeOwnedResources(token);
    } catch (error) {
      const resultError = this.lifecycleError(token, error) || error;
      await this.cleanupFailedInitialization(resultError);
      throw resultError;
    }
  }

  async initializeOwnedResources(token) {
    this.assertLifecycle(token);
    const parameters = new URLSearchParams(globalThis.location?.search || '');
    const debugRequested = parameters.has('t') || this.options.debugTime !== undefined;
    const textureOffset = textureSizeOffset(
      this.options.textureQuality ?? parameters.get('tex') ?? 'high',
    );
    this.dependencies.setBitmapTextureSizeOffset(textureOffset);
    this.setStatus('loading production data…');
    this.assertLifecycle(token);
    const production = await waitForLifecycle(
      this.dependencies.loadProductionData(token.signal), token.signal,
    );
    this.assertLifecycle(token);
    this.document = this.dependencies.parseKX(production.kx);
    this.assertLifecycle(token);
    const rendererOptions = {
      pixelRatio: this.options.pixelRatio || 0,
      maxPixelRatio: this.options.maxPixelRatio || 2,
      diagnostics: this.options.diagnostics ?? parameters.has('diag'),
      dxt5Mode: this.options.dxt5Mode ?? parameters.get('dxt5') ?? 'auto',
    };
    this.renderWidth = this.renderWidth || Math.max(0, Number(parameters.get('w')) | 0);
    this.renderHeight = this.renderHeight || Math.max(0, Number(parameters.get('h')) | 0);
    if (!this.options.pixelRatio && Number(parameters.get('dpr')) > 0) {
      rendererOptions.pixelRatio = Number(parameters.get('dpr'));
    }
    const createPlaybackRenderer = () => {
      this.assertLifecycle(token);
      let renderer = null;
      try {
        renderer = new this.dependencies.Renderer(this.canvas, rendererOptions);
        renderer.resize?.(
          this.renderWidth || undefined,
          this.renderHeight || undefined,
          rendererOptions.pixelRatio,
          PRODUCTION_ASPECT,
        );
        this.assertLifecycle(token);
        return renderer;
      } catch (error) {
        try { this.disposeRenderer(renderer); }
        catch (disposeError) { console.warn('renderer shutdown failed', disposeError); }
        throw error;
      }
    };
    this.resourceTelemetryEnabled ||= Boolean(rendererOptions.diagnostics);
    this.runtime = new this.dependencies.Runtime(this.document, {
      strictHandlers: this.options.strictHandlers !== false,
      handlers: this.options.handlers ?? this.dependencies.createOperatorHandlers(),
    });
    this.assertLifecycle(token);
    this.runtime.bitmapRendererHook = makeDisposableBitmapRenderer(
      this.runtime,
      this.canvas,
      { ...rendererOptions, pixelRatio: 1, maxPixelRatio: 1, diagnostics: false },
      report => {
        this.stats.precalcBitmapRenders = (this.stats.precalcBitmapRenders || 0) + 1;
        this.stats.precalcDiscardedGeometryBytes =
          (this.stats.precalcDiscardedGeometryBytes || 0) + (report.discarded?.bytes || 0);
        this.stats.precalcTemporaryResourcePeakBytes = Math.max(
          this.stats.precalcTemporaryResourcePeakBytes || 0,
          report.resources?.totalEstimatedBytes || 0,
        );
      },
      this.dependencies,
    );

    // The loader is forward-only and owns no seek checkpoints. Its V2 player
    // normally lives in a dedicated Worker whose MessagePort feeds the
    // AudioWorklet directly, so a long procedural operator cannot starve PCM
    // production. The playback renderer may overlap this Worker only for the
    // bounded resource warm-up below; close the loader before constructing the
    // main synth so the two soundtracks never overlap. Fixed-time/debug
    // rendering deliberately remains silent.
    let loaderAudio = null;
    let initializationError = null;
    let loaderCloseError = null;
    try {
      if (!debugRequested && this.options.loaderAudio !== false && production.loaderSong?.byteLength) {
        this.assertLifecycle(token);
        this.setStatus('starting loader music…');
        this.assertLifecycle(token);
        const loaderBlockFrames = this.options.loaderAudioBlockFrames || this.options.audioBlockFrames || 2048;
        const loaderQueueBlocks = this.options.loaderAudioQueueBlocks || this.options.audioQueueBlocks || 12;
        const loaderAudioOptions = {
          sampleRate: this.sampleRate,
          blockFrames: loaderBlockFrames,
          queueBlocks: loaderQueueBlocks,
          tailSeconds: 0,
          reportClock: false,
          workerSong: production.loaderSong,
          workerPlayerOptions: {
            checkpointMemoryBytes: 0,
            checkpointIntervalSamples: 0,
          },
          onUnderrun: () => {
            this.stats.loaderAudioUnderruns = (this.stats.loaderAudioUnderruns || 0) + 1;
          },
          onError: error => {
            this.stats.loaderAudioErrors = (this.stats.loaderAudioErrors || 0) + 1;
            console.warn('loader audio Worker failed', error);
          },
        };
        if (this.options.audioWorkerUrl) loaderAudioOptions.workerUrl = this.options.audioWorkerUrl;
        loaderAudio = new this.dependencies.AudioStream(loaderAudioOptions);
        this.loaderAudio = loaderAudio;
        this.assertLifecycle(token);
        await waitForLifecycle(loaderAudio.init(), token.signal);
        this.assertLifecycle(token);
        this.stats.loaderAudioBackend = loaderAudio.producerBackend;
        this.stats.loaderAudioOutputSampleRate = loaderAudio.outputSampleRate;
        this.stats.loaderAudioQueueBytes = loaderBlockFrames * loaderQueueBlocks * 2 * 4;
        await waitForLifecycle(loaderAudio.start(), token.signal);
        this.assertLifecycle(token);
      }

      await waitForLifecycle(nextFramePromise(), token.signal);
      this.assertLifecycle(token);
      const precalcStart = performance.now();
      this.setStatus('precalculating textures and geometry…');
      this.assertLifecycle(token);
      if (typeof this.runtime.precalcAsync === 'function') {
        await waitForLifecycle(this.runtime.precalcAsync(this.runtime.currentRoot, {
          budgetMilliseconds: this.options.precalcFrameBudget ?? 12,
          signal: token.signal,
          onProgress: progress => {
            if (!this.lifecycleIsActive(token)) return;
            const percent = Math.min(99, Math.floor(progress.completed * 100 / progress.total));
            this.setStatus(`precalculating textures and geometry… ${percent}%`);
          },
        }), token.signal);
      } else this.runtime.precalc();
      this.assertLifecycle(token);
      this.stats.precalcMilliseconds = performance.now() - precalcStart;

      this.assertLifecycle(token);
      this.setStatus('pruning immutable playback caches…');
      this.assertLifecycle(token);
      this.stats.playbackCaches = pruneImmutablePlaybackCaches(this.runtime);
      // Give the browser a collection opportunity after releasing immutable
      // cache references and before materializing terminal playback geometry.
      await waitForLifecycle(nextFramePromise(), token.signal);
      this.assertLifecycle(token);
      this.setStatus('preparing immutable playback geometry…');
      this.assertLifecycle(token);
      const playbackGeometry = await waitForLifecycle(prepareStaticTerminalGeometry(this.runtime, {
        budgetMilliseconds: this.options.precalcFrameBudget ?? 12,
        signal: token.signal,
      }), token.signal);
      this.assertLifecycle(token);
      this.stats.playbackGeometry = playbackGeometry;

      const workerWarmup = !debugRequested && this.options.playbackWarmup !== false &&
        loaderAudio?.producerBackend === 'worker' &&
        typeof this.dependencies.Renderer?.prototype?.prewarmResources === 'function';
      if (workerWarmup) {
        // Disposable Bitmap_Render contexts are gone and all immutable
        // release markers are installed. Create the sole playback renderer
        // now so local GPU uploads and topology preparation remain covered by
        // the loader Worker rather than appearing at scene transitions.
        await waitForLifecycle(nextFramePromise(), token.signal);
        this.assertLifecycle(token);
        this.renderer = createPlaybackRenderer();
        const plan = collectPlaybackResourcePlan(this.runtime);
        this.assertLifecycle(token);
        const requestedResidentBytes = Number(this.options.playbackWarmupMaxBytes);
        const maxResidentBytes = Number.isFinite(requestedResidentBytes) &&
          requestedResidentBytes >= 0
          ? Math.min(DEFAULT_PLAYBACK_WARMUP_BYTES, requestedResidentBytes)
          : DEFAULT_PLAYBACK_WARMUP_BYTES;
        const requestedResourceBytes = Number(this.options.playbackWarmupMaxResourceBytes);
        const maxResourceBytes = Number.isFinite(requestedResourceBytes) &&
          requestedResourceBytes >= 0
          ? Math.min(DEFAULT_PLAYBACK_WARMUP_RESOURCE_BYTES, requestedResourceBytes)
          : DEFAULT_PLAYBACK_WARMUP_RESOURCE_BYTES;
        this.setStatus('warming playback resources… 0%');
        this.assertLifecycle(token);
        const playbackRenderer = this.renderer;
        const playbackWarmup = await waitForLifecycle(playbackRenderer.prewarmResources(plan, {
          budgetMilliseconds: Math.min(12, this.options.precalcFrameBudget ?? 8),
          maxResidentBytes,
          maxResourceBytes,
          signal: token.signal,
          shouldAbort: () => !this.lifecycleIsActive(token),
          onProgress: progress => {
            if (!this.lifecycleIsActive(token)) return;
            const percent = Math.min(99, Math.floor(
              progress.completedTasks * 100 / Math.max(1, progress.plannedTasks),
            ));
            this.setStatus(`warming playback resources… ${percent}%`);
          },
        }), token.signal);
        this.assertLifecycle(token);
        this.stats.playbackWarmup = playbackWarmup;
      } else {
        this.stats.playbackWarmup = {
          enabled: false,
          reason: debugRequested ? 'fixed-time-debug'
            : this.options.playbackWarmup === false ? 'disabled'
              : loaderAudio?.producerBackend === 'worker'
                ? 'renderer-unavailable' : 'loader-audio-not-worker',
        };
      }
    } catch (error) {
      initializationError = error;
    } finally {
      // Drop every path back to the loader Worker/synth after bounded resource
      // warming, but before constructing the main soundtrack player.
      const closingLoaderAudio = loaderAudio;
      loaderAudio = null;
      if (this.loaderAudio === closingLoaderAudio) this.loaderAudio = null;
      if (closingLoaderAudio) {
        try { await this.closeAudioStream(closingLoaderAudio); }
        catch (error) {
          // Cleanup must never replace the actionable preparation failure.
          if (initializationError) console.warn('loader audio shutdown failed', error);
          loaderCloseError = error;
        } finally {
          if ('synth' in closingLoaderAudio) closingLoaderAudio.synth = null;
        }
      }
    }
    if (initializationError || loaderCloseError) {
      try { this.disposeRenderer(this.renderer); }
      catch (error) {
        if (initializationError) console.warn('renderer shutdown failed', error);
        else throw error;
      }
      this.renderer = null;
    }
    if (initializationError) throw initializationError;
    if (loaderCloseError) throw loaderCloseError;
    this.assertLifecycle(token);
    // Playback owns a fresh cache. Nothing uploaded for a precalc readback can
    // remain strongly reachable from this renderer. Non-Worker and fixed-time
    // paths construct it here because they deliberately skip warm-up.
    this.renderer ||= createPlaybackRenderer();
    this.assertLifecycle(token);
    this.runtime.bitmapRendererHook = makeBitmapRenderer(
      this.runtime, this.renderer, this.dependencies,
    );
    this.initialSnapshot = this.runtime.snapshot();
    this.snapshots.length = 0;
    this.snapshotBytes = 0;
    this.addSnapshot(0, 0, this.initialSnapshot);

    this.assertLifecycle(token);
    this.player = makeV2Player(
      this.document.song, { sampleRate: this.sampleRate }, this.dependencies,
    );
    this.assertLifecycle(token);
    this.duration = documentDuration(this.document, this.sampleRate, this.player);
    if (debugRequested) {
      const requestedTime = Number(this.options.debugTime ?? parameters.get('t'));
      const time = Number.isFinite(requestedTime)
        ? Math.max(0, Math.min(MAX_DEBUG_SECONDS, requestedTime)) : 0;
      this.debugSample = sampleAtSeconds(time, this.sampleRate);
    } else {
      this.setStatus('starting audio…');
      this.assertLifecycle(token);
      this.audio = new this.dependencies.AudioStream({
        sampleRate: this.sampleRate,
        blockFrames: this.options.audioBlockFrames || 2048,
        queueBlocks: this.options.audioQueueBlocks || 8,
        tailSeconds: this.options.audioTailSeconds ?? 12,
        onEnded: stream => this.handleAudioEnded(stream),
      });
      this.assertLifecycle(token);
      const playbackAudio = this.audio;
      await waitForLifecycle(playbackAudio.init(this.player), token.signal);
      this.assertLifecycle(token);
      // The original production, V2, and visual runtime stay in one fixed
      // 44.1-kHz coordinate. AudioStream converts only the final PCM sink when
      // WebAudio selects a different device rate.
      this.player = playbackAudio.synth;
      this.stats.audioOutputSampleRate = playbackAudio.outputSampleRate;
      this.duration = Number.isFinite(playbackAudio.durationSamples)
        ? playbackAudio.durationSamples / this.sampleRate
        : documentDuration(this.document, this.sampleRate, this.player);
    }
    this.assertLifecycle(token);
    this.ready = true;
    this.stopped = false;
    this.setStatus('');
    this.assertLifecycle(token);
    return this;
  }

  async start() {
    if (this.startPromise) return this.startPromise;
    const starting = this.startLifecycle();
    this.startPromise = starting;
    try { return await starting; }
    finally { if (this.startPromise === starting) this.startPromise = null; }
  }

  async startLifecycle() {
    const token = this.lifecycleToken();
    let startController = null;
    try {
      await waitForLifecycle(this.init(), token.signal);
      this.assertLifecycle(token);
      if (this.stopped) {
        throw new Error('DebrisApp cannot restart after stop; construct a new app');
      }
      if (this.running) return this;
      startController = new AbortController();
      this.activeStartController = startController;
      const assertStartActive = () => {
        this.assertLifecycle(token);
        if (startController.signal.aborted || !this.running) {
          throw lifecycleAbortError(startController.signal.reason || 'start was cancelled');
        }
      };
      this.running = true; this.paused = false; this.pauseRequested = false;
      this.stats.startedAt = performance.now();
      const audio = this.audio;
      if (audio) {
        await waitForLifecycle(audio.start(), startController.signal);
        assertStartActive();
      }
      this.options.onStart?.(this);
      assertStartActive();
      if (this.debugSample !== null) {
        await this.seekRuntime(this.debugSample, {
          yield: false,
          lifecycleToken: token,
          signal: startController.signal,
        });
        assertStartActive();
        this.renderSample(this.debugSample);
        this.running = false;
        // Fixed-time browser validation needs complete resource telemetry. Keep
        // the first-frame callback immediate, then perform this explicitly
        // requested scan outside renderSample's measured frame and republish.
        this.cancelResourceStatsTask();
        this.refreshResourceStats({ publish: true, forceTelemetry: true });
      } else {
        this.raf = requestAnimationFrame(time => this.frame(time));
      }
      return this;
    } catch (error) {
      this.running = false;
      throw this.lifecycleError(token, error) || error;
    } finally {
      if (this.activeStartController === startController) this.activeStartController = null;
    }
  }

  frame() {
    if (!this.running) return;
    this.raf = requestAnimationFrame(time => this.frame(time));
    if (this.paused) return;
    const sample = this.audio?.sample ?? this.currentSample;
    const audioFinished = this.audio
      ? this.audio.drained || (this.audio.ended && sample >= this.audio.durationSamples)
      : sample / this.sampleRate >= this.duration;
    if (audioFinished) {
      Promise.resolve(this.stop(true)).catch(error => {
        if (!isAbortError(error)) {
          this.setStatus(error.message);
          console.error(error);
        }
      });
      return;
    }
    try {
      this.renderSample(sample);
      const second = sample / this.sampleRate;
      if (second - this.lastSnapshotSecond >= this.snapshotIntervalSeconds) {
        this.requestRuntimeSnapshot();
      }
    } catch (error) {
      this.setStatus(error.message);
      console.error(error);
      Promise.resolve(this.stop(false)).catch(stopError => {
        if (!isAbortError(stopError)) console.error(stopError);
      });
    }
  }

  handleAudioEnded(stream) {
    if (stream !== this.audio) return;
    const endSample = Number(stream?.endSample);
    if (Number.isFinite(endSample) && endSample >= 0) this.duration = endSample / this.sampleRate;
    if (this.running) Promise.resolve(this.stop(true)).catch(error => {
      if (!isAbortError(error)) {
        this.setStatus(error.message);
        console.error(error);
      }
    });
  }

  renderSample(sample) {
    const frameStarted = performance.now();
    sample = normalizedSample(sample, this.currentSample);
    this.currentSample = sample;
    const environment = this.runtime.environment;
    environment.aspect = PRODUCTION_ASPECT;
    this.runtime.frameAtSample(sample, this.sampleRate);
    const runtimeEnded = performance.now();
    const result = this.renderer.render(environment.lastOutput, environment, {
      width: this.renderWidth || undefined,
      height: this.renderHeight || undefined,
      pixelRatio: this.options.pixelRatio || 0,
      presentationAspect: PRODUCTION_ASPECT,
    });
    const frameEnded = performance.now();
    this.stats.frames++;
    this.stats.drawCalls = result.drawCalls;
    this.stats.triangles = result.triangles;
    this.stats.culling = result.culling || {};
    this.stats.sample = sample;
    this.stats.seconds = sample / this.sampleRate;
    this.stats.runtimeMilliseconds = runtimeEnded - frameStarted;
    this.stats.renderMilliseconds = frameEnded - runtimeEnded;
    this.stats.frameMilliseconds = frameEnded - frameStarted;
    this.frameTimes.push(this.stats.frameMilliseconds);
    if (this.frameTimes.length > 240) this.frameTimes.shift();
    const heap = globalThis.performance?.memory;
    this.stats.jsHeapUsedBytes = heap?.usedJSHeapSize || 0;
    this.stats.jsHeapTotalBytes = heap?.totalJSHeapSize || 0;
    this.options.onFrame?.(this.stats, this, { frame: true });
    if (this.resourceTelemetryEnabled) this.requestResourceStats({ publish: true });
    return result;
  }

  redraw() {
    if (!this.ready || !this.renderer || !this.runtime) return false;
    this.renderSample(this.currentSample);
    return true;
  }

  cancelResourceStatsTask() {
    this.resourceStatsTaskCancel?.();
    this.resourceStatsTaskCancel = null;
    this.resourceStatsPublishPending = false;
  }

  cancelRuntimeSnapshotTask() {
    this.snapshotTaskCancel?.();
    this.snapshotTaskCancel = null;
  }

  setResourceTelemetry(enabled) {
    this.resourceTelemetryEnabled = Boolean(enabled);
    if (this.resourceTelemetryEnabled) {
      this.requestResourceStats({ publish: true, force: true });
    } else this.cancelResourceStatsTask();
    return this.resourceTelemetryEnabled;
  }

  refreshResourceStats(options = {}) {
    if (this.disposed || !this.runtime) return null;
    const resources = this.resourceStats();
    this.stats.resources = resources;
    this.lastResourceStatsAt = performance.now();
    if (options.publish) {
      this.options.onFrame?.(this.stats, this, {
        resources: true,
        forceTelemetry: Boolean(options.forceTelemetry),
      });
    }
    return resources;
  }

  requestResourceStats(options = {}) {
    if (this.disposed || !this.runtime || !this.renderer) return false;
    const publish = options.publish !== false;
    const now = performance.now();
    const due = options.force || !this.stats.resources ||
      now - this.lastResourceStatsAt >= this.resourceStatsIntervalMilliseconds;
    if (!due) return false;
    this.resourceStatsPublishPending ||= publish;
    if (this.resourceStatsTaskCancel) return false;
    let completedSynchronously = false;
    const run = () => {
      completedSynchronously = true;
      this.resourceStatsTaskCancel = null;
      const shouldPublish = this.resourceStatsPublishPending;
      this.resourceStatsPublishPending = false;
      if (this.disposed || !this.runtime || !this.renderer) return;
      this.refreshResourceStats({ publish: shouldPublish });
    };
    const cancel = this.scheduleMaintenance(run, 1000);
    if (!completedSynchronously) {
      this.resourceStatsTaskCancel = typeof cancel === 'function' ? cancel : () => {};
    }
    return true;
  }

  requestRuntimeSnapshot() {
    if (this.disposed || !this.ready || !this.running || !this.runtime ||
        this.snapshotTaskCancel) return false;
    const requestedSecond = this.currentSample / this.sampleRate;
    if (requestedSecond - this.lastSnapshotSecond < this.snapshotIntervalSeconds) return false;
    let completedSynchronously = false;
    const run = () => {
      completedSynchronously = true;
      this.snapshotTaskCancel = null;
      if (this.disposed || !this.ready || !this.running || !this.runtime) return;
      // The deferred callback snapshots whichever fully rendered sample is
      // current when idle time is actually granted. This keeps sample/state
      // coordinates exact even if another RAF ran before the callback.
      const sample = this.currentSample;
      const second = sample / this.sampleRate;
      if (second - this.lastSnapshotSecond < this.snapshotIntervalSeconds) return;
      const state = this.runtime.snapshot();
      this.lastSnapshotSecond = second;
      this.addSnapshot(second, sample, state);
    };
    const cancel = this.scheduleMaintenance(run, 2000);
    if (!completedSynchronously) {
      this.snapshotTaskCancel = typeof cancel === 'function' ? cancel : () => {};
    }
    return true;
  }

  addSnapshot(second, sample, state) {
    const bytes = estimateObjectBytes(state);
    this.snapshots.push({ second, sample, state, bytes });
    this.snapshotBytes += bytes;
    while (this.snapshots.length > 2 &&
      (this.snapshots.length > this.snapshotLimit || this.snapshotBytes > this.snapshotBudgetBytes)) {
      const [removed] = this.snapshots.splice(1, 1);
      this.snapshotBytes -= removed.bytes || 0;
    }
    this.stats.snapshotBytes = this.snapshotBytes;
    this.stats.snapshotCount = this.snapshots.length;
    return this.snapshots[this.snapshots.length - 1];
  }

  performanceStats() {
    const sorted = this.frameTimes.slice().sort((a, b) => a - b);
    return {
      ...this.stats,
      frameP50Milliseconds: percentile(sorted, 0.50),
      frameP95Milliseconds: percentile(sorted, 0.95),
      frameP99Milliseconds: percentile(sorted, 0.99),
      frameMaximumMilliseconds: sorted[sorted.length - 1] || 0,
      retainedHandlerTrace: this.runtime?.handlerCalls?.length || 0,
      totalHandlerCalls: this.runtime?.handlerCallCount || 0,
      resources: this.resourceStats() || this.stats.resources || null,
    };
  }

  resourceStats() {
    const renderer = this.renderer?.resourceStats?.() || null;
    const runtime = runtimeCacheStats(this.runtime, this.dependencies);
    if (!renderer) return { runtime, totalEstimatedBytes: runtime.estimatedBytes };
    return {
      ...renderer,
      runtime,
      totalEstimatedBytes: (renderer.totalEstimatedBytes || 0) + runtime.estimatedBytes,
    };
  }

  snapshotBefore(sample) {
    let best = this.snapshots[0];
    for (const snapshot of this.snapshots) if (snapshot.sample <= sample && snapshot.sample >= best.sample) best = snapshot;
    return best;
  }

  async seekRuntime(targetSample, options = {}) {
    const token = options.lifecycleToken || this.lifecycleToken();
    const signal = options.signal || token.signal;
    const assertSeekActive = () => {
      this.assertLifecycle(token);
      if (signal?.aborted) throw lifecycleAbortError(signal.reason);
    };
    assertSeekActive();
    // A deferred checkpoint must never observe a partially replayed seek.
    this.cancelRuntimeSnapshotTask();
    const seekStarted = performance.now();
    targetSample = normalizedSample(targetSample, this.currentSample);
    if (this.duration > 0) {
      targetSample = Math.min(targetSample, sampleAtSeconds(this.duration, this.sampleRate));
    }
    let replayStartSample = this.currentSample;
    // The live Runtime already represents currentSample. Forward seeks can
    // continue from it directly; only a backward seek (or a diagnostic exact
    // reset) needs a checkpoint restore. This keeps repeated +5 second seeks
    // linear instead of replaying the entire gap from the same older snapshot.
    if (options.forceRestore || targetSample < replayStartSample) {
      const snapshot = this.snapshotBefore(targetSample);
      if (!snapshot) throw new Error('runtime seeking requires an initial snapshot');
      this.runtime.restore(snapshot.state);
      replayStartSample = snapshot.sample;
    }
    this.runtime.environment.aspect = PRODUCTION_ASPECT;
    let iterations = 0;
    // Stateful effects such as ChainLine interpolate moving endpoints once per
    // visual frame, so replay at the reference capture's deterministic 30 fps
    // cadence. The sequence also yields a non-frame-aligned target remainder.
    for (const sample of replaySampleSequence(replayStartSample, targetSample, this.sampleRate)) {
      assertSeekActive();
      this.runtime.frameAtSample(sample, this.sampleRate);
      this.dependencies.advanceEffectFrame?.(this.runtime.environment);
      if (options.yield !== false && (++iterations % 120) === 0) {
        await waitForLifecycle(nextFramePromise(), signal);
        assertSeekActive();
      }
    }
    assertSeekActive();
    this.currentSample = targetSample;
    this.stats.seekMilliseconds = performance.now() - seekStarted;
    return targetSample;
  }

  seek(deltaSeconds) {
    // Audio and runtime seeking both mutate one forward state machine. Queue
    // repeated arrow-key requests so two asynchronous replays can never
    // restore/advance the same Runtime concurrently.
    const run = () => this.seekOwned(deltaSeconds);
    const request = this.seekQueue.then(run, run);
    this.seekQueue = request.then(() => undefined, () => undefined);
    return request;
  }

  async seekOwned(deltaSeconds) {
    const token = this.lifecycleToken();
    this.assertLifecycle(token);
    if (!this.ready || this.debugSample !== null || !this.audio) {
      return this.currentSample / this.sampleRate;
    }
    const seekController = new AbortController();
    this.activeSeekController = seekController;
    const durationSamples = Number.isFinite(this.audio?.durationSamples)
      ? normalizedSample(this.audio.durationSamples)
      : sampleAtSeconds(this.duration, this.sampleRate);
    const numericDelta = Number(deltaSeconds);
    const rawDeltaSamples = Number.isFinite(numericDelta)
      ? numericDelta * this.sampleRate : 0;
    const deltaSamples = Number.isFinite(rawDeltaSamples) ? Math.trunc(rawDeltaSamples) : 0;
    const targetSample = Math.max(0, Math.min(durationSamples, this.currentSample + deltaSamples));
    const targetSeconds = targetSample / this.sampleRate;
    this.pauseRequested = this.paused;
    this.paused = true;
    this.setStatus(`seeking ${Math.floor(targetSeconds / 60)}:${String(Math.floor(targetSeconds % 60)).padStart(2, '0')}…`);
    const audio = this.audio;
    try {
      await Promise.all([
        waitForLifecycle(audio.seek(targetSeconds), seekController.signal),
        this.seekRuntime(targetSample, {
          lifecycleToken: token,
          signal: seekController.signal,
        }),
      ]);
      this.assertLifecycle(token);
      this.setStatus('');
      this.paused = this.pauseRequested;
      audio.pause(this.paused);
      return targetSeconds;
    } catch (error) {
      // Promise.all does not cancel its sibling. Abort whichever side is still
      // pending so an audio failure cannot leave a background Runtime replay,
      // and a Runtime failure cannot leave a late audio seek publication.
      if (!seekController.signal.aborted) seekController.abort(error);
      const lifecycleActive = this.lifecycleIsActive(token);
      if (lifecycleActive) {
        this.setStatus('');
        this.paused = this.pauseRequested;
        if (this.audio === audio) audio.pause(this.paused);
      }
      // A non-cancellation failure may have advanced one of the two state
      // machines partially. Stop playback instead of resuming with divergent
      // audio/visual cursors; the normal onEnd owner will dispose the app.
      if (lifecycleActive && !isAbortError(error)) {
        Promise.resolve(this.stop(false)).catch(stopError => console.error(stopError));
      }
      throw this.lifecycleError(token, error) || error;
    } finally {
      if (this.activeSeekController === seekController) this.activeSeekController = null;
    }
  }

  pause(value) {
    const seeking = Boolean(this.activeSeekController);
    const current = seeking ? this.pauseRequested : this.paused;
    this.pauseRequested = value === undefined ? !current : Boolean(value);
    if (!seeking) {
      this.paused = this.pauseRequested;
      this.audio?.pause(this.paused);
    }
    return this.pauseRequested;
  }

  async stop(ended = false) {
    if (this.stopPromise) return this.stopPromise;
    if (this.stopped) return;
    if (!this.running && !this.ready) {
      // stop() is also the public cancellation path for an in-flight init.
      if (this.initializationPromise) this.dispose();
      return;
    }
    const token = this.lifecycleToken();
    const stopping = (async () => {
      this.activeStartController?.abort(lifecycleAbortError('start cancelled by stop'));
      this.activeSeekController?.abort(lifecycleAbortError('seek cancelled by stop'));
      this.running = false;
      this.stopped = true;
      cancelAnimationFrame(this.raf);
      this.raf = 0;
      this.cancelRuntimeSnapshotTask();
      this.cancelResourceStatsTask();
      const audio = this.audio;
      this.audio = null;
      if (audio) {
        try { await waitForLifecycle(this.closeAudioStream(audio), token.signal); }
        catch (error) {
          // Disposal invalidates the wait deliberately. A genuine device-close
          // failure is non-fatal too: returning to the launcher must not depend
          // on a broken AudioContext, and closeAudioStream consumes late errors.
          if (this.lifecycleIsActive(token)) console.warn('audio shutdown failed', error);
        }
      }
      if (this.disposed) return;
      this.options.onEnd?.(ended, this);
    })();
    this.stopPromise = stopping;
    try { return await stopping; }
    finally { if (this.stopPromise === stopping) this.stopPromise = null; }
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    this.lifecycleGeneration++;
    if (!this.lifecycleController.signal.aborted) {
      this.lifecycleController.abort(lifecycleAbortError());
    }
    this.activeStartController?.abort(lifecycleAbortError());
    this.activeStartController = null;
    this.activeSeekController?.abort(lifecycleAbortError());
    this.activeSeekController = null;
    this.running = false;
    this.stopped = true;
    this.ready = false;
    cancelAnimationFrame(this.raf);
    this.raf = 0;
    this.cancelRuntimeSnapshotTask();
    this.cancelResourceStatsTask();
    const audio = this.audio;
    const loaderAudio = this.loaderAudio;
    const renderer = this.renderer;
    const runtime = this.runtime;
    this.audio = null;
    this.loaderAudio = null;
    this.renderer = this.runtime = this.document = this.player = null;
    if (runtime) runtime.bitmapRendererHook = null;
    for (const stream of new Set([audio, loaderAudio].filter(Boolean))) {
      this.closeAudioStream(stream).catch(error => console.warn('audio shutdown failed', error));
    }
    try { this.disposeRenderer(renderer); }
    catch (error) { console.warn('renderer shutdown failed', error); }
    this.initialSnapshot = null;
    this.snapshots.length = 0;
    this.snapshotBytes = 0;
    this.frameTimes.length = 0;
  }
}

async function start(canvas, status, options = {}) {
  const app = new DebrisApp(canvas, status, options);
  await app.start();
  return app;
}

export {
  DEFAULT_DEPENDENCIES,
  DebrisApp,
  decodeRenderPixels,
  documentDuration,
  estimateObjectBytes,
  collectPlaybackResourcePlan,
  discardPreparedGeometry,
  terminalStaticGeometry,
  prepareStaticTerminalGeometry,
  pruneImmutablePlaybackCaches,
  replaySampleSequence,
  runtimeCacheStats,
  makeBitmapRenderer,
  makeDisposableBitmapRenderer,
  sampleAtSeconds,
  telemetryPublishDue,
  textureSizeOffset,
  start,
};
