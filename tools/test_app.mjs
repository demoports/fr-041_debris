import assert from 'node:assert/strict';
import fs from 'node:fs';
import * as appModule from '../src/app.js';
import { KC_MESH, KC_MINMESH, KC_SCENE } from '../src/runtime.js';
import {
  RGBA16_BYTES_PER_PIXEL,
  productionTextureQualityInventory,
} from './texture_quality_inventory.mjs';

const originalRequestAnimationFrame = globalThis.requestAnimationFrame;
const originalCancelAnimationFrame = globalThis.cancelAnimationFrame;
globalThis.requestAnimationFrame = callback => { callback(0); return 1; };
globalThis.cancelAnimationFrame = () => {};
const D = { ...appModule, KC_MESH, KC_MINMESH, KC_SCENE };

assert.equal(D.sampleAtSeconds(1.5, 44100), 66150);
assert.equal(D.sampleAtSeconds(-2, 44100), 0);
assert.equal(D.sampleAtSeconds(Number.NaN, 44100), 0);
assert.equal(D.sampleAtSeconds(Number.POSITIVE_INFINITY, 44100), 0,
  'a malformed fixed-time value cannot create an unbounded replay');
assert.equal(D.sampleAtSeconds(1, Number.POSITIVE_INFINITY), 44100,
  'an invalid sample rate falls back to the production rate');
assert.deepEqual(Array.from(D.replaySampleSequence(75, 245, 1001)),
  [100, 133, 166, 200, 233, 245],
  'replay uses global rational 30 fps boundaries and includes the target remainder');
assert.deepEqual(Array.from(D.replaySampleSequence(1470, 3000, 44100)), [2940, 3000]);
assert.deepEqual(Array.from(D.replaySampleSequence(3000, 3000, 44100)), []);
assert.deepEqual(Array.from(D.replaySampleSequence(75, Number.POSITIVE_INFINITY, 1001)), [],
  'an invalid replay target is a bounded no-op');
assert.equal(D.textureSizeOffset('normal'), -1);
assert.equal(D.textureSizeOffset('reduced'), -1);
assert.equal(D.textureSizeOffset('medium'), -1);
assert.equal(D.textureSizeOffset('low'), -2);
assert.equal(D.textureSizeOffset('high'), 0);
assert.equal(D.textureSizeOffset('ultra'), 0);
assert.equal(D.textureSizeOffset('full'), 0);
assert.equal(D.textureSizeOffset(undefined), 0);
assert.equal(D.textureSizeOffset(-3), -3);
assert.equal(D.textureSizeOffset(-20), -3);
assert.equal(D.telemetryPublishDue(-Infinity, 10, false, false, 250), true,
  'the first rendered frame publishes immediately');
assert.equal(D.telemetryPublishDue(10, 259, false, false, 250), false);
assert.equal(D.telemetryPublishDue(10, 260, false, false, 250), true,
  'hidden telemetry is capped at four publications per second');
assert.equal(D.telemetryPublishDue(10, 11, true, false, 250), true,
  'a visible stats overlay may update every frame');
assert.equal(D.telemetryPublishDue(10, 11, false, true, 250), true,
  'explicit fixed-frame telemetry can force a publication');

// Preparation normally yields to a task, reserving the much more expensive
// animation-frame handoff for the loader's visible 450ms paint cadence.
{
  let clock = 0;
  let visible = true;
  let taskYields = 0;
  let frameYields = 0;
  let timerId = 0;
  const scheduler = D.createPreparationScheduler({
    now: () => clock,
    isVisible: () => visible,
    taskYield: async () => { taskYields++; clock += 2; },
    requestFrame(callback) { frameYields++; clock += 8; callback(clock); return frameYields; },
    cancelFrame() {},
    setTimer: () => ++timerId,
    clearTimer() {},
    paintIntervalMilliseconds: 450,
  });
  await scheduler.yield('graph');
  assert.deepEqual({ taskYields, frameYields }, { taskYields: 1, frameYields: 0 });
  clock = 451;
  await scheduler.yield('graph');
  assert.deepEqual({ taskYields, frameYields }, { taskYields: 2, frameYields: 1 },
    'a paint handoff crosses both rAF and a posted-task boundary');
  clock = 912;
  visible = false;
  await scheduler.yield('geometry');
  assert.equal(frameYields, 1, 'a hidden loader never waits for throttled animation frames');
  visible = true;
  await scheduler.yield('geometry');
  assert.equal(frameYields, 2, 'an overdue loader paints immediately after becoming visible');
  const snapshot = scheduler.snapshot();
  assert.deepEqual({
    backend: snapshot.backend,
    totalYields: snapshot.totalYields,
    taskYields: snapshot.taskYields,
    paintAttempts: snapshot.paintAttempts,
    paintYields: snapshot.paintYields,
    paintTimeouts: snapshot.paintTimeouts,
    graphYields: snapshot.phases.graph.totalYields,
    geometryYields: snapshot.phases.geometry.totalYields,
    totalYieldMilliseconds: snapshot.totalYieldMilliseconds,
  }, {
    backend: 'injected', totalYields: 4, taskYields: 2,
    paintAttempts: 2, paintYields: 2, paintTimeouts: 0,
    graphYields: 2, geometryYields: 2, totalYieldMilliseconds: 24,
  });
  scheduler.dispose();
  scheduler.dispose();
  await scheduler.yield('late');
  assert.deepEqual(scheduler.snapshot(), snapshot,
    'scheduler snapshots are detached and disposal is idempotent');
}

// A throttled/occluded rAF cannot strand initialization indefinitely.
{
  let clock = 0;
  let timeout = null;
  let cancelledFrame = null;
  const scheduler = D.createPreparationScheduler({
    now: () => clock,
    taskYield: async () => {},
    requestFrame: () => 77,
    cancelFrame: handle => { cancelledFrame = handle; },
    setTimer(callback) { timeout = callback; return 12; },
    clearTimer() {},
    paintIntervalMilliseconds: 10,
    paintTimeoutMilliseconds: 100,
  });
  clock = 11;
  const yielding = scheduler.yield('warmup');
  clock = 111;
  timeout();
  await yielding;
  const snapshot = scheduler.snapshot();
  assert.equal(cancelledFrame, 77);
  assert.equal(snapshot.paintAttempts, 1);
  assert.equal(snapshot.paintTimeouts, 1);
  assert.equal(snapshot.paintYieldMilliseconds, 100);
  scheduler.dispose();
}

{
  let platformYields = 0;
  const schedulerApi = { async yield() { platformYields++; } };
  const scheduler = D.createPreparationScheduler({
    schedulerApi, MessageChannelCtor: null, requestFrame: null,
  });
  await scheduler.yield('graph');
  assert.equal(platformYields, 1);
  assert.equal(scheduler.snapshot().backend, 'scheduler');
  scheduler.dispose();
}

if (typeof globalThis.MessageChannel === 'function') {
  const scheduler = D.createPreparationScheduler({
    schedulerApi: null, requestFrame: null,
  });
  await scheduler.yield('graph');
  assert.equal(scheduler.snapshot().backend, 'message-channel');
  scheduler.dispose();
}

{
  const scheduler = D.createPreparationScheduler({
    schedulerApi: null, MessageChannelCtor: null, requestFrame: null,
  });
  await scheduler.yield('graph');
  assert.equal(scheduler.snapshot().backend, 'timeout');
  scheduler.dispose();
}

const indexSource = fs.readFileSync(new URL('../index.html', import.meta.url), 'utf8');
const appSource = fs.readFileSync(new URL('../src/app.js', import.meta.url), 'utf8');
assert.match(indexSource, /telemetryPublishDue\(/);
assert.match(indexSource, /STATS_PUBLISH_INTERVAL_MS\s*=\s*250/);
assert.match(indexSource, /<script type="module">\s*import \{ DebrisApp, telemetryPublishDue \}/);
assert.doesNotMatch(indexSource, /<script src=/,
  'the launcher must enter through the ESM graph rather than ordered classic scripts');
assert.doesNotMatch(indexSource, /texture-quality|TEXTURE_QUALITY_KEY|textureQuality/,
  'the launcher has no texture-quality selector or saved quality preference');
assert.match(indexSource, /starting = true; button\.disabled = true; setRunning\(true\)/,
  'starting the faithful full-resolution precalc disables the launcher button');
assert.match(indexSource, /id="loader-frame"/);
assert.match(indexSource, /left: 20px; right: 20px; top: calc\(50% - 20px\);\s*height: 40px; padding: 3px/,
  'the browser loader retains the native 20px margin, 40px frame, and 3px nested insets');
assert.match(indexSource, /#backdrop\s*\{[^}]*launcher-background\.jpg[^}]*center\s*\/\s*cover/s,
  'the launcher uses the compact frame from the original production as a cover background');
assert.match(indexSource, /linear-gradient\(rgba\(0,0,0,\.04\), rgba\(0,0,0,\.04\)\)/,
  'the pretreated launcher frame uses only a faint uniform readability wash');
assert.doesNotMatch(indexSource, /#backdrop\s*\{[^}]*(?:filter:|radial-gradient)/s,
  'the launcher treatment must not recolor, blur, or vignette the source frame');
assert.match(indexSource, /h1\s*\{[^}]*color:\s*#fff;/s,
  'the selected launcher frame uses a high-contrast light title palette');
assert.match(indexSource,
  /<a href="https:\/\/github\.com\/demoports\/fr-041_debris" target="_blank" rel="noopener noreferrer">github<\/a>/,
  'the launcher links to its GitHub repository without exposing the opener');
assert.match(indexSource, /#loader\s*\{[^}]*background:\s*#000;/s,
  'the native-style loader retains its plain black background');
assert.match(indexSource, /loaderPaintedAt \+ 450/,
  'the progress display keeps the released player\'s roughly 450ms presentation cadence');
assert.match(indexSource, /debrisPrecalcYieldBackend/,
  'production telemetry exposes the cooperative preparation backend');
assert.match(appModule.createPreparationScheduler.toString(),
  /schedulerApi\.yield[\s\S]*MessageChannelCtor[\s\S]*setTimer/,
  'preparation prefers Scheduler, then MessageChannel, then timer tasks');
assert.match(appSource,
  /yield: \(\) => preparationScheduler\.yield\('graph'\)[\s\S]*yield: \(\) => preparationScheduler\.yield\('geometry'\)[\s\S]*yield: \(\) => preparationScheduler\.yield\('warmup'\)/,
  'one app-owned scheduler covers graph, geometry, and resource preparation');
assert.match(indexSource, /\^precalculating textures and geometry…\\s\*\(\\d\+\)%\$/,
  'the native-style bar is driven by real graph-precalc completion');
assert.match(indexSource, /error\?\.name === 'AbortError'/,
  'Escape during asynchronous startup returns quietly to the launcher');
assert.match(indexSource, /wakeLockGeneration\+\+/,
  'a pending Wake Lock request is invalidated when the app is closed');
assert.match(indexSource, /if \(_sourceApp !== app\) return;/,
  'telemetry from a disposed launcher instance cannot overwrite its successor');
assert.match(indexSource, /fullscreen unavailable/,
  'rejected fullscreen requests are handled inside the async event listener');
assert.match(indexSource,
  /event\.key === 'ArrowLeft'[\s\S]*?event\.preventDefault\(\);\s*if \(event\.repeat\) return;/,
  'held arrow keys cannot enqueue repeated relative seeks');
const qualityInventory = await productionTextureQualityInventory();
assert.deepEqual(Object.fromEntries(Object.entries(qualityInventory).map(([quality, result]) => [
  quality,
  {
    sourceSizeOffset: result.sourceSizeOffset,
    bitmapOperators: result.bitmapOperators,
    rgba16Pixels: result.rgba16Pixels,
    rgba16OutputBytes: result.rgba16OutputBytes,
    roundedOutputMiB: result.roundedOutputMiB,
    retainedBitmapBoundaryOperations: result.retainedBitmapBoundaryOperations,
    retainedBitmapIdentities: result.retainedBitmapIdentities,
    retainedRgba16Pixels: result.retainedRgba16Pixels,
    retainedRgba16Bytes: result.retainedRgba16Bytes,
    roundedRetainedMiB: result.roundedRetainedMiB,
    inPlaceReuseCount: result.inPlaceReuseCount,
    inPlaceCopyBytesAvoided: result.inPlaceCopyBytesAvoided,
    roundedInPlaceCopyMiB: result.roundedInPlaceCopyMiB,
  },
])), {
  high: {
    sourceSizeOffset: 0,
    bitmapOperators: 2941,
    rgba16Pixels: 706916608,
    rgba16OutputBytes: 5655332864,
    roundedOutputMiB: 5393,
    retainedBitmapBoundaryOperations: 262,
    retainedBitmapIdentities: 260,
    retainedRgba16Pixels: 53117184,
    retainedRgba16Bytes: 424937472,
    roundedRetainedMiB: 405,
    inPlaceReuseCount: 1914,
    inPlaceCopyBytesAvoided: 3704356864,
    roundedInPlaceCopyMiB: 3533,
  },
  medium: {
    sourceSizeOffset: -1,
    bitmapOperators: 2941,
    rgba16Pixels: 180465664,
    rgba16OutputBytes: 1443725312,
    roundedOutputMiB: 1377,
    retainedBitmapBoundaryOperations: 262,
    retainedBitmapIdentities: 260,
    retainedRgba16Pixels: 13673472,
    retainedRgba16Bytes: 109387776,
    roundedRetainedMiB: 104,
    inPlaceReuseCount: 1914,
    inPlaceCopyBytesAvoided: 949682176,
    roundedInPlaceCopyMiB: 906,
  },
  low: {
    sourceSizeOffset: -2,
    bitmapOperators: 2941,
    rgba16Pixels: 48852928,
    rgba16OutputBytes: 390823424,
    roundedOutputMiB: 373,
    retainedBitmapBoundaryOperations: 262,
    retainedBitmapIdentities: 260,
    retainedRgba16Pixels: 3812544,
    retainedRgba16Bytes: 30500352,
    roundedRetainedMiB: 29,
    inPlaceReuseCount: 1914,
    inPlaceCopyBytesAvoided: 261013504,
    roundedInPlaceCopyMiB: 249,
  },
}, 'launcher estimates mirror retained playback boundaries and physical bitmap identities');
for (const result of Object.values(qualityInventory)) {
  assert.equal(result.rgba16OutputBytes, result.rgba16Pixels * RGBA16_BYTES_PER_PIXEL,
    'the static estimate uses the port\'s four 16-bit working channels');
  assert.equal(result.retainedRgba16Bytes,
    result.retainedRgba16Pixels * RGBA16_BYTES_PER_PIXEL,
    'retained playback texture data uses the same RGBA16 representation');
  assert.equal(Object.values(result.inPlaceBytesByClass)
    .reduce((sum, bytes) => sum + bytes, 0), result.inPlaceCopyBytesAvoided,
  'per-class runtime copy savings sum to the reported total');
}
assert.equal(qualityInventory.high.inPlaceBytesByClass['0x36'], 103 * 1024 * 1024,
  'RotateMul accounting includes only its source-faithful initial Color copy');
assert.equal(qualityInventory.medium.dimensionCounts['512x512'], 19,
  'authored no-scale bitmap chains remain full size at medium quality');
assert.equal(qualityInventory.low.dimensionCounts['512x512'], 19,
  'authored no-scale bitmap chains remain full size at low quality');
const browserSmokeSource = fs.readFileSync(new URL('./browser_smoke.mjs', import.meta.url), 'utf8');
assert.match(browserSmokeSource, /\['high', 'medium', 'low', 'normal', 'reduced', 'ultra', 'full'\]/,
  'the guarded runner accepts all three launcher names while retaining legacy aliases');
assert.match(browserSmokeSource, /must be high, medium, or low/);
assert.match(browserSmokeSource, /\['auto', 's3tc', 'rgba8'\]/,
  'the guarded runner exposes all DXT5 extension/fallback diagnostics');
assert.match(browserSmokeSource, /DEBRIS_SMOKE_DXT5/);
assert.match(browserSmokeSource,
  /\(\?:google chrome\|chromium\|chrome-headless-shell\)\/i/,
  'the guarded runner detects packaged and Playwright headless Chromium trees');
assert.match(browserSmokeSource,
  /seekRuntime\(sample, \{ yield: false, forceRestore: true \}\)/,
  'profile repetitions explicitly restore their retained baseline state');
assert.equal(D.documentDuration({ songLength: 65536 * 196, songBPM: 196 }, 44100), 60);
assert.equal(D.documentDuration({ songLength: 1, songBPM: 1 }, 10, { calcSongSamples: () => 125 }), 12.5);
assert.equal(D.documentDuration({ songLength: 1, songBPM: 0 }), 0);
assert.ok(D.estimateObjectBytes({ values: new Float32Array(16), name: 'snapshot' }) >= 64);
const snapshotBacking = new ArrayBuffer(256);
assert.ok(D.estimateObjectBytes({
  first: new Uint8Array(snapshotBacking, 0, 8),
  second: new Uint8Array(snapshotBacking, 128, 8),
}) >= 256 && D.estimateObjectBytes({
  first: new Uint8Array(snapshotBacking, 0, 8),
  second: new Uint8Array(snapshotBacking, 128, 8),
}) < 512,
'snapshot accounting counts a retained backing allocation once rather than its slices');
const sharedBitmapData = new Uint16Array(32);
const cacheStats = D.runtimeCacheStats({ operations: [
  { cache: { kind: 'bitmap', width: 2, height: 4, data: sharedBitmapData } },
  { cache: { kind: 'bitmap', width: 2, height: 4, data: sharedBitmapData } },
  { cache: { kind: 'scene' } },
] });
assert.equal(cacheStats.references, 3);
assert.equal(cacheStats.identities, 3);
assert.equal(cacheStats.bitmapIdentities, 2);
assert.equal(cacheStats.bitmapBytes, sharedBitmapData.byteLength);
assert.equal(cacheStats.estimatedBytes, sharedBitmapData.byteLength);
const largerBitmapAllocation = new ArrayBuffer(256);
const slicedBitmapStats = D.runtimeCacheStats({ operations: [{
  cache: {
    kind: 'bitmap', width: 2, height: 4,
    data: new Uint16Array(largerBitmapAllocation, 64, 32),
  },
}] });
assert.equal(slicedBitmapStats.bitmapBytes, 256,
  'telemetry reports the retained allocation rather than only a typed-array slice');

// Immutable playback keeps every cache identity at a dynamic/input/link or
// captured-object boundary, but drops the procedural DAG behind that boundary.
// Decisions are per identity: aliases are never partially released.
const staticOp = (id, cache, inputs = [], links = []) => ({
  id, cache, inputs, links, changed: false, _calcState: 2,
  classInfo: { dynamic: false },
});
const dynamicOp = (id, cache, inputs = [], links = []) => ({
  id, cache, inputs, links, changed: false, _calcState: 2,
  classInfo: { dynamic: true },
});
const bitmapCache = bytes => ({
  kind: 'bitmap', width: 1, height: Math.max(1, bytes / 8),
  data: new Uint16Array(new ArrayBuffer(bytes)),
});
const meshCache = (kind, bytes) => ({ kind, _compact: { values: new Uint8Array(bytes) } });

const intermediateBitmapOp = staticOp(20, bitmapCache(64));
const terminalBitmapOp = staticOp(21, bitmapCache(128), [intermediateBitmapOp]);
const capturedBitmapOp = staticOp(22, bitmapCache(32));
const finalMeshOp = staticOp(23, meshCache('mesh', 40));
const dynamicBitmapOp = dynamicOp(24, bitmapCache(24));
const sharedDeadCache = bitmapCache(256);
const sharedDeadA = staticOp(25, sharedDeadCache);
const sharedDeadB = staticOp(26, sharedDeadCache);
const sharedLiveCache = meshCache('mesh', 80);
const sharedLiveStatic = staticOp(27, sharedLiveCache);
const sharedLiveDynamic = dynamicOp(28, sharedLiveCache);
const unreachableMinMeshOp = staticOp(29, meshCache('minmesh', 96));
const changedBitmapOp = staticOp(30, bitmapCache(48));
changedBitmapOp.changed = true;
const rootMeshOp = staticOp(31, meshCache('mesh', 16));
const materialOp = dynamicOp(32, {
  kind: 'material', textures: [capturedBitmapOp.cache, terminalBitmapOp.cache],
}, [], [terminalBitmapOp]);
const sceneOp = dynamicOp(33, {
  kind: 'scene', children: [], drawMesh: finalMeshOp.cache,
}, [finalMeshOp]);
const immutableRuntime = {
  precalculated: true,
  operations: [
    intermediateBitmapOp, terminalBitmapOp, capturedBitmapOp, finalMeshOp,
    dynamicBitmapOp, sharedDeadA, sharedDeadB, sharedLiveStatic,
    sharedLiveDynamic, unreachableMinMeshOp, changedBitmapOp, rootMeshOp,
    materialOp, sceneOp,
  ],
  roots: [rootMeshOp], events: [], environment: {},
};
const pruning = D.pruneImmutablePlaybackCaches(immutableRuntime);
assert.equal(pruning.candidateReferences, 12);
assert.equal(pruning.candidateIdentities, 10);
assert.equal(pruning.retainedIdentities, 7);
assert.equal(pruning.releasedIdentities, 3);
assert.equal(pruning.clearedReferences, 4);
assert.equal(pruning.sharedIdentities, 2);
assert.deepEqual(JSON.parse(JSON.stringify(pruning.releasedKinds)), { bitmap: 2, minmesh: 1 });
assert.equal(pruning.candidateKnownBytes, 784);
assert.equal(pruning.estimatedReclaimableBytes, 416);
assert.equal(pruning.retainedKnownBytes, 368);
assert.equal(intermediateBitmapOp.cache, null, 'the DAG behind a terminal bitmap is pruned');
assert.ok(terminalBitmapOp.cache, 'a dynamic link boundary survives');
assert.ok(capturedBitmapOp.cache, 'a Material-captured bitmap survives');
assert.ok(finalMeshOp.cache, 'a Scene-captured/direct-input mesh survives');
assert.ok(dynamicBitmapOp.cache, 'a dynamic candidate cache survives');
assert.equal(sharedDeadA.cache, null);
assert.equal(sharedDeadB.cache, null, 'all dead aliases clear together');
assert.equal(sharedLiveStatic.cache, sharedLiveCache);
assert.equal(sharedLiveDynamic.cache, sharedLiveCache, 'one live alias retains the whole identity');
assert.equal(unreachableMinMeshOp.cache, null);
assert.ok(changedBitmapOp.cache, 'changed/incomplete ownership is never pruned');
assert.ok(rootMeshOp.cache, 'root caches survive even when their class is static');
assert.equal(D.pruneImmutablePlaybackCaches(immutableRuntime), pruning, 'pruning is idempotent');
assert.equal(D.runtimeCacheStats(immutableRuntime).pruning, pruning);

const sharedBacking = new ArrayBuffer(128);
const sharedBackingRuntime = {
  precalculated: true, roots: [], events: [], environment: {},
  operations: [
    staticOp(40, { kind: 'bitmap', width: 1, height: 8, data: new Uint16Array(sharedBacking) }),
    staticOp(41, { kind: 'bitmap', width: 1, height: 8, data: new Uint16Array(sharedBacking) }),
  ],
};
const sharedBackingPruning = D.pruneImmutablePlaybackCaches(sharedBackingRuntime);
assert.equal(sharedBackingPruning.releasedIdentities, 2);
assert.equal(sharedBackingPruning.candidateKnownBytes, 128);
assert.equal(sharedBackingPruning.estimatedReclaimableBytes, 128,
  'distinct dead cache shells sharing one allocation are counted once');

assert.throws(() => D.pruneImmutablePlaybackCaches({ operations: [], precalculated: false }),
  /requires completed precalc/);

// readPixels is bottom-up; the bitmap convention is top-down.
const pixels = new Uint8Array([
  1, 2, 3, 4, 5, 6, 7, 8,
  9, 10, 11, 12, 13, 14, 15, 16,
]);
const fakeGL = {
  RGBA: 1, UNSIGNED_BYTE: 2,
  readPixels(x, y, width, height, format, type, output) { output.set(pixels); },
};
assert.deepEqual(Array.from(D.decodeRenderPixels(fakeGL, 2, 2)), [
  9, 10, 11, 12, 13, 14, 15, 16,
  1, 2, 3, 4, 5, 6, 7, 8,
]);

// A precalc-time scene readback owns its renderer only until readPixels and
// drops the mesh's derivable prepared arrays before returning to graph precalc.
const preparedBuffer = new ArrayBuffer(96);
let compactCalls = 0, disposableRenderers = 0, disposableDisposals = 0;
const preparedMesh = {
  kind: 'minmesh',
  _prepared: {
    positions: new Float32Array(preparedBuffer, 0, 12),
    normals: new Float32Array(preparedBuffer, 48, 12),
  },
  compact() { compactCalls++; },
};
const disposableRuntime = {
  operations: [{ cache: preparedMesh }],
  environment: null,
};
D.newBitmap = () => ({ width: 1, height: 1 });
D.bitmapFromRGBA = (width, height, rgba) => ({ width, height, rgba });
D.Environment = class {
  constructor(runtime) { this.runtime = runtime; disposableRuntime.environment = this; }
  initView() {}
  initFrame() {}
  exitFrame() {}
};
D.Renderer = class {
  constructor() {
    disposableRenderers++;
    this.gl = {
      RGBA: 1, UNSIGNED_BYTE: 2,
      readPixels(x, y, width, height, format, type, output) { output.set([7, 8, 9, 10]); },
    };
  }
  render() {}
  resourceStats() { return { totalEstimatedBytes: 1234 }; }
  dispose() { disposableDisposals++; }
};
let disposableReport = null;
const disposableHook = D.makeDisposableBitmapRenderer(
  disposableRuntime, {}, {}, report => { disposableReport = report; }, D,
);
const disposableResult = disposableHook({
  widthExponent: 0, heightExponent: 0,
  op: { inputs: [{ exec(environment) { environment.lastOutput = {}; } }] },
});
assert.deepEqual(Array.from(disposableResult.rgba), [7, 8, 9, 10]);
assert.equal(disposableRenderers, 1);
assert.equal(disposableDisposals, 1);
assert.equal(preparedMesh._prepared, null);
assert.equal(compactCalls, 1);
assert.equal(disposableReport.discarded.geometries, 1);
assert.equal(disposableReport.discarded.bytes, 96);
assert.equal(disposableReport.resources.totalEstimatedBytes, 1234);

// Playback topology release is intentionally narrow: one static, completed
// operator cache, consumed by Scene only. Aliases, upstream geometry and
// animated caches remain fully reconstructable.
const makePlaybackMesh = (kind = 'mesh', animated = false) => ({
  kind, released: false, _sharedVertices: false,
  _compact: { values: new Uint8Array(64) },
  hasAnimation: () => animated,
  prepare(options = {}) {
    this._prepared ||= { positions: new Float32Array(12), indices: new Uint16Array(6) };
    if (options.releaseTopology || this._playbackTopologyRelease) {
      this.releaseTopologyForPlayback();
    }
    return this._prepared;
  },
  releaseTopologyForPlayback() {
    const deferredRelease = this._playbackTopologyRelease || null;
    this._playbackTopologyRelease = null;
    this.topologyReleasedForPlayback = true;
    this._compact = null;
    deferredRelease?.release?.(this._prepared);
  },
});
const eligibleMesh = makePlaybackMesh();
const eligiblePrepare = eligibleMesh.prepare;
eligibleMesh.prepare = function prepareWithReleasedLazyState(options) {
  const geometry = eligiblePrepare.call(this, options);
  Object.defineProperty(geometry, 'animation', {
    enumerable: true,
    get() { throw new Error('released lazy state must not be read for byte accounting'); },
  });
  return geometry;
};
const sharedMesh = makePlaybackMesh();
const upstreamMesh = makePlaybackMesh();
const animatedMesh = makePlaybackMesh('minmesh', true);
const eligibleOp = { id: 1, cache: eligibleMesh, changed: false, _calcState: 2,
  classInfo: { dynamic: false }, outputClassId: D.KC_MESH, inputs: [], links: [] };
const sharedA = { id: 2, cache: sharedMesh, changed: false, _calcState: 2,
  classInfo: { dynamic: false }, outputClassId: D.KC_MESH, inputs: [], links: [] };
const sharedB = { id: 3, cache: sharedMesh, changed: false, _calcState: 2,
  classInfo: { dynamic: false }, outputClassId: D.KC_MESH, inputs: [], links: [] };
const upstreamOp = { id: 4, cache: upstreamMesh, changed: false, _calcState: 2,
  classInfo: { dynamic: false }, outputClassId: D.KC_MESH, inputs: [], links: [] };
const animatedOp = { id: 5, cache: animatedMesh, changed: false, _calcState: 2,
  classInfo: { dynamic: false }, outputClassId: D.KC_MINMESH, inputs: [], links: [] };
const geometryConsumer = { id: 6, cache: {}, changed: false, _calcState: 2,
  classInfo: { dynamic: false }, outputClassId: D.KC_MESH, inputs: [upstreamOp], links: [] };
const sceneConsumer = { id: 7, cache: { kind: 'scene' }, changed: false, _calcState: 2,
  classInfo: { dynamic: true }, outputClassId: D.KC_SCENE,
  inputs: [eligibleOp, sharedA, sharedB, geometryConsumer, animatedOp], links: [] };
const playbackRuntime = { operations: [eligibleOp, sharedA, sharedB, upstreamOp,
  animatedOp, geometryConsumer, sceneConsumer] };
assert.deepEqual(Array.from(D.terminalStaticGeometry(playbackRuntime), item => item.operation.id), [1]);
const playbackStats = await D.prepareStaticTerminalGeometry(playbackRuntime, {
  budgetMilliseconds: 1000, yield: async () => {},
});
assert.equal(playbackStats.candidates, 1);
assert.equal(playbackStats.prepared, 1);
assert.equal(playbackStats.releasedTopologyBytes, 64);
assert.equal(playbackStats.preparedGeometryBytes, 60);
assert.equal(playbackStats.estimatedNetBytes, -4);
assert.equal(eligibleMesh.topologyReleasedForPlayback, true);
assert.ok(sharedMesh._compact && upstreamMesh._compact && animatedMesh._compact);

// Once pruning establishes immutable playback, aliases of the same captured
// Scene mesh are one preparation identity. Uncaptured execution boundaries,
// animated topology and incomplete owners stay reconstructable. Physical-byte
// telemetry also treats a compact buffer retained by another shell as live.
const immutableAliasedMesh = makePlaybackMesh();
const sharedTopology = new ArrayBuffer(128);
const capturedSharedBackingMesh = makePlaybackMesh();
capturedSharedBackingMesh._compact.values = new Uint8Array(sharedTopology);
const uncapturedSharedBackingMesh = makePlaybackMesh();
uncapturedSharedBackingMesh._compact.values = new Uint8Array(sharedTopology);
const immutableAnimatedMesh = makePlaybackMesh('minmesh', true);
const incompleteCapturedMesh = makePlaybackMesh();
const immutableAliasA = { ...eligibleOp, id: 50, cache: immutableAliasedMesh };
const immutableAliasB = { ...eligibleOp, id: 51, cache: immutableAliasedMesh };
const capturedBackingOp = { ...eligibleOp, id: 52, cache: capturedSharedBackingMesh };
const uncapturedBackingOp = { ...eligibleOp, id: 53, cache: uncapturedSharedBackingMesh };
const immutableAnimatedOp = { ...animatedOp, id: 54, cache: immutableAnimatedMesh };
const incompleteCapturedOp = { ...eligibleOp, id: 55, cache: incompleteCapturedMesh, changed: true };
const immutableSceneOp = {
  ...sceneConsumer, id: 56,
  cache: {
    kind: 'scene',
    children: [
      { drawMesh: immutableAliasedMesh },
      { drawMesh: capturedSharedBackingMesh },
      { drawMesh: immutableAnimatedMesh },
      { drawMesh: incompleteCapturedMesh },
    ],
  },
  inputs: [immutableAliasA, immutableAliasB, capturedBackingOp,
    immutableAnimatedOp, incompleteCapturedOp],
};
const immutablePlaybackRuntime = {
  immutablePlayback: true,
  playbackCachePruning: { immutable: true },
  operations: [
    immutableAliasA, immutableAliasB, capturedBackingOp, uncapturedBackingOp,
    immutableAnimatedOp, incompleteCapturedOp, immutableSceneOp,
  ],
};
assert.deepEqual(
  Array.from(D.terminalStaticGeometry(immutablePlaybackRuntime), item => item.operation.id),
  [50, 52],
);
const immutablePlaybackStats = await D.prepareStaticTerminalGeometry(immutablePlaybackRuntime, {
  budgetMilliseconds: 1000, yield: async () => {},
});
assert.equal(immutablePlaybackStats.immutable, true);
assert.equal(immutablePlaybackStats.candidates, 2);
assert.equal(immutablePlaybackStats.candidateReferences, 3);
assert.equal(immutablePlaybackStats.sharedCandidateIdentities, 1);
assert.equal(immutablePlaybackStats.uncapturedExecutionIdentities, 1);
assert.equal(immutablePlaybackStats.animatedIdentities, 1);
assert.equal(immutablePlaybackStats.incompleteOwnerIdentities, 1);
assert.equal(immutablePlaybackStats.candidateTopologyBytes, 192);
assert.equal(immutablePlaybackStats.retainedSharedTopologyBytes, 128);
assert.equal(immutablePlaybackStats.potentiallyReclaimableTopologyBytes, 64);
assert.equal(immutablePlaybackStats.releasedTopologyBytes, 0,
  'init does not prepare unseen timeline geometry');
assert.equal(immutablePlaybackStats.preparedGeometryBytes, 0);
assert.equal(immutablePlaybackStats.createdPreparedGeometryBytes, 0);
assert.equal(immutablePlaybackStats.queued, 2);
assert.ok(immutableAliasedMesh._compact && capturedSharedBackingMesh._compact &&
  uncapturedSharedBackingMesh._compact && immutableAnimatedMesh._compact &&
  incompleteCapturedMesh._compact);
assert.equal(
  await D.prepareStaticTerminalGeometry(immutablePlaybackRuntime),
  immutablePlaybackStats,
  'immutable topology compaction is idempotent',
);
assert.equal(D.runtimeCacheStats(immutablePlaybackRuntime).playbackGeometry,
  immutablePlaybackStats);
immutableAliasedMesh.prepare();
assert.equal(immutableAliasedMesh.topologyReleasedForPlayback, true);
assert.equal(immutablePlaybackStats.releasedTopologyBytes, 64);
assert.equal(immutablePlaybackStats.preparedGeometryBytes, 60);
assert.equal(immutablePlaybackStats.pendingIdentities, 1);
capturedSharedBackingMesh.prepare();
assert.equal(capturedSharedBackingMesh.topologyReleasedForPlayback, true);
assert.equal(immutablePlaybackStats.releasedTopologyBytes, 64,
  'a backing buffer still referenced by an excluded shell is not reclaimed');
assert.equal(immutablePlaybackStats.preparedGeometryBytes, 120);
assert.equal(immutablePlaybackStats.deferredPreparedGeometryBytes, 120);
assert.equal(immutablePlaybackStats.createdPreparedGeometryBytes, 0,
  'renderer-required lazy preparation is not attributed to compaction');
assert.equal(immutablePlaybackStats.conversionNetBytes, 56);
assert.equal(immutablePlaybackStats.estimatedNetBytes, -64);
assert.equal(immutablePlaybackStats.pendingIdentities, 0);

// Cancelling lazy immutable-geometry marking removes every not-yet-fired
// release closure and clears the incomplete summary. A discarded Runtime must
// not remain pinned through a deferred mesh marker.
{
  const abortMeshA = makePlaybackMesh();
  const abortMeshB = makePlaybackMesh();
  const abortOpA = { ...eligibleOp, id: 70, cache: abortMeshA };
  const abortOpB = { ...eligibleOp, id: 71, cache: abortMeshB };
  const abortScene = {
    ...sceneConsumer,
    id: 72,
    cache: {
      kind: 'scene',
      children: [{ drawMesh: abortMeshA }, { drawMesh: abortMeshB }],
    },
    inputs: [abortOpA, abortOpB],
  };
  const abortRuntime = {
    immutablePlayback: true,
    playbackCachePruning: { immutable: true },
    operations: [abortOpA, abortOpB, abortScene],
  };
  const controller = new AbortController();
  let armed = false;
  let announceYield;
  const enteredYield = new Promise(resolve => { announceYield = resolve; });
  const stalledYield = new Promise(() => {});
  const preparation = D.prepareStaticTerminalGeometry(abortRuntime, {
    budgetMilliseconds: 0,
    signal: controller.signal,
    onProgress() { armed = true; },
    yield() {
      if (!armed) return Promise.resolve();
      announceYield();
      return stalledYield;
    },
  });
  await enteredYield;
  controller.abort();
  await assert.rejects(preparation, error => error?.name === 'AbortError');
  assert.equal(abortRuntime.playbackGeometryCompaction, null);
  assert.equal(abortMeshA._playbackTopologyRelease, undefined);
  assert.equal(abortMeshB._playbackTopologyRelease, undefined);
  assert.ok(abortMeshA._compact && abortMeshB._compact,
    'deferred topology remains reconstructable when marking is cancelled');
}

// Playback warm-up planning follows event time, deduplicates aliases/cycles,
// sees compact MinMesh material slots without expanding topology, and never
// invokes unrelated accessors.
{
  const materialA = { kind: 'material', textures: [] };
  const materialB = { kind: 'material', textures: [] };
  const lateMesh = {
    kind: 'mesh', materials: [{ material: materialA }],
  };
  const earlyMinMesh = {
    kind: 'minmesh', _compact: { clusters: [{ material: materialB }] },
  };
  let accessorReads = 0;
  const earlyScene = { kind: 'scene', children: [], drawMesh: earlyMinMesh };
  Object.defineProperty(earlyScene, 'forbidden', {
    enumerable: true,
    get() { accessorReads++; throw new Error('warm-up planner invoked an accessor'); },
  });
  earlyScene.children.push(earlyScene);
  const lateScene = { kind: 'scene', children: [], drawMesh: lateMesh };
  const earlyMeshOp = { id: 201, cache: earlyMinMesh };
  const lateMeshOp = { id: 202, cache: lateMesh };
  const materialAOp = { id: 203, cache: materialA };
  const materialBOp = { id: 204, cache: materialB };
  const earlySceneOp = { id: 205, cache: earlyScene };
  const lateSceneOp = { id: 206, cache: lateScene };
  const plan = D.collectPlaybackResourcePlan({
    operations: [lateSceneOp, materialAOp, lateMeshOp, earlySceneOp, materialBOp, earlyMeshOp],
    roots: [],
    events: [
      { start: 20, op: lateSceneOp },
      { start: 10, op: earlySceneOp },
    ],
  });
  assert.equal(accessorReads, 0);
  assert.equal(plan.meshes, 2);
  assert.equal(plan.materials, 2);
  assert.deepEqual(plan.tasks.map(task => [task.kind, task.sourceId]), [
    ['mesh', 201], ['material', 204], ['mesh', 202], ['material', 203],
  ]);
}

// App retains the fixed production-rate player even when WebAudio selects a
// different sink rate. Duration and visual timing stay in source samples.
const document = { song: { id: 'song' }, songLength: 65536, songBPM: 60 };
const initialPlayer = { sampleRate: 44100, calcSongSamples: () => 44100 };
const loaderSong = new Uint8Array([1, 2, 3]);
const lifecycle = [];
const playerFactoryCalls = [];
class FakeRenderer {
  constructor(canvas, options = {}) {
    this.canvas = canvas; this.options = { ...options };
    lifecycle.push('renderer:construct');
  }
  async prewarmResources(plan) {
    lifecycle.push('renderer:warm');
    this.warmPlan = plan;
    return { enabled: true, plannedTasks: plan.tasks.length, completedTasks: plan.tasks.length };
  }
}
class FakeRuntime {
  constructor(doc, options = {}) {
    this.options = { ...options };
    this.document = doc; this.operations = []; this.roots = []; this.events = [];
    this.environment = {}; this.precalculated = false;
  }
  get playbackCachePruning() { return this._playbackCachePruning || null; }
  set playbackCachePruning(value) {
    this._playbackCachePruning = value;
    lifecycle.push('prune');
  }
  get playbackGeometryCompaction() { return this._playbackGeometryCompaction || null; }
  set playbackGeometryCompaction(value) {
    this._playbackGeometryCompaction = value;
    lifecycle.push('static-geometry');
  }
  precalc() { lifecycle.push('precalc'); this.precalculated = true; }
  snapshot() { return { frame: 0 }; }
}
class FakeAudioStream {
  constructor(options) {
    this.index = FakeAudioStream.instances.length;
    this.options = options; this.onEnded = options.onEnded;
    this.sampleRate = options.sampleRate; this.outputSampleRate = 48000;
    this.workerSong = options.workerSong || null;
    this.reportClock = options.reportClock;
    this.workerUrl = options.workerUrl;
    this.workerPlayerOptions = { ...(options.workerPlayerOptions || {}) };
    this.workerLoader = Boolean(this.workerSong);
    this.producerBackend = this.workerLoader ? 'worker' : 'main-worklet';
    this.durationSamples = 88200; this.endSample = null; this.paused = true;
    this.starts = 0; this.closes = 0;
    FakeAudioStream.instances.push(this);
    lifecycle.push(`stream:${this.index}:construct`);
  }
  async init(player) {
    lifecycle.push(`stream:${this.index}:init:${this.workerLoader ? 'loader-worker' : 'main'}`);
    this.initialPlayer = player;
    this.synth = this.workerLoader ? null : player;
    return this;
  }
  async start() {
    lifecycle.push(`stream:${this.index}:start`);
    this.starts++; this.paused = false; return this;
  }
  async seek(seconds) { this.seekSeconds = seconds; return seconds; }
  pause(value) { this.paused = value; return value; }
  async close() { lifecycle.push(`stream:${this.index}:close`); this.closes++; }
}
FakeAudioStream.instances = [];
D.loadProductionData = async () => ({ kx: new Uint8Array(0), loaderSong });
D.parseKX = () => document;
D.Renderer = FakeRenderer;
D.Runtime = FakeRuntime;
D.createV2Player = (song, options = {}) => {
  lifecycle.push('player:main');
  playerFactoryCalls.push({ song, options });
  return initialPlayer;
};
D.AudioStream = FakeAudioStream;

const app = new D.DebrisApp({}, null, {
  dependencies: D,
  sampleRate: 44100,
  audioQueueBlocks: 5,
  audioTailSeconds: 7,
  audioWorkerUrl: 'worker:test-loader',
  dxt5Mode: 'rgba8',
});
await app.init();
assert.equal(app.renderer.options.dxt5Mode, 'rgba8',
  'the app forwards its DXT5 diagnostic mode to the playback renderer');
assert.equal(app.runtime.options.reuseHandlerCallRecords, true,
  'the app opts its built-in, non-retaining handlers into call-record reuse');
assert.equal(FakeAudioStream.instances.length, 2);
const loaderStream = FakeAudioStream.instances[0];
assert.equal(loaderStream.initialPlayer, undefined,
  'the Worker-backed loader owns its V2 player instead of receiving one from app.js');
assert.equal(loaderStream.workerSong, loaderSong);
assert.equal(loaderStream.reportClock, false);
assert.equal(loaderStream.workerUrl, 'worker:test-loader');
assert.deepEqual(loaderStream.workerPlayerOptions, {
  checkpointMemoryBytes: 0,
  checkpointIntervalSamples: 0,
});
assert.equal(loaderStream.options.queueBlocks, 5);
assert.equal(loaderStream.options.tailSeconds, 0);
assert.equal(app.stats.loaderAudioBackend, 'worker');
assert.equal(app.stats.loaderAudioQueueBytes, 2048 * 5 * 2 * 4);
assert.equal(app.stats.loaderAudioUnderruns, 0);
assert.equal(app.stats.loaderAudioErrors, 0);
assert.equal(loaderStream.starts, 1);
assert.equal(loaderStream.closes, 1);
assert.equal(loaderStream.synth, null,
  'the Worker-backed loader never exposes or retains a main-thread synth');
assert.equal(app.loaderAudio, null);
assert.equal(playerFactoryCalls.length, 1,
  'only the main soundtrack is constructed through the app-level V2 factory');
assert.equal(playerFactoryCalls[0].song, document.song);
assert.equal(app.audio.initialPlayer, initialPlayer);
assert.equal(app.audio.starts, 0, 'main soundtrack still starts only in DebrisApp.start');
assert.equal(app.audio.closes, 0);
assert.equal(app.player, initialPlayer);
assert.equal(app.sampleRate, 44100);
assert.equal(app.stats.audioOutputSampleRate, 48000);
assert.equal(app.stats.loaderAudioOutputSampleRate, 48000);
assert.equal(app.stats.playbackWarmup.enabled, true);
assert.equal(app.renderer.warmPlan.tasks.length, 0);
assert.equal(app.duration, 2);
assert.equal(app.audio.options.queueBlocks, 5);
assert.equal(app.audio.options.tailSeconds, 7);
assert.deepEqual(lifecycle.slice(0, 12), [
  'stream:0:construct',
  'stream:0:init:loader-worker',
  'stream:0:start',
  'precalc',
  'prune',
  'static-geometry',
  'renderer:construct',
  'renderer:warm',
  'stream:0:close',
  'player:main',
  'stream:1:construct',
  'stream:1:init:main',
], 'bounded resource warming remains under loader Worker coverage and precedes the main player');

// A warm-up failure closes the loader, disposes the partially populated
// playback renderer, and cannot construct the main soundtrack.
{
  const warmupError = new Error('resource warm-up failure');
  let warmupRendererDisposals = 0;
  class WarmupFailRenderer extends FakeRenderer {
    async prewarmResources() { lifecycle.push('renderer:warm-fail'); throw warmupError; }
    dispose() { warmupRendererDisposals++; }
  }
  const savedRenderer = D.Renderer;
  D.Renderer = WarmupFailRenderer;
  const playerFactoryCallCount = playerFactoryCalls.length;
  const failingWarmupApp = new D.DebrisApp({}, null, { dependencies: D });
  try {
    await assert.rejects(failingWarmupApp.init(), error => error === warmupError);
    const stream = FakeAudioStream.instances.at(-1);
    assert.equal(stream.closes, 1);
    assert.equal(failingWarmupApp.loaderAudio, null);
    assert.equal(failingWarmupApp.renderer, null);
    assert.equal(warmupRendererDisposals, 1);
    assert.equal(playerFactoryCalls.length, playerFactoryCallCount);
  } finally {
    D.Renderer = savedRenderer;
  }
}

// Deterministic fixed-time rendering does not create either loader or main
// AudioStream, even though the embedded loader song is available.
const streamCountBeforeDebug = FakeAudioStream.instances.length;
const debugApp = new D.DebrisApp({}, null, { debugTime: 15, dependencies: D });
await debugApp.init();
assert.equal(FakeAudioStream.instances.length, streamCountBeforeDebug);
assert.equal(debugApp.loaderAudio, null);
assert.equal(debugApp.audio, null);
assert.equal(debugApp.debugSample, 15 * debugApp.sampleRate);
debugApp.dispose();

// Arbitrary injected handlers may retain their call arguments. The app must
// leave record reuse disabled at that extension boundary.
{
  const customHandlers = new Map([[123, { exec() {} }]]);
  const customHandlerApp = new D.DebrisApp({}, null, {
    debugTime: 0, dependencies: D, handlers: customHandlers,
  });
  await customHandlerApp.init();
  assert.equal(customHandlerApp.runtime.options.handlers, customHandlers);
  assert.equal(customHandlerApp.runtime.options.reuseHandlerCallRecords, false);
  customHandlerApp.dispose();
}

// A loader close failure must not replace the primary precalc exception.
const primaryPrecalcError = new Error('primary precalc failure');
const secondaryCloseError = new Error('secondary loader close failure');
class FailingRuntime extends FakeRuntime {
  precalc() { throw primaryPrecalcError; }
}
class FailingLoaderStream extends FakeAudioStream {
  async close() { this.closes++; throw secondaryCloseError; }
}
const savedRuntime = D.Runtime, savedAudioStream = D.AudioStream;
D.Runtime = FailingRuntime;
D.AudioStream = FailingLoaderStream;
const originalWarn = console.warn;
let loaderCloseWarning = null;
console.warn = (...values) => { loaderCloseWarning = values; };
try {
  const playerFactoryCallCount = playerFactoryCalls.length;
  const failingApp = new D.DebrisApp({}, null, { dependencies: D });
  await assert.rejects(failingApp.init(), error => error === primaryPrecalcError);
  assert.equal(failingApp.loaderAudio, null);
  const failingLoaderStream = FakeAudioStream.instances[FakeAudioStream.instances.length - 1];
  assert.equal(failingLoaderStream.initialPlayer, undefined);
  assert.equal(failingLoaderStream.workerSong, loaderSong);
  assert.equal(failingLoaderStream.synth, null,
    'the failing Worker loader never leaves a main-thread synth reachable');
  assert.equal(playerFactoryCalls.length, playerFactoryCallCount,
    'a preparation failure cannot construct the main player');
  assert.equal(loaderCloseWarning?.[0], 'loader audio shutdown failed');
  assert.equal(loaderCloseWarning?.[1], secondaryCloseError);
} finally {
  console.warn = originalWarn;
  D.Runtime = savedRuntime;
  D.AudioStream = savedAudioStream;
}

// dispose() also owns a loader stream if initialization is interrupted before
// the precalc finally block runs.
const partialApp = new D.DebrisApp({});
let partialLoaderCloses = 0;
partialApp.loaderAudio = { async close() { partialLoaderCloses++; } };
partialApp.dispose();
await Promise.resolve();
assert.equal(partialLoaderCloses, 1);
assert.equal(partialApp.loaderAudio, null);

const deferred = () => {
  let resolve, reject;
  const promise = new Promise((accept, decline) => { resolve = accept; reject = decline; });
  return { promise, resolve, reject };
};
class CancellationRuntime extends FakeRuntime {
  constructor(doc) { super(doc); this.currentRoot = 0; }
}

// Concurrent stop requests share one shutdown and emit one lifecycle callback.
{
  const closing = deferred();
  let closes = 0, ends = 0;
  const stopApp = new D.DebrisApp({}, null, { onEnd: () => { ends++; } });
  stopApp.ready = true;
  stopApp.running = true;
  stopApp.audio = { async close() { closes++; await closing.promise; } };
  const first = stopApp.stop(true);
  const second = stopApp.stop(true);
  assert.equal(closes, 1);
  closing.resolve();
  await Promise.all([first, second]);
  assert.equal(closes, 1);
  assert.equal(ends, 1);
  await stopApp.stop(false);
  assert.equal(ends, 1, 'a completed stop remains idempotent');
  stopApp.dispose();
}

// stop() cancels a pending audio.start wait. A late device-start resolution
// cannot emit onStart or queue playback after onEnd has already run.
{
  const enteredStart = deferred(), releaseStart = deferred();
  let starts = 0, closes = 0, startCallbacks = 0, ends = 0;
  const startStopApp = new D.DebrisApp({}, null, {
    onStart: () => { startCallbacks++; },
    onEnd: () => { ends++; },
  });
  startStopApp.ready = true;
  startStopApp.audio = {
    async start() { starts++; enteredStart.resolve(); await releaseStart.promise; },
    async close() { closes++; },
  };
  const starting = startStopApp.start();
  await enteredStart.promise;
  let concurrentStartSettled = false;
  const concurrentStart = startStopApp.start();
  const observeConcurrentStart = concurrentStart.then(
    () => { concurrentStartSettled = true; },
    () => { concurrentStartSettled = true; },
  );
  await Promise.resolve();
  assert.equal(concurrentStartSettled, false,
    'concurrent start callers share the pending device-start lifecycle');
  const stopping = startStopApp.stop(false);
  await assert.rejects(starting, error => error?.name === 'AbortError');
  await assert.rejects(concurrentStart, error => error?.name === 'AbortError');
  await observeConcurrentStart;
  await stopping;
  releaseStart.resolve();
  await Promise.resolve();
  assert.deepEqual({ starts, closes, startCallbacks, ends }, {
    starts: 1, closes: 1, startCallbacks: 0, ends: 1,
  });
  assert.equal(startStopApp.running, false);
  assert.equal(startStopApp.activeStartController, null);
  await assert.rejects(startStopApp.start(), /cannot restart after stop/,
    'a closed main AudioStream is a terminal app state');
  startStopApp.dispose();
}

// Disposal invalidates a stop wait without rejecting its public promise or
// invoking the stale end callback. The shared close remains exactly-once.
{
  const closing = deferred();
  let closes = 0, ends = 0;
  const stopDisposeApp = new D.DebrisApp({}, null, { onEnd: () => { ends++; } });
  stopDisposeApp.ready = true;
  stopDisposeApp.running = true;
  stopDisposeApp.audio = { async close() { closes++; await closing.promise; } };
  const stopping = stopDisposeApp.stop(true);
  stopDisposeApp.dispose();
  await stopping;
  assert.equal(closes, 1);
  assert.equal(ends, 0);
  closing.resolve();
  await Promise.resolve();
  assert.equal(closes, 1);
}

// Repeated seek requests are serialized. The second target is based on the
// first completed replay, rather than interleaving two Runtime restores.
{
  const firstReplay = deferred();
  const replayOrder = [];
  const audioTargets = [];
  const pauseValues = [];
  const seekApp = new D.DebrisApp({}, null, { sampleRate: 1000 });
  seekApp.ready = true;
  seekApp.duration = 5;
  seekApp.audio = {
    durationSamples: 5000,
    async seek(seconds) { audioTargets.push(seconds); },
    pause(value) { pauseValues.push(value); },
    async close() {},
  };
  seekApp.snapshots = [{ second: 0, sample: 0, state: {} }];
  seekApp.seekRuntime = async target => {
    replayOrder.push(`start:${target}`);
    if (!seekApp.currentSample) await firstReplay.promise;
    seekApp.currentSample = target;
    replayOrder.push(`end:${target}`);
  };
  const first = seekApp.seek(1);
  await Promise.resolve();
  const second = seekApp.seek(1);
  await Promise.resolve();
  assert.deepEqual(replayOrder, ['start:1000']);
  assert.equal(seekApp.pause(), true,
    'space during a seek toggles the requested playback state, not the forced seek pause');
  firstReplay.resolve();
  await Promise.all([first, second]);
  assert.deepEqual(replayOrder, ['start:1000', 'end:1000', 'start:2000', 'end:2000']);
  assert.deepEqual(audioTargets, [1, 2]);
  assert.equal(seekApp.paused, true);
  assert.deepEqual(pauseValues, [true, true]);
  seekApp.dispose();
}

// stop() aborts a long seek replay before closing audio and never lets that
// stale replay publish a later currentSample.
{
  const enteredReplay = deferred();
  let closes = 0, ends = 0;
  const seekStopApp = new D.DebrisApp({}, null, {
    sampleRate: 1000,
    onEnd: () => { ends++; },
  });
  seekStopApp.ready = true;
  seekStopApp.running = true;
  seekStopApp.duration = 5;
  seekStopApp.audio = {
    durationSamples: 5000,
    async seek() {},
    pause() {},
    async close() { closes++; },
  };
  seekStopApp.seekRuntime = async (target, options) => {
    enteredReplay.resolve();
    await new Promise((resolve, reject) => {
      if (options.signal.aborted) reject(options.signal.reason);
      else options.signal.addEventListener('abort', () => reject(options.signal.reason), {
        once: true,
      });
    });
    seekStopApp.currentSample = target;
  };
  const seeking = seekStopApp.seek(1);
  await enteredReplay.promise;
  const stopping = seekStopApp.stop(false);
  await assert.rejects(seeking, error => error?.name === 'AbortError');
  await stopping;
  assert.equal(seekStopApp.currentSample, 0);
  assert.equal(closes, 1);
  assert.equal(ends, 1);
  assert.equal(seekStopApp.activeSeekController, null);
  seekStopApp.dispose();
}

// A failure on either half of Promise.all aborts the other half. In particular,
// an audio seek rejection cannot leave Runtime replay mutating in the background.
{
  const audioError = new Error('audio seek failed');
  let runtimeAborted = false, closes = 0;
  const failingSeekApp = new D.DebrisApp({}, null, { sampleRate: 1000 });
  failingSeekApp.ready = true;
  failingSeekApp.duration = 5;
  failingSeekApp.audio = {
    durationSamples: 5000,
    async seek() { throw audioError; },
    pause() {},
    async close() { closes++; },
  };
  failingSeekApp.seekRuntime = async (target, options) => {
    await new Promise((resolve, reject) => {
      options.signal.addEventListener('abort', () => {
        runtimeAborted = true;
        reject(options.signal.reason);
      }, { once: true });
    });
  };
  await assert.rejects(failingSeekApp.seek(1), error => error === audioError);
  assert.equal(runtimeAborted, true);
  assert.equal(failingSeekApp.audio, null,
    'a partially failed audio/runtime seek terminates playback instead of resuming divergent state');
  assert.equal(closes, 1);
  failingSeekApp.dispose();
}

// Escape/dispose while the production-data promise is outstanding cannot
// construct any graph, renderer, player, or audio resource after it resolves.
{
  const load = deferred();
  let loadSignal = null;
  let parses = 0, runtimes = 0, renderers = 0, streams = 0, players = 0;
  class LoadRaceRuntime extends CancellationRuntime { constructor(doc) { super(doc); runtimes++; } }
  class LoadRaceRenderer { constructor() { renderers++; } }
  class LoadRaceStream { constructor() { streams++; } }
  const dependencies = {
    ...D,
    loadProductionData: signal => { loadSignal = signal; return load.promise; },
    parseKX: () => { parses++; return document; },
    Runtime: LoadRaceRuntime,
    Renderer: LoadRaceRenderer,
    AudioStream: LoadRaceStream,
    createV2Player: () => { players++; return initialPlayer; },
  };
  const loadRaceApp = new D.DebrisApp({}, null, { dependencies });
  const initializing = loadRaceApp.init();
  loadRaceApp.dispose();
  assert.equal(loadSignal?.aborted, true);
  await assert.rejects(initializing, error => error?.name === 'AbortError');
  load.resolve({ kx: new Uint8Array(0), loaderSong });
  await Promise.resolve();
  assert.deepEqual({ parses, runtimes, renderers, streams, players }, {
    parses: 0, runtimes: 0, renderers: 0, streams: 0, players: 0,
  });
  assert.equal(loadRaceApp.ready, false);
  assert.equal(loadRaceApp.runtime, null);
  assert.equal(loadRaceApp.renderer, null);
  assert.equal(loadRaceApp.player, null);
  assert.equal(loadRaceApp.audio, null);
}

// Even an uncooperative precalc implementation that resolves after dispose
// cannot reach pruning or renderer construction. The app passes its live
// AbortSignal into the real Runtime implementation.
{
  const entered = deferred(), release = deferred();
  let precalcSignal = null, renderers = 0, players = 0;
  class PrecalcRaceRuntime extends CancellationRuntime {
    async precalcAsync(root, options) {
      precalcSignal = options.signal;
      entered.resolve();
      await release.promise;
      this.precalculated = true;
      return {};
    }
  }
  class PrecalcRaceRenderer { constructor() { renderers++; } }
  const dependencies = {
    ...D,
    loadProductionData: async () => ({ kx: new Uint8Array(0), loaderSong: new Uint8Array(0) }),
    parseKX: () => document,
    Runtime: PrecalcRaceRuntime,
    Renderer: PrecalcRaceRenderer,
    createV2Player: () => { players++; return initialPlayer; },
  };
  const precalcRaceApp = new D.DebrisApp({}, null, { dependencies, debugTime: 0 });
  const initializing = precalcRaceApp.init();
  await entered.promise;
  precalcRaceApp.dispose();
  assert.equal(precalcSignal?.aborted, true);
  await assert.rejects(initializing, error => error?.name === 'AbortError');
  release.resolve();
  await Promise.resolve();
  assert.equal(renderers, 0);
  assert.equal(players, 0);
  assert.equal(precalcRaceApp.stats.playbackCaches, undefined,
    'a late precalc result never enters immutable-cache pruning');
  assert.equal(precalcRaceApp.runtime, null);
}

// Dispose during renderer warm-up closes the loader and disposes the renderer
// once. A late warm-up result cannot create the main player or main stream.
{
  const entered = deferred(), release = deferred();
  let rendererConstructs = 0, rendererDisposals = 0, warmSignal = null;
  let playerConstructs = 0;
  class WarmRaceRenderer {
    constructor() { rendererConstructs++; }
    async prewarmResources(plan, options) {
      warmSignal = options.signal;
      entered.resolve();
      await release.promise;
      return { enabled: true, plannedTasks: 0, completedTasks: 0 };
    }
    dispose() { rendererDisposals++; }
  }
  class WarmRaceStream {
    static instances = [];
    constructor(options) {
      this.workerSong = options.workerSong || null;
      this.producerBackend = this.workerSong ? 'worker' : 'main-worklet';
      this.outputSampleRate = 44100;
      this.closes = 0;
      WarmRaceStream.instances.push(this);
    }
    async init(player) { this.synth = player || null; return this; }
    async start() { return this; }
    async close() { this.closes++; }
  }
  const dependencies = {
    ...D,
    loadProductionData: async () => ({ kx: new Uint8Array(0), loaderSong }),
    parseKX: () => document,
    Runtime: CancellationRuntime,
    Renderer: WarmRaceRenderer,
    AudioStream: WarmRaceStream,
    createV2Player: () => { playerConstructs++; return initialPlayer; },
  };
  const warmRaceApp = new D.DebrisApp({}, null, { dependencies });
  const initializing = warmRaceApp.init();
  await entered.promise;
  warmRaceApp.dispose();
  assert.equal(warmSignal?.aborted, true);
  await assert.rejects(initializing, error => error?.name === 'AbortError');
  release.resolve();
  await Promise.resolve();
  assert.equal(rendererConstructs, 1);
  assert.equal(rendererDisposals, 1);
  assert.equal(WarmRaceStream.instances.length, 1);
  assert.equal(WarmRaceStream.instances[0].closes, 1);
  assert.equal(playerConstructs, 0);
  assert.equal(warmRaceApp.stats.playbackWarmup, undefined,
    'a late warm-up result cannot publish into the disposed app');
  assert.equal(warmRaceApp.renderer, null);
  assert.equal(warmRaceApp.runtime, null);
  assert.equal(warmRaceApp.ready, false);
}

// Main-audio initialization has the same ownership rule: dispose closes the
// stream and renderer once, and the late init resolution cannot mark ready.
{
  const entered = deferred(), release = deferred();
  let rendererConstructs = 0, rendererDisposals = 0;
  class AudioRaceRenderer {
    constructor() { rendererConstructs++; }
    dispose() { rendererDisposals++; }
  }
  class AudioRaceStream {
    static instances = [];
    constructor() { this.closes = 0; this.outputSampleRate = 44100; AudioRaceStream.instances.push(this); }
    async init(player) {
      this.synth = player;
      entered.resolve();
      await release.promise;
      return this;
    }
    async close() { this.closes++; }
  }
  const dependencies = {
    ...D,
    loadProductionData: async () => ({ kx: new Uint8Array(0), loaderSong: new Uint8Array(0) }),
    parseKX: () => document,
    Runtime: CancellationRuntime,
    Renderer: AudioRaceRenderer,
    AudioStream: AudioRaceStream,
    createV2Player: () => initialPlayer,
  };
  const audioRaceApp = new D.DebrisApp({}, null, {
    dependencies,
    loaderAudio: false,
    playbackWarmup: false,
  });
  const initializing = audioRaceApp.init();
  await entered.promise;
  audioRaceApp.dispose();
  await assert.rejects(initializing, error => error?.name === 'AbortError');
  release.resolve();
  await Promise.resolve();
  assert.equal(rendererConstructs, 1);
  assert.equal(rendererDisposals, 1);
  assert.equal(AudioRaceStream.instances.length, 1);
  assert.equal(AudioRaceStream.instances[0].closes, 1);
  assert.equal(audioRaceApp.ready, false);
  assert.equal(audioRaceApp.audio, null);
  assert.equal(audioRaceApp.player, null);
}

// Seek clamping uses the stream's dynamic sample end, not a stale duration in
// seconds. Both audio and visual replay receive exactly the same target sample.
app.currentSample = 1000;
app.audio.durationSamples = 2000;
let runtimeSeek = -1;
app.seekRuntime = async sample => { runtimeSeek = sample; app.currentSample = sample; return sample; };
assert.equal(await app.seek(100), 2000 / 44100);
assert.equal(runtimeSeek, 2000);
assert.equal(app.audio.seekSeconds, 2000 / 44100);

let stoppedAsEnded = null;
app.stop = ended => { stoppedAsEnded = ended; app.running = false; };
app.running = true;
app.audio.endSample = 66150;
app.audio.onEnded(app.audio);
assert.equal(app.duration, 1.5);
assert.equal(stoppedAsEnded, true);

// Draw-free seek replay drains deferred effect jobs at exact global 30 fps
// boundaries, followed by the non-frame-aligned target sample.
const replaySamples = [];
const replayAspects = [];
let replayRestores = 0;
D.advanceEffectFrame = environment => replaySamples.push(environment.sample);
const replayRuntime = {
  environment: {},
  restore(state) { replayRestores++; this.environment.sample = state.sample || 0; },
  frameAtSample(sample) {
    replayAspects.push(this.environment.aspect);
    this.environment.sample = sample;
  },
};
const replayApp = new D.DebrisApp({ clientWidth: 16, clientHeight: 9 }, null,
  { dependencies: D });
replayApp.runtime = replayRuntime;
replayApp.sampleRate = 1000;
replayApp.snapshots = [{ second: 0, sample: 0, state: { sample: 0 } }];
await replayApp.seekRuntime(350, { yield: false });
assert.deepEqual(replaySamples, [33, 66, 100, 133, 166, 200, 233, 266, 300, 333, 350]);
assert.deepEqual(replayAspects, new Array(11).fill(2),
  'draw-free seek replay retains the authored 2:1 camera');
assert.equal(replayRestores, 0, 'a forward seek continues from the live runtime state');
replaySamples.length = 0;
replayAspects.length = 0;
await replayApp.seekRuntime(500, { yield: false });
assert.deepEqual(replaySamples, [366, 400, 433, 466, 500],
  'a subsequent forward seek replays only the new timeline interval');
assert.deepEqual(replayAspects, new Array(5).fill(2));
assert.equal(replayRestores, 0);
replaySamples.length = 0;
replayRuntime.environment.aspect = 16 / 9;
await replayApp.seekRuntime(500, { yield: false });
assert.deepEqual(replaySamples, [], 'a seek to the live sample performs no replay work');
assert.equal(replayRuntime.environment.aspect, 2,
  'a no-op seek still restores the authored aspect');
replayApp.snapshots.push(
  { second: 0.3, sample: 300, state: { sample: 300 } },
  { second: 0.5, sample: 500, state: { sample: 500 } },
);
await replayApp.seekRuntime(500, { yield: false, forceRestore: true });
assert.equal(replayRestores, 1,
  'diagnostic repetitions can explicitly restore an exact retained state');
await replayApp.seekRuntime(450, { yield: false });
assert.deepEqual(replaySamples, [333, 366, 400, 433, 450],
  'a backward seek replays from its nearest retained snapshot');
assert.equal(replayRestores, 2);
replaySamples.length = 0;
replayRuntime.environment.aspect = 16 / 9;
await replayApp.seekRuntime(0, { yield: false });
assert.equal(replayRestores, 3, 'a backward seek still restores a retained snapshot');
assert.deepEqual(replaySamples, []);
assert.equal(replayRuntime.environment.aspect, 2,
  'an exact-snapshot seek restores the authored aspect without replay steps');

const boundedSnapshots = new D.DebrisApp({ clientWidth: 1, clientHeight: 1 }, null,
  { snapshotLimit: 2, snapshotBudgetBytes: 1024 * 1024 });
boundedSnapshots.addSnapshot(0, 0, { value: 0 });
boundedSnapshots.addSnapshot(30, 30, { value: 1 });
boundedSnapshots.addSnapshot(60, 60, { value: 2 });
assert.deepEqual(Array.from(boundedSnapshots.snapshots, snapshot => snapshot.sample), [0, 60]);
assert.equal(boundedSnapshots.stats.snapshotCount, 2);
boundedSnapshots.initialSnapshot = boundedSnapshots.snapshots[0].state;
boundedSnapshots.dispose();
assert.equal(boundedSnapshots.snapshots.length, 0);
assert.equal(boundedSnapshots.initialSnapshot, null);
const malformedSnapshotOptions = new D.DebrisApp({}, null, {
  snapshotLimit: Number.POSITIVE_INFINITY,
  snapshotBudgetBytes: Number.POSITIVE_INFINITY,
});
assert.equal(malformedSnapshotOptions.snapshotLimit, 18);
assert.equal(malformedSnapshotOptions.snapshotBudgetBytes, 32 * 1024 * 1024);
malformedSnapshotOptions.dispose();

// Renderer/runtime resource scans are deferred and coalesced outside the
// measured render call. The callback publishes the completed scan later.
const maintenanceTasks = [];
const scheduleMaintenance = callback => {
  const task = { callback, cancelled: false };
  maintenanceTasks.push(task);
  return () => { task.cancelled = true; };
};
const runMaintenance = () => {
  const task = maintenanceTasks.shift();
  if (task && !task.cancelled) task.callback();
  return task;
};
let resourceScans = 0;
const resourceRenderOptions = [];
const resourcePublications = [];
const resourceApp = new D.DebrisApp({ clientWidth: 16, clientHeight: 9 }, null, {
  resourceTelemetry: true,
  scheduleMaintenance,
  onFrame(stats, owner, telemetry) {
    resourcePublications.push({
      telemetry: { ...telemetry },
      hasResources: Boolean(stats.resources),
    });
  },
});
resourceApp.runtime = {
  operations: [],
  environment: { lastOutput: {} },
  frameAtSample(sample) {
    assert.equal(this.environment.aspect, 2,
      'the production aspect is fixed before frame graph execution');
    this.environment.sample = sample;
  },
};
resourceApp.renderer = {
  render(output, environment, options) {
    resourceRenderOptions.push({ ...options, aspect: environment.aspect });
    return { drawCalls: 4, triangles: 12, culling: {} };
  },
  resourceStats() { resourceScans++; return { totalEstimatedBytes: 4096 }; },
};
resourceApp.ready = true;
resourceApp.running = true;
resourceApp.sampleRate = 1000;
resourceApp.duration = 100;
resourceApp.renderSample(1000);
assert.equal(resourceScans, 0, 'renderSample never performs the full resource scan synchronously');
assert.equal(resourcePublications.length, 1);
assert.equal(resourcePublications[0].telemetry.frame, true);
assert.equal(resourcePublications[0].hasResources, false);
assert.equal(maintenanceTasks.length, 1, 'resource telemetry schedules one maintenance callback');
resourceApp.renderSample(1016);
assert.deepEqual(resourceRenderOptions, [
  { width: undefined, height: undefined, pixelRatio: 0, presentationAspect: 2, aspect: 2 },
  { width: undefined, height: undefined, pixelRatio: 0, presentationAspect: 2, aspect: 2 },
]);
assert.equal(maintenanceTasks.length, 1, 'pending resource scans coalesce across frames');
runMaintenance();
assert.equal(resourceScans, 1);
assert.equal(resourcePublications.length, 3);
assert.equal(resourcePublications[2].telemetry.resources, true);
assert.equal(resourcePublications[2].hasResources, true);
resourceApp.dispose();

// Fixed-time rendering publishes the frame immediately and then performs one
// explicit complete resource scan outside renderSample for browser-smoke DOM
// telemetry, marking the second publication as forced.
const fixedPublications = [];
let fixedResourceScans = 0;
const fixedApp = new D.DebrisApp({ clientWidth: 2, clientHeight: 1 }, null, {
  onFrame(stats, owner, telemetry) {
    fixedPublications.push({
      telemetry: { ...telemetry },
      hasResources: Boolean(stats.resources),
    });
  },
});
fixedApp.init = async () => { fixedApp.ready = true; return fixedApp; };
fixedApp.debugSample = 1500;
fixedApp.sampleRate = 1000;
fixedApp.runtime = {
  operations: [], environment: { lastOutput: {} },
  frameAtSample(sample) { this.environment.sample = sample; },
};
fixedApp.renderer = {
  render() { return { drawCalls: 2, triangles: 6, culling: {} }; },
  resourceStats() { fixedResourceScans++; return { totalEstimatedBytes: 2048 }; },
};
fixedApp.seekRuntime = async sample => { fixedApp.currentSample = sample; return sample; };
await fixedApp.start();
assert.equal(fixedPublications.length, 2);
assert.equal(fixedPublications[0].telemetry.frame, true);
assert.equal(fixedPublications[0].hasResources, false);
assert.equal(fixedPublications[1].telemetry.resources, true);
assert.equal(fixedPublications[1].telemetry.forceTelemetry, true);
assert.equal(fixedPublications[1].hasResources, true);
assert.equal(fixedResourceScans, 1);
fixedApp.dispose();

// The periodic runtime checkpoint is captured only by deferred maintenance,
// uses the fully rendered current sample, and is cancelled before seek replay.
const snapshotTasks = [];
const scheduleSnapshot = callback => {
  const task = { callback, cancelled: false };
  snapshotTasks.push(task);
  return () => { task.cancelled = true; };
};
const snapshotRuntime = {
  operations: [], environment: { lastOutput: {}, sample: 0 }, snapshots: 0,
  frameAtSample(sample) { this.environment.sample = sample; },
  snapshot() { this.snapshots++; return { sample: this.environment.sample }; },
  restore(state) { this.environment.sample = state.sample || 0; },
};
const snapshotApp = new D.DebrisApp({ clientWidth: 2, clientHeight: 1 }, null, {
  sampleRate: 1000,
  snapshotIntervalSeconds: 30,
  scheduleMaintenance: scheduleSnapshot,
});
snapshotApp.runtime = snapshotRuntime;
snapshotApp.renderer = {
  render() { return { drawCalls: 1, triangles: 1, culling: {} }; },
};
snapshotApp.ready = true;
snapshotApp.running = true;
snapshotApp.duration = 100;
snapshotApp.addSnapshot(0, 0, { sample: 0 });
const savedAnimationFrame = globalThis.requestAnimationFrame;
globalThis.requestAnimationFrame = () => 1;
snapshotApp.currentSample = 31000;
snapshotApp.frame();
assert.equal(snapshotRuntime.snapshots, 0,
  'the 30-second checkpoint cannot block the current RAF callback');
assert.equal(snapshotTasks.length, 1);
snapshotTasks.shift().callback();
assert.equal(snapshotRuntime.snapshots, 1);
assert.equal(snapshotApp.snapshots.at(-1).sample, 31000);
assert.equal(snapshotApp.snapshots.at(-1).state.sample, 31000);
snapshotApp.currentSample = 62000;
assert.equal(snapshotApp.requestRuntimeSnapshot(), true);
const cancelledSnapshot = snapshotTasks[0];
await snapshotApp.seekRuntime(100, { yield: false });
assert.equal(cancelledSnapshot.cancelled, true, 'seek cancels an outstanding checkpoint');
if (!cancelledSnapshot.cancelled) cancelledSnapshot.callback();
assert.equal(snapshotRuntime.snapshots, 1);
globalThis.requestAnimationFrame = savedAnimationFrame;
snapshotApp.dispose();

if (originalRequestAnimationFrame === undefined) delete globalThis.requestAnimationFrame;
else globalThis.requestAnimationFrame = originalRequestAnimationFrame;
if (originalCancelAnimationFrame === undefined) delete globalThis.cancelAnimationFrame;
else globalThis.cancelAnimationFrame = originalCancelAnimationFrame;

console.log('app clock, rate propagation, and render-readback tests passed');
