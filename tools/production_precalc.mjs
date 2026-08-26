#!/usr/bin/env node
// Runs the real Debris party graph without WebGL so geometry/bitmap precalc
// performance can be measured independently from browser rendering.

import { readFile } from 'node:fs/promises';
import { performance } from 'node:perf_hooks';

import {
  Bitmap,
  newBitmap,
  setBitmapFontAdapter,
  setBitmapTextureSizeOffset,
} from '../src/bitmap.js';
import { parseKX } from '../src/kx.js';
import { meshStorageStats } from '../src/mesh.js';
import {
  fontPolygonsToMinMesh,
  minMeshStorageStats,
  setMinMeshFontAdapter,
} from '../src/minmesh.js';
import { createOperatorHandlers } from '../src/operators.js';
import { Runtime } from '../src/runtime.js';
const argumentsMap = new Map();
for (let index = 2; index < process.argv.length; index++) {
  const argument = process.argv[index];
  if (!argument.startsWith('--')) continue;
  const [name, inlineValue] = argument.slice(2).split('=', 2);
  const next = process.argv[index + 1];
  const value = inlineValue ?? (next && !next.startsWith('--') ? process.argv[++index] : true);
  argumentsMap.set(name, value);
}

const numberOption = (name, fallback) => {
  const value = Number(argumentsMap.get(name));
  return Number.isFinite(value) && value >= 0 ? value : fallback;
};
const reportEvery = Math.max(1, numberOption('report-every', 250));
const slowMilliseconds = numberOption('slow-ms', 250);
const limitMilliseconds = numberOption('limit-ms', 0);
const quiet = argumentsMap.has('quiet');
const includeInventory = argumentsMap.has('inventory');
const stubBitmaps = argumentsMap.has('stub-bitmaps');
const asyncMode = argumentsMap.has('async');
const asyncBudgetMilliseconds = numberOption('async-budget-ms', 12);
const requestedTextureOffset = Number(argumentsMap.get('texture-offset') ?? 0);
const textureOffset = Number.isFinite(requestedTextureOffset)
  ? Math.max(-3, Math.min(0, requestedTextureOffset | 0)) : 0;
setBitmapTextureSizeOffset(textureOffset);

const memory = () => {
  const usage = process.memoryUsage();
  const megabytes = value => Math.round(value / 1048576);
  return {
    rssMB: megabytes(usage.rss),
    heapUsedMB: megabytes(usage.heapUsed),
    heapTotalMB: megabytes(usage.heapTotal),
    externalMB: megabytes(usage.external),
    arrayBuffersMB: megabytes(usage.arrayBuffers),
  };
};

const formatEntry = entry => entry
  ? `op ${entry.opId} class 0x${entry.classId.toString(16)} ${entry.handler || '(missing)'}`
  : '(startup)';

function cacheInventory(runtime) {
  const seen = new Set();
  const result = {
    references: 0, objects: 0, meshVertices: 0, meshFaces: 0, bitmapPixels: 0,
    kinds: {}, referenceKinds: {},
  };
  for (const operation of runtime.operations) {
    const cache = operation.cache;
    if (!cache || typeof cache !== 'object') continue;
    const kind = String(cache.kind || cache.type || cache.outputClass || cache.constructor?.name || 'object');
    result.references++;
    result.referenceKinds[kind] = (result.referenceKinds[kind] || 0) + 1;
    if (seen.has(cache)) continue;
    seen.add(cache); result.objects++;
    result.kinds[kind] = (result.kinds[kind] || 0) + 1;
    if (cache.kind === 'mesh' || cache.kind === 'minmesh') {
      const storage = cache.storageSummary?.();
      result.meshVertices += storage?.vertices ?? cache.vertices?.length ?? 0;
      result.meshFaces += storage?.faces ?? cache.faces?.length ?? 0;
    } else if (cache.kind === 'bitmap' || cache.data instanceof Uint16Array) {
      result.bitmapPixels += cache.width * cache.height || (cache.data?.length || 0) / 4;
    }
  }
  return result;
}

function storageInventory(runtime) {
  const mesh = meshStorageStats(runtime) || null;
  const minmesh = minMeshStorageStats(runtime) || null;
  const withMegabytes = value => value && ({
    ...value,
    compactMB: Number(((value.compactBytes || 0) / 1048576).toFixed(3)),
  });
  return { mesh: withMegabytes(mesh), minmesh: withMegabytes(minmesh) };
}

const kx = new Uint8Array(await readFile(new URL('../assets/debris_party.kx', import.meta.url)));
const document = parseKX(kx);

const handlers = createOperatorHandlers();
if (stubBitmaps) {
  for (const id of [
    0x21, 0x22, 0x23, 0x24, 0x25, 0x27, 0x29, 0x2a, 0x2b,
    0x2c, 0x2d, 0x2e, 0x30, 0x31, 0x32, 0x34, 0x35, 0x36,
    0x38, 0x39, 0x3a, 0x3b, 0x3d, 0x3e, 0x3f,
  ]) handlers.set(id, () => new Bitmap(1, 1));
}

// The two production Bitmap_Text calls only feed title textures. Node has no
// Canvas implementation, so use transparent glyph coverage for this CPU-only
// diagnostic. Browser runs still use the real Canvas adapter.
setBitmapFontAdapter(request => ({
  coverage: new Uint8Array(request.width * request.height),
  lineHeight: Math.max(1, Math.round(request.textHeight * request.height)),
}));

// This process-only graph profiler has no browser Canvas. Keep its Font3D
// dependency explicit and cheap; browser playback never installs this stub.
setMinMeshFontAdapter(({ height, extrude, text }) => {
  const width = Math.max(height * 0.25, String(text).length * height * 0.5);
  return fontPolygonsToMinMesh([{
    outer: [[0, 0], [0, height], [width, height], [width, 0]], holes: [],
  }], extrude);
});

let calls = 0;
let previousEntry = null;
let previousStarted = performance.now();
const started = previousStarted;
let lastReport = started;
let slowest = { milliseconds: 0, entry: null };
let asyncYields = 0;
let finalAsyncProgress = null;
let asyncYieldMilliseconds = 0;
const classCalls = {};

const runtime = new Runtime(document, {
  strictHandlers: true,
  handlers,
  onHandlerCall(call, phase, callback, entry) {
    if (phase !== 'init') return;
    const now = performance.now();
    const previousMilliseconds = now - previousStarted;
    if (previousEntry && previousMilliseconds > slowest.milliseconds) {
      slowest = { milliseconds: previousMilliseconds, entry: previousEntry };
    }
    if (!quiet && previousEntry && previousMilliseconds >= slowMilliseconds) {
      console.error(`slow ${previousMilliseconds.toFixed(1)} ms: ${formatEntry(previousEntry)}`);
    }
    calls++;
    const classKey = `0x${entry.classId.toString(16)}`;
    classCalls[classKey] = (classCalls[classKey] || 0) + 1;
    if (!quiet && (calls % reportEvery === 0 || now - lastReport >= 5000)) {
      console.error(JSON.stringify({
        calls,
        elapsedSeconds: Number(((now - started) / 1000).toFixed(2)),
        current: formatEntry(entry),
        memory: memory(),
        ...(includeInventory ? { inventory: cacheInventory(runtime) } : {}),
      }));
      lastReport = now;
    }
    if (limitMilliseconds && now - started > limitMilliseconds) {
      const error = new Error(`production precalc exceeded ${limitMilliseconds} ms before ${formatEntry(entry)}`);
      error.code = 'PRECALC_TIME_LIMIT';
      throw error;
    }
    previousEntry = entry;
    previousStarted = now;
  },
});

// Bitmap_Render normally precalculates an IPP subtree into a WebGL texture.
// A correctly sized blank bitmap preserves all graph/data dependencies while
// keeping this diagnostic independent from a GPU context.
runtime.bitmapRendererHook = record => newBitmap(record.widthExponent | 0, record.heightExponent | 0);

try {
  if (asyncMode) {
    await runtime.precalcAsync(0, {
      budgetMilliseconds: asyncBudgetMilliseconds,
      onProgress(progress) { finalAsyncProgress = progress; },
      async yield() {
        const yieldStarted = performance.now();
        await new Promise(resolve => setImmediate(resolve));
        const duration = performance.now() - yieldStarted;
        asyncYieldMilliseconds += duration;
        // Handler timing starts in onHandlerCall and would otherwise charge
        // the scheduler delay to the preceding operator.
        previousStarted += duration;
        asyncYields++;
      },
    });
  } else runtime.precalc();
  const ended = performance.now();
  const finalMilliseconds = ended - previousStarted;
  if (previousEntry && finalMilliseconds > slowest.milliseconds) {
    slowest = { milliseconds: finalMilliseconds, entry: previousEntry };
  }
  console.log(JSON.stringify({
    ok: true,
    operations: document.operations.length,
    handlerCalls: calls,
    elapsedSeconds: Number(((ended - started) / 1000).toFixed(3)),
    mode: asyncMode ? 'async' : 'sync',
    asyncBudgetMilliseconds: asyncMode ? asyncBudgetMilliseconds : null,
    asyncYields,
    asyncYieldMilliseconds: Number(asyncYieldMilliseconds.toFixed(3)),
    asyncProgress: finalAsyncProgress,
    textureOffset,
    stubBitmaps,
    classCalls,
    slowest: {
      milliseconds: Number(slowest.milliseconds.toFixed(3)),
      operation: formatEntry(slowest.entry),
    },
    memory: memory(),
    ...(includeInventory ? { inventory: cacheInventory(runtime) } : {}),
    storage: storageInventory(runtime),
    rootKind: runtime.root?.cache?.kind || runtime.root?.cache?.type || runtime.root?.cache?.outputClass || null,
    root: runtime.root?.cache?.summary?.() || runtime.root?.cache?.type || null,
  }, null, 2));
} catch (error) {
  console.error(JSON.stringify({
    ok: false,
    error: error?.stack || String(error),
    handlerCalls: calls,
    elapsedSeconds: Number(((performance.now() - started) / 1000).toFixed(3)),
    mode: asyncMode ? 'async' : 'sync',
    asyncBudgetMilliseconds: asyncMode ? asyncBudgetMilliseconds : null,
    asyncYields,
    asyncYieldMilliseconds: Number(asyncYieldMilliseconds.toFixed(3)),
    asyncProgress: finalAsyncProgress,
    textureOffset,
    stubBitmaps,
    classCalls,
    lastCompleted: formatEntry(previousEntry),
    slowest: {
      milliseconds: Number(slowest.milliseconds.toFixed(3)),
      operation: formatEntry(slowest.entry),
    },
    memory: memory(),
    ...(includeInventory ? { inventory: cacheInventory(runtime) } : {}),
    storage: storageInventory(runtime),
  }, null, 2));
  process.exitCode = 1;
}
