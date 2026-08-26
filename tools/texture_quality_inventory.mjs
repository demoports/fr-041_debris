#!/usr/bin/env node
// Static texture-work inventory for the released Debris KX. This reads only
// compact metadata and bitmap dimensions; it never evaluates the production
// graph, allocates bitmap pixels, or starts a browser.

import { readFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';

import { CLASS_REGISTRY } from '../src/classes.js';
import { parseKX } from '../src/kx.js';

const BITMAP_OUTPUT_CLASS = 'KC_BITMAP';
const PLAYBACK_CACHE_OUTPUT_CLASSES = new Set(['KC_BITMAP', 'KC_MESH', 'KC_MINMESH']);
const RGBA16_BYTES_PER_PIXEL = 4 * Uint16Array.BYTES_PER_ELEMENT;
const MEBIBYTE = 1024 * 1024;
const SIZED_BITMAP_CLASSES = new Set([0x21, 0x22, 0x32, 0x34, 0x3d, 0x3f]);
// Keep this static audit aligned with bitmap.js' runtime-only writable map.
// Values are handler input slots, not filtered bitmap-input ordinals.
const RUNTIME_WRITABLE_BITMAP_INPUTS = new Map([
  [0x23, 0], [0x24, 0], [0x27, 0], [0x29, 0], [0x2a, 0], [0x2b, 0],
  [0x2d, 1], [0x30, 0], [0x31, 0], [0x35, 0], [0x36, 0], [0x39, 0],
  [0x3b, 0],
]);

const QUALITY_SETTINGS = Object.freeze([
  Object.freeze({ name: 'high', sourceSizeOffset: 0 }),
  Object.freeze({ name: 'medium', sourceSizeOffset: -1 }),
  Object.freeze({ name: 'low', sourceSizeOffset: -2 }),
]);

function isBitmapOperation(document, operationId) {
  const operation = document.operations[operationId];
  return CLASS_REGISTRY[operation?.classId]?.outputClass === BITMAP_OUTPUT_CLASS;
}

function bitmapGeneratorDimensions(operation, sourceSizeOffset) {
  const encodedX = operation.parameters[0] | 0;
  const dontScale = (encodedX & 0x80) !== 0;
  let xExponent = encodedX & 0x7f;
  let yExponent = operation.parameters[1] | 0;
  if (!dontScale) {
    xExponent += sourceSizeOffset;
    yExponent += sourceSizeOffset;
  }
  xExponent = Math.max(0, Math.min(12, xExponent));
  yExponent = Math.max(0, Math.min(12, yExponent));
  return [2 ** xExponent, 2 ** yExponent];
}

function jpegDimensions(bytes) {
  const startOfFrame = new Set([
    0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7,
    0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf,
  ]);
  for (let offset = 0; offset + 8 < (bytes?.length || 0); offset++) {
    if (bytes[offset] !== 0xff || !startOfFrame.has(bytes[offset + 1])) continue;
    const height = (bytes[offset + 5] << 8) | bytes[offset + 6];
    const width = (bytes[offset + 7] << 8) | bytes[offset + 8];
    if (width > 0 && height > 0) return [width, height];
  }
  throw new Error('production Bitmap_Import is not a dimension-bearing JPEG');
}

function lowerPowerOfTwo(value) {
  return 2 ** Math.floor(Math.log2(value));
}

function referencedOperationId(value) {
  if (Number.isInteger(value)) return value;
  if (Number.isInteger(value?.id)) return value.id;
  if (Number.isInteger(value?.operation)) return value.operation;
  if (Number.isInteger(value?.op?.id)) return value.op.id;
  return null;
}

// Mirror pruneImmutablePlaybackCaches' execution-boundary walk. Dynamic
// operators, roots, and events retain the first Bitmap/Mesh/MinMesh cache on
// every path. The later captured-object scan adds no further bitmap identity in
// the released graph: all texture-bearing Material operators are dynamic and
// are therefore already roots of this walk.
function playbackBitmapBoundaryOperations(document) {
  const bitmapOperations = new Set(), visited = new Set();
  const visit = value => {
    const operationId = referencedOperationId(value);
    if (operationId === null || visited.has(operationId)) return;
    const operation = document.operations[operationId];
    if (!operation) return;
    visited.add(operationId);
    const outputClass = CLASS_REGISTRY[operation.classId]?.outputClass;
    if (PLAYBACK_CACHE_OUTPUT_CLASSES.has(outputClass)) {
      if (outputClass === BITMAP_OUTPUT_CLASS) bitmapOperations.add(operationId);
      return;
    }
    for (const input of operation.inputs || []) visit(input);
    for (const link of operation.links || []) visit(link);
  };

  for (const operation of document.operations) {
    if (CLASS_REGISTRY[operation.classId]?.dynamic !== false) visit(operation.id);
  }
  for (const root of document.roots || []) visit(root);
  for (const event of document.events || []) visit(event.operation ?? event.op);
  return bitmapOperations;
}

// Bitmap handlers transfer a safe designated final-consumed input in place;
// other operators retain the older post-result adoption fallback. Reproduce
// that identity ownership without allocating pixels so aliased boundary
// operations count the same physical RGBA16 cache only once.
function bitmapStorageIdentities(document, onWritableReuse = null) {
  const consumerCounts = new Map();
  const addConsumer = value => {
    const operationId = referencedOperationId(value);
    if (operationId !== null && document.operations[operationId]) {
      consumerCounts.set(operationId, (consumerCounts.get(operationId) || 0) + 1);
    }
  };
  for (const operation of document.operations) {
    for (const input of operation.inputs || []) addConsumer(input);
    for (const link of operation.links || []) addConsumer(link);
  }
  for (const root of document.roots || []) addConsumer(root);
  for (const event of document.events || []) addConsumer(event.operation ?? event.op);

  const remaining = new Map(consumerCounts), operationIdentities = new Map();
  for (const operation of document.operations) {
    if (!isBitmapOperation(document, operation.id)) continue;
    const owned = [];
    // Bitmap_Render bypasses finishBitmapCall because its deferred render may
    // retain its input. Its released-production input is IPP in any case.
    if (operation.classId !== 0x3f) {
      for (const input of operation.inputs || []) {
        if (!isBitmapOperation(document, input)) continue;
        const count = Math.max(0, (remaining.get(input) || 0) - 1);
        remaining.set(input, count);
        if (!count) {
          const identity = operationIdentities.get(input);
          if (identity && !owned.includes(identity)) owned.push(identity);
        }
      }
    }

    let writableIdentity = null;
    const writableIndex = RUNTIME_WRITABLE_BITMAP_INPUTS.get(operation.classId);
    if (writableIndex !== undefined) {
      const writableOperationId = referencedOperationId(operation.inputs[writableIndex]);
      const candidate = operationIdentities.get(writableOperationId);
      const inputIdentities = operation.inputs
        .filter(input => isBitmapOperation(document, referencedOperationId(input)))
        .map(input => operationIdentities.get(referencedOperationId(input)));
      const duplicate = candidate && inputIdentities.some((identity, index) =>
        identity === candidate && inputIdentities.indexOf(identity) !== index);
      // Bump and RotateMul return before allocating for these degenerate
      // parameters, so there is no source copy for the runtime path to avoid.
      const producesBitmap = operation.classId !== 0x30 || operation.parameters[8] !== 0;
      const rotateMulRuns = operation.classId !== 0x36 || operation.parameters[8] > 0;
      if (candidate && owned.includes(candidate) && !duplicate && producesBitmap && rotateMulRuns) {
        writableIdentity = candidate;
        onWritableReuse?.({ operation, inputOperationId: writableOperationId });
      }
    }

    let identity = writableIdentity || owned[0];
    if (!identity) identity = { active: true, operationId: operation.id };
    else {
      identity.active = true;
      identity.operationId = operation.id;
      for (const deadIdentity of owned) {
        if (deadIdentity !== identity) deadIdentity.active = false;
      }
    }
    operationIdentities.set(operation.id, identity);
  }
  return operationIdentities;
}

function textureQualityInventory(document, sourceSizeOffset) {
  if (!Number.isInteger(sourceSizeOffset) || sourceSizeOffset > 0 || sourceSizeOffset < -3) {
    throw new RangeError('source texture-size offset must be an integer from -3 through 0');
  }

  const dimensions = new Map();
  const visiting = new Set();
  const resolveDimensions = operationId => {
    if (dimensions.has(operationId)) return dimensions.get(operationId);
    if (visiting.has(operationId)) throw new Error(`bitmap dimension cycle at op ${operationId}`);
    visiting.add(operationId);
    const operation = document.operations[operationId];
    if (!operation || !isBitmapOperation(document, operationId)) {
      visiting.delete(operationId);
      return null;
    }

    let result = null;
    if (SIZED_BITMAP_CLASSES.has(operation.classId)) {
      result = bitmapGeneratorDimensions(operation, sourceSizeOffset);
    } else if (operation.classId === 0x3a) {
      // LoadBitmapCore crops imports to the next lower powers of two. Debris'
      // sole embedded import is already a 16x16 JPEG.
      const [sourceWidth, sourceHeight] = jpegDimensions(operation.blob);
      result = [lowerPowerOfTwo(sourceWidth), lowerPowerOfTwo(sourceHeight)];
    } else {
      const bitmapInputs = operation.inputs.filter(input =>
        isBitmapOperation(document, input));
      // Distort writes a copy of its displacement (second) bitmap. Authored
      // Debris dimensions match, but selecting the actual output makes this
      // inventory mirror the implementation rather than relying on that fact.
      const sourceIndex = operation.classId === 0x2d ? 1 : 0;
      const sourceOperation = bitmapInputs[sourceIndex] ?? bitmapInputs[0];
      result = sourceOperation === undefined ? null : resolveDimensions(sourceOperation);

      // Transform can request a new power-of-two output independently on each
      // axis. Its stored exponent is one above newBitmap's generator exponent.
      if (result && operation.classId === 0x2c) {
        const newWidth = operation.parameters[6] | 0;
        const newHeight = operation.parameters[7] | 0;
        const exponent = value => Math.max(0, value - 1 + sourceSizeOffset);
        result = [
          newWidth ? 2 ** exponent(newWidth) : result[0],
          newHeight ? 2 ** exponent(newHeight) : result[1],
        ];
      }
    }

    visiting.delete(operationId);
    if (!result) throw new Error(`cannot infer bitmap dimensions for op ${operationId}`);
    dimensions.set(operationId, result);
    return result;
  };

  let bitmapOperators = 0;
  let rgba16Pixels = 0;
  const dimensionCounts = {};
  for (const operation of document.operations) {
    if (!isBitmapOperation(document, operation.id)) continue;
    bitmapOperators++;
    const [width, height] = resolveDimensions(operation.id);
    const pixels = width * height;
    rgba16Pixels += pixels;
    const key = `${width}x${height}`;
    dimensionCounts[key] = (dimensionCounts[key] || 0) + 1;
  }

  const rgba16OutputBytes = rgba16Pixels * RGBA16_BYTES_PER_PIXEL;
  const retainedBoundaryOperations = playbackBitmapBoundaryOperations(document);
  let inPlaceCopyBytesAvoided = 0;
  let inPlaceReuseCount = 0;
  const inPlaceBytesByClass = {};
  const storageIdentities = bitmapStorageIdentities(document, ({ operation, inputOperationId }) => {
    const [width, height] = resolveDimensions(inputOperationId);
    const bytes = width * height * RGBA16_BYTES_PER_PIXEL;
    inPlaceCopyBytesAvoided += bytes;
    inPlaceReuseCount++;
    const key = `0x${operation.classId.toString(16)}`;
    inPlaceBytesByClass[key] = (inPlaceBytesByClass[key] || 0) + bytes;
  });
  const retainedIdentities = new Set();
  for (const operationId of retainedBoundaryOperations) {
    const identity = storageIdentities.get(operationId);
    if (identity?.active) retainedIdentities.add(identity);
  }
  let retainedRgba16Pixels = 0;
  for (const identity of retainedIdentities) {
    const [width, height] = resolveDimensions(identity.operationId);
    retainedRgba16Pixels += width * height;
  }
  const retainedRgba16Bytes = retainedRgba16Pixels * RGBA16_BYTES_PER_PIXEL;
  return {
    sourceSizeOffset,
    bitmapOperators,
    rgba16Pixels,
    rgba16OutputBytes,
    roundedOutputMiB: Math.round(rgba16OutputBytes / MEBIBYTE),
    retainedBitmapBoundaryOperations: retainedBoundaryOperations.size,
    retainedBitmapIdentities: retainedIdentities.size,
    retainedRgba16Pixels,
    retainedRgba16Bytes,
    roundedRetainedMiB: Math.round(retainedRgba16Bytes / MEBIBYTE),
    inPlaceReuseCount,
    inPlaceCopyBytesAvoided,
    inPlaceCopyMiB: inPlaceCopyBytesAvoided / MEBIBYTE,
    roundedInPlaceCopyMiB: Math.round(inPlaceCopyBytesAvoided / MEBIBYTE),
    inPlaceBytesByClass,
    dimensionCounts,
  };
}

async function productionTextureQualityInventory() {
  const bytes = await readFile(new URL('../assets/debris_party.kx', import.meta.url));
  const document = parseKX(bytes);
  return Object.fromEntries(QUALITY_SETTINGS.map(({ name, sourceSizeOffset }) => [
    name,
    textureQualityInventory(document, sourceSizeOffset),
  ]));
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const inventory = await productionTextureQualityInventory();
  console.log('quality\toffset\tretained RGBA16 bitmap caches\tcumulative bitmap output\tcopies avoided');
  for (const { name } of QUALITY_SETTINGS) {
    const result = inventory[name];
    console.log(`${name}\t${result.sourceSizeOffset}\t` +
      `${result.roundedRetainedMiB.toLocaleString('en-US')} MiB\t` +
      `${result.roundedOutputMiB.toLocaleString('en-US')} MiB\t` +
      `${result.roundedInPlaceCopyMiB.toLocaleString('en-US')} MiB`);
  }
  console.log('Width x height x four 16-bit channels; not download size, total memory, or GPU residency.');
}

export {
  QUALITY_SETTINGS,
  RGBA16_BYTES_PER_PIXEL,
  productionTextureQualityInventory,
  textureQualityInventory,
};
