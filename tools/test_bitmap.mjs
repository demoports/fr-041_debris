import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';

import { parseKX } from '../src/kx.js';
import * as BitmapAPI from '../src/bitmap.js';

const handlers = new Map(Object.entries(BitmapAPI.bitmapHandlers)
  .map(([id, handler]) => [Number(id), handler]));
const D = { ...BitmapAPI, parseKX, handlers };
const bitmapWordSha256 = bitmap => {
  // Native bitmap words are little-endian. Serialize that order explicitly so
  // the source-oracle hashes do not depend on the host CPU's byte order.
  const bytes = new Uint8Array(bitmap.data.length * 2);
  for (let index = 0; index < bitmap.data.length; index++) {
    const word = bitmap.data[index];
    bytes[index * 2] = word & 0xff;
    bytes[index * 2 + 1] = word >>> 8;
  }
  return createHash('sha256').update(bytes).digest('hex');
};
const bitmapIds = [
  0x21, 0x22, 0x23, 0x24, 0x25, 0x27, 0x29, 0x2a, 0x2b,
  0x2c, 0x2d, 0x2e, 0x30, 0x31, 0x32, 0x34, 0x35, 0x36,
  0x38, 0x39, 0x3a, 0x3b, 0x3d, 0x3e, 0x3f,
];
assert.equal(bitmapIds.filter(id => D.handlers.has(id)).length, 25);

assert.deepEqual([...D.packedColor(0xff123456)], [0x909, 0x1a1a, 0x2b2b, 0x7fff]);
assert.equal(D.clamp15(-1), 0);
assert.equal(D.clamp15(0x8000), 0x7fff);

const flat = D.Bitmap_Flat(2, 1, 0xff204080);
assert.equal(flat.width, 4);
assert.equal(flat.height, 2);
assert.deepEqual([...flat.data.slice(0, 4)], [...D.packedColor(0xff204080)]);
const flatCopy = flat.copy();
flatCopy.data[0] = 1;
assert.notEqual(flat.data[0], flatCopy.data[0]);
assert.deepEqual(flat.summary(), flat.summary());
assert.notEqual(flat.summary().hash, flatCopy.summary().hash);

const quad = new D.Bitmap(2, 2);
for (let c = 0; c < 4; c++) {
  quad.data[c] = 0;
  quad.data[4 + c] = 10000;
  quad.data[8 + c] = 20000;
  quad.data[12 + c] = 30000;
}
const context = new D.BilinearContext(quad, 0);
assert.deepEqual([...context.sample(0x8000, 0x8000)], [15000, 15000, 15000, 15000]);
// BilinearFilter's released MMX path halves each fraction, takes pmulhw of
// signed word deltas, and doubles the result. A conventional weighted sum
// produces 8374 here; the source instruction sequence produces 8372.
assert.deepEqual([...context.sample(0x7001, 0x3333)], [8372, 8372, 8372, 8372]);
// BilinearSetup's border bits select clamp, not wrap: a clear X bit masks the
// coordinate by XMax, while a set X bit preserves it for the signed clamp.
assert.deepEqual([...context.sample(-0x10000, 0)], [10000, 10000, 10000, 10000]);
assert.deepEqual([...new D.BilinearContext(quad, 1).sample(-0x10000, 0)], [0, 0, 0, 0]);
assert.deepEqual([...new D.BilinearContext(quad, 2).sample(0, -0x10000)], [0, 0, 0, 0]);
assert.deepEqual([...new D.BilinearContext(quad, 0).sample(0, -0x10000)], [20000, 20000, 20000, 20000]);
// PointFilter has the source's deliberate always-wrap behavior.
assert.deepEqual([...context.point(-0x10000, 0)], [10000, 10000, 10000, 10000]);

const white = D.Bitmap_Flat(1, 1, 0xffffffff);
const red = D.Bitmap_Flat(1, 1, 0xffff0000);
const merged = D.Bitmap_Merge(0, [red, white]);
assert.deepEqual([...merged.data.slice(0, 4)], [0x7fff, 0x7fff, 0x7fff, 0x7fff]);
assert.deepEqual([...red.data.slice(0, 4)], [...D.packedColor(0xffff0000)]);

// BI_GRAY and BI_ALPHA use two staged MMX word averages, not one scalar
// (r+2g+b)/4 expression. These values are source-oracle cases where the two
// formulas differ, and ALPHA's pand also clears retained high color bits.
{
  const grayDestination = new Uint16Array(4);
  D.bitmapInner(grayDestination, grayDestination, D.BI.GRAY,
    new Uint16Array([3, 0, 1, 0xffff]));
  assert.deepEqual([...grayDestination], [0, 0, 0, 0x7fff]);

  const alphaDestination = new Uint16Array(4);
  D.bitmapInner(alphaDestination, new Uint16Array([7, 2, 5, 0]), D.BI.ALPHA,
    new Uint16Array([0x8001, 0x9234, 0xffff, 0]));
  assert.deepEqual([...alphaDestination], [1, 0x1234, 0x7fff, 3]);

  const signedScaleSource = new D.Bitmap(1, 1);
  signedScaleSource.data.fill(0xffff);
  assert.deepEqual([...D.Bitmap_Color(signedScaleSource, 5, 0x01010101).data],
    [0x7fff, 0x7fff, 0x7fff, 0x7fff],
    'BI_SCALECOL keeps native psrld-before-pack behavior for negative words');
}

const ranged = D.Bitmap_Range(D.Bitmap_Flat(1, 1, 0xff808080), 0, 0xff000000, 0xffffffff);
assert.ok(ranged.data[0] >= 0x3f00 && ranged.data[0] <= 0x4100);
assert.equal(ranged.data[0], ranged.data[1]);

const identity = D.Bitmap_Rotate(quad, 0, 1, 1, 0.5, 0.5, 0, 0, 0);
assert.deepEqual([...identity.data], [...quad.data]);
quad.format = quad.Format = 9;
quad.texMipCount = quad.TexMipCount = 4;
quad.texMipThreshold = quad.TexMipTresh = 18;
quad.stripped = quad.Stripped = true;
const metadataIdentity = D.Bitmap_Rotate(quad, 0, 1, 1, 0.5, 0.5, 0, 0, 0);
assert.deepEqual(
  [metadataIdentity.format, metadataIdentity.texMipCount,
    metadataIdentity.texMipThreshold, metadataIdentity.stripped],
  [9, 4, 18, true],
  'Rotate changes native pixel storage without discarding texture policy',
);
// Pin Bitmap_Rotate's inlined copy of the same non-midpoint MMX kernel.
const translatedMMX = D.Bitmap_Rotate(quad, 0, 1, 1,
  (0x10000 + 0x7001) / (2 * 0x10000),
  (0x10000 + 0x3333) / (2 * 0x10000), 0, 0, 0);
assert.deepEqual([...translatedMMX.data.slice(0, 4)], [8372, 8372, 8372, 8372]);
const clampedQuarter = D.Bitmap_Rotate(quad, 0, 0.25, 0.25, 0.125, 0.125, 3, 0, 0);
const wrappedQuarter = D.Bitmap_Rotate(quad, 0, 0.25, 0.25, 0.125, 0.125, 0, 0, 0);
assert.deepEqual([...clampedQuarter.data.slice(0, 4)], [0, 0, 0, 0]);
assert.notDeepEqual([...wrappedQuarter.data.slice(0, 4)], [0, 0, 0, 0]);
const bulgedIdentity = D.Bitmap_Bulge(quad, 0);
assert.deepEqual([...bulgedIdentity.data], [...quad.data]);

const displacement = D.Bitmap_Flat(1, 1, 0xff808000);
const sample = D.Bitmap_Flat(1, 1, 0xff336699);
const distorted = D.Bitmap_Distort(sample, displacement, 1, 3);
assert.equal(distorted.width, sample.width);
assert.deepEqual([...distorted.data.slice(0, 4)], [...sample.data.slice(0, 4)]);

// Distort's source only requires equal pixel counts. BilinearSetup uses the
// displacement shape, so a differently shaped sample is reinterpreted as the
// same flat native sU64 buffer rather than sampled using its own metadata.
{
  const wideSample = new D.Bitmap(4, 2);
  for (let pixel = 0; pixel < wideSample.size; pixel++) {
    wideSample.data.fill(pixel * 1000 + 17, pixel * 4, pixel * 4 + 4);
  }
  const tallDisplacement = new D.Bitmap(2, 4);
  tallDisplacement.data.fill(0x4000);
  const reinterpreted = D.Bitmap_Distort(wideSample, tallDisplacement, 0, 0);
  assert.deepEqual([reinterpreted.width, reinterpreted.height], [2, 4]);
  assert.deepEqual([...reinterpreted.data], [...wideSample.data]);
}

const normal = D.Bitmap_Normals(D.Bitmap_Flat(2, 2, 0xff808080), 1, 1);
assert.deepEqual([...normal.data.slice(0, 4)], [0x4000, 0x4000, 0x7fff, 0xffff]);

// Native Bitmap_Normals uses x87 round-to-nearest-even for dist, then a
// wrapped signed 32-bit multiply and arithmetic right shift for each slope.
// These hand-authored height fields make each distinction observable without
// reproducing the implementation in the test.
{
  const tieDistance = new D.Bitmap(16, 1);
  tieDistance.data[(16 - 1) * 4 + 2] = 0x7fff;
  assert.deepEqual(
    [...D.Bitmap_Normals(tieDistance, 15.5 / 65536, 4).data.slice(0, 4)],
    [0x4001, 0x4000, 0x4000, 0xffff],
    'sFtol rounds a half-way distance to the even integer 16',
  );
  assert.deepEqual(
    [...D.Bitmap_Normals(tieDistance, 15.5 / 65536, 5).data.slice(0, 4)],
    [0x4001, 0x4000, 0x7fff, 0xffff],
    'normal Z rounds through x87 single precision before integer truncation',
  );

  const negativeSlope = new D.Bitmap(4, 1);
  negativeSlope.data[2] = 101;
  negativeSlope.data[(4 - 1) * 4 + 2] = 100;
  assert.deepEqual(
    [...D.Bitmap_Normals(negativeSlope, 1, 4).data.slice(0, 4)],
    [0x3fff, 0x4000, 0x4000, 0xffff],
    'SAR rounds a small negative fixed-point slope down to -1',
  );

  const wrappedProduct = new D.Bitmap(4, 1);
  wrappedProduct.data[(4 - 1) * 4 + 2] = 0x7fff;
  assert.deepEqual(
    [...D.Bitmap_Normals(wrappedProduct, 8, 4).data.slice(0, 4)],
    [0x3fff, 0x4000, 0x4000, 0xffff],
    'the signed 32-bit slope product wraps before its arithmetic shift',
  );

  // At (0,0) these heights produce the pre-normalized native slopes
  // (-16381,-1635). The released sF32 scale yields (-16302,-1627), one X
  // unit away from an all-double calculation.
  const saturatedSlope = new D.Bitmap(32, 32);
  saturatedSlope.data[2] = 32761;
  saturatedSlope.data[((31 * 32) + 0) * 4 + 2] = 29491;
  assert.deepEqual(
    [...D.Bitmap_Normals(saturatedSlope, 1, 5).data.slice(0, 4)],
    [82, 14757, 0x4000, 0xffff],
    'fallback normalization retains the native sF32 scale and products',
  );

  const channelOrder = new D.Bitmap(4, 1);
  const blueHeights = [0, 1000, 3000, 7000];
  for (let x = 0; x < 4; x++) {
    channelOrder.data[x * 4] = 777; // constant red must not affect height
    channelOrder.data[x * 4 + 1] = 123;
    channelOrder.data[x * 4 + 2] = blueHeights[x];
  }
  assert.deepEqual(
    [...D.Bitmap_Normals(channelOrder, 1, 0).data],
    [16743, 16384, 16384, 65535, 16399, 16384, 16384, 65535,
      16180, 16384, 16384, 65535, 16212, 16384, 16384, 65535],
    'native word zero is blue in BGRA, hence RGBA height generation reads lane 2',
  );
}

const perlinA = D.Bitmap_Perlin(3, 3, 1, 3, 0.5, 17, 0, 1, 1, 0xff000000, 0xffffffff);
const perlinB = D.Bitmap_Perlin(3, 3, 1, 3, 0.5, 17, 0, 1, 1, 0xff000000, 0xffffffff);
assert.deepEqual(perlinA.summary(), perlinB.summary());
assert.notEqual(perlinA.data[0], perlinA.data[4]);
// Generated independently by tools/oracles/perlin_oracle.cpp from the
// released #if 1 row/group path. Together these pin ordinary, absolute,
// sine-shaped, and absolute+sine noise plus non-square row strides.
assert.equal(perlinA.summary().hash, 0xdd38ecb4);
assert.equal(D.Bitmap_Perlin(4, 3, 1, 4, 0.625, 93, 1, 0.75, 1.25,
  0xff102030, 0xffe0c080).summary().hash, 0xdf2d5035);
assert.equal(D.Bitmap_Perlin(4, 3, 2, 3, 0.75, 7, 2, 1.125, 0.8,
  0x80402010, 0xffd0e0f0).summary().hash, 0x4b1a17aa);
assert.equal(D.Bitmap_Perlin(4, 3, 2, 3, 0.75, 7, 3, 1.125, 0.8,
  0x80402010, 0xffd0e0f0).summary().hash, 0xcbd5b91e);

const cellA = D.Bitmap_Cell(3, 3, 0xffffffff, 0xff000000, 0xffff0000, 12, 9, 1, 1, 1, 0, 64, 0);
const cellB = D.Bitmap_Cell(3, 3, 0xffffffff, 0xff000000, 0xffff0000, 12, 9, 1, 1, 1, 0, 64, 0);
assert.deepEqual(cellA.summary(), cellB.summary());
assert.equal(bitmapWordSha256(cellA),
  '1708418c72d03d42abfb549cf17ddabd7e0f2467d94bcefcc68c887a3c633822',
  'tiny Cell direct calls retain the exhaustive scalar fallback exactly');
const fallbackRectCells = D.Bitmap_Cell(4, 5, 0xffe0a060, 0xff102030, 0xff804020,
  90, 5, 0.8, 1.2, 7, 0.0625, 37, -1.25);
assert.equal(bitmapWordSha256(fallbackRectCells),
  '1b84141b92de04eb1f57a14ede9dcea4bb76d662afcf44561b48b399b7713aef',
  'a complete 16-pixel tile uses the released sorted-tile Cell path');

const gradient = D.Bitmap_Gradient(2, 2, 0xff000000, 0xffffffff, 0, 0, 1, 0);
assert.ok(gradient.data[0] < gradient.data[(gradient.width - 1) * 4]);

const bricksA = D.Bitmap_Bricks(3, 3, 0xff804020, 0xffc08040, 0xff101010,
  0.25, 0.25, 4, 4, 3, 100, 0, 0.25, 1);
const bricksB = D.Bitmap_Bricks(3, 3, 0xff804020, 0xffc08040, 0xff101010,
  0.25, 0.25, 4, 4, 3, 100, 0, 0.25, 1);
assert.deepEqual(bricksA.summary(), bricksB.summary());

const sourceBeforeText = D.Bitmap_Flat(1, 1, 0xff000000);
let textAdapterCalls = 0;
D.setBitmapFontAdapter(request => {
  textAdapterCalls++;
  const coverage = new Uint8Array(request.width * request.height);
  coverage[0] = 255;
  return { coverage };
});
const text = D.Bitmap_Text({}, {}, sourceBeforeText, 0, 0, 0.25, 0.25,
  0xffffffff, 0, 2, 0, 1, 'x', 'test');
assert.equal(textAdapterCalls, 1);
assert.deepEqual([...text.data.slice(0, 4)], [0x7ffe, 0x7ffe, 0x7ffe, 0x7ffe]);
assert.deepEqual([...sourceBeforeText.data.slice(0, 4)], [0, 0, 0, 0x7fff]);

// FontBegin receives explicit X then Y sizes: Bitmap_Text `height` supplies
// glyph width and `width` supplies glyph/line height. Also retain its
// extspace*4 + two-output-pixel origin on the 4x surface.
{
  const canvasCalls = [];
  const previousOffscreenCanvas = globalThis.OffscreenCanvas;
  globalThis.OffscreenCanvas = class {
    constructor(width, height) { this.width = width; this.height = height; }
    getContext() {
      return {
        clearRect() {}, fillStyle: '', textBaseline: '', font: '',
        measureText(value) {
          return {
            width: String(value).length * 4,
            fontBoundingBoxAscent: 6,
            fontBoundingBoxDescent: 3,
          };
        },
        save() {}, restore() {},
        translate(x, y) { canvasCalls.push(['translate', x, y]); },
        scale(x, y) { canvasCalls.push(['scale', x, y]); },
        fillText(value, x, y) { canvasCalls.push(['fillText', value, x, y, this.font]); },
        getImageData() { return { data: new Uint8ClampedArray(8 * 4 * 4 * 4 * 4) }; },
      };
    }
  };
  try {
    D.canvasFontAdapter({
      width: 8, height: 4, x: 0, y: 0,
      textWidth: 0.5, textHeight: 0,
      externalSpace: 2, lineSkip: 1, flags: 0, text: 'x\ny', font: '',
    });
    D.canvasFontAdapter({
      width: 8, height: 4, x: 0.1, y: 0.1,
      textWidth: 0.51, textHeight: 0,
      externalSpace: 2, lineSkip: 1, flags: 3, text: 'abc\nx', font: '',
    });
  } finally {
    if (previousOffscreenCanvas === undefined) delete globalThis.OffscreenCanvas;
    else globalThis.OffscreenCanvas = previousOffscreenCanvas;
  }
  assert.deepEqual(canvasCalls[0], ['translate', 16, 0]);
  assert.deepEqual(canvasCalls[1], ['scale', 1, 1]);
  assert.deepEqual(canvasCalls[2], ['fillText', 'x', 0, 16, '8px Arial']);
  assert.deepEqual(canvasCalls[5], ['fillText', 'y', 0, 25, '8px Arial']);
  // GDI receives integer font dimensions and draw origins, reports integer
  // extents/TEXTMETRIC, and performs integer division for both centering axes.
  assert.deepEqual(canvasCalls[6], ['translate', 13, 0]);
  assert.deepEqual(canvasCalls[8], ['fillText', 'abc', 0, 8, '8px Arial']);
  assert.deepEqual(canvasCalls[9], ['translate', 17, 0]);
  assert.deepEqual(canvasCalls[11], ['fillText', 'x', 0, 17, '8px Arial']);
}

// Browsers without the extended TextMetrics fields retain the released
// Arial tmHeight/em ratio, rather than collapsing line advance to character
// height and leaving the production meshes' authored UV strips empty.
{
  const canvasCalls = [];
  const previousOffscreenCanvas = globalThis.OffscreenCanvas;
  globalThis.OffscreenCanvas = class {
    constructor(width, height) { this.width = width; this.height = height; }
    getContext() {
      return {
        clearRect() {}, fillStyle: '', textBaseline: '', font: '',
        measureText(value) { return { width: String(value).length * 4 }; },
        save() {}, restore() {},
        translate(x, y) { canvasCalls.push(['translate', x, y]); },
        scale(x, y) { canvasCalls.push(['scale', x, y]); },
        fillText(value, x, y) { canvasCalls.push(['fillText', value, x, y, this.font]); },
        getImageData() { return { data: new Uint8ClampedArray(8 * 4 * 4 * 4 * 4) }; },
      };
    }
  };
  try {
    D.canvasFontAdapter({
      width: 8, height: 4, x: 0, y: 0,
      textWidth: 0.5, textHeight: 0,
      externalSpace: 2, lineSkip: 1, flags: 0, text: 'x\ny', font: '',
    });
  } finally {
    if (previousOffscreenCanvas === undefined) delete globalThis.OffscreenCanvas;
    else globalThis.OffscreenCanvas = previousOffscreenCanvas;
  }
  assert.deepEqual(canvasCalls[2], ['fillText', 'x', 0, 16, '8px Arial']);
  assert.deepEqual(canvasCalls[5], ['fillText', 'y', 0, 25, '8px Arial']);
}

const cachedCoverage = new Uint8Array(8 + 4);
cachedCoverage.set([3, 0, 2, 0, 2, 0, 0, 0, 31, 0, 0, 0]);
const cachedText = D.Bitmap_Text({ blob: cachedCoverage }, {}, sourceBeforeText,
  0, 0, 0, 0, 0xffffffff, 0x80, 0, 0, 1, '', '');
assert.equal(textAdapterCalls, 1);
assert.ok(cachedText.data[0] > 0x7000);

const kx = D.parseKX(new Uint8Array(await readFile(new URL('../assets/debris_party.kx', import.meta.url))));

// The same standalone native oracle covers two complete released Debris
// generators at their authored 512x512 size, including the near-one fadeoff
// and a non-black color range.
for (const [operationId, expectedHash] of [[11, 0x7c44dd6d], [49, 0xea938dd4]]) {
  const operation = kx.operations[operationId];
  assert.equal(operation.classId, 0x22);
  const bitmap = D.Bitmap_Perlin(...operation.parameters);
  assert.deepEqual([bitmap.width, bitmap.height], [512, 512]);
  assert.equal(bitmap.summary().hash, expectedHash);
}

// The M11 pass-8 facade overlay at op 1142 uses op 1141 as a MUL2 atlas.
// Its quarter-atlas sources deliberately request border=3 so their edge
// colors clamp outside each authored tile instead of repeating over all four
// quadrants. This production fixture keeps the BilinearSetup regression above
// tied to the graph where reversed border semantics blackened the windows.
const facadeOverlayAtlasOp = kx.operations[1141];
assert.equal(facadeOverlayAtlasOp.classId, 0x24);
assert.deepEqual(facadeOverlayAtlasOp.parameters, [2]);
assert.equal(facadeOverlayAtlasOp.inputs.length, 13);
for (const operationId of [952, 955, 1036, 1107, 1112, 1119, 1122, 1134]) {
  const tileTransform = kx.operations[operationId];
  assert.equal(tileTransform.classId, 0x2c);
  assert.equal(tileTransform.parameters[5], 3,
    `production facade tile transform ${operationId} clamps both axes`);
}

// The production wall-label atlas deliberately opts out of the global
// texture-quality offset (Flat's X exponent carries sBITMAP_DONT_SCALE). It
// therefore remains 512x512 at every quality, matching the released graph.
const labelFlatOp = kx.operations.find(operation => operation.id === 2711);
assert.deepEqual(labelFlatOp.parameters, [137, 9, 0xff000000]);
const labelTextureOffset = D.bitmapTextureSizeOffset;
try {
  for (const offset of [-3, 2]) {
    D.setBitmapTextureSizeOffset(offset);
    const atlas = D.Bitmap_Flat(...labelFlatOp.parameters);
    assert.deepEqual([atlas.width, atlas.height, atlas.texMipCount], [512, 512, 0]);
  }
} finally {
  D.setBitmapTextureSizeOffset(labelTextureOffset);
}

const labelTextOp = kx.operations.find(operation => operation.id === 2712);
assert.deepEqual(labelTextOp.parameters,
  [0.5, 0, 0.0625, 0.031982421875, 0xffffe0c0, 1, 2, 0, 1]);
const authoredLabels = kx.operations
  .filter(operation => operation.id >= 2712 && operation.id <= 2717)
  .flatMap(operation => operation.strings[0].split(/\s+/).filter(Boolean));
for (const label of ['code.', 'ryg.', 'kb.', 'chaos.', 'tron.']) {
  assert.ok(authoredLabels.includes(label), `production atlas contains ${label}`);
}

// 15 seconds at 196 BPM is beat 49. The active 32..64 scene event reaches
// both the production label material and the five facade strips used there.
const t15LabelEvent = kx.events.find(event => event.operation === 11785);
const t15Beat = 15 * kx.songBPM / 60;
assert.ok(t15LabelEvent.start / 65536 <= t15Beat && t15Beat < t15LabelEvent.end / 65536);
const operationsById = new Map(kx.operations.map(operation => [operation.id, operation]));
const t15Reachable = new Set();
const pendingOperationIds = [t15LabelEvent.operation];
while (pendingOperationIds.length) {
  const operationId = pendingOperationIds.pop();
  if (operationId == null || t15Reachable.has(operationId)) continue;
  t15Reachable.add(operationId);
  const operation = operationsById.get(operationId);
  if (operation) pendingOperationIds.push(...operation.inputs, ...operation.links);
}
for (const operationId of [2722, 11736, 11745, 11754, 11763, 11772]) {
  assert.ok(t15Reachable.has(operationId), `t15 graph reaches label op ${operationId}`);
}

const importOp = kx.operations.find(operation => operation.classId === 0x3a);
const imported = D.Bitmap_Import(importOp, importOp.strings[0]);
assert.equal(imported.width, 16);
assert.equal(imported.height, 16);
assert.deepEqual([...imported.data.slice(0, 4)], [...D.packedColor(0xff844eb0)]);

const deferredRecord = { runtime: {}, op: { id: 7 }, inputs: [{ kind: 'ipp' }], parameters: [3, 2] };
const renderHandler = D.handlers.get(0x3f);
assert.equal(typeof renderHandler.init, 'function');
assert.equal(typeof renderHandler.exec, 'function');
assert.equal(renderHandler.exec(deferredRecord), undefined,
  'released Exec_Bitmap_Render is an explicit no-op');
const deferred = renderHandler.init(deferredRecord);
assert.equal(deferred.width, 8);
assert.equal(deferred.height, 4);
assert.equal(deferred.deferredRender.input.kind, 'ipp');

// Smoke the remaining mutable operators on deliberately tiny images.
assert.ok(D.Bitmap_HSCB(flat, 0.1, 0.8, 1.1, 0.9) instanceof D.Bitmap);
assert.ok(D.Bitmap_Blur(flat, 1, 0.25, 0.25, 1) instanceof D.Bitmap);
assert.ok(D.Bitmap_GlowRect(flat, 0.5, 0.5, 0.2, 0.2, 0, 0,
  0xffffffff, 1, 1, 0, 0) instanceof D.Bitmap);
assert.ok(D.Bitmap_Sharpen(flat, 1, 0.1, 0.1, 1) instanceof D.Bitmap);
assert.ok(D.Bitmap_ColorBalance(flat, 0, 0, 0, 0, 0, 0, 0, 0, 0) instanceof D.Bitmap);
assert.ok(D.Bitmap_RotateMul(flat, 0.1, 1, 1, 0.5, 0.5, 3,
  0xffffffff, 0, 1, 0xffffffff) instanceof D.Bitmap);
assert.ok(D.Bitmap_Bump(flat, null, 2, 0.5, 0.5, 1, 0, 0,
  0xffffffff, 0xff202020, 0.5, 1, 1, 0xff000000, 1, 0) instanceof D.Bitmap);

// Bitmap_Bump converts lighting with x87 fistp under round-to-nearest-even.
// 384 * packed(1) / 32768 is exactly 1.5, so native produces the even 2
// instead of the truncating result 1.
{
  const tie = new D.Bitmap(1, 1);
  tie.data.fill(384);
  const rounded = D.Bitmap_Bump(tie, null, 2, 0.5, 0.5, 1, 0, 0,
    0x00000000, 0x01010101, 1, 1, 0, 0x00000000, 1, 0);
  assert.deepEqual([...rounded.data], [2, 2, 2, 2]);
}

// Deterministic performance-path oracles for RotateMul and the remaining
// operators below.
const performanceOracle = new D.Bitmap(8, 4);
for (let index = 0; index < performanceOracle.data.length; index++) {
  performanceOracle.data[index] = (Math.imul(index + 17, 7919) ^ (index * 313)) & 0x7fff;
}

// Whole-buffer fixtures generated by tools/oracles/bitmap_misc_oracle.cpp.
// They pin Gradient's sFtol/integer-half rules and the persistent sF32 locals
// in ColorBalance and Bulge, none of which is covered by the larger Perlin,
// Blur, or Cell source oracles below.
{
  const source = new D.Bitmap(16, 8);
  for (let index = 0; index < source.data.length; index++) {
    source.data[index] = (Math.imul(index + 17, 7919) ^ Math.imul(index, 313)) & 0x7fff;
  }
  assert.equal(D.Bitmap_Gradient(4, 3, 0xff102030, 0xffe0c080,
    0.3125, 0.173828125, 0.73193359375, 0).summary().hash, 4139632454);
  assert.equal(D.Bitmap_ColorBalance(source,
    -0.25, 0, 0.5, 0.5498046875, -0.125, 0.25,
    0.875, 0.125, -0.0478515625).summary().hash, 1592928610);
  const hscbSource = source.copy();
  hscbSource.data[0] = hscbSource.data[1] = hscbSource.data[2] = 234 * 32;
  assert.equal(D.Bitmap_HSCB(hscbSource,
    0, 1, 0.8447265625, 0.82275390625).summary().hash, 1843161851);
  assert.equal(D.Bitmap_GlowRect(D.Bitmap_Flat(10, 0, 0xff000000),
    0, 0, 1, 1, 0, 0, 0xffffffff, 1, 0.625, 0, 2).summary().hash,
  3495771130);
  assert.equal(D.Bitmap_Bricks(3, 3, 0xff000000, 0xffffffff, 0xff000000,
    0, 0, 1, 1, 138, 0, 0, 0, 0.25).summary().hash, 1635508293);
  assert.equal(D.Bitmap_Bulge(source, 1).summary().hash, 1148741333);

  const normals = source.copy();
  for (let index = 3; index < normals.data.length; index += 4) normals.data[index] = 0xffff;
  assert.equal(D.Bitmap_Bump(source, normals, 2, 0.5, 0.5, 0.5, 0.125, 1,
    0xffffffff, 0xffffffff, 0.75, 1, 1,
    0xffffffff, 16, 1).summary().hash, 3523518056,
  'directional Bump retains sPI(F) and persistent sF32 light arithmetic');
  assert.equal(D.Bitmap_Bump(source, normals, 2, 0.5, 0.5, 0.5, 0.25, -0.125,
    0xffffffff, 0xffffffff, 0.75, 1, 1,
    0xffffffff, 16, 1).summary().hash, 2042698802,
  'directional Bump rounds trig results before its x87-single products');
}

// Exact BlurCore fixtures generated by tools/oracles/bitmap_blur_oracle.cpp.
// Besides authored normal/sharpen parameters, these pin odd/even pass buffer
// parity, both transposes, a radius spanning the ring more than three times,
// signed 0x8000..0xffff source words, per-add unsigned saturation, sFtol's two
// half-way directions, and the distinct truncating fixed-amplitude conversion.
const signedBlurPattern = [
  0x0000, 0x0001, 0x7fff, 0x8000, 0xffff, 0x4000,
  0xc000, 0x1234, 0xfedc, 0x7ffe, 0x8001, 0x5555,
];
const blurOracleCases = [
  {
    name: 'production normal order 2', width: 16, height: 16, flags: 0x02,
    sx: 0.06298828125, sy: 0.06298828125, amplitude: 1, inputVariant: 0,
    hash: 3493745528,
    head: [6255, 11147, 16751, 27291, 11039, 11347, 12143, 26243,
      17535, 9859, 8887, 26523, 21455, 9707, 7047, 26963],
    tail: [13023, 13699, 19183, 25747, 7583, 12219, 21231, 28075,
      4191, 10859, 21239, 27203, 4047, 11427, 20407, 28075],
  },
  {
    name: 'production sharpen order 2', width: 16, height: 16, flags: 0x12,
    sx: 0.0999755859375, sy: 0.0999755859375, amplitude: 1, inputVariant: 0,
    hash: 3144209307,
    head: [8647, 11079, 18481, 25929, 9620, 10936, 16863, 25814,
      11457, 10830, 14861, 25519, 13928, 10810, 12830, 25202],
    tail: [13752, 13157, 16072, 23423, 12155, 12952, 17540, 24171,
      10788, 12684, 18571, 24667, 9861, 12408, 18966, 24863],
  },
  {
    name: 'production sharpen odd one axis', width: 16, height: 8, flags: 0x11,
    sx: 0.024993896484375, sy: 0, amplitude: 1, inputVariant: 0,
    hash: 3507585107,
    head: [3017, 11370, 17807, 27142, 3426, 11930, 17805, 28085,
      3455, 12099, 17656, 26702, 9223, 10406, 13835, 22122],
    tail: [11529, 23200, 23304, 11646, 7559, 27241, 22028, 14057,
      7290, 29477, 21138, 14729, 6911, 26898, 18670, 14400],
  },
  {
    name: 'multiwrap order 3', width: 8, height: 8, flags: 0x03,
    sx: 1.75, sy: 0.125, amplitude: 0.875, inputVariant: 0,
    hash: 3182156368,
    head: [7388, 5435, 5374, 11467, 7386, 5435, 5375, 11469,
      7385, 5435, 5375, 11469, 7385, 5435, 5374, 11469],
    tail: [6897, 4711, 5825, 10608, 6898, 4711, 5825, 10605,
      6900, 4710, 5825, 10605, 6901, 4709, 5826, 10605],
  },
  {
    name: 'signed saturating sharpen', width: 8, height: 8, flags: 0x12,
    sx: 0.029998779296875, sy: 0.029998779296875,
    amplitude: 1.25, inputVariant: 2, hash: 3105366885,
    head: [32767, 32767, 32767, 32767, 32767, 32767, 32767, 32767,
      32767, 32767, 21656, 32767, 32767, 23982, 32767, 32767],
    tail: [32767, 32767, 32767, 32767, 32767, 32767, 32767, 32767,
      32767, 32767, 21656, 32767, 32767, 23982, 32767, 32767],
  },
  {
    name: 'round even and fractional amplitude', width: 8, height: 8, flags: 0x01,
    sx: 0.00244140625, sy: 0.00341796875,
    amplitude: 0.10000000149011612, inputVariant: 1, hash: 421788143,
    head: [1652, 3275, 1643, 105, 1657, 3275, 1646, 121,
      1678, 1716, 3275, 1678, 1686, 1646, 3275, 1657],
    tail: [1689, 3275, 1692, 23, 1684, 3275, 1697, 54,
      1675, 3275, 3275, 94, 1699, 3275, 3275, 108],
  },
];
for (const fixture of blurOracleCases) {
  const source = new D.Bitmap(fixture.width, fixture.height);
  for (let index = 0; index < source.data.length; index++) {
    const mixed = (Math.imul(index + 17, 7919) ^ Math.imul(index, 313)) >>> 0;
    source.data[index] = fixture.inputVariant === 0 ? mixed & 0x7fff
      : fixture.inputVariant === 1 ? mixed & 0xffff
        : signedBlurPattern[(index * 5 + Math.trunc(index / 4)) % signedBlurPattern.length];
  }
  const sourceHash = source.summary().hash;
  const blurred = D.Bitmap_Blur(source, fixture.flags,
    fixture.sx, fixture.sy, fixture.amplitude);
  assert.equal(blurred.summary().hash, fixture.hash, fixture.name);
  assert.deepEqual([...blurred.data.slice(0, 16)], fixture.head, `${fixture.name} head`);
  assert.deepEqual([...blurred.data.slice(-16)], fixture.tail, `${fixture.name} tail`);
  assert.equal(source.summary().hash, sourceHash, `${fixture.name} leaves its input untouched`);
}
// GlowRect's released path builds a 1025-entry GetGamma table plus 32
// high-slope samples, then uses sFtol and signed interpolation. Cover the new
// falloff, old falloff with recursive wrapping, and square-distance modes.
assert.equal(D.Bitmap_GlowRect(performanceOracle, 0.5, 0.5, 0.008, 0.008,
  0.06, 0.09, 0xffe0a080, 0.8, 0.5, 0, 0).summary().hash, 2847753868);
assert.equal(D.Bitmap_GlowRect(performanceOracle, 0.02, 0.98, 0.04, 0.03,
  0.02, 0.01, 0xff1020ff, 0.65, 0.02, 1, 1).summary().hash, 1901478777);
assert.equal(D.Bitmap_GlowRect(performanceOracle, 0.5, 0.5, 0.2, 0.15,
  0, 0, 0x80ffffff, 1, 0, 0, 2).summary().hash, 3500825270);
const bumpOracleNormals = D.Bitmap_Normals(performanceOracle, 0.7, 1);
assert.equal(D.Bitmap_Bump(performanceOracle, bumpOracleNormals, 2,
  0.5, 0.5, 1, 0.125, 0.25, 0xffe0c080, 0xff202030,
  0.5, 1, 1.1, 0xffffffff, 8, 0.7).summary().hash, 1703125542,
  'directional Bump preserves exact lighting while reusing its halfway reciprocal');
assert.equal(D.Bitmap_Bump(performanceOracle, bumpOracleNormals, 2,
  0.5, 0.5, 1, 0.125, 0.25, 0xffe0c080, 0xff202030,
  0.5, 1, 1.1, 0xffffffff, 8, 0).summary().hash, 4008563335,
  'directional Bump without specular preserves the source output exactly');
assert.equal(D.Bitmap_RotateMul(performanceOracle, 0.1, 1.1, 0.9, 0.5, 0.5, 3,
  0xffc08040, 0, 3, 0xffe0d0c0).summary().hash, 4226596683);
assert.equal(D.Bitmap_RotateMul(performanceOracle, 0.125, 1.25, 0.75, 0.6, 0.4, 7,
  0xffffffff, 16, 3, 0xffc0a080).summary().hash, 1832531699);
// In source mode 16 only non-opaque fades are squared between iterations.
// Unconditionally squaring 0xffffffff changes it to 0xfefefefe and yields
// 3621358992 instead of this opaque source oracle.
assert.equal(D.Bitmap_RotateMul(performanceOracle, 0.125, 1.25, 0.75, 0.6, 0.4, 7,
  0xffffffff, 16, 3, 0xffffffff).summary().hash, 1027955);
assert.deepEqual(D.Bitmap_Merge(0, [performanceOracle, null]).summary(), performanceOracle.summary(),
  'null terminates the released Merge varargs');

// Exact whole-buffer production fixtures cover the released non-intro 16x16
// search at every launcher scale used below: flipped aspect, candidate culling,
// cell colors, strict nearest-distance ties, and a nearly full 255-cell set.
// Op 475 has ten equal-distance pixels where the stable per-tile insertion
// order deliberately selects a different cell color than an exhaustive search
// in the original unsorted order; its digest pins the released behavior.
const savedTextureOffset = D.bitmapTextureSizeOffset;
const productionCellCases = [
  {
    operationId: 452, textureOffset: -2, dimensions: [128, 128], hash: 2239341598,
    sha256: '4137c522b028ec17c88e589f661e3d128a7b6f5edecdfb4d86a2f93090076a5f',
  },
  {
    operationId: 475, textureOffset: -1, dimensions: [256, 256], hash: 1131615569,
    sha256: '26bd02e11d8694c15d0a00ca592bd807969d804055e42e877f06f48cacf66470',
  },
  {
    operationId: 481, textureOffset: 0, dimensions: [512, 512], hash: 273346632,
    sha256: 'b1f0d6167d769b0e1d890f9642a5bc22fe0e770c9fc60ecd0d20b3a1f821cd81',
  },
];
try {
  for (const fixture of productionCellCases) {
    const operation = kx.operations[fixture.operationId];
    assert.equal(operation.classId, 0x32);
    D.setBitmapTextureSizeOffset(fixture.textureOffset);
    const bitmap = D.Bitmap_Cell(...operation.parameters);
    assert.deepEqual([bitmap.width, bitmap.height], fixture.dimensions);
    assert.equal(bitmap.summary().hash, fixture.hash);
    assert.equal(bitmapWordSha256(bitmap), fixture.sha256);
  }
} finally {
  D.setBitmapTextureSizeOffset(savedTextureOffset);
}

// Runtime handlers may write a source-faithful operator into its designated
// final-consumed input. The unexported opt-in keeps every direct Bitmap_* API
// copy-on-write. These cases verify both output bytes and physical storage.
function makeOwnershipBitmap(seed = 1, width = 8, height = 8) {
  const bitmap = new D.Bitmap(width, height);
  for (let index = 0; index < bitmap.data.length; index++) {
    bitmap.data[index] = (Math.imul(index + seed, 7919) ^ Math.imul(index, 313)) & 0x7fff;
  }
  return bitmap;
}

const unaryInPlaceCases = [
  {
    classId: 0x23, name: 'Color', parameters: [0, 0xff80c040],
    direct: (source, parameters) => D.Bitmap_Color(source, ...parameters),
  },
  {
    classId: 0x27, name: 'GlowRect new',
    parameters: [0.5, 0.5, 0.2, 0.2, 0, 0, 0xffffffff, 1, 1, 0, 0],
    direct: (source, parameters) => D.Bitmap_GlowRect(source, ...parameters),
  },
  {
    classId: 0x39, name: 'GlowRect old',
    parameters: [0.5, 0.5, 0.2, 0.2, 0, 0, 0xffffffff, 1, 1, 0, 1],
    direct: (source, parameters) => D.Bitmap_GlowRect(source, ...parameters),
  },
  {
    classId: 0x2b, name: 'HSCB', parameters: [0.1, 0.8, 1.1, 0.9],
    direct: (source, parameters) => D.Bitmap_HSCB(source, ...parameters),
  },
  {
    classId: 0x29, name: 'Blur', parameters: [2, 0.06298828125, 0.06298828125, 1],
    direct: (source, parameters) => D.Bitmap_Blur(source, ...parameters),
  },
  {
    classId: 0x30, name: 'Bump',
    parameters: [2, 0.5, 0.5, 1, 0, 0, 0xffffffff, 0xff202020,
      0.5, 1, 1, 0xff000000, 1, 0],
    inputs: source => [source, null],
    direct: (source, parameters) => D.Bitmap_Bump(source, null, ...parameters),
  },
  {
    classId: 0x31, name: 'Text',
    parameters: [0, 0, 0.25, 0.25, 0xffffffff, 0, 0, 0, 1], strings: ['x', 'Arial'],
    direct: (source, parameters, strings) =>
      D.Bitmap_Text({}, {}, source, ...parameters, strings[0], strings[1]),
  },
  {
    classId: 0x35, name: 'Range', parameters: [3, 0xff102030, 0xffe0c080],
    direct: (source, parameters) => D.Bitmap_Range(source, ...parameters),
  },
  {
    classId: 0x3b, name: 'ColorBalance',
    parameters: [4, -3, 2, 1, -2, 3, -1, 2, -4],
    direct: (source, parameters) => D.Bitmap_ColorBalance(source, ...parameters),
  },
];

for (const [caseIndex, fixture] of unaryInPlaceCases.entries()) {
  const source = makeOwnershipBitmap(caseIndex + 1);
  const sourceHash = source.summary().hash;
  const sourceBuffer = source.data.buffer;
  const strings = fixture.strings || [];
  const oracle = fixture.direct(source, fixture.parameters, strings);
  assert.notEqual(oracle, source, `${fixture.name} direct result is a copy`);
  assert.equal(source.summary().hash, sourceHash, `${fixture.name} direct call preserves input bytes`);
  assert.equal(source.data.buffer, sourceBuffer, `${fixture.name} direct call preserves input storage`);

  const inputs = fixture.inputs ? fixture.inputs(source) : [source];
  const inputOps = inputs.map(input => input instanceof D.Bitmap ? {} : null);
  const consumerOp = { inputs: inputOps, links: [] };
  const runtime = { operations: [consumerOp], roots: [], events: [] };
  const result = D.handlers.get(fixture.classId)({
    runtime, op: consumerOp, inputs, parameters: fixture.parameters,
    strings, environment: {},
  });
  assert.equal(result, source, `${fixture.name} returns its final-consumed primary input`);
  assert.equal(result.data.buffer, sourceBuffer, `${fixture.name} retains the primary buffer`);
  assert.deepEqual(result.summary(), oracle.summary(), `${fixture.name} in-place bytes match copy path`);
}

{
  const first = makeOwnershipBitmap(21), second = makeOwnershipBitmap(22);
  const firstHash = first.summary().hash, secondHash = second.summary().hash;
  const firstBuffer = first.data.buffer;
  const oracle = D.Bitmap_Merge(0, [first, second]);
  assert.equal(first.summary().hash, firstHash);
  assert.equal(second.summary().hash, secondHash);
  const firstOp = {}, secondOp = {}, consumerOp = { inputs: [firstOp, secondOp], links: [] };
  const runtime = { operations: [consumerOp], roots: [], events: [] };
  const result = D.handlers.get(0x24)({
    runtime, op: consumerOp, inputs: [first, second], parameters: [0],
  });
  assert.equal(result, first);
  assert.equal(result.data.buffer, firstBuffer, 'Merge reuses only its first final-consumed input');
  assert.equal(second.released, true, 'Merge still releases another final-consumed input');
  assert.deepEqual(result.summary(), oracle.summary());
}

{
  const mask = makeOwnershipBitmap(31), b = makeOwnershipBitmap(32), c = makeOwnershipBitmap(33);
  const inputHashes = [mask, b, c].map(bitmap => bitmap.summary().hash);
  const maskBuffer = mask.data.buffer;
  const oracle = D.Bitmap_Mask(mask, b, c, 0);
  assert.deepEqual([mask, b, c].map(bitmap => bitmap.summary().hash), inputHashes,
    'direct Mask preserves all three inputs');
  const maskOp = {}, bOp = {}, cOp = {}, consumerOp = { inputs: [maskOp, bOp, cOp], links: [] };
  const runtime = { operations: [consumerOp], roots: [], events: [] };
  const result = D.handlers.get(0x2a)({
    runtime, op: consumerOp, inputs: [mask, b, c], parameters: [0],
  });
  assert.equal(result, mask);
  assert.equal(result.data.buffer, maskBuffer, 'Mask retains only its primary buffer');
  assert.equal(b.released, true);
  assert.equal(c.released, true);
  assert.deepEqual(result.summary(), oracle.summary());
}

{
  const sample = makeOwnershipBitmap(41), displacement = makeOwnershipBitmap(42);
  const sampleHash = sample.summary().hash;
  const displacementHash = displacement.summary().hash;
  const displacementBuffer = displacement.data.buffer;
  const oracle = D.Bitmap_Distort(sample, displacement, 0.1, 3);
  assert.equal(displacement.summary().hash, displacementHash,
    'direct Distort preserves its displacement input');
  assert.equal(sample.summary().hash, sampleHash, 'direct Distort preserves its sample input');
  const sampleOp = {}, displacementOp = {};
  const consumerOp = { inputs: [sampleOp, displacementOp], links: [] };
  const runtime = { operations: [consumerOp], roots: [], events: [] };
  const result = D.handlers.get(0x2d)({
    runtime, op: consumerOp, inputs: [sample, displacement], parameters: [0.1, 3],
  });
  assert.equal(result, displacement, 'Distort selects its displacement as writable storage');
  assert.equal(result.data.buffer, displacementBuffer);
  assert.equal(sample.released, true);
  assert.deepEqual(result.summary(), oracle.summary());
}

{
  // RotateMul consumes its primary only for the initial Color. Native then
  // copies bb and performs the accumulation in separate buffers.
  const source = makeOwnershipBitmap(51), sourceBuffer = source.data.buffer;
  const parameters = [0.1, 1.1, 0.9, 0.5, 0.5, 3,
    0xffffffff, 0, 1, 0xffffffff];
  const oracle = D.Bitmap_RotateMul(source, ...parameters);
  assert.equal(source.data.buffer, sourceBuffer);
  const sourceOp = {}, consumerOp = { inputs: [sourceOp], links: [] };
  const runtime = { operations: [consumerOp], roots: [], events: [] };
  const result = D.handlers.get(0x36)({
    runtime, op: consumerOp, inputs: [source], parameters,
  });
  assert.equal(result, source);
  assert.notEqual(result.data.buffer, sourceBuffer,
    'RotateMul keeps its later accumulation copies instead of leaking the reuse flag');
  assert.deepEqual(result.summary(), oracle.summary());
}

{
  // A shared producer remains untouched until its final graph consumer.
  const sourceOp = {};
  const firstConsumer = { inputs: [sourceOp], links: [] };
  const finalConsumer = { inputs: [sourceOp], links: [] };
  const runtime = { operations: [firstConsumer, finalConsumer], roots: [], events: [] };
  const source = makeOwnershipBitmap(61), sourceBuffer = source.data.buffer;
  const sourceHash = source.summary().hash;
  const first = D.handlers.get(0x23)({
    runtime, op: firstConsumer, inputs: [source], parameters: [0, 0xff808080],
  });
  assert.notEqual(first, source);
  assert.equal(source.summary().hash, sourceHash);
  const final = D.handlers.get(0x23)({
    runtime, op: finalConsumer, inputs: [source], parameters: [0, 0xffc08040],
  });
  assert.equal(final, source);
  assert.equal(final.data.buffer, sourceBuffer);

  // Production bitmap precalc is one-shot. If tooling invokes this same op
  // again, the ownership claim is not repeated and its input is not mutated.
  const finalHash = final.summary().hash;
  const recalculated = D.handlers.get(0x23)({
    runtime, op: finalConsumer, inputs: [final], parameters: [0, 0xffffffff],
  });
  assert.notEqual(recalculated, final);
  assert.equal(final.summary().hash, finalHash);
  assert.equal(final.data.buffer, sourceBuffer);
}

{
  // Bitmap.adopt() can give distinct producer caches the same backing store.
  // Finishing one producer must copy before writing while the other cache is
  // still live, even when the two wrappers never meet in one operator call.
  const firstProducer = {}, aliasedProducer = {};
  const firstConsumer = { inputs: [firstProducer], links: [] };
  const aliasedConsumer = { inputs: [aliasedProducer], links: [] };
  const runtime = {
    operations: [firstProducer, aliasedProducer, firstConsumer, aliasedConsumer],
    roots: [], events: [],
  };
  const first = makeOwnershipBitmap(62);
  const alias = new D.Bitmap(first.width, first.height).adopt(first);
  const sharedStorage = first.data.buffer;
  const aliasHash = alias.summary().hash;
  firstProducer.cache = first;
  aliasedProducer.cache = alias;

  const firstOracle = D.Bitmap_Color(first, 0, 0xffc08040).summary();
  const firstResult = D.handlers.get(0x23)({
    runtime, op: firstConsumer, inputs: [first], parameters: [0, 0xffc08040],
  });
  assert.equal(firstResult, first);
  assert.notEqual(firstResult.data.buffer, sharedStorage,
    'a live cache alias forces copy-before-write and result adoption');
  assert.equal(alias.data.buffer, sharedStorage);
  assert.equal(alias.summary().hash, aliasHash,
    'mutating one final-consumed wrapper preserves another live cache');
  assert.deepEqual(firstResult.summary(), firstOracle);

  // Public aliases remain permanently conservative. This also protects an
  // alias created after runtime lifetime tracking has already initialized;
  // the small false-negative costs one copy and never affects production's
  // private ownership transfers.
  const aliasOracle = D.Bitmap_Color(alias, 0, 0xff80c040).summary();
  const aliasResult = D.handlers.get(0x23)({
    runtime, op: aliasedConsumer, inputs: [alias], parameters: [0, 0xff80c040],
  });
  assert.equal(aliasResult, alias);
  assert.notEqual(aliasResult.data.buffer, sharedStorage,
    'a public-adopt participant never becomes an in-place candidate');
  assert.deepEqual(aliasResult.summary(), aliasOracle);
}

{
  // Public adoption can happen after the runtime's consumer state exists.
  // The per-Bitmap flag must still block the later final-consumer write.
  const sourceProducer = {}, aliasProducer = {};
  const earlyConsumer = { inputs: [sourceProducer], links: [] };
  const finalConsumer = { inputs: [sourceProducer], links: [] };
  const aliasConsumer = { inputs: [aliasProducer], links: [] };
  const runtime = {
    operations: [sourceProducer, aliasProducer, earlyConsumer, finalConsumer, aliasConsumer],
    roots: [], events: [],
  };
  const source = makeOwnershipBitmap(66);
  sourceProducer.cache = source;
  const earlyResult = D.handlers.get(0x23)({
    runtime, op: earlyConsumer, inputs: [source], parameters: [0, 0xff808080],
  });
  assert.notEqual(earlyResult, source, 'the first shared use initializes lifetime state without transfer');

  const alias = new D.Bitmap(source.width, source.height).adopt(source);
  aliasProducer.cache = alias;
  const sharedStorage = source.data.buffer;
  const aliasHash = alias.summary().hash;
  const oracle = D.Bitmap_Color(source, 0, 0xffc08040).summary();
  const result = D.handlers.get(0x23)({
    runtime, op: finalConsumer, inputs: [source], parameters: [0, 0xffc08040],
  });
  assert.equal(result, source);
  assert.notEqual(result.data.buffer, sharedStorage,
    'post-initialization public alias still forces copy-before-write');
  assert.equal(alias.data.buffer, sharedStorage);
  assert.equal(alias.summary().hash, aliasHash);
  assert.deepEqual(result.summary(), oracle);
}

{
  // Private runtime transfers must preserve the ordinary linear fast path:
  // dead upstream Ops may still expose the same transferred Bitmap identity.
  const producer = {};
  const middle = { inputs: [producer], links: [] };
  const terminal = { inputs: [middle], links: [] };
  const runtime = { operations: [producer, middle, terminal], roots: [], events: [] };
  const source = makeOwnershipBitmap(63);
  const sourceStorage = source.data.buffer;
  producer.cache = source;
  const middleResult = D.handlers.get(0x23)({
    runtime, op: middle, inputs: [source], parameters: [0, 0xffc0c0c0],
  });
  middle.cache = middleResult;
  assert.equal(middleResult, source);
  assert.equal(middleResult.data.buffer, sourceStorage);
  const terminalOracle = D.Bitmap_Color(middleResult, 0, 0xff808080).summary();
  const terminalResult = D.handlers.get(0x23)({
    runtime, op: terminal, inputs: [middleResult], parameters: [0, 0xff808080],
  });
  assert.equal(terminalResult, source);
  assert.equal(terminalResult.data.buffer, sourceStorage,
    'dead upstream cache identities do not block linear reuse');
  assert.deepEqual(terminalResult.summary(), terminalOracle);
}

{
  // Small external runtimes may encode graph edges as numeric operation IDs
  // or array indices. Keep public-alias safety and final-consumer ownership for
  // ID zero, a non-index ID, and the index form.
  const numericCases = [
    { firstProducer: { id: 0 }, aliasedProducer: { id: 42 }, firstKey: 0, aliasKey: 42, label: 'IDs' },
    { firstProducer: {}, aliasedProducer: {}, firstKey: 0, aliasKey: 1, label: 'indices' },
  ];
  for (let index = 0; index < numericCases.length; index++) {
    const fixture = numericCases[index];
    const firstConsumer = { inputs: [fixture.firstKey], links: [] };
    const aliasedConsumer = { inputs: [fixture.aliasKey], links: [] };
    const runtime = {
      operations: [fixture.firstProducer, fixture.aliasedProducer, firstConsumer, aliasedConsumer],
      roots: [], events: [],
    };
    const first = makeOwnershipBitmap(64 + index);
    const alias = new D.Bitmap(first.width, first.height).adopt(first);
    const sharedStorage = first.data.buffer;
    const aliasHash = alias.summary().hash;
    fixture.firstProducer.cache = first;
    fixture.aliasedProducer.cache = alias;
    const result = D.handlers.get(0x23)({
      runtime, op: firstConsumer, inputs: [first], parameters: [0, 0xffc08040],
    });
    assert.equal(result, first);
    assert.notEqual(result.data.buffer, sharedStorage,
      `numeric producer ${fixture.label} still protect another live cache backing`);
    assert.equal(alias.data.buffer, sharedStorage);
    assert.equal(alias.summary().hash, aliasHash);
  }
}

{
  // Numeric operation id zero is a valid ownership key.
  const sourceProducer = { inputs: [], links: [] };
  const consumerOp = { inputs: [0], links: [] };
  const runtime = { operations: [sourceProducer, consumerOp], roots: [], events: [] };
  const source = makeOwnershipBitmap(71), sourceBuffer = source.data.buffer;
  const result = D.handlers.get(0x23)({
    runtime, op: consumerOp, inputs: [source], parameters: [0, 0xff808080],
  });
  assert.equal(result, source);
  assert.equal(result.data.buffer, sourceBuffer);
}

{
  // Guard the ownership optimization's asymptotic shape without a fragile
  // wall-clock assertion. Consumer counts may inspect a large operation list
  // once; writable dispatch must not walk all operations again per call.
  const callCount = 1900, operationCount = 15000;
  const sources = new Array(callCount);
  const producers = new Array(callCount);
  const consumers = new Array(callCount);
  for (let index = 0; index < callCount; index++) {
    const source = D.Bitmap_Flat(0, 0, (0xff000000 | index) >>> 0);
    const producer = { inputs: [], links: [], cache: source };
    sources[index] = source;
    producers[index] = producer;
    consumers[index] = { inputs: [producer], links: [] };
  }
  const operations = producers.concat(consumers);
  while (operations.length < operationCount) operations.push({ inputs: [], links: [] });
  let indexedReads = 0;
  const observedOperations = new Proxy(operations, {
    get(target, property, receiver) {
      if (typeof property === 'string' && /^(0|[1-9][0-9]*)$/.test(property)) indexedReads++;
      return Reflect.get(target, property, receiver);
    },
  });
  const runtime = { operations: observedOperations, roots: [], events: [] };
  for (let index = 0; index < callCount; index++) {
    const source = sources[index], storage = source.data.buffer;
    const result = D.handlers.get(0x23)({
      runtime, op: consumers[index], inputs: [source], parameters: [0, 0xffffffff],
    });
    assert.equal(result, source);
    assert.equal(result.data.buffer, storage);
  }
  assert.ok(indexedReads <= operationCount + 8,
    `writable dispatch reread the ${operationCount}-op runtime ${indexedReads} times`);
}

// Runtime handlers transfer storage only at the final graph consumer. Direct
// Bitmap_* APIs above remain ordinary copy-on-write calls.
{
  const sourceOp = {};
  const consumerOp = { inputs: [sourceOp], links: [] };
  const runtime = { operations: [consumerOp], roots: [], events: [] };
  const source = D.Bitmap_Flat(1, 1, 0xff123456);
  const sourceStorage = source.data.buffer;
  const result = D.handlers.get(0x25)({
    runtime, op: consumerOp, inputs: [source], parameters: [7, 3, 11],
  });
  assert.equal(result, source);
  assert.equal(result.data.buffer, sourceStorage,
    'a uniquely consumed Format changes metadata without copying pixel storage');
  assert.equal(source.format, 7);
  assert.equal(source.texMipCount, 3);
}

{
  const sourceOp = {};
  const firstConsumer = { inputs: [sourceOp], links: [] };
  const finalConsumer = { inputs: [sourceOp], links: [] };
  const runtime = { operations: [firstConsumer, finalConsumer], roots: [], events: [] };
  const source = D.Bitmap_Flat(1, 1, 0xff123456);
  const first = D.handlers.get(0x25)({
    runtime, op: firstConsumer, inputs: [source], parameters: [2, 0, 0],
  });
  assert.notEqual(first, source);
  const final = D.handlers.get(0x25)({
    runtime, op: finalConsumer, inputs: [source], parameters: [3, 0, 0],
  });
  assert.equal(final, source);
  assert.equal(first.format, 2);
  assert.equal(final.format, 3);

  // A later incremental recalculation must not transfer the shared producer
  // again: the final branch cache already aliases it from the initial DFS.
  const recalculated = D.handlers.get(0x25)({
    runtime, op: firstConsumer, inputs: [source], parameters: [4, 0, 0],
  });
  assert.notEqual(recalculated, source);
  assert.equal(recalculated.format, 4);
  assert.equal(source.format, 3, 'incremental branch did not mutate aliased final cache');
}

{
  const sourceOp = {};
  const bitmapConsumer = { inputs: [sourceOp], links: [] };
  const retainedMaterial = { inputs: [], links: [sourceOp] };
  const runtime = { operations: [bitmapConsumer, retainedMaterial], roots: [], events: [] };
  const source = D.Bitmap_Flat(1, 1, 0xff123456);
  const result = D.handlers.get(0x25)({
    runtime, op: bitmapConsumer, inputs: [source], parameters: [9, 0, 0],
  });
  assert.notEqual(result, source, 'a material link pins the original texture object');
  assert.notEqual(source.format, 9);
}

{
  const sourceOpA = {}, sourceOpB = {};
  const consumerOp = { inputs: [sourceOpA, sourceOpB], links: [] };
  const runtime = { operations: [consumerOp], roots: [], events: [] };
  const sourceA = D.Bitmap_Flat(1, 1, 0xff000000);
  const sourceB = D.Bitmap_Flat(1, 1, 0xffffffff);
  const result = D.handlers.get(0x24)({
    runtime, op: consumerOp, inputs: [sourceA, sourceB], parameters: [0],
  });
  assert.equal(result, sourceA);
  assert.equal(sourceB.released, true);
  assert.deepEqual([...result.data.slice(0, 4)], [0x7fff, 0x7fff, 0x7fff, 0x7fff]);
}

{
  // Duplicate slots count as two graph consumers, but the final slot may
  // safely transfer the one physical Bitmap identity after Merge has read it.
  const sourceOp = {};
  const consumerOp = { inputs: [sourceOp, sourceOp], links: [] };
  const runtime = { operations: [consumerOp], roots: [], events: [] };
  const source = D.Bitmap_Flat(1, 1, 0xff204080);
  const sourceStorage = source.data.buffer;
  const oracle = D.Bitmap_Merge(0, [source, source]).summary();
  const result = D.handlers.get(0x24)({
    runtime, op: consumerOp, inputs: [source, source], parameters: [0],
  });
  assert.equal(result, source);
  assert.equal(source.released, false);
  assert.notEqual(result.data.buffer, sourceStorage,
    'duplicate Merge inputs take the copy path before final-result adoption');
  assert.deepEqual(result.summary(), oracle);
}

{
  // Distort must not overwrite displacement coordinates while the sample
  // context aliases the same physical bitmap.
  const sourceOp = {};
  const consumerOp = { inputs: [sourceOp, sourceOp], links: [] };
  const runtime = { operations: [consumerOp], roots: [], events: [] };
  const source = makeOwnershipBitmap(81);
  const sourceStorage = source.data.buffer;
  const oracle = D.Bitmap_Distort(source, source, 0.1, 3).summary();
  const result = D.handlers.get(0x2d)({
    runtime, op: consumerOp, inputs: [source, source], parameters: [0.1, 3],
  });
  assert.equal(result, source);
  assert.notEqual(result.data.buffer, sourceStorage,
    'aliased Distort sample/displacement cannot take the in-place path');
  assert.deepEqual(result.summary(), oracle);
}

{
  // Render hooks/deferred records retain their input by identity. Even an
  // unusual Bitmap input must never be adopted into the blank render target.
  const sourceOp = {};
  const renderOp = { classId: 0x3f, inputs: [sourceOp], links: [] };
  const runtime = { operations: [renderOp], roots: [], events: [] };
  const source = D.Bitmap_Flat(1, 1, 0xff123456);
  const sourceHash = source.summary().hash;
  const rendered = D.handlers.get(0x3f).init({
    runtime, op: renderOp, inputs: [source], parameters: [1, 1],
  });
  assert.notEqual(rendered, source);
  assert.equal(rendered.deferredRender.input, source);
  assert.equal(source.summary().hash, sourceHash);
  assert.equal(source.released, undefined);
}

{
  const sourceOp = {};
  const consumerOp = { inputs: [sourceOp], links: [] };
  const runtime = { operations: [consumerOp], roots: [sourceOp], events: [] };
  const source = D.Bitmap_Flat(1, 1, 0xff123456);
  const result = D.handlers.get(0x25)({
    runtime, op: consumerOp, inputs: [source], parameters: [6, 0, 0],
  });
  assert.notEqual(result, source, 'runtime root pins the producer cache');
}

console.log('bitmap ownership, exact algorithms, and deterministic performance tests passed');
