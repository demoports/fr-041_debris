import assert from 'node:assert/strict';
import * as CoreAPI from '../src/core.js';
import * as GLAPI from '../src/gl.js';

const D = { ...CoreAPI, ...GLAPI };

function approxArray(actual, expected, epsilon = 1e-7) {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, index) => assert.ok(Math.abs(value - expected[index]) < epsilon));
}
approxArray(Array.from(D.colorARGB(0x80402010)), [64 / 255, 32 / 255, 16 / 255, 128 / 255]);
approxArray(Array.from(D.colorRGB(0x00402010)), [64 / 255, 32 / 255, 16 / 255]);

const model = D.mat4SRT(new Float32Array([2, 3, 4, 0.1, -0.2, 0.3, 5, -6, 7]));
const inverse = D.mat4Inverse(model);
assert.ok(inverse);
const identity = D.mat4Mul(model, inverse);
for (let i = 0; i < 16; i++) assert.ok(Math.abs(identity[i] - (i % 5 === 0 ? 1 : 0)) < 2e-5);
assert.equal(D.mat4Inverse(new Float32Array(16)), null);

const camera = {
  cameraSpace: D.mat4Identity(), nearClip: 1, farClip: 11,
  zoomX: 2, zoomY: 3, centerX: 0.25, centerY: -0.5,
};
const projection = D.legacyProjection(camera);
assert.equal(projection[0], 2);
assert.equal(projection[5], 3);
assert.equal(projection[8], 0.25);
assert.equal(projection[9], -0.5);
assert.equal(projection[11], 1);
assert.equal(projection[10], 1);
assert.equal(projection[14], -2);
assert.equal(projection[12], 0);
assert.equal(projection[13], 0);
// Native D3D9 MakeProjectionMatrix's (-1/W,+1/H) terms move screen space by
// (-0.5,-0.5) pixels under D3D's viewport transform. WebGL already moves its
// sample centers by that half pixel, so the port must not retain the terms.
const nativeViewportWidth = 640, nativeViewportHeight = 360;
assert.deepEqual([
  (-1 / nativeViewportWidth) * nativeViewportWidth / 2,
  -(1 / nativeViewportHeight) * nativeViewportHeight / 2,
], [-0.5, -0.5]);
// The shadow shader's extruded point has homogeneous w=0. A +Z direction
// therefore reaches the infinite far boundary with clip z exactly equal to w.
assert.equal(projection[10], projection[11]);
const fartherProjection = D.legacyProjection({ ...camera, farClip: 1e9 });
assert.deepEqual(Array.from(fartherProjection), Array.from(projection));

const bitmap = { width: 2, height: 1, data: new Uint16Array([
  0, 0x4000, 0x7fff, 0xffff,
  0x7fff, 0x2000, 0, 0x7fff,
]) };
assert.deepEqual(Array.from(D.bitmapRGBA8(bitmap)), [0, 128, 255, 255, 255, 64, 0, 255]);

// UpdateTexture takes bits 7..14 directly instead of normalizing 0..0x7fff.
// Lock the half-byte boundaries and the native masked behavior above 0x7fff.
const unsignedQuantizerEdges = { width: 2, height: 1, data: new Uint16Array([
  64, 65, 127, 128,
  0x7fff, 0x8000, 0xffff, 0,
]) };
assert.deepEqual(Array.from(D.bitmapRGBA8(unsignedQuantizerEdges)), [
  0, 0, 0, 1,
  255, 0, 255, 0,
]);

// Production op 1767 uses TexMipTresh 0x12: average the first two reductions,
// then reverse the threshold and point-pick all remaining levels.
const reverseThresholdData = new Uint16Array(8 * 8 * 4);
for (let index = 0; index < 8 * 8; index++) {
  reverseThresholdData[index * 4] = index * 128;
  reverseThresholdData[index * 4 + 3] = 0x7fff;
}
const reverseThresholdLevels = [];
assert.equal(D.forEachBitmapMip({
  width: 8, height: 8, texMipCount: 0, texMipThreshold: 0x12,
  data: reverseThresholdData,
}, (level, width, height, pixels) => {
  reverseThresholdLevels.push({
    level, width, height,
    red: Array.from(pixels.filter((value, index) => (index & 3) === 0)),
  });
}), 4);
assert.deepEqual(reverseThresholdLevels, [
  { level: 0, width: 8, height: 8, red: Array.from({ length: 64 }, (_, index) => index) },
  { level: 1, width: 4, height: 4, red: [4, 6, 8, 10, 20, 22, 24, 26, 36, 38, 40, 42, 52, 54, 56, 58] },
  { level: 2, width: 2, height: 2, red: [13, 17, 45, 49] },
  { level: 3, width: 1, height: 1, red: [13] },
]);
const explicitMipLevels = [];
D.forEachBitmapMip({
  width: 8, height: 8, texMipCount: 2, data: reverseThresholdData,
}, (level, width, height) => explicitMipLevels.push([level, width, height]));
assert.deepEqual(explicitMipLevels, [[0, 8, 8], [1, 4, 4]],
  'native explicit mip counts are clamped and include the base level');
const nonPowerOfTwoLevels = [];
D.forEachBitmapMip({
  width: 3, height: 2, texMipCount: 0, data: new Uint16Array(3 * 2 * 4),
}, (level, width, height) => nonPowerOfTwoLevels.push([level, width, height]));
assert.deepEqual(nonPowerOfTwoLevels, [[0, 3, 2]],
  'native AddTexture disables mipmaps for non-power-of-two textures');

// Native Q8W8V8U8 upload reverses GenBitmap's first/third words and normalizes
// each texel. The shader then sees logical (vz,vy,vx), matching its N,B,T
// basis. The second texel is deliberately non-unit to cover UnitSafe3.
const signedNormalBitmap = {
  width: 4, height: 1, format: 5,
  data: new Uint16Array([
    0x4000, 0x4000, 0x7fff, 0xffff,
    0x7fff, 0x4000, 0x7fff, 0x4000,
    0x4000, 0x4000, 0x7fff, 0x7fff,
    0x4000, 0x4000, 0x7fff, 0,
  ]),
};
const signedNormalPixels = D.bitmapRGBA8(signedNormalBitmap);
assert.ok(signedNormalPixels instanceof Int8Array,
  'Q8W8V8U8 remains signed through the RGBA8_SNORM upload');
assert.deepEqual(Array.from(signedNormalPixels),
  [127, 0, 0, 127, 89, 0, 89, 0,
    127, 0, 0, 127, 127, 0, 0, -128]);

// Bitmap_Normals writes (vx,vy,vz), UpdateTexture reverses the first and
// third words, and the released shaders consume the result as (N,B,T).
// Keep all three positive axes explicit so a future conventional XYZ normal
// conversion cannot silently invert or rotate Debris' tangent frame.
const signedNormalAxes = {
  width: 3, height: 1, format: 5,
  data: new Uint16Array([
    0x4000, 0x4000, 0x7fff, 0x4000, // +N (vz)
    0x4000, 0x7fff, 0x4000, 0x4000, // +B (vy)
    0x7fff, 0x4000, 0x4000, 0x4000, // +T (vx)
  ]),
};
assert.deepEqual(Array.from(D.bitmapRGBA8(signedNormalAxes)), [
  127, 0, 0, 0,
  0, 127, 0, 0,
  0, 0, 127, 0,
]);

// This vector lands on a signed-byte boundary. Native UnitSafe3's float32
// rounding produces signed blue -111 (packed 0x91); an all-double Math.hypot
// normalization produces -110 and no longer matches UpdateTexture.
const signedNormalQuantizerEdge = {
  width: 1, height: 1, format: 5,
  data: new Uint16Array([0x0cbb, 0x23cc, 0x3be1, 0x4000]),
};
assert.deepEqual(Array.from(D.bitmapRGBA8(signedNormalQuantizerEdge)),
  [-8, -61, -111, 0]);

// SNORM converts each signed texel before filtering: both -128 and -127 are
// exactly -1, while -128/+127 interpolate to zero. The former biased-UNORM
// shader decode produced -1/254 at that latter midpoint.
const signedFilterEndpoints = D.bitmapRGBA8({
  width: 2, height: 1, format: 5,
  data: new Uint16Array([
    0x4000, 0x4000, 0x7fff, 0,
    0x4000, 0x4000, 0x7fff, 0x7fff,
  ]),
});
assert.deepEqual(Array.from(signedFilterEndpoints), [
  127, 0, 0, -128, 127, 0, 0, 127,
]);
const snorm8 = value => Math.max(value / 127, -1);
assert.equal((snorm8(signedFilterEndpoints[3]) +
  snorm8(signedFilterEndpoints[7])) * 0.5, 0);
assert.equal((snorm8(-128) + snorm8(-127)) * 0.5, -1);

// Native UpdateTexture averages the 16-bit GenBitmap words first and then
// repeats UnitSafe3 for every signed-normal mip. Two +X and two +Y texels
// therefore become signed unit-diagonal bytes (89,89), rather than the
// shortened vector produced by averaging already-quantized normal bytes.
const signedMipBitmap = {
  width: 2, height: 2, format: 5, texMipCount: 0, texMipThreshold: 0,
  data: new Uint16Array([
    0x4000, 0x4000, 0x7fff, 0xffff,
    0x4000, 0x4000, 0x7fff, 0xffff,
    0x4000, 0x7fff, 0x4000, 0x4000,
    0x4000, 0x7fff, 0x4000, 0x4000,
  ]),
};
const signedMipLevels = [];
assert.equal(D.forEachSignedNormalMip(signedMipBitmap, (level, width, height, pixels) => {
  signedMipLevels.push({ level, width, height, pixels: Array.from(pixels) });
}), 2);
assert.deepEqual(signedMipLevels, [
  {
    level: 0, width: 2, height: 2,
    pixels: [
      127, 0, 0, 127, 127, 0, 0, 127,
      0, 127, 0, 0, 0, 127, 0, 0,
    ],
  },
  { level: 1, width: 1, height: 1, pixels: [89, 89, 0, -64] },
]);
// TexMipTresh follows the source's level comparison and optional point-pick
// path. Threshold one leaves the first generated mip unfiltered.
const pointMipLevels = [];
D.forEachSignedNormalMip({ ...signedMipBitmap, texMipThreshold: 1 },
  (level, width, height, pixels) => pointMipLevels.push(Array.from(pixels)));
assert.deepEqual(pointMipLevels[1], [127, 0, 0, 127]);

const normal = D.normalMatrix3(D.mat4SRT(new Float32Array([2, 4, 8, 0, 0, 0, 0, 0, 0])));
assert.deepEqual(Array.from(normal), [0.5, 0, 0, 0, 0.25, 0, 0, 0, 0.125]);

const samplerParameters = [];
const anisotropyParameters = [];
const textureUploads = [];
const deletedTextures = [];
let textureUploadFailure = null;
let generatedMipmapCount = 0;
const anisotropyExtension = {
  TEXTURE_MAX_ANISOTROPY_EXT: 17,
  MAX_TEXTURE_MAX_ANISOTROPY_EXT: 18,
};
let textureId = 0;
const textureGL = {
  TEXTURE_2D: 1, TEXTURE0: 10, RGBA8: 2, RGBA: 3, UNSIGNED_BYTE: 4,
  RGBA8_SNORM: 21, BYTE: 22,
  UNPACK_ALIGNMENT: 5, TEXTURE_MIN_FILTER: 6, TEXTURE_MAG_FILTER: 7,
  TEXTURE_WRAP_S: 8, TEXTURE_WRAP_T: 9, REPEAT: 11, CLAMP_TO_EDGE: 12,
  LINEAR: 13, NEAREST: 14, LINEAR_MIPMAP_NEAREST: 15, LINEAR_MIPMAP_LINEAR: 16,
  NEAREST_MIPMAP_NEAREST: 19, TEXTURE_MAX_LEVEL: 20,
  createTexture: () => ({ id: ++textureId }), bindTexture() {}, activeTexture() {},
  pixelStorei() {},
  texImage2D(target, level, internalFormat, width, height, border, format, type, pixels) {
    if (textureUploadFailure) throw textureUploadFailure;
    textureUploads.push({
      level, internalFormat, width, height, format, type,
      pixelType: pixels?.constructor?.name,
      pixels: pixels ? Array.from(pixels) : null,
    });
  },
  generateMipmap() { generatedMipmapCount++; },
  deleteTexture(texture) { deletedTextures.push(texture); },
  getExtension(name) { return name === 'EXT_texture_filter_anisotropic' ? anisotropyExtension : null; },
  getParameter(name) {
    assert.equal(name, anisotropyExtension.MAX_TEXTURE_MAX_ANISOTROPY_EXT);
    return 8;
  },
  texParameteri(target, name, value) { samplerParameters.push([target, name, value]); },
  texParameterf(target, name, value) { anisotropyParameters.push([target, name, value]); },
};

// Public cache callers can hand us released graph values or malformed plain
// objects. Validate those before creating a per-bitmap WebGL texture: the only
// allocation here must be the single shared fallback, and no incomplete DXT5
// entry may appear in telemetry.
const malformedTextureCache = new D.TextureCache(textureGL, { dxt5Mode: 'rgba8' });
const malformedCreateStart = textureId;
const malformedUploadStart = textureUploads.length;
const missingDxt5Data = { width: 4, height: 4, format: 9 };
assert.equal(malformedTextureCache.estimatedUploadBytes(missingDxt5Data), 4,
  'a malformed bitmap predicts only the shared fallback allocation');
const malformedFallback = malformedTextureCache.get(missingDxt5Data);
assert.equal(textureId, malformedCreateStart + 1,
  'missing DXT5 data creates only the shared fallback texture');
assert.deepEqual(textureUploads.slice(malformedUploadStart).map(upload => [
  upload.level, upload.width, upload.height, upload.pixels,
]), [[0, 1, 1, [255, 255, 255, 255]]]);
assert.equal(malformedTextureCache.get({
  width: 4, height: 4, format: 9, data: new Uint16Array(4),
}), malformedFallback, 'short DXT5 data reuses the fallback without allocating');
assert.equal(malformedTextureCache.get({
  width: 2, height: 2, data: new Uint16Array(15),
}), malformedFallback, 'ordinary short bitmap data is rejected before allocation');
assert.equal(textureId, malformedCreateStart + 1);
assert.deepEqual(malformedTextureCache.resourceStats(), {
  textures: 1, bitmapTextures: 0,
  dxt5CompressedTextures: 0, dxt5FallbackTextures: 0,
  estimatedBytes: 4,
}, 'malformed bitmaps never register an incomplete cache entry');
assert.equal(malformedTextureCache.estimatedUploadBytes(missingDxt5Data), 0,
  'an existing fallback adds no further warm-up residency');

const unsupportedDxt5Quality = {
  width: 4, height: 4, format: 9, texMipThreshold: 0x40,
  data: new Uint16Array(4 * 4 * 4),
};
assert.throws(() => malformedTextureCache.get(unsupportedDxt5Quality),
  /production quality-0/, 'unsupported DXT5 quality still fails clearly');
assert.equal(textureId, malformedCreateStart + 1,
  'DXT5 quality rejection also happens before WebGL allocation');

// If a formerly valid graph bitmap releases its storage, its old GPU entry is
// no longer reachable through get(). Retire it immediately, including DXT
// counters and byte telemetry, before installing/returning the fallback.
const releasedTextureCache = new D.TextureCache(textureGL, { dxt5Mode: 'rgba8' });
const releasedBitmap = {
  width: 4, height: 4, format: 9, texMipCount: 0,
  data: new Uint16Array(4 * 4 * 4),
};
const releasedHandle = releasedTextureCache.get(releasedBitmap);
assert.deepEqual(releasedTextureCache.resourceStats(), {
  textures: 1, bitmapTextures: 1,
  dxt5CompressedTextures: 0, dxt5FallbackTextures: 1,
  estimatedBytes: 84,
});
const releasedDeleteStart = deletedTextures.length;
releasedBitmap.data = new Uint16Array();
const releasedFallback = releasedTextureCache.get(releasedBitmap);
assert.notEqual(releasedFallback, releasedHandle);
assert.deepEqual(deletedTextures.slice(releasedDeleteStart), [releasedHandle],
  'released bitmap storage immediately deletes its formerly cached handle');
assert.deepEqual(releasedTextureCache.resourceStats(), {
  textures: 1, bitmapTextures: 0,
  dxt5CompressedTextures: 0, dxt5FallbackTextures: 0,
  estimatedBytes: 4,
}, 'released bitmap eviction removes every entry telemetry contribution');
releasedTextureCache.dispose();
assert.deepEqual(deletedTextures.slice(releasedDeleteStart), [releasedHandle, releasedFallback],
  'dispose only retains the replacement fallback after invalidation');

// A data-reference miss with unchanged dimensions/mips is still a real
// reupload. Stage it on a fresh handle so a mid-upload exception deletes only
// the candidate and leaves the previous entry valid and exactly accounted.
const transactionalTextureCache = new D.TextureCache(textureGL);
const transactionalSource = new Uint16Array(2 * 2 * 4);
const transactionalReplacement = new Uint16Array(2 * 2 * 4);
transactionalReplacement.fill(0x7fff);
const transactionalBitmap = {
  width: 2, height: 2, data: transactionalSource,
};
const transactionalHandle0 = transactionalTextureCache.get(transactionalBitmap);
const transactionalDeleteStart = deletedTextures.length;
const failedCreateStart = textureId;
transactionalBitmap.data = transactionalReplacement;
textureUploadFailure = new Error('synthetic texture upload failure');
try {
  assert.throws(() => transactionalTextureCache.get(transactionalBitmap),
    /synthetic texture upload failure/);
} finally {
  textureUploadFailure = null;
}
const failedCandidate = deletedTextures.at(-1);
assert.equal(failedCandidate.id, failedCreateStart + 1);
assert.notEqual(failedCandidate, transactionalHandle0);
assert.deepEqual(deletedTextures.slice(transactionalDeleteStart), [failedCandidate],
  'failed reupload deletes its fresh candidate, not the cached texture');
assert.deepEqual(transactionalTextureCache.resourceStats(), {
  textures: 1, bitmapTextures: 1,
  dxt5CompressedTextures: 0, dxt5FallbackTextures: 0,
  estimatedBytes: 20,
}, 'failed reupload leaves prior telemetry unchanged');

transactionalBitmap.data = transactionalSource;
const immutableHitCreateStart = textureId;
assert.equal(transactionalTextureCache.get(transactionalBitmap), transactionalHandle0);
assert.equal(textureId, immutableHitCreateStart,
  'restored immutable cache hit performs no allocation');
transactionalBitmap.data = transactionalReplacement;
const transactionalHandle1 = transactionalTextureCache.get(transactionalBitmap);
assert.notEqual(transactionalHandle1, transactionalHandle0);
assert.deepEqual(deletedTextures.slice(transactionalDeleteStart), [
  failedCandidate, transactionalHandle0,
], 'successful retry atomically replaces and retires the previous handle');
transactionalTextureCache.dispose();
assert.deepEqual(deletedTextures.slice(transactionalDeleteStart), [
  failedCandidate, transactionalHandle0, transactionalHandle1,
]);

// Redefining the same bitmap object with a different size, mip shape, or
// internal format must retire its old allocation. In particular, shrinking a
// full chain cannot leave the old upper levels resident but unreported.
const mutableTextureCache = new D.TextureCache(textureGL);
const mutableBitmap = {
  width: 4, height: 4, texMipCount: 0, data: new Uint16Array(4 * 4 * 4),
};
const mutableHandle0 = mutableTextureCache.get(mutableBitmap);
const mutableDeleteStart = deletedTextures.length;
mutableBitmap.width = mutableBitmap.height = 2;
mutableBitmap.data = new Uint16Array(2 * 2 * 4);
const mutableHandle1 = mutableTextureCache.get(mutableBitmap);
assert.notEqual(mutableHandle1, mutableHandle0);
assert.deepEqual(deletedTextures.slice(mutableDeleteStart), [mutableHandle0],
  'a size change recreates storage and deletes the longer old mip chain');
assert.equal(mutableTextureCache.resourceStats().estimatedBytes, 20);

mutableBitmap.texMipCount = 1;
const mutableHandle2 = mutableTextureCache.get(mutableBitmap);
assert.notEqual(mutableHandle2, mutableHandle1);
assert.deepEqual(deletedTextures.slice(mutableDeleteStart), [mutableHandle0, mutableHandle1],
  'a mip-count change recreates storage instead of retaining stale levels');
assert.equal(mutableTextureCache.resourceStats().estimatedBytes, 16);

mutableBitmap.format = 5;
const mutableHandle3 = mutableTextureCache.get(mutableBitmap);
assert.notEqual(mutableHandle3, mutableHandle2);
assert.deepEqual(deletedTextures.slice(mutableDeleteStart), [
  mutableHandle0, mutableHandle1, mutableHandle2,
], 'an internal-format change recreates the texture object');
assert.deepEqual(mutableTextureCache.resourceStats(), {
  textures: 1, bitmapTextures: 1,
  dxt5CompressedTextures: 0, dxt5FallbackTextures: 0,
  estimatedBytes: 16,
}, 'replacement uploads preserve one-entry resource telemetry');
mutableTextureCache.dispose();
assert.deepEqual(deletedTextures.slice(mutableDeleteStart), [
  mutableHandle0, mutableHandle1, mutableHandle2, mutableHandle3,
], 'dispose deletes only the live replacement after retired handles');

const textureCache = new D.TextureCache(textureGL);
const samplerBitmap = { width: 2, height: 2, data: new Uint16Array(16) };
assert.equal(textureCache.estimatedUploadBytes(samplerBitmap), 20,
  'warm-up predicts the exact RGBA8 mip-chain residency before upload');
const lastParameter = name => samplerParameters.findLast(value => value[1] === name)?.[2];
const unsignedUploadStart = textureUploads.length;
textureCache.bind(samplerBitmap, 0, { filterMode: 0 });
assert.equal(textureCache.estimatedUploadBytes(samplerBitmap), 0,
  'an immutable cache hit adds no warm-up residency');
assert.equal(generatedMipmapCount, 0,
  'ordinary generated formats upload source-word mip levels without gl.generateMipmap');
assert.deepEqual(textureUploads.slice(unsignedUploadStart).map(upload => [
  upload.level, upload.width, upload.height, upload.pixels,
]), [
  [0, 2, 2, new Array(16).fill(0)],
  [1, 1, 1, new Array(4).fill(0)],
]);
assert.equal(lastParameter(textureGL.TEXTURE_MAX_LEVEL), 1);
assert.equal(lastParameter(textureGL.TEXTURE_MAG_FILTER), textureGL.NEAREST);
assert.equal(lastParameter(textureGL.TEXTURE_MIN_FILTER), textureGL.NEAREST);
textureCache.bind(samplerBitmap, 0, { filterMode: 2 });
assert.equal(lastParameter(textureGL.TEXTURE_MIN_FILTER), textureGL.LINEAR_MIPMAP_NEAREST);
textureCache.bind(samplerBitmap, 0, { filterMode: 3 });
assert.equal(lastParameter(textureGL.TEXTURE_MIN_FILTER), textureGL.LINEAR_MIPMAP_LINEAR);
// Material11's released table differs from Material20's for the production
// mode-2 textures, and also retains its point/linear point-mip modes.
textureCache.bind(samplerBitmap, 0, { filterMode: 2, filterProfile: 'material11' });
assert.equal(lastParameter(textureGL.TEXTURE_MIN_FILTER), textureGL.LINEAR_MIPMAP_LINEAR);
textureCache.bind(samplerBitmap, 0, { filterMode: 4, filterProfile: 'material11' });
assert.equal(lastParameter(textureGL.TEXTURE_MAG_FILTER), textureGL.NEAREST);
assert.equal(lastParameter(textureGL.TEXTURE_MIN_FILTER), textureGL.NEAREST_MIPMAP_NEAREST);
textureCache.bind(samplerBitmap, 0, { filterMode: 5, filterProfile: 'material11' });
assert.equal(lastParameter(textureGL.TEXTURE_MAG_FILTER), textureGL.LINEAR);
assert.equal(lastParameter(textureGL.TEXTURE_MIN_FILTER), textureGL.LINEAR_MIPMAP_NEAREST);
const lastAnisotropy = () => anisotropyParameters.at(-1)?.[2];
textureCache.bind(samplerBitmap, 0, { filterMode: 3, filterProfile: 'material11' });
assert.equal(lastAnisotropy(), 4);
textureCache.bind(samplerBitmap, 0, { filterMode: 4, filterProfile: 'material20' });
assert.equal(lastAnisotropy(), 4);
textureCache.bind(samplerBitmap, 0, { filterMode: 2, filterProfile: 'material20' });
assert.equal(lastAnisotropy(), 1, 'non-anisotropic reuse resets texture state');

// Sampler parameters belong to the texture object, not the texture unit. An
// identical material bind must therefore reuse the installed wrap/filter/
// anisotropy state, while later changes must still be applied exactly once.
let integerParameterCount = samplerParameters.length;
let floatingParameterCount = anisotropyParameters.length;
textureCache.bind(samplerBitmap, 0, {
  clamp: false, filterMode: 2, filterProfile: 'material20',
});
assert.equal(samplerParameters.length, integerParameterCount,
  'an identical bitmap sampler does not repeat integer texture parameters');
assert.equal(anisotropyParameters.length, floatingParameterCount,
  'an identical bitmap sampler does not repeat anisotropy');

textureCache.bind(samplerBitmap, 0, {
  clamp: true, filterMode: 2, filterProfile: 'material20',
});
assert.deepEqual(samplerParameters.slice(integerParameterCount).map(([, name]) => name), [
  textureGL.TEXTURE_WRAP_S, textureGL.TEXTURE_WRAP_T,
], 'a clamp change only reapplies the two wrap axes');
assert.equal(anisotropyParameters.length, floatingParameterCount);

integerParameterCount = samplerParameters.length;
textureCache.bind(samplerBitmap, 0, {
  clamp: true, filterMode: 2, filterProfile: 'material11',
});
assert.deepEqual(samplerParameters.slice(integerParameterCount).map(([, name]) => name), [
  textureGL.TEXTURE_MIN_FILTER,
], 'a profile change reapplies the differing native minification filter');

floatingParameterCount = anisotropyParameters.length;
textureCache.bind(samplerBitmap, 0, {
  clamp: true, filterMode: 3, filterProfile: 'material11',
});
assert.equal(anisotropyParameters.length, floatingParameterCount + 1);
assert.equal(lastAnisotropy(), 4, 'a later anisotropic mode is installed');
integerParameterCount = samplerParameters.length;
floatingParameterCount = anisotropyParameters.length;
textureCache.bind(samplerBitmap, 0, {
  clamp: true, filterMode: 3, filterProfile: 'material11',
});
assert.equal(samplerParameters.length, integerParameterCount);
assert.equal(anisotropyParameters.length, floatingParameterCount,
  'the changed bitmap sampler is cached after its first application');
textureCache.bind(samplerBitmap, 0, {
  clamp: true, filterMode: 4, filterProfile: 'material20',
});
assert.equal(samplerParameters.length, integerParameterCount);
assert.equal(anisotropyParameters.length, floatingParameterCount,
  'different sampler words with identical GL state do not emit redundant calls');

const anisotropyCallCount = anisotropyParameters.length;
const fallbackTextureCache = new D.TextureCache({
  ...textureGL,
  getExtension() { return null; },
});
fallbackTextureCache.bind(samplerBitmap, 0, { filterMode: 4, filterProfile: 'material20' });
assert.equal(lastParameter(textureGL.TEXTURE_MIN_FILTER), textureGL.LINEAR_MIPMAP_LINEAR);
assert.equal(anisotropyParameters.length, anisotropyCallCount,
  'trilinear filtering remains a safe fallback when anisotropy is unavailable');

// Null material slots share the 1x1 fallback texture. Cache its state too, but
// do not confuse it with the independently cached state of a generated bitmap.
const missingTextureCache = new D.TextureCache(textureGL);
missingTextureCache.bind(null, 0, {
  clamp: true, filterMode: 3, filterProfile: 'material20',
});
integerParameterCount = samplerParameters.length;
floatingParameterCount = anisotropyParameters.length;
missingTextureCache.bind(null, 2, {
  clamp: true, filterMode: 3, filterProfile: 'material20',
});
assert.equal(samplerParameters.length, integerParameterCount,
  'fallback state is reused across texture units');
assert.equal(anisotropyParameters.length, floatingParameterCount,
  'fallback anisotropy is not repeated across texture units');

missingTextureCache.bind(null, 1, {
  clamp: false, filterMode: 0, filterProfile: 'material20',
});
assert.deepEqual(samplerParameters.slice(integerParameterCount).map(([, name]) => name), [
  textureGL.TEXTURE_WRAP_S, textureGL.TEXTURE_WRAP_T,
  textureGL.TEXTURE_MAG_FILTER, textureGL.TEXTURE_MIN_FILTER,
], 'fallback clamp and point-filter changes are both reapplied');
assert.equal(anisotropyParameters.length, floatingParameterCount,
  'unchanged fallback anisotropy remains cached');

integerParameterCount = samplerParameters.length;
floatingParameterCount = anisotropyParameters.length;
missingTextureCache.bind(null, 1, {
  clamp: false, filterMode: 3, filterProfile: 'material11',
});
assert.deepEqual(samplerParameters.slice(integerParameterCount).map(([, name]) => name), [
  textureGL.TEXTURE_MAG_FILTER, textureGL.TEXTURE_MIN_FILTER,
], 'a later fallback profile/filter mode restores linear filtering');
assert.equal(anisotropyParameters.length, floatingParameterCount + 1);
assert.equal(lastAnisotropy(), 4,
  'a later fallback profile correctly enables anisotropy');

const noMipBitmap = { width: 2, height: 2, texMipCount: 1, data: new Uint16Array(16) };
textureCache.bind(noMipBitmap, 0, { filterMode: 4 });
assert.equal(lastParameter(textureGL.TEXTURE_MIN_FILTER), textureGL.LINEAR);
integerParameterCount = samplerParameters.length;
floatingParameterCount = anisotropyParameters.length;
textureCache.bind(samplerBitmap, 0, {
  clamp: true, filterMode: 4, filterProfile: 'material20',
});
assert.equal(samplerParameters.length, integerParameterCount,
  'binding another texture does not invalidate this texture sampler cache');
assert.equal(anisotropyParameters.length, floatingParameterCount);

const signedUploadStart = textureUploads.length;
const generatedBeforeSigned = generatedMipmapCount;
textureCache.bind(signedMipBitmap, 0, { filterMode: 3, filterProfile: 'material20' });
assert.equal(generatedMipmapCount, generatedBeforeSigned,
  'signed normals upload source-derived levels instead of gl.generateMipmap');
assert.deepEqual(textureUploads.slice(signedUploadStart).map(upload => [
  upload.level, upload.internalFormat, upload.width, upload.height,
  upload.format, upload.type, upload.pixelType, upload.pixels,
]), [
  [0, textureGL.RGBA8_SNORM, 2, 2, textureGL.RGBA, textureGL.BYTE,
    'Int8Array', signedMipLevels[0].pixels],
  [1, textureGL.RGBA8_SNORM, 1, 1, textureGL.RGBA, textureGL.BYTE,
    'Int8Array', signedMipLevels[1].pixels],
]);

const signedBaseUploadStart = textureUploads.length;
textureCache.bind({ ...signedNormalQuantizerEdge, texMipCount: 1 }, 0,
  { filterMode: 1, filterProfile: 'material20' });
assert.deepEqual(textureUploads.slice(signedBaseUploadStart).map(upload => [
  upload.internalFormat, upload.format, upload.type, upload.pixelType, upload.pixels,
]), [[textureGL.RGBA8_SNORM, textureGL.RGBA, textureGL.BYTE, 'Int8Array',
  [-8, -61, -111, 0]]], 'single-level signed textures use the same SNORM path');

const dxt5Bitmap = {
  width: 4, height: 4, format: 9, texMipCount: 0, texMipThreshold: 0,
  data: new Uint16Array(4 * 4 * 4),
};
for (let offset = 0; offset < dxt5Bitmap.data.length; offset += 4) {
  dxt5Bitmap.data[offset] = 0x7fff; // logical red; the codec gathers native BGRA
  dxt5Bitmap.data[offset + 3] = 0x7fff;
}
const s3tcExtension = { COMPRESSED_RGBA_S3TC_DXT5_EXT: 0x83f3 };
const compressedUploads = [];
const s3tcGL = {
  ...textureGL,
  getExtension(name) {
    if (name === 'EXT_texture_filter_anisotropic') return anisotropyExtension;
    if (name === 'WEBGL_compressed_texture_s3tc') return s3tcExtension;
    return null;
  },
  compressedTexImage2D(target, level, internalFormat, width, height, border, pixels) {
    compressedUploads.push({ target, level, internalFormat, width, height, pixels: Array.from(pixels) });
  },
};
const generatedBeforeDxt5 = generatedMipmapCount;
const regularUploadsBeforeDxt5 = textureUploads.length;
const compressedTextureCache = new D.TextureCache(s3tcGL);
assert.equal(compressedTextureCache.estimatedUploadBytes(dxt5Bitmap), 48,
  'S3TC warm-up admission uses exact block-rounded BC3 levels');
compressedTextureCache.bind(dxt5Bitmap, 0, { filterMode: 3, filterProfile: 'material20' });
assert.deepEqual(compressedUploads.map(upload => [
  upload.level, upload.width, upload.height, upload.internalFormat, upload.pixels.length,
]), [
  [0, 4, 4, s3tcExtension.COMPRESSED_RGBA_S3TC_DXT5_EXT, 16],
  [1, 2, 2, s3tcExtension.COMPRESSED_RGBA_S3TC_DXT5_EXT, 16],
  [2, 1, 1, s3tcExtension.COMPRESSED_RGBA_S3TC_DXT5_EXT, 16],
]);
assert.equal(textureUploads.length, regularUploadsBeforeDxt5,
  'S3TC-capable upload never materializes RGBA texture levels');
assert.equal(generatedMipmapCount, generatedBeforeDxt5,
  'authored compressed mip levels never call gl.generateMipmap');
assert.deepEqual(compressedTextureCache.resourceStats(), {
  textures: 1, bitmapTextures: 1,
  dxt5CompressedTextures: 1, dxt5FallbackTextures: 0,
  estimatedBytes: 48,
});
compressedTextureCache.bind(dxt5Bitmap, 1, { filterMode: 3, filterProfile: 'material20' });
assert.equal(compressedUploads.length, 3, 'cached DXT5 texture is not recompressed on rebind');

const fallbackUploadStart = textureUploads.length;
const fallbackDxt5Cache = new D.TextureCache(s3tcGL, { dxt5Mode: 'rgba8' });
assert.equal(fallbackDxt5Cache.estimatedUploadBytes(dxt5Bitmap), 84,
  'portable DXT5 warm-up admission accounts for decoded RGBA8 levels');
fallbackDxt5Cache.bind(dxt5Bitmap, 0, { filterMode: 3, filterProfile: 'material20' });
const fallbackUploads = textureUploads.slice(fallbackUploadStart);
assert.deepEqual(fallbackUploads.map(upload => [
  upload.level, upload.width, upload.height, upload.pixels.length,
]), [[0, 4, 4, 64], [1, 2, 2, 16], [2, 1, 1, 4]]);
for (const upload of fallbackUploads) {
  for (let offset = 0; offset < upload.pixels.length; offset += 4) {
    assert.deepEqual(upload.pixels.slice(offset, offset + 4), [255, 0, 0, 255]);
  }
}
assert.equal(generatedMipmapCount, generatedBeforeDxt5,
  'CPU DXT5 round-trip uploads every source-derived mip explicitly');
assert.deepEqual(fallbackDxt5Cache.resourceStats(), {
  textures: 1, bitmapTextures: 1,
  dxt5CompressedTextures: 0, dxt5FallbackTextures: 1,
  estimatedBytes: 84,
});

assert.throws(() => new D.TextureCache(textureGL, { dxt5Mode: 's3tc' }),
  /requires WEBGL_compressed_texture_s3tc/,
  'forced S3TC mode fails clearly instead of silently changing diagnostics');
assert.throws(() => new D.TextureCache(textureGL, { dxt5Mode: 'invalid' }),
  /dxt5Mode/);

// Resource telemetry must describe the levels actually allocated, including
// explicit counts and D3D's one-level NPOT rule. The former 4/3 estimate was
// only exact for an infinite square chain and over-counted all three cases.
const exactTelemetryCache = new D.TextureCache(textureGL, { dxt5Mode: 'rgba8' });
exactTelemetryCache.bind({
  width: 2, height: 2, texMipCount: 0, data: new Uint16Array(2 * 2 * 4),
});
exactTelemetryCache.bind({
  width: 2, height: 2, texMipCount: 1, data: new Uint16Array(2 * 2 * 4),
});
exactTelemetryCache.bind({
  width: 3, height: 2, texMipCount: 0, data: new Uint16Array(3 * 2 * 4),
});
assert.deepEqual(exactTelemetryCache.resourceStats(), {
  textures: 3, bitmapTextures: 3,
  dxt5CompressedTextures: 0, dxt5FallbackTextures: 0,
  estimatedBytes: 20 + 16 + 24,
}, 'RGBA8 memory telemetry sums every allocated mip exactly');

console.log('WebGL math and bitmap upload tests passed');
