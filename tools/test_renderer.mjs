import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as core from '../src/core.js';
import * as gl from '../src/gl.js';
import * as lookup from '../src/legacy_lookup.js';
import * as renderer from '../src/renderer.js';

const D = { ...core, ...gl, ...lookup, ...renderer };
const material20EnvironmentVertexReference = readFileSync(new URL(
  '../vendor/wz3/materials/material20_envi.vsh', import.meta.url), 'utf8');
const material11VertexReference = readFileSync(new URL(
  '../vendor/wz3/materials/material11.vsh', import.meta.url), 'utf8');
const effectExecutor = (environment, job) => {
  if (job.effectGeometry) environment.frame.effectGeometry.push(job.effectGeometry);
  if (job.postJob) environment.frame.postJobs.push(job.postJob);
};

// Catch shader-interface mistakes without starting a browser/WebGL context.
// Duplicate declarations or a vertex/fragment type mismatch are compile-time
// failures even though the numerical renderer tests can still run in Node.
const shaderVaryings = (source, qualifier) => source.split('\n')
  .filter(line => line.startsWith(`${qualifier} `))
  .map(line => {
    const fields = line.split(/\s+/);
    return [fields[1], fields[2].replace(';', '')];
  });
const vertexVaryings = shaderVaryings(D.materialVertexSource, 'out');
const fragmentVaryings = shaderVaryings(D.materialFragmentSource, 'in');
assert.equal(new Set(vertexVaryings.map(([, name]) => name)).size, vertexVaryings.length,
  'the material vertex shader has no duplicate varying declarations');
assert.equal(new Set(fragmentVaryings.map(([, name]) => name)).size, fragmentVaryings.length,
  'the material fragment shader has no duplicate varying declarations');
assert.deepEqual(fragmentVaryings, vertexVaryings,
  'the material vertex and fragment varying interfaces match exactly');
const varyingScalars = vertexVaryings.reduce((sum, [type]) => sum +
  ({ vec2: 2, vec3: 3, vec4: 4 }[type] || 0), 0);
assert.equal(varyingScalars, 44,
  'source-exact vertex lighting replaces P/N/T without increasing the varying budget');

// MakeCubeNormalizer and MakeAttenuationVolume are fixed renderer resources,
// not production bitmap nodes. Lock target orientation, sampler state, byte
// accounting, texture-unit binding and disposal without creating a browser.
{
  let nextTexture = 0;
  const uploads2D = [], uploads3D = [], parameters = [], activeUnits = [];
  const bindings = [], deletedTextures = [], pixelStores = [];
  const lookupGL = {
    UNPACK_ALIGNMENT: 1, UNPACK_FLIP_Y_WEBGL: 2,
    TEXTURE_CUBE_MAP: 10, TEXTURE_CUBE_MAP_POSITIVE_X: 20, TEXTURE_3D: 30,
    RGBA8: 40, RGBA: 41, UNSIGNED_BYTE: 42,
    TEXTURE_MIN_FILTER: 50, TEXTURE_MAG_FILTER: 51,
    TEXTURE_WRAP_S: 52, TEXTURE_WRAP_T: 53, TEXTURE_WRAP_R: 54,
    TEXTURE_BASE_LEVEL: 55, TEXTURE_MAX_LEVEL: 56,
    LINEAR: 60, REPEAT: 61, CLAMP_TO_EDGE: 62, TEXTURE0: 100,
    createTexture() { return { texture: ++nextTexture }; },
    pixelStorei(name, value) { pixelStores.push([name, value]); },
    bindTexture(target, texture) { bindings.push([target, texture]); },
    texImage2D(target, level, internalFormat, width, height, border, format, type, bytes) {
      uploads2D.push({ target, level, internalFormat, width, height, border,
        format, type, byteLength: bytes.byteLength, byteOffset: bytes.byteOffset });
    },
    texImage3D(target, level, internalFormat, width, height, depth, border,
        format, type, bytes) {
      uploads3D.push({ target, level, internalFormat, width, height, depth,
        border, format, type, byteLength: bytes.byteLength });
    },
    texParameteri(target, name, value) { parameters.push([target, name, value]); },
    activeTexture(unit) { activeUnits.push(unit); },
    deleteTexture(texture) { deletedTextures.push(texture); },
  };
  const lookups = new D.Material11LookupTextures(lookupGL);
  assert.deepEqual(pixelStores, [[lookupGL.UNPACK_ALIGNMENT, 1],
    [lookupGL.UNPACK_FLIP_Y_WEBGL, false]]);
  assert.deepEqual(uploads2D.map(upload => [upload.target, upload.width,
    upload.height, upload.byteLength]), Array.from({ length: 6 }, (_, face) =>
    [lookupGL.TEXTURE_CUBE_MAP_POSITIVE_X + face, 64, 64, 64 * 64 * 4]));
  assert.deepEqual(uploads2D.map(upload => upload.byteOffset),
    Array.from({ length: 6 }, (_, face) => face * 64 * 64 * 4),
    'cube uploads use consecutive native face views without copies');
  assert.deepEqual(uploads3D, [{
    target: lookupGL.TEXTURE_3D, level: 0, internalFormat: lookupGL.RGBA8,
    width: 32, height: 32, depth: 32, border: 0,
    format: lookupGL.RGBA, type: lookupGL.UNSIGNED_BYTE,
    byteLength: 32 ** 3 * 4,
  }]);
  for (const parameter of [
    [lookupGL.TEXTURE_CUBE_MAP, lookupGL.TEXTURE_MIN_FILTER, lookupGL.LINEAR],
    [lookupGL.TEXTURE_CUBE_MAP, lookupGL.TEXTURE_MAG_FILTER, lookupGL.LINEAR],
    [lookupGL.TEXTURE_CUBE_MAP, lookupGL.TEXTURE_WRAP_S, lookupGL.REPEAT],
    [lookupGL.TEXTURE_CUBE_MAP, lookupGL.TEXTURE_WRAP_T, lookupGL.REPEAT],
    [lookupGL.TEXTURE_CUBE_MAP, lookupGL.TEXTURE_MAX_LEVEL, 0],
    [lookupGL.TEXTURE_3D, lookupGL.TEXTURE_MIN_FILTER, lookupGL.LINEAR],
    [lookupGL.TEXTURE_3D, lookupGL.TEXTURE_MAG_FILTER, lookupGL.LINEAR],
    [lookupGL.TEXTURE_3D, lookupGL.TEXTURE_WRAP_S, lookupGL.CLAMP_TO_EDGE],
    [lookupGL.TEXTURE_3D, lookupGL.TEXTURE_WRAP_T, lookupGL.CLAMP_TO_EDGE],
    [lookupGL.TEXTURE_3D, lookupGL.TEXTURE_WRAP_R, lookupGL.CLAMP_TO_EDGE],
    [lookupGL.TEXTURE_3D, lookupGL.TEXTURE_MAX_LEVEL, 0],
  ]) assert.ok(parameters.some(actual => actual.every((value, index) => value === parameter[index])));
  assert.deepEqual(lookups.resourceStats(), {
    textures: 2,
    normalizerCubeBytes: D.LEGACY_NORMALIZER_CUBE_BYTE_LENGTH,
    attenuationVolumeBytes: D.LEGACY_ATTENUATION_VOLUME_BYTE_LENGTH,
    estimatedBytes: D.LEGACY_NORMALIZER_CUBE_BYTE_LENGTH +
      D.LEGACY_ATTENUATION_VOLUME_BYTE_LENGTH,
  });
  const bindingStart = bindings.length;
  lookups.bind(5, 6);
  assert.deepEqual(activeUnits.slice(-2), [lookupGL.TEXTURE0 + 5, lookupGL.TEXTURE0 + 6]);
  assert.deepEqual(bindings.slice(bindingStart), [
    [lookupGL.TEXTURE_CUBE_MAP, lookups.normalizerCube],
    [lookupGL.TEXTURE_3D, lookups.attenuationVolume],
  ]);
  lookups.dispose();
  lookups.dispose();
  assert.equal(deletedTextures.length, 2, 'lookup texture disposal is idempotent');
  assert.deepEqual(lookups.resourceStats(), {
    textures: 0, normalizerCubeBytes: 0, attenuationVolumeBytes: 0, estimatedBytes: 0,
  });

  const rendererStats = D.Renderer.prototype.resourceStats.call({
    geometry: { resourceStats: () => ({ gpuBytes: 10, cpuReferencedBytes: 20, shadowBytes: 30 }) },
    textures: { resourceStats: () => ({
      textures: 3, bitmapTextures: 3, dxt5CompressedTextures: 1,
      dxt5FallbackTextures: 0, estimatedBytes: 1000,
    }) },
    material11Lookups: { resourceStats: () => ({
      textures: 2, normalizerCubeBytes: 98304,
      attenuationVolumeBytes: 131072, estimatedBytes: 229376,
    }) },
    masterTarget: { estimatedBytes: () => 100 },
    prelightTarget: { estimatedBytes: () => 50 },
    targets: [], glareTargets: [],
    canvasWidth: 10, canvasHeight: 5, width: 8, height: 4,
    presentationRegion: { x: 1, y: 2, width: 8, height: 4 },
  });
  assert.equal(rendererStats.textures.textures, 5);
  assert.equal(rendererStats.textures.bitmapTextures, 3);
  assert.equal(rendererStats.textures.lookupTextures, 2);
  assert.equal(rendererStats.textures.lookupEstimatedBytes, 229376);
  assert.equal(rendererStats.textures.estimatedBytes, 230376);
  assert.equal(rendererStats.totalEstimatedBytes, 10 + 20 + 30 + 230376 + 150 + 400);
}

// Material objects survive from precalc through playback. Animated Material
// exec replaces their parameter arrays, so the view cache must validate the
// material generation rather than returning the precalc-era state forever.
{
  const renderer = {
    materialViews: new WeakMap(), defaultMaterialViews: new Map(),
    contextLost: false, width: 1, height: 1, canvasWidth: 1, canvasHeight: 1,
    presentationRegion: { x: 0, y: 0, width: 1, height: 1 },
    drawCalls: 0, triangles: 0,
    masterTarget: { color: null }, geometry: { beginFrame() {} },
    resize() {}, clearMasterTarget() {}, renderNode() {}, bindDestination() {}, present() {},
    configureFullscreenState() {}, drawFullscreen() {},
  };
  const material = {
    system: '1.1', parameters: new Array(64).fill(0), textures: [], version: 0,
  };
  const pass = { usage: 'other', state: 'material11-single' };
  material.parameters[28] = 0xff102030;
  const first = D.Renderer.prototype.viewMaterial.call(renderer, material, pass);
  const firstState = D.materialState(first);
  assert.equal(D.materialState(first), firstState,
    'a compiled material view reuses its render-state object');
  const defaultFirst = D.Renderer.prototype.viewMaterial.call(renderer, null, pass);
  const reused = D.Renderer.prototype.viewMaterial.call(renderer, material, pass);
  assert.equal(reused, first);
  const materialViews = renderer.materialViews;
  const defaultMaterialViews = renderer.defaultMaterialViews;
  D.Renderer.prototype.render.call(renderer, { type: 'viewport' }, {});
  assert.equal(renderer.materialViews, materialViews,
    'material views remain weakly cached across renderer frames');
  assert.equal(renderer.defaultMaterialViews, defaultMaterialViews,
    'default material views remain cached across renderer frames');
  assert.equal(D.Renderer.prototype.viewMaterial.call(renderer, material, pass), first);
  assert.equal(D.Renderer.prototype.viewMaterial.call(renderer, null, pass), defaultFirst);
  material.parameters = material.parameters.slice();
  material.parameters[0] = 0x0300;
  material.parameters[28] = 0xffa0b0c0;
  material.version++;
  D.Renderer.prototype.render.call(renderer, { type: 'viewport' }, {});
  const animated = D.Renderer.prototype.viewMaterial.call(renderer, material, pass);
  assert.notEqual(animated, first);
  assert.equal(animated.color, 0xffa0b0c0);
  assert.notEqual(D.materialState(animated), firstState,
    'a material generation change compiles a fresh render state');
  assert.equal(D.materialState(animated).depthTest, true);
}

// D3DCULL_CCW keeps WZ3's authored clockwise faces. The port projection does
// not reflect X or Y, so WebGL must use CW as its front-face classification.
let configuredFrontFace = null;
let configuredDepthBias = null;
let configuredCullFace = null;
const enabledCaps = [], disabledCaps = [], stencilOperations = [];
const windingGL = {
  DEPTH_TEST: 1, CULL_FACE: 2, BLEND: 3, STENCIL_TEST: 4,
  ALWAYS: 5, EQUAL: 6, LESS: 7, LEQUAL: 8, CW: 9, FRONT: 10, BACK: 11,
  ZERO: 12, ONE: 13, SRC_ALPHA: 14, ONE_MINUS_SRC_ALPHA: 15,
  SRC_COLOR: 16, ONE_MINUS_SRC_COLOR: 17, DST_COLOR: 18,
  ONE_MINUS_DST_COLOR: 19, DST_ALPHA: 20, FUNC_ADD: 21, FUNC_REVERSE_SUBTRACT: 22,
  KEEP: 23, POLYGON_OFFSET_FILL: 24, DECR_WRAP: 25, INCR_WRAP: 26,
  enable(cap) { enabledCaps.push(cap); }, disable(cap) { disabledCaps.push(cap); },
  depthFunc() {}, depthMask() {},
  polygonOffset(factor, units) { configuredDepthBias = [factor, units]; },
  frontFace(value) { configuredFrontFace = value; },
  cullFace(value) { configuredCullFace = value; },
  blendEquation() {}, blendFunc() {}, colorMask() {}, stencilMask() {}, stencilFunc() {}, stencilOp() {},
  stencilOpSeparate(...values) { stencilOperations.push(values); },
  useProgram() {}, bindVertexArray() {}, uniformMatrix4fv() {}, uniform3fv() {},
};
D.Renderer.prototype.configureMaterialState.call({ gl: windingGL }, {
  baseFlags: 0x0300, blend: 0, system: '1.1',
});
assert.equal(configuredFrontFace, windingGL.CW);
D.Renderer.prototype.configureMaterialState.call({ gl: windingGL }, {
  baseFlags: 0x0308, blend: 0, system: '1.1',
});
assert.deepEqual(configuredDepthBias, [1 / 65536, 256]);
D.Renderer.prototype.configureMaterialState.call({ gl: windingGL }, {
  baseFlags: 0x0310, blend: 0, system: '1.1',
});
assert.deepEqual(configuredDepthBias, [-1 / 65536, -256]);
D.Renderer.prototype.configureMaterialState.call({ gl: windingGL }, {
  baseFlags: 0x0318, blend: 0, system: '1.1',
});
assert.deepEqual(configuredDepthBias, [-1 / 65536, -256], 'foreground bias wins like native state emission');
D.Renderer.prototype.configureMaterialState.call({ gl: windingGL }, {
  baseFlags: 0x0304, blend: 0, system: '1.1',
});
assert.equal(configuredCullFace, windingGL.FRONT);
const doubleSidedDisableStart = disabledCaps.length;
D.Renderer.prototype.configureMaterialState.call({ gl: windingGL }, {
  baseFlags: 0x0302, blend: 0, system: '1.1',
});
assert.deepEqual(disabledCaps.slice(doubleSidedDisableStart, doubleSidedDisableStart + 2),
  [windingGL.POLYGON_OFFSET_FILL, windingGL.CULL_FACE]);

const shadowStateGeometry = {
  positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
  indices: new Uint16Array([0, 1, 2]), groups: [{ start: 0, count: 3 }],
  shadowTopologies: new Map(),
};
const shadowStateTopology = D.prepareShadowTopology(shadowStateGeometry);
shadowStateGeometry.shadowTopologies.set('0:3', shadowStateTopology);
const shadowStencilStart = stencilOperations.length;
const shadowDisableStart = disabledCaps.length;
D.Renderer.prototype.drawShadowItem.call({
  gl: windingGL, shadowUniforms: { uViewProjection: 1, uLightPosition: 2 },
  shadowProgram: {}, shadowVAO: {},
  instanceMatrices: () => new Float32Array(0),
}, {
  light: { position: new Float32Array([0, 0, -1]) },
  groups: shadowStateGeometry.groups, geometry: shadowStateGeometry,
}, { viewProjection: new Float32Array(16) });
assert.equal(configuredFrontFace, windingGL.CW);
assert.ok(disabledCaps.slice(shadowDisableStart).includes(windingGL.CULL_FACE));
assert.ok(disabledCaps.slice(shadowDisableStart).includes(windingGL.POLYGON_OFFSET_FILL));
assert.deepEqual(stencilOperations.slice(shadowStencilStart), [
  [windingGL.FRONT, windingGL.KEEP, windingGL.DECR_WRAP, windingGL.KEEP],
  [windingGL.BACK, windingGL.KEEP, windingGL.INCR_WRAP, windingGL.KEEP],
]);
const shadowZPassStart = stencilOperations.length;
D.Renderer.prototype.drawShadowItem.call({
  gl: windingGL, shadowUniforms: { uViewProjection: 1, uLightPosition: 2 },
  shadowProgram: {}, shadowVAO: {},
  instanceMatrices: () => new Float32Array(0),
}, {
  light: { position: new Float32Array([0, 0, -1]) },
  groups: shadowStateGeometry.groups, geometry: shadowStateGeometry,
  shadowZFail: false,
}, { viewProjection: new Float32Array(16) });
assert.deepEqual(stencilOperations.slice(shadowZPassStart), [
  [windingGL.FRONT, windingGL.KEEP, windingGL.KEEP, windingGL.INCR_WRAP],
  [windingGL.BACK, windingGL.KEEP, windingGL.KEEP, windingGL.DECR_WRAP],
], 'native z-pass updates stencil on depth pass with the opposite face signs');

const insertBlendCalls = [];
const insertGL = {
  DEPTH_TEST: 1, CULL_FACE: 2, POLYGON_OFFSET_FILL: 3, STENCIL_TEST: 4,
  BLEND: 5, FUNC_ADD: 6, ZERO: 7, SRC_COLOR: 8, DST_ALPHA: 9, ONE: 10,
  disable() {}, depthMask() {}, enable() {}, blendEquation() {}, colorMask() {},
  blendFunc(source, destination) { insertBlendCalls.push([source, destination]); },
};
const insertDraws = [];
const insertRenderer = {
  gl: insertGL,
  textures: { fallbackTexture: () => 'white' },
  drawFullscreen(image, mode) { insertDraws.push([image, mode]); },
};
D.Renderer.prototype.drawMaterial11Insert.call(insertRenderer, 'clear-destination-alpha');
D.Renderer.prototype.drawMaterial11Insert.call(insertRenderer, 'add-destination-alpha');
assert.deepEqual(insertBlendCalls, [
  [insertGL.ZERO, insertGL.SRC_COLOR],
  [insertGL.DST_ALPHA, insertGL.ONE],
]);
assert.deepEqual(insertDraws, [['white', 7], ['white', 7]]);

const mesh = {
  prepare() {
    return {
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      indices: new Uint16Array([0, 1, 2]),
      colors: new Uint32Array([0xff112233, 0x80445566, 0xffffffff]),
      groups: [{ material: 2, start: 0, count: 3 }], materials: [null, null, { material: null }],
    };
  },
};
const geometry = D.normalizePreparedGeometry(mesh);
assert.equal(geometry.vertexCount, 3);
assert.deepEqual(Array.from(geometry.colors.subarray(0, 8)), [17, 34, 51, 255, 68, 85, 102, 128]);
assert.deepEqual(JSON.parse(JSON.stringify(geometry.groups)), [{ material: 2, start: 0, count: 3 }]);
assert.deepEqual(Array.from(geometry.normals), [0, 1, 0, 0, 1, 0, 0, 1, 0]);
const dualUVGeometry = D.normalizePreparedGeometry({
  positions: new Float32Array([0, 0, 0, 1, 0, 0]),
  uv0: new Float32Array([0.1, 0.2, 0.3, 0.4]),
  uv1: new Float32Array([0.5, 0.6, 0.7, 0.8]),
  indices: new Uint16Array([0, 1]),
});
assert.deepEqual(Array.from(dualUVGeometry.uvs), Array.from(new Float32Array([0.1, 0.2, 0.3, 0.4])));
assert.deepEqual(Array.from(dualUVGeometry.uv1), Array.from(new Float32Array([0.5, 0.6, 0.7, 0.8])));

const topologyDiagnostic = D.geometryTopologyStats({
  positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
  normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
  indices: new Uint16Array([
    0, 1, 2, 0, 1, 2, 0, 2, 1, 0, 0, 1,
  ]),
});
assert.equal(topologyDiagnostic.degenerateTriangles, 1);
assert.equal(topologyDiagnostic.duplicateTriangles, 2);
assert.equal(topologyDiagnostic.oppositeDuplicateTriangles, 1);
assert.equal(topologyDiagnostic.sameOrientationDuplicateTriangles, 1);
assert.equal(topologyDiagnostic.normalAlignedWinding, 2);
assert.equal(topologyDiagnostic.normalOpposedWinding, 1);
assert.equal(topologyDiagnostic.unexpectedWinding, 1);

// Once both orientations exist, later copies must be compared against both.
// Comparing only to the first triangle incorrectly classified CW,CCW,CCW as
// having no same-winding overlap (the exact z-fighting candidate).
const mixedDuplicateDiagnostic = D.geometryTopologyStats({
  positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
  normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
  indices: new Uint16Array([0, 1, 2, 0, 2, 1, 0, 2, 1]),
});
assert.equal(mixedDuplicateDiagnostic.duplicateTriangles, 2);
assert.equal(mixedDuplicateDiagnostic.oppositeDuplicateTriangles, 1);
assert.equal(mixedDuplicateDiagnostic.sameOrientationDuplicateTriangles, 1);

// Exact position duplicates are classified by their closest earlier match.
// This distinguishes redundant triangles in one draw from UV/normal variants
// and separate material layers, while retaining the old near-coincident count.
const duplicateMaterialA = { kind: 'material', system: '1.1', parameters: [] };
const duplicateMaterialB = { kind: 'material', system: '1.1', parameters: [] };
const classifiedPositions = [], classifiedNormals = [], classifiedUVs = [], classifiedIndices = [];
const addClassifiedTriangle = (offset = 0, uvOffset = 0) => {
  const first = classifiedPositions.length / 3;
  classifiedPositions.push(offset, offset, 0, 1 + offset, offset, 0, offset, 1 + offset, 0);
  classifiedNormals.push(0, 0, 1, 0, 0, 1, 0, 0, 1);
  classifiedUVs.push(uvOffset, uvOffset, 1 + uvOffset, uvOffset, uvOffset, 1 + uvOffset);
  classifiedIndices.push(first, first + 1, first + 2);
};
addClassifiedTriangle(); // reference
addClassifiedTriangle(); // same group, identical attributes
addClassifiedTriangle(0, 2); // same group, UV variant
addClassifiedTriangle(); // separate group, same material
addClassifiedTriangle(); // separate material layer
addClassifiedTriangle(4e-6); // within diagnostic tolerance, but not exact
const classifiedDuplicates = D.geometryTopologyStats({
  positions: new Float32Array(classifiedPositions),
  normals: new Float32Array(classifiedNormals),
  uvs: new Float32Array(classifiedUVs),
  uv1: new Float32Array(classifiedUVs),
  indices: new Uint16Array(classifiedIndices),
  groups: [
    { material: duplicateMaterialA, start: 0, count: 9 },
    { material: duplicateMaterialA, start: 9, count: 3 },
    { material: duplicateMaterialB, start: 12, count: 6 },
  ],
});
assert.equal(classifiedDuplicates.sameOrientationDuplicateTriangles, 5);
assert.equal(classifiedDuplicates.exactSameOrientationDuplicateTriangles, 4);
assert.equal(classifiedDuplicates.nearOnlySameOrientationDuplicateTriangles, 1);
assert.equal(classifiedDuplicates.exactSameGroupIdenticalAttributeTriangles, 1);
assert.equal(classifiedDuplicates.exactSameGroupAttributeVariantTriangles, 1);
assert.equal(classifiedDuplicates.exactCrossGroupSameMaterialTriangles, 1);
assert.equal(classifiedDuplicates.exactCrossMaterialTriangles, 1);
assert.equal(classifiedDuplicates.exactDegenerateSameOrientationTriangles, 0);
assert.equal(
  classifiedDuplicates.exactSameGroupIdenticalAttributeTriangles +
  classifiedDuplicates.exactSameGroupAttributeVariantTriangles +
  classifiedDuplicates.exactCrossGroupSameMaterialTriangles +
  classifiedDuplicates.exactCrossMaterialTriangles +
  classifiedDuplicates.exactDegenerateSameOrientationTriangles,
  classifiedDuplicates.exactSameOrientationDuplicateTriangles,
);

const material11 = {
  kind: 'material', system: '1.1', parameters: new Array(64).fill(0), textures: [],
};
material11.parameters[0] = 0x2300;
material11.parameters[11] = 0x10; // sMCS_TEX0 = SET
material11.parameters[28] = 0xffabcdef;
material11.parameters[32] = 24;
material11.parameters[51] = 48;
material11.parameters[52] = 0xff010203;
let view = D.materialView(material11, { usage: 'other' });
assert.equal(view.color, 0xffabcdef);
assert.equal(view.specularPower, 24);
assert.equal(D.renderMode(view), 7);
const chainedOwner = {
  kind: 'material', system: '1.1', parameters: material11.parameters.slice(), textures: [],
};
chainedOwner.parameters[0] = 0x0300;
chainedOwner.parameters[28] = 0xff102938;
const chainedView = D.materialView(material11,
  { usage: 'other', material: chainedOwner });
assert.equal(chainedView.baseFlags, 0x0300,
  'a Material_Add pass retains its upstream compiled state');
assert.equal(chainedView.color, 0xffabcdef,
  'downstream Exec words update every pass in the combined Material_Add result');
const compiledOwner = {
  kind: 'material', system: '1.1',
  initialParameters: chainedOwner.parameters.slice(),
  parameters: chainedOwner.parameters.slice(), textures: [],
};
compiledOwner.parameters[0] = 0;
compiledOwner.parameters[28] = 0xff556677;
const compiledView = D.materialView(compiledOwner,
  { usage: 'other', material: compiledOwner });
assert.equal(compiledView.baseFlags, 0x0300,
  'Material11 base flags remain compiled even if animation touches the source word');
assert.equal(compiledView.color, 0xff556677,
  'Material11 Exec still updates its dynamic Color block');
const textureSet = D.evaluateMaterial11Combiner(view, {
  textures: [new Float32Array([0.75, 0.5, 0.25, 1])],
});
assert.deepEqual(Array.from(textureSet), [0.75, 0.5, 0.25, 1]);
const colorFadeIgnoresAlphaInvert = D.evaluateMaterial11Combiner({
  combiners: [0, 0x114], // Color0 SET faded by Tex0 alpha; bit 8 is ignored here.
  colors: [0xffff0000], alphaCombiner: 0,
}, { textures: [new Float32Array([1, 1, 1, 0.25])] });
assert.deepEqual(Array.from(colorFadeIgnoresAlphaInvert), [0.25, 0, 0, 1],
  'sMCA_INVERTA is not encoded into Material11 color-combiner AlphaSrc');
const mix01 = D.evaluateMaterial11Combiner({
  combiners: [0, 0x10], colors: [0x80ffffff],
  // High alpha nibble 14 selects the source's dedicated TEX0/TEX1/COL0 path;
  // low source and both inversion flags are intentionally bypassed.
  alphaCombiner: 0x03ed,
}, { textures: [new Float32Array([1, 1, 1, 0.25]),
  new Float32Array([1, 1, 1, 0.5])] });
assert.ok(Math.abs(mix01[3] - 0.75 * (128 / 255)) < 1e-7);
assert.match(D.materialFragmentSource,
  /if \(alphaB == 14\)[\s\S]*?\(t0\.a \+ t1\.a\) \* uMaterialColors\[0\]\.a/,
  'Material11 MIX01 alpha follows the released two-instruction special case');

// GenMesh::Init and GenMinMesh::AddCluster resolve material slot 1 to the
// player's GenOverlayManager->DefaultMat. It is a per-light Material11 pass,
// not a constant unlit fallback: the distinction keeps unlinked Font3D text
// below glare threshold where the authored light faces away or attenuates.
const defaultMeshMaterialView = D.materialView({
  kind: 'material', system: 'default', passes: [],
}, { usage: 'other' });
assert.equal(defaultMeshMaterialView.system, '1.1');
assert.equal(defaultMeshMaterialView.usage, 'light');
assert.equal(defaultMeshMaterialView.baseFlags, 0x0300);
assert.equal(defaultMeshMaterialView.lightFlags, 0x0005);
assert.equal(defaultMeshMaterialView.color, 0x00c0c0c0);
assert.equal(defaultMeshMaterialView.colors[0], 0x00c0c0c0);
assert.equal(defaultMeshMaterialView.colors[1], 0x00404040);
assert.equal(defaultMeshMaterialView.specularPower, 32);
assert.equal(defaultMeshMaterialView.specularStrength, 1);
assert.equal(D.renderMode(defaultMeshMaterialView), 2);
assert.equal(D.material11LightConstant(
  defaultMeshMaterialView.color, 0xffffffff, 1,
)[3], 0, 'DefaultMat Color0 alpha suppresses its PS1.1 specular channel');
assert.deepEqual(JSON.parse(JSON.stringify(D.materialState(defaultMeshMaterialView))), {
  depthTest: true, depthWrite: true, depthFunc: 'lequal', colorWrite: true,
  cull: 'back', blend: null, stencilTest: false,
});

// Production wall-label Material11 op 2722 is a single unlit texture pass.
// Its mode-3 sampler is trilinear/anisotropic (covered by test_gl), while its
// smooth blend makes the direct+glow atlas self-emissive over the facade.
const debrisLabelMaterial = {
  kind: 'material', system: '1.1', parameters: new Array(64).fill(0), textures: [{}],
};
debrisLabelMaterial.parameters[0] = 0x5200; // ZREAD | smooth blend
debrisLabelMaterial.parameters[4] = 3; // Material11 anisotropic filter
debrisLabelMaterial.parameters[11] = 0x10; // Tex0 SET
debrisLabelMaterial.parameters[23] = 0; // output alpha ONE
debrisLabelMaterial.parameters[48] = 0x201; // single-pass other material
const debrisLabelView = D.materialView(debrisLabelMaterial, { usage: 'other' });
assert.equal(D.renderMode(debrisLabelView), 7);
assert.equal(debrisLabelView.material11Combiner, true);
assert.equal(debrisLabelView.slotTextureFlags[0], 3);
assert.deepEqual(Array.from(D.evaluateMaterial11Combiner(debrisLabelView, {
  textures: [new Float32Array([1, 0.875, 0.75, 0.625])],
})), [1, 0.875, 0.75, 1]);
assert.deepEqual(JSON.parse(JSON.stringify(D.materialState(debrisLabelView))), {
  depthTest: true, depthWrite: false, depthFunc: 'lequal', colorWrite: true,
  cull: 'back', blend: ['one', 'one-minus-src-color', 'add'], stencilTest: false,
});
assert.equal(D.groupRenderPass({ materials: [{ pass: 9 }] }, { materialIndex: 0 }), 9);

view = D.materialView(material11, { usage: 'light' });
assert.equal(view.color, 0xff010203);
assert.equal(view.specularPower, 48);
assert.equal(D.renderMode(view), 7);

material11.parameters[48] = 0x0f;
material11.parameters[53] = 0xff102030;
view = D.materialView(material11, { usage: 'base', state: 'material11-base' });
assert.equal(view.baseFlags, 0x020300);
assert.equal(view.color, 0xff102030);
assert.equal(view.vertexColorMode, 4);
assert.equal(view.generatedBaseAlpha, 'zero');
assert.deepEqual(Array.from(view.textureMap), [null, null, null, null]);
const m11BaseZeroView = view;
const m11AlphaBaseMaterial = {
  ...material11,
  parameters: material11.parameters.slice(),
  textures: [{}],
};
m11AlphaBaseMaterial.parameters[48] |= 0x0400;
const m11BaseTextureAlphaView = D.materialView(m11AlphaBaseMaterial,
  { usage: 'base', state: 'material11-base' });
assert.equal(m11BaseTextureAlphaView.generatedBaseAlpha, 'texture0');
assert.deepEqual(Array.from(m11BaseTextureAlphaView.textureMap), [0, null, null, null]);
assert.match(D.materialFragmentSource,
  /uVertexColorMode == 4[\s\S]*?uBaseColor\.rgb \+ vColor\.rgb, uBaseColor\.a \* t0\.a/,
  'generated Material11 BASE uses Tex0 only for alpha, not for RGB');
const m11AlphaPostView = D.materialView(m11AlphaBaseMaterial,
  { usage: 'postlight', state: 'material11-postlight' });
assert.equal(m11AlphaPostView.colors[3], 0xff808080,
  'alpha-tested postlight Exec forces native Color[3] after its animated copy');
view = D.materialView(material11, { usage: 'light', state: 'material11-light' });
assert.equal(view.baseFlags, 0x302280);
assert.deepEqual(Array.from(view.textureMap), [null, 4, null, null]);
assert.equal(view.specularStrength, 0);
assert.equal(D.materialState(view).depthFunc, 'equal');
assert.equal(D.materialState(view).stencilTest, true);
assert.deepEqual(Array.from(D.materialState(view).blend), ['one', 'one', 'add']);

// The one production Material11 specular path (op 9765, MultiFlags 0x20b)
// uses the PS1.1 multiply approximation and framebuffer alpha, not Blinn pow
// added directly to RGB. Its authored diffuse alpha is the specular amplitude.
material11.parameters[48] = 0x20b;
material11.parameters[51] = 32;
material11.parameters[52] = 0x20ffffff;
view = D.materialView(material11, { usage: 'light', state: 'material11-light' });
assert.equal(view.specularStrength, 1);
assert.equal((view.color >>> 24) & 255, 0x20);
const m11PostlightView = D.materialView(material11,
  { usage: 'postlight', state: 'material11-postlight' });
assert.equal(m11PostlightView.alphaCombiner, 0x0c);
assert.deepEqual(Array.from(D.materialState(m11PostlightView).blend),
  ['dst-color', 'src-color', 'add']);
// src alpha 1/2 with DESTCOLOR,SRCCOLOR gives .5*dstA + dstA*.5 = dstA.
assert.equal(0.5 * 0.375 + 0.375 * 0.5, 0.375);
assert.ok(Math.abs(D.evaluateMaterial11Specular(0.9, 32) - 0.023106477086107406) < 1e-12);
assert.equal(D.evaluateMaterial11Specular(1, 32), 1);
assert.equal(D.evaluateMaterial11Specular(0, 32), 0);
const highwayWarmC0 = D.material11LightConstant(0xffffe8b8, 0xffffe090, 1.75);
assert.deepEqual(Array.from(highwayWarmC0).slice(0, 2), [1, 1],
  'PS1.1 c0 clips the highway light red/green products before shader arithmetic');
assert.ok(Math.abs(highwayWarmC0[2] - 0.7130796) < 1e-6);
assert.equal(highwayWarmC0[3], 1);
assert.match(D.materialFragmentSource,
  /diffuseLighting = \(uMaterial11 != 0\s+\? uM11LightConstant\.rgb : uLightColor\[0\]\.rgb\)/);
assert.match(D.materialFragmentSource,
  /result = vec4\(diffuseLighting,\s+material11SpecularLighting \* uSpecularStrength\)/);

const m11InsertMaterial = { system: '1.1', multiFlags: 0x20b, parameters: [] };
const m20InsertMaterial = { system: '2.0', parameters: [] };
assert.equal(D.materialInsertKind(m11InsertMaterial), 'material11');
assert.equal(D.materialInsertKind({ system: '1.1', multiFlags: 0x201 }), null);
let insertPlan = D.materialInsertPlan([
  { renderPass: 4, material: m11InsertMaterial },
  { renderPass: 4, material: m20InsertMaterial },
  { renderPass: 5, material: m11InsertMaterial },
]);
assert.equal(insertPlan.get(4), 'material20', 'Material20 has the native higher insert priority');
assert.equal(insertPlan.get(5), 'material11');
assert.equal(D.material11InsertAction('material11', 'base'), 'clear-destination-alpha');
assert.equal(D.material11InsertAction('material11', 'postlight'), 'add-destination-alpha');
assert.equal(D.material11InsertAction('material20', 'postlight'), null);
const insertTracker = D.createMaterialInsertTracker([
  { renderPass: 5, material: m11InsertMaterial, pass: { usage: 'base' } },
  { renderPass: 5, material: m11InsertMaterial, pass: { usage: 'light' }, light: { id: 1 } },
  { renderPass: 5, material: m11InsertMaterial, pass: { usage: 'postlight' } },
]);
assert.equal(insertTracker.transition(
  { renderPass: 5, material: m11InsertMaterial, pass: { usage: 'base' } }), null);
assert.equal(insertTracker.transition(
  { renderPass: 5, material: m11InsertMaterial, pass: { usage: 'light' }, light: { id: 1 } }),
  'clear-destination-alpha');
assert.equal(insertTracker.transition(
  { renderPass: 5, material: m11InsertMaterial, pass: { usage: 'postlight' } }), null);
assert.equal(insertTracker.finish(), 'add-destination-alpha',
  'the final AfterUsage must flush destination-alpha specular at end of the list');

const materialUniformCalls = {
  oneI: new Map(), oneF: new Map(), three: new Map(), four: new Map(), oneFV: new Map(),
  matrixFour: new Map(), lightArrayCalls: 0, lightArrayFloats: 0,
};
const lightArrayUniforms = new Set([
  'uLightPosition[0]', 'uLightAttenuation[0]', 'uLightColor[0]', 'uLightSpecular[0]',
]);
function materialUniformSlice(value, sourceOffset = 0,
    sourceLength = value.length - sourceOffset) {
  return Array.from(value).slice(sourceOffset, sourceOffset + sourceLength);
}
function recordMaterialUniform(map, location, value, sourceOffset, sourceLength) {
  const uploaded = materialUniformSlice(value, sourceOffset, sourceLength);
  map.set(location, uploaded);
  if (lightArrayUniforms.has(location)) {
    materialUniformCalls.lightArrayCalls++;
    materialUniformCalls.lightArrayFloats += uploaded.length;
  }
}
function resetMaterialLightUploads() {
  materialUniformCalls.lightArrayCalls = 0;
  materialUniformCalls.lightArrayFloats = 0;
}
const materialBindGL = {
  TEXTURE4: 4, TEXTURE_2D: 3553,
  uniform1i(location, value) { materialUniformCalls.oneI.set(location, value); },
  uniform1f(location, value) { materialUniformCalls.oneF.set(location, value); },
  uniform1fv(location, value, sourceOffset, sourceLength) {
    recordMaterialUniform(materialUniformCalls.oneFV,
      location, value, sourceOffset, sourceLength);
  },
  uniform3fv(location, value) { materialUniformCalls.three.set(location, Array.from(value)); },
  uniform3f(location, x, y, z) { materialUniformCalls.three.set(location, [x, y, z]); },
  uniform4fv(location, value, sourceOffset, sourceLength) {
    recordMaterialUniform(materialUniformCalls.four,
      location, value, sourceOffset, sourceLength);
  },
  uniformMatrix4fv(location, transpose, value) {
    assert.equal(transpose, false);
    materialUniformCalls.matrixFour.set(location, Array.from(value));
  }, uniform2iv() {}, uniform1iv() {},
  uniform2f() {}, uniform4iv() {}, activeTexture() {}, bindTexture() {},
};
const identity = D.mat4Identity();
const legacyModel = Float32Array.from(identity);
legacyModel[0] = 2; legacyModel[5] = 3; legacyModel[10] = 4;
legacyModel[12] = 10; legacyModel[13] = 20; legacyModel[14] = 30;
assert.deepEqual(Array.from(D.legacyTransRVector(legacyModel, [14, 26, 38], true)),
  [8, 18, 32]);
assert.deepEqual(Array.from(D.legacyTransRVector(legacyModel, [1, 2, 3], false)),
  [2, 6, 12]);
const rigidLegacyModel = D.mat4SRT(new Float32Array([
  1, 1, 1, 0.125, -0.2, 0.33, 4, -5, 6,
]));
const rigidInverse = D.mat4Inverse(rigidLegacyModel);
const nativePoint = D.legacyTransRVector(rigidLegacyModel, [9, 7, -2], true);
const inversePoint = D.mat4TransformPoint(rigidInverse, [9, 7, -2, 1]);
const nativeDirection = D.legacyTransRVector(rigidLegacyModel, [0.25, -0.5, 0.75], false);
const inverseDirection = D.mat4TransformPoint(rigidInverse, [0.25, -0.5, 0.75, 0]);
for (let axis = 0; axis < 3; axis++) {
  assert.ok(Math.abs(nativePoint[axis] - inversePoint[axis]) < 2e-5,
    'TransR matches a rigid inverse for rotated points');
  assert.ok(Math.abs(nativeDirection[axis] - inverseDirection[axis]) < 2e-5,
    'TransR matches inverse rotation for directions');
}
const materialCameraSpace = Float32Array.from(identity);
materialCameraSpace[12] = 14; materialCameraSpace[13] = 26; materialCameraSpace[14] = 38;
const materialScratch = {
  baseColor: new Float32Array(4), color: new Float32Array(4),
  materialColors: new Float32Array(16), specularColor: new Float32Array(3),
  ambient: new Float32Array(3), materialCameraPosition: new Float32Array(3),
  fogColor: new Float32Array(3), lightColor: new Float32Array(3),
  lightPositions: new Float32Array(64), lightAttenuation: new Float32Array(64),
  lightColors: new Float32Array(64),
  lightSpecular: new Float32Array(16), m11LightConstant: new Float32Array(4),
  selectedTextures: new Array(4),
  m20SamplerFlags: new Int32Array(4), m20SamplerScales: new Float32Array(4),
  m11SamplerFlags: new Int32Array(4), m11SamplerScales: new Float32Array(4),
  identityUVTransform: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0]),
  identityMatrix: identity, m11WorldToModel: new Float32Array(16),
  emptyCombiners: new Int32Array(13), emptyDetailOps: new Int32Array(2),
};
const signedBindView = {
  ...view,
  textures: [null, { kind: 'bitmap', format: 5 }],
  textureMap: [null, 1, null, null],
};
const materialBindingRenderer = {
  gl: materialBindGL,
  uniforms: new Proxy({}, { get: (_target, property) => String(property) }),
  materialScratch,
  textures: { bind() {}, fallbackTexture: () => 'white' },
  currentPrelightTexture: null,
};
resetMaterialLightUploads();
D.Renderer.prototype.bindMaterial.call(materialBindingRenderer, signedBindView, {
  ambientLight: 0xff000000,
  camera: { cameraSpace: materialCameraSpace, fogColor: 0xff000000, fogStart: 0, fogEnd: 100 },
}, { view: identity }, [{
  kind: 'point', position: [11, 22, 33], range: 10, amplify: 2, color: 0x80ffffff,
}], legacyModel);
assert.equal(materialUniformCalls.lightArrayCalls, 4,
  'a light pass uploads exactly the four active-light arrays');
assert.equal(materialUniformCalls.lightArrayFloats, 13,
  'one active light uploads 13 floats instead of all 208 scratch floats');
assert.equal(materialScratch.lightPositions.length + materialScratch.lightAttenuation.length +
  materialScratch.lightColors.length + materialScratch.lightSpecular.length, 208);
assert.equal(materialUniformCalls.four.get('uLightPosition[0]').length, 4);
assert.equal(materialUniformCalls.four.get('uLightAttenuation[0]').length, 4);
assert.equal(materialUniformCalls.four.get('uLightColor[0]').length, 4);
assert.equal(materialUniformCalls.oneFV.get('uLightSpecular[0]').length, 1);
assert.equal(materialUniformCalls.oneI.get('uM11MultipassLight'), 1);
assert.equal(materialUniformCalls.oneI.has('uSignedTextureMask'), false,
  'signed normalization is texture storage semantics, not a shader mask');
assert.equal(materialUniformCalls.oneF.get('uSpecularStrength'), 1);
assert.ok(Math.abs(materialUniformCalls.four.get('uBaseColor')[3] - 0x20 / 255) < 1e-7);
assert.ok(Math.abs(materialUniformCalls.oneFV.get('uLightSpecular[0]')[0] - 2 * 0x80 / 255) < 1e-7);
assert.deepEqual(materialUniformCalls.four.get('uM11LightConstant').slice(0, 3), [1, 1, 1]);
assert.ok(Math.abs(materialUniformCalls.four.get('uM11LightConstant')[3] -
  (0x20 / 255) * (0x80 / 255) * 2) < 1e-7,
  'Material11 uploads the combined and PS1.1-clamped c0 alpha');
assert.equal(materialUniformCalls.oneI.get('uLegacyLightingMode'), 1,
  'static legacy light passes use the native unconditioned N/T variant');
assert.equal(materialUniformCalls.oneI.get('uConditionLegacyBasis'), 0);
assert.deepEqual(materialUniformCalls.matrixFour.get('uM11ModelToWorld'),
  Array.from(legacyModel));
assert.deepEqual(materialUniformCalls.three.get('uCameraPosition'), [14, 26, 38],
  'the world camera remains available to environment and fog paths');
assert.deepEqual(materialUniformCalls.three.get('uMaterialCameraPosition'), [8, 18, 32]);
assert.deepEqual(materialUniformCalls.four.get('uLightPosition[0]').slice(0, 4), [2, 6, 12, 1]);
assert.deepEqual(materialUniformCalls.four.get('uLightAttenuation[0]').slice(0, 3), [2, 6, 12]);
assert.ok(Math.abs(materialUniformCalls.four.get('uLightAttenuation[0]')[3] - 0.1) < 1e-7);
D.Renderer.prototype.bindMaterial.call(materialBindingRenderer, signedBindView, {
  ambientLight: 0xff000000,
  camera: { cameraSpace: materialCameraSpace, fogColor: 0xff000000, fogStart: 0, fogEnd: 100 },
}, { view: identity }, [{
  kind: 'directional', direction: [1, 2, 3], position: [14, 26, 38],
  range: 4096, amplify: 1, color: 0xffffffff,
}], legacyModel);
assert.deepEqual(materialUniformCalls.four.get('uLightPosition[0]').slice(0, 4), [2, 6, 12, 0]);
assert.deepEqual(materialUniformCalls.four.get('uLightAttenuation[0]').slice(0, 3), [2, 6, 12],
  'Material11 directional attenuation is centered on its direction constant like native');
assert.ok(Math.abs(materialUniformCalls.four.get('uLightAttenuation[0]')[3] - 1 / 4096) < 1e-12);
const singleLitParameters = new Array(64).fill(0);
singleLitParameters[0] = 0x0200;
singleLitParameters[2] = 5; // sMLF_BUMPX.
singleLitParameters[48] = 5 << 20; // authored ENGU_LIGHT phase, no generated multipass.
singleLitParameters[52] = 0x80ffffff;
const singleLitM11 = {
  kind: 'material', system: '1.1', parameters: singleLitParameters,
  initialParameters: singleLitParameters.slice(), textures: [],
};
const singleLitM11View = D.materialView(singleLitM11,
  { usage: 'light', state: 'material11-single' });
assert.equal(D.renderMode(singleLitM11View), 2);
assert.equal(singleLitM11View.specularStrength, 1,
  'Material11 light alpha already contains its authored specular amplitude');
D.Renderer.prototype.bindMaterial.call(materialBindingRenderer, singleLitM11View, {
  ambientLight: 0x00ffffff,
  camera: { cameraSpace: materialCameraSpace, fogColor: 0xff000000,
    fogStart: 0, fogEnd: 100 },
}, { view: identity }, [{
  kind: 'point', position: [11, 22, 33], range: 10, amplify: 1,
  color: 0xffffffff,
}], legacyModel);
assert.equal(materialUniformCalls.oneI.get('uM11MultipassLight'), 1,
  'all compiled Material11 light shaders use diffuse-RGB/specular-alpha output semantics');
// material11.vsh:87-115,242-280 and material20_light.vsh:51-111 emit raw
// tangent-space L/H/attenuation at vertices. M20 normalizes and attenuates
// analytically; M11 samples the released generated cube and volume lookups.
assert.doesNotMatch(D.materialVertexSource, /vMaterial(?:Position|Normal|Tangent)/);
assert.doesNotMatch(D.materialFragmentSource, /vMaterial(?:Position|Normal|Tangent)/);
assert.match(D.materialVertexSource,
  /vLegacyLight = vec3\(dot\(lightVector, legacyNormal\),\s+dot\(lightVector, legacyBitangent\), dot\(lightVector, legacyTangent\)\)/);
assert.match(D.materialVertexSource,
  /vec3 halfwayVector = normalize\(lightVector\) \+ eyeVector/);
assert.match(D.materialVertexSource,
  /vec3 legacyAttenuationVector =\s+\(modelPosition\.xyz - uLightAttenuation\[0\]\.xyz\) \*\s+uLightAttenuation\[0\]\.w;[\s\S]*?vLegacyAttenuation = uMaterial11 != 0\s+\? legacyAttenuationVector \* 0\.5 \+ vec3\(0\.5\)\s+: legacyAttenuationVector/);
assert.match(D.materialVertexSource,
  /if \(uLegacyLightingMode == 2\) \{[\s\S]*?legacyNormal = normalize\(legacyNormal\);[\s\S]*?legacyTangent = normalize/);
assert.match(D.materialFragmentSource,
  /uniform samplerCube uM11NormalizerCube;[\s\S]*uniform sampler3D uM11AttenuationVolume;/);
assert.match(D.materialFragmentSource, /precision highp sampler3D;/,
  'the Material11 attenuation volume declares fragment-sampler precision for ANGLE');
assert.match(D.materialFragmentSource,
  /vec3 lightDirection = uMaterial11 != 0[\s\S]*?texture\(uM11NormalizerCube, vLegacyLight\)\.rgb \* 2\.0 - 1\.0[\s\S]*?: normalize\(vLegacyLight\)/);
assert.match(D.materialFragmentSource,
  /vec3 halfwayDirection = uMaterial11 != 0[\s\S]*?texture\(uM11NormalizerCube, vLegacyHalfway\)\.rgb \* 2\.0 - 1\.0[\s\S]*?: normalize\(vLegacyHalfway\)/);
assert.match(D.materialFragmentSource,
  /radiusSquared = clamp\(dot\(vLegacyAttenuation,\s+vLegacyAttenuation\), 0\.0, 1\.0\)/);
assert.match(D.materialFragmentSource,
  /float attenuation = uMaterial11 != 0[\s\S]*?texture\(uM11AttenuationVolume, vLegacyAttenuation\)\.r[\s\S]*?: 1\.0 - radiusSquared/);
assert.doesNotMatch(D.materialFragmentSource, /legacyMaterialSpace|surfacePosition/,
  'legacy lighting is no longer reconstructed per fragment');
assert.match(D.materialFragmentSource,
  /uMaterial20 != 0 \? material20AttenuationAlpha : albedo\.a/,
  'Material20 light alpha retains native saturated radial distance squared');
assert.match(D.materialFragmentSource,
  /if \(uMaterial20 != 0 && \(uM20Flags & 8\) != 0\) result \*= vColor;/,
  'Material20 COLOR0 multiplies diffuse, specular and radial alpha after lighting');
assert.ok(D.materialFragmentSource.indexOf('litColor = clamp(litColor, 0.0, 1.0)') <
  D.materialFragmentSource.indexOf('result *= vColor'),
  'Material20 attenuation MAD saturates before its final COLOR0 multiplication');
assert.doesNotMatch(D.materialFragmentSource, /uSignedTextureMask|decodeTexture/,
  'RGBA8_SNORM samples arrive in the native signed range before filtering');
assert.match(D.materialFragmentSource, /mapped = bumpVector\(t1\)/);
assert.match(D.materialFragmentSource,
  /vec3 bumpVector\(vec4 sampleValue\) \{[\s\S]*?return sampleValue\.xyz;\s*\}/);
assert.doesNotMatch(D.materialFragmentSource,
  /sampleValue\.xyz\s*\*\s*2\.0\s*-\s*1\.0/);
// Native M20 normalizes two maps and the explicit renormalize flag; M11 does
// neither. Only a non-legacy embedder material reaches the generic fallback.
assert.match(D.materialFragmentSource,
  /bool renormalizeBump = uMaterial20 != 0\s*\? bumpCount > 1 \|\| \(uM20Flags & 32\) != 0\s*: uMaterial11 == 0;/);
assert.match(D.materialFragmentSource,
  /specular \*= \(uTextureMask & 2\) != 0 \? t1\.a : t2\.a/);

// material20_envi.vsh:45-68 always preserves raw N/T, forms B before mModel,
// and has no animated-basis conditioning branch. The port shares the three
// legacy light varyings with this mutually exclusive environment phase.
assert.match(material20EnvironmentVertexReference,
  /vmov\s+N,vNormal[\s\S]*?mov\s+T,vTangent[\s\S]*?mul\s+B\.xyz,N\.yzx,T\.zxy[\s\S]*?mad\s+B\.xyz,-N\.zxy,T\.yzx,B/);
assert.match(material20EnvironmentVertexReference,
  /m3x3\s+oT0\.xyz,N,mModel[\s\S]*?m3x3\s+oT3\.xyz,B,mModel[\s\S]*?m3x3\s+oT4\.xyz,T,mModel/);
assert.doesNotMatch(material20EnvironmentVertexReference, /LgtType\[8\]/);
assert.match(D.materialVertexSource,
  /if \(uLegacyLightingMode == 3\) \{[\s\S]*?vLegacyLight = worldNormal;[\s\S]*?vLegacyHalfway = mat3\(model\) \* cross\(aNormal, aTangent\.xyz\);[\s\S]*?vLegacyAttenuation = worldTangent;/);
assert.match(D.materialFragmentSource,
  /bool legacyVertexLighting =\s+uLegacyLightingMode == 1 \|\| uLegacyLightingMode == 2;/,
  'the environment basis mode cannot accidentally enter the light path');
assert.match(D.materialFragmentSource,
  /environmentNormal = vLegacyLight \* mapped\.x \+\s+vLegacyHalfway \* mapped\.y \+ vLegacyAttenuation \* mapped\.z;/);
assert.match(D.materialFragmentSource,
  /if \(legacyEnvironmentBasis\) environmentNormal = normalize\(environmentNormal\);/);
assert.match(D.materialFragmentSource,
  /legacyEnvironmentBasis && \(uM20RuntimeEnvironmentFlags & 65536\) == 0[\s\S]*?environmentNormal = mat3\(uM11View\) \* environmentNormal/,
  'the non-reflection M20 environment variant maps its raw basis in eye space');
assert.match(D.materialFragmentSource,
  /mat3\(uM11ModelToWorld\) \* \(uMaterialCameraPosition -\s+\(uM11WorldToModel \* vec4\(vWorld, 1\.0\)\)\.xyz\)/,
  'M20 environment reflection preserves the source TransR model-space eye path');

// material11.vsh variant 1 conditions skinned N/T before environment UV
// generation. This is independent of whether the phase happens to be a light.
assert.match(material11VertexReference,
  /CodegenFlags\[18\.\.19\] == 1[\s\S]*?rsq\s+N\.w,N\.w[\s\S]*?mul\s+N\.xyz,vNormal,N\.w[\s\S]*?mad\s+T\.xyz,N,-N\.w,vTangent/);
assert.match(D.materialVertexSource,
  /if \(uConditionLegacyBasis != 0\) \{[\s\S]*?conditionedInputNormal = normalize\(conditionedInputNormal\);[\s\S]*?conditionedInputTangent = normalize/);
assert.match(D.materialVertexSource,
  /material11UV\(2, world, material11WorldNormal, modelPosition,\s+material11ModelNormal\)/);

// CPU oracle: static/instance shaders keep authored lengths and skew, while
// the skinned variant alone normalizes N and Gram-Schmidt-orthogonalizes T.
const rawLegacyLighting = D.legacyVertexLighting(
  [1, 2, 3], [0, 2, 0], [2, 1, 0], [5, 2, 3, 1], [1, 2, 1, 0.5], [1, 2, 7], false);
assert.deepEqual(Array.from(rawLegacyLighting.normal), [0, 2, 0]);
assert.deepEqual(Array.from(rawLegacyLighting.tangent), [2, 1, 0]);
assert.deepEqual(Array.from(rawLegacyLighting.bitangent), [0, 0, -4]);
assert.deepEqual(Array.from(rawLegacyLighting.light), [0, 0, 8]);
assert.deepEqual(Array.from(rawLegacyLighting.halfway), [0, -4, 2]);
assert.deepEqual(Array.from(rawLegacyLighting.attenuation), [0, 0, 1]);
const skinnedLegacyLighting = D.legacyVertexLighting(
  [1, 2, 3], [0, 2, 0], [2, 1, 0], [5, 2, 3, 1], [1, 2, 1, 0.5], [1, 2, 7], true);
assert.deepEqual(Array.from(skinnedLegacyLighting.normal), [0, 1, 0]);
assert.deepEqual(Array.from(skinnedLegacyLighting.tangent), [1, 0, 0]);
assert.deepEqual(Array.from(skinnedLegacyLighting.bitangent), [0, 0, -1]);
assert.deepEqual(Array.from(skinnedLegacyLighting.light), [0, 0, 4]);
assert.deepEqual(Array.from(skinnedLegacyLighting.halfway), [0, -1, 1]);
D.Renderer.prototype.bindMaterial.call(materialBindingRenderer, signedBindView, {
  ambientLight: 0xff000000,
  camera: { cameraSpace: materialCameraSpace, fogColor: 0xff000000, fogStart: 0, fogEnd: 100 },
}, { view: identity }, [{
  kind: 'point', position: [11, 22, 33], range: 10, amplify: 2, color: 0x80ffffff,
}], legacyModel, true);
assert.equal(materialUniformCalls.oneI.get('uLegacyLightingMode'), 2,
  'skinned legacy light passes select the source conditioning variant');
assert.equal(materialUniformCalls.oneI.get('uConditionLegacyBasis'), 1,
  'animated Material11 phases condition their basis independently of light mode');
D.Renderer.prototype.bindMaterial.call(materialBindingRenderer, signedBindView, {
  ambientLight: 0xff000000,
  camera: { cameraSpace: materialCameraSpace, fogColor: 0xff000000, fogStart: 0, fogEnd: 100 },
}, { view: identity }, [], legacyModel, true, true);
assert.equal(materialUniformCalls.oneI.get('uLegacyLightingMode'), 1,
  'shader instancing overrides the animated light variant with raw transformed N/T');
assert.equal(materialUniformCalls.oneI.get('uConditionLegacyBasis'), 0,
  'shader instancing has native variant-2 precedence over BoneData variant 1');
assert.match(D.Renderer.prototype.drawMeshItem.toString(),
  /meshHasAnimation\(job\.mesh\)/,
  'draw submission derives the native N/T shader variant from the source mesh');
assert.match(D.Renderer.prototype.drawMeshItem.toString(),
  /pass\.program === 'instances'/,
  'draw submission preserves native shader-instance variant precedence');
D.Renderer.prototype.bindMaterial.call(materialBindingRenderer, m11BaseZeroView, {
  ambientLight: 0xff000000,
  camera: { cameraSpace: identity, fogColor: 0xff000000, fogStart: 0, fogEnd: 100 },
}, { view: identity }, [], legacyModel);
assert.equal(materialUniformCalls.oneI.get('uLegacyLightingMode'), 0,
  'unlit legacy phases do not run the vertex lighting path');
assert.equal(materialUniformCalls.oneI.get('uConditionLegacyBasis'), 0);
assert.equal(materialUniformCalls.four.get('uBaseColor')[3], 0,
  'generated Material11 base writes ZERO alpha without alpha test');
D.Renderer.prototype.bindMaterial.call(materialBindingRenderer, m11BaseZeroView, {
  ambientLight: 0xff000000,
  camera: { cameraSpace: identity, fogColor: 0xff000000, fogStart: 0, fogEnd: 100 },
}, { view: identity }, [], legacyModel, true);
assert.equal(materialUniformCalls.oneI.get('uLegacyLightingMode'), 0,
  'animated unlit Material11 phases remain outside the light path');
assert.equal(materialUniformCalls.oneI.get('uConditionLegacyBasis'), 1,
  'animated unlit Material11 phases still select source basis conditioning');
D.Renderer.prototype.bindMaterial.call(materialBindingRenderer, m11BaseTextureAlphaView, {
  ambientLight: 0xff000000,
  camera: { cameraSpace: identity, fogColor: 0xff000000, fogStart: 0, fogEnd: 100 },
}, { view: identity }, [], legacyModel);
assert.equal(materialUniformCalls.four.get('uBaseColor')[3], 1,
  'generated Material11 alpha-test base writes TEX0 alpha independently of ambient alpha');
view = D.materialView(material11, { usage: 'shadow', state: 'shadow-volume' });
assert.equal(view.baseFlags, 0x040220);
assert.equal(D.materialState(view).depthFunc, 'less');
assert.equal(D.materialState(view).colorWrite, false);

// material11.vsh gives every logical Tex0..Tex3 slot its own coordinate path:
// UV channel selection, SpecialFlags space generation and TScale/SRT mode.
const uvMaterial11 = {
  kind: 'material', system: '1.1', parameters: new Array(64).fill(0), textures: [],
};
const uvp = uvMaterial11.parameters;
uvp[1] = 0x0210; // Tex2 sphere mapping, Tex3 world projection.
uvp[4] = 0x1000; // Tex0 scale (an authored zero must remain zero).
uvp[5] = 0x0010; // Tex1 UV1, direct.
uvp[6] = 0x1000; // Tex2 scale.
uvp[7] = 0x2000; // Tex3 SRT1.
uvp[24] = 0; uvp[26] = 3;
uvp.splice(34, 9, 2, 3, 1, 0, 0, 0, 0.25, -0.5, 0);
uvp.splice(43, 5, 2, 4, Math.PI / 2, 0.25, 0.5);
const uvView11 = D.materialView(uvMaterial11, { usage: 'other' });
assert.deepEqual(Array.from(uvView11.slotTextureFlags), [0x1000, 0x10, 0x1000, 0x2000]);
assert.deepEqual(Array.from(uvView11.slotTextureScales), [0, 0, 3, 0]);
const uvInputs11 = {
  uv0: [0.25, 0.5], uv1: [0.75, 0.875], normal: [0.2, -0.4, 0.8],
  eyeNormal: [0.2, -0.4, 0.8], position: [4, 5, 6, 1],
  worldPosition: [1, 2, 3, 1], eyePosition: [7, 8, 9, 1],
};
const roundedUV = value => Array.from(value, component => Number(component.toFixed(6)));
assert.deepEqual(roundedUV(D.material11UV(uvView11, 0, uvInputs11)), [0, 0]);
assert.deepEqual(roundedUV(D.material11UV(uvView11, 1, uvInputs11)), [0.75, 0.875]);
assert.deepEqual(roundedUV(D.material11UV(uvView11, 2, uvInputs11)), [1.8, 2.1]);
assert.deepEqual(roundedUV(D.material11UV(uvView11, 3, uvInputs11)), [2.25, 5.5]);

uvp[1] = 0x20; uvp[6] = 0; // Native reflection uses 2*dot(E,N)/dot(N,N).
let reflectionView11 = D.materialView(uvMaterial11, { usage: 'other' });
assert.deepEqual(roundedUV(D.material11UV(reflectionView11, 2, {
  eyePosition: [0, 1, 0, 1], eyeNormal: [0, 1, 0], normal: [0, 1, 0],
})), [0.5, 0]);

uvp[1] = 0; uvp[4] = 0x3000; // SRT2 angle is radians, unlike SRT1's turns.
const srt2View11 = D.materialView(uvMaterial11, { usage: 'other' });
assert.deepEqual(roundedUV(D.material11UV(srt2View11, 0, { uv0: [0.25, 0.5] })), [2.25, 0]);

uvp[48] = 0x0110; uvp[58] = 0x1012; uvp[62] = 7;
const environmentPass11 = D.materialView(uvMaterial11,
  { usage: 'postlight2', state: 'material11-postlight2' });
assert.equal(environmentPass11.specialFlags, 0x10);
assert.equal(environmentPass11.slotTextureFlags[2], 0x1012);
assert.equal(environmentPass11.slotTextureScales[2], 7);
assert.match(D.materialVertexSource, /layout\(location=9\) in vec2 aUV1/);
assert.match(D.materialVertexSource, /vM11UV2 = uMaterial11 != 0\s+\? material11UV\(2/);
assert.match(D.materialVertexSource, /source\.xy = eyeNormal\.xy \* vec2\(0\.5, -0\.5\) \+ 0\.5/);
assert.match(D.materialFragmentSource, /uMaterial11 != 0 \? vM11UV3 : vM20UV3/);
assert.match(D.materialFragmentSource,
  /if \(\(uTextureMask & 1\) != 0\) environment \*= t0\.a;\s+result = vec4\(environment, 1\.0\);/,
  'Material11 environment masking and output alpha match the generated combiner');

const material20 = {
  kind: 'material', system: '2.0', parameters: new Array(38).fill(0),
  textures: [null, null, null, { kind: 'bitmap' }, null, null, { kind: 'bitmap' }],
};
material20.parameters[0] = 0x14;
material20.parameters[1] = 0xffaabbcc;
material20.parameters[9] = 0x00010000; // detail texture 1: mul2
view = D.materialView(material20, { usage: 'base', state: 'material20-zfill' });
assert.equal(view.baseFlags, 0x0320);
assert.equal(view.alphaCutoff, 128 / 255);
assert.equal(view.vertexColorMode, 0);
assert.deepEqual(Array.from(view.textureMap), [3, null, null, null]);
view = D.materialView(material20, { usage: 'prelight', state: 'material20-texture' });
const material20PrelightView = view;
assert.equal(D.renderMode(view), 6);
assert.deepEqual(Array.from(view.textureMap), [0, 1, 2, null]);
assert.deepEqual(Array.from(view.detailOps), [1, 0]);
assert.deepEqual(Array.from(D.evaluateMaterial20Prelight(view, [
  [0.4, 0.5, 0.75, 1], [0.5, 0.25, 1, 1], null,
])).map(value => Number(value.toFixed(4))), [0.4, 0.25, 1.5, 1]);
assert.deepEqual(Array.from(D.evaluateMaterial20Prelight(view)), [1, 1, 1, 0],
  'material20_tex uses native c8=(1,1,1,0) when the main texture is absent');
assert.match(D.materialFragmentSource,
  /\(uTextureMask & 1\) != 0 \? t0 : vec4\(1\.0, 1\.0, 1\.0, 0\.0\)/);

// M20 UpdatePara replaces runtime constants but never recompiles setup state,
// sampler flags or shader branches. Exercise both halves of that ownership.
const compiledM20Parameters = new Array(42).fill(0);
compiledM20Parameters[0] = 0x85; // specular + smooth-light blend + spec map.
compiledM20Parameters[1] = 0xff102030;
compiledM20Parameters[2] = 0xff405060;
compiledM20Parameters[3] = 12;
compiledM20Parameters[8] = 0x0101;
compiledM20Parameters[9] = 0x00030002; // detail ADD, trilinear sampler.
compiledM20Parameters[16] = 0x1203;
compiledM20Parameters[18] = 0x30002;
const animatedM20Parameters = compiledM20Parameters.slice();
animatedM20Parameters[0] = 0x40; // runtime-only 2x remains observable.
animatedM20Parameters[1] = 0xffabcdef;
animatedM20Parameters[2] = 0xff112233;
animatedM20Parameters[3] = 48;
animatedM20Parameters[8] = 0x3000;
animatedM20Parameters[9] = 0x00010000;
animatedM20Parameters[12] = 7;
animatedM20Parameters[16] = 0x2301;
animatedM20Parameters[18] = 0;
animatedM20Parameters[20] = 9;
animatedM20Parameters[24] = 2;
const compiledM20 = {
  kind: 'material', system: '2.0', initialParameters: compiledM20Parameters,
  parameters: animatedM20Parameters, textures: new Array(7).fill(null),
};
const compiledM20Prelight = D.materialView(compiledM20,
  { usage: 'prelight', state: 'material20-texture' });
assert.equal(compiledM20Prelight.flags, 0x85);
assert.equal(compiledM20Prelight.runtimeFlags, 0x40);
assert.equal(compiledM20Prelight.textureFlags[0], 0x0101);
assert.deepEqual(Array.from(compiledM20Prelight.detailOps), [3, 0]);
assert.equal(compiledM20Prelight.textureScales[0], 7);
assert.equal(compiledM20Prelight.uvTransform1[0], 2);
const compiledM20Light = D.materialView(compiledM20,
  { usage: 'light', state: 'material20-light' });
assert.equal(compiledM20Light.baseFlags, 0x305280,
  'M20 blend setup remains compiled from initial Flags bit 2');
assert.equal(compiledM20Light.color, 0xffabcdef);
assert.equal(compiledM20Light.specularColor, 0xff112233);
assert.equal(compiledM20Light.specularPower, 48);
assert.equal(compiledM20Light.specularStrength, 1);
assert.equal(compiledM20Light.lightScale, 2);
assert.equal(compiledM20Light.lightFlags[0], 0x1203);
assert.equal(compiledM20Light.lightScales[0], 9);
assert.equal(compiledM20Light.environmentFlags, 0x30002);
assert.equal(compiledM20Light.runtimeEnvironmentFlags, 0);
assert.match(D.materialFragmentSource,
  /uM20RuntimeEnvironmentFlags & 65536/,
  'M20 environment matrix choice remains a runtime Set value');
const material20LightView = D.materialView(material20,
  { usage: 'light', state: 'material20-light' });
D.Renderer.prototype.bindMaterial.call(materialBindingRenderer, material20LightView, {
  ambientLight: 0xff000000,
  camera: { cameraSpace: materialCameraSpace, fogColor: 0xff000000, fogStart: 0, fogEnd: 100 },
}, { view: identity }, [{
  kind: 'directional', direction: [1, 2, 3], position: [14, 26, 38],
  range: 4096, amplify: 1, color: 0xffffffff,
}], legacyModel);
assert.deepEqual(materialUniformCalls.four.get('uLightPosition[0]').slice(0, 4), [2, 6, 12, 0]);
assert.deepEqual(materialUniformCalls.four.get('uLightAttenuation[0]').slice(0, 3), [8, 18, 32],
  'Material20 directional attenuation retains the independently transformed light position');
assert.ok(Math.abs(materialUniformCalls.four.get('uLightAttenuation[0]')[3] - 1 / 4096) < 1e-12);
const material20AmbientView = D.materialView(material20,
  { usage: 'ambient', state: 'material20-ambient' });
resetMaterialLightUploads();
D.Renderer.prototype.bindMaterial.call(materialBindingRenderer, material20AmbientView, {
  ambientLight: 0x00102030,
  camera: { cameraSpace: materialCameraSpace, fogColor: 0xff000000,
    fogStart: 0, fogEnd: 100 },
}, { view: identity }, [{
  kind: 'point', position: [11, 22, 33], range: 10, amplify: 1,
  color: 0xffffffff,
}], legacyModel);
assert.equal(materialUniformCalls.oneI.get('uMode'), 5);
assert.equal(materialUniformCalls.oneI.get('uLightCount'), 1,
  'non-light passes preserve the selected-light count uniform');
assert.equal(materialUniformCalls.lightArrayCalls, 0,
  'non-light passes do not upload any light arrays');
assert.equal(materialUniformCalls.lightArrayFloats, 0);
assert.deepEqual(materialUniformCalls.three.get('uAmbient'),
  Array.from(D.colorRGB(0x00102030)),
  'Material20 AMBIENT consumes the viewport-owned Scene_Ambient snapshot');
assert.equal(materialUniformCalls.oneI.get('uTextureMask'), 0,
  'Material20 AMBIENT is the texture-free VColor pass');
D.Renderer.prototype.bindMaterial.call(materialBindingRenderer, material20PrelightView, {
  ambientLight: 0x00ffffff,
  camera: { cameraSpace: materialCameraSpace, fogColor: 0xff000000,
    fogStart: 0, fogEnd: 100 },
}, { view: identity }, [], legacyModel);
assert.equal(materialUniformCalls.oneI.get('uMode'), 6,
  'Material20 PRELIGHT retains UseAmbient=false even with nonzero Scene_Ambient');

const ambientM11Parameters = new Array(64).fill(0);
ambientM11Parameters[0] = 0x0200;
ambientM11Parameters[9] = 0x10; // Color0 SET.
ambientM11Parameters[28] = 0xff204060;
ambientM11Parameters[48] = 3 << 20; // authored ENGU_AMBIENT phase.
const ambientM11View = D.materialView({
  kind: 'material', system: '1.1', parameters: ambientM11Parameters,
  initialParameters: ambientM11Parameters.slice(), textures: [],
}, { usage: 'ambient', state: 'material11-single' });
assert.equal(D.renderMode(ambientM11View), 7,
  'Material11 AMBIENT phase remains its authored combiner, not M20 VColor');
D.Renderer.prototype.bindMaterial.call(materialBindingRenderer, ambientM11View, {
  ambientLight: 0x00ffffff,
  camera: { cameraSpace: materialCameraSpace, fogColor: 0xff000000,
    fogStart: 0, fogEnd: 100 },
}, { view: identity }, [], legacyModel);
assert.equal(materialUniformCalls.oneI.get('uMode'), 7);
assert.deepEqual(materialUniformCalls.four.get('uBaseColor'),
  Array.from(D.colorARGB(0xff204060)),
  'Material11 does not substitute the frame Scene_Ambient for authored Color0');
assert.match(D.materialFragmentSource,
  /result = vec4\(albedo\.rgb \* uAmbient, 0\.0\);/,
  'Material20 ambient preserves the native zero alpha constant');
assert.match(D.materialFragmentSource,
  /float eyeZ = \(uM11View \* vec4\(vWorld, 1\.0\)\)\.z;[\s\S]*?eyeZ - uFogRange\.x/,
  'Material11 linear fog is based on camera-space Z rather than radial distance');
// Released M20 light shaders use the basis N,B=cross(N,T),T: normal-map X is
// the geometric normal, Y the bitangent and Z the tangent.
assert.deepEqual(Array.from(D.tangentBasisNormal([0, 0, 1], [1, 0, 0], [1, 0, 0])), [0, 0, 1]);
assert.deepEqual(Array.from(D.tangentBasisNormal([0, 0, 1], [1, 0, 0], [0, 1, 0])), [0, 1, 0]);
assert.deepEqual(Array.from(D.tangentBasisNormal([0, 0, 1], [1, 0, 0], [0, 0, 1])), [1, 0, 0]);
// _start.cpp MakeAttenuationVolume encodes max(1-dot(v,v),0). Material11
// samples that same volume as Material20; the former (1-r)^2 approximation
// was only 0.25 at half range instead of the native 0.75.
assert.equal(D.materialLightAttenuation(0), 1);
assert.equal(D.materialLightAttenuation(0.5), 0.75);
assert.equal(D.materialLightAttenuation(1024 / 4096), 0.9375,
  'finite-range directional lights use the same native radial curve');
assert.equal(D.materialLightAttenuation(1), 0);
assert.equal(D.materialLightAttenuation(2), 0);
assert.equal(D.materialLightAttenuation(Number.POSITIVE_INFINITY), 0);

const material20UVParameters = new Array(38).fill(0);
material20UVParameters.splice(24, 9, 2, 3, 1, 0, 0, 0, 0.25, 0.5, 0);
material20UVParameters.splice(33, 5, 2, 3, Math.PI / 2, 0.25, 0.5);
material20UVParameters[8] = 0x1000; material20UVParameters[12] = 4;
material20UVParameters[9] = 0x2000;
material20UVParameters[16] = 0x3000;
assert.deepEqual(Array.from(D.material20UV(material20UVParameters, 0, [0.25, 0.5])), [1, 2]);
assert.deepEqual(Array.from(D.material20UV(material20UVParameters, 1, [0.25, 0.5])), [0.75, 2]);
assert.deepEqual(Array.from(D.material20UV(material20UVParameters, 4, [0.25, 0.5]),
  value => Number(value.toFixed(6))), [1.75, 0]);
assert.deepEqual(Array.from(D.material20EnvironmentUV([1, 0, 0])), [1, 0.5]);
assert.deepEqual(Array.from(D.material20EnvironmentUV([0, 1, 0])), [0.5, 0]);
const environmentBasisModel = D.mat4Identity();
environmentBasisModel[0] = 3; environmentBasisModel[5] = 5;
environmentBasisModel[10] = 7;
assert.deepEqual(Array.from(D.material20EnvironmentBumpDirection(
  [0, 2, 0], [2, 1, 0], [0.25, 0.5, 0.75], environmentBasisModel)),
  [4.5, 6.25, -14],
  'raw B=cross(N,T) is formed before a non-uniform model transform');
// cross(worldN,worldT) would contribute -30 here instead of source-exact -14.
assert.notDeepEqual(Array.from(D.material20EnvironmentBumpDirection(
  [0, 2, 0], [2, 1, 0], [0.25, 0.5, 0.75], environmentBasisModel)),
  [4.5, 6.25, -30]);
const environmentViewRotation = D.mat4Identity();
environmentViewRotation[0] = 0; environmentViewRotation[1] = 1;
environmentViewRotation[4] = -1; environmentViewRotation[5] = 0;
assert.deepEqual(Array.from(D.material20EnvironmentBumpDirection(
  [0, 2, 0], [2, 1, 0], [0.25, 0.5, 0.75], environmentBasisModel,
  environmentViewRotation, false)), [-6.25, 4.5, -14],
  'non-reflection M20 environment mapping rotates the combined basis to eye space');
assert.deepEqual(Array.from(D.material20EnvironmentEye(
  legacyModel, [1, 2, 3], [14, 26, 38])), [14, 48, 116],
  'authored scale retains the released TransR camera behavior');
const staticM11EnvironmentNormal = D.material11EnvironmentNormals(
  [0, 2, 0], legacyModel, false);
const skinnedM11EnvironmentNormal = D.material11EnvironmentNormals(
  [0, 2, 0], legacyModel, true);
assert.deepEqual(Array.from(staticM11EnvironmentNormal.model), [0, 2, 0]);
assert.deepEqual(Array.from(staticM11EnvironmentNormal.world), [0, 6, 0]);
assert.deepEqual(Array.from(skinnedM11EnvironmentNormal.model), [0, 1, 0]);
assert.deepEqual(Array.from(skinnedM11EnvironmentNormal.world), [0, 3, 0]);
const reflectedEnvironment = D.material20EnvironmentDirection(
  [0, 1, 0], [Math.SQRT1_2, Math.SQRT1_2, 0], true);
assert.deepEqual(Array.from(reflectedEnvironment, value => Number(value.toFixed(6))),
  [0.707107, -0.707107, 0]);

const colorCorrectParameters = new Array(11).fill(0);
colorCorrectParameters[1] = 0.25;
colorCorrectParameters[2] = 0.75;
colorCorrectParameters[3] = 0xff102030;
colorCorrectParameters[4] = 0xff405060;
colorCorrectParameters[5] = 0xff8090a0;
colorCorrectParameters[6] = 0xff010203;
colorCorrectParameters[8] = 0xff708090;
colorCorrectParameters[9] = 0xffa0b0c0;
colorCorrectParameters[10] = 1.5;
let correctionConstants = D.colorCorrectionConstants(colorCorrectParameters);
assert.deepEqual(Array.from(correctionConstants.subarray(16, 20)), [-0.5, 2, 0, 0]);
assert.deepEqual(Array.from(correctionConstants.subarray(0, 3), value => Number(value.toFixed(6))),
  [16 / 255, 32 / 255, 48 / 255].map(value => Number(value.toFixed(6))));
const expectedBrightness = new Float32Array([128 / 255, 144 / 255, 160 / 255]);
for (let channel = 0; channel < 3; channel++) expectedBrightness[channel] *= 1.5;
assert.deepEqual(Array.from(correctionConstants.subarray(8, 11), value => Number(value.toFixed(6))),
  Array.from(expectedBrightness, value => Number(value.toFixed(6))));

// With direct-color vectors at one and grayscale vectors/gamma at zero, the
// native seven-vector shader is the identity for every threshold position.
const identityCorrection = new Array(11).fill(0);
identityCorrection[1] = 0; identityCorrection[2] = 1;
identityCorrection[3] = identityCorrection[4] = identityCorrection[5] = 0xffffffff;
identityCorrection[6] = identityCorrection[8] = identityCorrection[9] = 0xff000000;
identityCorrection[10] = 1;
const correctionSource = new Float32Array([0.2, 0.4, 0.8, 0.6]);
assert.deepEqual(Array.from(D.evaluateColorCorrection(identityCorrection, correctionSource),
  value => Number(value.toFixed(6))), Array.from(correctionSource, value => Number(value.toFixed(6))));
const amplifiedCorrection = identityCorrection.slice();
amplifiedCorrection[10] = 4;
assert.deepEqual(Array.from(D.evaluateColorCorrection(amplifiedCorrection,
  [0.5, 0.25, 0.125, 0.75])), [1, 1, 0.5, 0.75],
  'the CPU oracle includes the RGBA8 target saturation after brightness amplification');

// Fake mode ignores all authored colors and displays only the threshold range.
// effect_colorcorrect.psh uses an equal-weight 1/3 gray, not luma weights.
const fakeCorrection = new Array(11).fill(0);
fakeCorrection[1] = 0; fakeCorrection[2] = 1; fakeCorrection[7] = 1;
assert.deepEqual(Array.from(D.evaluateColorCorrection(fakeCorrection, [1, 0, 0, 0.25]),
  value => Number(value.toFixed(6))), [1 / 3, 1 / 3, 1 / 3, 0.25].map(value => Number(value.toFixed(6))));
fakeCorrection[1] = 0.25; fakeCorrection[2] = 0.75;
assert.deepEqual(Array.from(D.evaluateColorCorrection(fakeCorrection, [0.5, 0.5, 0.5, 1])),
  [0.5, 0.5, 0.5, 1]);

const subtractCorrection = identityCorrection.slice();
subtractCorrection[6] = 0xff408000;
assert.deepEqual(Array.from(D.evaluateColorCorrection(subtractCorrection, [0.5, 0.5, 0.5, 0.75]),
  value => Number(value.toFixed(6))),
  [0.5 - 64 / 255, 0, 0.5, 0.75].map(value => Number(Math.max(0, value).toFixed(6))));

const correctionDraws = [];
D.Renderer.prototype.applyPost.call({
  configureFullscreenState(material) { assert.equal(material, null); },
  drawFullscreen(image, mode, options) { correctionDraws.push({ image, mode, options }); },
}, 'captured-frame', { kind: 'color-correction', parameters: colorCorrectParameters },
{ uvRect: [0.1, 0.2, 0.9, 0.8] });
assert.equal(correctionDraws[0].image, 'captured-frame');
assert.equal(correctionDraws[0].mode, 2);
assert.deepEqual(Array.from(correctionDraws[0].options.colorCorrect), Array.from(correctionConstants));
assert.deepEqual(Array.from(correctionDraws[0].options.uvRect), [0.1, 0.2, 0.9, 0.8]);

// Glare follows geneffectipp.cpp's fixed target selection and all three
// released shaders, rather than the former single full-resolution blur.
const glareParameters = [
  31, 0, 0xff808080, 0xffffe0a0, 0xffffffff,
  0.008, 0.004, 1.25, 1, 0, 1,
];
let plan = D.glarePlan(glareParameters);
assert.deepEqual(Array.from(plan.downsample), [true, false]);
assert.equal(plan.copyDownsample, false);
assert.deepEqual(JSON.parse(JSON.stringify(plan.stages)),
  [{ blur: 0.008, amplify: 1.25 }, { blur: 0.004, amplify: 1 }]);
assert.deepEqual(Array.from(plan.glareColor, value => Number(value.toFixed(6))),
  [1, 224 / 255, 160 / 255, 1].map(value => Number(value.toFixed(6))));
assert.match(D.fullscreenFragmentSource,
  /source\.a \+= weights\[0\] \* weights\[0\];/,
  'the 3x3 glare filter retains effect_glare.psh first-MAD alpha bias');
assert.match(D.fullscreenFragmentSource,
  /source\.a \+= weights\[0\];\s+source = clamp\(source \* uParameters\.z/,
  'the 9-tap glare filter scales the packed first-MAD alpha bias with amplification');
for (const [flags, expected] of [[1, [false, true]], [2, [true, true]], [3, [true, false]]]) {
  const flagged = glareParameters.slice(); flagged[1] = flags;
  assert.deepEqual(Array.from(D.glarePlan(flagged).downsample), expected);
}
const copiedGlare = glareParameters.slice(); copiedGlare[1] = 4;
assert.equal(D.glarePlan(copiedGlare).copyDownsample, true);

const grayGlare = glareParameters.slice();
grayGlare[2] = 0xff000000; grayGlare[9] = 0;
assert.deepEqual(Array.from(D.evaluateGlareTone(grayGlare, [1, 0.5, 0, 0.75]),
  value => Number(value.toFixed(6))), [0.5, 0.5, 0.5, 0.5]);
grayGlare[9] = 1;
assert.deepEqual(Array.from(D.evaluateGlareTone(grayGlare, [1, 0.5, 0, 0.75])),
  [1, 0.5, 0, 0.75]);
const composite = D.evaluateGlareComposite(glareParameters,
  [0.4, 0.4, 0.4, 1], [0.5, 0.25, 0.75, 1]);
const glareTint = [1, 224 / 255, 160 / 255, 1];
assert.deepEqual(Array.from(composite, value => Number(value.toFixed(6))),
  glareTint.map((tint, channel) => {
    const original = [0.5, 0.25, 0.75, 1][channel];
    const blurred = 0.4 * tint;
    return Number(Math.min(1, blurred + original - blurred * original).toFixed(6));
  }));

const glareDraws = [], glareBinds = [];
const glareTargets = [0, 1, 2].map(index => ({
  color: `glare-${index}`, width: 1, height: 1,
  resize(width, height) { this.width = width; this.height = height; },
}));
const glareRenderer = {
  width: 640, height: 360, glareTargets,
  configureFullscreenState(material) { assert.equal(material, null); },
  glareTarget(index, width, height) {
    const target = glareTargets[index]; target.resize(width, height); return target;
  },
  bindDestination(target) { glareBinds.push(target?.color || target || 'screen'); },
  setViewportRegion(viewport) { assert.equal(viewport.id, 'post-viewport'); },
  drawFullscreen(image, mode, options) { glareDraws.push({ image, mode, options }); },
  applyGlare: D.Renderer.prototype.applyGlare,
};
D.Renderer.prototype.applyPost.call(glareRenderer, 'original-frame',
  { kind: 'glare', parameters: glareParameters }, {
    destination: { color: 'final-frame' }, viewport: { id: 'post-viewport' },
    uvRect: [0.1, 0.2, 0.9, 0.8],
  });
assert.deepEqual(glareDraws.map(draw => draw.mode), [1, 4, 5, 5, 5, 5, 6]);
assert.deepEqual(glareBinds, ['glare-0', 'glare-2', 'glare-0', 'glare-2', 'glare-0', 'glare-2', 'final-frame']);
assert.deepEqual(Array.from(glareDraws[0].options.texel), [1 / 640, 1 / 360]);
assert.deepEqual(glareDraws[0].options.uvRect, [0.1, 0.2, 0.9, 0.8]);
assert.deepEqual(Array.from(glareDraws[2].options.parameters), [0.008, 0, 1, 0]);
assert.deepEqual(Array.from(glareDraws[3].options.parameters), [0, 0.016, 1.25, 0]);
assert.deepEqual(Array.from(glareDraws[5].options.parameters), [0, 0.008, 1, 0]);
assert.equal(glareDraws[6].options.secondImage, 'original-frame');
assert.deepEqual(Array.from(glareDraws[6].options.color0), Array.from(plan.glareColor));
assert.deepEqual(Array.from(glareDraws[6].options.color1), Array.from(plan.originalColor));
assert.deepEqual(Array.from(glareDraws[6].options.parameters), [1, 0, 0, 0]);

view = D.materialView(material20, { usage: 'ambient', state: 'material20-ambient' });
assert.equal(view.baseFlags, 0x3280);
assert.deepEqual(Array.from(D.materialState(view).blend), ['zero', 'src-color', 'add']);
view = D.materialView(material20, { usage: 'light', state: 'material20-light' });
assert.equal(view.baseFlags, 0x305280);
assert.equal(view.color, 0xffaabbcc);
assert.deepEqual(Array.from(view.textureMap), [null, 4, 5, null]);
assert.equal(view.usePrelight, true);
view = D.materialView(material20, { usage: 'postlight', state: 'material20-environment' });
assert.equal(D.renderMode(view), 4);
assert.deepEqual(Array.from(view.textureMap), [null, 4, 6, null]);
material20.parameters[18] = 0x30002;
material20.textures[4] = { kind: 'bitmap', format: 5 };
const material20EnvironmentView = D.materialView(material20,
  { usage: 'postlight', state: 'material20-environment' });
D.Renderer.prototype.bindMaterial.call(materialBindingRenderer,
  material20EnvironmentView, {
    ambientLight: 0xff000000,
    camera: { cameraSpace: materialCameraSpace, fogColor: 0xff000000,
      fogStart: 0, fogEnd: 100 },
  }, { view: identity }, [], legacyModel, true);
assert.equal(materialUniformCalls.oneI.get('uLegacyLightingMode'), 3,
  'M20 environment selects the raw N/B/T varying mode even for a skinned mesh');
assert.equal(materialUniformCalls.oneI.get('uConditionLegacyBasis'), 0,
  'material20_envi.vsh has no animated basis-conditioning variant');
assert.deepEqual(materialUniformCalls.three.get('uMaterialCameraPosition'), [8, 18, 32]);

assert.deepEqual(JSON.parse(JSON.stringify(D.materialState({ baseFlags: 0x0100 }))), {
  depthTest: true, depthWrite: true, depthFunc: 'always', colorWrite: true,
  cull: 'back', blend: null, stencilTest: false,
});
assert.equal(D.groupRenderPass({ materials: [{ pass: 9 }] }, { material: 0, pass: 4 }), 4);
assert.equal(D.groupRenderPass({ materials: [] }, { material: null, renderPass: 7 }), 7);

material11.parameters[48] = 0;
material11.parameters[19] = 0x40;
assert.equal(D.materialView(material11, { usage: 'other' }).vertexColorMode, 1);
material11.parameters[19] = 0x10;
assert.equal(D.materialView(material11, { usage: 'other' }).vertexColorMode, 3);

const items = [
  { renderPass: 1, pass: { usage: 'base' }, job: { opId: 1 } },
  { renderPass: 0, pass: { usage: 'light' }, job: { opId: 2 } },
  { renderPass: 0, pass: { usage: 'base' }, job: { opId: 3 } },
];
D.sortRenderItems(items);
assert.deepEqual(items.map(item => item.job.opId), [3, 2, 1]);
const equalKeyItems = [
  { renderPass: 7, pass: { usage: 'other' }, job: { opId: 11 }, sortOrder: 0 },
  { renderPass: 7, pass: { usage: 'other' }, job: { opId: 10 }, sortOrder: 1 },
];
D.sortRenderItems(equalKeyItems);
assert.deepEqual(equalKeyItems.map(item => item.job.opId), [11, 10],
  'equal-key paint jobs retain their native BuildPaintJobs order');

// AddPaintJob reverses the outer linked job list only. BuildPaintJobs still
// walks each EngMesh's materials and passes forwards, and the stable radix
// sort retains that inner order when keys are equal.
const nativeBuildItems = [];
D.forEachNativeHeadInsertedJob([
  { id: 'first' }, { id: 'second' },
], job => {
  for (const inner of [0, 1]) nativeBuildItems.push({
    renderPass: 7, pass: { usage: 'other' }, job,
    sortMaterial: 1, sortOrder: nativeBuildItems.length, inner,
  });
});
D.sortRenderItems(nativeBuildItems);
assert.deepEqual(nativeBuildItems.map(item => `${item.job.id}:${item.inner}`), [
  'second:0', 'second:1', 'first:0', 'first:1',
], 'head insertion reverses jobs without reversing their material/pass loops');

const identityPassA = {}, identityPassB = {};
const identityA = D.materialPassSortIdentity({}, identityPassA);
assert.equal(D.materialPassSortIdentity({}, identityPassA), identityA,
  'one material-pass object keeps a stable native-pointer analogue');
const identityB = D.materialPassSortIdentity({}, identityPassB);
assert.notEqual(identityA, identityB, 'distinct material-pass records receive distinct sort identities');
const materialKeyItems = [
  { renderPass: 7, pass: { usage: 'other' }, job: { opId: 20 },
    sortMaterial: identityB, sortOrder: -2 },
  { renderPass: 7, pass: { usage: 'other' }, job: { opId: 21 },
    sortMaterial: identityA, sortOrder: -1 },
  { renderPass: 7, pass: { usage: 'other' }, job: { opId: 22 },
    sortMaterial: identityA, sortOrder: -3 },
];
D.sortRenderItems(materialKeyItems);
assert.deepEqual(materialKeyItems.map(item => item.job.opId), [22, 21, 20],
  'material-pass identity groups jobs before reverse equal-key insertion order');

const lightItems = [
  { renderPass: 0, pass: { usage: 'postlight' }, job: { opId: 6 } },
  { renderPass: 0, lightIndex: 1, pass: { usage: 'light' }, job: { opId: 5 } },
  { renderPass: 0, lightIndex: 0, pass: { usage: 'light' }, job: { opId: 3 } },
  { renderPass: 0, pass: { usage: 'base' }, job: { opId: 1 } },
  { renderPass: 0, lightIndex: 1, pass: { usage: 'shadow' }, job: { opId: 4 } },
  { renderPass: 0, lightIndex: 0, pass: { usage: 'shadow' }, job: { opId: 2 } },
];
D.sortRenderItems(lightItems);
assert.deepEqual(lightItems.map(item => item.job.opId), [1, 2, 3, 4, 5, 6]);

const adjustedEffect = D.effectRenderItem({
  op: { id: 99 }, effect: { pass: 31, usage: 'base' }, passAdjust: -1,
}, 4);
assert.equal(adjustedEffect.renderPass, 30);
const samePassMesh = {
  renderPass: 30, pass: { usage: 'base' }, job: { opId: 1 }, sortMaterial: 1,
};
const samePassEffect = D.effectRenderItem({
  op: { id: 2 }, effect: { pass: 30, usage: 'base' }, passAdjust: 0,
});
const unifiedItems = [samePassMesh, samePassEffect];
D.sortRenderItems(unifiedItems);
assert.equal(unifiedItems[0], samePassEffect,
  'native effect material id zero runs before same-pass/same-usage mesh paint');

const selectedLights = D.selectActiveLights({
  camera: { cameraSpace: D.mat4Identity() },
  lightJobs: [
    { opId: 1, position: [10, 0, 0], range: 10, amplify: 1 },
    { opId: 2, position: [2, 0, 0], range: 10, amplify: 1 },
    { opId: 3, position: [1, 0, 0], range: 100, amplify: 0 },
  ],
}, 1);
assert.deepEqual(selectedLights.map(light => light.opId), [2]);

const fallbackLight = {
  opId: 0, kind: 'point', position: [0, 0, 0], range: 32, amplify: 1,
};
const selectedFallback = D.selectActiveLights({
  camera: { cameraSpace: D.mat4Identity() },
  defaultLight: fallbackLight,
  lightJobs: [
    { opId: 7, position: [0, 0, 0], range: 32, amplify: 0 },
    { opId: 8, position: [1e6, 0, 0], range: 1, amplify: 1 },
  ],
});
assert.deepEqual(selectedFallback, [fallbackLight],
  'the native camera light replaces authored lights rejected by AddLightJob');

const selectedAuthored = D.selectActiveLights({
  camera: { cameraSpace: D.mat4Identity() },
  defaultLight: fallbackLight,
  lightJobs: [{ opId: 9, position: [0, 0, 1], range: 32, amplify: 1 }],
});
assert.deepEqual(selectedAuthored.map(light => light.opId), [9],
  'a surviving authored light suppresses the default camera light');

// Loader-time warming touches only immutable cache identities. It uploads one
// static geometry/texture, prepares the reusable light-independent shadow
// topology, yields cooperatively, and becomes a no-op on a second pass.
{
  const warmBitmap = { width: 2, height: 2, data: new Uint16Array(16) };
  const warmMaterial = { kind: 'material', passes: [{ usage: 'shadow' }] };
  const warmMesh = {
    kind: 'mesh',
    hasAnimation: () => false,
    storageSummary: () => ({ vertices: 3, faces: 1 }),
  };
  const animatedMesh = {
    kind: 'minmesh',
    hasAnimation: () => true,
    storageSummary: () => ({ vertices: 3, faces: 1, bones: 1 }),
  };
  const entry = {
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    normals: new Float32Array(9),
    uvs: new Float32Array(6),
    uv1: new Float32Array(6),
    colors: new Uint8Array(12),
    tangents: new Float32Array(12),
    indices: new Uint16Array([0, 1, 2]),
    groups: [{ material: warmMaterial, start: 0, count: 3 }],
    materials: [],
    shadowTopologies: new Map(),
    uploadBytes: 100,
  };
  const geometryEntries = new WeakMap();
  const allEntries = new Set();
  const textureEntries = new WeakMap();
  let textureBytes = 0, yields = 0, flushes = 0, finishes = 0, layerCreates = 0;
  const target = bytes => ({ estimatedBytes: () => bytes });
  const warmRenderer = {
    gl: {
      flush() { flushes++; },
      finish() { finishes++; },
    },
    contextLost: false,
    shadows: true,
    width: 16,
    height: 8,
    geometry: {
      entries: geometryEntries,
      allEntries,
      get(mesh) {
        let result = geometryEntries.get(mesh);
        if (!result) {
          assert.equal(mesh, warmMesh);
          result = entry;
          geometryEntries.set(mesh, result);
          allEntries.add(result);
        }
        return result;
      },
    },
    textures: {
      entries: textureEntries,
      get estimatedBytes() { return textureBytes; },
      estimatedUploadBytes(bitmap) { return textureEntries.has(bitmap) ? 0 : 20; },
      get(bitmap) {
        if (!textureEntries.has(bitmap)) {
          textureEntries.set(bitmap, {});
          textureBytes += 20;
        }
      },
      fallbackTexture() {
        if (!this.fallback) { this.fallback = {}; textureBytes += 4; }
        return this.fallback;
      },
    },
    masterTarget: target(0),
    prelightTarget: target(0),
    targets: [],
    glareTargets: [],
    viewMaterial() { return { textureMap: [0], textures: [warmBitmap] }; },
    warmupResidentBytes: D.Renderer.prototype.warmupResidentBytes,
    ensureLayerGeometry() { if (!this.layerVAO) { this.layerVAO = {}; layerCreates++; } },
    target(index) {
      while (this.targets.length <= index) this.targets.push(target(16));
      return this.targets[index];
    },
    glareTarget(index, width, height) {
      while (this.glareTargets.length <= index) this.glareTargets.push(target(0));
      this.glareTargets[index] = target(width * height * 4);
      return this.glareTargets[index];
    },
  };
  let clock = 0;
  const warmPlan = { tasks: [
    { kind: 'mesh', value: warmMesh, sourceId: 91 },
    { kind: 'material', value: warmMaterial, sourceId: 92 },
    { kind: 'mesh', value: animatedMesh, sourceId: 93 },
  ] };
  const warmOptions = {
    budgetMilliseconds: 8,
    maxResidentBytes: 4 * 1024 * 1024,
    maxResourceBytes: 4 * 1024 * 1024,
    now: () => (clock += 10),
    yield: async () => { yields++; },
  };
  const firstWarm = await D.Renderer.prototype.prewarmResources.call(
    warmRenderer, warmPlan, warmOptions,
  );
  assert.equal(firstWarm.warmedMeshes, 1);
  assert.equal(firstWarm.warmedTextures, 1);
  assert.equal(firstWarm.warmedShadowTopologies, 1);
  assert.equal(firstWarm.skippedAnimatedMeshes, 1);
  assert.equal(firstWarm.plannedTextures, 1);
  assert.ok(firstWarm.newResidentBytes > 0);
  assert.ok(yields > 0);
  assert.equal(finishes, 1);
  assert.equal(layerCreates, 1);

  clock = 0;
  const secondWarm = await D.Renderer.prototype.prewarmResources.call(
    warmRenderer, warmPlan, warmOptions,
  );
  assert.equal(secondWarm.cachedMeshes, 1);
  assert.equal(secondWarm.cachedTextures, 1);
  assert.equal(secondWarm.warmedShadowTopologies, 0);
  assert.equal(secondWarm.newResidentBytes, 0,
    'an identical warm-up does not add renderer residency');
  assert.equal(layerCreates, 1);
  assert.equal(finishes, 2);

  const hugeMesh = {
    kind: 'mesh', hasAnimation: () => false,
    storageSummary: () => ({ vertices: 2_000_000, faces: 1 }),
  };
  const cappedWarm = await D.Renderer.prototype.prewarmResources.call(warmRenderer, {
    tasks: [{ kind: 'mesh', value: hugeMesh }],
  }, { ...warmOptions, maxResourceBytes: 64 * 1024 * 1024 });
  assert.equal(cappedWarm.skippedLargeResources, 1,
    'one oversized identity cannot defeat the per-resource safety bound');
  assert.equal(geometryEntries.has(hugeMesh), false);

  const abortedWarm = await D.Renderer.prototype.prewarmResources.call(
    warmRenderer, warmPlan, { ...warmOptions, signal: { aborted: true } },
  );
  assert.equal(abortedWarm.aborted, true);
  assert.equal(finishes, 3,
    'an aborted warm-up flushes but does not synchronously drain the GPU');
  assert.ok(flushes >= 4);
}

// Scene planning selects one concrete array per viewport. Every mesh/effect
// draw receives that same array so material binding does not repeat camera,
// frustum, filter and sort work for each paint item.
{
  let cameraProjectionBuilds = 0;
  const threadedLight = {
    opId: 80, kind: 'point', position: [0, 0, 4], range: 20,
    amplify: 1, color: 0xffffffff, flags: 0,
  };
  const viewport = {
    flags: 0, clearColor: 0,
    camera: {
      cameraSpace: D.mat4Identity(), nearClip: 1, farClip: 100,
      zoomX: 1, zoomY: 1,
      get centerX() { cameraProjectionBuilds++; return 0; },
      centerY: 0,
    },
    lightJobs: [threadedLight],
    effectJobs: [{ op: { id: 81 }, effect: { pass: 0, usage: 'other' } }],
    meshJobs: [{ opId: 82, mesh: {}, matrix: D.mat4Identity(), passAdjust: 0 }],
  };
  const geometry = {
    groups: [{ material: null, start: 0, count: 3 }], materials: [],
  };
  const received = [];
  const gl = {
    DEPTH_TEST: 1, BLEND: 2, CULL_FACE: 3, BACK: 4,
    COLOR_BUFFER_BIT: 8, DEPTH_BUFFER_BIT: 16, STENCIL_BUFFER_BIT: 32,
    STENCIL_TEST: 64,
    enable() {}, disable() {}, depthMask() {}, colorMask() {}, cullFace() {},
    clearColor() {}, clearDepth() {}, clearStencil() {}, stencilMask() {}, clear() {},
  };
  const renderer = {
    gl, shadows: true, cullingStats: {}, prelightTarget: {},
    geometry: { get: () => geometry },
    setViewportRegion() {}, instanceMatrices: () => new Float32Array(identity),
    setLightScissor() {},
    prepareInstanceUploads() {},
    drawMaterial11Insert() {}, captureFramebuffer() {}, bindDestination() {},
    drawEffectItem(...args) { received.push(['effect', args[7]]); },
    drawMeshItem(_item, _viewport, _matrices, lights) { received.push(['mesh', lights]); },
  };
  D.Renderer.prototype.drawViewportScene.call(renderer, viewport,
    { frame: { effectGeometry: [], postJobs: [] } });
  assert.deepEqual(received.map(([kind]) => kind).sort(), ['effect', 'mesh']);
  assert.equal(received[0][1], received[1][1],
    'mesh and effect draws share the exact selected-light array');
  assert.equal(received[0][1][0], threadedLight);
  assert.equal(cameraProjectionBuilds, 1,
    'viewport planning builds the camera projection and its frustum only once');
}

// RenderPaintJobs removes the previous light scissor before every
// pass/usage/light transition. Inserts and stencil clears therefore see the
// whole viewport; only the shadow/light draws see the projected LightRect.
{
  const events = [];
  const baseRegion = { x: 10, y: 20, width: 200, height: 100,
    uvRect: new Float32Array([0, 0, 1, 1]) };
  const material = {
    kind: 'material', system: '1.1', multiFlags: 2, parameters: [], textures: [],
    passes: [
      { usage: 'base', renderPass: 0 },
      { usage: 'shadow', renderPass: 0 },
      { usage: 'light', renderPass: 0 },
      { usage: 'postlight', renderPass: 0 },
    ],
  };
  const lights = [
    { opId: 84, kind: 'point', position: [0, 0, 5], range: 1,
      amplify: 1, color: 0xffffffff, flags: 2 },
    { opId: 85, kind: 'point', position: [2, 0, 5], range: 1,
      amplify: 1, color: 0xffffffff, flags: 2 },
  ];
  const viewport = {
    flags: 0, clearColor: 0,
    camera: {
      cameraSpace: D.mat4Identity(), nearClip: 1, farClip: 100,
      zoomX: 1, zoomY: 1, centerX: 0, centerY: 0,
    },
    lightJobs: lights, effectJobs: [],
    meshJobs: [{ opId: 86, mesh: {}, matrix: D.mat4Identity(), passAdjust: 0 }],
  };
  const geometry = {
    groups: [{ material, start: 0, count: 3 }], materials: [],
  };
  const lifecycleGL = {
    DEPTH_TEST: 1, BLEND: 2, CULL_FACE: 3, BACK: 4,
    COLOR_BUFFER_BIT: 8, DEPTH_BUFFER_BIT: 16, STENCIL_BUFFER_BIT: 32,
    STENCIL_TEST: 64,
    enable() {}, disable() {}, depthMask() {}, colorMask() {}, cullFace() {},
    clearColor() {}, clearDepth() {}, clearStencil() {}, stencilMask() {},
    clear(mask) { events.push(['clear', mask]); },
  };
  const lifecycleRenderer = {
    gl: lifecycleGL, shadows: true, cullingStats: {}, prelightTarget: {},
    geometry: { get: () => geometry },
    setViewportRegion() { events.push(['viewport']); return baseRegion; },
    setLightScissor(rectangle, target) {
      assert.equal(target, baseRegion);
      const clipped = D.lightScissorRegion(rectangle, target);
      events.push(['light-scissor', clipped]);
      return clipped;
    },
    instanceMatrices: () => new Float32Array(identity),
    prepareInstanceUploads() {}, captureFramebuffer() {
      assert.fail('material11 lifecycle should not capture prelight');
    },
    bindDestination() {},
    drawMaterial11Insert(action) { events.push(['insert', action]); },
    drawEffectItem() {},
    drawMeshItem(item) { events.push(['draw', item.pass.usage, item.lightIndex ?? -1]); },
  };
  D.Renderer.prototype.drawViewportScene.call(lifecycleRenderer, viewport,
    { frame: { effectGeometry: [], postJobs: [] } });

  let activeScissor = 'viewport';
  for (const event of events) {
    if (event[0] === 'viewport') activeScissor = 'viewport';
    if (event[0] === 'light-scissor') activeScissor = 'light';
    if (event[0] === 'clear' || event[0] === 'insert') {
      assert.equal(activeScissor, 'viewport', `${event[0]} runs with SetScissor(0)`);
    }
    if (event[0] === 'draw') {
      const lit = event[1] === 'shadow' || event[1] === 'light';
      assert.equal(activeScissor, lit ? 'light' : 'viewport',
        `${event[1]} draw uses its native scissor extent`);
    }
  }
  assert.equal(activeScissor, 'viewport', 'the viewport scissor is restored after all jobs');
  assert.deepEqual(events.filter(event => event[0] === 'insert').map(event => event[1]),
    ['clear-destination-alpha', 'add-destination-alpha']);
  assert.equal(events.filter(event => event[0] === 'clear').length, 3,
    'stencil clears occur on light-group entry/change and light exit');
  const lightRegions = events.filter(event => event[0] === 'light-scissor')
    .map(event => event[1]);
  assert.equal(lightRegions.length, 4, 'each shadow/light usage reapplies its light rectangle');
  assert.ok(lightRegions.every(value => value.width < baseRegion.width),
    'point-light bounds are tighter than the complete viewport');
}

// NeedCurrentRender is a property of the complete native render pass. Its
// target is captured once after the preceding pass's AfterUsage insert, then
// shared by every matching effect rather than being recaptured at each job.
{
  const events = [];
  const passSnapshot = { color: 'pass-snapshot' };
  const material = {
    kind: 'material', system: '1.1', multiFlags: 2, parameters: [], textures: [],
    passes: [{ usage: 'base', renderPass: 3 }],
  };
  const viewport = {
    flags: 0, clearColor: 0,
    camera: {
      cameraSpace: D.mat4Identity(), nearClip: 1, farClip: 100,
      zoomX: 1, zoomY: 1, centerX: 0, centerY: 0,
    },
    lightJobs: [],
    meshJobs: [{ opId: 91, mesh: {}, matrix: D.mat4Identity(), passAdjust: 0 }],
    effectJobs: [
      { op: { id: 92 }, effect: { pass: 4, usage: 'base', needCurrentRender: true } },
      { op: { id: 93 }, effect: { pass: 4, usage: 'base', needCurrentRender: true } },
    ],
  };
  const geometry = {
    groups: [{ material, start: 0, count: 3 }], materials: [],
  };
  const passGL = {
    DEPTH_TEST: 1, BLEND: 2, CULL_FACE: 3, BACK: 4,
    COLOR_BUFFER_BIT: 8, DEPTH_BUFFER_BIT: 16, STENCIL_BUFFER_BIT: 32,
    STENCIL_TEST: 64,
    enable() {}, disable() {}, depthMask() {}, colorMask() {}, cullFace() {},
    clearColor() {}, clearDepth() {}, clearStencil() {}, stencilMask() {}, clear() {},
  };
  const passRenderer = {
    gl: passGL, shadows: true, cullingStats: {}, prelightTarget: {},
    geometry: { get: () => geometry },
    setViewportRegion() { return { x: 0, y: 0, width: 100, height: 50 }; },
    instanceMatrices: () => new Float32Array(identity), prepareInstanceUploads() {},
    bindDestination() {}, postTarget() { return passSnapshot; },
    captureFramebuffer(target) { events.push(['capture', target]); },
    drawMaterial11Insert(action) { events.push(['insert', action]); },
    drawMeshItem(item) { events.push(['mesh', item.renderPass]); },
    drawEffectItem(item, _frame, _environment, _viewport, _matrices, _destination,
        _depth, _lights, snapshot) {
      events.push(['effect', item.job.opId, snapshot]);
    },
  };
  D.Renderer.prototype.drawViewportScene.call(passRenderer, viewport,
    { frame: { effectGeometry: [], postJobs: [] } });
  assert.deepEqual(events, [
    ['mesh', 3],
    ['insert', 'clear-destination-alpha'],
    ['capture', passSnapshot],
    ['effect', 93, passSnapshot],
    ['effect', 92, passSnapshot],
  ]);
}

// One packed instance stream retains every unique raster job for the whole
// viewport. Reusing job A after job B therefore needs only an attribute-offset
// change, not another orphan/upload. Shadow items stay on their CPU volume
// path and effect items only reserve a future one-matrix slot.
{
  const bufferDataCalls = [], bufferSubDataCalls = [];
  const pointerCalls = [], divisorCalls = [], draws = [];
  const gl = {
    ARRAY_BUFFER: 1, DYNAMIC_DRAW: 2, FLOAT: 3, TRIANGLES: 4,
    bindBuffer() {}, bindVertexArray() {}, useProgram() {},
    bufferData(_target, value) { bufferDataCalls.push(value); },
    bufferSubData(_target, offset, value) {
      bufferSubDataCalls.push({ offset, values: Array.from(value) });
    },
    vertexAttribPointer(location, size, type, normalized, stride, offset) {
      pointerCalls.push({ location, size, type, normalized, stride, offset });
    },
    vertexAttribDivisor(location, divisor) { divisorCalls.push({ location, divisor }); },
    uniformMatrix4fv() {}, uniform4f() {},
    drawElementsInstanced(...args) { draws.push(args); },
  };
  const renderer = Object.assign(Object.create(D.Renderer.prototype), {
    gl, instanceBuffer: {}, instanceMatrixCache: new WeakMap(),
    instanceUploadCache: new WeakMap(), instanceUploadRecords: [],
    instanceUploadUsedBytes: 0, instanceBufferCapacity: 0, instanceUploadActive: false,
    materialScratch: { singleLight: new Array(1) }, uniforms: {}, program: {},
    drawCalls: 0, triangles: 0,
    viewMaterial: () => ({ uvScale: 1 }), configureMaterialState() {}, bindMaterial() {},
  });
  const matrixA = new Float32Array(identity);
  const matrixB = new Float32Array(identity); matrixB[12] = 3;
  const jobA = { opId: 90, matrix: matrixA };
  const jobB = { opId: 91, matrix: matrixB };
  const emptyJob = { opId: 96, matrix: matrixA, instances: [] };
  const geometry = { vao: {}, indexType: 0, indexBytes: 2 };
  const group = { start: 0, count: 3 };
  const rasterA = { geometry, group, material: null, pass: { usage: 'other' }, job: jobA };
  const rasterB = { geometry, group, material: null, pass: { usage: 'other' }, job: jobB };
  const emptyRaster = {
    geometry, group, material: null, pass: { usage: 'other' }, job: emptyJob,
  };
  const lightA = {
    geometry, group, material: null, pass: { usage: 'light' }, job: jobA,
    light: { opId: 92 },
  };
  renderer.prepareInstanceUploads([
    rasterA, rasterB, lightA, emptyRaster,
    { geometry, group, pass: { usage: 'shadow' }, job: jobA },
    { effectJob: { op: { id: 93 } }, pass: { usage: 'other' } },
  ]);
  assert.equal(bufferDataCalls.length, 1, 'the viewport instance store is orphaned once');
  assert.equal(typeof bufferDataCalls[0], 'number');
  assert.ok(bufferDataCalls[0] >= 3 * 64, 'unique jobs plus one effect slot fit the store');
  assert.deepEqual(bufferSubDataCalls.map(call => call.offset), [0, 64]);

  renderer.drawMeshItem(rasterA, {}, { viewProjection: identity }, []);
  renderer.drawMeshItem(rasterB, {}, { viewProjection: identity }, []);
  renderer.drawMeshItem(lightA, {}, { viewProjection: identity }, []);
  renderer.drawMeshItem(emptyRaster, {}, { viewProjection: identity }, []);
  assert.equal(bufferDataCalls.length, 1, 'material/pass draws do not orphan the packed store');
  assert.equal(bufferSubDataCalls.length, 2, 'each unique job uploads exactly once');
  assert.equal(draws.length, 3);
  assert.equal(renderer.instanceMatrices(emptyJob).length, 0,
    'an explicit zero-instance MPP_INSTANCES job cannot become one base draw');
  assert.deepEqual(pointerCalls.map(call => call.offset), [
    0, 16, 32, 48,
    64, 80, 96, 112,
    0, 16, 32, 48,
  ]);
  assert.equal(pointerCalls.every(call => call.stride === 64), true);
  assert.deepEqual(divisorCalls, [],
    'repeated draw binding keeps the instance divisors already stored in each VAO');

  const effectJob = { opId: 94, matrix: new Float32Array(identity) };
  const effectMatrices = renderer.instanceMatrices(effectJob);
  const effectPointerStart = pointerCalls.length;
  renderer.bindInstanceMatrices(effectJob, effectMatrices);
  renderer.bindInstanceMatrices(effectJob, effectMatrices);
  assert.equal(bufferDataCalls.length, 1, 'the reserved effect slot needs no mid-viewport grow');
  assert.deepEqual(bufferSubDataCalls.map(call => call.offset), [0, 64, 128]);
  assert.deepEqual(pointerCalls.slice(effectPointerStart).map(call => call.offset), [
    128, 144, 160, 176,
    128, 144, 160, 176,
  ]);

  // If an effect emits more matrices than its conservative one-slot reserve,
  // growing the orphaned store must refill every earlier packed record at the
  // same offset before appending the larger effect block. Otherwise a later
  // sorted pass would silently read invalidated instance data.
  const overflowJob = { opId: 95, matrix: new Float32Array(identity) };
  const overflowMatrices = new Float32Array(4 * 16);
  for (let index = 0; index < 4; index++) {
    overflowMatrices.set(identity, index * 16);
    overflowMatrices[index * 16 + 12] = 10 + index;
  }
  const overflowPointerStart = pointerCalls.length;
  renderer.bindInstanceMatrices(overflowJob, overflowMatrices);
  assert.deepEqual(bufferDataCalls, [256, 512],
    'an underestimated effect grows the packed store geometrically once');
  assert.deepEqual(bufferSubDataCalls.slice(3).map(call => call.offset), [
    0, 64, 128, 192,
  ], 'a grow refills earlier records at stable offsets before the new upload');
  assert.deepEqual(bufferSubDataCalls.slice(3, 6).map(call => call.values[12]), [0, 3, 0]);
  assert.deepEqual(bufferSubDataCalls[6].values.filter((_, index) => index % 16 === 12),
    [10, 11, 12, 13]);
  assert.deepEqual(pointerCalls.slice(overflowPointerStart).map(call => call.offset), [
    192, 208, 224, 240,
  ]);
  assert.equal(renderer.instanceUploadRecords.length, 4);
  renderer.bindInstanceMatrices(overflowJob, overflowMatrices);
  assert.equal(bufferSubDataCalls.length, 7,
    'the grown effect block remains cached for later passes');

  // A new viewport/frame recomposes live matrices, orphans once again and
  // drops old job records rather than retaining per-job GPU allocations. Its
  // much smaller demand also releases the temporary oversized capacity.
  matrixA[12] = 9;
  renderer.instanceMatrixCache = new WeakMap();
  renderer.prepareInstanceUploads([rasterA, lightA]);
  assert.deepEqual(bufferDataCalls, [256, 512, 64]);
  assert.equal(bufferSubDataCalls.length, 8);
  assert.equal(bufferSubDataCalls[7].values[12], 9);
  assert.equal(renderer.instanceUploadRecords.length, 1);

  // Ordinary fluctuations reuse a nearby retained allocation, but neither an
  // old oversize peak nor a request above the 4 MiB retention ceiling becomes
  // a permanently retained ordinary viewport buffer.
  renderer.instanceBufferCapacity = 256;
  assert.equal(renderer.instanceBufferSize(128), 256);
  renderer.instanceBufferCapacity = 8 * 1024 * 1024;
  assert.equal(renderer.instanceBufferSize(4 * 1024 * 1024), 4 * 1024 * 1024);
  const oversizeCapacity = 4 * 1024 * 1024 + 64;
  assert.equal(renderer.instanceBufferSize(oversizeCapacity), oversizeCapacity);
  renderer.instanceBufferCapacity = oversizeCapacity;
  assert.equal(renderer.instanceBufferSize(64), 64);

  // The unbatched fallback rewrites the same instance buffer from offset zero.
  // It still only changes the four pointers; the bound geometry VAO retains
  // the divisors established by GeometryCache.
  renderer.instanceUploadActive = false;
  const fallbackJob = { opId: 97, matrix: new Float32Array(identity) };
  const fallbackMatrices = renderer.instanceMatrices(fallbackJob);
  const fallbackPointerStart = pointerCalls.length;
  const fallbackBufferStart = bufferDataCalls.length;
  renderer.bindInstanceMatrices(fallbackJob, fallbackMatrices);
  renderer.bindInstanceMatrices(fallbackJob, fallbackMatrices);
  assert.equal(bufferDataCalls.length, fallbackBufferStart + 2);
  assert.equal(bufferDataCalls.at(-1), fallbackMatrices);
  assert.deepEqual(pointerCalls.slice(fallbackPointerStart).map(call => call.offset), [
    0, 16, 32, 48,
    0, 16, 32, 48,
  ]);
  assert.deepEqual(divisorCalls, []);
}

const ippA = { type: 'viewport', id: 'a' }, ippB = { type: 'viewport', id: 'b' };
assert.deepEqual(Array.from(D.resolveIPPOutputs(ippA, {})), [ippA]);
assert.deepEqual(Array.from(D.resolveIPPOutputs(null, { frameOutputs: [ippA, null, ippB] })), [ippA, ippB]);
assert.deepEqual(Array.from(D.resolveIPPOutputs(ippA, { frameOutputs: [ippB] })), [ippB]);
assert.deepEqual(Array.from(D.resolveIPPOutputs([ippB, ippA], { frameOutputs: [ippA] })), [ippB, ippA]);
assert.equal(D.viewportClearFlags({ flags: 0 }), 0);
assert.equal(D.viewportClearFlags({ flags: 0x102 }), 2);
assert.equal(D.viewportClearFlags({}), 3);
const plainRegion = value => JSON.parse(JSON.stringify(value));
assert.deepEqual(plainRegion(D.fitAspectRegion(1920, 1080, 2)),
  { x: 0, y: 60, width: 1920, height: 960 });
assert.deepEqual(plainRegion(D.fitAspectRegion(1280, 960, 2)),
  { x: 0, y: 160, width: 1280, height: 640 });
assert.deepEqual(plainRegion(D.fitAspectRegion(1280, 640, 2)),
  { x: 0, y: 0, width: 1280, height: 640 });
assert.deepEqual(plainRegion(D.fitAspectRegion(3440, 1440, 2)),
  { x: 280, y: 0, width: 2880, height: 1440 });
assert.deepEqual(plainRegion(D.fitAspectRegion(321, 181, 2)),
  { x: 0, y: 10, width: 320, height: 160 },
  'odd outer dimensions retain an exact integer 2:1 production raster');
assert.deepEqual(plainRegion(D.fitAspectRegion(320, 180)),
  { x: 0, y: 0, width: 320, height: 180 });

const targetResizeCalls = [];
const resizeCanvas = { clientWidth: 160, clientHeight: 90, width: 0, height: 0 };
const resizeRenderer = {
  canvas: resizeCanvas, pixelRatio: 0, maxPixelRatio: 2,
  masterTarget: { resize: (width, height) => targetResizeCalls.push(['master', width, height]) },
  prelightTarget: { resize: (width, height) => targetResizeCalls.push(['prelight', width, height]) },
  targets: [{ resize: (width, height) => targetResizeCalls.push(['scratch', width, height]) }],
};
const resized = D.Renderer.prototype.resize.call(resizeRenderer, 160, 90, 2, 2);
assert.deepEqual([resizeCanvas.width, resizeCanvas.height], [320, 180],
  'DPR scales the complete outer drawing buffer');
assert.deepEqual([resizeRenderer.width, resizeRenderer.height], [320, 160],
  'full-resolution render targets use the inner 2:1 dimensions');
assert.deepEqual(plainRegion(resizeRenderer.presentationRegion),
  { x: 0, y: 10, width: 320, height: 160 });
assert.deepEqual(targetResizeCalls, [
  ['master', 320, 160], ['prelight', 320, 160], ['scratch', 320, 160],
]);
assert.deepEqual(plainRegion(resized.presentationRegion),
  { x: 0, y: 10, width: 320, height: 160 });

let region = D.viewportRegion({ crop: [0.25, 0.1, 0.75, 0.6] }, 200, 100);
assert.deepEqual([region.x, region.y, region.width, region.height], [50, 40, 100, 50]);
assert.deepEqual(Array.from(region.uvRect, value => Number(value.toFixed(4))), [0.25, 0.4, 0.75, 0.9]);
region = D.viewportRegion({ crop: [0, 0, 0, 0] }, 200, 100);
assert.deepEqual([region.x, region.y, region.width, region.height], [0, 0, 200, 100]);

// sSystem_::SetScissor truncates normalized D3D light bounds against the
// current viewport, then flips its top-down Y interval for WebGL.
const lightViewport = { x: 50, y: 40, width: 100, height: 50 };
assert.deepEqual(D.lightScissorRegion([-0.5, -0.25, 0.75, 0.5], lightViewport),
  { x: 75, y: 59, width: 62, height: 19 });
assert.deepEqual(D.lightScissorRegion([-1, -1, 1, 1], lightViewport), lightViewport,
  'the native full normalized rectangle exactly restores the cropped viewport');
assert.deepEqual(D.lightScissorRegion([-4, -3, 2, 5], lightViewport), lightViewport,
  'out-of-range sphere edges are intersected with the active viewport');
assert.deepEqual(D.lightScissorRegion([NaN, 0, 1, 1], lightViewport), lightViewport,
  'unknown bounds conservatively retain the full viewport');
const lightScissorCalls = [];
const lightScissorGL = {
  SCISSOR_TEST: 9,
  enable(cap) { lightScissorCalls.push(['enable', cap]); },
  scissor(...values) { lightScissorCalls.push(['scissor', ...values]); },
};
const appliedLightRegion = D.Renderer.prototype.setLightScissor.call(
  { gl: lightScissorGL }, [-0.5, -0.25, 0.75, 0.5], lightViewport,
);
assert.deepEqual(appliedLightRegion, { x: 75, y: 59, width: 62, height: 19 });
assert.deepEqual(lightScissorCalls, [
  ['enable', lightScissorGL.SCISSOR_TEST], ['scissor', 75, 59, 62, 19],
]);

const layerGeometry = D.layerQuadGeometry({
  screen: [0.25, 0.1, 0.75, 0.6], uv: [0.2, 0.3, 0.8, 0.9], z: 0.75,
});
assert.deepEqual(Array.from(layerGeometry.positions, value => Number(value.toFixed(6))), [
  -0.5, 0.8, 0.5, 0.5, 0.8, 0.5,
  0.5, -0.2, 0.5, -0.5, -0.2, 0.5,
]);
assert.deepEqual(Array.from(layerGeometry.uvs, value => Number(value.toFixed(6))),
  [0.2, 0.3, 0.8, 0.3, 0.8, 0.9, 0.2, 0.9]);
const layerArea = (layerGeometry.positions[3] - layerGeometry.positions[0]) *
  (layerGeometry.positions[7] - layerGeometry.positions[1]) -
  (layerGeometry.positions[4] - layerGeometry.positions[1]) *
  (layerGeometry.positions[6] - layerGeometry.positions[0]);
assert.ok(layerArea < 0, 'Layer2D source order is clockwise like the native quad');

// The player build has no overlay render-target pool: Layer2D renders its
// child directly into the destination (preserving depth/stencil), then applies
// its authored clear and every linked material pass.
const layerCalls = [];
const layerMaterial = { passes: [{ id: 1 }, { id: 2 }] };
const layerGL = {
  COLOR_BUFFER_BIT: 1, DEPTH_BUFFER_BIT: 2, STENCIL_BUFFER_BIT: 4,
  clearColor() { layerCalls.push(['clear-color']); },
  colorMask() {}, depthMask() {}, stencilMask() {},
  clear(mask) { layerCalls.push(['clear', mask]); },
};
const layerRenderer = {
  gl: layerGL, currentPrelightTexture: 'stale-prelight',
  renderNode(input, environment, target, depth) {
    layerCalls.push(['child', input, target, depth]);
  },
  bindDestination(target) { layerCalls.push(['bind', target]); },
  drawLayerQuad(layer, material, pass) { layerCalls.push(['quad', material, pass.id]); },
};
const layerInput = { type: 'viewport' }, layerDestination = { color: 'destination' };
D.Renderer.prototype.renderLayer.call(layerRenderer, {
  input: layerInput, material: layerMaterial, clearFlags: 3,
}, {}, layerDestination, 2);
assert.deepEqual(layerCalls, [
  ['child', layerInput, layerDestination, 3],
  ['bind', layerDestination],
  ['clear-color'], ['clear', 7],
  ['quad', layerMaterial, 1], ['quad', layerMaterial, 2],
]);
assert.equal(layerRenderer.currentPrelightTexture, null);

// Renderer.render must clear the native master viewport first and then prefer
// the ordered Demo outputs even when a trailing inactive branch left
// lastOutput null. These flags model the no-clear Layer2D/Viewport roots at
// production beats 1010 and 1410: neither is allowed to inherit the prior
// frame's color, depth, or stencil.
const sequenceCalls = [];
let sequenceDestination = null;
let sequenceClearColor = null;
let sequencePresentedPixel = null;
const sequenceGL = {
  COLOR_BUFFER_BIT: 1, DEPTH_BUFFER_BIT: 2, STENCIL_BUFFER_BIT: 4,
  colorMask(...values) { sequenceCalls.push(['color-mask', ...values]); },
  depthMask(value) { sequenceCalls.push(['depth-mask', value]); },
  stencilMask(value) { sequenceCalls.push(['stencil-mask', value]); },
  clearColor(...values) {
    sequenceClearColor = values;
    sequenceCalls.push(['clear-color', ...values]);
  },
  clearDepth(value) { sequenceCalls.push(['clear-depth', value]); },
  clearStencil(value) { sequenceCalls.push(['clear-stencil', value]); },
  clear(mask) {
    if (mask & this.COLOR_BUFFER_BIT) sequenceDestination.pixel = sequenceClearColor.slice();
    sequenceCalls.push(['clear', mask]);
  },
};
const sequenceRenderer = Object.assign(Object.create(D.Renderer.prototype), {
  contextLost: false, width: 1, height: 1, drawCalls: 0, triangles: 0,
  masterTarget: { color: 'master-rgba8', pixel: [1, 0, 1, 0] }, gl: sequenceGL,
  geometry: { beginFrame() { sequenceCalls.push(['begin-frame']); } },
  resize(width, height) {
    sequenceCalls.push(['resize', width, height]);
    this.width = this.canvasWidth = width; this.height = this.canvasHeight = height;
    this.presentationRegion = { x: 0, y: 0, width, height };
  },
  bindDestination(target) {
    sequenceDestination = target;
    sequenceCalls.push(['bind', target.color]);
  },
  renderNode(node) { sequenceCalls.push(['node', node.id]); this.drawCalls++; },
  present(image) {
    sequencePresentedPixel = this.masterTarget.pixel.slice();
    sequenceCalls.push(['present', image]); this.drawCalls++;
  },
});
const sequenceResult = D.Renderer.prototype.render.call(sequenceRenderer, null,
  { beatTime: Math.round(1010 * 65536), frameOutputs: [ippA, ippB] },
  { width: 320, height: 180, pixelRatio: 1 });
assert.deepEqual(sequenceCalls, [
  ['resize', 320, 180], ['begin-frame'], ['bind', 'master-rgba8'],
  ['color-mask', true, true, true, true], ['depth-mask', true], ['stencil-mask', 0xff],
  ['clear-color', 0, 0, 0, 1], ['clear-depth', 1], ['clear-stencil', 0], ['clear', 7],
  ['node', 'a'], ['node', 'b'], ['present', 'master-rgba8'],
]);
assert.equal(sequenceResult.outputs, 2);
assert.equal(sequenceResult.drawCalls, 3);

sequenceCalls.length = 0;
sequenceRenderer.masterTarget.pixel = [1, 0, 1, 0];
const beat1410NoClear = {
  type: 'layer2d', id: 'beat-1410-no-clear', flags: 0x20, clearFlags: 0,
  input: { type: 'viewport', flags: 0 },
};
const beat1410Result = D.Renderer.prototype.render.call(sequenceRenderer, null,
  { beatTime: Math.round(1410 * 65536), frameOutputs: [beat1410NoClear] },
  { width: 320, height: 180, pixelRatio: 1 });
assert.ok(sequenceCalls.findIndex(call => call[0] === 'clear') <
  sequenceCalls.findIndex(call => call[0] === 'node'),
  'beat 1410 clears the master before its authored no-clear Layer2D root');
assert.deepEqual(sequencePresentedPixel, [0, 0, 0, 1]);
assert.equal(beat1410Result.outputs, 1);

// At production beat 1424.5 there is no active Demo output. The native player
// still clears and flips an opaque-black frame; the port must not leave the
// previous canvas contents visible.
sequenceCalls.length = 0;
const emptyResult = D.Renderer.prototype.render.call(sequenceRenderer, null,
  { beatTime: Math.round(1424.5 * 65536), frameOutputs: [] },
  { width: 320, height: 180, pixelRatio: 1 });
assert.deepEqual(sequenceCalls, [
  ['resize', 320, 180], ['begin-frame'], ['bind', 'master-rgba8'],
  ['color-mask', true, true, true, true], ['depth-mask', true], ['stencil-mask', 0xff],
  ['clear-color', 0, 0, 0, 1], ['clear-depth', 1], ['clear-stencil', 0], ['clear', 7],
  ['present', 'master-rgba8'],
]);
assert.deepEqual({ drawCalls: emptyResult.drawCalls, triangles: emptyResult.triangles,
  outputs: emptyResult.outputs }, { drawCalls: 1, triangles: 0, outputs: 0 });
assert.deepEqual(sequencePresentedPixel, [0, 0, 0, 1],
  'the zero-output frame presents opaque black rather than stale master color');

// A lost context remains the sole early return and must not issue GL calls.
sequenceCalls.length = 0;
sequenceRenderer.contextLost = true;
assert.deepEqual(D.Renderer.prototype.render.call(sequenceRenderer, ippA, {}, {
  width: 320, height: 180, pixelRatio: 1,
}), { drawCalls: 0, triangles: 0, outputs: 0 });
assert.deepEqual(sequenceCalls, []);
sequenceRenderer.contextLost = false;

// Viewport rendering hands the environment to one unified mesh/effect stream;
// there is no delayed post-only phase after the scene.
const postCalls = [];
const scratch0 = { color: 'screen-copy-0' }, scratch1 = { color: 'screen-copy-1' };
const postRenderer = {
  width: 320, height: 180,
  bindDestination(destination) { postCalls.push(['bind', destination]); },
  drawViewportScene(viewport, environment, destination, depth) {
    postCalls.push(['scene', viewport, environment, destination, depth]);
  },
};
const postEnvironment = { frame: { effectGeometry: [], postJobs: [] } };
const postViewport = {
  flags: 0, crop: [0, 0, 1, 1],
  effectJobs: [{ effect: { pass: 0, usage: 'base' }, postJob: { kind: 'glare' } }],
};
D.Renderer.prototype.renderViewport.call(postRenderer, postViewport, postEnvironment, null, 3);
assert.deepEqual(postCalls, [
  ['bind', null], ['scene', postViewport, postEnvironment, null, 3],
]);

// An ordinary inline post effect captures and replaces the current destination
// at its sorted job, then restores the viewport for later mesh jobs.
const inlineCalls = [];
const inlineFrame = { effectGeometry: [], postJobs: [] };
const inlineViewport = { crop: [0, 0, 1, 1] };
const inlineDestination = { color: 'destination' };
const inlineRenderer = {
  effectExecutor,
  postTarget() { return scratch0; },
  captureFramebuffer(target, source) { inlineCalls.push(['capture', target, source]); },
  bindDestination(target) { inlineCalls.push(['bind', target]); },
  setViewportRegion(target) { inlineCalls.push(['region', target]); return { uvRect: [0, 0, 1, 1] }; },
  applyPost(image, job, options) { inlineCalls.push(['post', image, job.kind, options.destination]); },
  drawEffectGeometry() { inlineCalls.push(['geometry']); },
};
D.Renderer.prototype.drawEffectItem.call(inlineRenderer, {
  effectJob: { effect: { needCurrentRender: false }, postJob: { kind: 'glare' },
    variables: [], matrix: D.mat4Identity(), op: {} },
}, inlineFrame, { frame: inlineFrame, vars: [], markers: [], matrixStack: { push() {}, pop() {} } },
inlineViewport, {}, inlineDestination, 2);
assert.deepEqual(inlineCalls, [
  ['region', inlineViewport], ['capture', scratch0, inlineDestination],
  ['bind', inlineDestination],
  ['region', inlineViewport], ['post', 'screen-copy-0', 'glare', inlineDestination],
  ['bind', inlineDestination], ['region', inlineViewport],
]);

// ColorCorrection consumes the pass-boundary snapshot supplied by
// drawViewportScene and must not overwrite it with a later job-time capture.
inlineCalls.length = 0;
D.Renderer.prototype.drawEffectItem.call(inlineRenderer, {
  effectJob: { effect: { needCurrentRender: true },
    postJob: { kind: 'color-correction' }, variables: [], matrix: D.mat4Identity(), op: {} },
}, inlineFrame, { frame: inlineFrame, vars: [], markers: [], matrixStack: { push() {}, pop() {} } },
inlineViewport, {}, inlineDestination, 2, null, scratch1);
assert.deepEqual(inlineCalls, [
  ['bind', inlineDestination],
  ['region', inlineViewport], ['post', 'screen-copy-1', 'color-correction', inlineDestination],
  ['bind', inlineDestination], ['region', inlineViewport],
]);

// Runtime-generated effect geometry must keep the viewport-selected array all
// the way to its material bind. Water delegates to drawMeshItem; ChainLine
// binds its dynamic ribbon directly.
{
  const activeLights = [{ opId: 120 }];
  const effectGeometry = { kind: 'test-generated-geometry' };
  const frame = { effectGeometry: [], postJobs: [] };
  let generatedLights = null;
  const effectRenderer = {
    effectExecutor,
    drawEffectGeometry(_geometry, _viewport, _matrices, lights) {
      generatedLights = lights;
    },
  };
  D.Renderer.prototype.drawEffectItem.call(effectRenderer, {
    effectJob: { effectGeometry },
  }, frame, { frame }, {}, {}, null, 0, activeLights);
  assert.equal(generatedLights, activeLights);

  let waterLights = null, waterMatrix = null;
  const waterTransform = D.mat4Identity(); waterTransform[12] = 17;
  D.Renderer.prototype.drawEffectGeometry.call({
    geometry: { get: () => ({ groups: [{ start: 0, count: 3 }] }) },
    drawMeshItem(item, _viewport, _matrices, lights) {
      waterLights = lights;
      waterMatrix = item.job.matrix;
    },
  }, {
    kind: 'water', geometry: {}, material: null, matrix: waterTransform, opId: 121,
  }, {}, {}, activeLights);
  assert.equal(waterLights, activeLights);
  assert.equal(waterMatrix, waterTransform,
    'Water retains its authored deferred-effect transform');

  const chainGL = {
    ARRAY_BUFFER: 1, ELEMENT_ARRAY_BUFFER: 2, DYNAMIC_DRAW: 3,
    FLOAT: 4, TRIANGLES: 5, UNSIGNED_SHORT: 6,
    createVertexArray: () => ({}), createBuffer: () => ({}),
    bindVertexArray() {}, bindBuffer() {}, bufferData() {},
    enableVertexAttribArray() {}, disableVertexAttribArray() {},
    vertexAttribPointer() {}, vertexAttrib3f() {}, vertexAttrib4f() {},
    vertexAttrib4fv() {}, vertexAttribDivisor() {}, useProgram() {},
    uniformMatrix4fv() {}, uniform4f() {}, drawElements() {},
  };
  const chainColumns = [];
  chainGL.vertexAttrib4fv = (location, values) => {
    chainColumns[location - 5] = Array.from(values);
  };
  let chainLights = null, chainMaterialMatrix = null;
  const chainRenderer = {
    gl: chainGL, program: {}, uniforms: {}, drawCalls: 0, triangles: 0,
    viewMaterial: () => ({}), configureMaterialState() {},
    bindMaterial(_view, _viewport, _matrices, lights, modelMatrix) {
      chainLights = lights;
      chainMaterialMatrix = modelMatrix;
    },
  };
  const chainParent = D.mat4Identity();
  chainParent[0] = 2; chainParent[5] = 3; chainParent[10] = 4;
  chainParent[12] = 50;
  D.Renderer.prototype.drawEffectGeometry.call(chainRenderer, {
    kind: 'chain-line', points: [new Float32Array([0, 0, 0]), new Float32Array([1, 0, 0])],
    matrix: chainParent, material: { passes: [{ usage: 'other' }] }, thickness: 1,
  }, {}, { view: identity, viewProjection: identity }, activeLights);
  assert.equal(chainLights, activeLights);
  assert.deepEqual(chainColumns, [
    [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
  ], 'world-space ChainLine vertices bind an identity instance matrix');
  assert.deepEqual(Array.from(chainMaterialMatrix), Array.from(identity),
    'ChainLine material coordinates also remain in world space');

  // Empty arrays are deliberate (Layer2D and lightless viewport paths), while
  // explicit per-light passes must still override the viewport-wide array.
  const meshGL = {
    ARRAY_BUFFER: 1, DYNAMIC_DRAW: 2, TRIANGLES: 3,
    bindVertexArray() {}, bindBuffer() {}, bufferData() {},
    useProgram() {}, uniformMatrix4fv() {}, uniform4f() {}, drawElementsInstanced() {},
  };
  const boundLights = [];
  const meshRenderer = {
    gl: meshGL, uniforms: {}, program: {}, instanceBuffer: {},
    materialScratch: { singleLight: new Array(1) }, drawCalls: 0, triangles: 0,
    viewMaterial: () => ({ uvScale: 1 }), configureMaterialState() {},
    instanceMatrices: () => new Float32Array(identity),
    bindInstanceMatrices() {},
    bindMaterial(_view, _viewport, _matrices, lights) { boundLights.push(lights); },
  };
  const meshItem = {
    geometry: { vao: {}, indexType: 0, indexBytes: 2 },
    group: { start: 0, count: 3 }, material: null,
    pass: { usage: 'other' }, job: { matrix: identity },
  };
  const noLights = [];
  D.Renderer.prototype.drawMeshItem.call(meshRenderer, meshItem, {},
    { viewProjection: identity }, noLights);
  const explicitLight = { opId: 122 };
  D.Renderer.prototype.drawMeshItem.call(meshRenderer, {
    ...meshItem, pass: { usage: 'light' }, light: explicitLight,
  }, {}, { viewProjection: identity }, activeLights);
  assert.equal(boundLights[0], noLights);
  assert.notEqual(boundLights[1], activeLights);
  assert.deepEqual(boundLights[1], [explicitLight]);
}

// Animated MinMeshes may occur more than once with distinct scene times in a
// single viewport. Each time needs a stable entry until all sorted passes draw,
// while the bounded entry pool must recycle its VAOs/buffers next frame.
const gpuCounts = {
  createVAO: 0, createBuffer: 0, deleteVAO: 0, deleteBuffer: 0, uploads: 0,
  divisorCalls: [],
};
const fakeGL = {
  ARRAY_BUFFER: 1, ELEMENT_ARRAY_BUFFER: 2, STATIC_DRAW: 3, DYNAMIC_DRAW: 4,
  FLOAT: 5, UNSIGNED_BYTE: 6, UNSIGNED_SHORT: 7, UNSIGNED_INT: 8,
  createVertexArray() { gpuCounts.createVAO++; return { vao: gpuCounts.createVAO }; },
  createBuffer() { gpuCounts.createBuffer++; return { buffer: gpuCounts.createBuffer }; },
  bindVertexArray() {}, bindBuffer() {},
  bufferData() { gpuCounts.uploads++; },
  enableVertexAttribArray() {}, vertexAttribPointer() {},
  vertexAttribDivisor(location, divisor) { gpuCounts.divisorCalls.push({ location, divisor }); },
  deleteVertexArray() { gpuCounts.deleteVAO++; },
  deleteBuffer() { gpuCounts.deleteBuffer++; },
};
const preparedTimes = [], preparedSlots = [];
const animatedMesh = {
  kind: 'minmesh',
  hasAnimation: () => true,
  prepare({ time, animationSlot }) {
    preparedTimes.push(time);
    preparedSlots.push(animationSlot);
    return {
      positions: new Float32Array([time, 0, 0, time + 1, 0, 0, time, 1, 0]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      indices: new Uint16Array([0, 1, 2]), groups: [{ start: 0, count: 3 }],
    };
  },
};
const animatedCache = new D.GeometryCache(fakeGL, { instance: true });
animatedCache.beginFrame();
const animatedAtOne = animatedCache.get(animatedMesh, 1, 101);
assert.equal(animatedCache.get(animatedMesh, 1), animatedAtOne, 'all passes for one job retain one entry');
const animatedAtTwo = animatedCache.get(animatedMesh, 2);
assert.notEqual(animatedAtTwo, animatedAtOne, 'distinct times in one frame cannot share mutable buffers');
assert.equal(animatedAtOne.positions[0], 1);
assert.equal(animatedAtTwo.positions[0], 2);
assert.deepEqual(preparedTimes, [1, 2]);
assert.deepEqual(preparedSlots, [0, 1]);
assert.equal(gpuCounts.createVAO, 2);
assert.equal(gpuCounts.createBuffer, 14);
assert.deepEqual(gpuCounts.divisorCalls, [
  { location: 5, divisor: 1 }, { location: 6, divisor: 1 },
  { location: 7, divisor: 1 }, { location: 8, divisor: 1 },
  { location: 5, divisor: 1 }, { location: 6, divisor: 1 },
  { location: 7, divisor: 1 }, { location: 8, divisor: 1 },
], 'each GeometryCache VAO configures all four instance-column divisors once');
for (let frame = 0; frame < 8; frame++) {
  animatedCache.beginFrame();
  assert.equal(animatedCache.get(animatedMesh, frame + 10), animatedAtOne);
  assert.equal(animatedCache.get(animatedMesh, frame + 20), animatedAtTwo);
}
assert.deepEqual(preparedSlots.slice(2), Array.from({ length: 16 }, (_, index) => index & 1),
  'animation output slots track the bounded geometry-entry pool');
assert.equal(gpuCounts.createVAO, 2, 'animated rendering does not allocate a VAO per frame');
assert.equal(gpuCounts.createBuffer, 14, 'animated rendering does not allocate buffers per frame');
const animatedStats = animatedCache.resourceStats();
assert.equal(animatedStats.entries, 2);
assert.equal(animatedStats.animatedEntries, 2);
assert.ok(animatedStats.gpuBytes > 0);
assert.deepEqual(Array.from(animatedAtOne.sourceIds), [101]);
animatedCache.dispose();
assert.equal(gpuCounts.deleteVAO, 2);
assert.equal(gpuCounts.deleteBuffer, 14);

// Static prepared geometry can bypass normalization only while both its
// prepared identity and generation match. A changed version refreshes the
// existing buffers; a changed prepared object creates a distinct cache entry.
const staticCache = new D.GeometryCache(fakeGL, { instance: true });
let staticPrepareCalls = 0;
const makeStaticPrepared = version => ({
  kind: 'mesh-buffer', version,
  positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
  indices: new Uint16Array([0, 1, 2]), groups: [{ start: 0, count: 3 }],
});
const staticMesh = {
  kind: 'mesh', _prepared: makeStaticPrepared(0),
  prepare() { staticPrepareCalls++; return this._prepared; },
};
let uploadStart = gpuCounts.uploads;
const staticEntry = staticCache.get(staticMesh);
const staticUploadCount = gpuCounts.uploads - uploadStart;
assert.equal(staticPrepareCalls, 1);
assert.equal(staticCache.get(staticMesh), staticEntry);
assert.equal(staticPrepareCalls, 1,
  'an unchanged static prepared source skips prepare and normalization');
assert.equal(gpuCounts.uploads - uploadStart, staticUploadCount);
staticMesh._prepared.version++;
assert.equal(staticCache.get(staticMesh), staticEntry);
assert.equal(staticPrepareCalls, 2, 'a changed prepared generation is normalized again');
assert.equal(gpuCounts.uploads - uploadStart, staticUploadCount * 2);
staticMesh._prepared = makeStaticPrepared(0);
const replacedStaticEntry = staticCache.get(staticMesh);
assert.notEqual(replacedStaticEntry, staticEntry);
assert.equal(staticPrepareCalls, 3, 'a changed prepared identity cannot use the old entry');

// Direct dynamic producers such as Water use themselves as the prepared
// source. Their stable version is fast, but a generation bump still performs
// the existing in-place GPU refresh.
const dynamicGeometry = makeStaticPrepared(0);
uploadStart = gpuCounts.uploads;
const dynamicEntry = staticCache.get(dynamicGeometry);
const dynamicUploadCount = gpuCounts.uploads - uploadStart;
assert.equal(staticCache.get(dynamicGeometry), dynamicEntry);
assert.equal(gpuCounts.uploads - uploadStart, dynamicUploadCount);
dynamicGeometry.version++;
assert.equal(staticCache.get(dynamicGeometry), dynamicEntry);
assert.equal(gpuCounts.uploads - uploadStart, dynamicUploadCount * 2);

// Procedural geometry can declare a narrow dirty set. UV0/UV1 share source
// storage on Water but have separate GL buffers, so changing UVs refreshes
// both without resending colors, tangents, or indices.
const partialGeometry = makeStaticPrepared(0);
partialGeometry.normals = new Float32Array(9);
partialGeometry.uvs = new Float32Array(6);
partialGeometry.dynamicAttributes = ['positions', 'normals'];
uploadStart = gpuCounts.uploads;
const partialEntry = staticCache.get(partialGeometry);
const partialCreateUploads = gpuCounts.uploads - uploadStart;
assert.equal(partialCreateUploads, 7);
partialGeometry.version++;
staticCache.get(partialGeometry);
assert.equal(gpuCounts.uploads - uploadStart, partialCreateUploads + 2,
  'only position and normal buffers refresh');
partialGeometry.dynamicAttributes = ['positions', 'normals', 'uvs'];
partialGeometry.version++;
staticCache.get(partialGeometry);
assert.equal(gpuCounts.uploads - uploadStart, partialCreateUploads + 6,
  'shared UV0/UV1 buffers both refresh when UV layout changes');
staticCache.dispose();

// Updating an animated entry must refresh its own VAO. ELEMENT_ARRAY_BUFFER is
// VAO state in WebGL2, so doing this against the previously drawn VAO silently
// replaces that mesh's indices. Color storage can also legitimately switch
// between packed bytes and authored floats across producer revisions.
let trackedVAO = null, trackedArrayBuffer = null, trackedId = 0;
const trackedElementBuffers = new Map(), trackedColorPointers = new Map();
const trackedGL = {
  ARRAY_BUFFER: 1, ELEMENT_ARRAY_BUFFER: 2, STATIC_DRAW: 3, DYNAMIC_DRAW: 4,
  FLOAT: 5, UNSIGNED_BYTE: 6, UNSIGNED_SHORT: 7, UNSIGNED_INT: 8,
  createVertexArray() { return { vao: ++trackedId }; },
  createBuffer() { return { buffer: ++trackedId }; },
  bindVertexArray(vao) { trackedVAO = vao; },
  bindBuffer(target, buffer) {
    if (target === this.ARRAY_BUFFER) trackedArrayBuffer = buffer;
    else trackedElementBuffers.set(trackedVAO, buffer);
  },
  bufferData() {}, enableVertexAttribArray() {}, vertexAttribDivisor() {},
  vertexAttribPointer(location, size, type, normalized) {
    if (location === 3) trackedColorPointers.set(trackedVAO, {
      buffer: trackedArrayBuffer, size, type, normalized,
    });
  },
  deleteVertexArray() {}, deleteBuffer() {},
};
const trackedPackedColors = new Uint8Array(12).fill(255);
const trackedFloatColors = new Float32Array(12).fill(1);
const colorChangingMesh = {
  kind: 'minmesh', hasAnimation: () => true,
  prepare({ time }) {
    return {
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      indices: new Uint16Array([0, 1, 2]),
      colors: time === 1 ? trackedPackedColors : trackedFloatColors,
    };
  },
};
const trackedCache = new D.GeometryCache(trackedGL, { instance: true });
trackedCache.beginFrame();
const trackedEntry = trackedCache.get(colorChangingMesh, 1);
assert.equal(trackedColorPointers.get(trackedEntry.vao).type, trackedGL.UNSIGNED_BYTE);
assert.equal(trackedColorPointers.get(trackedEntry.vao).normalized, true);
const unrelatedVAO = { vao: 'unrelated' }, unrelatedIndexBuffer = { buffer: 'unrelated' };
trackedGL.bindVertexArray(unrelatedVAO);
trackedGL.bindBuffer(trackedGL.ELEMENT_ARRAY_BUFFER, unrelatedIndexBuffer);
trackedCache.beginFrame();
assert.equal(trackedCache.get(colorChangingMesh, 2), trackedEntry);
assert.equal(trackedElementBuffers.get(unrelatedVAO), unrelatedIndexBuffer,
  'entry refresh cannot mutate the previously bound VAO');
assert.equal(trackedElementBuffers.get(trackedEntry.vao), trackedEntry.buffers[6]);
assert.equal(trackedColorPointers.get(trackedEntry.vao).buffer, trackedEntry.buffers[3]);
assert.equal(trackedColorPointers.get(trackedEntry.vao).type, trackedGL.FLOAT);
assert.equal(trackedColorPointers.get(trackedEntry.vao).normalized, false);

// Two triangles use duplicated render vertices along their shared diagonal.
// Shadow topology welds those wedges back to the four source positions.
const shadowGeometry = {
  positions: new Float32Array([
    0, 0, 0, 1, 0, 0, 1, 1, 0,
    0, 0, 0, 1, 1, 0, 0, 1, 0,
  ]),
  indices: new Uint16Array([0, 1, 2, 3, 4, 5]),
  groups: [{ start: 0, count: 6 }],
};
let volume = D.buildShadowVolume(shadowGeometry, shadowGeometry.groups, [0.5, 0.5, -1]);
assert.equal(volume.topology.positions.length / 3, 4);
assert.equal(volume.topology.boundaryEdges, 4);
assert.equal(volume.topology.nonManifoldEdges, 0);
assert.equal(volume.topology.maxEdgeIncidence, 2);
assert.deepEqual(Array.from(volume.faceFront), [0, 0]);
assert.equal(volume.silhouetteIndexCount, 24);
assert.equal(volume.capIndexCount, 6);
assert.equal(volume.indices.length, 30);
assert.deepEqual(Array.from(volume.extrusions), [0, 1, 0, 1, 0, 1, 0, 1]);
const caplessVolume = D.buildShadowVolume(
  shadowGeometry, shadowGeometry.groups, [0.5, 0.5, -1], identity, false);
assert.equal(caplessVolume.silhouetteIndexCount, volume.silhouetteIndexCount);
assert.equal(caplessVolume.capIndexCount, 0);
assert.equal(caplessVolume.indices.length, volume.silhouetteIndexCount,
  'native z-pass uploads only silhouette sides');
volume = D.buildShadowVolume(shadowGeometry, shadowGeometry.groups, [0.5, 0.5, 1]);
assert.deepEqual(Array.from(volume.faceFront), [1, 1]);
assert.equal(volume.silhouetteIndexCount, 0);
assert.equal(volume.capIndexCount, 6);
const maskedShadowGeometry = { ...shadowGeometry, shadowTriangleMask: new Uint8Array([1, 0]) };
volume = D.buildShadowVolume(maskedShadowGeometry, maskedShadowGeometry.groups, [0.5, 0.5, -1]);
assert.equal(volume.faceFront.length, 1);
assert.equal(volume.silhouetteIndexCount, 18);
assert.equal(volume.capIndexCount, 3);

// Animated MinMesh keeps indices, weld maps and groups stable while skinning
// positions in-place. Preserve its expensive edge topology and refresh the
// canonical/doubled position arrays from the retained render-vertex sources.
const rendererTypedArray = (Type, values) => {
  const constructors = { Float32Array, Uint16Array, Uint32Array, Uint8Array };
  return new constructors[Type](values);
};
const movingShadowGeometry = {
  kind: 'indexed-geometry', version: 0,
  positions: rendererTypedArray('Float32Array', shadowGeometry.positions),
  normals: rendererTypedArray('Float32Array', new Float32Array(18)),
  tangents: rendererTypedArray('Float32Array', new Float32Array(24)),
  indices: rendererTypedArray('Uint16Array', shadowGeometry.indices),
  groups: [{ start: 0, count: 6 }],
  shadowVertexMap: rendererTypedArray('Uint32Array', [0, 1, 2, 0, 2, 3]),
  shadowTriangleMask: rendererTypedArray('Uint8Array', [1, 1]),
  dynamicAttributes: ['positions', 'normals', 'tangents'],
};
const movingShadowCache = new D.GeometryCache(fakeGL, { instance: true });
const movingEntry = movingShadowCache.get(movingShadowGeometry);
const movingTopology = D.prepareShadowTopology(movingEntry, movingEntry.groups);
assert.deepEqual(Array.from(movingTopology.sourceIndices), [0, 1, 2, 5]);
movingEntry.shadowTopologies.set('0:6', movingTopology);
const movingTopologyMap = movingEntry.shadowTopologies;
movingShadowGeometry.positions.set([1, 1, 0.75], 2 * 3);
movingShadowGeometry.positions.set([1, 1, 0.75], 4 * 3);
movingShadowGeometry.positions.set([0, 1, 0.25], 5 * 3);
movingShadowGeometry.version++;
assert.equal(movingShadowCache.get(movingShadowGeometry), movingEntry);
assert.equal(movingEntry.shadowTopologies, movingTopologyMap);
assert.equal(movingEntry.shadowTopologies.get('0:6'), movingTopology);
assert.deepEqual(Array.from(movingTopology.positions), [
  0, 0, 0, 1, 0, 0, 1, 1, 0.75, 0, 1, 0.25,
]);
assert.deepEqual(Array.from(movingTopology.volumePositions), [
  0, 0, 0, 0, 0, 0,
  1, 0, 0, 1, 0, 0,
  1, 1, 0.75, 1, 1, 0.75,
  0, 1, 0.25, 0, 1, 0.25,
]);
const movingLight = [0.5, 0.5, -1];
const reusedMovingVolume = D.buildShadowVolume(
  movingEntry, movingTopology, movingLight, identity, true);
const freshMovingVolume = D.buildShadowVolume(
  movingEntry, movingEntry.groups, movingLight, identity, true);
assert.deepEqual(Array.from(reusedMovingVolume.faceFront), Array.from(freshMovingVolume.faceFront));
assert.deepEqual(Array.from(reusedMovingVolume.indices), Array.from(freshMovingVolume.indices));
assert.equal(reusedMovingVolume.silhouetteIndexCount, freshMovingVolume.silhouetteIndexCount);
assert.equal(reusedMovingVolume.capIndexCount, freshMovingVolume.capIndexCount);

// Structural references and group ranges are deliberately conservative cache
// seams. A producer replacing any of them gets a fresh topology map even when
// the new values happen to be numerically identical.
const assertShadowTopologyInvalidated = mutate => {
  const before = movingEntry.shadowTopologies;
  if (!before.size) before.set('0:6', D.prepareShadowTopology(movingEntry, movingEntry.groups));
  mutate();
  movingShadowGeometry.version++;
  movingShadowCache.get(movingShadowGeometry);
  assert.notEqual(movingEntry.shadowTopologies, before);
};
assertShadowTopologyInvalidated(() => {
  movingShadowGeometry.indices = movingShadowGeometry.indices.slice();
});
assertShadowTopologyInvalidated(() => {
  movingShadowGeometry.shadowVertexMap = movingShadowGeometry.shadowVertexMap.slice();
});
assertShadowTopologyInvalidated(() => {
  movingShadowGeometry.shadowTriangleMask = movingShadowGeometry.shadowTriangleMask.slice();
});
assertShadowTopologyInvalidated(() => {
  movingShadowGeometry.groups = [{ start: 0, count: 3 }, { start: 3, count: 3 }];
});
movingShadowCache.dispose();

// Renderer-owned shadow scratch reuses its large typed storage sequentially;
// the exported helper remains isolated when no scratch object is supplied.
const scratchTopology = D.prepareShadowTopology(shadowGeometry);
const isolatedShadowA = D.buildShadowVolume(
  shadowGeometry, scratchTopology, [0.5, 0.5, -1], identity, true);
const isolatedShadowB = D.buildShadowVolume(
  shadowGeometry, scratchTopology, [0.5, 0.5, 1], identity, true);
assert.notEqual(isolatedShadowA.faceFront.buffer, isolatedShadowB.faceFront.buffer);
assert.notEqual(isolatedShadowA.indices.buffer, isolatedShadowB.indices.buffer);
const shadowScratch = {};
const scratchShadowA = D.buildShadowVolume(
  shadowGeometry, scratchTopology, [0.5, 0.5, -1], identity, true, shadowScratch);
const scratchLightBuffer = shadowScratch.lightPosition.buffer;
const scratchFaceBuffer = scratchShadowA.faceFront.buffer;
const scratchIndexBuffer = scratchShadowA.indices.buffer;
const scratchShadowB = D.buildShadowVolume(
  shadowGeometry, scratchTopology, [0.5, 0.5, 1], identity, true, shadowScratch);
assert.equal(shadowScratch.lightPosition.buffer, scratchLightBuffer);
assert.equal(scratchShadowB.faceFront.buffer, scratchFaceBuffer);
assert.equal(scratchShadowB.indices.buffer, scratchIndexBuffer);
assert.deepEqual(Array.from(scratchShadowB.faceFront), Array.from(isolatedShadowB.faceFront));
assert.deepEqual(Array.from(scratchShadowB.indices), Array.from(isolatedShadowB.indices));
assert.ok(shadowScratch.indices.length >=
  scratchTopology.edges.length * 6 + scratchTopology.faces.length);

const uploadedShadowLights = [];
const scratchDrawGL = {
  DEPTH_TEST: 1, LESS: 2, BLEND: 3, CULL_FACE: 4, POLYGON_OFFSET_FILL: 5,
  CW: 6, STENCIL_TEST: 7, ALWAYS: 8, FRONT: 9, BACK: 10, KEEP: 11,
  DECR_WRAP: 12, INCR_WRAP: 13, ARRAY_BUFFER: 14, DYNAMIC_DRAW: 15,
  FLOAT: 16, UNSIGNED_BYTE: 17, ELEMENT_ARRAY_BUFFER: 18, TRIANGLES: 19,
  UNSIGNED_SHORT: 20, UNSIGNED_INT: 21,
  useProgram() {}, bindVertexArray() {}, uniformMatrix4fv() {},
  uniform3fv(_location, value) { uploadedShadowLights.push(Array.from(value)); },
  enable() {}, disable() {}, depthFunc() {}, depthMask() {}, colorMask() {},
  frontFace() {}, stencilMask() {}, stencilFunc() {}, stencilOpSeparate() {},
  bindBuffer() {}, bufferData() {}, enableVertexAttribArray() {},
  vertexAttribPointer() {}, drawElements() {},
};
const rendererShadowScratch = {};
const scratchDrawRenderer = {
  gl: scratchDrawGL, shadowUniforms: {}, shadowProgram: {}, shadowVAO: {},
  shadowPositionBuffer: {}, shadowExtrusionBuffer: {}, shadowIndexBuffer: {},
  shadowVolumeScratch: rendererShadowScratch, drawCalls: 0, triangles: 0,
  instanceMatrices: () => new Float32Array(identity),
};
const scratchDrawGeometry = {
  ...shadowGeometry, shadowTopologies: new Map([['0:6', scratchTopology]]),
};
D.Renderer.prototype.drawShadowItem.call(scratchDrawRenderer, {
  light: { position: [0.5, 0.5, -1] }, groups: shadowGeometry.groups,
  geometry: scratchDrawGeometry, job: {},
}, { viewProjection: identity });
assert.equal(rendererShadowScratch.lightPosition.BYTES_PER_ELEMENT,
  Float32Array.BYTES_PER_ELEMENT);
const rendererLightBuffer = rendererShadowScratch.lightPosition.buffer;
D.Renderer.prototype.drawShadowItem.call(scratchDrawRenderer, {
  light: { position: [0.5, 0.5, 1] }, groups: shadowGeometry.groups,
  geometry: scratchDrawGeometry, job: {},
}, { viewProjection: identity });
assert.equal(rendererShadowScratch.lightPosition.buffer, rendererLightBuffer);

// Shadow topology vertices and extrusion flags are identical for every
// instance in one paint job. Upload and configure them once, while preserving
// the per-instance light/model constants and silhouette index streams.
{
  const shadowPositionBuffer = { name: 'positions' };
  const shadowExtrusionBuffer = { name: 'extrusions' };
  const shadowIndexBuffer = { name: 'indices' };
  const uploads = [], pointers = [], enabledAttributes = [];
  const shadowDraws = [], lightUploads = [], matrixUploads = [];
  let boundBuffer = null;
  const shadowUploadGL = {
    ...scratchDrawGL,
    bindBuffer(_target, buffer) { boundBuffer = buffer; },
    bufferData(target, value) {
      uploads.push({ buffer: boundBuffer, target, value, length: value.length });
    },
    enableVertexAttribArray(location) { enabledAttributes.push(location); },
    vertexAttribPointer(location, size, type, normalized, stride, offset) {
      pointers.push({ location, size, type, normalized, stride, offset });
    },
    uniform3fv(location, value) {
      lightUploads.push({ location, value: Array.from(value) });
    },
    uniformMatrix4fv(location, transpose, value) {
      matrixUploads.push({ location, transpose, value: Array.from(value) });
    },
    drawElements(mode, count, type, offset) {
      shadowDraws.push({ mode, count, type, offset });
    },
  };
  const repeatedMatrices = new Float32Array(3 * 16);
  for (let instance = 0; instance < 3; instance++) {
    repeatedMatrices.set(identity, instance * 16);
  }
  const multiShadowRenderer = {
    gl: shadowUploadGL,
    shadowUniforms: { uViewProjection: 'viewProjection', uLightPosition: 'light', uModel: 'model' },
    shadowProgram: {}, shadowVAO: {}, shadowPositionBuffer,
    shadowExtrusionBuffer, shadowIndexBuffer, shadowVolumeScratch: {},
    drawCalls: 0, triangles: 0, instanceMatrices: () => repeatedMatrices,
  };
  D.Renderer.prototype.drawShadowItem.call(multiShadowRenderer, {
    light: { position: [0.5, 0.5, -1] }, groups: shadowGeometry.groups,
    geometry: scratchDrawGeometry, job: {},
  }, { viewProjection: identity });
  assert.equal(uploads.length, 5);
  assert.equal(uploads.filter(upload => upload.buffer === shadowPositionBuffer).length, 1);
  assert.equal(uploads.filter(upload => upload.buffer === shadowExtrusionBuffer).length, 1);
  assert.equal(uploads.filter(upload => upload.buffer === shadowIndexBuffer).length, 3);
  assert.equal(uploads.find(upload => upload.buffer === shadowPositionBuffer).value,
    scratchTopology.volumePositions);
  assert.equal(uploads.find(upload => upload.buffer === shadowExtrusionBuffer).value,
    scratchTopology.extrusions);
  assert.deepEqual(enabledAttributes, [0, 1]);
  assert.deepEqual(pointers.map(pointer => pointer.location), [0, 1]);
  assert.equal(lightUploads.length, 3);
  assert.equal(matrixUploads.filter(upload => upload.location === 'model').length, 3);
  assert.deepEqual(shadowDraws.map(draw => draw.count), [30, 30, 30]);
  assert.equal(multiShadowRenderer.drawCalls, 3);
  assert.equal(multiShadowRenderer.triangles, 30);

  // A capless volume viewed from its fully front-facing side emits no
  // silhouette indices. The lazy topology setup must remain untouched when
  // every instance is empty.
  uploads.length = 0; pointers.length = 0; enabledAttributes.length = 0;
  shadowDraws.length = 0; lightUploads.length = 0; matrixUploads.length = 0;
  const emptyShadowRenderer = {
    ...multiShadowRenderer, shadowVolumeScratch: {}, drawCalls: 0, triangles: 0,
  };
  D.Renderer.prototype.drawShadowItem.call(emptyShadowRenderer, {
    light: { position: [0.5, 0.5, 1] }, groups: shadowGeometry.groups,
    geometry: scratchDrawGeometry, job: {}, shadowZFail: false,
  }, { viewProjection: identity });
  assert.equal(uploads.length, 0);
  assert.deepEqual(enabledAttributes, []);
  assert.deepEqual(pointers, []);
  assert.equal(lightUploads.length, 0);
  assert.equal(matrixUploads.filter(upload => upload.location === 'model').length, 0);
  assert.deepEqual(shadowDraws, []);
  assert.equal(emptyShadowRenderer.drawCalls, 0);
  assert.equal(emptyShadowRenderer.triangles, 0);
}

// Engine_::RenderPaintJobs transforms a shadow light with the released
// transpose-only sMatrix::TransR, and material11.vsh subtracts that c4 value
// in object space before Model. A mathematically correct world-space
// extrusion is observably different for authored non-uniform model scale.
assert.ok(D.shadowVertexSource.includes(
  'vec4 modelPosition = vec4(aPosition, 1.0) -\n    vec4(uLightPosition, 1.0) * aExtrude;'));
assert.ok(D.shadowVertexSource.includes(
  'gl_Position = uViewProjection * (uModel * modelPosition);'));
const scaledShadowGeometry = {
  positions: new Float32Array([0, 0, 0, 1, -1, 0, 1, 0, -1]),
  indices: new Uint16Array([0, 1, 2]), groups: [{ start: 0, count: 3 }],
};
const scaledShadowModel = new Float32Array(identity);
scaledShadowModel[0] = 2; scaledShadowModel[5] = 3; scaledShadowModel[10] = 4;
const scaledShadowLight = [1, -1, 0];
const scaledShadowVolume = D.buildShadowVolume(
  scaledShadowGeometry, scaledShadowGeometry.groups, scaledShadowLight,
  scaledShadowModel, true,
);
assert.deepEqual(Array.from(scaledShadowVolume.lightPosition), [2, -3, 0]);
assert.deepEqual(Array.from(scaledShadowVolume.faceFront), [0],
  'shadow plane tests consume the native TransR object-space light');
scratchDrawRenderer.instanceMatrices = () => scaledShadowModel;
D.Renderer.prototype.drawShadowItem.call(scratchDrawRenderer, {
  light: { position: scaledShadowLight }, groups: scaledShadowGeometry.groups,
  geometry: { ...scaledShadowGeometry, shadowTopologies: new Map() }, job: {},
}, { viewProjection: identity });
assert.deepEqual(uploadedShadowLights.at(-1), [2, -3, 0],
  'the shadow shader receives native c4 rather than the world-space light');

// Incidence alone cannot prove a closed shadow volume: two manifold faces
// that traverse their shared edge in the same direction reveal a local
// winding inversion. Three incident faces remain the separate nonmanifold
// condition.
const conflictedShadowGeometry = {
  positions: new Float32Array([
    0, 0, 0, 1, 0, 0, 0, 1, 0, 0, -1, 0,
  ]),
  indices: new Uint16Array([0, 1, 2, 0, 1, 3]),
  groups: [{ start: 0, count: 6 }],
};
const conflictedTopology = D.prepareShadowTopology(conflictedShadowGeometry);
assert.equal(conflictedTopology.boundaryEdges, 4);
assert.equal(conflictedTopology.nonManifoldEdges, 0);
assert.equal(conflictedTopology.windingConflictEdges, 1);
const nonManifoldShadowGeometry = {
  positions: new Float32Array([
    0, 0, 0, 1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1,
  ]),
  indices: new Uint16Array([0, 1, 2, 1, 0, 3, 0, 1, 4]),
  groups: [{ start: 0, count: 9 }],
};
const nonManifoldTopology = D.prepareShadowTopology(nonManifoldShadowGeometry);
assert.equal(nonManifoldTopology.nonManifoldEdges, 1);
assert.equal(nonManifoldTopology.maxEdgeIncidence, 3);

// GeometryCache retains only scalar diagnostic summaries/offenders. The
// transient triangle-key Map is not attached to an entry.
const diagnosticCache = new D.GeometryCache({}, null, { diagnostics: true });
diagnosticCache.allEntries.add({
  source: { kind: 'diagnostic-geometry' }, sourceIds: new Set([77]),
  vertexCount: 4, indices: conflictedShadowGeometry.indices, uploadBytes: 0,
  diagnostics: D.geometryTopologyStats({
    ...conflictedShadowGeometry,
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, -1]),
  }),
  shadowTopologies: new Map([['conflict', conflictedTopology]]),
});
const diagnosticResources = diagnosticCache.resourceStats().diagnostics;
assert.equal(diagnosticResources.triangleCount, 2);
assert.equal(diagnosticResources.auditedTriangles, 2);
assert.equal(diagnosticResources.unauditedTriangles, 0);
assert.equal(diagnosticResources.shadowWindingConflictEdges, 1);
assert.equal(diagnosticResources.offenderCount, 1);
assert.equal(diagnosticResources.offenders[0].sourceIds[0], 77);
assert.equal(diagnosticResources.offenders[0].shadowWindingConflictEdges, 1);
assert.equal('triangleKeys' in diagnosticCache.allEntries.values().next().value.diagnostics, false);

// Coverage must be numerical, not just a count of truncated entries. Without
// the skipped-triangle total a clean audited prefix can be mistaken for a
// complete topology result on a large production mesh.
const truncatedDiagnosticCache = new D.GeometryCache({}, null, { diagnostics: true });
truncatedDiagnosticCache.allEntries.add({
  source: { kind: 'truncated-diagnostic-geometry' }, sourceIds: new Set([78]),
  vertexCount: 3, indices: new Uint16Array(6), uploadBytes: 0,
  diagnostics: D.geometryTopologyStats({
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    indices: new Uint16Array(6),
  }, { maxTriangles: 1 }),
  shadowTopologies: new Map(),
});
const truncatedDiagnosticResources = truncatedDiagnosticCache.resourceStats().diagnostics;
assert.equal(truncatedDiagnosticResources.triangleCount, 2);
assert.equal(truncatedDiagnosticResources.auditedTriangles, 1);
assert.equal(truncatedDiagnosticResources.unauditedTriangles, 1);
assert.equal(truncatedDiagnosticResources.truncatedEntries, 1);

const ribbon = D.buildChainRibbon({
  points: [
    new Float32Array([0, 0, 0]),
    new Float32Array([1, 0, 0]),
    new Float32Array([2, 0, 0]),
  ],
  matrix: D.mat4Identity(), thickness: 2, ripped: 1,
}, { view: D.mat4Identity() });
assert.deepEqual(Array.from(ribbon.positions), [
  0, 1, 0, 0, -1, 0,
  1, 1, 0, 1, -1, 0,
  2, 1, 0, 2, -1, 0,
]);
assert.deepEqual(Array.from(ribbon.uvs), [0, 0, 0, 1, 1, 0, 1, 1, 2, 0, 2, 1]);
assert.deepEqual(Array.from(ribbon.indices), [3, 2, 0, 3, 0, 1]);

// Cross(d,viewAxis) is intentionally not normalized by the native effect.
// An oblique cable therefore narrows by sin(view angle). The nonidentity
// deferred parent must have no influence because marker endpoints are already
// world-space.
const obliqueView = D.mat4Identity();
const diagonal = Math.fround(Math.SQRT1_2);
obliqueView[0] = diagonal; obliqueView[2] = diagonal;
obliqueView[8] = -diagonal; obliqueView[10] = diagonal;
const ignoredChainParent = D.mat4Identity();
ignoredChainParent[0] = 9; ignoredChainParent[5] = 7; ignoredChainParent[10] = 5;
ignoredChainParent[12] = 100;
const obliqueRibbon = D.buildChainRibbon({
  points: [new Float32Array([0, 0, 0]), new Float32Array([0, 0, 1])],
  matrix: ignoredChainParent, thickness: 2, ripped: -1,
}, { view: obliqueView });
assert.deepEqual(Array.from(obliqueRibbon.positions, value => Number(value.toFixed(6))), [
  0, -0.707107, 0, 0, 0.707107, 0,
  0, -0.707107, 1, 0, 0.707107, 1,
]);

console.log('renderer geometry and material planning tests passed');
