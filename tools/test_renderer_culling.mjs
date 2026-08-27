import assert from 'node:assert/strict';
import * as core from '../src/core.js';
import * as gl from '../src/gl.js';
import * as renderer from '../src/renderer.js';

const D = { ...core, ...gl, ...renderer };

const missingBounds = D.normalizePreparedGeometry({
  positions: new Float32Array([-2, 3, 8, 4, -5, 1, 0, 2, -3]),
  indices: new Uint16Array([0, 1, 2]),
});
assert.deepEqual(Array.from(missingBounds.bounds.minimum), [-2, -5, -3]);
assert.deepEqual(Array.from(missingBounds.bounds.maximum), [4, 3, 8]);

// Animated prepared positions are already skinned; bind-pose bounds cannot
// safely represent them.
const animatedBounds = D.normalizePreparedGeometry({
  prepare({ time }) {
    return {
      positions: new Float32Array([time, -2, 3, time + 2, 4, 8, time + 1, 0, 5]),
      indices: new Uint16Array([0, 1, 2]),
      bounds: { minimum: [-100, -100, -100], maximum: [100, 100, 100] },
    };
  },
}, { time: 10 });
assert.deepEqual(Array.from(animatedBounds.bounds.minimum), [10, -2, 3]);
assert.deepEqual(Array.from(animatedBounds.bounds.maximum), [12, 4, 8]);

const geometry = {
  bounds: { minimum: new Float32Array([-1, -1, -1]), maximum: new Float32Array([1, 1, 1]) },
};
const translate = (x, y, z) => {
  const matrix = D.mat4Identity();
  matrix[12] = x; matrix[13] = y; matrix[14] = z;
  return matrix;
};
const scaled = D.mat4Identity();
scaled[0] = 2; scaled[5] = 3; scaled[10] = -4;
scaled[12] = 10; scaled[13] = 20; scaled[14] = 30;
let world = D.meshJobWorldBounds(geometry, { matrix: scaled });
assert.deepEqual(Array.from(world.minimum), [8, 17, 26]);
assert.deepEqual(Array.from(world.maximum), [12, 23, 34]);
assert.equal(D.boundsIntersectsSphere(world, [15, 20, 30], 3), true);
assert.equal(D.boundsIntersectsSphere(world, [16, 20, 30], 3), false);

const composed = D.composeInstanceMatrices({ matrix: scaled, instances: [translate(1, 2, 3)] });
assert.deepEqual(Array.from(composed), Array.from(D.mat4Mul(scaled, translate(1, 2, 3))));
world = D.meshJobWorldBounds(geometry, {
  matrix: D.mat4Identity(), instances: [translate(10, 0, 0), translate(20, 0, 0)],
});
assert.deepEqual(Array.from(world.minimum), [9, -1, -1]);
assert.deepEqual(Array.from(world.maximum), [21, 1, 1]);
assert.equal(D.lightIntersectsWorldBounds({
  kind: 'point', position: [0, 0, 0], range: 7,
}, world), false, 'an all-outside instance batch is rejected');
world = D.meshJobWorldBounds(geometry, {
  matrix: D.mat4Identity(), instances: [translate(0, 0, 0), translate(20, 0, 0)],
});
assert.equal(D.lightIntersectsWorldBounds({
  kind: 'point', position: [0, 0, 0], range: 0.5,
}, world), true, 'one inside instance conservatively retains the batch');
assert.equal(D.lightIntersectsWorldBounds({
  kind: 'directional', position: [1e6, 0, 0], range: 1,
}, world), true, 'directional lights bypass point-sphere rejection');

const projective = D.mat4Identity(); projective[3] = 0.25;
assert.equal(D.meshJobWorldBounds(geometry, { matrix: projective }), null,
  'unknown projective model bounds disable culling');
assert.equal(D.lightIntersectsWorldBounds({
  kind: 'point', position: [100, 0, 0], range: 1,
}, null), true);

const camera = {
  cameraSpace: D.mat4Identity(), nearClip: 1, farClip: 100,
  zoomX: 1, zoomY: 1, centerX: 0, centerY: 0,
};
const matrices = D.cameraMatrices(camera);
const viewportPlanes = D.viewFrustumPlanes(matrices.viewProjection);
assert.equal(viewportPlanes.length, 5, 'the infinite-far camera exposes four side planes and near');
assert.equal(D.lightIntersectsViewFrustum({
  kind: 'point', position: [6, 0, 5], range: 1,
}, viewportPlanes), true, 'a sphere crossing a side plane remains visible');
assert.equal(D.lightIntersectsViewFrustum({
  kind: 'point', position: [7, 0, 5], range: 1,
}, viewportPlanes), false, 'a sphere wholly beyond a side plane is rejected');
assert.equal(D.lightIntersectsViewFrustum({
  kind: 'point', position: [0, 0, 0], range: 1,
}, viewportPlanes), true, 'a sphere tangent to the near plane remains visible');
assert.equal(D.lightIntersectsViewFrustum({
  kind: 'point', position: [0, 0, -1], range: 1,
}, viewportPlanes), false, 'a sphere wholly behind the near plane is rejected');
assert.equal(D.lightIntersectsViewFrustum({
  kind: 'directional', position: [1e6, 1e6, -1e6], range: 1,
}, viewportPlanes), true, 'semantic directional lights affect the entire viewport');

const visibleLowImportance = {
  opId: 900, kind: 'point', position: [0, 0, 10], range: 1, amplify: 1,
};
const offscreenHighImportance = Array.from({ length: 16 }, (_, index) => ({
  opId: 901 + index, kind: 'point', position: [100 + index, 0, 10], range: 20, amplify: 1,
}));
const selectedAfterViewportCull = D.selectActiveLights({
  camera,
  lightJobs: [...offscreenHighImportance, visibleLowImportance],
}, 1);
assert.deepEqual(selectedAfterViewportCull.map(light => light.opId), [900],
  'offscreen high-importance lights cannot evict a visible light from the capped list');
const selectedWithPreparedViewportCull = D.selectActiveLights({
  camera,
  lightJobs: [...offscreenHighImportance, visibleLowImportance],
}, 1, viewportPlanes);
assert.deepEqual(selectedWithPreparedViewportCull.map(light => light.opId),
  selectedAfterViewportCull.map(light => light.opId),
  'reusing prepared viewport planes preserves light selection and culling');

const sideCaster = D.meshJobWorldBounds(geometry, { matrix: translate(20, 0, 3) });
assert.equal(D.shadowCasterMayAffectView({
  kind: 'point', position: [0, 0, 3], range: 30,
}, sideCaster, matrices), false, 'side caster outside the enlarged five-plane frustum is rejected');
assert.equal(D.shadowCasterMayAffectView({
  kind: 'directional', position: [1e6, 0, 0], range: 1,
}, sideCaster, matrices), true, 'directional shadow-frustum rejection stays disabled');

// Engine::CalcSphereBounds and sFrustum::ZFailVolume jointly choose the
// stencil algorithm per caster. A centered bbox crossing the near-volume
// needs caps/z-fail; a lateral bbox outside one plane can use capless z-pass.
const zFailLight = { kind: 'point', flags: 2, position: [0, 0, 5], range: 1 };
const lightRectangle = D.lightSphereBounds(zFailLight, matrices.view,
  camera.zoomX, camera.zoomY);
assert.deepEqual(Array.from(lightRectangle, value => Number(value.toFixed(6))),
  [-0.204124, -0.204124, 0.204124, 0.204124]);
const zFailPlanes = D.shadowZFailVolumePlanes(camera, zFailLight, matrices);
assert.equal(zFailPlanes.length, 6);
assert.equal(D.shadowCasterUsesZFail({
  minimum: [-0.1, -0.1, 0.5], maximum: [0.1, 0.1, 1.5],
}, zFailPlanes), true);
assert.equal(D.shadowCasterUsesZFail({
  minimum: [10, -1, 2], maximum: [11, 1, 3],
}, zFailPlanes), false);
assert.equal(D.shadowCasterUsesZFail(null, zFailPlanes), true,
  'unknown/animated/instanced bounds conservatively retain native z-fail');
const nearPlaneLight = { kind: 'point', position: [0, 0, 1], range: 1 };
assert.equal(D.shadowZFailVolumePlanes(camera, nearPlaneLight, matrices,
  new Float64Array([-1, -1, 1, 1])).length, 2,
  'a light on znear takes the source two-plane epsilon slab');
const behindNearLight = { kind: 'point', position: [0, 0, 0], range: 2 };
const flippedZFailPlanes = D.shadowZFailVolumePlanes(camera, behindNearLight, matrices);
assert.equal(flippedZFailPlanes.length, 6);
assert.ok(flippedZFailPlanes[0][2] < 0,
  'a light behind znear flips the five source volume planes inward');

const material = {
  kind: 'material', system: '1.1', parameters: [], textures: [],
  passes: [
    { usage: 'base', renderPass: 0 },
    { usage: 'shadow', renderPass: 0 },
    { usage: 'light', renderPass: 0 },
  ],
};
const renderGeometry = {
  ...geometry, groups: [{ material, start: 0, count: 3 }], materials: [],
};
const job = { opId: 50, matrix: translate(0, 0, 3), passAdjust: 0 };
const lights = [
  { opId: 51, kind: 'point', flags: 2, position: [0, 0, 3], range: 4 },
  { opId: 52, kind: 'point', flags: 2, position: [100, 0, 3], range: 1 },
  { opId: 53, kind: 'directional', flags: 2, position: [1e6, 0, 0], range: 1 },
];
const items = [], stats = {};
D.appendMeshRenderItems(items, job, renderGeometry, lights, matrices,
  D.composeInstanceMatrices(job), stats);
assert.equal(items.length, 5, 'culling reduces the fake viewport planner from seven items to five');
assert.equal(items.every(item => Number.isSafeInteger(item.sortMaterial)), true,
  'every mesh paint item carries a stable material-pass sort identity');
for (const pass of material.passes) {
  const passItems = items.filter(item => item.pass === pass);
  if (passItems.length > 1) {
    assert.equal(new Set(passItems.map(item => item.sortMaterial)).size, 1,
      'all duplicated light/shadow jobs for one pass share its sort identity');
  }
}
assert.deepEqual(items.map(item => item.pass.usage).sort(),
  ['base', 'light', 'light', 'shadow', 'shadow']);
assert.deepEqual(stats, {
  candidateViewItems: 2,
  candidateLightItems: 3, culledLightItems: 1,
  candidateShadowItems: 3, culledShadowSphereItems: 1,
});

const stencilChoiceLight = {
  opId: 58, kind: 'point', flags: 2, position: [0, 0, 5], range: 30,
};
const stencilChoicePlanes = [D.shadowZFailVolumePlanes(camera, stencilChoiceLight, matrices)];
for (const [x, expected] of [[0, true], [10, false]]) {
  const choiceJob = { opId: 59 + x, matrix: translate(x, 0, 3), passAdjust: 0 };
  const choiceItems = [];
  D.appendMeshRenderItems(choiceItems, choiceJob, renderGeometry,
    [stencilChoiceLight], matrices, D.composeInstanceMatrices(choiceJob), null,
    [null], stencilChoicePlanes);
  const shadowItem = choiceItems.find(item => item.pass.usage === 'shadow');
  assert.equal(shadowItem?.shadowZFail, expected,
    `caster at x=${x} receives the source stencil mode in its paint item`);
}

// Native BuildPaintJobs assigns animated meshes an enormous bbox. The
// animation can move vertices outside bind-pose/current-frame bounds, so every
// light and shadow candidate must survive the spatial culling stage.
const animatedJob = {
  opId: 56, matrix: translate(0, 0, 3), passAdjust: 0,
  mesh: { hasAnimation: () => true },
};
const animatedItems = [], animatedStats = {};
D.appendMeshRenderItems(animatedItems, animatedJob, renderGeometry, lights, matrices,
  D.composeInstanceMatrices(animatedJob), animatedStats);
assert.deepEqual(animatedItems.map(item => item.pass.usage).sort(),
  ['base', 'light', 'light', 'light', 'shadow', 'shadow', 'shadow'],
  'animated meshes conservatively retain every light and shadow job');
assert.deepEqual(animatedStats, {
  candidateViewItems: 2, candidateLightItems: 3, candidateShadowItems: 3,
});

// Scene_Particles' batched path is rendered with native MPP_INSTANCES. Native
// BuildPaintJobs gives that program the same enormous bbox as animated meshes,
// even when every authored instance lies outside a point light's sphere.
const instanceMaterial = {
  kind: 'material', system: '1.1', parameters: [], textures: [],
  passes: [
    { usage: 'base', renderPass: 0, program: 'instances' },
    { usage: 'light', renderPass: 0, program: 'instances' },
  ],
};
const instanceGeometry = {
  ...geometry, groups: [{ material: instanceMaterial, start: 0, count: 3 }], materials: [],
};
const instanceJob = {
  opId: 57, matrix: D.mat4Identity(), passAdjust: 0,
  instances: [translate(100, 0, 3)],
};
const instanceItems = [], instanceStats = {};
D.appendMeshRenderItems(instanceItems, instanceJob, instanceGeometry, lights, matrices,
  D.composeInstanceMatrices(instanceJob), instanceStats);
assert.deepEqual(instanceItems.map(item => item.pass.usage).sort(),
  ['base', 'light', 'light', 'light'],
  'native MPP_INSTANCES conservatively retains every per-light job');
assert.deepEqual(instanceStats, { candidateViewItems: 2, candidateLightItems: 3 });

const sideJob = { opId: 54, matrix: translate(20, 0, 3), passAdjust: 0 };
const sideItems = [], sideStats = {};
D.appendMeshRenderItems(sideItems, sideJob, renderGeometry, [{
  opId: 55, kind: 'point', flags: 2, position: [0, 0, 3], range: 30,
}], matrices, D.composeInstanceMatrices(sideJob), sideStats);
assert.deepEqual(sideItems.map(item => item.pass.usage), [],
  'ordinary base/light paint jobs outside the native five-plane frustum are omitted');
assert.deepEqual(sideStats, {
  candidateViewItems: 2, culledViewItems: 2,
  candidateShadowItems: 1, culledShadowFrustumItems: 1,
});

// Camera culling never suppresses a shadow solely because the caster itself
// is offscreen. A directional source can still project that caster into view.
const offscreenShadowItems = [], offscreenShadowStats = {};
D.appendMeshRenderItems(offscreenShadowItems, sideJob, renderGeometry, [{
  opId: 56, kind: 'directional', flags: 2,
  position: [1e6, 0, 0], range: 1,
}], matrices, D.composeInstanceMatrices(sideJob), offscreenShadowStats);
assert.deepEqual(offscreenShadowItems.map(item => item.pass.usage), ['shadow']);
assert.deepEqual(offscreenShadowStats, {
  candidateViewItems: 2, culledViewItems: 2, candidateShadowItems: 1,
});

// Retain boxes touching a plane and unknown projective boxes. Both are
// conservative safeguards against dropping visible geometry at float edges.
const tangentBounds = {
  minimum: new Float32Array([3, 0, 3]),
  maximum: new Float32Array([3, 0, 3]),
};
assert.equal(D.boundsOutsidePlanes(tangentBounds, viewportPlanes), false);
const tangentItems = [];
D.appendMeshRenderItems(tangentItems,
  { opId: 57, matrix: D.mat4Identity(), passAdjust: 0 },
  { ...renderGeometry, bounds: tangentBounds }, [], matrices);
assert.deepEqual(tangentItems.map(item => item.pass.usage), ['base']);

const unknownItems = [];
D.appendMeshRenderItems(unknownItems,
  { opId: 58, matrix: projective, passAdjust: 0 }, renderGeometry, [], matrices);
assert.deepEqual(unknownItems.map(item => item.pass.usage), ['base'],
  'unknown projective bounds disable camera culling');

// Ordinary native jobs use per-material/cluster boxes rather than the union
// for the complete mesh. An offscreen cluster must not keep its own pass alive
// merely because a different cluster is visible.
const baseOnly = {
  kind: 'material', system: '1.1', parameters: [], textures: [],
  passes: [{ usage: 'base', renderPass: 0 }],
};
const splitGeometry = D.normalizePreparedGeometry({
  positions: new Float32Array([
    -0.5, -0.5, 3, 0.5, -0.5, 3, 0, 0.5, 3,
    19.5, -0.5, 3, 20.5, -0.5, 3, 20, 0.5, 3,
  ]),
  indices: new Uint16Array([0, 1, 2, 3, 4, 5]),
  groups: [
    { material: baseOnly, start: 0, count: 3 },
    { material: baseOnly, start: 3, count: 3 },
  ],
});
const splitItems = [], splitStats = {};
D.appendMeshRenderItems(splitItems,
  { opId: 59, matrix: D.mat4Identity(), passAdjust: 0 },
  splitGeometry, [], matrices, null, splitStats);
assert.deepEqual(splitItems.map(item => item.group.start), [0]);
assert.deepEqual(splitStats, { candidateViewItems: 2, culledViewItems: 1 });

console.log('renderer light and shadow culling tests passed');
