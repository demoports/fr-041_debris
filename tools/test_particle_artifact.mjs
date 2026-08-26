import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

import { parseKX } from '../src/kx.js';
import {
  Mesh_Cube,
  Mesh_TransformEx,
  MESH_DEFAULT_VERTEX_MASK,
  MESH_FEATURE,
} from '../src/mesh.js';
import { meshToMin } from '../src/minmesh.js';
import { createOperatorHandlers } from '../src/operators.js';
import {
  appendMeshRenderItems,
  composeInstanceMatrices,
  material11LightConstant,
  materialView,
  normalizePreparedGeometry,
} from '../src/renderer.js';
import { Runtime } from '../src/runtime.js';
import { beginRenderFrame } from '../src/scene.js';

const bytes = await readFile(new URL('../assets/debris_party.kx', import.meta.url));
const document = parseKX(bytes);

// Event 51 (zero-based, beats 360..384) is the looping-highway shot shown by
// the artifact report. Event 92 is the preceding beats-336..360 shot; the KX
// event array is grouped by authored page rather than sorted chronologically.
// Keep this fixture tied to the exact released highway graph path:
// old Cube -> TransEx -> Mesh_ToMin -> MatLink(M11) -> Scene_Particles.
const event = document.events[51];
assert.equal(event.operation, 12267);
assert.deepEqual([event.start / 65536, event.end / 65536], [360, 384]);
assert.equal(document.events[92].operation, 14526);
assert.deepEqual(
  [document.events[92].start / 65536, document.events[92].end / 65536],
  [336, 360],
);

const material = document.operations[12236];
assert.equal(material.classId, 0xd0);
assert.equal(material.parameters[48] >>> 0, 0x0803,
  'the particle material uses the instanced Material11 multipass path');
assert.equal(material.parameters[53] >>> 0, 0xff102028,
  'the authored multipass ambient is dark blue-grey, not white');
const nativeWarmLightConstant = material11LightConstant(
  material.parameters[52], 0xffffe090, 1.75);
assert.deepEqual(Array.from(nativeWarmLightConstant).slice(0, 2), [1, 1],
  'PS1.1 clips the amplified highway light constant before diffuse shading');
assert.ok(nativeWarmLightConstant[2] > 0.71 && nativeWarmLightConstant[2] < 0.72,
  'the unsaturated blue channel retains the authored warm-light balance');

const systems = [
  { particle: 12239, spline: 12238, matLink: 12237, toMin: 12232, transform: 12231, cube: 12230, count: 1000, seed: 2, startAnim: 0, startCount: 0 },
  { particle: 12247, spline: 12246, matLink: 12245, toMin: 12244, transform: 12243, cube: 12242, count: 1000, seed: 2, startAnim: 0.2, startCount: 227 },
  { particle: 12255, spline: 12254, matLink: 12253, toMin: 12252, transform: 12251, cube: 12250, count: 750, seed: 1, startAnim: 0.125, startCount: 106 },
];

const reachable = new Set();
const pending = [event.operation];
while (pending.length) {
  const id = pending.pop();
  if (reachable.has(id)) continue;
  reachable.add(id);
  pending.push(...document.operations[id].inputs);
}

for (const ids of systems) {
  const particle = document.operations[ids.particle];
  const spline = document.operations[ids.spline];
  const matLink = document.operations[ids.matLink];
  const toMin = document.operations[ids.toMin];
  const transform = document.operations[ids.transform];
  const cube = document.operations[ids.cube];

  assert.ok(reachable.has(particle.id),
    `particle ${particle.id} must remain below highway root ${event.operation}`);
  assert.equal(particle.classId, 0xc5);
  assert.equal(particle.inputs[0], matLink.id);
  assert.equal(particle.inputs[1], spline.id);
  assert.deepEqual(particle.parameters.slice(0, 3), [0x154, ids.count, ids.seed]);
  assert.equal(spline.classId, 0x18);
  assert.equal(matLink.classId, 0x110);
  assert.equal(matLink.inputs[0], toMin.id);
  assert.equal(matLink.links[0], material.id);
  assert.equal(toMin.classId, 0xb5);
  assert.equal(toMin.inputs[0], transform.id);
  assert.equal(transform.classId, 0x89);
  assert.equal(transform.inputs[0], cube.id);
  assert.equal(cube.classId, 0x81);
}

// sGMI_COLOR0 is attribute 3 and sGMF_COLOR0 is bit 3. The intro build's
// sGMF_DEFAULT therefore has a present COLOR0 channel. GenMesh::Init clears
// that channel to zero, and native Mesh_ToMin packs that zero with GetColor().
assert.equal(MESH_FEATURE.COLOR0, 1 << 3);
assert.equal(MESH_DEFAULT_VERTEX_MASK, 0x2f);
assert.ok(MESH_DEFAULT_VERTEX_MASK & MESH_FEATURE.COLOR0);

const convertedSystems = systems.map(ids => {
  const cubeOperation = document.operations[ids.cube];
  const transformOperation = document.operations[ids.transform];
  const oldCube = Mesh_Cube(
    cubeOperation.parameters[0],
    cubeOperation.parameters[1],
    cubeOperation.parameters[2],
    cubeOperation.parameters[3],
    cubeOperation.parameters.slice(4, 13),
  );
  const transformed = Mesh_TransformEx(
    oldCube,
    transformOperation.parameters[0],
    transformOperation.parameters.slice(1, 10),
    transformOperation.parameters[10],
    transformOperation.parameters[11],
  );
  assert.ok(transformed.vertexMask & MESH_FEATURE.COLOR0);
  assert.ok(transformed.vertices.every(vertex =>
    vertex.color[0] === 0 && vertex.color[1] === 0 &&
    vertex.color[2] === 0 && vertex.color[3] === 0));

  const converted = meshToMin(transformed);
  assert.ok(converted.vertices.every(vertex => vertex.color === 0x00000000),
    `highway particle ${ids.particle} must preserve its present zero COLOR0`);
  assert.ok(converted.prepare().colors.every(component => component === 0),
    `particle ${ids.particle} renderer bytes must preserve the zero channel`);
  return converted;
});

// sVector::GetColor packs z | y<<8 | x<<16 | w<<24 after truncating each
// float component to an unsigned byte. Exercise a non-zero value as a direct
// executable oracle for the native packing used by Mesh_ToMin.
const colorProbe = Mesh_Cube(1, 1, 1, 0, [1, 1, 1, 0, 0, 0, 0, 0, 0]);
for (const vertex of colorProbe.vertices) vertex.color.set([1, 0.5, 0.25, 0.75]);
assert.ok(meshToMin(colorProbe).vertices.every(vertex => vertex.color === 0xbfff7f3f),
  'present COLOR0 must use native sVector::GetColor channel order and truncation');

const cubeWithoutColor = Mesh_Cube(1, 1, 1, 0, [1, 1, 1, 0, 0, 0, 0, 0, 0]);
cubeWithoutColor.vertexMask &= ~MESH_FEATURE.COLOR0;
assert.ok(meshToMin(cubeWithoutColor).vertices.every(vertex => vertex.color === 0xffffffff),
  'white is the native fallback only when COLOR0 is absent from the format');

const baseView = materialView({
  kind: 'material',
  system: '1.1',
  parameters: material.parameters,
  textures: [],
}, { usage: 'base' });
assert.equal(baseView.vertexColorMode, 4,
  'generated Material11 base keeps the native Color0 SET + vertex ADD program');
assert.equal(baseView.generatedBaseAlpha, 'zero',
  'the generated base keeps its source AlphaCombiner at ZERO');
assert.deepEqual(baseView.textureMap, [null, null, null, null],
  'the non-alpha-tested highway base does not sample a texture');
assert.equal(baseView.color >>> 0, 0xff102028);

// With the correct zero vertex color the base is the authored dark blue-grey.
// Substituting the absent-channel white fallback saturates the same ADD phase,
// which is the exact pure-white particle symptom from the highway screenshot.
const addRgb = (packedBase, packedVertex) => [16, 8, 0].map(shift =>
  Math.min(255, ((packedBase >>> shift) & 255) + ((packedVertex >>> shift) & 255)));
assert.deepEqual(addRgb(baseView.color, convertedSystems[0].vertices[0].color), [16, 32, 40]);
assert.deepEqual(addRgb(baseView.color, 0xffffffff), [255, 255, 255]);

// Exercise the real operator chain as well as the isolated conversion above.
// This catches a later renderer normalization regression that might replace a
// valid all-zero packed channel with its missing-channel white default.
const runtime = new Runtime(document, {
  handlers: createOperatorHandlers(),
  strictHandlers: true,
});
const environment = runtime.environment;
environment.exitFrame();
environment.initFrame(0, 0);
try {
  for (const ids of systems) runtime.operations[ids.particle].precalc(environment);
} finally {
  environment.exitFrame();
}

// The event animation drives parameter 12 to 0, 0.2 and 0.125 at the first
// frame. Mode 0x40 suppresses particles until their unwrapped fraction reaches
// one, so op12239 intentionally has no instances there. Native PaintJob exits
// for InstanceCount==0; the GPU path must preserve that empty cardinality.
for (const ids of systems) {
  environment.initFrame(event.start, 0);
  try {
    beginRenderFrame(environment);
    runtime.events[51].instances.clear();
    const particleOp = runtime.operations[ids.particle];
    particleOp.execEvent(environment, runtime.events[51]);
    assert.ok(Math.abs(particleOp.animParameters[12] - ids.startAnim) < 1e-7,
      `particle ${ids.particle} evaluates its authored start animation`);
    assert.equal(environment.frame.meshJobs[0].instances.length, ids.startCount,
      `particle ${ids.particle} preserves its native first-frame instance count`);
    if (ids.startCount === 0) {
      assert.equal(composeInstanceMatrices(environment.frame.meshJobs[0]).length, 0,
        'zero native instances do not fall back to the paint-job base matrix');
    }
  } finally {
    environment.exitFrame();
  }
}

// Four beats into the shot has a non-empty come-out set while keeping this
// fixture independent of the audio device/sample rate.
for (const ids of systems) {
  environment.initFrame(event.start + 4 * 65536, 0);
  try {
    beginRenderFrame(environment);
    runtime.events[51].instances.clear();
    runtime.operations[ids.particle].execEvent(environment, runtime.events[51]);
    assert.equal(environment.frame.meshJobs.length, 1);
    const job = environment.frame.meshJobs[0];
    assert.ok(job.instances.length > 0,
      `highway particle ${ids.particle} emits visible instances`);

    const geometry = normalizePreparedGeometry(job.mesh);
    assert.equal(geometry.vertexCount, 24);
    assert.ok(geometry.colors instanceof Uint8Array);
    assert.ok(geometry.colors.every(component => component === 0),
      `particle ${ids.particle} reaches the renderer with zero COLOR0`);

    const liveMaterial = geometry.groups[0].material;
    assert.equal(liveMaterial.parameters[48] >>> 0, 0x0803);
    const liveBase = materialView(liveMaterial, liveMaterial.passes[0]);
    const liveLight = materialView(liveMaterial, liveMaterial.passes[1]);
    assert.equal(liveBase.vertexColorMode, 4,
      'the live generated base retains its dedicated RGB-add/alpha-zero shader path');
    assert.equal(liveBase.generatedBaseAlpha, 'zero');
    assert.equal(liveBase.color >>> 0, 0xff102028);
    assert.equal(liveLight.vertexColorMode, 0,
      'the additive light phase neither substitutes nor consumes vertex COLOR0');

    // Native BuildPaintJobs emits the base once and duplicates only the light
    // phase per selected light. Instanced jobs deliberately skip sphere culling.
    const lights = [
      { kind: 'point', position: [-96, 100, -128], range: 256, amplify: 1.75 },
      { kind: 'point', position: [48, -100, 148], range: 320, amplify: 0.5 },
    ];
    const renderItems = [];
    const matrices = composeInstanceMatrices(job);
    appendMeshRenderItems(renderItems, job, geometry, lights, {}, matrices);
    assert.deepEqual(renderItems.map(item => item.pass.usage), ['base', 'light', 'light']);
    assert.deepEqual(renderItems.filter(item => item.pass.usage === 'light')
      .map(item => item.light), lights);
  } finally {
    environment.exitFrame();
  }
}

console.log('highway event 51 particle artifact regression passed');
