import assert from 'node:assert/strict';
import * as CoreAPI from '../src/core.js';
import * as SceneAPI from '../src/scene.js';

const handlers = new Map(Object.entries(SceneAPI.sceneHandlers)
  .map(([id, handler]) => [Number(id), handler]));
const D = { ...CoreAPI, ...SceneAPI, handlers };

function environment() {
  const instances = new Map();
  return {
    vars: Array.from({ length: 32 }, () => new Float32Array(4)),
    matrixStack: new D.MatrixStack(),
    getInstance(op, factory) {
      let value = instances.get(op);
      if (!value) {
        value = factory();
        value.reset = true;
        instances.set(op, value);
      } else {
        value.reset = false;
      }
      return value;
    },
    instanceFor(op) { return instances.get(op); },
    restoreInstance(op, value) { instances.set(op, value); },
  };
}

function cloneSnapshotState(value) {
  if (value === null || value === undefined || typeof value !== 'object') return value;
  if (ArrayBuffer.isView(value)) return new value.constructor(value);
  if (Array.isArray(value)) return value.map(cloneSnapshotState);
  const result = {};
  for (const [key, item] of Object.entries(value)) result[key] = cloneSnapshotState(item);
  return result;
}

function staticOp(id, kind) {
  return {
    id,
    cache: { kind },
    classInfo: { outputClass: kind === 'mesh' ? 'KC_MESH' : 'KC_SCENE' },
    exec() {},
  };
}

function callFor(classId, parameters, inputs, env, op = {}) {
  op.id ??= classId;
  op.inputs ??= inputs.map((_, index) => staticOp(index, 'mesh'));
  op.execInputs ??= current => op.inputs.forEach(input => input.exec(current));
  const call = {
    runtime: null,
    environment: env,
    op,
    inputs,
    links: [],
    parameters,
    strings: [],
    splines: [],
  };
  return call;
}

const env = environment();
env.vars[0].fill(3.5);
const mesh = { kind: 'mesh', name: 'test' };

const sceneHandler = D.handlers.get(0xc0);
const sceneCall = callFor(0xc0, [1, 1, 1, 0, 0, 0, 5, 6, 7, 0], [mesh], env);
sceneCall.op.cache = sceneHandler.init(sceneCall);
sceneHandler.exec(sceneCall);
assert.equal(env.frame.meshJobs.length, 1);
assert.deepEqual(Array.from(env.frame.meshJobs[0].matrix.subarray(12, 15)), [5, 6, 7]);
assert.equal(env.frame.meshJobs[0].time, 3.5);

D.beginRenderFrame(env);
const multiplyHandler = D.handlers.get(0xc2);
const multiplyCall = callFor(0xc2, [1, 1, 1, 0, 0, 0, 2, 0, 0, 3], [mesh], env);
multiplyCall.op.cache = multiplyHandler.init(multiplyCall);
env.vars[3].fill(99);
multiplyHandler.exec(multiplyCall);
assert.deepEqual(env.frame.meshJobs.map(job => job.matrix[12]), [0, 2, 4]);
assert.deepEqual(Array.from(env.vars[3]), [99, 99, 99, 99]);
assert.equal(env.matrixStack.stack.length, 1);

D.beginRenderFrame(env);
const lightHandler = D.handlers.get(0xc4);
const lightCall = callFor(
  0xc4,
  [0, 0, 0, 10, 20, 30, 4, 0, 0xff804020, 2, 0],
  [],
  env,
);
lightHandler.exec(lightCall);
assert.equal(env.frame.lightJobs[0].kind, 'directional');
assert.deepEqual(Array.from(env.frame.lightJobs[0].direction), [0, 0, 1]);
assert.deepEqual(Array.from(env.frame.lightJobs[0].position), [0, 0, 1e6]);

const ambientHandler = D.handlers.get(0x184);
ambientHandler.exec(callFor(0x184, [0x00f01020], [], env));
ambientHandler.exec(callFor(0x184, [0x0040f0f0], [], env));
assert.equal(env.frame.ambientLight, 0xffffff);

D.beginRenderFrame(env);
assert.equal(env.frame.ambientLight, 0,
  'Engine::StartFrame ownership resets Scene_Ambient before the next viewport');
const particleHandler = D.handlers.get(0xc5);
const particleParameters = new Array(30).fill(0);
particleParameters[0] = 0x130;
particleParameters[1] = 3;
particleParameters[2] = 2;
particleParameters[9] = 0.25;
particleParameters[12] = 0.1;
particleParameters[13] = 1;
const particleCall = callFor(0xc5, particleParameters, [mesh], env);
particleCall.op.cache = particleHandler.init(particleCall);
particleHandler.exec(particleCall);
assert.equal(env.frame.meshJobs.length, 1);
assert.equal(env.frame.meshJobs[0].instances.length, 3);
const translations = env.frame.meshJobs[0].instances.map(matrix => matrix[12]);
assert.ok(Math.abs(translations[0] - 0.1) < 1e-6);
assert.ok(Math.abs(translations[1] - (0.1 + 1 / 3)) < 1e-6);
assert.ok(Math.abs(translations[2] - (0.1 + 2 / 3)) < 1e-6);

// Particle traversal only needs one position/rotation workspace. Instanced
// particles additionally retain one output matrix per particle because the
// renderer consumes all of them after traversal has completed.
const firstInstances = env.frame.meshJobs[0].instances;
const firstMatrices = firstInstances.slice();
const particleMemory = env.instanceFor(particleCall.op);
assert.ok(particleMemory.pos instanceof Float32Array);
assert.ok(particleMemory.speed instanceof Float32Array);
assert.ok(particleMemory.rot instanceof Float32Array);
assert.ok(particleMemory.rotSpeed instanceof Float32Array);
assert.equal(particleMemory.pos.length, particleParameters[1] * 4);
assert.equal(particleMemory.speed.length, particleParameters[1] * 4);
assert.equal(particleMemory.rot.length, particleParameters[1] * 3);
assert.equal(particleMemory.rotSpeed.length, particleParameters[1] * 3);
const scratchDescriptor = Object.getOwnPropertyDescriptor(particleMemory, '_particleScratch');
assert.ok(scratchDescriptor);
assert.equal(scratchDescriptor.enumerable, false,
  'derivable particle scratch is excluded from runtime snapshots');
const firstScratch = scratchDescriptor.value;
assert.equal(Object.hasOwn(firstScratch, 'particles'), false,
  'particle scratch has no per-particle wrapper objects');
assert.ok(firstScratch.position instanceof Float32Array);
assert.equal(firstScratch.position.length, 3);
assert.equal(firstScratch.matrix, null,
  'instanced traversal does not allocate a redundant shared output matrix');
assert.ok(firstScratch.rotation instanceof Float32Array);
assert.equal(firstScratch.rotation.length, 16);
assert.equal(firstScratch.matrices.length, particleParameters[1]);
assert.deepEqual(firstScratch.matrices, firstMatrices);
assert.ok(firstScratch.matrices.every(matrix => matrix instanceof Float32Array));
const firstPosition = firstScratch.position;
const firstRotation = firstScratch.rotation;

particleParameters[12] = 0.35;
D.beginRenderFrame(env);
particleHandler.exec(particleCall);
const secondInstances = env.frame.meshJobs[0].instances;
const secondScratch = particleMemory._particleScratch;
assert.equal(secondInstances, firstInstances, 'the instances array is reused across frames');
for (let index = 0; index < secondInstances.length; index++) {
  assert.equal(secondInstances[index], firstMatrices[index],
    `particle ${index} reuses its output matrix`);
}
assert.equal(secondScratch, firstScratch, 'later frames reuse the compact particle scratch');
assert.equal(secondScratch.position, firstPosition,
  'all instanced particles and later frames share one position workspace');
assert.equal(secondScratch.rotation, firstRotation,
  'all instanced particles and later frames share one rotation workspace');
for (let index = 0; index < secondInstances.length; index++) {
  const stateOffset = index * 4;
  let fraction = particleParameters[12] + particleMemory.pos[stateOffset + 3] +
    particleMemory.randForw * particleMemory.speed[stateOffset + 3] * particleParameters[12];
  while (fraction >= 1) fraction -= 1;
  const translated = D.mat4Identity();
  translated[12] += fraction;
  const legacyMatrix = D.mat4MulA(
    D.mat4Euler(fraction * particleParameters[9], 0, 0), translated,
  );
  assert.deepEqual(Array.from(secondInstances[index]), Array.from(legacyMatrix),
    `particle ${index} retains the released allocation path's Float32 matrix`);
}

// A fresh calculation is the numerical oracle: scratch reuse must not alter a
// single Float32 result or the final authored multiplication order.
const oracleEnvironment = environment();
const oracleCall = callFor(0xc5, particleParameters.slice(), [mesh], oracleEnvironment);
oracleCall.op.cache = particleHandler.init(oracleCall);
particleHandler.exec(oracleCall);
const oracleMemory = oracleEnvironment.instanceFor(oracleCall.op);
assert.deepEqual(Array.from(particleMemory.pos), Array.from(oracleMemory.pos));
assert.deepEqual(Array.from(particleMemory.speed), Array.from(oracleMemory.speed));
assert.deepEqual(Array.from(particleMemory.rot), Array.from(oracleMemory.rot));
assert.deepEqual(Array.from(particleMemory.rotSpeed), Array.from(oracleMemory.rotSpeed));
assert.deepEqual(
  secondInstances.map(matrix => Array.from(matrix)),
  oracleEnvironment.frame.meshJobs[0].instances.map(matrix => Array.from(matrix)),
  'reused particle matrices remain bit-identical to a fresh calculation',
);

// Non-instanced traversal may reuse a single matrix while walking particles,
// but each deferred mesh job must own a stable copy. The equivalent instanced
// path is an oracle for the authored transforms because bit 0x100 only selects
// how those transforms are submitted.
{
  const traversalEnvironment = environment();
  const traversalParameters = particleParameters.slice();
  traversalParameters[0] &= ~0x100;
  const traversalCall = callFor(0xc5, traversalParameters, [mesh], traversalEnvironment);
  traversalCall.op.cache = particleHandler.init(traversalCall);
  particleHandler.exec(traversalCall);

  const traversalMemory = traversalEnvironment.instanceFor(traversalCall.op);
  const traversalScratch = traversalMemory._particleScratch;
  const traversalJobs = traversalEnvironment.frame.meshJobs;
  const oracleMatrices = oracleEnvironment.frame.meshJobs[0].instances;
  assert.equal(Object.hasOwn(traversalScratch, 'particles'), false);
  assert.ok(traversalScratch.position instanceof Float32Array);
  assert.ok(traversalScratch.matrix instanceof Float32Array);
  assert.ok(traversalScratch.rotation instanceof Float32Array);
  assert.equal(traversalScratch.matrices, null,
    'non-instanced traversal has no persistent per-particle output matrices');
  assert.equal(traversalJobs.length, oracleMatrices.length);
  assert.equal(new Set(traversalJobs.map(job => job.matrix)).size, traversalJobs.length,
    'each non-instanced deferred job owns an independent matrix');
  for (let index = 0; index < traversalJobs.length; index++) {
    assert.notEqual(traversalJobs[index].matrix, traversalScratch.matrix,
      `particle ${index} job does not retain the shared traversal matrix`);
    assert.deepEqual(Array.from(traversalJobs[index].matrix), Array.from(oracleMatrices[index]),
      `particle ${index} non-instanced transform is bit-identical to the oracle`);
  }

  const firstTraversalJobs = traversalJobs.map(job => job.matrix);
  traversalParameters[12] = 0.47;
  D.beginRenderFrame(traversalEnvironment);
  particleHandler.exec(traversalCall);
  assert.equal(traversalMemory._particleScratch, traversalScratch,
    'non-instanced frames reuse the compact traversal scratch');
  assert.ok(traversalEnvironment.frame.meshJobs.every((job, index) =>
    job.matrix !== firstTraversalJobs[index]),
  'later non-instanced jobs remain independent of earlier frame output');
}

// Runtime cloneState walks enumerable properties. Mimic snapshot/restore here:
// the cache must be absent from the clone and reconstructed on the next frame.
const snapshotMemory = cloneSnapshotState(particleMemory);
assert.equal(Object.hasOwn(snapshotMemory, '_particleScratch'), false);
assert.notEqual(snapshotMemory.pos, particleMemory.pos);
assert.notEqual(snapshotMemory.speed, particleMemory.speed);
assert.notEqual(snapshotMemory.rot, particleMemory.rot);
assert.notEqual(snapshotMemory.rotSpeed, particleMemory.rotSpeed);
assert.deepEqual(Array.from(snapshotMemory.pos), Array.from(particleMemory.pos));
assert.deepEqual(Array.from(snapshotMemory.speed), Array.from(particleMemory.speed));
assert.deepEqual(Array.from(snapshotMemory.rot), Array.from(particleMemory.rot));
assert.deepEqual(Array.from(snapshotMemory.rotSpeed), Array.from(particleMemory.rotSpeed));

particleParameters[12] = 0.6;
D.beginRenderFrame(env);
particleHandler.exec(particleCall);
const expectedRestoredMatrices = env.frame.meshJobs[0].instances
  .map(matrix => Array.from(matrix));

env.restoreInstance(particleCall.op, snapshotMemory);
D.beginRenderFrame(env);
particleHandler.exec(particleCall);
const restoredMemory = env.instanceFor(particleCall.op);
const restoredDescriptor = Object.getOwnPropertyDescriptor(restoredMemory, '_particleScratch');
assert.ok(restoredDescriptor, 'particle scratch is recreated after snapshot restore');
assert.equal(restoredDescriptor.enumerable, false);
assert.notEqual(restoredDescriptor.value, firstScratch);
assert.equal(Object.hasOwn(restoredDescriptor.value, 'particles'), false);
assert.ok(restoredDescriptor.value.position instanceof Float32Array);
assert.equal(restoredDescriptor.value.matrix, null);
assert.equal(restoredDescriptor.value.matrices.length, particleParameters[1]);
assert.notEqual(env.frame.meshJobs[0].instances, firstInstances,
  'restored state receives a fresh derivable instances cache');
assert.deepEqual(
  env.frame.meshJobs[0].instances.map(matrix => Array.from(matrix)),
  expectedRestoredMatrices,
  'snapshot restoration reproduces bit-identical particle matrices',
);

const restoredInstances = env.frame.meshJobs[0].instances;
D.beginRenderFrame(env);
particleParameters[12] = 0.7;
particleHandler.exec(particleCall);
assert.equal(env.frame.meshJobs[0].instances, restoredInstances,
  'the recovered instances cache is reused by later frames');

// The authored random state remains stable while only animation inputs change.
// Any parameter that authors that state must replace all packed buffers so a
// snapshot cannot retain stale particle data.
{
  const packedEnvironment = environment();
  const packedParameters = new Array(30).fill(0);
  packedParameters[0] = 0x110;
  packedParameters[1] = 5;
  packedParameters[2] = 0x1234;
  packedParameters[3] = 2;
  packedParameters[4] = 3;
  packedParameters[5] = 4;
  packedParameters[6] = 0.2;
  packedParameters[7] = 0.3;
  packedParameters[8] = 0.4;
  packedParameters[9] = 0.15;
  packedParameters[10] = 0.1;
  packedParameters[11] = 0.05;
  packedParameters[12] = 0.35;
  packedParameters[13] = 1;
  packedParameters[16] = 0.5;
  packedParameters[22] = 0.25;
  packedParameters[23] = 0.1;
  packedParameters[24] = 0.02;
  packedParameters[25] = 0.03;
  packedParameters[26] = 0.04;
  const packedCall = callFor(0xc5, packedParameters, [mesh], packedEnvironment);
  packedCall.op.cache = particleHandler.init(packedCall);
  particleHandler.exec(packedCall);
  const packedMemory = packedEnvironment.instanceFor(packedCall.op);
  const initialBuffers = {
    pos: packedMemory.pos,
    speed: packedMemory.speed,
    rot: packedMemory.rot,
    rotSpeed: packedMemory.rotSpeed,
  };

  const twinEnvironment = environment();
  const twinCall = callFor(0xc5, packedParameters.slice(), [mesh], twinEnvironment);
  twinCall.op.cache = particleHandler.init(twinCall);
  particleHandler.exec(twinCall);
  assert.deepEqual(
    packedEnvironment.frame.meshJobs[0].instances.map(matrix => Array.from(matrix)),
    twinEnvironment.frame.meshJobs[0].instances.map(matrix => Array.from(matrix)),
    'independently initialized packed particle calls produce bit-identical matrices',
  );

  packedParameters[12] = 0.55;
  D.beginRenderFrame(packedEnvironment);
  particleHandler.exec(packedCall);
  assert.equal(packedMemory.pos, initialBuffers.pos);
  assert.equal(packedMemory.speed, initialBuffers.speed);
  assert.equal(packedMemory.rot, initialBuffers.rot);
  assert.equal(packedMemory.rotSpeed, initialBuffers.rotSpeed);

  packedParameters[2]++;
  D.beginRenderFrame(packedEnvironment);
  particleHandler.exec(packedCall);
  assert.notEqual(packedMemory.pos, initialBuffers.pos);
  assert.notEqual(packedMemory.speed, initialBuffers.speed);
  assert.notEqual(packedMemory.rot, initialBuffers.rot);
  assert.notEqual(packedMemory.rotSpeed, initialBuffers.rotSpeed);
}

// Generated BlobSpline paths can write directly into each particle's cached
// output matrix. Their evaluation scratch belongs to the particle operator,
// and is shared by every particle and later frame.
{
  const splineCalls = [];
  const splineScratches = [];
  const generatedSpline = {
    createEvalScratch() {
      const scratch = { marker: splineScratches.length };
      splineScratches.push(scratch);
      return scratch;
    },
    evalInto(time, phase, matrix, scratch) {
      splineCalls.push({ time, phase, matrix, scratch });
      D.mat4Identity(matrix);
      matrix[12] = time * 10;
      return { matrix, zoom: 1 };
    },
    eval() {
      assert.fail('particles should prefer the allocation-reuse spline API');
    },
  };
  const splineEnvironment = environment();
  const splineParameters = new Array(30).fill(0);
  splineParameters[0] = 0x100;
  splineParameters[1] = 3;
  splineParameters[2] = 7;
  splineParameters[12] = 0.15;
  const splineCall = callFor(0xc5, splineParameters, [mesh, generatedSpline], splineEnvironment);
  splineCall.op.cache = particleHandler.init(splineCall);

  particleHandler.exec(splineCall);
  const firstSplineInstances = splineEnvironment.frame.meshJobs[0].instances;
  assert.equal(splineScratches.length, 1, 'one generated-spline scratch is created per particle operator');
  assert.equal(splineCalls.length, firstSplineInstances.length);
  for (let index = 0; index < splineCalls.length; index++) {
    assert.equal(splineCalls[index].matrix, firstSplineInstances[index],
      `particle ${index} supplies its cached matrix as the spline output`);
    assert.equal(splineCalls[index].scratch, splineScratches[0],
      `particle ${index} reuses the generated-spline scratch`);
  }

  const firstCallCount = splineCalls.length;
  splineParameters[12] = 0.45;
  D.beginRenderFrame(splineEnvironment);
  particleHandler.exec(splineCall);
  const secondSplineInstances = splineEnvironment.frame.meshJobs[0].instances;
  assert.equal(splineScratches.length, 1, 'later frames reuse the generated-spline scratch');
  assert.equal(secondSplineInstances, firstSplineInstances);
  for (let index = 0; index < secondSplineInstances.length; index++) {
    const call = splineCalls[firstCallCount + index];
    assert.equal(call.matrix, firstSplineInstances[index],
      `particle ${index} keeps passing its cached output matrix on later frames`);
    assert.equal(call.scratch, splineScratches[0],
      `particle ${index} keeps reusing the same generated-spline scratch`);
  }
}

// The three generic scene modes are present in the released player even
// though the Debris production has no operator instances for them.
{
  D.beginRenderFrame(env);
  const adjustedMeshOp = staticOp(70, 'mesh');
  const adjustOp = {
    id: 71, inputs: [adjustedMeshOp], cache: null,
    execInput(current, index) {
      D.execSceneInput({ environment: current, op: this }, index);
    },
  };
  const adjustCall = callFor(0x180, [5], [mesh], env, adjustOp);
  const adjust = D.handlers.get(0x180);
  adjustOp.cache = adjust.init(adjustCall);
  env.renderPassAdjust = 2;
  adjust.exec(adjustCall);
  assert.equal(env.frame.meshJobs[0].passAdjust, 7);
  assert.equal(env.renderPassAdjust, 2, 'AdjustPass restores its enclosing adjustment');

  D.beginRenderFrame(env);
  const splineMatrix = D.mat4Identity(); splineMatrix[12] = 9;
  const generatedSpline = { eval: () => ({ matrix: splineMatrix, zoom: 1 }) };
  const applyOp = {
    id: 72, inputs: [staticOp(73, 'mesh'), staticOp(74, 'scene')], cache: null,
  };
  const applyCall = callFor(0x181, [0.25], [mesh, generatedSpline], env, applyOp);
  const apply = D.handlers.get(0x181);
  applyOp.cache = apply.init(applyCall);
  apply.exec(applyCall);
  assert.equal(env.frame.meshJobs[0].matrix[12], 9);
  assert.equal(env.matrixStack.stack.length, 1);

  const lod = D.handlers.get(0x183);
  const highOp = staticOp(75, 'mesh'), lowOp = staticOp(76, 'mesh');
  const lodOp = { id: 77, inputs: [highOp, lowOp], cache: null };
  const lodCall = callFor(0x183, [5], [mesh, mesh], env, lodOp);
  lodOp.cache = lod.init(lodCall);
  env.currentCamera = { cameraSpace: D.mat4Identity() };
  const nearModel = D.mat4Identity(); nearModel[14] = 3;
  env.matrixStack.push(nearModel);
  D.beginRenderFrame(env);
  lod.exec(lodCall);
  assert.equal(env.frame.meshJobs[0].opId, highOp.id);
  env.matrixStack.pop();
  const farModel = D.mat4Identity(); farModel[14] = 8;
  env.matrixStack.push(farModel);
  D.beginRenderFrame(env);
  lod.exec(lodCall);
  assert.equal(env.frame.meshJobs[0].opId, lowOp.id);
  env.matrixStack.pop();

  delete env.currentCamera;
  env.matrixStack.push(farModel);
  D.beginRenderFrame(env);
  lod.exec(lodCall);
  assert.equal(env.frame.meshJobs[0].opId, lowOp.id,
    'LOD uses the neutral native camera when invoked before a viewport');
  env.matrixStack.pop();
}

console.log('scene traversal, lighting, and particles tests passed');
