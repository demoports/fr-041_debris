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

function float32Bits(values) {
  return Array.from(new Uint32Array(
    values.buffer, values.byteOffset, values.byteLength / Uint32Array.BYTES_PER_ELEMENT,
  ));
}

function legacySceneMatrix(parameters, base = D.mat4Identity()) {
  const srt = new Float32Array([
    parameters[0] ?? 1, parameters[1] ?? 1, parameters[2] ?? 1,
    parameters[3] ?? 0, parameters[4] ?? 0, parameters[5] ?? 0,
    parameters[6] ?? 0, parameters[7] ?? 0, parameters[8] ?? 0,
  ]);
  return D.mat4Mul(base, D.mat4SRT(srt));
}

function transformScratchPool(op) {
  const symbol = Object.getOwnPropertySymbols(op)
    .find(value => value.description === 'sceneTransformScratch');
  assert.ok(symbol, 'scene transform scratch is private symbol state on its operator');
  const descriptor = Object.getOwnPropertyDescriptor(op, symbol);
  assert.equal(descriptor.enumerable, false);
  return descriptor.value;
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

// Scene SRT operators refill per-op, depth-indexed scratch on every execution.
// Fresh core helpers are the numerical oracle; compare raw Float32 bits so a
// changed rounding boundary, signed zero or animated-value reuse cannot hide.
{
  const transformEnvironment = environment();
  const transformParameters = [
    1.25, -0.75, 2.5,
    0.125, -0.375, 0.625,
    3.25, -4.5, 5.75, 0,
  ];
  const transformOp = {};
  const transformCall = callFor(0xc0, transformParameters, [mesh],
    transformEnvironment, transformOp);
  transformOp.cache = sceneHandler.init(transformCall);
  sceneHandler.exec(transformCall);
  const firstJobMatrix = transformEnvironment.frame.meshJobs[0].matrix;
  const firstJobBits = float32Bits(firstJobMatrix);
  assert.deepEqual(firstJobBits, float32Bits(legacySceneMatrix(transformParameters)),
    'Scene transform scratch is bit-identical to fresh srtFrom/mat4SRT output');

  const transformPool = transformScratchPool(transformOp);
  assert.equal(transformPool.depth, 0);
  assert.equal(transformPool.entries.length, 1);
  const transformScratch = transformPool.entries[0];
  assert.notEqual(firstJobMatrix, transformScratch.matrix,
    'a deferred mesh job does not retain the per-op transform scratch');

  transformParameters[0] = -1.5;
  transformParameters[3] = -0.2;
  transformParameters[6] = -8.125;
  D.beginRenderFrame(transformEnvironment);
  sceneHandler.exec(transformCall);
  assert.equal(transformPool.entries[0], transformScratch,
    'repeated Scene execution reuses its transform scratch');
  assert.deepEqual(float32Bits(transformEnvironment.frame.meshJobs[0].matrix),
    float32Bits(legacySceneMatrix(transformParameters)),
    'animated Scene parameters are recomputed on every execution');
  assert.deepEqual(float32Bits(firstJobMatrix), firstJobBits,
    'later scratch reuse cannot mutate an already retained frame job');

  const transform3Handler = D.handlers.get(0xc3);
  const transform3Parameters = [
    0.5, 1.5, 2, -0.125, 0.25, -0.5, 11, 12, 13, 4,
  ];
  const transform3Op = {};
  const transform3Call = callFor(0xc3, transform3Parameters, [mesh],
    transformEnvironment, transform3Op);
  transform3Op.cache = transform3Handler.init(transform3Call);
  D.beginRenderFrame(transformEnvironment);
  transform3Handler.exec(transform3Call);
  const transform3Pool = transformScratchPool(transform3Op);
  assert.notEqual(transform3Pool, transformPool,
    'separate scene operators never share mutable transform scratch');
  assert.notEqual(transform3Pool.entries[0].matrix, transformScratch.matrix);
  assert.deepEqual(float32Bits(transformEnvironment.frame.meshJobs[0].matrix),
    float32Bits(legacySceneMatrix(transform3Parameters)));
}

// Multiply reuses both its local SRT matrix and cumulative multiplication
// output, while every submitted job remains an independent snapshot.
{
  const multiplyEnvironment = environment();
  const multiplyParameters = [
    1.25, 0.625, -0.75, 0.0625, -0.1875, 0.3125, 2.5, -1.25, 0.75, 4,
  ];
  const multiplyOp = {};
  const scratchMultiplyCall = callFor(0xc2, multiplyParameters, [mesh],
    multiplyEnvironment, multiplyOp);
  multiplyOp.cache = multiplyHandler.init(scratchMultiplyCall);
  multiplyHandler.exec(scratchMultiplyCall);

  const multiplySRT = () => new Float32Array(multiplyParameters.slice(0, 9));
  const expectedMultiplyJobs = () => {
    const step = D.mat4SRT(multiplySRT());
    const result = [];
    let current = D.mat4Identity();
    for (let index = 0; index < (multiplyParameters[9] | 0); index++) {
      result.push(new Float32Array(current));
      current = D.mat4MulA(step, current);
    }
    return result;
  };
  assert.deepEqual(
    multiplyEnvironment.frame.meshJobs.map(job => float32Bits(job.matrix)),
    expectedMultiplyJobs().map(float32Bits),
    'Multiply scratch preserves every cumulative transform bit',
  );
  const retainedMultiplyJobs = multiplyEnvironment.frame.meshJobs.map(job => job.matrix);
  const retainedMultiplyBits = retainedMultiplyJobs.map(float32Bits);
  const multiplyPool = transformScratchPool(multiplyOp);
  const multiplyScratch = multiplyPool.entries[0];
  assert.ok(retainedMultiplyJobs.every(matrix =>
    matrix !== multiplyScratch.matrix && matrix !== multiplyScratch.next));

  multiplyParameters[3] = -0.35;
  multiplyParameters[6] = -3.75;
  D.beginRenderFrame(multiplyEnvironment);
  multiplyHandler.exec(scratchMultiplyCall);
  assert.equal(multiplyPool.entries[0], multiplyScratch,
    'later Multiply executions reuse their per-op scratch entry');
  assert.deepEqual(
    multiplyEnvironment.frame.meshJobs.map(job => float32Bits(job.matrix)),
    expectedMultiplyJobs().map(float32Bits),
    'Multiply recomputes animated parameters on every execution',
  );
  assert.deepEqual(retainedMultiplyJobs.map(float32Bits), retainedMultiplyBits,
    'Multiply scratch reuse does not alter prior deferred jobs');
}

// Multiply retains its step matrix across child traversal. A depth-indexed pool
// keeps an inner execution of the same op from overwriting the outer step.
{
  const nestedEnvironment = environment();
  const nestedParameters = [1, 1, 1, 0, 0, 0, 2, 0, 0, 2];
  let reentered = false;
  const seenTranslations = [];
  let nestedCall;
  const nestedChild = {
    id: 801,
    cache: { kind: 'scene' },
    classInfo: { outputClass: 'KC_SCENE' },
    exec(current) {
      seenTranslations.push(current.matrixStack.top[12]);
      if (reentered) return;
      reentered = true;
      const translation = nestedParameters[6], count = nestedParameters[9];
      nestedParameters[6] = 7;
      nestedParameters[9] = 1;
      multiplyHandler.exec(nestedCall);
      nestedParameters[6] = translation;
      nestedParameters[9] = count;
    },
  };
  const nestedOp = { inputs: [nestedChild] };
  nestedCall = callFor(0xc2, nestedParameters, [nestedChild.cache],
    nestedEnvironment, nestedOp);
  nestedOp.cache = multiplyHandler.init(nestedCall);
  multiplyHandler.exec(nestedCall);
  assert.deepEqual(seenTranslations, [0, 0, 2],
    'same-op reentrancy leaves the outer Multiply step matrix intact');
  const nestedPool = transformScratchPool(nestedOp);
  assert.equal(nestedPool.depth, 0);
  assert.equal(nestedPool.entries.length, 2,
    'one scratch entry is retained for each observed reentrancy depth');
  assert.equal(nestedEnvironment.matrixStack.depth, 1);
}

// Light traversal can share the same local SRT machinery because retained light
// jobs copy their direction and position rather than exposing the scratch matrix.
{
  const scratchLightEnvironment = environment();
  const scratchLightParameters = [
    0.125, -0.25, 0.375, 4, 5, 6, 0, 0, 0xff204080, 1.5, 8,
  ];
  const scratchLightOp = {};
  const scratchLightCall = callFor(0xc4, scratchLightParameters, [],
    scratchLightEnvironment, scratchLightOp);
  lightHandler.exec(scratchLightCall);
  const firstLight = scratchLightEnvironment.frame.lightJobs[0];
  const firstPositionBits = float32Bits(firstLight.position);
  const firstDirectionBits = float32Bits(firstLight.direction);
  const lightPool = transformScratchPool(scratchLightOp);
  const lightScratch = lightPool.entries[0];

  scratchLightParameters[0] = -0.4;
  scratchLightParameters[3] = 10;
  D.beginRenderFrame(scratchLightEnvironment);
  lightHandler.exec(scratchLightCall);
  assert.equal(lightPool.entries[0], lightScratch,
    'repeated Light execution reuses its private transform scratch');
  const lightSRT = new Float32Array([
    1, 1, 1,
    scratchLightParameters[0], scratchLightParameters[1], scratchLightParameters[2],
    scratchLightParameters[3], scratchLightParameters[4], scratchLightParameters[5],
  ]);
  const lightMatrix = D.mat4Mul(D.mat4Identity(), D.mat4SRT(lightSRT));
  assert.deepEqual(float32Bits(scratchLightEnvironment.frame.lightJobs[0].direction),
    float32Bits(new Float32Array(lightMatrix.subarray(8, 11))));
  assert.deepEqual(float32Bits(scratchLightEnvironment.frame.lightJobs[0].position),
    float32Bits(new Float32Array(lightMatrix.subarray(12, 15))));
  assert.deepEqual(float32Bits(firstLight.position), firstPositionBits);
  assert.deepEqual(float32Bits(firstLight.direction), firstDirectionBits,
    'later Light scratch reuse leaves retained light jobs unchanged');
}

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
