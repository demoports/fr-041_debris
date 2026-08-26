import assert from 'node:assert/strict';
import * as AppAPI from '../src/app.js';
import * as CoreAPI from '../src/core.js';
import * as EffectsAPI from '../src/effects.js';
import * as RuntimeAPI from '../src/runtime.js';
import * as SceneAPI from '../src/scene.js';

const handlers = new Map([
  ...Object.entries(SceneAPI.sceneHandlers),
  ...Object.entries(EffectsAPI.effectHandlers),
].map(([id, handler]) => [Number(id), handler]));
const D = { ...CoreAPI, ...RuntimeAPI, ...SceneAPI, ...EffectsAPI, ...AppAPI, handlers };
assert.deepEqual(Array.from(D.perlin3D([0, 0, 0])), [1, -1, 0]);
assert.deepEqual(
  Array.from(D.perlin3D([-0.25, 0.5, 8])),
  [-0.396484375, -0.5517578125, -0.0517578125],
);

const instances = new Map();
const env = {
  vars: Array.from({ length: 32 }, () => new Float32Array(4)),
  markers: Array.from({ length: 32 }, () => D.mat4Identity()),
  matrixStack: new D.MatrixStack(),
  timeSlices: 0,
  getInstance(op, factory) {
    let value = instances.get(op);
    if (!value) {
      value = factory();
      value.reset = true;
      instances.set(op, value);
    } else value.reset = false;
    return value;
  },
};
const material = { kind: 'material', passes: [{ usage: 'other', renderPass: 0 }] };
const parameters = new Array(21).fill(0);
parameters.splice(0, 6, 0, 10, 0, 10, 10, 0);
parameters[6] = 0;
parameters[7] = 1;
parameters[8] = 12;
parameters[9] = 0.1;
parameters[10] = 0.005;
parameters[11] = 0.99;
parameters[12] = 0.02;
parameters[13] = 48;
parameters[16] = 8;
parameters[20] = 16;
const op = { id: 100, cache: null };
const call = {
  environment: env,
  op,
  inputs: [],
  links: [material],
  parameters,
};
const chain = D.handlers.get(0x6b);
op.cache = chain.init(call);
// Marker positions are already world-space when the deferred effect runs.
// Keep a distinct nonidentity effect-job parent on the stack so renderer
// tests can prove that it is not applied to these positions a second time.
env.markers[0][12] = 3;
env.markers[1][14] = 5;
const chainParent = D.mat4Identity();
chainParent[0] = 2; chainParent[5] = 3; chainParent[10] = 4;
chainParent[12] = 100;
env.matrixStack.push(chainParent);
chain.exec(call);
env.matrixStack.pop();
assert.equal(env.frame.effectGeometry.length, 1);
const geometry = env.frame.effectGeometry[0];
assert.equal(geometry.kind, 'chain-line');
assert.equal(geometry.points.length, 32);
assert.deepEqual(Array.from(geometry.points[0]), [3, 10, 0]);
assert.deepEqual(Array.from(geometry.points[31]), [10, 10, 5]);
assert.deepEqual(Array.from(geometry.matrix), Array.from(D.mat4Identity()));
assert.ok(geometry.points[15][1] < 8);

const glareOp = { id: 101, cache: null };
const glareCall = { environment: env, op: glareOp, inputs: [], links: [], parameters: new Array(11).fill(0) };
const glare = D.handlers.get(0x74);
glareOp.cache = glare.init(glareCall);
glare.exec(glareCall);
assert.equal(env.frame.postJobs[0].kind, 'glare');

const colorOp = { id: 102, cache: null };
const colorCall = { environment: env, op: colorOp, inputs: [], links: [], parameters: new Array(11).fill(0) };
const color = D.handlers.get(0x75);
colorOp.cache = color.init(colorCall);
color.exec(colorCall);
assert.equal(colorOp.cache.needCurrentRender, true);
assert.equal(env.frame.postJobs[1].kind, 'color-correction');

const water = D.buildWaterGeometry({ blobSpline: { keys: [] } }, [8, 1, 2, 1, 1, 3], 0);
assert.equal(water.width, 8);
assert.equal(water.positions.length, 8 * 8 * 3);
assert.equal(water.indices.length, 7 * 7 * 6);
assert.ok(water.indices instanceof Uint16Array);
assert.ok(water.colors instanceof Uint8Array);
assert.equal(water.colors.length, 8 * 8 * 4);
assert.equal(water.tangents.length, 8 * 8 * 4);
assert.ok(Array.from(water.tangents).every(value => value === 0),
  'native Water explicitly submits zero tangent-space vectors');
assert.deepEqual(Array.from(water.positions.subarray(0, 3)), [-4, 0, -4]);
assert.deepEqual(Array.from(water.uvs.subarray(water.uvs.length - 2)), [2, 2]);
assert.deepEqual(Array.from(water.normals.subarray(0, 3)), [0, 1, 0]);

const waveSpline = { blobSpline: { keys: [{
  time: 0, px: 0, pz: 0, rx: 1, ry: 1, rz: 2, zoom: 1,
}] } };
const waved = D.buildWaterGeometry(waveSpline, [8, 1, 2, 1, 1, 3], 0.1, water);
assert.equal(waved, water);
const retainedWaveTable = waved._waterScratch.waves[0].values;
D.buildWaterGeometry(waveSpline, [8, 1, 2, 1, 1, 3], 0.2, water);
assert.equal(water._waterScratch.waves[0].values, retainedWaveTable);
assert.equal(waved.waveCount, 1);
assert.equal(waved.version, 3);
assert.ok(Array.from(waved.positions).some((value, index) => index % 3 === 1 && value !== 0));

// The maximum authored Water grid has exactly 65,536 vertices. Its largest
// valid index therefore still fits in Uint16, and repeated updates must reuse
// every sizeable buffer plus the private layout/wave scratch.
const maximumWaterParameters = [8, 1, 2, 1, 1, 8];
const maximumWater = D.buildWaterGeometry({ blobSpline: { keys: [] } }, maximumWaterParameters, 0);
assert.equal(maximumWater.width, 256);
assert.equal(maximumWater.positions.length / 3, 65_536);
assert.ok(maximumWater.indices instanceof Uint16Array);
let maximumWaterIndex = 0;
for (const index of maximumWater.indices) maximumWaterIndex = Math.max(maximumWaterIndex, index);
assert.equal(maximumWaterIndex, 65_535);

const retainedMaximumWater = {
  positions: maximumWater.positions,
  normals: maximumWater.normals,
  tangents: maximumWater.tangents,
  uvs: maximumWater.uvs,
  colors: maximumWater.colors,
  indices: maximumWater.indices,
  shadowVertexMap: maximumWater.shadowVertexMap,
  bounds: maximumWater.bounds,
  boundsMinimum: maximumWater.bounds.minimum,
  boundsMaximum: maximumWater.bounds.maximum,
  scratch: maximumWater._waterScratch,
  lastRow: maximumWater._waterScratch.lastRow,
  waves: maximumWater._waterScratch.waves,
};
assert.equal(
  D.buildWaterGeometry({ blobSpline: { keys: [] } }, maximumWaterParameters, 1, maximumWater),
  maximumWater,
);
for (const [name, identity] of Object.entries(retainedMaximumWater)) {
  const current = name === 'boundsMinimum' ? maximumWater.bounds.minimum
    : name === 'boundsMaximum' ? maximumWater.bounds.maximum
      : name === 'scratch' ? maximumWater._waterScratch
        : name === 'lastRow' ? maximumWater._waterScratch.lastRow
          : name === 'waves' ? maximumWater._waterScratch.waves
            : maximumWater[name];
  assert.equal(current, identity, `same-layout Water update retains ${name}`);
}
assert.deepEqual(maximumWater.dynamicAttributes, ['positions', 'normals']);

const oldFirstX = maximumWater.positions[0];
const oldLastU = maximumWater.uvs[maximumWater.uvs.length - 2];
const changedMaximumWaterParameters = [12, 1, 4, 1, 1, 8];
assert.equal(
  D.buildWaterGeometry({ blobSpline: { keys: [] } }, changedMaximumWaterParameters, 2, maximumWater),
  maximumWater,
);
assert.equal(maximumWater.positions, retainedMaximumWater.positions);
assert.equal(maximumWater.normals, retainedMaximumWater.normals);
assert.equal(maximumWater.uvs, retainedMaximumWater.uvs);
assert.equal(maximumWater.bounds, retainedMaximumWater.bounds);
assert.equal(maximumWater._waterScratch, retainedMaximumWater.scratch);
assert.deepEqual(maximumWater.dynamicAttributes, ['positions', 'normals', 'uvs']);
assert.notEqual(maximumWater.positions[0], oldFirstX);
assert.notEqual(maximumWater.uvs[maximumWater.uvs.length - 2], oldLastU);
assert.equal(maximumWater.positions[0], -6);
assert.equal(maximumWater.uvs[maximumWater.uvs.length - 2], 4);

// Deferred effects execute in the native paint order even when a seek drains
// a nested IPP tree without drawing it.
{
  const order = [];
  const environment = new D.Environment(null);
  const makeJob = (id, pass, usage) => {
    const effect = { pass, usage };
    const jobOp = {
      id, cache: effect, sceneInstances: null,
      exec: () => { order.push(id); },
    };
    return {
      op: jobOp, effect, matrix: D.mat4Identity(),
      variables: environment.vars.map(value => new Float32Array(value)),
    };
  };
  const viewport = {
    type: 'viewport',
    effectJobs: [
      makeJob(1, 2, 'base'),
      makeJob(2, 1, 'other'),
      makeJob(3, 1, 'ambient'),
    ],
  };
  environment.frameOutputs.push({ type: 'layer2d', input: viewport });
  assert.equal(D.advanceEffectFrame(environment), 3);
  assert.deepEqual(order, [3, 2, 1]);
  assert.deepEqual(environment.frame.effectGeometry, []);
  assert.deepEqual(environment.frame.postJobs, []);
}

// AddPaintJob prepends native effect jobs. Stable key sorting therefore keeps
// equal pass/usage jobs in reverse traversal order, including the seek-only
// path that advances state without rendering.
{
  const order = [];
  const environment = new D.Environment(null);
  const makeJob = (id, passAdjust = 0) => {
    const effect = { pass: 5, usage: 'other' };
    const jobOp = { id, cache: effect, sceneInstances: null, exec: () => order.push(id) };
    return {
      op: jobOp, effect, passAdjust, matrix: D.mat4Identity(),
      variables: environment.vars.map(value => new Float32Array(value)),
    };
  };
  environment.frameOutputs.push({
    type: 'viewport',
    effectJobs: [makeJob(1), makeJob(2), makeJob(3, -1)],
  });
  assert.deepEqual(D.nativeEffectJobOrder(environment.frameOutputs[0].effectJobs)
    .map(job => job.op.id), [3, 2, 1]);
  assert.equal(D.advanceEffectFrame(environment), 3);
  assert.deepEqual(order, [3, 2, 1]);
}

// Effect passes clamp before sorting, so distinct out-of-range authored values
// become an equal native key and retain reverse AddPaintJob insertion order.
{
  const jobs = [
    { op: { id: 1 }, effect: { pass: -2, usage: 'base' }, passAdjust: 0 },
    { op: { id: 2 }, effect: { pass: 0, usage: 'base' }, passAdjust: -4 },
  ];
  assert.deepEqual(D.nativeEffectJobOrder(jobs).map(job => job.op.id), [2, 1]);
}

// The native player paints each viewport before traversing the next Demo IPP
// branch. Deferred JS rendering happens after traversal, so jobs must retain
// the marker table as it stood at the end of their own viewport.
{
  const observedMarkers = [];
  const environment = new D.Environment(null);
  const effect = { kind: 'effect', pass: 0, usage: 'other' };
  const effectOp = {
    id: 400, cache: effect, sceneInstances: null,
    exec(activeEnvironment) { observedMarkers.push(activeEnvironment.markers[0][12]); },
  };
  const sceneOp = { cache: new D.Scene(), inputs: [effectOp] };
  const sceneCall = { environment, op: sceneOp };
  const markerHandler = D.handlers.get(0x182);
  const markerOp = { cache: new D.Scene(), inputs: [] };
  const setMarker = x => {
    const matrix = D.mat4Identity(); matrix[12] = x;
    environment.matrixStack.push(matrix);
    markerHandler.exec({ environment, op: markerOp, parameters: [0] });
    environment.matrixStack.pop();
  };
  const makeViewport = marker => {
    D.beginRenderFrame(environment);
    // Queue first: native Engine::Paint still sees markers written later in
    // this viewport's scene traversal.
    D.execSceneInput(sceneCall, 0);
    setMarker(marker);
    return { type: 'viewport', effectJobs: environment.frame.effectJobs.slice() };
  };

  const first = makeViewport(11);
  const second = makeViewport(22);
  assert.equal(D.advanceEffectFrame(environment, [first, second]), 2);
  assert.deepEqual(observedMarkers, [11, 22]);
  // Executing the first job restored the later branch's live marker table,
  // allowing the second job to install its own snapshot independently.
  assert.equal(environment.markers[0][12], 22);
}

// Production has 53 ChainLine operators. Replay from a snapshot must consume
// every queued viewport job at the reference video's exact global 30 fps
// boundaries, yielding the same private SceneMemLink state as a sequential
// 30 fps oracle. Coarse 100 ms batching is deliberately not equivalent because
// ChainLine interpolates moving endpoints independently inside every frame.
{
  const chainHandler = D.handlers.get(0x6b);
  const chainParameters = parameters.slice();
  chainParameters[10] = 0.01;
  chainParameters[11] = 0.985;
  chainParameters[12] = 0.04;
  chainParameters[14] = 0.003;
  chainParameters[15] = 0.2;
  chainParameters[16] = 0;

  function makeChainRuntime() {
    const environment = new D.Environment(null);
    const chainOp = { id: 700, cache: null, sceneInstances: null };
    const chainCall = {
      environment, op: chainOp, inputs: [], links: [material],
      parameters: chainParameters,
    };
    chainOp.cache = chainHandler.init(chainCall);
    chainOp.exec = activeEnvironment => chainHandler.exec({
      ...chainCall, environment: activeEnvironment,
    });

    const timingState = () => ({
      beatTime: environment.beatTime,
      currentTime: environment.currentTime,
      lastTime: environment.lastTime,
      timeDelta: environment.timeDelta,
      timeSlices: environment.timeSlices,
      timeJitter: environment.timeJitter,
      timeReset: environment.timeReset,
    });
    return {
      environment,
      chainOp,
      frameAtSample(sample, sampleRate) {
        const milliseconds = Math.floor(sample * 1000 / sampleRate);
        environment.initFrame(0, milliseconds);
        const phase = milliseconds / 170;
        D.mat4Identity(environment.markers[0]);
        D.mat4Identity(environment.markers[1]);
        environment.markers[0][12] = Math.sin(phase) * 2;
        environment.markers[1][12] = 10 + Math.cos(phase * 0.7) * 2;
        const viewport = {
          type: 'viewport',
          effectJobs: [{
            op: chainOp,
            effect: chainOp.cache,
            matrix: D.mat4Identity(),
            variables: environment.vars.map(value => new Float32Array(value)),
          }],
        };
        environment.lastOutput = { type: 'layer2d', input: viewport };
        environment.frameOutputs.push(environment.lastOutput);
        environment.exitFrame();
      },
      snapshot() {
        return {
          timing: timingState(),
          sceneInstances: chainOp.sceneInstances?.snapshot() ?? null,
        };
      },
      restore(state) {
        Object.assign(environment, state.timing);
        if (state.sceneInstances === null) chainOp.sceneInstances = null;
        else {
          chainOp.sceneInstances ||= new D.InstanceChain();
          chainOp.sceneInstances.restore(state.sceneInstances);
        }
      },
    };
  }

  const sampleRate = 1000;
  const targetSample = 350;
  const samples = Array.from(D.replaySampleSequence(0, targetSample, sampleRate));
  const sequential = makeChainRuntime();
  for (const sample of samples) {
    sequential.frameAtSample(sample, sampleRate);
    const viewport = sequential.environment.lastOutput.input;
    for (const job of viewport.effectJobs.slice().sort(D.compareEffectJobs)) {
      D.executeEffectJob(sequential.environment, job);
    }
  }

  const coarse = makeChainRuntime();
  for (const sample of [100, 200, 300, targetSample]) {
    coarse.frameAtSample(sample, sampleRate);
    D.advanceEffectFrame(coarse.environment);
  }
  assert.notDeepEqual(coarse.chainOp.sceneInstances.snapshot(),
    sequential.chainOp.sceneInstances.snapshot(),
    '100 ms endpoint batching must not masquerade as reference-frame replay');

  const replay = makeChainRuntime();
  const initial = replay.snapshot();
  const app = new D.DebrisApp({ clientWidth: 320, clientHeight: 180 });
  app.runtime = replay;
  app.sampleRate = sampleRate;
  app.snapshots = [{ second: 0, sample: 0, state: initial }];
  await app.seekRuntime(targetSample, { yield: false });

  assert.deepEqual(
    replay.chainOp.sceneInstances.snapshot(),
    sequential.chainOp.sceneInstances.snapshot(),
  );
  assert.equal(replay.chainOp.sceneInstances.items[0].timeCounter, 350);
}

console.log('Perlin, ChainLine, Water, post-effect, and seek replay tests passed');
