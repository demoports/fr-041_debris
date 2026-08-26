import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

import { CLASS_REGISTRY } from '../src/classes.js';
import { parseKX } from '../src/kx.js';
import * as RuntimeAPI from '../src/runtime.js';

const handlers = new Map(Object.entries(RuntimeAPI.runtimeHandlers)
  .map(([id, handler]) => [Number(id), handler]));
const D = { ...RuntimeAPI, CLASS_REGISTRY, parseKX, handlers };

function floatBytes(...values) {
  return Array.from(new Uint8Array(new Float32Array(values).buffer));
}

function makeDocument(classes, operations, options = {}) {
  return {
    classes,
    operations: operations.map((operation, id) => ({
      id,
      classIndex: operation.classIndex,
      classId: classes[operation.classIndex].id,
      inputs: operation.inputs || [],
      links: operation.links || [],
      parameters: operation.parameters || [],
      strings: operation.strings || [],
      splines: operation.splines || [],
      blob: operation.blob || null,
      animation: new Uint8Array(operation.animation || [1]),
    })),
    events: options.events || [],
    splines: options.splines || [],
    roots: [options.root ?? operations.length - 1],
    songBPMFixed: options.songBPMFixed ?? 196 * 65536,
    buzzTiming: options.buzzTiming ?? true,
  };
}

// Released Buzz timing uses an intentionally quantized integer denominator.
assert.equal(D.sampleToBeat(1687, 196 * 65536, true), 8192);
assert.equal(D.sampleToBeat(44100, 196 * 65536, true), 214147);

// Cubic document spline, including the endpoint tangent behavior in kdoc.cpp.
const cubic = new D.Spline({
  interpolation: 0,
  channels: [[
    { time: 0, value: 0 },
    { time: 0.5, value: 1 },
    { time: 1, value: 0 },
  ]],
});
assert.equal(cubic.eval(0.25)[0], 0.625);
assert.equal(cubic.eval(0.75)[0], 0.625);
assert.equal(cubic.eval(-1)[0], 0);
assert.equal(cubic.eval(2)[0], 0);

// sHermite's source spells the cubic as ((2*fade)*fade)*fade. Under the
// released player's 24-bit x87 precision that association differs by one ULP
// from 2*((fade*fade)*fade), including on authored mode-0 camera splines.
{
  const key = (time, px) => ({
    select: 0, time, px, py: 0, pz: 0, rx: 0, ry: 0, rz: 0, zoom: 1,
  });
  const spline = new D.BlobSplinePath({ mode: 0, tension: 1, keys: [key(0, 0), key(1, 1)] });
  const value = spline.eval(Math.fround(0.001));
  assert.equal(value.matrix[12], 0.000002998000127263367);
}

const genericRegistry = {
  900: { id: 900, convention: 2, outputClass: 'KC_ANY', name: 'VM' },
};
const genericHandlers = new Map([[900, {
  init: call => ({ value: call.parameters.slice() }),
  exec: () => {},
}]]);

// Handler tracing is diagnostic-only and bounded. Production frames retain a
// scalar count instead of an ever-growing call log, and return compact frame
// telemetry rather than cloning the complete runtime summary on every seek step.
{
  const document = makeDocument(
    [{ id: 900, convention: 0, packing: '' }],
    [{ classIndex: 0 }],
  );
  const registry = { 900: { id: 900, convention: 0, outputClass: 'KC_DEMO' } };
  const handlers = new Map([[900, {
    init: () => ({ classId: D.KC_DEMO }),
    exec: () => {},
  }]]);
  const runtime = new D.Runtime(document, { registry, handlers });
  runtime.precalc();
  let result = null;
  for (let frame = 0; frame < 1000; frame++) result = runtime.frame(frame, frame * 16);
  assert.deepEqual(result, { beatTime: 999, timeMilliseconds: 15984, handlerCalls: 1, outputs: 0 });
  assert.equal(runtime.handlerCallCount, 1001);
  assert.equal(runtime.handlerCalls.length, 0);
  assert.equal(runtime.summary({ operations: false, events: false }).handlerCalls.length, 0);

  const traced = new D.Runtime(document, { registry, handlers, handlerTraceLimit: 3 });
  traced.precalc();
  for (let frame = 0; frame < 5; frame++) traced.frame(frame, frame * 16);
  assert.equal(traced.handlerCallCount, 6);
  assert.deepEqual(traced.handlerCalls.map(entry => entry.sequence), [3, 4, 5]);
}

// Byte stores address the four lanes inside one 32-bit word.
{
  const document = makeDocument(
    [{ id: 900, convention: 2, packing: 'cc' }],
    [{
      classIndex: 0,
      parameters: [0, 0],
      animation: [0x16, ...floatBytes(-0.1, 0.5, 1.1, 0.999), 0xbf, 1, 1],
    }],
  );
  const runtime = new D.Runtime(document, { registry: genericRegistry, handlers: genericHandlers });
  runtime.precalc();
  assert.equal(runtime.operations[0].animBits[1], 0xfeff7f00);
}

// LOADPARA always reads immutable DataEdit, even after DataAnim changed.
{
  const document = makeDocument(
    [{ id: 900, convention: 2, packing: 'ff' }],
    [{
      classIndex: 0,
      parameters: [2, 0],
      animation: [
        0x18, ...floatBytes(9), 0x91, 0,
        0x03, 0, 0x91, 1,
        1,
      ],
    }],
  );
  const runtime = new D.Runtime(document, { registry: genericRegistry, handlers: genericHandlers });
  runtime.precalc();
  assert.deepEqual(runtime.operations[0].animParameters, [9, 2]);
}

// Source permutation/noise golden at zero.
{
  const document = makeDocument(
    [{ id: 900, convention: 1, packing: 'f' }],
    [{ classIndex: 0, parameters: [0], animation: [0x18, ...floatBytes(0), 0x1c, 0x91, 0, 1] }],
  );
  const runtime = new D.Runtime(document, {
    registry: { 900: { id: 900, convention: 1, outputClass: 'KC_ANY' } },
    handlers: genericHandlers,
  });
  runtime.precalc();
  assert.equal(runtime.operations[0].animParameters[0], 0.4470588266849518);
}

// STOREVAR remains scoped across recursive exec, duplicate inputs remain
// duplicates, and instance memory is an ordered tape rather than an op map.
{
  const classes = [
    { id: 901, convention: 0, packing: '' },
    { id: 902, convention: 0x100, packing: '' },
    { id: 903, convention: 0x200, packing: '' },
  ];
  const parentAnimation = [
    0x02, 0,
    0x18, ...floatBytes(2),
    0x0d,
    0x8f, 0,
    1,
  ];
  const document = makeDocument(classes, [
    { classIndex: 0 },
    { classIndex: 1, inputs: [0], animation: parentAnimation },
    { classIndex: 2, inputs: [1, 0] },
  ]);
  const observations = [];
  const handlers = new Map([
    [901, {
      init: () => ({}),
      exec: call => {
        const mem = call.environment.getInstance(call.op, () => ({ count: 0 }));
        mem.count++;
        observations.push({ time: call.environment.vars[0][0], reset: mem.reset, count: mem.count });
      },
    }],
    [902, { init: () => ({}), exec: call => call.op.execInputs(call.environment) }],
    [903, { init: () => ({}), exec: call => call.op.execInputs(call.environment) }],
  ]);
  const registry = Object.fromEntries(classes.map(item => [item.id, {
    id: item.id, convention: item.convention, outputClass: 'KC_ANY', exec: 'custom',
  }]));
  registry[903].outputClass = 'KC_DEMO';
  const runtime = new D.Runtime(document, { registry, handlers });
  runtime.precalc();
  runtime.frame(0x18000, 0);
  assert.deepEqual(observations, [
    { time: 3, reset: true, count: 1 },
    { time: 1.5, reset: true, count: 1 },
  ]);
  assert.equal(runtime.environment.defaultInstances.items.length, 2);

  runtime.operations[1].sceneInstances = new D.InstanceChain([
    { opId: 0, count: 7, position: new Float32Array([1, 2, 3]) },
  ]);

  const snapshot = runtime.snapshot();
  assert.equal(snapshot.version, 2);
  assert.equal(snapshot.operationCount, 3);
  assert.deepEqual(snapshot.operations.map(state => state.index), [1]);
  runtime.operations[1].sceneInstances.items[0].count = 99;
  observations.length = 0;
  runtime.frame(0x18000, 16);
  assert.deepEqual(observations.map(item => [item.reset, item.count]), [[false, 2], [false, 2]]);
  runtime.restore(snapshot);
  assert.equal(runtime.operations[1].sceneInstances.items[0].count, 7);
  assert.deepEqual(Array.from(runtime.operations[1].sceneInstances.items[0].position), [1, 2, 3]);
  observations.length = 0;
  runtime.frame(0x18000, 16);
  assert.deepEqual(observations.map(item => [item.reset, item.count]), [[false, 2], [false, 2]]);
}

// Static events are attached on [start,end), prepended, interval-remapped,
// and dispatched through Event -> child ExecEvent.
{
  const leafId = 904;
  const classes = [
    { id: leafId, convention: 0, packing: '' },
    { id: 0x06, convention: 0x101, packing: 'f' },
    { id: 0x0d, convention: 0xa5000000, packing: '' },
  ];
  const events = [0.2, 0.8].map(value => ({
    operation: 1, start: 0, end: 0x10000, velocity: 1, modulation: 0, select: 0,
    scale: [1, 1, 1], rotate: [0, 0, 0], translate: [0, 0, 0], color: 0xffffffff,
    spline: null, startInterval: value, endInterval: value, flags: 0,
  }));
  const document = makeDocument(classes, [
    { classIndex: 0 },
    { classIndex: 1, inputs: [0], parameters: [0] },
    { classIndex: 2, inputs: [1] },
  ], { events });
  const seen = [];
  const handlers = new Map(D.handlers);
  handlers.set(leafId, {
    init: () => ({ classId: D.KC_DEMO }),
    exec: call => seen.push(call.environment.vars[0][0]),
  });
  const registry = { ...D.CLASS_REGISTRY,
    [leafId]: { id: leafId, convention: 0, outputClass: 'KC_DEMO' },
  };
  const runtime = new D.Runtime(document, { registry, handlers });
  runtime.precalc();
  runtime.frame(0x8000, 0);
  assert.deepEqual(seen, [Math.fround(0.8), Math.fround(0.2)]);
  assert.equal(runtime.operations[1].firstEvent, null);
  seen.length = 0;
  runtime.frame(0x10000, 16);
  assert.deepEqual(seen, []);
}

// Exec_Misc_Demo commits every active IPP branch in input order.  Resetting
// LastOutput at each branch is important: production has many inactive Event
// wrappers after an active viewport, and the native overlay manager does not
// let those wrappers inherit an earlier render target.
{
  const ippLeafId = 905;
  const classes = [
    { id: ippLeafId, convention: 0, packing: '' },
    { id: 0x06, convention: 0x101, packing: 'f' },
    { id: 0x0d, convention: 0xa5000000, packing: '' },
  ];
  const event = operation => ({
    operation, start: 0, end: 0x10000, velocity: 1, modulation: 0, select: 0,
    scale: [1, 1, 1], rotate: [0, 0, 0], translate: [0, 0, 0], color: 0xffffffff,
    spline: null, startInterval: 0, endInterval: 1, flags: 0,
  });
  const document = makeDocument(classes, [
    { classIndex: 0, strings: ['main'] },
    { classIndex: 1, inputs: [0], parameters: [0] },
    { classIndex: 0, strings: ['inactive-middle'] },
    { classIndex: 1, inputs: [2], parameters: [0] },
    { classIndex: 0, strings: ['glare'] },
    { classIndex: 1, inputs: [4], parameters: [0] },
    { classIndex: 0, strings: ['inactive-tail'] },
    { classIndex: 1, inputs: [6], parameters: [0] },
    { classIndex: 2, inputs: [1, 3, 5, 7] },
  ], { events: [event(1), event(5)] });
  const handlers = new Map(D.handlers);
  const ippCacheSizes = [];
  handlers.set(ippLeafId, {
    init: () => ({ classId: D.KC_IPP, kind: 'ipp' }),
    exec: call => {
      ippCacheSizes.push(call.environment.ippOutputs.size);
      call.environment.lastOutput = {
        kind: 'ipp', type: call.strings[0], opId: call.op.id,
      };
      call.environment.ippOutputs.set(call.op, call.environment.lastOutput);
    },
  });
  const registry = { ...D.CLASS_REGISTRY,
    [ippLeafId]: { id: ippLeafId, convention: 0, outputClass: 'KC_IPP' },
  };
  const runtime = new D.Runtime(document, { registry, handlers });
  runtime.precalc();
  runtime.frame(0x8000, 0);
  assert.deepEqual(runtime.environment.frameOutputs.map(output => output.type), ['main', 'glare']);
  assert.deepEqual(ippCacheSizes, [0, 0],
    'each root IPP branch receives a fresh GenOverlayManager owner cache');
  assert.equal(runtime.environment.lastOutput, null);

  runtime.frame(0x10000, 16);
  assert.deepEqual(runtime.environment.frameOutputs, []);
  assert.equal(runtime.environment.lastOutput, null);
}

// KEnvironment::InitFrame resets both material camera environments and the
// per-root overlay owner cache, exactly as the released runtime does.
{
  const environment = new D.Environment({});
  environment.currentCamera.cameraSpace[12] = 7;
  environment.gameCamera.zoomX = 3;
  environment.ippOutputs.set({}, {});
  environment.initFrame(0, 16);
  assert.deepEqual(Array.from(environment.currentCamera.cameraSpace),
    Array.from(D.MatrixStack ? new D.MatrixStack().top : []));
  assert.equal(environment.gameCamera.zoomX, 1);
  assert.equal(environment.currentCamera.fogColor, 0xff808080);
  assert.equal(environment.ippOutputs.size, 0);
}

// Matrix stack API used by scene Multiply.
{
  const stack = new D.MatrixStack();
  stack.duplicate();
  assert.equal(stack.depth, 2);
  const matrix = new Float32Array(stack.top);
  matrix[12] = 3;
  stack.pop();
  stack.push(matrix);
  assert.equal(stack.top[12], 3);
  stack.pop();
}

// The yielding evaluator preserves the synchronous DFS handler order and
// parent animation-variable scope while returning control between operators.
{
  const classes = [
    { id: 920, convention: 0, packing: '' },
    { id: 921, convention: 0x100, packing: '' },
  ];
  const document = makeDocument(classes, [
    { classIndex: 0 },
    { classIndex: 1, inputs: [0] },
  ]);
  const calls = [];
  const handlers = new Map([
    [920, { init: call => { calls.push([call.op.id, call.environment.vars[0][0]]); return { value: 3 }; } }],
    [921, { init: call => { calls.push([call.op.id, call.environment.vars[0][0]]); return { value: call.inputs[0].value + 1 }; } }],
  ]);
  const registry = {
    920: { id: 920, convention: 0, outputClass: 'KC_ANY' },
    921: { id: 921, convention: 0x100, outputClass: 'KC_ANY' },
  };
  let yields = 0, finalProgress = null;
  const runtime = new D.Runtime(document, { registry, handlers });
  const result = await runtime.precalcAsync(0, {
    budgetMilliseconds: 0,
    yield: async () => { yields++; },
    onProgress: progress => { finalProgress = progress; },
  });
  assert.equal(result.value, 4);
  assert.deepEqual(calls, [[0, 0], [1, 0]]);
  assert.ok(yields >= 1);
  assert.equal(finalProgress.completed, 2);
  assert.equal(runtime.precalculated, true);
}

// Cancellation is observed at every DFS edge and immediately after a yield.
// Active animation scopes unwind, while already-completed immutable children
// remain reusable by a later clean precalc.
{
  const classes = [
    { id: 922, convention: 0, packing: '' },
    { id: 923, convention: 0x100, packing: '' },
  ];
  const document = makeDocument(classes, [
    { classIndex: 0 },
    { classIndex: 1, inputs: [0] },
  ]);
  const calls = [];
  const handlers = new Map([
    [922, { init: call => { calls.push(call.op.id); return { value: 7 }; } }],
    [923, { init: call => { calls.push(call.op.id); return { value: call.inputs[0].value + 1 }; } }],
  ]);
  const registry = {
    922: { id: 922, convention: 0, outputClass: 'KC_ANY' },
    923: { id: 923, convention: 0x100, outputClass: 'KC_ANY' },
  };
  const controller = new AbortController();
  const runtime = new D.Runtime(document, { registry, handlers });
  let announceYield;
  const enteredYield = new Promise(resolve => { announceYield = resolve; });
  const stalledYield = new Promise(() => {});
  const evaluation = runtime.precalcAsync(0, {
    budgetMilliseconds: 0,
    signal: controller.signal,
    yield() { announceYield(); return stalledYield; },
  });
  await enteredYield;
  controller.abort();
  await assert.rejects(evaluation, error => error?.name === 'AbortError');
  assert.equal(runtime.precalculated, false);
  assert.equal(runtime.operations[0]._calcState, 2,
    'a completed child remains a valid immutable cache boundary');
  assert.equal(runtime.operations[1]._calcState, 0,
    'the active parent is reset so precalc can be retried');
  assert.equal(runtime.environment._varSaves.length, 0,
    'all active animation variable saves unwind on abort');
  assert.deepEqual(calls, [0]);
  const result = await runtime.precalcAsync(0, { budgetMilliseconds: 1000 });
  assert.equal(result.value, 8);
  assert.deepEqual(calls, [0, 1]);
  assert.equal(runtime.precalculated, true);
}

// Animation decode errors are transactional in both evaluators. In
// particular, STOREVAR scopes created before the bad opcode cannot leave an
// operator stuck in the cycle-detection state.
{
  const classes = [{ id: 924, convention: 0, packing: '' }];
  const document = makeDocument(classes, [{
    classIndex: 0,
    animation: [0x18, ...floatBytes(2), 0x8f, 0, 0x7f],
  }]);
  const registry = { 924: { id: 924, convention: 0, outputClass: 'KC_ANY' } };
  const handlers = new Map([[924, { init: () => ({}) }]]);
  const runtime = new D.Runtime(document, { registry, handlers });
  assert.throws(() => runtime.precalc(), /unknown animation opcode/);
  assert.equal(runtime.operations[0]._calcState, 0);
  assert.equal(runtime.environment._varSaves.length, 0);
  assert.equal(runtime.environment.vars[0][0], 0);
  await assert.rejects(runtime.precalcAsync(0), /unknown animation opcode/);
  assert.equal(runtime.operations[0]._calcState, 0);
  assert.equal(runtime.environment._varSaves.length, 0);
  assert.equal(runtime.environment.vars[0][0], 0);
}

// The real released blobs exercise Spline, Shaker, PipeSpline and SplineScale
// without requiring canvas or rendering handlers.
{
  const bytes = await readFile(new URL('../assets/debris_party.kx', import.meta.url));
  const document = D.parseKX(bytes);
  // The player fixes x87 arithmetic to single precision. This production
  // sample differs materially if the cubic keeps binary64 intermediates.
  const documentSpline = new D.Spline(document.splines[9], 9);
  assert.equal(documentSpline.eval(Math.fround(0.448))[1], 0.0009689813596196473);
  const runtime = new D.Runtime(document, { handlers: D.handlers, strictHandlers: false });
  const spline = runtime.operations[3340].precalc();
  assert.equal(spline.classId, D.KC_SPLINE);
  assert.equal(spline.keys.length, 4);
  const shaker = runtime.operations[3341].precalc();
  assert.equal(shaker.type, 'ShakerSpline');
  assert.equal(runtime.operations[3341].callRecord().inputs.length, 2);
  const pipe = runtime.operations[15759].precalc();
  assert.equal(pipe.mode, 4);
  assert.equal(pipe.keys.length, 20);
  const scaled = runtime.operations[14830].precalc();
  assert.equal(scaled.keys.length, 7);
  assert.equal(scaled.keys[0].rx, 0);
  assert.equal(scaled.keys[0].rz, 0);

  // Every authored production viewport uses the ordinary fixed-aspect camera
  // path. Camera inputs all terminate in mode-0 BlobSplines; the released KX
  // never selects the game/orbit/stereo/aspect-bypass branches or Scene_Camera.
  // Keep this inventory beside the numeric samples below so a class-map,
  // packing, or graph-link regression cannot masquerade as a camera-angle fix.
  const viewports = runtime.operations.filter(op => op.classId === 0x00f0);
  assert.equal(viewports.length, 104);
  assert.deepEqual([...new Set(viewports.map(op => op.parameters[1] >>> 0))].sort((a, b) => a - b),
    [0, 2, 3, 0x23]);
  assert.equal(viewports.some(op => (op.parameters[1] & (4 | 0x40 | 0x100 | 0x200 | 0x1000)) !== 0),
    false);
  assert.equal(runtime.operations.some(op => op.classId === 0x00c6), false);
  const cameraSplines = viewports.map(op => op.inputs[1]).filter(Boolean);
  assert.equal(cameraSplines.length, 80);
  for (const operation of cameraSplines) {
    let spline = operation.precalc();
    while (spline?.parent) spline = spline.parent;
    assert.equal(spline?.blobSpline?.mode, 0,
      `viewport camera op ${operation.id} must terminate in an Euler BlobSpline`);
  }

  // Source-derived production camera oracles. These cross the 360-beat cut,
  // exercise two event interval remaps, and include the nested EventTime chain
  // used at 420s. Values are the released Buzz sample clock followed by
  // KOp::ExecEvent -> BlobSpline::Calc -> GenSplineShaker::Eval.
  const cameraCases = [
    { seconds: 15, eventId: 15, cameraOp: 11783, beat: 3212215,
      localTime: 0.5317034721374512,
      forward: [0.8037875890731812, 0.3081922233104706, 0.5088571906089783],
      position: [-173.9819793701172, 2.9175710678100586, 42.549407958984375],
      zoom: 1.0664629936218262 },
    { seconds: 60, eventId: 48, cameraOp: 12321, beat: 12848863,
      localTime: 0.15207931399345398,
      forward: [-0.7172273397445679, 0.2745212912559509, 0.6404834389686584],
      position: [-34.25, 1.75, -22], zoom: 1.024321436882019 },
    { seconds: 109.9, eventId: 92, cameraOp: 14524, beat: 23534834,
      localTime: 0.9630444645881653,
      forward: [0.7412523031234741, -0.5333618521690369, 0.40750807523727417],
      position: [-107.06708526611328, 70.04745483398438, -50.48607635498047],
      zoom: 1.2407610416412354 },
    { seconds: 110.3, eventId: 51, cameraOp: 12265, beat: 23620493,
      localTime: 0.017505010589957237,
      forward: [0.7818527817726135, 0.2692064940929413, 0.5623464584350586],
      position: [-48, 2, -122], zoom: 1.7632888555526733 },
    { seconds: 111, eventId: 51, cameraOp: 12265, beat: 23770396,
      localTime: 0.11281076818704605,
      forward: [0.7776069045066833, 0.374447762966156, 0.5050895810127258],
      position: [-48, 2, -122], zoom: 1.8429549932479858 },
    { seconds: 120, eventId: 11, cameraOp: 14715, beat: 25697726,
      localTime: 0.2536306381225586,
      forward: [-0.2575424015522003, 0.8744032382965088, 0.411197692155838],
      position: [127, 12, 53], zoom: 1.4712904691696167 },
    { seconds: 420, eventId: 37, cameraOp: 13611, beat: 89942041,
      localTime: 0.6292357444763184,
      forward: [-0.547887921333313, 0.1179218515753746, 0.8281948566436768],
      position: [12, 1.75, -19.5], zoom: 1.3278101682662964 },
  ];
  for (const expected of cameraCases) {
    const sample = Math.floor(expected.seconds * 44100);
    const beat = D.sampleToBuzzBeat(sample, document.songBPMFixed);
    const event = document.events[expected.eventId];
    let time = event.start < event.end
      ? Math.fround((beat - event.start) / (event.end - event.start)) : 0;
    time = Math.max(0, Math.min(1, time));
    time = Math.fround(event.startInterval + time * (event.endInterval - event.startInterval));
    assert.equal(beat, expected.beat);
    assert.equal(time, expected.localTime);
    runtime.environment.beatTime = beat;
    runtime.environment.vars[0].fill(time);
    const value = runtime.operations[expected.cameraOp].precalc().eval(
      time, 0, new Float32Array(16), runtime.environment,
    );
    assert.deepEqual(Array.from(value.matrix.subarray(8, 11)), expected.forward,
      `camera forward at ${expected.seconds}s`);
    assert.deepEqual(Array.from(value.matrix.subarray(12, 15)), expected.position,
      `camera position at ${expected.seconds}s`);
    assert.equal(value.zoom, expected.zoom, `camera zoom at ${expected.seconds}s`);
  }
}

console.log('runtime tests passed');
