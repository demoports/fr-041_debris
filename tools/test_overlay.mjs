import assert from 'node:assert/strict';
import * as CoreAPI from '../src/core.js';
import * as OverlayAPI from '../src/overlay.js';
import * as SceneAPI from '../src/scene.js';

const handlers = new Map(Object.entries(OverlayAPI.overlayHandlers)
  .map(([id, handler]) => [Number(id), handler]));
const D = { ...CoreAPI, ...SceneAPI, ...OverlayAPI, handlers };
const env = {
  aspect: 16 / 9,
  vars: Array.from({ length: 32 }, () => new Float32Array(4)),
  matrixStack: new D.MatrixStack(),
};
let sceneExecutions = 0;
const sceneOp = {
  exec(current) {
    sceneExecutions++;
    D.ensureFrame(current).meshJobs.push({ mesh: { kind: 'mesh' }, matrix: D.mat4Identity() });
  },
};
const parameters = [
  4, 0, 0xff102030,
  0, 0, 0,
  10, 20, 30,
  4096, 0.125,
  0, 0,
  1, 2,
  0xff000000, 4096, 16,
  0.015, 2,
  0, 0, 1, 1,
];
const viewportOp = { id: 1, inputs: [sceneOp] };
const viewportCall = {
  environment: env,
  op: viewportOp,
  inputs: [{ kind: 'scene' }],
  links: [],
  parameters,
};
const viewport = D.handlers.get(0xf0);
viewportOp.cache = viewport.init(viewportCall);
viewport.exec(viewportCall);
assert.equal(sceneExecutions, 1);
assert.equal(env.lastOutput.type, 'viewport');
assert.equal(env.lastOutput.meshJobs.length, 1);
assert.deepEqual(Array.from(env.lastOutput.camera.cameraSpace.subarray(12, 15)), [10, 20, 30]);
assert.equal(env.lastOutput.camera.zoomY, 2 * 16 / 9);
assert.equal(env.lastOutput.lightJobs.length, 1);
assert.equal(env.lastOutput.lightJobs[0], env.lastOutput.defaultLight);
assert.deepEqual(Array.from(env.lastOutput.defaultLight.position), [10, 20, 30]);

// Exec_IPP_Viewport applies the authored aspect before a camera spline
// replaces CameraSpace, then multiplies both zoom lanes by the spline zoom.
// This is the production path for 80 of Debris' 104 viewports.
env.vars[0][0] = 0.25;
const splineCameraSpace = D.mat4EulerTurns(new Float32Array([0.125, -0.25, 0.375]));
splineCameraSpace[12] = 40; splineCameraSpace[13] = 50; splineCameraSpace[14] = 60;
const splineCamera = {
  eval(time, phase) {
    assert.equal(time, 0.25);
    assert.equal(phase, 0);
    return { matrix: splineCameraSpace, zoom: 1.5 };
  },
};
const evaluatedCamera = D.buildCamera({
  ...viewportCall,
  inputs: [{ kind: 'scene' }, splineCamera],
});
assert.deepEqual(Array.from(evaluatedCamera.cameraSpace), Array.from(splineCameraSpace));
assert.equal(evaluatedCamera.zoomX, 1.5);
assert.equal(evaluatedCamera.zoomY, 2 * (16 / 9) * 1.5);
const aspectBypassed = D.buildCamera({
  ...viewportCall,
  inputs: [{ kind: 'scene' }, splineCamera],
  parameters: parameters.map((value, index) => index === 1 ? 0x1000 : value),
});
assert.equal(aspectBypassed.zoomY, 3);

const firstViewport = env.lastOutput;
viewport.exec(viewportCall);
assert.equal(sceneExecutions, 1, 'a viewport target is reused within one root IPP graph');
assert.equal(env.lastOutput, firstViewport);

const base = env.lastOutput;
const material = { kind: 'material', passes: [{ usage: 'other' }] };
let layerInputExecutions = 0;
const layerOp = {
  inputs: [{}],
  execInputs(current) { layerInputExecutions++; current.lastOutput = base; },
};
const layerCall = {
  environment: env,
  op: layerOp,
  inputs: [base],
  links: [material],
  parameters: [2, 0, 0, 1, 2, 0.5, 0.5, 1, 1, 0, 0x30],
};
const layer = D.handlers.get(0xe9);
layerOp.cache = layer.init(layerCall);
layer.exec(layerCall);
assert.equal(env.lastOutput.type, 'layer2d');
assert.equal(env.lastOutput.input, base);
assert.equal(env.lastOutput.size, base.size,
  'Layer2D inherits an existing input render target size');
assert.deepEqual(env.lastOutput.screen, [-0.5, -1, 0.5, 1]);
assert.deepEqual(env.lastOutput.uv, [0, 0, 1, 1]);

const firstLayer = env.lastOutput;
layer.exec(layerCall);
assert.equal(layerInputExecutions, 1, 'Layer2D also reuses its native owner target');
assert.equal(env.lastOutput, firstLayer);

env.ippOutputs.clear();
viewport.exec(viewportCall);
assert.equal(sceneExecutions, 2, 'overlay reset permits a fresh viewport execution');
assert.notEqual(env.lastOutput, firstViewport);

const sentinelOutput = { type: 'sentinel' };
const sentinelCamera = env.currentCamera;
env.lastOutput = sentinelOutput;
const emptyOp = { id: 2, inputs: [] };
viewport.exec({ ...viewportCall, op: emptyOp, inputs: [] });
assert.equal(env.lastOutput, sentinelOutput,
  'a viewport without a scene leaves native LastOutput untouched');
assert.equal(env.currentCamera, sentinelCamera);

// Scene_Ambient belongs to one Engine::StartFrame/viewport execution. The
// viewport snapshots it, then the next fresh viewport starts from zero.
const ambientHandler = SceneAPI.sceneHandlers[0x184];
let ambientExecutions = 0;
const ambientSceneOp = {
  exec(current) {
    ambientExecutions++;
    ambientHandler.exec({ environment: current, op: { id: 0x184 },
      parameters: [0x00102030] });
  },
};
const ambientViewportOp = { id: 3, inputs: [ambientSceneOp] };
ambientViewportOp.cache = viewport.init({ ...viewportCall, op: ambientViewportOp });
viewport.exec({ ...viewportCall, op: ambientViewportOp, inputs: [{ kind: 'scene' }] });
const ambientViewport = env.lastOutput;
assert.equal(ambientExecutions, 1);
assert.equal(ambientViewport.ambientLight, 0x00102030);
assert.equal(ambientViewport.lightJobs.length, 1);
assert.equal(ambientViewport.lightJobs[0], ambientViewport.defaultLight,
  'ambient accumulation does not count as an authored light');

const quietSceneOp = { exec() {} };
const quietViewportOp = { id: 4, inputs: [quietSceneOp] };
quietViewportOp.cache = viewport.init({ ...viewportCall, op: quietViewportOp });
viewport.exec({ ...viewportCall, op: quietViewportOp, inputs: [{ kind: 'scene' }] });
assert.equal(env.lastOutput.ambientLight, 0,
  'a fresh viewport resets Scene_Ambient like Engine::StartFrame');
assert.equal(ambientViewport.ambientLight, 0x00102030,
  'an earlier viewport retains its immutable ambient snapshot');

const lightHandler = SceneAPI.sceneHandlers[0xc4];
const authoredLightSceneOp = {
  exec(current) {
    lightHandler.exec({ environment: current, op: { id: 77 },
      parameters: [0, 0, 0, 10, 20, 30, 0, 0, 0xff804020, 2, 64] });
  },
};
const litViewportOp = { id: 5, inputs: [authoredLightSceneOp] };
litViewportOp.cache = viewport.init({ ...viewportCall, op: litViewportOp });
viewport.exec({ ...viewportCall, op: litViewportOp, inputs: [{ kind: 'scene' }] });
assert.equal(env.lastOutput.ambientLight, 0);
assert.equal(env.lastOutput.lightJobs.length, 1);
assert.equal(env.lastOutput.lightJobs[0].opId, 77);
assert.notEqual(env.lastOutput.lightJobs[0], env.lastOutput.defaultLight,
  'a surviving authored light suppresses the viewport camera-light fallback');

console.log('viewport camera and Layer2D graph tests passed');
