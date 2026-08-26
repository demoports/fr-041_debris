// The two IPP operators present in Debris. They build a render graph; WebGL2
// executes that graph after the operator/event traversal has finished.
import {
  mat4Copy,
  mat4EulerTurns,
  mat4Identity,
  vec3Cross,
  vec3Normalize,
} from './core.js';
import { beginRenderFrame } from './scene.js';

const KV_TIME = 0;

class IPP {
  constructor(type = 'empty') {
    this.kind = 'ipp';
    this.type = type;
  }

  summary() {
    return { kind: this.kind, type: this.type };
  }
}

function cameraDefaults() {
  return {
    cameraSpace: mat4Identity(),
    modelSpace: mat4Identity(),
    farClip: 4096,
    nearClip: 0.125,
    centerX: 0,
    centerY: 0,
    zoomX: 1,
    zoomY: 1,
    fogColor: 0,
    fogStart: 0,
    fogEnd: 4096,
  };
}

function evalCameraSpline(spline, time) {
  if (!spline) return null;
  if (typeof spline.evalMatrix === 'function') return spline.evalMatrix(time, 0);
  if (typeof spline.eval !== 'function') return null;
  const value = spline.eval(time, 0);
  if (value?.matrix || (ArrayBuffer.isView(value) && value.length === 16)) return value;
  return null;
}

function buildCamera(call) {
  const { environment, parameters: p } = call;
  const flags = p[1] >>> 0;
  const camera = cameraDefaults();
  camera.farClip = p[9];
  camera.nearClip = p[10];
  camera.centerX = p[11];
  camera.centerY = p[12];
  camera.zoomX = p[13];
  camera.zoomY = p[14];

  const rotation = p.slice(3, 6);
  const position = p.slice(6, 9);
  const game = environment.gameCamera || cameraDefaults();
  if (flags & 4) {
    camera.cameraSpace = mat4Copy(game.cameraSpace);
    if ((flags & 0x300) === 0) {
      camera.centerX = game.centerX;
      camera.centerY = game.centerY;
      camera.zoomX = game.zoomX;
      camera.zoomY = game.zoomY;
      camera.farClip = game.farClip;
      camera.nearClip = game.nearClip;
    }
  } else {
    camera.cameraSpace = mat4EulerTurns(rotation);
    camera.cameraSpace[12] = position[0];
    camera.cameraSpace[13] = position[1];
    camera.cameraSpace[14] = position[2];
  }

  if ((flags & 0x300) === 0x100) {
    const matrix = camera.cameraSpace;
    const k = new Float32Array(matrix.subarray(8, 11));
    const i = vec3Normalize(vec3Cross([0, 1, 0], k));
    const nextK = vec3Cross(i, [0, 1, 0]);
    matrix[0] = i[0]; matrix[1] = i[1]; matrix[2] = i[2];
    matrix[4] = 0; matrix[5] = 1; matrix[6] = 0;
    matrix[8] = nextK[0]; matrix[9] = nextK[1]; matrix[10] = nextK[2];
    matrix[12] = nextK[0] * position[2];
    matrix[13] = nextK[1] * position[2];
    matrix[14] = nextK[2] * position[2];
  } else if ((flags & 0x300) === 0x200) {
    const matrix = camera.cameraSpace;
    matrix[12] = matrix[8] * position[2];
    matrix[13] = matrix[9] * position[2];
    matrix[14] = matrix[10] * position[2];
  }
  camera.cameraSpace[15] = 1;
  camera.fogColor = p[15] >>> 0;
  camera.fogEnd = p[16];
  camera.fogStart = p[17];
  if (!(flags & 0x1000)) camera.zoomY *= environment.aspect ?? 1;

  const spline = call.inputs[1];
  const evaluated = evalCameraSpline(spline, environment.vars?.[KV_TIME]?.[0] || 0);
  if (evaluated) {
    camera.cameraSpace = mat4Copy(evaluated.matrix || evaluated);
    const zoom = evaluated.zoom ?? 1;
    camera.zoomX *= zoom;
    camera.zoomY *= zoom;
  }
  return camera;
}

function initViewport() {
  return new IPP('viewport');
}

function defaultCameraLight(camera) {
  return {
    kind: 'point',
    opId: 0,
    position: new Float32Array(camera.cameraSpace.subarray(12, 15)),
    direction: new Float32Array([0, 0, 1]),
    flags: 0,
    color: 0x78787878,
    amplify: 1,
    range: 32,
    event: null,
  };
}

function execViewport(call) {
  const { environment, op, parameters: p } = call;
  // The released viewport does nothing at all without its scene input; in
  // particular it does not allocate a target or replace LastOutput.
  if (!op.inputs[0]) return;
  environment.ippOutputs ||= new Map();
  const cached = environment.ippOutputs.get(op);
  if (cached) {
    environment.lastOutput = cached;
    return;
  }
  const frame = beginRenderFrame(environment);
  const camera = buildCamera(call);
  environment.currentCamera = camera;
  op.inputs[0]?.exec(environment);

  // Native AddLightJob rejects invisible authored lights immediately. The
  // viewport then adds this camera light if none survived. WebGL performs the
  // equivalent rejection later in selectActiveLights(), so retain the fallback
  // on the viewport even when the raw authored list is non-empty.
  const defaultLight = defaultCameraLight(camera);
  if (frame.lightJobs.length === 0) {
    frame.lightJobs.push(defaultLight);
  }

  const viewport = Object.assign(new IPP('viewport'), {
    size: p[0] | 0,
    flags: p[1] >>> 0,
    clearColor: p[2] >>> 0,
    camera,
    crop: [p[20], p[21], p[22], p[23]],
    eyeDistance: p[18],
    focalDistance: p[19],
    paintMode: (p[1] >>> 4) & 3,
    meshJobs: frame.meshJobs.slice(),
    effectJobs: frame.effectJobs.slice(),
    lightJobs: frame.lightJobs.slice(),
    defaultLight,
    ambientLight: frame.ambientLight >>> 0,
  });
  environment.lastOutput = viewport;
  environment.ippOutputs.set(op, viewport);
  frame.output = viewport;
}

function initLayer2D(call) {
  if (call.links[0]?.kind !== 'material') return null;
  return new IPP('layer2d');
}

function centeredRect(values, centered) {
  if (!centered) return values.slice();
  return [
    values[0] - values[2] * 0.5,
    values[1] - values[3] * 0.5,
    values[0] + values[2] * 0.5,
    values[1] + values[3] * 0.5,
  ];
}

function execLayer2D(call) {
  const { environment, op, parameters: p } = call;
  environment.ippOutputs ||= new Map();
  const cached = environment.ippOutputs.get(op);
  if (cached) {
    environment.lastOutput = cached;
    return;
  }
  environment.lastOutput = null;
  op.execInputs(environment);
  const flags = p[10] >>> 0;
  const input = environment.lastOutput;
  environment.lastOutput = Object.assign(new IPP('layer2d'), {
    input,
    material: call.links[0],
    // Native GENOVER_RTSIZES propagation inherits an existing input's target
    // size, regardless of the layer's requested fallback size.
    size: input?.size ?? (p[0] | 0),
    screen: centeredRect(p.slice(1, 5), Boolean(flags & 0x10)),
    uv: centeredRect(p.slice(5, 9), Boolean(flags & 0x20)),
    z: p[9],
    flags,
    clearFlags: flags & 3,
  });
  environment.ippOutputs.set(op, environment.lastOutput);
}

const overlayHandlers = {
  0x00e9: { init: initLayer2D, exec: execLayer2D },
  0x00f0: { init: initViewport, exec: execViewport },
};

export { IPP, buildCamera, overlayHandlers };
