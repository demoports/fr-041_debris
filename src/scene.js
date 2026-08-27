// Scene traversal and the scene operators used by fr-041. Rendering remains a
// separate phase: exec handlers only produce deterministic frame jobs.
import {
  Random,
  f32,
  mat4Copy,
  mat4Direction,
  mat4Euler,
  mat4Identity,
  mat4MulA,
  mat4SRT,
  vec3Normalize,
} from './core.js';

const KV_TIME = 0x00;
const KV_SELECT = 0x03;
const KV_FRACTION = 0x0b;
const PARTICLE_SCRATCH = '_particleScratch';

class Scene {
  constructor() {
    this.kind = 'scene';
    this.children = [];
    this.drawMesh = null;
    this.effect = null;
    this.srt = new Float32Array([1, 1, 1, 0, 0, 0, 0, 0, 0]);
    this.count = 0;
  }

  summary() {
    return {
      kind: this.kind,
      children: this.children.length,
      drawable: this.drawMesh?.kind || null,
      effect: this.effect?.kind || null,
      srt: Array.from(this.srt),
      count: this.count,
    };
  }
}

function objectKind(value) {
  if (!value) return null;
  if (value.kind) return value.kind;
  if (value.outputClass === 'KC_MESH') return 'mesh';
  if (value.outputClass === 'KC_MINMESH') return 'minmesh';
  if (value.outputClass === 'KC_EFFECT') return 'effect';
  if (value.outputClass === 'KC_SCENE') return 'scene';
  return null;
}

function makeScene(value) {
  if (!value) return null;
  const kind = objectKind(value);
  if (kind === 'scene') return value;
  const scene = new Scene();
  if (kind === 'mesh' || kind === 'minmesh') scene.drawMesh = value;
  else if (kind === 'effect') scene.effect = value;
  else return null;
  return scene;
}

function srtFrom(parameters, offset = 0) {
  return new Float32Array([
    parameters[offset] ?? 1,
    parameters[offset + 1] ?? 1,
    parameters[offset + 2] ?? 1,
    parameters[offset + 3] ?? 0,
    parameters[offset + 4] ?? 0,
    parameters[offset + 5] ?? 0,
    parameters[offset + 6] ?? 0,
    parameters[offset + 7] ?? 0,
    parameters[offset + 8] ?? 0,
  ]);
}

function ensureFrame(environment) {
  const frame = environment.frame ||= {};
  frame.meshJobs ||= [];
  frame.effectJobs ||= [];
  frame.lightJobs ||= [];
  frame.effectGeometry ||= [];
  frame.postJobs ||= [];
  frame.effectMarkers ||= (environment.markers || []).map(marker => mat4Copy(marker));
  frame.ambientLight ??= 0;
  return frame;
}

function beginRenderFrame(environment) {
  const frame = ensureFrame(environment);
  frame.meshJobs.length = 0;
  frame.effectJobs.length = 0;
  frame.lightJobs.length = 0;
  frame.effectGeometry.length = 0;
  frame.postJobs.length = 0;
  // Effects execute after the viewport scene has finished traversing. Keep a
  // viewport-local marker table so later Demo/IPP branches cannot change what
  // an earlier viewport's deferred effects observe.
  frame.effectMarkers = (environment.markers || []).map(marker => mat4Copy(marker));
  frame.ambientLight = 0;
  return frame;
}

function copyVariables(environment) {
  const variables = environment.vars || [];
  const result = new Float32Array(variables.length * 4);
  for (let index = 0; index < variables.length; index++) result.set(variables[index], index * 4);
  return result;
}

function outputKind(op) {
  return objectKind(op?.cache) || ({
    KC_MESH: 'mesh',
    KC_MINMESH: 'minmesh',
    KC_EFFECT: 'effect',
    KC_SCENE: 'scene',
  })[op?.classInfo?.outputClass] || null;
}

function execSceneInput(call, index) {
  const { environment, op } = call;
  const inputOp = op.inputs[index];
  if (!inputOp) return;
  const kind = outputKind(inputOp);
  const frame = ensureFrame(environment);
  const matrix = mat4Copy(environment.matrixStack.top);

  if (kind === 'mesh' || kind === 'minmesh') {
    const scene = op.cache;
    const wrapper = scene?.children?.length ? scene.children[index] : scene;
    const mesh = wrapper?.drawMesh || inputOp.cache;
    if (!mesh) return;
    frame.meshJobs.push({
      kind: 'mesh',
      opId: inputOp.id,
      mesh,
      matrix,
      time: environment.vars?.[KV_TIME]?.[0] || 0,
      passAdjust: environment.renderPassAdjust | 0,
      instances: null,
    });
  } else if (kind === 'effect') {
    frame.effectJobs.push({
      kind: 'effect',
      op: inputOp,
      effect: inputOp.cache,
      matrix,
      variables: copyVariables(environment),
      // All jobs in one viewport share this snapshot. Exec_Scene_Marker keeps
      // it current through the end of that viewport's scene traversal.
      markers: frame.effectMarkers,
      passAdjust: environment.renderPassAdjust | 0,
    });
  } else {
    inputOp.exec(environment);
  }
}

function execSceneInputs(call, srt = null) {
  const { environment, op } = call;
  if (srt) environment.matrixStack.pushMul(mat4SRT(srt));
  for (let i = 0; i < op.inputs.length; i++) execSceneInput(call, i);
  if (srt) environment.matrixStack.pop();
}

function initScene(call) {
  const scene = makeScene(call.inputs[0]) || new Scene();
  scene.srt.set(srtFrom(call.parameters));
  return scene;
}

function initAdd(call) {
  const scene = new Scene();
  for (const input of call.inputs) {
    const child = makeScene(input);
    if (child) scene.children.push(child);
  }
  return scene;
}

function initMultiply(call) {
  const child = makeScene(call.inputs[0]);
  if (!child) return null;
  const scene = new Scene();
  scene.children.push(child);
  scene.srt.set(srtFrom(call.parameters));
  scene.count = call.parameters[9] | 0;
  return scene;
}

function execMultiply(call) {
  const { environment, parameters } = call;
  const count = parameters[9] | 0;
  const matrix = mat4SRT(srtFrom(parameters));
  const select = environment.vars[KV_SELECT];
  const saved = new Float32Array(select);

  environment.matrixStack.duplicate();
  for (let i = 0; i < count; i++) {
    select.fill(i);
    execSceneInputs(call);
    const next = mat4MulA(matrix, environment.matrixStack.top);
    environment.matrixStack.pop();
    environment.matrixStack.push(next);
  }
  environment.matrixStack.pop();
  select.set(saved);
}

function initLight() {
  return new Scene();
}

function execLight(call) {
  const { environment, op, parameters: p } = call;
  const srt = new Float32Array([
    1, 1, 1,
    p[0] || 0, p[1] || 0, p[2] || 0,
    p[3] || 0, p[4] || 0, p[5] || 0,
  ]);
  environment.matrixStack.pushMul(mat4SRT(srt));
  const matrix = environment.matrixStack.top;
  const flags = p[6] >>> 0;
  const directional = Boolean(flags & 4);
  const direction = new Float32Array(matrix.subarray(8, 11));
  const range = p[10] || 0;
  const position = directional && !range
    ? new Float32Array([direction[0] * 1e6, direction[1] * 1e6, direction[2] * 1e6])
    : new Float32Array(matrix.subarray(12, 15));
  ensureFrame(environment).lightJobs.push({
    kind: directional ? 'directional' : 'point',
    opId: op.id,
    position,
    direction,
    flags,
    color: p[8] >>> 0,
    amplify: f32(p[9] || 0),
    range: range ? f32(range) : 1e15,
    event: environment.currentEvent || null,
  });
  environment.matrixStack.pop();
}

function initMarker(call) {
  const scene = new Scene();
  const child = makeScene(call.inputs[0]);
  if (child) scene.children.push(child);
  return scene;
}

function execMarker(call) {
  const marker = call.parameters[0] | 0;
  if (marker < 0 || marker >= 32) throw new RangeError(`scene marker ${marker} is outside 0..31`);
  call.environment.markers ||= Array.from({ length: 32 }, () => mat4Identity());
  const value = mat4Copy(call.environment.matrixStack.top);
  call.environment.markers[marker] = value;
  const frame = ensureFrame(call.environment);
  frame.effectMarkers[marker] = mat4Copy(value);
  execSceneInputs(call);
}

function initAdjustPass(call) {
  return makeScene(call.inputs[0]);
}

function execAdjustPass(call) {
  const { environment, op } = call;
  const previous = environment.renderPassAdjust | 0;
  environment.renderPassAdjust = previous + (call.parameters[0] | 0);
  try { op.execInput(environment, 0); }
  finally { environment.renderPassAdjust = previous; }
}

function initApplySpline(call) {
  const child = makeScene(call.inputs[0]);
  if (!child || !call.inputs[1]) return null;
  const scene = new Scene();
  scene.children.push(child);
  return scene;
}

function execApplySpline(call) {
  const evaluated = evalGeneratedSpline(call.inputs[1], call.parameters[0]);
  if (!evaluated) return;
  call.environment.matrixStack.pushMul(evaluated.matrix || evaluated);
  try { execSceneInputs(call); }
  finally { call.environment.matrixStack.pop(); }
}

function initLOD(call) {
  const high = makeScene(call.inputs[0]);
  if (!high) return null;
  const scene = new Scene();
  scene.children.push(high);
  const low = makeScene(call.inputs[1]);
  if (low) scene.children.push(low);
  return scene;
}

function execLOD(call) {
  // CurrentCam is a neutral sMaterialEnv at frame start in the native player,
  // not an absent value. Environment supplies that default, but retain the
  // identity fallback for focused/direct handler use too.
  const current = call.environment.currentCamera?.cameraSpace;
  const camera = current?.length >= 16 ? current : mat4Identity();
  const model = call.environment.matrixStack.top;
  const dx = camera[12] - model[12];
  const dy = camera[13] - model[13];
  const dz = camera[14] - model[14];
  const distance = -(dx * camera[8] + dy * camera[9] + dz * camera[10]);
  if (distance < call.parameters[0]) execSceneInput(call, 0);
  else if (call.op.inputs.length === 2) execSceneInput(call, 1);
}

function addAmbient(frame, color) {
  const old = frame.ambientLight >>> 0;
  const r = Math.min(((old >>> 16) & 255) + ((color >>> 16) & 255), 255);
  const g = Math.min(((old >>> 8) & 255) + ((color >>> 8) & 255), 255);
  const b = Math.min((old & 255) + (color & 255), 255);
  frame.ambientLight = ((r << 16) | (g << 8) | b) >>> 0;
}

function initParticles(call) {
  const scene = new Scene();
  const child = makeScene(call.inputs[0]);
  if (child) scene.children.push(child);
  return scene;
}

function evalValueSpline(spline, time, out = new Float32Array(4)) {
  if (!spline) return null;
  out.fill(0);
  let result;
  if (typeof spline.eval === 'function') result = spline.eval(time, out);
  else if (typeof spline.evaluate === 'function') result = spline.evaluate(time, out);
  else return null;
  if (ArrayBuffer.isView(result) || Array.isArray(result)) return result;
  if (result?.value) return result.value;
  return out;
}

function evalGeneratedSpline(spline, time, matrix, scratch) {
  if (!spline) return null;
  if (typeof spline.evalInto === 'function') {
    if (scratch.generatedSpline !== spline) {
      scratch.generatedSpline = spline;
      scratch.generatedSplineEval = typeof spline.createEvalScratch === 'function'
        ? spline.createEvalScratch() : null;
    }
    return spline.evalInto(time, 0, matrix, scratch.generatedSplineEval);
  }
  if (typeof spline.evalMatrix === 'function') return spline.evalMatrix(time, 0, matrix);
  if (typeof spline.eval !== 'function') return null;
  const result = spline.eval(time, 0, matrix);
  if (result?.matrix || (ArrayBuffer.isView(result) && result.length === 16)) return result;
  return null;
}

function initParticleMemory(mem, p) {
  const mode = p[0] | 0;
  const count = Math.max(0, p[1] | 0);
  const seed = p[2] | 0;
  mem.mode = mode;
  mem.count = count;
  mem.seed = seed;
  mem.rand = new Float32Array(p.slice(3, 6));
  mem.randRot = new Float32Array(p.slice(6, 9));
  mem.randSpeed = new Float32Array(p.slice(16, 19));
  mem.randForw = f32(p[22] || 0);
  mem.randRotSpeed = new Float32Array(p.slice(24, 27));
  mem.comeOut = f32(p[23] || 0);
  mem.pos = new Array(count);
  mem.speed = new Array(count);
  mem.rot = mode & 0x10 ? new Array(count) : null;
  mem.rotSpeed = mode & 0x10 ? new Array(count) : null;

  const random = new Random();
  random.setSeed(seed);
  for (let i = 0; i < count; i++) {
    const pos = new Float32Array(4);
    do {
      pos[0] = f32(random.float(2) - 1);
      pos[1] = f32(random.float(2) - 1);
      pos[2] = f32(random.float(2) - 1);
      if ((mode & 12) === 4) pos.set(vec3Normalize(pos), 0);
    } while ((mode & 12) === 8 && pos[0] ** 2 + pos[1] ** 2 + pos[2] ** 2 > 1);
    pos[3] = mode & 0x20
      ? f32(i / count)
      : f32(random.float(1 - mem.comeOut) + mem.comeOut);
    pos[0] = f32(pos[0] * mem.rand[0] * 0.5);
    pos[1] = f32(pos[1] * mem.rand[1] * 0.5);
    pos[2] = f32(pos[2] * mem.rand[2] * 0.5);
    mem.pos[i] = pos;

    const speed = new Float32Array(4);
    do {
      speed[0] = f32(random.float(2) - 1);
      speed[1] = f32(random.float(2) - 1);
      speed[2] = f32(random.float(2) - 1);
    } while (speed[0] ** 2 + speed[1] ** 2 + speed[2] ** 2 > 1);
    speed[3] = random.float();
    mem.speed[i] = speed;

    if (mem.rot) {
      mem.rot[i] = new Float32Array([
        random.float(mem.randRot[0]),
        random.float(mem.randRot[1]),
        random.float(mem.randRot[2]),
      ]);
      mem.rotSpeed[i] = new Float32Array([
        random.float(mem.randRotSpeed[0]),
        random.float(mem.randRotSpeed[1]),
        random.float(mem.randRotSpeed[2]),
      ]);
    }
  }
}

function particleMemoryChanged(mem, p) {
  if (mem.reset || mem.count === undefined) return true;
  return mem.mode !== (p[0] | 0) || mem.count !== Math.max(0, p[1] | 0) ||
    mem.seed !== (p[2] | 0) ||
    mem.rand[0] !== f32(p[3]) || mem.rand[1] !== f32(p[4]) || mem.rand[2] !== f32(p[5]) ||
    mem.randRot[0] !== f32(p[6]) || mem.randRot[1] !== f32(p[7]) || mem.randRot[2] !== f32(p[8]) ||
    mem.randForw !== f32(p[22] || 0) || mem.comeOut !== f32(p[23] || 0) ||
    mem.randRotSpeed[0] !== f32(p[24]) || mem.randRotSpeed[1] !== f32(p[25]) ||
    mem.randRotSpeed[2] !== f32(p[26]);
}

function particleScratch(mem, mode, count) {
  let scratch = mem[PARTICLE_SCRATCH];
  if (scratch?.mode === mode && scratch.count === count) return scratch;
  const rotating = Boolean(mode & 0x10);
  scratch = {
    mode,
    count,
    instances: mode & 0x100 ? [] : null,
    savedSelect: new Float32Array(4),
    savedFraction: new Float32Array(4),
    splineValue: new Float32Array(4),
    splineAhead: new Float32Array(4),
    splineDirection: [0, 0, 0],
    generatedSpline: null,
    generatedSplineEval: null,
    composedMatrix: rotating ? new Float32Array(16) : null,
    particles: Array.from({ length: count }, () => ({
      position: new Float32Array(3),
      matrix: new Float32Array(16),
      rotation: rotating ? new Float32Array(16) : null,
    })),
  };
  // Particle positions and output matrices are entirely derived from the
  // enumerable random state plus the current parameters. Runtime snapshots use
  // Object.entries(), so keep this potentially large frame cache out of them;
  // particleScratch() recreates it after restore.
  Object.defineProperty(mem, PARTICLE_SCRATCH, {
    configurable: true, enumerable: false, writable: true, value: scratch,
  });
  return scratch;
}

function execParticles(call) {
  const { environment, op, parameters: p } = call;
  const mode = p[0] | 0;
  const mem = environment.getInstance(op, () => ({ reset: true }));
  if (particleMemoryChanged(mem, p)) initParticleMemory(mem, p);

  let anim = p[12] || 0;
  if (!(mode & 0x80)) anim %= 1;
  if (anim < 0) anim += 1;

  const documentSpline = call.splines[0];
  const generatedSpline = call.inputs.length >= 2 ? call.inputs[1] : null;
  const scratch = particleScratch(mem, mode, mem.count);
  const instances = mode & 0x100 ? scratch.instances : null;
  if (instances) instances.length = 0;
  const savedSelect = scratch.savedSelect;
  const savedFraction = scratch.savedFraction;
  savedSelect.set(environment.vars[KV_SELECT]);
  savedFraction.set(environment.vars[KV_FRACTION]);
  const randomSpeed = p[16];
  const gravityX = p[19], gravityY = p[20], gravityZ = p[21];
  const lineX = p[13], lineY = p[14], lineZ = p[15];
  const pulseRate = p[27], pulsePhase = p[28], pulseAmount = p[29];

  for (let i = 0; i < mem.count; i++) {
    let fraction = anim + mem.pos[i][3] + mem.randForw * mem.speed[i][3] * anim;
    if (fraction < 1 && (mode & 64)) continue;
    while (fraction >= 1) fraction -= 1;

    const particle = scratch.particles[i];
    const matrix = mat4Identity(particle.matrix);
    const g = fraction * fraction;
    const position = particle.position;
    position[0] = mem.pos[i][0] + mem.speed[i][0] * fraction * randomSpeed + g * gravityX;
    position[1] = mem.pos[i][1] + mem.speed[i][1] * fraction * randomSpeed + g * gravityY;
    position[2] = mem.pos[i][2] + mem.speed[i][2] * fraction * randomSpeed + g * gravityZ;
    if (pulseAmount) {
      const scale = Math.sin(anim * Math.PI * 2 * pulseRate +
        fraction * Math.PI * 2 * pulsePhase) * pulseAmount + 1;
      position[0] *= scale; position[1] *= scale; position[2] *= scale;
    }

    if ((mode & 1) && documentSpline) {
      const value = evalValueSpline(documentSpline, fraction, scratch.splineValue);
      if (value) {
        if (mode & 0x10) {
          const ahead = evalValueSpline(documentSpline, fraction + 0.01, scratch.splineAhead);
          if (ahead) {
            const direction = scratch.splineDirection;
            direction[0] = ahead[0] - value[0];
            direction[1] = ahead[1] - value[1];
            direction[2] = ahead[2] - value[2];
            mat4Direction(direction, matrix);
          }
        }
        matrix[12] += position[0] + value[0];
        matrix[13] += position[1] + value[1];
        matrix[14] += position[2] + value[2];
      }
    }

    const generated = evalGeneratedSpline(generatedSpline, fraction, matrix, scratch);
    if (generated) {
      const generatedMatrix = generated.matrix || generated;
      if (generatedMatrix !== matrix) mat4Copy(generatedMatrix, matrix);
      const x = matrix[0] * position[0] + matrix[4] * position[1] + matrix[8] * position[2];
      const y = matrix[1] * position[0] + matrix[5] * position[1] + matrix[9] * position[2];
      const z = matrix[2] * position[0] + matrix[6] * position[1] + matrix[10] * position[2];
      matrix[12] += x; matrix[13] += y; matrix[14] += z;
    } else {
      matrix[12] += position[0] + lineX * fraction;
      matrix[13] += position[1] + lineY * fraction;
      matrix[14] += position[2] + lineZ * fraction;
    }

    if (mem.rot || mem.rotSpeed) {
      let rx = fraction * p[9];
      let ry = fraction * p[10];
      let rz = fraction * p[11];
      if (mem.rot) {
        rx += mem.rot[i][0]; ry += mem.rot[i][1]; rz += mem.rot[i][2];
      }
      if (mem.rotSpeed) {
        rx += mem.rotSpeed[i][0] * fraction;
        ry += mem.rotSpeed[i][1] * fraction;
        rz += mem.rotSpeed[i][2] * fraction;
      }
      mat4Euler(rx, ry, rz, particle.rotation);
      mat4MulA(particle.rotation, matrix, scratch.composedMatrix);
      matrix.set(scratch.composedMatrix);
    }

    if (instances) {
      instances.push(matrix);
    } else {
      environment.vars[KV_SELECT].fill(i);
      environment.vars[KV_FRACTION].fill(fraction);
      environment.matrixStack.pushMul(matrix);
      execSceneInput(call, 0);
      environment.matrixStack.pop();
    }
  }

  environment.vars[KV_SELECT].set(savedSelect);
  environment.vars[KV_FRACTION].set(savedFraction);
  if (instances) {
    const scene = op.cache;
    const mesh = scene?.children?.[0]?.drawMesh || call.inputs[0];
    if (mesh) ensureFrame(environment).meshJobs.push({
      kind: 'mesh',
      opId: op.inputs[0]?.id,
      mesh,
      matrix: mat4Copy(environment.matrixStack.top),
      time: 0,
      passAdjust: 0,
      instances,
    });
  }
}

const sceneHandlers = {
  0x00c0: { init: initScene, exec: call => execSceneInputs(call, srtFrom(call.parameters)) },
  0x00c1: { init: initAdd, exec: call => execSceneInputs(call) },
  0x00c2: { init: initMultiply, exec: execMultiply },
  0x00c3: { init: initMultiply, exec: call => execSceneInputs(call, srtFrom(call.parameters)) },
  0x00c4: { init: initLight, exec: execLight },
  0x00c5: { init: initParticles, exec: execParticles },
  0x00d2: { init: initLight, exec: call => call.op.execInputs(call.environment) },
  0x0182: { init: initMarker, exec: execMarker },
  0x0180: { init: initAdjustPass, exec: execAdjustPass },
  0x0181: { init: initApplySpline, exec: execApplySpline },
  0x0183: { init: initLOD, exec: execLOD },
  0x0184: {
    init: initLight,
    exec: call => addAmbient(ensureFrame(call.environment), call.parameters[0] >>> 0),
  },
};

export {
  Scene,
  beginRenderFrame,
  ensureFrame,
  execSceneInput,
  execSceneInputs,
  makeScene,
  sceneHandlers,
};
