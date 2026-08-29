// Debris-specific dynamic effects. Effects are scheduled by scene traversal
// and executed later in material/pass order by the renderer.
import {
  Random,
  f32,
  mat4Copy,
  mat4Identity,
  mat4TransformPoint,
  sFPow,
  vec3NormalizeSafe,
} from './core.js';
import { InstanceChain } from './runtime.js';
import { ensureFrame } from './scene.js';

const CHAIN_POINTS = 32;
const EFFECT_USAGE_ORDER = Object.freeze({
  base: 0, prelight: 1, ambient: 2, shadow: 3,
  light: 4, postlight: 5, postlight2: 6, other: 7,
});

class Effect {
  constructor(type, material = null) {
    this.kind = 'effect';
    this.type = type;
    this.material = material;
    this.pass = material?.passes?.[0]?.renderPass || 0;
    this.usage = material?.passes?.[0]?.usage || 'other';
    this.needCurrentRender = false;
  }

  summary() {
    return {
      kind: this.kind,
      type: this.type,
      pass: this.pass,
      usage: this.usage,
      needCurrentRender: this.needCurrentRender,
      material: this.material?.system || null,
    };
  }
}

function makeEffect(type, material) {
  return new Effect(type, material?.kind === 'material' ? material : null);
}

function roundEven(value) {
  const lower = Math.floor(value);
  const fraction = value - lower;
  if (fraction < 0.5) return lower;
  if (fraction > 0.5) return lower + 1;
  return lower & 1 ? lower + 1 : lower;
}

let perlinState;
function getPerlinState() {
  if (perlinState) return perlinState;
  const random = new Random();
  random.setSeed(1);
  const keys = new Uint32Array(256);
  const permutation = new Uint8Array(512);
  for (let i = 0; i < 256; i++) {
    keys[i] = random.int(0x10000);
    permutation[i] = i;
  }
  for (let i = 0; i < 255; i++) for (let j = i + 1; j < 256; j++) {
    if (keys[i] > keys[j]) {
      const key = keys[i]; keys[i] = keys[j]; keys[j] = key;
      const item = permutation[i]; permutation[i] = permutation[j]; permutation[j] = item;
    }
  }
  permutation.set(permutation.subarray(0, 256), 256);
  const source = [
    1, 1, 0, 0, 2, 1, 0, 0, 1, 2, 0, 0, 2, 2, 0, 0,
    1, 0, 1, 0, 2, 0, 1, 0, 1, 0, 2, 0, 2, 0, 2, 0,
    0, 1, 1, 0, 0, 2, 1, 0, 0, 1, 2, 0, 0, 2, 2, 0,
    1, 1, 0, 0, 2, 1, 0, 0, 0, 2, 1, 0, 0, 2, 2, 0,
  ];
  const values = [0, 1, -1];
  const gradients = new Float32Array(source.map(index => values[index]));
  perlinState = {
    gradients,
    gradientViews: Array.from({ length: 16 }, (_, index) =>
      gradients.subarray(index * 4, index * 4 + 3)),
    permutation,
    scratch: {
      integer: new Int32Array(3), fade: new Float32Array(3),
      t0: new Float32Array(3), ty: new Float32Array(3),
      t1: new Float32Array(3), t2: new Float32Array(3),
    },
  };
  return perlinState;
}

function lerp3(a, b, t, out = new Float32Array(3)) {
  out[0] = f32(a[0] + (b[0] - a[0]) * t);
  out[1] = f32(a[1] + (b[1] - a[1]) * t);
  out[2] = f32(a[2] + (b[2] - a[2]) * t);
  return out;
}

function perlin3D(position, out = new Float32Array(3)) {
  const { gradientViews, permutation: p, scratch } = getPerlinState();
  const { integer, fade, t0, ty, t1, t2 } = scratch;
  for (let axis = 0; axis < 3; axis++) {
    const base = roundEven(position[axis] - 0.5);
    let fraction = Math.max(0, Math.min(1, position[axis] - base));
    integer[axis] = base & 255;
    fraction = fraction ** 3 * (10 + fraction * (6 * fraction - 15));
    fade[axis] = f32(fraction);
  }
  const gradient = (x, y, z) => {
    const index = p[p[p[x] + y] + z] & 15;
    return gradientViews[index];
  };
  const x = integer[0], y = integer[1], z = integer[2];
  lerp3(gradient(x, y, z), gradient(x + 1, y, z), fade[0], t0);
  lerp3(gradient(x, y + 1, z), gradient(x + 1, y + 1, z), fade[0], ty);
  lerp3(t0, ty, fade[1], t0);
  lerp3(gradient(x, y, z + 1), gradient(x + 1, y, z + 1), fade[0], t1);
  lerp3(gradient(x, y + 1, z + 1), gradient(x + 1, y + 1, z + 1), fade[0], t2);
  lerp3(t1, t2, fade[1], t1);
  return lerp3(t0, t1, fade[2], out);
}

function ensureLineScratch(memory) {
  if (!memory.renderPoints) Object.defineProperty(memory, 'renderPoints', {
    value: memory.lines.map(line => line.position), configurable: true,
  });
  if (!memory._scratch) Object.defineProperty(memory, '_scratch', { configurable: true, value: {
    endpointA: new Float32Array(3), endpointB: new Float32Array(3),
    windPosition: new Float32Array(3), wind: new Float32Array(3),
    windDirection: new Float32Array(3),
  } });
  return memory;
}

function lineMemory() {
  const lines = Array.from({ length: CHAIN_POINTS }, () => ({
    old: new Float32Array(3),
    position: new Float32Array(3),
    next: new Float32Array(3),
  }));
  const memory = {
    reset: true,
    lines,
    ripped: -1,
    collisionAxis: 0,
    timeCounter: 0,
    firstCycles: 250,
    oldA: new Float32Array(3),
    oldB: new Float32Array(3),
  };
  // Render references and arithmetic scratch are derivable from the line
  // state. Keep them non-enumerable so snapshots retain only authored state.
  return ensureLineScratch(memory);
}

function initializeLine(mem, a, b, segmentLength, flags) {
  const difference = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  if (flags & 4) {
    for (let i = 0; i < CHAIN_POINTS; i++) {
      const position = mem.lines[i].position;
      position[0] = a[0]; position[1] = a[1] - i * segmentLength; position[2] = a[2];
      mem.lines[i].old.set(position);
    }
    mem.ripped = CHAIN_POINTS - 1;
  }
  const absolute = difference.map(Math.abs);
  mem.collisionAxis = absolute[0] > absolute[1]
    ? absolute[0] > absolute[2] ? 0 : 2
    : absolute[1] > absolute[2] ? 1 : 2;
  mem.timeCounter = 0;
  mem.firstCycles = 250;
  mem.oldA.set(a);
  mem.oldB.set(b);
}

function staticCatenary(mem, a, b, segmentLength) {
  const dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
  const delta = dx * dx + dy * dy + dz * dz;
  let height = (segmentLength * (CHAIN_POINTS - 1)) ** 2 - delta;
  height = height <= 0 ? 0 : Math.sqrt(height) * 0.4;
  for (let i = 0; i < CHAIN_POINTS; i++) {
    const n = i / (CHAIN_POINTS - 1);
    const m = (i - (CHAIN_POINTS - 1) / 2) / ((CHAIN_POINTS - 1) / 2);
    const position = mem.lines[i].position;
    position[0] = a[0] + (b[0] - a[0]) * n;
    position[1] = Math.max(0, a[1] + (b[1] - a[1]) * n - (1 - m * m) * height);
    position[2] = a[2] + (b[2] - a[2]) * n;
    mem.lines[i].old.set(position);
  }
  mem.ripped = -1;
}

function execChainLine(call) {
  const { environment, op, parameters: p } = call;
  const markerA = p[6] | 0;
  const markerB = p[7] | 0;
  if (markerA < 0 || markerA >= 32 || markerB < 0 || markerB >= 32) {
    throw new RangeError(`ChainLine markers ${markerA}, ${markerB} are outside 0..31`);
  }
  const markers = environment.markers;
  const a = mat4TransformPoint(markers[markerA], [p[0], p[1], p[2], 1]).subarray(0, 3);
  const b = mat4TransformPoint(markers[markerB], [p[3], p[4], p[5], 1]).subarray(0, 3);
  const segmentLength = p[8] / (CHAIN_POINTS - 1);
  const gravity = p[10] * p[10];
  const flags = p[16] >>> 0;
  const mem = ensureLineScratch(environment.getInstance(op, lineMemory));
  if (mem.reset) initializeLine(mem, a, b, segmentLength, flags);
  if ((mem.reset && !(flags & 4)) || (flags & 8)) staticCatenary(mem, a, b, segmentLength);

  const timeslices = Math.max(0, environment.timeSlices | 0);
  const scratch = mem._scratch;
  if (!(flags & 8)) for (let tick = 0; tick < timeslices * 10; tick++) {
    const interpolation = (tick + 1) / (timeslices * 10);
    const endpointA = lerp3(mem.oldA, a, interpolation, scratch.endpointA);
    const endpointB = lerp3(mem.oldB, b, interpolation, scratch.endpointB);
    const axis = mem.collisionAxis;
    const collisionMin = Math.min(endpointA[axis], endpointB[axis]);
    const collisionMax = Math.max(endpointA[axis], endpointB[axis]);
    const windTime = (mem.timeCounter + tick) * p[14];
    scratch.windPosition[0] = windTime;
    scratch.windPosition[1] = windTime * 1.1;
    scratch.windPosition[2] = windTime * 1.2;
    const wind = perlin3D(scratch.windPosition, scratch.wind);
    wind[0] = f32(wind[0] * p[15] + p[17]);
    wind[1] = f32(wind[1] * p[15] * 0.5 + p[18]);
    wind[2] = f32(wind[2] * p[15] + p[19]);
    // Exec_Effect_ChainLine calls UnitSafe3 here, whose 1e-20 epsilon applies
    // to squared length and falls back to +X for a near-zero wind vector.
    const windDirection = vec3NormalizeSafe(wind, scratch.windDirection);

    for (const line of mem.lines) {
      const damping = mem.firstCycles === 0 ? p[11] : 0.975;
      for (let axisIndex = 0; axisIndex < 3; axisIndex++) {
        line.next[axisIndex] = f32(
          line.position[axisIndex] + line.position[axisIndex] * damping - line.old[axisIndex] * damping,
        );
      }
      line.next[1] = f32(line.next[1] - gravity);
    }

    for (let i = 0; i < CHAIN_POINTS - 1; i++) {
      const first = mem.lines[i];
      const second = mem.lines[i + 1];
      const dx = f32(second.position[0] - first.position[0]);
      const dy = f32(second.position[1] - first.position[1]);
      const dz = f32(second.position[2] - first.position[2]);
      const crossX = f32(dy * windDirection[2] - dz * windDirection[1]);
      const crossY = f32(dz * windDirection[0] - dx * windDirection[2]);
      const crossZ = f32(dx * windDirection[1] - dy * windDirection[0]);
      const windForce = Math.sqrt(crossX * crossX + crossY * crossY + crossZ * crossZ) * 0.0001;
      first.next[0] += wind[0] * windForce;
      first.next[1] += wind[1] * windForce;
      first.next[2] += wind[2] * windForce;
      if (i !== mem.ripped) {
        const length = Math.sqrt(dx * dx + dy * dy + dz * dz);
        const inverseLength = length > 1e-10 ? 1 / length : 0;
        const directionX = length > 1e-10 ? f32(dx * inverseLength) : 1;
        const directionY = length > 1e-10 ? f32(dy * inverseLength) : 0;
        const directionZ = length > 1e-10 ? f32(dz * inverseLength) : 0;
        if (length > segmentLength) for (let component = 0; component < 3; component++) {
          const direction = component === 0 ? directionX : component === 1 ? directionY : directionZ;
          const force = direction * (length - segmentLength) * p[12];
          first.next[component] += force;
          second.next[component] -= force;
        }
      }
    }

    for (const line of mem.lines) {
      if (line.next[1] < 0) line.next[1] = 0;
      if ((flags & 3) === 1) {
        if (line.next[axis] < collisionMin) {
          line.position[axis] -= line.next[axis] - collisionMin;
          line.next[axis] = collisionMin;
        }
        if (line.next[axis] > collisionMax) {
          line.position[axis] -= line.next[axis] - collisionMax;
          line.next[axis] = collisionMax;
        }
      } else if ((flags & 3) === 2) {
        line.next[axis] = Math.max(collisionMin, Math.min(collisionMax, line.next[axis]));
      }
    }
    mem.lines[0].next.set(endpointA);
    if (!(flags & 4)) mem.lines[CHAIN_POINTS - 1].next.set(endpointB);
    if (mem.ripped === -1) {
      const dx = endpointA[0] - endpointB[0];
      const dy = endpointA[1] - endpointB[1];
      const dz = endpointA[2] - endpointB[2];
      if (dx * dx + dy * dy + dz * dz > p[13] * p[13]) mem.ripped = p[20] | 0;
    }
    for (const line of mem.lines) {
      line.old.set(line.position);
      line.position.set(line.next);
    }
    if (mem.firstCycles > 0) mem.firstCycles--;
  }
  mem.timeCounter += timeslices * 10;
  mem.oldA.set(a);
  mem.oldB.set(b);

  const effect = op.cache;
  if (effect.material?.passes?.length) ensureFrame(environment).effectGeometry.push({
    kind: 'chain-line',
    opId: op.id,
    material: effect.material,
    // Marker positions were transformed by their marker matrices above and
    // are therefore already in world space. The native effect submits those
    // vertices directly; retaining the deferred scene matrix here applies the
    // parent transform a second time in the WebGL draw path.
    matrix: mat4Identity(),
    points: mem.renderPoints,
    thickness: p[9],
    ripped: mem.ripped,
    color: 0xff404040,
  });
}

function initWater(call) {
  return makeEffect('water', call.links[0]);
}

function createWaterGeometry(grid) {
  grid = Math.max(0, Math.min(8, grid | 0));
  const width = 1 << grid, height = width, vertexCount = width * height;
  const positions = new Float32Array(vertexCount * 3);
  const normals = new Float32Array(vertexCount * 3);
  const tangents = new Float32Array(vertexCount * 4);
  const uvs = new Float32Array(vertexCount * 2);
  const colors = new Uint8Array(vertexCount * 4);
  const shadowVertexMap = new Uint32Array(vertexCount);
  // grid=8 has exactly 65,536 vertices, so every valid index still fits in
  // Uint16. Keeping this static buffer compact matters because the water mesh
  // is updated throughout the final part.
  const indices = new Uint16Array(Math.max(0, (width - 1) * (height - 1) * 6));
  let at = 0;
  for (let z = 0; z < height - 1; z++) for (let x = 0; x < width - 1; x++) {
    const i = z * width + x;
    // sQuad(ip,i+Width,i+1+Width,i+1,i)
    indices[at++] = i + width; indices[at++] = i + 1 + width; indices[at++] = i + 1;
    indices[at++] = i + width; indices[at++] = i + 1; indices[at++] = i;
  }
  for (let index = 0; index < vertexCount; index++) {
    const color = index * 4;
    colors[color] = 0xff; colors[color + 1] = 0xaa;
    colors[color + 2] = 0x88; colors[color + 3] = 0xff;
    // Exec_Effect_Water submits sFVF_TSPACE and explicitly zeroes both
    // tangent-space vectors (fp[10..15]). Keep the authored zero tangent
    // rather than installing the renderer's generic +X fallback.
    shadowVertexMap[index] = index;
  }
  const geometry = {
    kind: 'water-buffer', grid, width, height, positions, normals, tangents,
    uvs, uv0: uvs, uv1: uvs, colors, indices, shadowVertexMap,
    groups: [{ material: 0, start: 0, count: indices.length }], materials: [], version: 0,
    bounds: { minimum: new Float32Array(3), maximum: new Float32Array(3) },
  };
  // Wave tables and the previous-row scratch used to be freshly allocated on
  // every water frame. They are private derivable workspace and remain bounded
  // by the authored spline's maximum simultaneously active wave count.
  Object.defineProperty(geometry, '_waterScratch', {
    value: {
      waves: [], lastRow: new Float32Array(width),
      layoutScale: NaN, layoutUVScale: NaN,
    },
  });
  return geometry;
}

function buildWaterGeometry(spline, parameters, time, geometry = null) {
  const grid = Math.max(0, Math.min(8, parameters[5] | 0));
  if (!geometry || geometry.grid !== grid) geometry = createWaterGeometry(grid);
  const { width, height, positions, normals, uvs } = geometry;
  const scale = parameters[0] || 0;
  const scalePosition = parameters[1] || 0;
  const scaleUV = parameters[2] || 0;
  const speed = f32((parameters[3] || 0) * 32);
  const scaleAmplitude = f32(parameters[4] || 0);
  const blobSpline = spline?.blobSpline || spline;
  const keys = blobSpline?.keys || [];
  const scratch = geometry._waterScratch;
  const waves = scratch.waves;
  let waveCount = 0;
  for (const key of keys) {
    if (key.time > time) continue;
    const deltaTime = f32(f32(time) - key.time);
    const decay = f32(1 / sFPow(key.rz, deltaTime));
    if (!(decay > 0.0001)) continue;
    const wave = waves[waveCount] ||= { x: 0, z: 0, values: new Float32Array(1024) };
    const values = wave.values;
    for (let index = 0; index < 1024; index++) {
      const distance = f32((index / 1024 * 16) * key.ry);
      const phase = f32(f32(deltaTime * speed) - f32(distance * key.rx));
      let damping = f32(scaleAmplitude);
      damping = f32(damping * f32(1 / sFPow(key.rz, phase)));
      damping = f32(damping * f32(1 / sFPow(20, distance)));
      damping = f32(damping * key.zoom);
      values[index] = phase < 0 ? 0 : f32(Math.sin(f32(phase * 3.14)) * damping);
    }
    wave.x = key.px * scalePosition * (width - 1);
    wave.z = key.pz * scalePosition * (height - 1);
    waveCount++;
  }

  const distanceScale = (width * width + height * height) / 1024;
  const widthDenominator = Math.max(1, width - 1);
  const heightDenominator = Math.max(1, height - 1);
  const layoutChanged = scratch.layoutScale !== scale || scratch.layoutUVScale !== scaleUV;
  if (layoutChanged) {
    for (let z = 0; z < height; z++) for (let x = 0; x < width; x++) {
      const vertex = z * width + x, p = vertex * 3, uv = vertex * 2;
      positions[p] = f32(scale * (x - (width - 1) / 2) / widthDenominator);
      positions[p + 2] = f32(scale * (z - (height - 1) / 2) / heightDenominator);
      uvs[uv] = f32(x * scaleUV / widthDenominator);
      uvs[uv + 1] = f32(z * scaleUV / heightDenominator);
    }
    scratch.layoutScale = scale;
    scratch.layoutUVScale = scaleUV;
  }
  const lastRow = scratch.lastRow;
  lastRow.fill(0);
  let lastColumn = 0;
  let minimumY = Infinity, maximumY = -Infinity;
  for (let z = 0; z < height; z++) for (let x = 0; x < width; x++) {
    let py = 0;
    for (let waveIndex = 0; waveIndex < waveCount; waveIndex++) {
      const wave = waves[waveIndex];
      const dx = wave.x - (x - (width - 1) / 2);
      const dz = wave.z - (z - (height - 1) / 2);
      let distance = (dx * dx + dz * dz) / distanceScale;
      let index = Math.trunc(distance);
      distance -= index;
      if (index > 1022) { index = 1022; distance = 0; }
      py -= wave.values[index + 1] * distance + wave.values[index] * (1 - distance);
    }
    const vertex = z * width + x, p = vertex * 3;
    const positionY = f32(scale * py);
    positions[p + 1] = positionY;
    minimumY = Math.min(minimumY, positionY); maximumY = Math.max(maximumY, positionY);
    let nx = py - lastColumn, ny = scale / width, nz = py - lastRow[x];
    const inverseLength = 1 / (Math.sqrt(nx * nx + ny * ny + nz * nz) || 1);
    normals[p] = f32(nx * inverseLength); normals[p + 1] = f32(ny * inverseLength);
    normals[p + 2] = f32(nz * inverseLength);
    lastColumn = py;
    lastRow[x] = py;
  }
  const halfScale = Math.abs(scale) * 0.5;
  geometry.bounds.minimum.set([-halfScale, Number.isFinite(minimumY) ? minimumY : 0, -halfScale]);
  geometry.bounds.maximum.set([halfScale, Number.isFinite(maximumY) ? maximumY : 0, halfScale]);
  geometry.version++;
  geometry.waveCount = waveCount;
  geometry.dynamicAttributes = layoutChanged
    ? ['positions', 'normals', 'uvs'] : ['positions', 'normals'];
  return geometry;
}

function execWater(call) {
  const effect = call.op.cache;
  if (!effect.material?.passes?.length || !call.inputs[0]) return;
  const time = call.environment.vars?.[0]?.[0] || 0;
  effect.waterGeometry = buildWaterGeometry(call.inputs[0], call.parameters, time, effect.waterGeometry);
  ensureFrame(call.environment).effectGeometry.push({
    kind: 'water',
    opId: call.op.id,
    material: effect.material,
    spline: call.inputs[0],
    matrix: mat4Copy(call.environment.matrixStack.top),
    parameters: Array.from(call.parameters),
    geometry: effect.waterGeometry,
    time,
  });
}

function initGlare(call) {
  const effect = makeEffect('glare', null);
  effect.pass = call.parameters[0] | 0;
  effect.usage = 'base';
  return effect;
}

function execGlare(call) {
  ensureFrame(call.environment).postJobs.push({
    kind: 'glare',
    opId: call.op.id,
    parameters: Array.from(call.parameters),
  });
}

function initColorCorrection(call) {
  const effect = makeEffect('color-correction', null);
  effect.pass = call.parameters[0] | 0;
  effect.usage = 'base';
  effect.needCurrentRender = true;
  return effect;
}

function execColorCorrection(call) {
  ensureFrame(call.environment).postJobs.push({
    kind: 'color-correction',
    opId: call.op.id,
    parameters: Array.from(call.parameters),
  });
}

function executeEffectJob(environment, job) {
  const variableCount = environment.vars.length;
  let saved = environment.effectVariableScratch;
  if (!(saved instanceof Float32Array) || saved.length !== variableCount * 4) {
    saved = environment.effectVariableScratch = new Float32Array(variableCount * 4);
  }
  const savedMarkers = environment.markers;
  for (let i = 0; i < variableCount; i++) {
    saved.set(environment.vars[i], i * 4);
    const source = ArrayBuffer.isView(job.variables)
      ? job.variables.subarray(i * 4, i * 4 + 4)
      : job.variables?.[i];
    if (source) environment.vars[i].set(source);
  }
  if (job.markers) environment.markers = job.markers;
  environment.matrixStack.push(job.matrix);
  try {
    // Engine::Paint gives every deferred effect operator its own SceneMemLink
    // instance chain, separate from the event/default scene traversal tape.
    job.op.sceneInstances ||= new InstanceChain();
    if (job.op.sceneInstances && typeof environment.withInstanceChain === 'function') {
      environment.withInstanceChain(job.op.sceneInstances, () => job.op.exec(environment));
    } else {
      job.op.exec(environment);
    }
  } finally {
    environment.matrixStack.pop();
    environment.markers = savedMarkers;
    for (let i = 0; i < variableCount; i++) {
      environment.vars[i].set(saved.subarray(i * 4, i * 4 + 4));
    }
  }
}

function compareEffectJobs(a, b) {
  const pass = job => Math.max(0, Math.min(0xff,
    (job?.effect?.pass | 0) + (job?.passAdjust | 0)));
  return (pass(a) - pass(b)) ||
    ((EFFECT_USAGE_ORDER[a.effect?.usage] ?? 99) -
      (EFFECT_USAGE_ORDER[b.effect?.usage] ?? 99));
}

// Engine::AddPaintJob inserts at the head of its EffectJobs linked list.
// BuildPaintJobs visits that list from the head and the radix sort is stable,
// so effects with identical pass/usage keys execute in reverse scene-traversal
// order. The renderer models this while flattening paint jobs; seeking uses
// this helper because it executes effects without constructing render items.
function nativeEffectJobOrder(jobs) {
  return (jobs || []).slice().reverse().sort(compareEffectJobs);
}

// Native Engine::Paint executes deferred effects while painting every IPP
// viewport. Seeking does not draw those intermediate viewports, but stateful
// effects still need the same calls so their private SceneMemLink tapes consume
// KEnvironment::TimeSlices. Walk the IPP tree in renderer order, execute the
// sorted jobs, and discard the generated draw data.
function advanceEffectFrame(environment, outputs = null) {
  if (!environment) return 0;
  let nodes;
  if (outputs === null || outputs === undefined) {
    nodes = environment.frameOutputs?.length
      ? environment.frameOutputs
      : [environment.lastOutput];
  } else nodes = Array.isArray(outputs) ? outputs : [outputs];

  const frame = ensureFrame(environment);
  let count = 0;
  const visit = node => {
    if (!node) return;
    if (node.type === 'layer2d') {
      visit(node.input);
      return;
    }
    if (node.type !== 'viewport') return;
    frame.effectGeometry.length = 0;
    frame.postJobs.length = 0;
    const jobs = nativeEffectJobOrder(node.effectJobs);
    for (const job of jobs) {
      executeEffectJob(environment, job);
      count++;
    }
    frame.effectGeometry.length = 0;
    frame.postJobs.length = 0;
  };
  for (const node of nodes) visit(node);
  return count;
}

const effectHandlers = {
  0x006b: { init: call => makeEffect('chain-line', call.links[0]), exec: execChainLine },
  0x0073: { init: initWater, exec: execWater },
  0x0074: { init: initGlare, exec: execGlare },
  0x0075: { init: initColorCorrection, exec: execColorCorrection },
};

export {
  Effect,
  advanceEffectFrame,
  buildWaterGeometry,
  compareEffectJobs,
  executeEffectJob,
  nativeEffectJobOrder,
  perlin3D,
  effectHandlers,
};
