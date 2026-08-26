// werkkzeug3 KX graph, animation, event, and spline runtime.
import { CLASS_REGISTRY } from './classes.js';
import { MatrixStack, Random, f32, f32ToBits, mat4Euler } from './core.js';
import {
  KC_ANY,
  KC_BITMAP,
  KC_DEMO,
  KC_EFFECT,
  KC_IPP,
  KC_MATERIAL,
  KC_MESH,
  KC_MINMESH,
  KC_SCENE,
  KC_SPLINE,
  OPC_ALTEXEC,
  OPC_ALTINIT,
  OPC_BLOB,
  OPC_DONTCALLLINK,
  OPC_FLEXINPUT,
  OPC_KENV,
  OPC_KOP,
  OPC_SKIPEXEC,
  OPC_STRIPPEDIN,
  OPC_VARIABLEINPUT,
} from './abi.js';

const KA = Object.freeze({
  NOP: 0x00, END: 0x01, LOADVAR: 0x02,
  LOADPARA1: 0x03, LOADPARA2: 0x04, LOADPARA3: 0x05, LOADPARA4: 0x06,
  SWIZZLEX: 0x07, SWIZZLEY: 0x08, SWIZZLEZ: 0x09, SWIZZLEW: 0x0a,
  ADD: 0x0b, SUB: 0x0c, MUL: 0x0d, DIV: 0x0e, MOD: 0x0f,
  INVERT: 0x10, NEG: 0x11, SIN: 0x12, COS: 0x13,
  PULSE: 0x14, RAMP: 0x15, CONSTV: 0x16, CONSTC: 0x17, CONSTS: 0x18,
  SPLINE: 0x19, EVENTSPLINE: 0x1a, MATRIX: 0x1b, NOISE: 0x1c,
  EASE: 0x1d, TIMESHIFT: 0x1e, POW: 0x1f, LOG2: 0x20, POW2: 0x21,
  STOREVAR: 0x80, STOREPARAFLOAT: 0x90, STOREPARAINT: 0xa0,
  STOREPARABYTE: 0xb0, CHANGEPARAFLOAT: 0xd0,
  CHANGEPARAINT: 0xe0, CHANGEPARABYTE: 0xf0,
});

const KEF_NOTIME = 0x0001;
const MAX_VARS = 32;
const MAX_VAR_SAVES = 256;
const KV_TIME = 0;
const KV_MATRIX_I = 0x1c;
const TWO_PI = f32(Math.PI * 2);

// The native player installs x87's 24-bit precision control at startup.
// These helpers make the corresponding rounding points explicit in routines
// whose intermediate values otherwise remain binary64 in JavaScript.
function addF(a, b) { return f32(a + b); }
function subF(a, b) { return f32(a - b); }
function mulF(a, b) { return f32(a * b); }
function divF(a, b) { return f32(a / b); }

function cancellationError(reason = 'operation was cancelled') {
  if (reason?.name === 'AbortError') return reason;
  let error;
  if (typeof DOMException === 'function') error = new DOMException(String(reason || 'operation was cancelled'), 'AbortError');
  else {
    error = new Error(String(reason || 'operation was cancelled'));
    error.name = 'AbortError';
  }
  if (reason instanceof Error && reason !== error) {
    try { error.cause = reason; } catch (_) {}
  }
  return error;
}

// sMaterialEnv::Init() is applied to GameCam and CurrentCam at every native
// KEnvironment::InitFrame(). Keep the same neutral camera available even
// before a viewport has run; generic scene operators such as LOD read it.
function defaultCameraEnvironment() {
  return {
    cameraSpace: identityMatrix(),
    modelSpace: identityMatrix(),
    farClip: 4096,
    nearClip: 0.125,
    centerX: 0,
    centerY: 0,
    zoomX: 1,
    zoomY: 1,
    fogColor: 0xff808080,
    fogStart: 0,
    fogEnd: 4096,
    orthogonal: 0,
  };
}

function throwIfCancelled(options = {}) {
  if (options.signal?.aborted) throw cancellationError(options.signal.reason);
  if (options.shouldAbort?.()) throw cancellationError();
}

function waitForCancellation(value, options = {}) {
  const signal = options.signal;
  if (!signal) return Promise.resolve(value);
  if (signal.aborted) return Promise.reject(cancellationError(signal.reason));
  return new Promise((resolve, reject) => {
    let settled = false;
    const finish = (callback, result) => {
      if (settled) return;
      settled = true;
      signal.removeEventListener('abort', onAbort);
      callback(result);
    };
    const onAbort = () => finish(reject, cancellationError(signal.reason));
    signal.addEventListener('abort', onAbort, { once: true });
    Promise.resolve(value).then(
      result => finish(resolve, result),
      error => finish(reject, error),
    );
  });
}

const OUTPUT_CLASS_IDS = Object.freeze({
  KC_BITMAP, KC_MINMESH, KC_SCENE, KC_MATERIAL, KC_MESH, KC_IPP,
  KC_EFFECT, KC_DEMO, KC_SPLINE, KC_ANY,
});

function inputSlots(convention) { return (convention & 0x00000f00) >>> 8; }
function linkSlots(convention) { return (convention & 0x0000f000) >>> 12; }
function stringSlots(convention) { return (convention & 0x00070000) >>> 16; }
function splineSlots(convention) { return (convention & 0x00700000) >>> 20; }
function dataWords(convention) { return convention & 0xff; }

function asBytes(value) {
  if (!value) return new Uint8Array();
  if (value instanceof Uint8Array) return value;
  if (value instanceof ArrayBuffer) return new Uint8Array(value);
  if (ArrayBuffer.isView(value)) {
    return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
  }
  throw new TypeError('expected an ArrayBuffer or typed-array view');
}

function vector(x = 0, y = 0, z = 0, w = 0) {
  return new Float32Array([x, y, z, w]);
}

function copyVector(source) {
  return new Float32Array(source);
}

function setScalar(out, value) {
  value = f32(value);
  out[0] = out[1] = out[2] = out[3] = value;
  return out;
}

function identityMatrix(out = new Float32Array(16)) {
  out.fill(0);
  out[0] = out[5] = out[10] = out[15] = 1;
  return out;
}

function copyMatrix(source, out = new Float32Array(16)) {
  out.set(source);
  return out;
}

// sMatrix::MulA(a,b): the source stores i,j,k,l as four consecutive vectors.
function mulAffine(a, b, out = new Float32Array(16)) {
  const target = out === a || out === b ? new Float32Array(16) : out;
  for (let n = 0; n < 4; n++) {
    const o = n * 4;
    const ax = a[o], ay = a[o + 1], az = a[o + 2];
    target[o] = f32(b[0] * ax + b[4] * ay + b[8] * az);
    target[o + 1] = f32(b[1] * ax + b[5] * ay + b[9] * az);
    target[o + 2] = f32(b[2] * ax + b[6] * ay + b[10] * az);
    target[o + 3] = 0;
  }
  target[12] = f32(target[12] + b[12]);
  target[13] = f32(target[13] + b[13]);
  target[14] = f32(target[14] + b[14]);
  target[15] = 1;
  if (target !== out) out.set(target);
  return out;
}

function mulRotation(a, b, out = new Float32Array(16)) {
  const target = out === a || out === b ? new Float32Array(16) : out;
  identityMatrix(target);
  for (let n = 0; n < 3; n++) {
    const o = n * 4;
    const ax = a[o], ay = a[o + 1], az = a[o + 2];
    target[o] = f32(b[0] * ax + b[4] * ay + b[8] * az);
    target[o + 1] = f32(b[1] * ax + b[5] * ay + b[9] * az);
    target[o + 2] = f32(b[2] * ax + b[6] * ay + b[10] * az);
    target[o + 3] = 0;
  }
  if (target !== out) out.set(target);
  return out;
}

function initEuler(a, b, c, out = new Float32Array(16)) {
  return mat4Euler(f32(a), f32(b), f32(c), out);
}

function initEulerTurns(x, y, z, out = new Float32Array(16)) {
  if (Math.abs(x) < 1e-6 && Math.abs(y) < 1e-6 && Math.abs(z) < 1e-6) {
    return identityMatrix(out);
  }
  return initEuler(f32(f32(x) * TWO_PI), f32(f32(y) * TWO_PI),
    f32(f32(z) * TWO_PI), out);
}

function normalize3(value, out = new Float32Array(3)) {
  const length2 = value[0] * value[0] + value[1] * value[1] + value[2] * value[2];
  if (length2 > 1e-20) {
    const scale = 1 / Math.sqrt(length2);
    out[0] = f32(value[0] * scale);
    out[1] = f32(value[1] * scale);
    out[2] = f32(value[2] * scale);
  } else {
    out[0] = 1; out[1] = out[2] = 0;
  }
  return out;
}

function cross3(a, b, out = new Float32Array(3)) {
  const x = a[1] * b[2] - a[2] * b[1];
  const y = a[2] * b[0] - a[0] * b[2];
  const z = a[0] * b[1] - a[1] * b[0];
  out[0] = f32(x); out[1] = f32(y); out[2] = f32(z);
  return out;
}

function initDirection(direction, out = new Float32Array(16)) {
  const k = normalize3(direction);
  const i = normalize3(cross3([0, 1, 0], k));
  const j = cross3(k, i);
  out.set([i[0], i[1], i[2], 0], 0);
  out.set([j[0], j[1], j[2], 0], 4);
  out.set([k[0], k[1], k[2], 0], 8);
  out.set([0, 0, 0, 1], 12);
  return out;
}

function initAxisAngle(axis, angle, out = new Float32Array(16)) {
  const u = normalize3(axis);
  const c = Math.cos(angle), s = Math.sin(angle);
  const x = u[0], y = u[1], z = u[2];
  out[0] = f32(x * x + c * (1 - x * x));
  out[1] = f32(x * y + c * (-x * y) + s * z);
  out[2] = f32(x * z + c * (-x * z) - s * y); out[3] = 0;
  out[4] = f32(y * x + c * (-y * x) - s * z);
  out[5] = f32(y * y + c * (1 - y * y));
  out[6] = f32(y * z + c * (-y * z) + s * x); out[7] = 0;
  out[8] = f32(z * x + c * (-z * x) + s * y);
  out[9] = f32(z * y + c * (-z * y) - s * x);
  out[10] = f32(z * z + c * (1 - z * z)); out[11] = 0;
  out[12] = out[13] = out[14] = 0; out[15] = 1;
  return out;
}

function quaternionFromMatrix(matrix) {
  const result = new Float32Array(4); // w,x,y,z
  const trace = matrix[0] + matrix[5] + matrix[10];
  let s;
  if (trace >= 0) {
    s = Math.sqrt(trace + 1);
    result[0] = f32(s * 0.5);
    s = 0.5 / s;
    result[1] = f32((matrix[9] - matrix[6]) * s);
    result[2] = f32((matrix[2] - matrix[8]) * s);
    result[3] = f32((matrix[4] - matrix[1]) * s);
  } else {
    let index;
    if (matrix[5] > matrix[0]) index = matrix[10] > matrix[5] ? 2 : 1;
    else index = matrix[10] > matrix[0] ? 2 : 0;
    if (index === 0) {
      s = Math.sqrt((matrix[0] - (matrix[5] + matrix[10])) + 1);
      result[1] = f32(s * 0.5); s = 0.5 / s;
      result[2] = f32((matrix[1] + matrix[4]) * s);
      result[3] = f32((matrix[8] + matrix[2]) * s);
      result[0] = f32((matrix[9] - matrix[6]) * s);
    } else if (index === 1) {
      s = Math.sqrt((matrix[5] - (matrix[10] + matrix[0])) + 1);
      result[2] = f32(s * 0.5); s = 0.5 / s;
      result[3] = f32((matrix[6] + matrix[9]) * s);
      result[1] = f32((matrix[1] + matrix[4]) * s);
      result[0] = f32((matrix[2] - matrix[8]) * s);
    } else {
      s = Math.sqrt((matrix[10] - (matrix[0] + matrix[5])) + 1);
      result[3] = f32(s * 0.5); s = 0.5 / s;
      result[1] = f32((matrix[8] + matrix[2]) * s);
      result[2] = f32((matrix[6] + matrix[9]) * s);
      result[0] = f32((matrix[4] - matrix[1]) * s);
    }
  }
  return result;
}

function quaternionToMatrix(quaternion, out = new Float32Array(16)) {
  const w = quaternion[0], x = quaternion[1], y = quaternion[2], z = quaternion[3];
  const xx = 2 * x * x, xy = 2 * x * y, xz = 2 * x * z;
  const yy = 2 * y * y, yz = 2 * y * z, zz = 2 * z * z;
  const xw = 2 * x * w, yw = 2 * y * w, zw = 2 * z * w;
  out[0] = f32(1 - yy - zz); out[1] = f32(xy - zw); out[2] = f32(xz + yw); out[3] = 0;
  out[4] = f32(xy + zw); out[5] = f32(1 - xx - zz); out[6] = f32(yz - xw); out[7] = 0;
  out[8] = f32(xz - yw); out[9] = f32(yz + xw); out[10] = f32(1 - xx - yy); out[11] = 0;
  out[12] = out[13] = out[14] = 0; out[15] = 1;
  return out;
}

function quaternionLerp(a, b, time) {
  const out = new Float32Array(4);
  const dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  const sign = dot < 0 ? -1 : 1;
  for (let i = 0; i < 4; i++) out[i] = f32(a[i] + (sign * b[i] - a[i]) * time);
  const scale = 1 / Math.sqrt(out[0] * out[0] + out[1] * out[1] + out[2] * out[2] + out[3] * out[3]);
  for (let i = 0; i < 4; i++) out[i] = f32(out[i] * scale);
  return out;
}

function rotateVector4(matrix, value, out = new Float32Array(4)) {
  const x = value[0], y = value[1], z = value[2], w = value[3];
  out[0] = f32(matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12] * w);
  out[1] = f32(matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13] * w);
  out[2] = f32(matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14] * w);
  out[3] = f32(matrix[3] * x + matrix[7] * y + matrix[11] * z + matrix[15] * w);
  return out;
}

function roundEven(value) {
  if (!Number.isFinite(value) || value < -2147483648 || value >= 2147483648) return -2147483648;
  const floor = Math.floor(value);
  const fraction = value - floor;
  if (fraction < 0.5) return floor | 0;
  if (fraction > 0.5) return (floor + 1) | 0;
  return (floor & 1 ? floor + 1 : floor) | 0;
}

function truncInt(value) {
  if (!Number.isFinite(value) || value < -2147483648 || value >= 2147483648) return -2147483648;
  return Math.trunc(value) | 0;
}

function clamp(value, maximum, minimum) {
  return value >= maximum ? maximum : value <= minimum ? minimum : value;
}

function fnvWord(hash, word) {
  hash ^= word >>> 0;
  return Math.imul(hash, 0x01000193) >>> 0;
}

function hashFloatRecords(records, fields) {
  let hash = 0x811c9dc5;
  for (const record of records) {
    for (const field of fields) hash = fnvWord(hash, f32ToBits(record[field] || 0));
  }
  return hash >>> 0;
}

function cloneState(value) {
  if (value === null || value === undefined || typeof value !== 'object') return value;
  if (ArrayBuffer.isView(value)) return new value.constructor(value);
  if (value instanceof ArrayBuffer) return value.slice(0);
  if (Array.isArray(value)) return value.map(cloneState);
  const result = {};
  for (const [key, item] of Object.entries(value)) {
    if (key === 'op' || key === 'runtime' || key === 'environment') continue;
    if (typeof item !== 'function') result[key] = cloneState(item);
  }
  return result;
}

function stateSummary(value) {
  if (value === null || value === undefined || typeof value !== 'object') return value;
  if (ArrayBuffer.isView(value)) return Array.from(value);
  if (Array.isArray(value)) return value.map(stateSummary);
  const result = {};
  for (const key of Object.keys(value).sort()) {
    if (key === 'op' || key === 'runtime' || key === 'environment') continue;
    if (typeof value[key] !== 'function') result[key] = stateSummary(value[key]);
  }
  return result;
}

function registryGet(registry, id) {
  if (!registry) return null;
  if (typeof registry.get === 'function') return registry.get(id) || null;
  return registry[id] || registry[String(id)] || null;
}

function handlerGet(handlers, id) {
  if (!handlers) return null;
  if (typeof handlers.get === 'function') return handlers.get(id) || null;
  return handlers[id] || handlers[String(id)] || null;
}

function outputClassId(classInfo) {
  if (!classInfo) return KC_ANY;
  if (Number.isInteger(classInfo.outputClassId)) return classInfo.outputClassId;
  return OUTPUT_CLASS_IDS[classInfo.outputClass] ?? KC_ANY;
}

function parameterIsFloat(kind) {
  kind = (kind || '-').toLowerCase();
  return kind === 'g' || kind === 'f' || kind === 'e';
}

function parameterIsSigned(kind) {
  kind = (kind || '-').toLowerCase();
  return kind === 'i' || kind === 's';
}

class InstanceChain {
  constructor(items = []) {
    this.items = items;
  }

  clear() { this.items.length = 0; }
  snapshot() { return this.items.map(cloneState); }
  restore(items) { this.items = (items || []).map(cloneState); }
  summary() {
    return this.items.map(item => ({ opId: item.opId, state: stateSummary(item) }));
  }
}

class Spline {
  constructor(raw = {}, id = -1) {
    this.id = id;
    this.interpolation = raw.interpolation | 0;
    this.channels = (raw.channels || []).map(channel => channel.map(key => ({
      time: f32(key.time),
      value: f32(key.value),
    })));
  }

  eval(time, out = vector()) {
    out.fill(0);
    for (let i = 0; i < this.channels.length && i < 4; i++) {
      out[i] = this.evalChannel(this.channels[i], time);
    }
    return out;
  }

  evaluate(time, out) { return this.eval(time, out); }

  evalChannel(keys, time) {
    const count = keys.length;
    if (!count) return 0;
    if (time <= keys[0].time) return keys[0].value;
    if (time >= keys[count - 1].time) return keys[count - 1].value;
    for (let k = 0; k < count - 1; k++) {
      if (keys[k + 1].time > time) {
        const key = keys[k], next = keys[k + 1];
        // kdoc.cpp runs with the x87 precision control set to 24 bits. Model
        // every arithmetic instruction as a rounded float rather than keeping
        // JavaScript's binary64 intermediates until the final assignment.
        const difference = f32(next.value - key.value);
        const timeDifference = f32(next.time - key.time);
        const t = f32(f32(f32(time) - key.time) / timeDifference);
        if (this.interpolation === 1) return f32(f32(t * difference) + key.value);
        if (this.interpolation === 2) return key.value;
        const d0 = k === 0
          ? difference
          : f32(f32(timeDifference * f32(next.value - keys[k - 1].value)) /
            f32(next.time - keys[k - 1].time));
        const d1 = k === count - 2
          ? difference
          : f32(f32(timeDifference * f32(keys[k + 2].value - key.value)) /
            f32(keys[k + 2].time - key.time));
        let value = f32(d0 + d1);
        value = f32(value - f32(2 * difference));
        value = f32(value * t);
        value = f32(value + f32(3 * difference));
        value = f32(value - f32(2 * d0));
        value = f32(value - d1);
        value = f32(value * t);
        value = f32(value + d0);
        value = f32(value * t);
        return f32(value + key.value);
      }
    }
    return 0;
  }

  summary() {
    let hash = 0x811c9dc5;
    for (const channel of this.channels) {
      hash = fnvWord(hash, channel.length);
      for (const key of channel) {
        hash = fnvWord(hash, f32ToBits(key.time));
        hash = fnvWord(hash, f32ToBits(key.value));
      }
    }
    return {
      id: this.id,
      interpolation: this.interpolation,
      channelKeyCounts: this.channels.map(channel => channel.length),
      hash: hash >>> 0,
    };
  }
}

function hermite(p0, p1, p2, p3, fade, tension, continuity, bias, ignoreTime, derivative = false) {
  fade = f32(fade); tension = f32(tension); continuity = f32(continuity); bias = f32(bias);
  const oneMinusTension = subF(1, tension);
  let a1 = mulF(mulF(oneMinusTension, subF(1, continuity)), addF(1, bias));
  const b1 = mulF(mulF(oneMinusTension, addF(1, continuity)), subF(1, bias));
  const a2 = mulF(mulF(oneMinusTension, addF(1, continuity)), addF(1, bias));
  let b2 = mulF(mulF(oneMinusTension, subF(1, continuity)), subF(1, bias));
  let t0, t1 = p1.time, t2 = p2.time, t3;
  if (!p0) { p0 = p2; a1 = -a1; t0 = addF(t1, subF(t1, t2)); } else t0 = p0.time;
  if (!p3) { p3 = p1; b2 = -b2; t3 = addF(t2, subF(t2, t1)); } else t3 = p3.time;

  const ff = mulF(fade, fade), fff = mulF(ff, fade);
  // Preserve the source expression's left association. With the player's
  // 24-bit x87 precision, ((2*fade)*fade)*fade is observably different from
  // 2*((fade*fade)*fade) for a handful of authored camera keys.
  const twoFade2 = mulF(mulF(2, fade), fade);
  const twoFade3 = mulF(twoFade2, fade);
  const threeFade2 = mulF(mulF(3, fade), fade);
  const sixFade2 = mulF(mulF(6, fade), fade);
  const factors = [
    addF(subF(twoFade3, threeFade2), 1),
    addF(-twoFade3, threeFade2),
    addF(subF(fff, twoFade2), fade),
    subF(fff, ff),
  ];
  const derivatives = [
    subF(sixFade2, mulF(6, fade)),
    addF(-sixFade2, mulF(6, fade)),
    addF(subF(threeFade2, mulF(4, fade)), 1),
    subF(threeFade2, mulF(2, fade)),
  ];
  if (!ignoreTime) {
    t0 = subF(t1, t0); t1 = subF(t2, t1); t2 = subF(t3, t2);
    factors[2] = divF(mulF(factors[2], t1), addF(t1, t0));
    factors[3] = divF(mulF(factors[3], t1), addF(t1, t2));
    derivatives[2] = divF(mulF(derivatives[2], t1), addF(t1, t0));
    derivatives[3] = divF(mulF(derivatives[3], t1), addF(t1, t2));
  } else {
    factors[2] = mulF(factors[2], 0.5); factors[3] = mulF(factors[3], 0.5);
    derivatives[2] = mulF(derivatives[2], 0.5); derivatives[3] = mulF(derivatives[3], 0.5);
  }
  for (let i = 0; i < 4; i++) derivatives[i] = mulF(derivatives[i], 0.25);

  const fields = ['px', 'py', 'pz', 'rx', 'ry', 'rz', 'zoom'];
  const value = {}, delta = {};
  for (const field of fields) {
    const tangent1 = addF(mulF(a1, subF(p1[field], p0[field])),
      mulF(b1, subF(p2[field], p1[field])));
    const tangent2 = addF(mulF(a2, subF(p2[field], p1[field])),
      mulF(b2, subF(p3[field], p2[field])));
    value[field] = addF(addF(addF(mulF(factors[0], p1[field]),
      mulF(factors[1], p2[field])), mulF(factors[2], tangent1)),
    mulF(factors[3], tangent2));
    if (derivative) {
      delta[field] = addF(addF(addF(mulF(derivatives[0], p1[field]),
        mulF(derivatives[1], p2[field])), mulF(derivatives[2], tangent1)),
      mulF(derivatives[3], tangent2));
    }
  }
  return { value, derivative: delta };
}

class BlobSplinePath {
  constructor(data = {}) {
    this.classId = KC_SPLINE;
    this.type = 'Spline';
    this.version = data.version ?? 3;
    this.select = data.select ?? -1;
    this.mode = data.mode ?? 0;
    this.target = new Float32Array(data.target || [0, 0, 0, 1]);
    this.continuity = f32(data.continuity || 0);
    this.tension = f32(data.tension || 0);
    this.uniform = data.uniform | 0;
    this.keys = (data.keys || []).map(key => ({
      select: key.select | 0, time: f32(key.time),
      px: f32(key.px), py: f32(key.py), pz: f32(key.pz),
      rx: f32(key.rx), ry: f32(key.ry), rz: f32(key.rz), zoom: f32(key.zoom),
    }));
    this.pipe = data.pipe || null;
  }

  static fromBlob(blob) {
    const bytes = asBytes(blob);
    if (bytes.byteLength < 64) throw new Error(`Spline blob is only ${bytes.byteLength} bytes`);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const count = view.getInt32(0, true);
    if (count < 2 || 64 + count * 36 > bytes.byteLength) {
      throw new Error(`invalid Spline blob key count ${count} for ${bytes.byteLength} bytes`);
    }
    const keys = [];
    for (let i = 0; i < count; i++) {
      const at = 64 + i * 36;
      keys.push({
        select: view.getInt32(at, true), time: view.getFloat32(at + 4, true),
        px: view.getFloat32(at + 8, true), py: view.getFloat32(at + 12, true),
        pz: view.getFloat32(at + 16, true), rx: view.getFloat32(at + 20, true),
        ry: view.getFloat32(at + 24, true), rz: view.getFloat32(at + 28, true),
        zoom: view.getFloat32(at + 32, true),
      });
    }
    const result = new BlobSplinePath({
      version: view.getInt32(4, true), select: view.getInt32(8, true),
      mode: view.getInt32(12, true),
      target: [view.getFloat32(16, true), view.getFloat32(20, true),
        view.getFloat32(24, true), view.getFloat32(28, true)],
      continuity: view.getFloat32(32, true), tension: view.getFloat32(36, true),
      uniform: view.getInt32(40, true), keys,
    });
    if (result.version < 3) {
      result.uniform = 0; result.tension = 0; result.continuity = 0;
    }
    result.version = 3;
    return result;
  }

  clone() {
    return new BlobSplinePath(this);
  }

  get blobSpline() { return this; }

  _segment(time) {
    if (this.keys.length < 2) throw new Error('Spline needs at least two keys');
    let index;
    if (time <= this.keys[0].time) { index = 1; time = 0; }
    else if (time >= this.keys[this.keys.length - 1].time) {
      index = this.keys.length - 1; time = this.keys[index].time;
    } else {
      index = 1;
      while (index < this.keys.length && this.keys[index].time <= time) index++;
    }
    const p1 = this.keys[index - 1], p2 = this.keys[index];
    const fade = p2.time === p1.time ? p2.time
      : divF(subF(f32(time), p1.time), subF(p2.time, p1.time));
    return { p0: index >= 2 ? this.keys[index - 2] : null, p1, p2,
      p3: index + 1 < this.keys.length ? this.keys[index + 1] : null, fade };
  }

  eval(time, phase = 0, matrix = new Float32Array(16)) {
    void phase;
    const segment = this._segment(time);
    let key, keyDerivative = null;
    if (this.mode === 4) {
      const t = f32(segment.fade);
      const tt = mulF(t, t), ttt = mulF(tt, t);
      const f1 = addF(subF(mulF(2, ttt), mulF(3, tt)), 1);
      const f2 = addF(mulF(-2, ttt), mulF(3, tt));
      const f3 = mulF(addF(subF(ttt, mulF(2, tt)), t), subF(1, this.tension));
      const f4 = mulF(subF(ttt, tt), subF(1, this.tension));
      const q1 = new Float32Array([segment.p1.zoom, segment.p1.rx, segment.p1.ry, segment.p1.rz]);
      const q2 = new Float32Array([segment.p2.zoom, segment.p2.rx, segment.p2.ry, segment.p2.rz]);
      const quaternion = quaternionLerp(q1, q2, t);
      quaternionToMatrix(quaternion, matrix);
      const m1 = quaternionToMatrix(q1), m2 = quaternionToMatrix(q2);
      matrix[12] = addF(addF(addF(mulF(segment.p1.px, f1), mulF(segment.p2.px, f2)),
        mulF(m1[8], f3)), mulF(m2[8], f4));
      matrix[13] = addF(addF(addF(mulF(segment.p1.py, f1), mulF(segment.p2.py, f2)),
        mulF(m1[9], f3)), mulF(m2[9], f4));
      matrix[14] = addF(addF(addF(mulF(segment.p1.pz, f1), mulF(segment.p2.pz, f2)),
        mulF(m1[10], f3)), mulF(m2[10], f4));
      matrix[15] = 1;
      return { matrix, zoom: 1 };
    }
    const interpolated = hermite(segment.p0, segment.p1, segment.p2, segment.p3,
      segment.fade, this.tension, this.continuity, 0, Boolean(this.uniform), this.mode === 3);
    key = interpolated.value; keyDerivative = interpolated.derivative;
    switch (this.mode) {
      case 1:
        initDirection([this.target[0] - key.px, this.target[1] - key.py, this.target[2] - key.pz], matrix);
        break;
      case 2:
        initDirection([key.rx - key.px, key.ry - key.py, key.rz - key.pz], matrix);
        break;
      case 3:
        initDirection([keyDerivative.px, keyDerivative.py, keyDerivative.pz], matrix);
        break;
      case 5: {
        const mx = initEuler(key.rx * TWO_PI, 0, 0);
        const my = initEuler(0, key.ry * TWO_PI, 0);
        const mz = initEuler(0, 0, key.rz * TWO_PI);
        mulRotation(mulRotation(mz, mx), my, matrix);
        break;
      }
      default:
        initEulerTurns(key.rx, key.ry, key.rz, matrix);
        break;
    }
    matrix[12] = key.px; matrix[13] = key.py; matrix[14] = key.pz; matrix[15] = 1;
    return { matrix, zoom: key.zoom };
  }

  evaluate(time, phase, matrix) { return this.eval(time, phase, matrix); }

  normalize() {
    this.uniform = 1;
    this.keys.sort((a, b) => a.time - b.time);
    const distances = new Float32Array(Math.max(0, this.keys.length - 1));
    let distance = 0;
    let previous = this.eval(this.keys[0].time).matrix.slice(12, 15);
    for (let i = 0; i < this.keys.length - 1; i++) {
      for (let j = 1; j <= 16; j++) {
        const time = this.keys[i].time + (this.keys[i + 1].time - this.keys[i].time) * (j / 16);
        const current = this.eval(time).matrix.slice(12, 15);
        const dx = current[0] - previous[0], dy = current[1] - previous[1], dz = current[2] - previous[2];
        distance = f32(distance + Math.sqrt(dx * dx + dy * dy + dz * dz));
        previous = current;
      }
      distances[i] = distance;
    }
    if (distance === 0) distance = 1;
    this.keys[0].time = 0;
    for (let i = 1; i < this.keys.length; i++) this.keys[i].time = f32(distances[i - 1] / distance);
    return distance;
  }

  summary() {
    return {
      classId: this.classId, type: this.type, mode: this.mode, version: this.version,
      uniform: this.uniform, keyCount: this.keys.length,
      hash: hashFloatRecords(this.keys, ['time', 'px', 'py', 'pz', 'rx', 'ry', 'rz', 'zoom']),
    };
  }
}

function parsePipeBlob(blob) {
  const bytes = asBytes(blob);
  if (bytes.byteLength < 64) throw new Error(`PipeSpline blob is only ${bytes.byteLength} bytes`);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const count = view.getInt32(0, true);
  if (count < 1 || 64 + count * 32 > bytes.byteLength) {
    throw new Error(`invalid PipeSpline key count ${count} for ${bytes.byteLength} bytes`);
  }
  const pipe = {
    count, version: view.getInt32(4, true), mode: view.getInt32(8, true),
    tension: view.getFloat32(12, true),
    start: [view.getFloat32(16, true), view.getFloat32(20, true), view.getFloat32(24, true)],
    rotation: view.getFloat32(28, true), keys: [],
  };
  for (let i = 0; i < count; i++) {
    const at = 64 + i * 32;
    pipe.keys.push({
      x: view.getFloat32(at, true), y: view.getFloat32(at + 4, true),
      z: view.getFloat32(at + 8, true), radius: view.getFloat32(at + 12, true),
      flags: view.getInt32(at + 16, true),
    });
  }
  return pipe;
}

function pipeToSpline(pipe) {
  let matrix = initEuler(0, 0, pipe.rotation * TWO_PI);
  let oldRadius = 0;
  let current = new Float32Array(pipe.start);
  const keys = [];
  for (let i = 0; i < pipe.keys.length; i++) {
    const previous = current;
    const pipeKey = pipe.keys[i];
    current = new Float32Array([pipeKey.x, pipeKey.y, pipeKey.z]);
    const direction = normalize3([current[0] - previous[0], current[1] - previous[1], current[2] - previous[2]]);
    let ac = matrix[8] * direction[0] + matrix[9] * direction[1] + matrix[10] * direction[2];
    ac = clamp(ac, 1, -1);
    const angle = Math.acos(ac);
    const axis = cross3(matrix.subarray(8, 11), direction);
    matrix = mulRotation(matrix, initAxisAngle(axis, angle));
    const quaternion = quaternionFromMatrix(matrix);
    const oldPoint = [previous[0] + direction[0] * oldRadius,
      previous[1] + direction[1] * oldRadius, previous[2] + direction[2] * oldRadius];
    keys.push({ select: 0, time: i, px: oldPoint[0], py: oldPoint[1], pz: oldPoint[2],
      rx: quaternion[1], ry: quaternion[2], rz: quaternion[3], zoom: quaternion[0] });
    oldRadius = pipeKey.radius;
    const newPoint = [current[0] - direction[0] * oldRadius,
      current[1] - direction[1] * oldRadius, current[2] - direction[2] * oldRadius];
    keys.push({ select: 0, time: i + 0.5, px: newPoint[0], py: newPoint[1], pz: newPoint[2],
      rx: quaternion[1], ry: quaternion[2], rz: quaternion[3], zoom: quaternion[0] });
  }
  const result = new BlobSplinePath({ mode: 4, tension: pipe.tension, keys, pipe: cloneState(pipe) });
  result.normalize();
  return result;
}

let perlinPermutation = null;
function makePerlinPermutation() {
  if (perlinPermutation) return perlinPermutation;
  const random = new Random();
  random.setSeed(1);
  const ranks = new Uint32Array(256), permutation = new Uint8Array(512);
  for (let i = 0; i < 256; i++) { ranks[i] = random.uint32() % 0x10000; permutation[i] = i; }
  for (let i = 0; i < 255; i++) {
    for (let j = i + 1; j < 256; j++) {
      if (ranks[i] > ranks[j]) {
        const rank = ranks[i]; ranks[i] = ranks[j]; ranks[j] = rank;
        const value = permutation[i]; permutation[i] = permutation[j]; permutation[j] = value;
      }
    }
  }
  permutation.copyWithin(256, 0, 256);
  perlinPermutation = permutation;
  return permutation;
}

const PERLIN_GRADIENTS = new Float32Array([
  1,1,0,0, -1,1,0,0, 1,-1,0,0, -1,-1,0,0,
  1,0,1,0, -1,0,1,0, 1,0,-1,0, -1,0,-1,0,
  0,1,1,0, 0,-1,1,0, 0,1,-1,0, 0,-1,-1,0,
  1,1,0,0, -1,1,0,0, 0,-1,1,0, 0,-1,-1,0,
]);

function perlin3D(position, out = vector()) {
  const permutation = makePerlinPermutation();
  const integer = new Int32Array(3), fade = new Float32Array(3);
  for (let j = 0; j < 3; j++) {
    // sPerlin3D receives an sVector and runs under the player's x87 24-bit
    // precision mode. Round the Scale3 result and every polynomial operation,
    // rather than letting JavaScript retain binary64 intermediates.
    const coordinate = f32(position[j]);
    const lattice = roundEven(subF(coordinate, 0.5));
    integer[j] = lattice & 255;
    const fraction = clamp(subF(coordinate, lattice), 1, 0);
    const fraction2 = mulF(fraction, fraction);
    const fraction3 = mulF(fraction2, fraction);
    const polynomial = addF(10, mulF(fraction, subF(mulF(6, fraction), 15)));
    fade[j] = mulF(fraction3, polynomial);
  }
  const gradient = (x, y, z) => {
    const index = permutation[permutation[permutation[x] + y] + z] & 15;
    return PERLIN_GRADIENTS.subarray(index * 4, index * 4 + 4);
  };
  const lerp3 = (a, b, t) => vector(
    addF(a[0], mulF(subF(b[0], a[0]), t)),
    addF(a[1], mulF(subF(b[1], a[1]), t)),
    addF(a[2], mulF(subF(b[2], a[2]), t)),
    0,
  );
  let t0 = lerp3(gradient(integer[0], integer[1], integer[2]), gradient(integer[0] + 1, integer[1], integer[2]), fade[0]);
  let t1 = lerp3(gradient(integer[0], integer[1] + 1, integer[2]), gradient(integer[0] + 1, integer[1] + 1, integer[2]), fade[0]);
  t0 = lerp3(t0, t1, fade[1]);
  t1 = lerp3(gradient(integer[0], integer[1], integer[2] + 1), gradient(integer[0] + 1, integer[1], integer[2] + 1), fade[0]);
  let t2 = lerp3(gradient(integer[0], integer[1] + 1, integer[2] + 1), gradient(integer[0] + 1, integer[1] + 1, integer[2] + 1), fade[0]);
  t1 = lerp3(t1, t2, fade[1]);
  out.set(lerp3(t0, t1, fade[2]));
  return out;
}

class ShakerSpline {
  constructor(runtime, op, parent, amplitude, parameters) {
    this.classId = KC_SPLINE;
    this.type = 'ShakerSpline';
    this.runtime = runtime;
    this.op = op;
    this.parent = parent || null;
    this.amplitude = amplitude || null;
    this.mode = parameters[0] >>> 0;
    this.translateAmp = new Float32Array(parameters.slice(1, 4));
    this.rotateAmp = new Float32Array(parameters.slice(4, 7));
    this.translateFreq = new Float32Array(parameters.slice(7, 10));
    this.rotateFreq = new Float32Array(parameters.slice(10, 13));
    this.center = new Float32Array(parameters.slice(14, 17));
  }

  eval(time2, phase = 0, matrix = new Float32Array(16), environment = this.runtime.environment) {
    let zoom = 1, amplitude = 1;
    if (this.amplitude) amplitude = this.amplitude.eval(time2, phase, new Float32Array(16), environment).zoom;
    if (this.parent) {
      const parent = this.parent.eval(time2, phase, matrix, environment);
      matrix = parent.matrix; zoom = parent.zoom;
    } else identityMatrix(matrix);

    let event = this.op.firstEvent;
    let fakeMode = !event;
    while (event || fakeMode) {
      let time = fakeMode
        ? environment.vars[KV_TIME][0]
        : f32((event.end - environment.beatTime) / (event.end - event.start));
      if (time > 0 && time < 1) {
        let ramp;
        switch (this.mode & 15) {
          case 1: ramp = time; time = subF(1, time); break;
          case 2: ramp = f32(Math.sin(mulF(time, f32(Math.PI)))); break;
          case 3: ramp = 1; break;
          default: ramp = subF(1, time); break;
        }
        ramp = mulF(ramp, amplitude);
        let tx, ty, tz, rx, ry, rz;
        if (this.mode & 32) {
          const translation = perlin3D([
            mulF(this.translateFreq[0], time), mulF(this.translateFreq[1], time),
            mulF(this.translateFreq[2], time),
          ]);
          const rotation = perlin3D([
            mulF(this.rotateFreq[0], time), mulF(this.rotateFreq[1], time),
            mulF(this.rotateFreq[2], time),
          ]);
          [tx, ty, tz] = translation; [rx, ry, rz] = rotation;
        } else {
          const wave = frequency => Math.sin(mulF(mulF(time, frequency), TWO_PI));
          tx = mulF(mulF(wave(this.translateFreq[0]), this.translateAmp[0]), ramp);
          ty = mulF(mulF(wave(this.translateFreq[1]), this.translateAmp[1]), ramp);
          tz = mulF(mulF(wave(this.translateFreq[2]), this.translateAmp[2]), ramp);
          rx = mulF(mulF(wave(this.rotateFreq[0]), this.rotateAmp[0]), ramp);
          ry = mulF(mulF(wave(this.rotateFreq[1]), this.rotateAmp[1]), ramp);
          rz = mulF(mulF(wave(this.rotateFreq[2]), this.rotateAmp[2]), ramp);
        }
        // These repeated amplitude/ramp multiplies (and tz's y amplitude) are
        // intentional source behavior in GenSplineShaker::Eval.
        tx = mulF(mulF(tx, this.translateAmp[0]), ramp);
        ty = mulF(mulF(ty, this.translateAmp[1]), ramp);
        tz = mulF(mulF(tz, this.translateAmp[1]), ramp);
        rx = mulF(mulF(rx, this.rotateAmp[0]), ramp);
        ry = mulF(mulF(ry, this.rotateAmp[1]), ramp);
        rz = mulF(mulF(rz, this.rotateAmp[2]), ramp);
        const shake = initEuler(rx, ry, rz);
        shake[12] = tx; shake[13] = ty; shake[14] = tz;
        if (this.mode & 16) {
          matrix[12] -= this.center[0]; matrix[13] -= this.center[1]; matrix[14] -= this.center[2];
          mulAffine(matrix, shake, matrix);
          matrix[12] += this.center[0]; matrix[13] += this.center[1]; matrix[14] += this.center[2];
        } else mulAffine(shake, matrix, matrix);
      }
      if (fakeMode) fakeMode = false;
      else event = event.nextOp;
    }
    return { matrix, zoom };
  }

  evaluate(time, phase, matrix, environment) { return this.eval(time, phase, matrix, environment); }
  summary() { return { classId: this.classId, type: this.type, mode: this.mode, opId: this.op.id }; }
}

class Event {
  constructor(runtime, raw = {}, id = -1) {
    this.runtime = runtime;
    this.id = id;
    this.op = null;
    this.nextOp = null;
    this.start = raw.start | 0;
    this.end = raw.end | 0;
    this.velocity = f32(raw.velocity ?? 0);
    this.modulation = f32(raw.modulation ?? 0);
    this.select = raw.select | 0;
    this.scale = new Float32Array([...(raw.scale || [0, 0, 0]), 0].slice(0, 4));
    this.rotate = new Float32Array([...(raw.rotate || [0, 0, 0]), 0].slice(0, 4));
    this.translate = new Float32Array([...(raw.translate || [0, 0, 0]), 0].slice(0, 4));
    this.color = (raw.color ?? 0) >>> 0;
    this.spline = null;
    this.startInterval = f32(raw.startInterval ?? 0);
    this.endInterval = f32(raw.endInterval ?? 1);
    this.flags = raw.flags | 0;
    this.matrix = identityMatrix();
    this.instances = new InstanceChain();
    this.dynamic = Boolean(raw.dynamic);
    this.removeMe = false;
  }

  stop() { this.start = 0; this.end = 1; }
  summary() {
    return {
      id: this.id, opId: this.op ? this.op.id : null, start: this.start, end: this.end,
      velocity: this.velocity, modulation: this.modulation, select: this.select,
      splineId: this.spline ? this.spline.id : null, flags: this.flags,
      instanceCount: this.instances.items.length,
    };
  }
}

class Op {
  constructor(runtime, raw, documentClass) {
    this.runtime = runtime;
    this.id = raw.id | 0;
    this.classIndex = raw.classIndex | 0;
    this.classId = raw.classId | 0;
    this.documentClass = documentClass;
    this.classInfo = registryGet(runtime.registry, this.classId);
    this.convention = documentClass.convention >>> 0;
    this.inputSlots = inputSlots(this.convention);
    this.linkSlots = linkSlots(this.convention);
    this.dataWords = dataWords(this.convention);
    this.stringSlots = stringSlots(this.convention);
    this.splineSlots = splineSlots(this.convention);
    this.outputClassId = outputClassId(this.classInfo);
    this.parameters = (raw.parameters || []).slice();
    this.strings = (raw.strings || []).slice();
    this.blob = raw.blob ? asBytes(raw.blob) : new Uint8Array();
    this.animation = raw.animation ? asBytes(raw.animation) : new Uint8Array([KA.END]);
    this.inputs = [];
    this.links = [];
    this.splines = [];
    this.cache = null;
    this.firstEvent = null;
    this.instanceState = {};
    this.changed = true;
    this.calcError = false;
    this._calcState = 0;
    this._executing = false;

    this._editBuffer = new ArrayBuffer(this.dataWords * 4);
    this._animBuffer = new ArrayBuffer(this.dataWords * 4);
    this.editBits = new Uint32Array(this._editBuffer);
    this.editInts = new Int32Array(this._editBuffer);
    this.editFloats = new Float32Array(this._editBuffer);
    this.animBits = new Uint32Array(this._animBuffer);
    this.animInts = new Int32Array(this._animBuffer);
    this.animFloats = new Float32Array(this._animBuffer);
    this.animBytes = new Uint8Array(this._animBuffer);
    const packing = documentClass.packing || '';
    for (let i = 0; i < this.dataWords; i++) {
      const value = this.parameters[i] ?? 0;
      const kind = packing[i] || '-';
      if (parameterIsFloat(kind)) this.editFloats[i] = f32(value);
      else if (parameterIsSigned(kind)) this.editInts[i] = value | 0;
      else this.editBits[i] = value >>> 0;
    }
    this.animBits.set(this.editBits);
    this.animParameters = new Array(this.dataWords);
    this.syncAnimParameters();
  }

  syncAnimParameters() {
    const packing = this.documentClass.packing || '';
    for (let i = 0; i < this.dataWords; i++) {
      const kind = packing[i] || '-';
      this.animParameters[i] = parameterIsFloat(kind)
        ? this.animFloats[i]
        : parameterIsSigned(kind) ? this.animInts[i] : this.animBits[i] >>> 0;
    }
    return this.animParameters;
  }

  callRecord(environment = this.runtime.environment) {
    return this.runtime.makeCallRecord(this, environment);
  }

  precalc(environment = this.runtime.environment) {
    return this.runtime._precalcOp(this, environment);
  }

  exec(environment = this.runtime.environment) {
    if (this.outputClassId === KC_BITMAP) return;
    if (this._executing) throw new Error(`operator execution cycle at op ${this.id}`);
    this._executing = true;
    const popCount = environment.executeAnimation(this, this.animation);
    try {
      return this.runtime._invokeHandler('exec', this, environment);
    } finally {
      environment.pop(popCount);
      this._executing = false;
    }
  }

  execInputs(environment = this.runtime.environment) {
    for (const input of this.inputs) {
      if (!input) throw new Error(`null input while executing op ${this.id}`);
      input.exec(environment);
    }
    if (!(this.convention & OPC_DONTCALLLINK)) {
      for (const link of this.links) if (link) link.exec(environment);
    }
  }

  execInput(environment, index) {
    const input = this.inputs[index];
    if (input) return input.exec(environment);
  }

  execEvent(environment, event) {
    const savedVariables = environment.vars.slice(0, 8).map(copyVector);
    const oldSpline = environment.eventSpline;
    const oldEvent = environment.currentEvent;
    let time = 0;
    if (event.start < event.end) time = f32((environment.beatTime - event.start) / (event.end - event.start));
    time = clamp(time, 1, 0);
    time = f32(event.startInterval + time * (event.endInterval - event.startInterval));
    if (!(event.flags & KEF_NOTIME)) setScalar(environment.vars[0], time);
    setScalar(environment.vars[1], event.velocity);
    setScalar(environment.vars[2], event.modulation);
    setScalar(environment.vars[3], event.select);
    environment.vars[4].set(event.scale);
    environment.vars[5].set(event.rotate);
    environment.vars[6].set(event.translate);
    const color = event.color >>> 0;
    environment.vars[7].set([
      ((color >>> 16) & 255) / 255, ((color >>> 8) & 255) / 255,
      (color & 255) / 255, ((color >>> 24) & 255) / 255,
    ]);
    if (event.spline) environment.eventSpline = event.spline;
    environment.currentEvent = event;
    environment.matrixStack.pushMul(event.matrix);
    try {
      return environment.withInstanceChain(event.instances, () => this.exec(environment));
    } finally {
      environment.matrixStack.pop();
      environment.eventSpline = oldSpline;
      environment.currentEvent = oldEvent;
      for (let i = 0; i < 8; i++) environment.vars[i].set(savedVariables[i]);
    }
  }

  summary() {
    let cache = null;
    if (this.cache) cache = typeof this.cache.summary === 'function'
      ? this.cache.summary() : { classId: this.cache.classId ?? null, type: this.cache.type || this.cache.constructor?.name };
    return {
      id: this.id, classId: this.classId, name: this.classInfo?.name || null,
      inputIds: this.inputs.map(input => input ? input.id : null),
      linkIds: this.links.map(link => link ? link.id : null),
      animParameters: this.syncAnimParameters().slice(), changed: this.changed,
      calcError: this.calcError, cache,
    };
  }
}

class Environment {
  constructor(runtime) {
    this.runtime = runtime;
    this.vars = Array.from({ length: MAX_VARS }, () => vector());
    this.Var = this.vars;
    this._varSaves = [];
    this.eventSpline = null;
    this.currentEvent = null;
    this.dynamicEvents = [];
    this.eventOpsCleanup = [];
    this.defaultInstances = new InstanceChain();
    this._instanceChain = this.defaultInstances;
    this._instanceCursor = 0;
    this.clearInstanceMemory = false;
    this.matrixStack = new MatrixStack();
    this.execStack = this.matrixStack;
    this.beatTime = 0;
    this.currentTime = 0;
    this.lastTime = 0;
    this.timeDelta = 0;
    this.timeSlices = 0;
    this.timeJitter = 0;
    this.timeReset = true;
    this.aspect = 1;
    this.markers = Array.from({ length: 32 }, () => identityMatrix());
    this.frame = null;
    // Exec_Misc_Demo renders every KC_IPP input to the master target in
    // input order.  lastOutput is only the scratch result of the current IPP
    // branch; keep the committed branch roots separately for the renderer.
    this.frameOutputs = [];
    // GenOverlayManager::Find() reuses an operator's render target within one
    // root IPP graph. Exec_Misc_Demo/Reset clears that ownership for each root.
    this.ippOutputs = new Map();
    this.renderPassAdjust = 0;
    this.initFrame(0, 0);
  }

  initView() {
    for (const event of this.dynamicEvents) event.instances.clear();
    this.dynamicEvents.length = 0;
    this.clearInstanceMemory = true;
    this.defaultInstances.clear();
    this._instanceChain = this.defaultInstances;
    this._instanceCursor = 0;
    this.aspect = 1;
    for (const marker of this.markers) identityMatrix(marker);
    this.beatTime = this.currentTime = this.lastTime = 0;
    this.timeDelta = this.timeSlices = this.timeJitter = 0;
    this.timeReset = true;
  }

  initFrame(beatTime, timeMilliseconds) {
    for (const item of this.vars) item.fill(0);
    setScalar(this.vars[0], f32((beatTime | 0) / 65536));
    this.vars[4].set([1, 1, 1, 0]);
    this._varSaves.length = 0;
    this.matrixStack.popAll();
    this.frame = null;
    this.lastOutput = null;
    this.frameOutputs.length = 0;
    this.ippOutputs.clear();
    this.renderPassAdjust = 0;
    this.eventSpline = null;
    this.currentEvent = null;
    this.gameCamera = defaultCameraEnvironment();
    this.currentCamera = defaultCameraEnvironment();
    this.beatTime = beatTime | 0;
    this.lastTime = this.currentTime | 0;
    this.currentTime = timeMilliseconds | 0;
    this.timeDelta = (this.currentTime - this.lastTime) | 0;
    if (this.timeDelta < 0) this.timeReset = true;
    else {
      this.timeSlices = (Math.trunc(this.currentTime / 10) - Math.trunc(this.lastTime / 10)) | 0;
      this.timeJitter = this.currentTime % 10;
      this.timeReset = false;
    }
    if (this.timeReset) this.timeDelta = this.timeSlices = this.timeJitter = 0;
    if (this.eventOpsCleanup.length) throw new Error('initFrame called before exitFrame');
    for (const event of this.dynamicEvents) this.addStaticEvent(event);
    this._instanceChain = this.defaultInstances;
    this._instanceCursor = 0;
    return this;
  }

  exitFrame() {
    for (const op of this.eventOpsCleanup) op.firstEvent = null;
    this.eventOpsCleanup.length = 0;
    for (let i = 0; i < this.dynamicEvents.length;) {
      const event = this.dynamicEvents[i];
      if (event.end !== event.start && this.beatTime >= event.end) {
        event.instances.clear();
        this.dynamicEvents[i] = this.dynamicEvents[this.dynamicEvents.length - 1];
        this.dynamicEvents.pop();
      } else i++;
    }
    this.clearInstanceMemory = false;
    return this;
  }

  addDynamicEvent(event) {
    event.dynamic = true;
    this.dynamicEvents.push(event);
    return event;
  }

  addStaticEvent(event) {
    if (event.start === event.end || (this.beatTime >= event.start && this.beatTime < event.end)) {
      if (this.clearInstanceMemory) event.instances.clear();
      if (!event.op.firstEvent) this.eventOpsCleanup.push(event.op);
      event.nextOp = event.op.firstEvent;
      event.op.firstEvent = event;
      return true;
    }
    return false;
  }

  withInstanceChain(chain, callback) {
    const previousChain = this._instanceChain, previousCursor = this._instanceCursor;
    this._instanceChain = chain;
    this._instanceCursor = 0;
    try { return callback(); }
    finally { this._instanceChain = previousChain; this._instanceCursor = previousCursor; }
  }

  getInstance(op, factory = () => ({})) {
    const index = this._instanceCursor++;
    let instance = this._instanceChain.items[index];
    if (instance) {
      if ((instance.opId | 0) !== op.id) {
        throw new Error(`instance chain mismatch: expected op ${instance.opId}, got ${op.id}`);
      }
      instance.reset = false;
      return instance;
    }
    instance = factory() || {};
    if (typeof instance !== 'object') throw new TypeError('instance factory must return an object');
    instance.opId = op.id;
    instance.reset = true;
    this._instanceChain.items.push(instance);
    return instance;
  }

  pop(count) {
    while (count-- > 0) {
      const saved = this._varSaves.pop();
      if (!saved) throw new Error('animation variable-save stack underflow');
      this.vars[saved.index].set(saved.value);
    }
  }

  executeAnimation(op, bytecode = op.animation) {
    const bytes = asBytes(bytecode);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const stack = [];
    let cursor = 0, popCount = 0, changed = false;
    const require = count => {
      if (cursor + count > bytes.byteLength) throw new Error(`truncated animation for op ${op.id}`);
    };
    const readU8 = () => { require(1); return bytes[cursor++]; };
    const readU16 = () => { require(2); const value = view.getUint16(cursor, true); cursor += 2; return value; };
    const readF32 = () => { require(4); const value = view.getFloat32(cursor, true); cursor += 4; return value; };
    const top = () => {
      if (!stack.length) throw new Error(`animation stack underflow at op ${op.id}`);
      return stack[stack.length - 1];
    };
    const pop = () => {
      if (!stack.length) throw new Error(`animation stack underflow at op ${op.id}`);
      return stack.pop();
    };
    for (;;) {
      const command = readU8();
      const base = command >= 0x80 ? command & 0xf0 : command;
      switch (base) {
        case KA.NOP: break;
        case KA.END:
          if (stack.length) throw new Error(`animation stack is not empty at op ${op.id}`);
          if (changed) op.changed = true;
          op.syncAnimParameters();
          return popCount;
        case KA.LOADVAR: {
          const index = readU8();
          if (index >= MAX_VARS) throw new Error(`animation variable ${index} outside range`);
          stack.push(copyVector(this.vars[index]));
          break;
        }
        case KA.LOADPARA1:
        case KA.LOADPARA2:
        case KA.LOADPARA3:
        case KA.LOADPARA4: {
          const index = readU8(), count = base - KA.LOADPARA1 + 1;
          if (index + count > op.dataWords) throw new Error(`animation parameter ${index}+${count} outside op ${op.id}`);
          const value = vector();
          if (count === 1) setScalar(value, op.editFloats[index]);
          else for (let i = 0; i < count; i++) value[i] = op.editFloats[index + i];
          stack.push(value);
          break;
        }
        case KA.SWIZZLEX: setScalar(top(), top()[0]); break;
        case KA.SWIZZLEY: setScalar(top(), top()[1]); break;
        case KA.SWIZZLEZ: setScalar(top(), top()[2]); break;
        case KA.SWIZZLEW: setScalar(top(), top()[3]); break;
        case KA.ADD:
        case KA.SUB:
        case KA.MUL:
        case KA.DIV:
        case KA.MOD:
        case KA.POW: {
          const right = pop(), left = top();
          for (let i = 0; i < 4; i++) {
            if (base === KA.ADD) left[i] = f32(left[i] + right[i]);
            else if (base === KA.SUB) left[i] = f32(left[i] - right[i]);
            else if (base === KA.MUL) left[i] = f32(left[i] * right[i]);
            else if (base === KA.DIV) left[i] = f32(left[i] / right[i]);
            else if (base === KA.MOD) left[i] = f32(left[i] % right[i]);
            else left[i] = f32(Math.pow(left[i], right[i]));
          }
          break;
        }
        case KA.NEG: for (let i = 0; i < 4; i++) top()[i] = f32(-top()[i]); break;
        case KA.INVERT: for (let i = 0; i < 4; i++) top()[i] = f32(1 - top()[i]); break;
        case KA.SIN: setScalar(top(), Math.sin(top()[0] * TWO_PI)); break;
        case KA.COS: setScalar(top(), Math.cos(top()[0] * TWO_PI)); break;
        case KA.PULSE: {
          let value = top()[0] % 1; if (value < 0) value += 1;
          setScalar(top(), value < 0.5 ? 1 : 0); break;
        }
        case KA.RAMP: {
          let value = top()[0] % 1; if (value < 0) value += 1;
          setScalar(top(), value); break;
        }
        case KA.CONSTV: {
          const value = vector(readF32(), readF32(), readF32(), readF32());
          stack.push(value); break;
        }
        case KA.CONSTC:
          stack.push(vector(readU8() / 255, readU8() / 255, readU8() / 255, readU8() / 255));
          break;
        case KA.CONSTS: stack.push(setScalar(vector(), readF32())); break;
        case KA.SPLINE: {
          const index = readU16(), value = top()[0], spline = this.runtime.splines[index];
          top().fill(0); if (!spline) throw new Error(`animation spline ${index} outside document`);
          spline.eval(value, top()); break;
        }
        case KA.EVENTSPLINE: {
          const value = top()[0]; top().fill(0);
          if (this.eventSpline) this.eventSpline.eval(value, top());
          break;
        }
        case KA.MATRIX: {
          const matrix = new Float32Array(16);
          for (let i = 0; i < 4; i++) matrix.set(this.vars[KV_MATRIX_I + i], i * 4);
          rotateVector4(matrix, top(), top()); break;
        }
        case KA.NOISE: {
          let value = top()[0] % 1; if (value < 0) value += 1;
          const integer = Math.trunc(value * 65536) | 0, permutation = makePerlinPermutation();
          const result = permutation[(integer >>> 8) & 255] | (permutation[integer & 255] << 8);
          setScalar(top(), result / 65535); break;
        }
        case KA.EASE: {
          let easeIn = readF32(), easeOut = readF32();
          if (easeIn > 0.5) easeIn = 0.5; if (easeOut > 0.5) easeOut = 0.5;
          easeOut = 1 - easeOut;
          let distance = easeOut - easeIn;
          if (distance < 0.1) {
            easeOut += (0.1 - distance) / 2; easeIn -= (0.1 - distance) / 2;
            distance = easeOut - easeIn;
          }
          const scale = 1 / distance;
          const value = top();
          for (let i = 0; i < 4; i++) {
            let x = value[i], a, c;
            if (x < 0) x = -easeIn * scale / 2;
            else if (x < easeIn) { a = -easeIn * scale / 2; c = scale / (2 * easeIn); x = a + x * x * c; }
            else if (x < easeOut) x = (x - easeIn) * scale;
            else if (x < 1) { a = -(1 - easeOut) * scale / 2; c = scale / (2 * (1 - easeOut)); x = 1 - a - (1 - x) * (1 - x) * c; }
            else x = 1 + (1 - easeOut) * scale / 2;
            value[i] = f32((x + easeIn * scale / 2) /
              (1 + easeIn * scale / 2 + (1 - easeOut) * scale / 2));
          }
          break;
        }
        case KA.TIMESHIFT: {
          const t0 = readU8() / 256, t1 = readU8() / 256, value = top();
          for (let i = 0; i < 4; i++) value[i] = value[i] <= t0 ? 0 : value[i] >= t1 ? 1 : f32((value[i] - t0) / (t1 - t0));
          break;
        }
        case KA.LOG2:
          // This multiplier is intentionally ln(2), exactly as in kdoc.cpp.
          for (let i = 0; i < 4; i++) top()[i] = f32(Math.log(top()[i]) * 0.6931471805599453);
          break;
        case KA.POW2: for (let i = 0; i < 4; i++) top()[i] = f32(Math.pow(2, top()[i])); break;
        case KA.STOREVAR: {
          const index = readU8(), value = pop();
          if (index >= MAX_VARS || this._varSaves.length >= MAX_VAR_SAVES) throw new Error('animation variable save outside range');
          this._varSaves.push({ index, value: copyVector(this.vars[index]) });
          popCount++;
          for (let i = 0; i < 4; i++) if (command & (1 << i)) this.vars[index][i] = value[i];
          break;
        }
        case KA.STOREPARAFLOAT:
        case KA.CHANGEPARAFLOAT:
        case KA.STOREPARAINT:
        case KA.CHANGEPARAINT:
        case KA.STOREPARABYTE:
        case KA.CHANGEPARABYTE: {
          const index = readU8(), value = pop();
          let didChange = false;
          for (let i = 0; i < 4; i++) {
            if (!(command & (1 << i))) continue;
            if (base === KA.STOREPARABYTE || base === KA.CHANGEPARABYTE) {
              const at = index * 4 + i;
              if (at >= op.animBytes.length) throw new Error(`animation byte parameter ${index}:${i} outside op ${op.id}`);
              const next = clamp(truncInt(value[i] * 255), 255, 0);
              if (op.animBytes[at] !== next) didChange = true;
              op.animBytes[at] = next;
            } else {
              const at = index + i;
              if (at >= op.dataWords) throw new Error(`animation parameter ${at} outside op ${op.id}`);
              const next = base === KA.STOREPARAINT || base === KA.CHANGEPARAINT ? truncInt(value[i]) : f32(value[i]);
              const old = base === KA.STOREPARAINT || base === KA.CHANGEPARAINT ? op.animInts[at] : op.animFloats[at];
              if (old !== next) didChange = true;
              if (base === KA.STOREPARAINT || base === KA.CHANGEPARAINT) op.animInts[at] = next;
              else op.animFloats[at] = next;
            }
          }
          if (base === KA.CHANGEPARAFLOAT || base === KA.CHANGEPARAINT || base === KA.CHANGEPARABYTE) changed ||= didChange;
          break;
        }
        default: throw new Error(`unknown animation opcode 0x${command.toString(16)} at op ${op.id}`);
      }
      if (stack.length > 256) throw new Error(`animation stack overflow at op ${op.id}`);
    }
  }

  summary() {
    return {
      beatTime: this.beatTime, currentTime: this.currentTime, lastTime: this.lastTime,
      timeDelta: this.timeDelta, timeSlices: this.timeSlices,
      timeJitter: this.timeJitter, timeReset: this.timeReset,
      variables: this.vars.map(value => Array.from(value)),
      defaultInstances: this.defaultInstances.summary(),
      dynamicEvents: this.dynamicEvents.map(event => event.summary()),
      matrixStack: this.matrixStack.summary(),
    };
  }
}

for (const [pascal, camel] of [
  ['BeatTime', 'beatTime'], ['CurrentTime', 'currentTime'], ['LastTime', 'lastTime'],
  ['TimeDelta', 'timeDelta'], ['TimeSlices', 'timeSlices'], ['TimeJitter', 'timeJitter'],
  ['TimeReset', 'timeReset'], ['EventSpline', 'eventSpline'], ['CurrentEvent', 'currentEvent'],
  ['Aspect', 'aspect'],
]) {
  Object.defineProperty(Environment.prototype, pascal, {
    get() { return this[camel]; }, set(value) { this[camel] = value; }, configurable: true,
  });
}

class MissingCache {
  constructor(op) {
    this.classId = op.outputClassId;
    this.type = 'MissingHandler';
    this.opId = op.id;
    this.operatorClassId = op.classId;
  }
  summary() { return { classId: this.classId, type: this.type, opId: this.opId, operatorClassId: this.operatorClassId }; }
}

class DemoObject {
  constructor() { this.classId = KC_DEMO; this.type = 'Demo'; }
  summary() { return { classId: this.classId, type: this.type }; }
}

class Runtime {
  constructor(document, options = {}) {
    if (!document || !Array.isArray(document.operations) || !Array.isArray(document.classes)) {
      throw new TypeError('Runtime expects a parsed KX document');
    }
    this.document = document;
    this.registry = options.registry || CLASS_REGISTRY;
    this.handlers = options.handlers || new Map();
    this.strictHandlers = options.strictHandlers !== false;
    this.onHandlerCall = options.onHandlerCall || null;
    this.handlerTraceLimit = Math.max(0, Math.floor(Number(options.handlerTraceLimit) || 0));
    this.handlerCallCount = 0;
    this.handlerCalls = [];
    this.currentRoot = options.rootIndex | 0;
    this.splines = document.splines.map((spline, index) => new Spline(spline, index));
    this.operations = document.operations.map(raw => {
      const documentClass = document.classes[raw.classIndex];
      if (!documentClass) throw new Error(`operator ${raw.id} has no document class ${raw.classIndex}`);
      const classInfo = registryGet(this.registry, raw.classId);
      if (classInfo && (classInfo.convention >>> 0) !== (documentClass.convention >>> 0)) {
        throw new Error(`class 0x${raw.classId.toString(16)} convention mismatch`);
      }
      return new Op(this, raw, documentClass);
    });
    for (let i = 0; i < this.operations.length; i++) {
      const raw = document.operations[i], op = this.operations[i];
      op.inputs = raw.inputs.map(index => this._operationAt(index, `input of op ${op.id}`));
      op.links = raw.links.map(index => index === null ? null : this._operationAt(index, `link of op ${op.id}`));
      op.splines = raw.splines.map(index => index === null ? null : this.splines[index]);
    }
    this.events = document.events.map((raw, index) => {
      const event = new Event(this, raw, index);
      event.op = this._operationAt(raw.operation, `event ${index}`);
      event.spline = raw.spline === null ? null : this.splines[raw.spline];
      return event;
    });
    this.roots = document.roots.map(index => index >= 0 && index < this.operations.length ? this.operations[index] : null);
    this.environment = options.environment || new Environment(this);
    this.environment.runtime = this;
    this.precalculated = false;
    this._snapshots = new Map();
  }

  _operationAt(index, what) {
    const op = this.operations[index];
    if (!op) throw new Error(`${what} refers to missing operator ${index}`);
    return op;
  }

  get root() { return this.roots[this.currentRoot] || null; }

  setRoot(index) {
    index |= 0;
    if (!this.roots[index]) throw new Error(`root ${index} is not present`);
    if (index !== this.currentRoot) {
      this.currentRoot = index;
      this.environment.initView();
      this.precalculated = false;
    }
    return this.root;
  }

  makeCallRecord(op, environment = this.environment) {
    const fixedInputs = Array.from({ length: op.inputSlots }, (_, index) => op.inputs[index]?.cache || null);
    const variableInputs = op.convention & OPC_VARIABLEINPUT
      ? op.inputs.map(input => input?.cache || null) : [];
    const inputs = op.convention & OPC_VARIABLEINPUT
      ? variableInputs
      : Array.from({ length: Math.max(op.inputSlots, op.inputs.length) }, (_, index) => op.inputs[index]?.cache || null);
    const links = Array.from({ length: op.linkSlots }, (_, index) => {
      const link = op.links[index] || null;
      return op.convention & OPC_DONTCALLLINK ? link : link?.cache || null;
    });
    op.syncAnimParameters();
    return {
      runtime: this, environment, op, inputs, links,
      parameters: op.animParameters.slice(), strings: op.strings.slice(), splines: op.splines.slice(),
      fixedInputs, variableInputs,
      callInputs: fixedInputs.concat(variableInputs),
    };
  }

  _invokeHandler(phase, op, environment) {
    const handler = handlerGet(this.handlers, op.classId);
    let callback = null;
    if (typeof handler === 'function') callback = phase === 'init' ? handler : null;
    else if (handler) callback = handler[phase] || null;
    if (!callback && phase === 'exec' && op.classInfo?.exec === 'Exec_Misc_Nop') callback = execMiscNop;
    const call = this.makeCallRecord(op, environment);
    const entry = {
      sequence: this.handlerCallCount++, phase, opId: op.id, classId: op.classId,
      handler: callback?.handlerName || callback?.name || null,
    };
    if (this.handlerTraceLimit > 0) {
      this.handlerCalls.push(entry);
      if (this.handlerCalls.length > this.handlerTraceLimit) this.handlerCalls.shift();
    }
    if (this.onHandlerCall) this.onHandlerCall(call, phase, callback, entry);
    if (!callback) {
      if (this.strictHandlers) throw new Error(`missing ${phase} handler for class 0x${op.classId.toString(16)} at op ${op.id}`);
      return phase === 'init' ? new MissingCache(op) : undefined;
    }
    return callback(call);
  }

  _precalcOp(op, environment) {
    if (op._calcState === 2 && !op.changed) return op.cache;
    if (op._calcState === 1) throw new Error(`operator calculation cycle at op ${op.id}`);
    op._calcState = 1;
    if (op.changed || !op.cache) op.animBits.set(op.editBits);
    const savedVariables = environment._varSaves.length;
    try {
      environment.executeAnimation(op, op.animation);
      for (const input of op.inputs) input.precalc(environment);
      if (!(op.convention & OPC_DONTCALLLINK)) for (const link of op.links) if (link) link.precalc(environment);
      op.cache = this._invokeHandler('init', op, environment);
      if (op.cache && typeof op.cache === 'object' && Object.isExtensible(op.cache)) {
        if (op.cache.classId === undefined) op.cache.classId = op.outputClassId;
        if (op.cache.outputClass === undefined && op.classInfo?.outputClass) {
          op.cache.outputClass = op.classInfo.outputClass;
        }
      }
      op.calcError = op.cache == null;
      op.changed = false;
      op._calcState = 2;
      return op.cache;
    } catch (error) {
      op._calcState = 0;
      throw error;
    } finally {
      environment.pop(environment._varSaves.length - savedVariables);
    }
  }

  precalc(rootIndex = this.currentRoot) {
    if (rootIndex !== this.currentRoot) this.setRoot(rootIndex);
    if (!this.root) throw new Error(`root ${this.currentRoot} is not present`);
    this.handlerCallCount = 0;
    this.handlerCalls.length = 0;
    this.environment.initFrame(0, 0);
    try {
      this.root.precalc(this.environment);
      this.precalculated = true;
      return this.root.cache;
    } finally {
      this.environment.exitFrame();
    }
  }

  async precalcAsync(rootIndex = this.currentRoot, options = {}) {
    if (rootIndex !== this.currentRoot) this.setRoot(rootIndex);
    if (!this.root) throw new Error(`root ${this.currentRoot} is not present`);
    throwIfCancelled(options);
    const environment = this.environment;
    const budgetMilliseconds = Math.max(0, Number(options.budgetMilliseconds ?? 12));
    const now = () => typeof performance !== 'undefined' && typeof performance.now === 'function'
      ? performance.now()
      : Date.now();
    const yieldThread = typeof options.yield === 'function' ? options.yield : () => new Promise(resolve => {
      if (typeof requestAnimationFrame === 'function') requestAnimationFrame(() => resolve());
      else setTimeout(resolve, 0);
    });
    const stack = [];
    let completed = 0;
    let deadline = now() + budgetMilliseconds;
    this.handlerCallCount = 0;
    this.handlerCalls.length = 0;
    environment.initFrame(0, 0);

    const enter = op => {
      throwIfCancelled(options);
      if (op._calcState === 2 && !op.changed) return false;
      if (op._calcState === 1) throw new Error(`operator calculation cycle at op ${op.id}`);
      op._calcState = 1;
      if (op.changed || !op.cache) op.animBits.set(op.editBits);
      const savedVariables = environment._varSaves.length;
      try {
        const popCount = environment.executeAnimation(op, op.animation);
        stack.push({ op, popCount, stage: 0, index: 0 });
      } catch (error) {
        op._calcState = 0;
        environment.pop(environment._varSaves.length - savedVariables);
        throw error;
      }
      return true;
    };

    try {
      enter(this.root);
      while (stack.length) {
        // Checking every DFS edge keeps cancellation latency bounded by one
        // operator handler. A handler itself is synchronous and therefore
        // cannot observe an AbortSignal until it returns to the event loop.
        throwIfCancelled(options);
        const frame = stack[stack.length - 1];
        const op = frame.op;
        if (frame.stage === 0) {
          if (frame.index < op.inputs.length) {
            const input = op.inputs[frame.index++];
            enter(input);
            continue;
          }
          frame.stage = 1; frame.index = 0;
        }
        if (frame.stage === 1) {
          if (!(op.convention & OPC_DONTCALLLINK) && frame.index < op.links.length) {
            const link = op.links[frame.index++];
            if (link) enter(link);
            continue;
          }
          frame.stage = 2;
        }
        if (frame.stage === 2) {
          op.cache = this._invokeHandler('init', op, environment);
          if (op.cache && typeof op.cache === 'object' && Object.isExtensible(op.cache)) {
            if (op.cache.classId === undefined) op.cache.classId = op.outputClassId;
            if (op.cache.outputClass === undefined && op.classInfo?.outputClass) {
              op.cache.outputClass = op.classInfo.outputClass;
            }
          }
          op.calcError = op.cache == null;
          op.changed = false;
          op._calcState = 2;
          environment.pop(frame.popCount);
          stack.pop();
          completed++;
          if (!stack.length || now() >= deadline) {
            options.onProgress?.({
              completed,
              total: this.document.operations.length,
              opId: op.id,
              handlerCalls: this.handlerCallCount,
            });
            throwIfCancelled(options);
            if (stack.length) {
              await waitForCancellation(yieldThread(), options);
              throwIfCancelled(options);
              deadline = now() + budgetMilliseconds;
            }
          }
        }
      }
      throwIfCancelled(options);
      this.precalculated = true;
      return this.root.cache;
    } catch (error) {
      // Each active frame owns animation variable saves that must unwind in
      // reverse DFS order just as the synchronous finally blocks do.
      while (stack.length) {
        const frame = stack.pop();
        frame.op._calcState = 0;
        environment.pop(frame.popCount);
      }
      throw error;
    } finally {
      environment.exitFrame();
    }
  }

  addEvents(environment = this.environment) {
    for (const event of this.events) environment.addStaticEvent(event);
  }

  frame(beatTime, timeMilliseconds, options = {}) {
    if (!this.precalculated && options.precalc !== false) this.precalc();
    const environment = this.environment;
    const callsBefore = this.handlerCallCount;
    environment.initFrame(beatTime, timeMilliseconds);
    try {
      this.addEvents(environment);
      if ((beatTime | 0) >= 0 && this.root?.cache?.classId === KC_DEMO) {
        this.root.exec(environment);
      }
      if (options.summary) return this.summary({ operations: false });
      return {
        beatTime: environment.beatTime,
        timeMilliseconds: environment.currentTime,
        handlerCalls: this.handlerCallCount - callsBefore,
        outputs: environment.frameOutputs.length,
      };
    } finally {
      environment.exitFrame();
    }
  }

  frameAtSample(sample, sampleRate = 44100, options = {}) {
    const beat = audioSampleToBeat(sample, this.document, sampleRate);
    const milliseconds = Number((BigInt(sample | 0) * 1000n) / BigInt(sampleRate | 0)) | 0;
    return this.frame(beat, milliseconds, options);
  }

  snapshot() {
    const operations = [];
    for (const op of this.operations) {
      const animated = op.animation.length > 1 || op.animation[0] !== KA.END;
      const hasInstanceState = Object.keys(op.instanceState).length > 0;
      const hasSceneInstances = op.sceneInstances !== null && op.sceneInstances !== undefined;
      const canSnapshotCache = op.cache && typeof op.cache.snapshot === 'function';
      if (!animated && !op.changed && !hasInstanceState && !hasSceneInstances && !canSnapshotCache) continue;
      const state = { index: op.id };
      if (animated) state.animBits = Array.from(op.animBits);
      if (op.changed) state.changed = true;
      if (hasInstanceState) state.instanceState = cloneState(op.instanceState);
      if (hasSceneInstances) state.sceneInstances = op.sceneInstances.snapshot();
      if (canSnapshotCache) state.cacheState = cloneState(op.cache.snapshot());
      operations.push(state);
    }
    return {
      version: 2, currentRoot: this.currentRoot, precalculated: this.precalculated,
      operationCount: this.operations.length,
      operations,
      events: this.events.map(event => ({
        start: event.start, end: event.end, instances: event.instances.snapshot(), removeMe: event.removeMe,
      })),
      environment: {
        beatTime: this.environment.beatTime, currentTime: this.environment.currentTime,
        lastTime: this.environment.lastTime, timeDelta: this.environment.timeDelta,
        timeSlices: this.environment.timeSlices, timeJitter: this.environment.timeJitter,
        timeReset: this.environment.timeReset, variables: this.environment.vars.map(value => Array.from(value)),
        defaultInstances: this.environment.defaultInstances.snapshot(),
        dynamicEvents: this.environment.dynamicEvents.map(event => event.id),
      },
    };
  }

  restore(snapshot) {
    const version1 = snapshot?.version === 1 && snapshot.operations?.length === this.operations.length;
    const version2 = snapshot?.version === 2 && snapshot.operationCount === this.operations.length &&
      Array.isArray(snapshot.operations);
    if (!version1 && !version2) {
      throw new Error('incompatible runtime snapshot');
    }
    this.currentRoot = snapshot.currentRoot | 0;
    this.precalculated = Boolean(snapshot.precalculated);
    for (const op of this.operations) {
      op.animBits.set(op.editBits);
      op.syncAnimParameters();
      op.changed = false;
      op.instanceState = {};
      op.sceneInstances = null;
      op.firstEvent = null;
    }
    const operationStates = version1
      ? snapshot.operations.map((state, index) => ({ ...state, index }))
      : snapshot.operations;
    for (const state of operationStates) {
      const op = this.operations[state.index | 0];
      if (!op) throw new Error(`snapshot references missing operator ${state.index}`);
      if (state.animBits) { op.animBits.set(state.animBits); op.syncAnimParameters(); }
      op.changed = Boolean(state.changed);
      op.instanceState = cloneState(state.instanceState || {});
      if (state.sceneInstances !== null && state.sceneInstances !== undefined) {
        op.sceneInstances = new InstanceChain();
        op.sceneInstances.restore(state.sceneInstances);
      }
      if (state.cacheState !== null && state.cacheState !== undefined &&
          op.cache && typeof op.cache.restore === 'function') {
        op.cache.restore(cloneState(state.cacheState));
      }
    }
    this.events.forEach((event, index) => {
      const state = snapshot.events[index];
      event.start = state.start | 0; event.end = state.end | 0;
      event.instances.restore(state.instances); event.removeMe = Boolean(state.removeMe); event.nextOp = null;
    });
    const env = snapshot.environment;
    this.environment.beatTime = env.beatTime | 0;
    this.environment.currentTime = env.currentTime | 0;
    this.environment.lastTime = env.lastTime | 0;
    this.environment.timeDelta = env.timeDelta | 0;
    this.environment.timeSlices = env.timeSlices | 0;
    this.environment.timeJitter = env.timeJitter | 0;
    this.environment.timeReset = Boolean(env.timeReset);
    env.variables.forEach((value, index) => this.environment.vars[index].set(value));
    this.environment.defaultInstances.restore(env.defaultInstances);
    this.environment.dynamicEvents = (env.dynamicEvents || []).map(index => this.events[index]).filter(Boolean);
    this.environment.eventOpsCleanup.length = 0;
    this.environment.matrixStack.popAll();
    return this;
  }

  saveSnapshot(name = 'default') {
    const snapshot = this.snapshot();
    this._snapshots.set(name, snapshot);
    return snapshot;
  }

  restoreSnapshot(name = 'default') {
    const snapshot = this._snapshots.get(name);
    if (!snapshot) throw new Error(`runtime snapshot ${name} does not exist`);
    return this.restore(snapshot);
  }

  summary(options = {}) {
    const result = {
      rootIndex: this.currentRoot, rootOpId: this.root?.id ?? null,
      precalculated: this.precalculated,
      operationCount: this.operations.length, eventCount: this.events.length,
      splineCount: this.splines.length, handlerCallCount: this.handlerCallCount,
      handlerCalls: this.handlerCalls.map(entry => ({ ...entry })),
      environment: this.environment.summary(),
    };
    if (options.operations !== false) result.operations = this.operations.map(op => op.summary());
    if (options.events !== false) result.events = this.events.map(event => event.summary());
    return result;
  }
}

function exactMulDiv(a, b, c) {
  if (!c) throw new RangeError('division by zero');
  return Number((BigInt(a | 0) * BigInt(b | 0)) / BigInt(c | 0)) | 0;
}

function exactDivShift(a, b) {
  if (!b) throw new RangeError('division by zero');
  return Number((BigInt(a | 0) << 16n) / BigInt(b | 0)) | 0;
}

function sampleToBeat(sample, songBPMFixed, buzzTiming = false, sampleRate = 44100) {
  sample |= 0; songBPMFixed |= 0; sampleRate |= 0;
  if (!buzzTiming) return exactMulDiv(sample, songBPMFixed, Math.imul(sampleRate, 60));
  const bpm = songBPMFixed >> 16;
  const denominator = Math.trunc(Math.imul(60, sampleRate) / Math.imul(bpm, 8)) | 0;
  return Math.trunc(exactDivShift(sample, denominator) / 8) | 0;
}

function audioSampleToBeat(sample, documentOrBPM, sampleRate = 44100) {
  if (typeof documentOrBPM === 'number') return sampleToBeat(sample, documentOrBPM, false, sampleRate);
  return sampleToBeat(sample, documentOrBPM.songBPMFixed, Boolean(documentOrBPM.buzzTiming), sampleRate);
}

function sampleToBuzzBeat(sample, songBPMFixed, sampleRate = 44100) {
  return sampleToBeat(sample, songBPMFixed, true, sampleRate);
}

function initMiscEvent(call) { return call.inputs[0] || null; }
function execMiscEvent(call) {
  const { op, environment } = call;
  const duration = call.parameters[0];
  for (let event = op.firstEvent; event; event = event.nextOp) {
    if (event.start === event.end) {
      event.start = environment.beatTime;
      event.end = (event.start + roundEven(f32(duration * 65536))) | 0;
    }
    op.inputs[0].execEvent(environment, event);
  }
}

function initMiscEventTime(call) { return call.inputs[0] || null; }
function execMiscEventTime(call) {
  const { op, environment } = call;
  const memory = environment.getInstance(op, () => ({ endTime: 0x7fffffff }));
  if (memory.reset) memory.endTime = 0x7fffffff;
  const event = op.firstEvent;
  if (event) {
    op.inputs[0].execEvent(environment, event);
    memory.endTime = event.end;
  } else {
    const saved = copyVector(environment.vars[KV_TIME]);
    setScalar(environment.vars[KV_TIME], environment.beatTime > memory.endTime ? 1 : 0);
    try { op.execInputs(environment); }
    finally { environment.vars[KV_TIME].set(saved); }
  }
}

function initMiscDemo(call) {
  for (const cache of call.inputs) {
    if (!cache || (cache.classId !== KC_DEMO && cache.classId !== KC_IPP)) return null;
  }
  return new DemoObject();
}

function execMiscDemo(call) {
  const { op, environment, runtime } = call;
  for (let i = 0; i < op.inputs.length; i++) {
    const input = op.inputs[i], classId = input.cache?.classId;
    if (classId === KC_DEMO) input.exec(environment);
    else if (classId === KC_IPP) {
      // GenOverlayManager::Reset clears LastOutput before every root IPP.
      // A branch that has no active Event must therefore not inherit the
      // preceding branch's output.
      environment.lastOutput = null;
      environment.ippOutputs.clear();
      runtime.overlay?.reset?.(environment);
      input.exec(environment);
      if (environment.lastOutput) environment.frameOutputs.push(environment.lastOutput);
      runtime.overlay?.preserveLastOutput?.(environment);
    }
  }
}

function initMiscSpline(call) { return BlobSplinePath.fromBlob(call.op.blob); }
function initMiscPipeSpline(call) { return pipeToSpline(parsePipeBlob(call.op.blob)); }
function initMiscShaker(call) {
  return new ShakerSpline(call.runtime, call.op, call.inputs[0], call.inputs[1], call.parameters);
}
function initMiscSplineScale(call) {
  const input = call.inputs[0];
  if (!input || input.classId !== KC_SPLINE || !input.blobSpline) return null;
  const spline = input.blobSpline.clone();
  const [px, py, pz, bits] = call.parameters;
  for (const key of spline.keys) {
    key.px = f32(key.px * px); key.py = f32(key.py * py); key.pz = f32(key.pz * pz);
    if (bits & 1) key.rx = 0; if (bits & 2) key.ry = 0; if (bits & 4) key.rz = 0;
  }
  return spline;
}
function execMiscNop(call) { return call.op.execInputs(call.environment); }

const runtimeHandlers = {
  0x06: { init: initMiscEvent, exec: execMiscEvent },
  0x0d: { init: initMiscDemo, exec: execMiscDemo },
  0x16: { init: initMiscEventTime, exec: execMiscEventTime },
  0x18: { init: initMiscSpline, exec: execMiscNop },
  0x19: { init: initMiscShaker, exec: execMiscNop },
  0x1a: { init: initMiscPipeSpline, exec: execMiscNop },
  0x1c: { init: initMiscSplineScale, exec: execMiscNop },
};

export {
  Environment,
  Event,
  InstanceChain,
  MatrixStack,
  Op,
  Runtime,
  Spline,
  BlobSplinePath,
  ShakerSpline,
  KA as ANIMATION_OPS,
  OPC_BLOB,
  OPC_KENV,
  OPC_VARIABLEINPUT,
  OPC_SKIPEXEC,
  OPC_ALTEXEC,
  OPC_STRIPPEDIN,
  OPC_DONTCALLLINK,
  OPC_ALTINIT,
  OPC_KOP,
  OPC_FLEXINPUT,
  KC_BITMAP,
  KC_MINMESH,
  KC_SCENE,
  KC_MATERIAL,
  KC_MESH,
  KC_IPP,
  KC_EFFECT,
  KC_DEMO,
  KC_SPLINE,
  KC_ANY,
  KEF_NOTIME,
  audioSampleToBeat,
  sampleToBeat,
  sampleToBuzzBeat,
  runtimeHandlers,
};
