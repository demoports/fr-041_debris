// A direct JavaScript implementation of the Farbrausch V2 v6 player.  The
// implementation follows vendor/v2redux's v2load/v2seq/v2core split: parsing
// never mutates the source bytes, the player owns the event/sample clock, and
// the synthesizer can be driven independently with MIDI byte streams.

const f = Math.fround;
const TWO32 = 4294967296;
const TWO31 = 2147483648;
const PI = f(Math.PI);
const TWO_PI = f(Math.PI * 2);
const HALF_PI = f(Math.PI * 0.5);
const THREE_HALF_PI = f(Math.PI * 1.5);
const DC = f(3.814697265625e-6);
const LOWEST = f(1.220703125e-4);
const SUSTAIN_MULTIPLIER = f(0.0019375);
const MOOG_POINT_EIGHT = f(0.8);
const MOOG_FIVE_POINT_SIX = f(5.6);
const MOD_DELAY_LFO_MULTIPLIER = f(1973915.49);
const COMPRESSOR_PEAK_FALLOFF = f(0.9998);
const COMPRESSOR_RMS_SCALE = f(0.011048543456039805);
const MIX_GAIN = f(0.6);
const PATCH_PARMS = 89;
const GLOBAL_PARMS = 23;
const VOICE_PARMS = 59;
const CHANNEL_PARMS = 29;
const MAX_POLY = 64;
const CHANNELS = 16;

const bitsBuffer = new ArrayBuffer(4);
const bitsView = new DataView(bitsBuffer);

function bitsToFloat(value) {
  bitsView.setUint32(0, value >>> 0, true);
  return bitsView.getFloat32(0, true);
}

function floatToBits(value) {
  bitsView.setFloat32(0, value, true);
  return bitsView.getUint32(0, true);
}

function clamp(value, lo, hi) {
  return value < lo ? lo : value > hi ? hi : value;
}

// x87 FISTP's default mode: nearest, ties to even.
function roundEven(value) {
  if (!Number.isFinite(value)) return -2147483648;
  const lo = Math.floor(value);
  const d = value - lo;
  const rounded = d < 0.5 ? lo : d > 0.5 ? lo + 1 : (lo & 1) ? lo + 1 : lo;
  // Invalid/out-of-range FISTP stores the signed-integer indefinite value.
  return rounded < -2147483648 || rounded > 2147483647 ? -2147483648 : rounded;
}

// Project-owned 2^x kernel from v2math.h.  V2's x87 implementation rounds
// the f2xm1 mantissa to float32 before applying its (exact) power-of-two
// scale.  Keeping that rounding point explicit makes renders independent of
// the host JavaScript engine's libm implementation.
const EXP2_TAB = [
  [1, 0], [1.0218971486541166, 5.109225028973444e-17],
  [1.0442737824274138, 8.551889705537965e-17], [1.0671404006768237, -7.899853966841582e-17],
  [1.0905077326652577, -3.046782079812471e-17], [1.1143867425958924, 1.0410278456845571e-16],
  [1.1387886347566916, 8.912812676025408e-17], [1.1637248587775775, 3.8292048369240935e-17],
  [1.189207115002721, 3.982015231465646e-17], [1.215247359980469, -7.712630692681488e-17],
  [1.241857812073484, 4.658027591836937e-17], [1.2690509571917332, 2.667932131342186e-18],
  [1.2968395546510096, 2.5382502794888315e-17], [1.3252366431597413, -2.8587312100388614e-17],
  [1.3542555469368927, 7.70094837980299e-17], [1.383909881963832, -6.770511658794786e-17],
  [1.4142135623730951, -9.667293313452913e-17], [1.4451808069770467, -3.0237581349939873e-17],
  [1.4768261459394993, -3.483994556892796e-17], [1.5091644275934228, -1.016455327754295e-16],
  [1.5422108254079407, 7.949834809697621e-17], [1.5759808451078865, -1.0136916471278304e-17],
  [1.6104903319492543, 2.4707192569797888e-17], [1.645755478153965, -1.0125679913674773e-16],
  [1.681792830507429, 8.199010020581497e-17], [1.718619298122478, -1.851380418263111e-17],
  [1.7562521603732995, 2.960140695448873e-17], [1.7947090750031072, 1.8227458427912087e-17],
  [1.8340080864093424, 3.283107224245627e-17], [1.8741676341103, -6.122763413004143e-17],
  [1.9152065613971474, -1.0619946056195963e-16], [1.9571441241754002, 8.960767791036668e-17],
];
const EXP2_M1_C = [
  0.6931471805599453, 0.24022650695910072, 0.05550410866482158,
  0.009618129107628477, 0.0013333558146428443, 0.0001540353039338161,
  1.5252733804059841e-5,
];
const LOG2_TAB = [
  [0, 0], [0.044394119358453436, 1.3338680039226223e-18],
  [0.0874628412503394, 6.765321226991275e-18], [0.12928301694496647, -1.147571414337692e-17],
  [0.16992500144231237, -1.0448980122780218e-17], [0.20945336562894978, -1.747801539116594e-18],
  [0.2479275134435855, 3.8662183541602335e-18], [0.28540221886224837, -2.726283638197372e-17],
  [0.32192809488736235, -3.717019964142682e-19], [0.3575520046180837, 1.8984820907705057e-17],
  [0.3923174227787603, -1.6328502208352762e-17], [0.42626475470209796, -1.9932012137193316e-17],
  [0.45943161863729726, -3.8053583859449705e-19], [0.4918530963296747, -1.0820682119194486e-17],
  [0.5235619560570128, 3.838472289082233e-17], [0.5545888516776374, -1.2269989151629687e-17],
  [0.5849625007211562, -5.224490061390109e-18], [0.6147098441152082, -2.2208024293925304e-17],
  [0.6438561897747247, -7.434039928285364e-19], [0.6724253419714956, -2.6214744450027748e-17],
  [0.7004397181410922, -2.2038346320583612e-17], [0.7279204545631992, -2.476475356878588e-17],
  [0.7548875021634686, -1.5673470184170328e-17], [0.7813597135246596, -7.522378350087652e-19],
  [0.8073549220576041, 4.4407139084295174e-17], [0.8328900141647416, 5.415287952402795e-17],
  [0.8579809951275721, 3.2653869625311436e-17], [0.8826430493618412, 2.2296523086165164e-17],
  [0.9068905956085185, 4.991495917345345e-17], [0.9307373375628862, 4.094087911381388e-17],
  [0.9541963103868752, -3.7239566747188146e-17], [0.9772799234999164, 3.395815896151496e-17],
  [1, 0],
];
const ATAN_TAB = [
  [0, 0], [0.12435499454676144, -3.1253241424539383e-18],
  [0.24497866312686414, 1.0698755618734451e-17],
  [0.35877067027057225, -2.4623815582638635e-17],
  [0.4636476090008061, 2.2698777452961687e-17],
  [0.5585993153435624, -5.4556305485916264e-18],
  [0.6435011087932844, 1.5834785051444286e-17],
  [0.7188299996216245, -2.1478388444456983e-17],
  [0.7853981633974483, 3.061616997868383e-17],
];
function exp2Core(value) {
  const k = roundEven(value * 32);
  const t = value - k * 0.03125;
  const c = EXP2_M1_C;
  const q = t * (c[0] + t * (c[1] + t * (c[2] + t * (c[3] + t * (c[4] + t * (c[5] + t * c[6]))))));
  const kk = k + 32;
  const table = EXP2_TAB[kk & 31];
  const mantissa = table[0] + (table[0] * q + (table[1] + table[1] * q));
  return mantissa * (2 ** ((kk >> 5) - 1));
}
function pow2(value) {
  const y = f(value);
  const integer = Math.trunc(y);
  const mantissa = f(exp2Core(y - integer));
  return f(mantissa * (2 ** integer));
}
function log2Core(value) {
  const x = f(value);
  const bits = floatToBits(x);
  let exponent = ((bits >>> 23) & 255) - 127;
  let mantissa;
  if (((bits >>> 23) & 255) === 0) {
    // V2's power bases are normal, but retaining subnormal support keeps the
    // kernel well-defined for custom patch banks.
    mantissa = x;
    exponent = 0;
    while (mantissa < 1) { mantissa *= 2; exponent--; }
  } else {
    mantissa = 1 + (bits & 0x7fffff) / 0x800000;
  }
  const j = roundEven((mantissa - 1) * 32);
  const tableValue = 1 + j * 0.03125;
  const s = (mantissa - tableValue) / (mantissa + tableValue);
  const s2 = s * s;
  const p = 2.8853900817779268 * s * (1 + s2 * (1 / 3 + s2 * (1 / 5 + s2 * (1 / 7))));
  const table = LOG2_TAB[j];
  return exponent + (table[0] + (p + table[1]));
}
function powf24(base, value) {
  const t = f(value) * log2Core(f(base));
  const integer = Math.trunc(t);
  const mantissa = f(exp2Core(t - integer));
  return f(mantissa * (2 ** integer));
}
function atan01(value) {
  const j = roundEven(value * 8);
  const center = j * 0.125;
  const t = (value - center) / (1 + value * center);
  const square = t * t;
  const polynomial = t + t * square * (-1 / 3 + square * (1 / 5 + square * (-1 / 7 +
    square * (1 / 9 + square * (-1 / 11)))));
  const table = ATAN_TAB[j];
  return table[0] + (polynomial + table[1]);
}
function atanCore(value) {
  const magnitude = Math.abs(value);
  const result = magnitude > 1
    ? 3.141592653589793 * 0.5 - (atan01(1 / magnitude) - 6.123233995736766e-17)
    : atan01(magnitude);
  return value < 0 ? -result : result;
}
function overdriveGain(value, gain) {
  return f(value / atanCore(gain));
}

// glibc's TYPE_3 additive generator, which is the deterministic replacement
// used by the pinned v6 core for the original libc rand() LFO seed stream.
class V2Rand {
  constructor(seed = 1) {
    this.ring = new Uint32Array(31);
    this.reset(seed);
  }
  reset(seed = 1) {
    seed >>>= 0;
    if (!seed) seed = 1;
    const values = new Uint32Array(344);
    values[0] = seed;
    for (let i = 1; i < 31; i++) {
      let value = (16807 * (values[i - 1] | 0)) % 2147483647;
      if (value < 0) value += 2147483647;
      values[i] = value >>> 0;
    }
    for (let i = 31; i < 34; i++) values[i] = values[i - 31];
    for (let i = 34; i < 344; i++) values[i] = (values[i - 31] + values[i - 3]) >>> 0;
    for (let i = 0; i < 31; i++) this.ring[i] = values[313 + i];
    this.head = 0;
  }
  next() {
    const value = (this.ring[this.head] + this.ring[(this.head + 28) % 31]) >>> 0;
    this.ring[this.head] = value;
    this.head = (this.head + 1) % 31;
    return value >>> 1;
  }
}
const SIN_C = [-1 / 6, 1 / 120, -1 / 5040, 1 / 362880, -1 / 39916800, 1 / 6227020800, -1 / 1307674368000, 1 / 355687428096000];
const COS_C = [-1 / 2, 1 / 24, -1 / 720, 1 / 40320, -1 / 3628800, 1 / 479001600, -1 / 87178291200, 1 / 20922789888000];
function sinPoly(value) {
  const square = value * value;
  const c = SIN_C;
  return value + value * square * (c[0] + square * (c[1] + square * (c[2] + square * (c[3] + square * (c[4] + square * (c[5] + square * (c[6] + square * c[7])))))));
}
function cosPoly(value) {
  const square = value * value;
  const c = COS_C;
  return 1 + square * (c[0] + square * (c[1] + square * (c[2] + square * (c[3] + square * (c[4] + square * (c[5] + square * (c[6] + square * c[7])))))));
}
function sinCosCore(value, cosine) {
  const n = roundEven(value * 0.6366197723675814);
  let reduced = value - n * 1.5707963267341256;
  reduced -= n * 6.077100506303966e-11;
  reduced -= n * 2.0222662487959506e-21;
  switch ((n + (cosine ? 1 : 0)) & 3) {
    case 0: return sinPoly(reduced);
    case 1: return cosPoly(reduced);
    case 2: return -sinPoly(reduced);
    default: return -cosPoly(reduced);
  }
}
function sinf24(value) { return f(sinCosCore(f(value), false)); }
function cosf24(value) { return f(sinCosCore(f(value), true)); }
function calcFreq(value) { return pow2(f(f(value - 1) * 10)); }
function calcFreq2(value) { return pow2(f(f(value - 1) * 11)); }
function utof23(value) {
  return f(bitsToFloat(((value >>> 9) | 0x3f800000) >>> 0) - 1);
}
function ftou32(value) {
  return (2 * (roundEven(f(value * TWO31)) >>> 0)) >>> 0;
}

function fastSin(value) {
  const x = f(value);
  const x2 = f(x * x);
  let t = f(-0.00018542 * x2);
  t = f(t + 0.0083143);
  t = f(t * x2);
  t = f(t - 0.16666);
  t = f(t * x2);
  t = f(1 + t);
  return f(t * x);
}

function fastSinRC(value) {
  let x = f(value % TWO_PI);
  if (x > THREE_HALF_PI) x = f(x - TWO_PI);
  else if (x > HALF_PI) x = f(Math.PI - x);
  return fastSin(x);
}

function fastAtan(value) {
  let x = f(value);
  let sign = 1;
  if (x < 0) { sign = -1; x = -x; }
  // Preserve the original V2 table-selection bug for x >= 2.
  const c = x >= 1 && x < 2
    ? [-0.431597974, -1, 0.05831938, 0.76443945, 1, Math.PI / 2]
    : [1, 0.43157974, 1, 0.76443945, 0.05831938, 0];
  const x2 = f(x * x);
  let a = f(x2 * c[1]);
  let b = f(x2 * c[4]);
  a = f(a + c[0]);
  b = f(b + c[3]);
  const num = f(a * x);
  let den = f(b * x2);
  den = f(den + c[2]);
  return f(f(f(num / den) + c[5]) * sign);
}

function asBytes(value) {
  if (value instanceof Uint8Array) return value;
  if (value instanceof ArrayBuffer) return new Uint8Array(value);
  if (ArrayBuffer.isView(value)) return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
  throw new TypeError('V2M data must be an ArrayBuffer or Uint8Array');
}

class Reader {
  constructor(bytes) {
    this.bytes = bytes;
    this.view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    this.pos = 0;
  }
  need(size) {
    if (size < 0 || this.pos + size > this.bytes.length) throw new Error('Truncated V2M data');
  }
  u32() { this.need(4); const v = this.view.getUint32(this.pos, true); this.pos += 4; return v; }
  take(size) { this.need(size); const v = this.bytes.subarray(this.pos, this.pos + size); this.pos += size; return v; }
}

function delta24(data, count, index) {
  return data[index] + (data[count + index] << 8) + (data[count * 2 + index] << 16);
}

function decodeStream(data, count, columns, kind, channel, controller = 0) {
  const events = [];
  let time = 0;
  let previousTime = -1;
  let sameTimePass = 0;
  const values = new Uint8Array(columns - 3);
  for (let i = 0; i < count; i++) {
    time += delta24(data, count, i);
    for (let c = 3; c < columns; c++) values[c - 3] = (values[c - 3] + data[c * count + i]) & 255;
    const event = { time, kind, channel, controller, values: Array.from(values), order: i };
    // v2seq consumes every same-time note in one Tick(), but only one row from
    // each program/controller/pitch stream. A following zero-delta row is
    // therefore emitted in another zero-sample sequencer pass, after every
    // channel has completed the current pass.
    if (kind !== 'note') {
      sameTimePass = time === previousTime ? sameTimePass + 1 : 0;
      if (sameTimePass) event.sameTimePass = sameTimePass;
      previousTime = time;
    }
    events.push(event);
  }
  return events;
}

function parseSpeech(bytes) {
  if (!bytes.length) return Array(256).fill(' ');
  if (bytes.length < 4) return Array(256).fill(' ');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const count = Math.min(256, view.getUint32(0, true));
  if (4 + count * 4 > bytes.length) return Array(256).fill(' ');
  const decoder = new TextDecoder('windows-1252');
  const result = Array(256).fill(' ');
  for (let i = 0; i < count; i++) {
    const offset = view.getUint32(4 + i * 4, true);
    if (offset >= bytes.length) continue;
    let end = offset;
    while (end < bytes.length && bytes[end]) end++;
    result[i] = decoder.decode(bytes.subarray(offset, end));
  }
  return result;
}

function parseV2M(input) {
  const bytes = asBytes(input);
  const r = new Reader(bytes);
  const timediv = r.u32();
  const maxtime = r.u32();
  const globalCount = r.u32();
  if (!timediv || globalCount > 1_000_000) throw new Error('Invalid V2M header');
  const globalRaw = r.take(globalCount * 10);
  const globalEvents = [];
  let globalTime = 0;
  let previousGlobalTime = -1;
  let sameTimeGlobalPass = 0;
  const globalView = new DataView(globalRaw.buffer, globalRaw.byteOffset, globalRaw.byteLength);
  for (let i = 0; i < globalCount; i++) {
    globalTime += delta24(globalRaw, globalCount, i);
    const event = {
      time: globalTime,
      kind: 'global',
      tempo: globalView.getUint32(globalCount * 3 + i * 4, true),
      numerator: globalRaw[globalCount * 7 + i],
      denominator: globalRaw[globalCount * 8 + i],
      ticksPerQuarter: globalRaw[globalCount * 9 + i],
      order: i,
    };
    sameTimeGlobalPass = globalTime === previousGlobalTime ? sameTimeGlobalPass + 1 : 0;
    if (sameTimeGlobalPass) event.sameTimePass = sameTimeGlobalPass;
    previousGlobalTime = globalTime;
    globalEvents.push(event);
  }

  const channels = [];
  const events = globalEvents.slice();
  for (let channel = 0; channel < CHANNELS; channel++) {
    const noteCount = r.u32();
    const c = { noteCount, notes: [], programs: [], pitchBends: [], controllers: Array.from({ length: 7 }, () => []) };
    if (noteCount) {
      c.notes = decodeStream(r.take(noteCount * 5), noteCount, 5, 'note', channel);
      const programCount = r.u32();
      c.programs = decodeStream(r.take(programCount * 4), programCount, 4, 'program', channel);
      const bendCount = r.u32();
      c.pitchBends = decodeStream(r.take(bendCount * 5), bendCount, 5, 'pitchBend', channel);
      for (let controller = 1; controller <= 7; controller++) {
        const count = r.u32();
        c.controllers[controller - 1] = decodeStream(r.take(count * 4), count, 4, 'controller', channel, controller);
      }
      events.push(...c.programs);
      for (const stream of c.controllers) events.push(...stream);
      events.push(...c.pitchBends, ...c.notes);
    }
    channels.push(c);
  }

  const globalsSize = r.u32();
  if (globalsSize !== GLOBAL_PARMS) throw new Error(`Unsupported V2 global layout (${globalsSize}; expected v6 size 23)`);
  const globals = r.take(globalsSize).slice();
  const patchBlockSize = r.u32();
  if (!patchBlockSize || patchBlockSize > 1_048_576) throw new Error('Invalid V2 patch block size');
  const patchBlock = r.take(patchBlockSize);
  if (patchBlock.length < 4) throw new Error('Missing V2 patch map');
  const patchView = new DataView(patchBlock.buffer, patchBlock.byteOffset, patchBlock.byteLength);
  const patchCount = patchView.getUint32(0, true) / 4;
  if (!Number.isInteger(patchCount) || patchCount < 1 || patchCount > 128 || patchCount * 4 > patchBlock.length) {
    throw new Error('Invalid V2 patch offset table');
  }
  const patches = [];
  for (let i = 0; i < patchCount; i++) {
    const offset = patchView.getUint32(i * 4, true);
    if (offset + PATCH_PARMS + 1 > patchBlock.length) throw new Error(`Invalid V2 patch ${i}`);
    const params = patchBlock.subarray(offset, offset + PATCH_PARMS).slice();
    const modCount = patchBlock[offset + PATCH_PARMS];
    if (offset + PATCH_PARMS + 1 + modCount * 3 > patchBlock.length) throw new Error(`Truncated V2 patch ${i}`);
    const mods = [];
    for (let m = 0; m < modCount; m++) {
      const p = offset + PATCH_PARMS + 1 + m * 3;
      mods.push({ source: patchBlock[p], value: patchBlock[p + 1], destination: patchBlock[p + 2] });
    }
    patches.push({
      params,
      voice: params.subarray(0, VOICE_PARMS),
      channel: params.subarray(VOICE_PARMS, VOICE_PARMS + CHANNEL_PARMS),
      maxPoly: params[88],
      mods,
    });
  }

  let speechBytes = new Uint8Array(0);
  if (r.pos < bytes.length) {
    const speechSize = r.u32();
    if (speechSize && speechSize < 8192) speechBytes = r.take(speechSize).slice();
  }

  const priority = { program: 0, controller: 1, pitchBend: 2, note: 3 };
  events.sort((a, b) => {
    const time = a.time - b.time;
    if (time) return time;
    const pass = (a.sameTimePass || 0) - (b.sameTimePass || 0);
    if (pass) return pass;
    // v2seq applies the global tempo row first, then builds one MIDI packet by
    // ascending channel; only within a channel is program -> CC1..7 -> bend ->
    // notes the ordering key. Cross-channel order is observable because MIDI
    // program changes can free voices before a later channel allocates one.
    if (a.kind === 'global' || b.kind === 'global') {
      if (a.kind === b.kind) return a.order - b.order;
      return a.kind === 'global' ? -1 : 1;
    }
    return a.channel - b.channel || priority[a.kind] - priority[b.kind] ||
      a.controller - b.controller || a.order - b.order;
  });

  return Object.freeze({
    version: 6,
    bytes,
    timediv,
    maxtime,
    globalCount,
    globalEvents,
    channels,
    globals,
    patchBlock,
    patches,
    patchCount,
    speechBytes,
    lyrics: parseSpeech(speechBytes),
    events,
  });
}

class LRC {
  constructor() { this.l = 0; this.b = 0; }
  reset() { this.l = this.b = 0; }
  step(input, frequency, resonance) {
    this.l = f(this.l + f(frequency * this.b));
    const h = f(f(input - f(this.b * resonance)) - this.l);
    this.b = f(this.b + f(frequency * h));
    return h;
  }
  step2(input, frequency, resonance) {
    const x = f(input + DC);
    this.l = f(this.l + f(f(frequency * this.b) - DC));
    this.b = f(this.b + f(frequency * f(f(x - f(this.b * resonance)) - this.l)));
    this.l = f(this.l + f(frequency * this.b));
    const h = f(f(x - f(this.b * resonance)) - this.l);
    this.b = f(this.b + f(frequency * h));
    return h;
  }
}

class Moog {
  constructor() { this.b = new Float32Array(5); }
  reset() { this.b.fill(0); }
  step(realInput, frequency, pole, resonance) {
    let input = f(realInput + DC);
    input = f(input - f(resonance * this.b[4]));
    const t1 = this.b[1];
    this.b[1] = f(f(f(input + this.b[0]) * pole) + f(this.b[1] * frequency));
    const t2 = this.b[2];
    this.b[2] = f(f(f(t1 + this.b[1]) * pole) + f(this.b[2] * frequency));
    const t3 = this.b[3];
    this.b[3] = f(f(f(t2 + this.b[2]) * pole) + f(this.b[3] * frequency));
    let b4 = f(f(f(t3 + this.b[3]) * pole) + f(this.b[4] * frequency));
    b4 = f(b4 - f(f(f(b4 * b4) * b4) * f(1 / 6)));
    b4 = f(b4 - DC);
    this.b[4] = b4;
    this.b[0] = realInput;
    return b4;
  }
}

class DCFilter {
  constructor(r = 0.9971428571428571) { this.r = f(r); this.x = [0, 0]; this.y = [0, 0]; }
  reset() { this.x[0] = this.x[1] = this.y[0] = this.y[1] = 0; }
  step(input, channel = 0) {
    const y = f(f(f(f(f(this.r * this.y[channel]) - this.x[channel]) + input) + DC) - DC);
    this.x[channel] = input;
    this.y[channel] = y;
    return y;
  }
}

class DelayLine {
  constructor(length) { this.data = new Float32Array(length); this.pos = 0; }
  reset() { this.data.fill(0); this.pos = 0; }
  fetch() { return this.data[this.pos]; }
  feed(value) { this.data[this.pos] = value; if (++this.pos === this.data.length) this.pos = 0; }
}

class Oscillator {
  constructor(synth, index) {
    this.synth = synth;
    this.index = index;
    this.filter = new LRC();
    this.reset();
  }
  reset() {
    this.mode = 0;
    this.ring = false;
    this.counter = 0;
    this.frequency = 0;
    this.breakpoint = 0;
    this.noiseFrequency = 0;
    this.noiseResonance = 0;
    this.noiseSeed = [0xdeadbeef, 0xbaadf00d, 0xd3adc0de][this.index] >>> 0;
    this.gain = 0;
    this.note = 0;
    this.pitch = 0;
    this.filter.reset();
  }
  changePitch() {
    this.noiseFrequency = f(this.synth.linearFrequency * calcFreq(f(f(this.pitch + 64) / 128)));
    const exponent = f(f(f(this.pitch + this.note) - 60) * f(1 / 12));
    this.frequency = roundEven(f(pow2(exponent) * this.synth.oscBaseFrequency)) | 0;
  }
  set(params, offset) {
    this.mode = params[offset] | 0;
    this.ring = (params[offset + 1] & 1) !== 0;
    this.pitch = f(f(params[offset + 2] - 64) + f(f(params[offset + 3] - 64) / 128));
    this.changePitch();
    this.gain = f(params[offset + 5] / 128);
    const color = f(params[offset + 4] / 128);
    this.breakpoint = ftou32(color);
    this.noiseResonance = f(1 - f(Math.sqrt(color)));
  }
  output(dest, index, value) {
    dest[index] = this.ring ? f(dest[index] * value) : f(dest[index] + value);
  }
  transition(state, oldCounter, frequency, breakpoint) {
    const newState = (((state << 1) | ((oldCounter >>> 0) < (breakpoint >>> 0) ? 1 : 0)) & 3) >>> 0;
    return [newState, (newState | ((oldCounter >>> 0) < (frequency >>> 0) ? 4 : 0)) >>> 0];
  }
  renderTriSaw(dest, count) {
    const frequency = this.frequency >>> 0;
    const fphase = utof23(frequency);
    if (!(fphase > 0)) return;
    const omf = f(1 - fphase);
    const reciprocal = f(1 / fphase);
    const color = utof23(this.breakpoint);
    // syOscRender keeps these working values at x87 PC=24 precision.  The
    // pinned portable oracle expresses that as one float32 rounding after
    // every operation, including the otherwise-unused 1/0 coefficient at the
    // legal color=0 endpoint.
    const c1 = f(this.gain / color);
    const c2 = f(-this.gain / f(1 - color));
    let state = ((((this.counter - frequency) >>> 0) < this.breakpoint) ? 3 : 0) >>> 0;
    let counter = this.counter >>> 0;
    for (let i = 0; i < count; i++) {
      const p = f(utof23(counter) - color);
      state = (((state << 1) | (counter < this.breakpoint ? 1 : 0)) & 3) >>> 0;
      const code = (state | (counter < frequency ? 4 : 0)) >>> 0;
      counter = (counter + frequency) >>> 0;
      let value;
      switch (code) {
        case 3: value = f(c1 * f(f(p + p) - fphase)); break;
        case 0: value = f(c2 * f(f(p + p) - fphase)); break;
        case 2: {
          const pp = f(p * p);
          const pf = f(p - fphase);
          value = f(reciprocal * f(f(c2 * pp) - f(c1 * f(pf * pf))));
          break;
        }
        case 5: {
          const t = f(f(p + 1) - fphase);
          const terms = f(f(c2 * f(t * t)) - f(c1 * f(p * p)));
          value = f(f(-(f(terms + this.gain))) * reciprocal);
          break;
        }
        case 7: {
          const area = f(omf * f(f(p + p) + omf));
          value = f(f(-f(f(c1 * area) + this.gain)) * reciprocal);
          break;
        }
        case 4: {
          const area = f(omf * f(f(p + p) + omf));
          value = f(f(-f(f(c2 * area) + this.gain)) * reciprocal);
          break;
        }
        default: value = 0; break;
      }
      this.output(dest, i, f(value + this.gain));
    }
    this.counter = counter;
  }
  renderPulse(dest, count) {
    const frequency = this.frequency >>> 0;
    const fp = utof23(frequency);
    if (!(fp > 0)) return;
    const gainDiv = f(this.gain / fp);
    const color = utof23(this.breakpoint);
    const cc121 = f(f(f(gainDiv * 2) * f(color - 1)) + this.gain);
    const cc212 = f(f(f(gainDiv * 2) * color) - this.gain);
    let state = ((((this.counter - frequency) >>> 0) < this.breakpoint) ? 3 : 0) >>> 0;
    let counter = this.counter >>> 0;
    for (let i = 0; i < count; i++) {
      const p = utof23(counter);
      state = (((state << 1) | (counter < this.breakpoint ? 1 : 0)) & 3) >>> 0;
      const code = (state | (counter < frequency ? 4 : 0)) >>> 0;
      counter = (counter + frequency) >>> 0;
      let value;
      switch (code) {
        case 3: value = this.gain; break;
        case 0: value = -this.gain; break;
        case 2: value = f(f(f(gainDiv * 2) * f(color - p)) + this.gain); break;
        case 5: value = f(f(f(gainDiv * 2) * p) - this.gain); break;
        case 7: value = cc121; break;
        case 4: value = cc212; break;
        default: value = 0; break;
      }
      this.output(dest, i, f(value));
    }
    this.counter = counter;
  }
  renderSin(dest, count) {
    let counter = this.counter >>> 0;
    const frequency = this.frequency >>> 0;
    for (let i = 0; i < count; i++) {
      let phase = (counter + 0x40000000) >>> 0;
      counter = (counter + frequency) >>> 0;
      if (phase & 0x80000000) phase = (~phase) >>> 0;
      let t = bitsToFloat(((phase >>> 8) | 0x3f800000) >>> 0);
      t = f(f(t * PI) - THREE_HALF_PI);
      this.output(dest, i, f(this.gain * fastSin(t)));
    }
    this.counter = counter;
  }
  renderNoise(dest, count) {
    let seed = this.noiseSeed >>> 0;
    for (let i = 0; i < count; i++) {
      seed = (Math.imul(seed, 196314165) + 907633515) >>> 0;
      const noise = f(bitsToFloat(((seed >>> 9) | 0x40000000) >>> 0) - 3);
      const high = this.filter.step(noise, this.noiseFrequency, this.noiseResonance);
      const value = f(f(this.noiseResonance * f(this.filter.l + high)) + this.filter.b);
      this.output(dest, i, f(this.gain * value));
    }
    this.noiseSeed = seed;
  }
  renderFM(dest, count) {
    let counter = this.counter >>> 0;
    const frequency = this.frequency >>> 0;
    for (let i = 0; i < count; i++) {
      const mod = f(dest[i] * 2);
      const phase = bitsToFloat(((counter >>> 9) | 0x3f800000) >>> 0);
      const value = f(this.gain * fastSinRC(f(f(mod + phase) * TWO_PI)));
      counter = (counter + frequency) >>> 0;
      dest[i] = this.ring ? f(dest[i] * value) : value;
    }
    this.counter = counter;
  }
  renderAux(dest, count, left, right) {
    for (let i = 0; i < count; i++) {
      let value = f(f(f(left[i] + right[i]) * this.gain) * MIX_GAIN);
      if (this.ring) value = f(value * dest[i]);
      dest[i] = value;
    }
  }
  render(dest, count) {
    switch (this.mode & 7) {
      case 0: break;
      case 1: this.renderTriSaw(dest, count); break;
      case 2: this.renderPulse(dest, count); break;
      case 3: this.renderSin(dest, count); break;
      case 4: this.renderNoise(dest, count); break;
      case 5: this.renderFM(dest, count); break;
      case 6: this.renderAux(dest, count, this.synth.auxAL, this.synth.auxAR); break;
      case 7: this.renderAux(dest, count, this.synth.auxBL, this.synth.auxBR); break;
    }
  }
}

class Envelope {
  constructor() { this.reset(); }
  reset() { this.out = 0; this.state = 0; this.value = 0; this.attack = 0; this.decay = 0; this.sustain = 0; this.sustainFactor = 1; this.release = 0; this.gain = 0; }
  set(params, offset) {
    this.attack = pow2(f(f(params[offset] * -0.09375) + 7));
    this.decay = f(1 - calcFreq2(f(1 - f(params[offset + 1] / 128))));
    this.sustain = params[offset + 2];
    this.sustainFactor = pow2(f(SUSTAIN_MULTIPLIER * f(params[offset + 3] - 64)));
    this.release = f(1 - calcFreq2(f(1 - f(params[offset + 4] / 128))));
    this.gain = f(params[offset + 5] / 128);
  }
  tick(gate) {
    if (this.state <= 1 && gate) this.state = 2;
    else if (this.state >= 2 && !gate) this.state = 1;
    switch (this.state) {
      case 0: this.value = 0; break;
      case 1: this.value = f(this.value * this.release); break;
      case 2:
        this.value = f(this.value + this.attack);
        if (this.value >= 128) { this.value = 128; this.state = 3; }
        break;
      case 3:
        this.value = f(this.value * this.decay);
        if (this.value <= this.sustain) { this.value = this.sustain; this.state = 4; }
        break;
      case 4:
        this.value = f(this.value * this.sustainFactor);
        if (this.value > 128) this.value = 128;
        break;
    }
    if (this.value <= LOWEST) { this.value = 0; this.state = 0; }
    this.out = f(this.value * this.gain);
  }
}

class LFO {
  constructor(noiseSeed = 0) { this.noiseSeed = noiseSeed >>> 0; this.reset(); }
  reset(noiseSeed = this.noiseSeed) {
    this.out = 0; this.mode = 0; this.sync = false; this.envelopeMode = false;
    this.frequency = 0; this.counter = 0; this.phase = 0; this.gain = 0; this.dc = 0;
    this.noiseSeed = noiseSeed >>> 0; this.last = 0;
  }
  set(params, offset) {
    this.mode = params[offset] | 0;
    this.sync = params[offset + 1] !== 0;
    this.envelopeMode = params[offset + 2] !== 0;
    this.frequency = roundEven(f(f(calcFreq(f(params[offset + 3] / 128)) * TWO31) * 0.5)) | 0;
    this.phase = ftou32(f(params[offset + 4] / 128));
    const amplitude = params[offset + 6];
    switch (params[offset + 5] | 0) {
      case 0: this.gain = amplitude; this.dc = 0; break;
      case 1: this.gain = -amplitude; this.dc = 0; break;
      default: this.gain = amplitude; this.dc = f(-0.5 * amplitude); break;
    }
  }
  keyOn() { if (this.sync) { this.counter = this.phase; this.last = 0xffffffff; } }
  tick() {
    let value;
    switch (this.mode & 7) {
      case 0: value = utof23(this.counter); break;
      case 1: {
        const signed = this.counter | 0;
        value = utof23((((this.counter << 1) ^ (signed >> 31)) >>> 0));
        break;
      }
      case 2: value = utof23(((this.counter | 0) >> 31) >>> 0); break;
      case 3: value = f(f(fastSinRC(f(utof23(this.counter) * TWO_PI)) * 0.5) + 0.5); break;
      case 4:
      default:
        if (this.counter < this.last) this.noiseSeed = (Math.imul(this.noiseSeed, 196314165) + 907633515) >>> 0;
        this.last = this.counter;
        value = utof23(this.noiseSeed);
        break;
    }
    this.out = f(f(value * this.gain) + this.dc);
    const old = this.counter;
    this.counter = (this.counter + (this.frequency >>> 0)) >>> 0;
    if (this.counter < (this.frequency >>> 0) && this.envelopeMode) this.counter = 0xffffffff;
    return old;
  }
}

class Filter {
  constructor(synth) { this.synth = synth; this.lrc = new LRC(); this.moog = new Moog(); this.reset(); }
  reset() { this.mode = 0; this.frequency = 0; this.resonance = 0; this.moogFrequency = 0; this.moogPole = 0; this.moogResonance = 0; this.lrc.reset(); this.moog.reset(); }
  set(params, offset) {
    this.mode = params[offset] | 0;
    let frequency = f(calcFreq(f(params[offset + 1] / 128)) * this.synth.linearFrequency);
    const resonance = f(params[offset + 2] / 128);
    if ((this.mode & 7) < 6) {
      this.frequency = frequency;
      this.resonance = f(1 - resonance);
    } else {
      frequency = f(frequency * 0.25);
      const t = f(1 - frequency);
      this.moogPole = f(frequency + f(MOOG_POINT_EIGHT * f(frequency * t)));
      this.moogFrequency = f(1 - f(this.moogPole + this.moogPole));
      const qx = f(f(MOOG_FIVE_POINT_SIX * f(t * t)) + f(1 - t));
      const qb = f(resonance * f(1 + f(0.5 * f(t * qx))));
      this.moogResonance = f(f(qb + qb) + f(qb + qb));
    }
  }
  render(dest, source, count) {
    switch (this.mode & 7) {
      case 0: if (dest !== source) dest.set(source.subarray(0, count), 0); break;
      case 1:
        for (let i = 0; i < count; i++) { this.lrc.step2(source[i], this.frequency, this.resonance); dest[i] = this.lrc.l; }
        break;
      case 2:
        for (let i = 0; i < count; i++) { this.lrc.step2(source[i], this.frequency, this.resonance); dest[i] = this.lrc.b; }
        break;
      case 3:
        for (let i = 0; i < count; i++) dest[i] = this.lrc.step2(source[i], this.frequency, this.resonance);
        break;
      case 4:
        for (let i = 0; i < count; i++) { const h = this.lrc.step2(source[i], this.frequency, this.resonance); dest[i] = f(this.lrc.l + h); }
        break;
      case 5:
        for (let i = 0; i < count; i++) { const h = this.lrc.step2(source[i], this.frequency, this.resonance); dest[i] = f(f(h + this.lrc.b) + this.lrc.l); }
        break;
      case 6:
        for (let i = 0; i < count; i++) { const input = source[i]; this.moog.step(input, this.moogFrequency, this.moogPole, this.moogResonance); dest[i] = this.moog.step(input, this.moogFrequency, this.moogPole, this.moogResonance); }
        break;
      case 7:
        for (let i = 0; i < count; i++) { const input = source[i]; this.moog.step(input, this.moogFrequency, this.moogPole, this.moogResonance); dest[i] = f(input - this.moog.step(input, this.moogFrequency, this.moogPole, this.moogResonance)); }
        break;
    }
  }
}

class Distortion {
  constructor(synth) {
    this.synth = synth; this.leftFilter = new Filter(synth); this.rightFilter = new Filter(synth);
    this.stereoScratch = new Float32Array(2); this.reset();
  }
  reset() {
    this.mode = 0; this.gain1 = this.gain2 = this.offset = 0; this.crushScale = 0;
    this.crushFactor = this.crushXor = 0; this.decimateCounter = this.decimateFrequency = 0;
    this.lastLeft = this.lastRight = 0; this.leftFilter.reset(); this.rightFilter.reset();
  }
  set(params, offset) {
    this.mode = params[offset] | 0;
    this.gain1 = pow2(f(f(params[offset + 1] - 32) / 16));
    switch (this.mode) {
      case 1: {
        const p = f(params[offset + 2] / 128);
        this.gain2 = overdriveGain(p, this.gain1);
        this.offset = f(f(this.gain1 * 2) * f(f(params[offset + 3] / 128) - 0.5));
        break;
      }
      case 2:
        this.gain2 = f(params[offset + 2] / 128);
        this.offset = f(f(this.gain1 * 2) * f(f(params[offset + 3] / 128) - 0.5));
        break;
      case 3: {
        const value = f(params[offset + 2] * 256 + 1);
        this.crushFactor = value | 0;
        this.crushScale = f(this.gain1 * f(32768 / value));
        this.crushXor = (params[offset + 3] << 9) | 0;
        break;
      }
      case 4: this.decimateFrequency = ftou32(calcFreq(f(params[offset + 2] / 128))); break;
      default:
        if (this.mode > 4) {
          const p = new Float32Array([this.mode - 4, params[offset + 2], params[offset + 3]]);
          this.leftFilter.set(p, 0); this.rightFilter.set(p, 0);
        }
        break;
    }
  }
  monoValue(value) {
    switch (this.mode) {
      case 0: return value;
      case 1: return f(this.gain2 * fastAtan(f(f(value * this.gain1) + this.offset)));
      case 2: return f(this.gain2 * clamp(f(f(value * this.gain1) + this.offset), -1, 1));
      case 3: {
        let q = roundEven(f(value * this.crushScale)) | 0;
        q = clamp(Math.imul(q, this.crushFactor), -0x7fff, 0x7fff) ^ this.crushXor;
        return f(q / 32768);
      }
      default: return value;
    }
  }
  renderMono(dest, source, count) {
    if (this.mode >= 5) { this.leftFilter.render(dest, source, count); return; }
    if (this.mode === 4) {
      for (let i = 0; i < count; i++) {
        const before = this.decimateCounter;
        this.decimateCounter = (before + this.decimateFrequency) >>> 0;
        if (this.decimateCounter < this.decimateFrequency) this.lastLeft = source[i];
        dest[i] = this.lastLeft;
      }
      return;
    }
    for (let i = 0; i < count; i++) dest[i] = this.monoValue(source[i]);
  }
  renderStereo(left, right, count) {
    if (this.mode >= 5 && this.mode <= 9) {
      this.leftFilter.render(left, left, count); this.rightFilter.render(right, right, count); return;
    }
    if (this.mode === 4) {
      for (let i = 0; i < count; i++) {
        this.decimateCounter = (this.decimateCounter + this.decimateFrequency) >>> 0;
        if (this.decimateCounter < this.decimateFrequency) { this.lastLeft = left[i]; this.lastRight = right[i]; }
        left[i] = this.lastLeft; right[i] = this.lastRight;
      }
      return;
    }
    if (this.mode >= 10) {
      // The original stereo dispatcher deliberately falls through to its mono
      // renderer for both Moog modes, processing interleaved L/R samples with
      // the single left filter state.
      for (let i = 0; i < count; i++) {
        this.stereoScratch[0] = left[i]; this.stereoScratch[1] = right[i];
        this.leftFilter.render(this.stereoScratch, this.stereoScratch, 2);
        left[i] = this.stereoScratch[0]; right[i] = this.stereoScratch[1];
      }
      return;
    }
    for (let i = 0; i < count; i++) { left[i] = this.monoValue(left[i]); right[i] = this.monoValue(right[i]); }
  }
}

class Voice {
  constructor(synth, index) {
    this.synth = synth;
    this.index = index;
    this.oscillators = Array.from({ length: 3 }, (_, i) => new Oscillator(synth, i));
    this.filters = [new Filter(synth), new Filter(synth)];
    this.envelopes = [new Envelope(), new Envelope()];
    this.lfos = [new LFO(synth.random.next()), new LFO(synth.random.next())];
    this.distortion = new Distortion(synth);
    this.dc = new DCFilter(synth.dcCoefficient);
    this.params = new Float32Array(VOICE_PARMS);
    this.reset();
  }
  reset(reseedLfos = false) {
    this.note = 0; this.velocity = 0; this.gate = false; this.currentVolume = 0; this.volumeRamp = 0;
    this.transpose = 0; this.filterMode = 0; this.leftVolume = this.rightVolume = 0;
    this.filter1Gain = this.filter2Gain = 1; this.keySync = 0;
    for (const osc of this.oscillators) osc.reset();
    for (const filter of this.filters) filter.reset();
    for (const env of this.envelopes) env.reset();
    for (const lfo of this.lfos) lfo.reset(reseedLfos ? this.synth.random.next() : lfo.noiseSeed);
    this.distortion.reset(); this.dc.reset();
  }
  updateNote() {
    const note = f(this.transpose + this.note);
    for (const osc of this.oscillators) { osc.note = note; osc.changePitch(); }
  }
  set(params) {
    this.params.set(params);
    this.transpose = f(params[1] - 64);
    this.updateNote();
    this.filterMode = params[26] | 0;
    this.keySync = params[58] | 0;
    const pan = f(params[0] / 128);
    this.leftVolume = f(Math.sqrt(f(1 - pan)));
    this.rightVolume = f(Math.sqrt(pan));
    const balance = f(f(params[27] - 64) / 64);
    if (roundEven(params[27] - 64) >= 0) { this.filter2Gain = 1; this.filter1Gain = f(1 - balance); }
    else { this.filter1Gain = 1; this.filter2Gain = f(1 + balance); }
    for (let i = 0; i < 3; i++) this.oscillators[i].set(params, 2 + i * 6);
    this.filters[0].set(params, 20); this.filters[1].set(params, 23);
    this.distortion.set(params, 28);
    this.envelopes[0].set(params, 32); this.envelopes[1].set(params, 38);
    this.lfos[0].set(params, 44); this.lfos[1].set(params, 51);
  }
  noteOn(note, velocity) {
    this.note = note | 0; this.velocity = f(velocity); this.gate = true; this.updateNote();
    for (const env of this.envelopes) env.state = 2;
    if (this.keySync === 2) {
      for (const env of this.envelopes) env.value = 0;
      this.currentVolume = 0;
      for (const osc of this.oscillators) osc.reset();
      for (const filter of this.filters) filter.reset();
      this.distortion.reset();
      // Reapply the already-modulated parameters to freshly reset blocks.
      this.set(this.params);
    }
    if (this.keySync >= 1) for (const osc of this.oscillators) osc.counter = 0;
    for (const osc of this.oscillators) osc.changePitch();
    for (const lfo of this.lfos) lfo.keyOn();
    this.dc.reset();
  }
  noteOff() { this.gate = false; }
  tick() {
    for (const env of this.envelopes) env.tick(this.gate);
    for (const lfo of this.lfos) lfo.tick();
    this.volumeRamp = f(f(f(this.envelopes[0].out / 128) - this.currentVolume) * this.synth.inverseFrame);
  }
  render(channelLeft, channelRight, count) {
    const voice = this.synth.voiceBuffer;
    const voice2 = this.synth.voiceBuffer2;
    voice.fill(0, 0, count);
    for (const oscillator of this.oscillators) oscillator.render(voice, count);
    switch (this.filterMode) {
      case 0: this.filters[0].render(voice, voice, count); break;
      case 2:
        this.filters[1].render(voice2, voice, count);
        this.filters[0].render(voice, voice, count);
        for (let i = 0; i < count; i++) voice[i] = f(f(voice[i] * this.filter1Gain) + f(voice2[i] * this.filter2Gain));
        break;
      case 1:
      default:
        this.filters[0].render(voice, voice, count);
        this.filters[1].render(voice, voice, count);
        break;
    }
    this.distortion.renderMono(voice, voice, count);
    for (let i = 0; i < count; i++) voice[i] = this.dc.step(voice[i], 0);
    let volume = this.currentVolume;
    for (let i = 0; i < count; i++) {
      const value = f(voice[i] * volume);
      volume = f(volume + this.volumeRamp);
      const left = f(f(this.leftVolume * value) + DC);
      const right = f(f(this.rightVolume * value) + DC);
      channelLeft[i] = f(channelLeft[i] + left);
      channelRight[i] = f(channelRight[i] + right);
    }
    this.currentVolume = volume;
  }
}

class BassBoost {
  constructor(synth) { this.synth = synth; this.x1 = [0, 0]; this.x2 = [0, 0]; this.y1 = [0, 0]; this.y2 = [0, 0]; this.enabled = false; }
  reset() { this.x1.fill(0); this.x2.fill(0); this.y1.fill(0); this.y2.fill(0); }
  set(amount) {
    this.enabled = (amount | 0) !== 0;
    if (!this.enabled) return;
    const A = pow2(f(amount / 128));
    const beta = f(Math.sqrt(f(f(A * A + 1) - f(f(A - 1) * f(A - 1)))));
    const bs = f(beta * this.synth.boostSin);
    const am1 = f(A - 1), ap1 = f(A + 1);
    const cam1 = f(am1 * this.synth.boostCos), cap1 = f(ap1 * this.synth.boostCos);
    const ia0 = f(1 / f(f(bs + cam1) + ap1));
    this.b1 = f(f(f(2 * A) * f(am1 - cap1)) * ia0);
    this.a1 = f(f(-2 * f(am1 + cap1)) * ia0);
    this.a2 = f(f(f(ap1 + cam1) - bs) * ia0);
    const aia0 = f(A * ia0);
    this.b0 = f(f(f(ap1 - cam1) + bs) * aia0);
    this.b2 = f(f(f(ap1 - cam1) - bs) * aia0);
  }
  render(left, right, count) {
    if (!this.enabled) return;
    const buffers = [left, right];
    for (let channel = 0; channel < 2; channel++) {
      let xm1 = this.x1[channel], xm2 = this.x2[channel], ym1 = this.y1[channel], ym2 = this.y2[channel];
      const buffer = buffers[channel];
      for (let i = 0; i < count; i++) {
        const x = f(buffer[i] + DC);
        const y = f(f(this.b0 * x) + f(f(f(this.b1 * xm1) - f(this.a1 * ym1)) + f(f(this.b2 * xm2) - f(this.a2 * ym2))));
        ym2 = ym1; ym1 = y; xm2 = xm1; xm1 = x; buffer[i] = y;
      }
      this.x1[channel] = xm1; this.x2[channel] = xm2; this.y1[channel] = ym1; this.y2[channel] = ym2;
    }
  }
}

class ModDelay {
  constructor(synth, length) {
    this.synth = synth; this.left = new Float32Array(length); this.right = new Float32Array(length);
    this.mask = length - 1; this.reset();
  }
  reset() { this.left.fill(0); this.right.fill(0); this.position = 0; this.modCounter = 0; }
  set(params, offset) {
    this.wet = f(f(params[offset] - 64) / 64);
    this.dry = f(1 - Math.abs(this.wet));
    this.feedback = f(f(params[offset + 1] - 64) / 64);
    const lengthScale = f(f(this.mask - 1023) / 128);
    this.offsetLeft = roundEven(f(params[offset + 2] * lengthScale)) >>> 0;
    this.offsetRight = roundEven(f(params[offset + 3] * lengthScale)) >>> 0;
    this.modFrequency = roundEven(f(f(calcFreq(f(params[offset + 4] / 128)) *
      MOD_DELAY_LFO_MULTIPLIER) * this.synth.linearFrequency)) | 0;
    this.maxOffset = roundEven(f(f(f(params[offset + 5] / 128) * 1023))) >>> 0;
    this.modPhase = (2 * (roundEven(f(f(f(params[offset + 6] - 64) / 128) * TWO31)) >>> 0)) >>> 0;
  }
  processChannel(input, channel, dry) {
    let counter = (this.modCounter + (channel ? this.modPhase : 0)) >>> 0;
    counter = counter < 0x80000000 ? (counter * 2) >>> 0 : (0xffffffff - ((counter * 2) >>> 0)) >>> 0;
    // maxOffset is at most 1023, so this 32.32 product is exactly representable
    // by JavaScript's 53-bit integer mantissa (and avoids BigInt in the DSP).
    const product = counter * this.maxOffset;
    const whole = Math.floor(product / TWO32) >>> 0;
    const fraction = (product - whole * TWO32) >>> 0;
    const baseOffset = channel ? this.offsetRight : this.offsetLeft;
    const index = (this.position - whole - baseOffset - (channel ? 1 : 0)) >>> 0;
    const buffer = channel ? this.right : this.left;
    const a = buffer[index & this.mask], b = buffer[(index + 1) & this.mask];
    const delayed = f(a + f(f(1 - utof23(fraction)) * f(b - a)));
    buffer[this.position] = f(input + f(delayed * this.feedback));
    return f(f(input * dry) + f(delayed * this.wet));
  }
  renderChannel(left, right, count) {
    if (!this.wet) return;
    for (let i = 0; i < count; i++) {
      left[i] = this.processChannel(f(left[i] + DC), 0, this.dry);
      right[i] = this.processChannel(f(right[i] + DC), 1, this.dry);
      this.modCounter = (this.modCounter + (this.modFrequency >>> 0)) >>> 0;
      this.position = (this.position + 1) & this.mask;
    }
  }
  renderAux(left, right, input, count) {
    if (!this.wet) return;
    for (let i = 0; i < count; i++) {
      const x = f(input[i] + DC);
      left[i] = f(left[i] + this.processChannel(x, 0, 0));
      right[i] = f(right[i] + this.processChannel(x, 1, 0));
      this.modCounter = (this.modCounter + (this.modFrequency >>> 0)) >>> 0;
      this.position = (this.position + 1) & this.mask;
    }
  }
}

class Compressor {
  constructor(synth) {
    this.synth = synth;
    this.delayLeft = new Float32Array(5700); this.delayRight = new Float32Array(5700);
    this.rmsLeft = new Float32Array(8192); this.rmsRight = new Float32Array(8192);
    this.mode = 2; this.oldMode = 0; this.currentGain = [0, 0]; this.peak = [0, 0]; this.rms = [0, 0];
    this.delayCount = this.rmsCount = 0;
  }
  reset() {
    this.currentGain[0] = this.currentGain[1] = 1; this.peak[0] = this.peak[1] = 0; this.rms[0] = this.rms[1] = 0;
    this.rmsLeft.fill(0); this.rmsRight.fill(0); this.rmsCount = 0;
  }
  set(params, offset) {
    const requested = params[offset] | 0;
    this.mode = requested === 0 ? 4 : requested === 1 ? 0 : 1;
    if (params[offset + 1] !== 0) this.mode |= 2;
    if (this.mode !== this.oldMode) { this.oldMode = this.mode; this.reset(); }
    this.delayLength = clamp(roundEven(f(params[offset + 3] * this.synth.samplesPerMs)), 0, 5699) >>> 0;
    let threshold = f(8 * calcFreq(f(params[offset + 4] / 128)));
    this.inputVolume = f(1 / threshold);
    if (params[offset + 2] !== 0) threshold = 1;
    this.outputVolume = f(threshold * pow2(f(f(params[offset + 8] - 64) / 16)));
    this.ratio = f(params[offset + 5] / 128);
    this.attack = pow2(f(f(-params[offset + 6] * 12) / 128));
    this.release = pow2(f(f(-params[offset + 7] * 16) / 128));
  }
  detectPeak(value, channel) {
    this.peak[channel] = Math.max(
      f(f(this.peak[channel] * COMPRESSOR_PEAK_FALLOFF) + DC),
      Math.abs(value),
    );
    return this.peak[channel];
  }
  detectRMS(value, channel, accumDC) {
    const ring = channel ? this.rmsRight : this.rmsLeft;
    let sum = f(this.rms[channel] - ring[this.rmsCount]);
    sum = f(sum + accumDC);
    const square = f(value * value);
    sum = f(sum + square);
    ring[this.rmsCount] = square; this.rms[channel] = sum;
    return f(f(Math.sqrt(Math.max(0, sum))) * COMPRESSOR_RMS_SCALE);
  }
  render(left, right, count) {
    if (this.mode & 4) return;
    const levelsLeft = this.synth.levelLeft, levelsRight = this.synth.levelRight;
    const rmsMode = (this.mode & 1) !== 0, stereo = (this.mode & 2) !== 0;
    for (let i = 0; i < count; i++) {
      if (!stereo) {
        let value = f(0.5 * f(left[i] + right[i]));
        let level;
        if (rmsMode) level = this.detectRMS(f(value + DC), 0, 0);
        else level = this.detectPeak(value, 0);
        levelsLeft[i] = levelsRight[i] = f(this.inputVolume * level);
      } else if (rmsMode) {
        levelsLeft[i] = f(this.inputVolume * this.detectRMS(left[i], 0, DC));
        levelsRight[i] = f(this.inputVolume * this.detectRMS(right[i], 1, DC));
      } else {
        levelsLeft[i] = f(this.inputVolume * this.detectPeak(left[i], 0));
        levelsRight[i] = f(this.inputVolume * this.detectPeak(right[i], 1));
      }
      if (rmsMode) this.rmsCount = (this.rmsCount + 1) & 8191;
    }
    const buffers = [left, right], delay = [this.delayLeft, this.delayRight], levels = [levelsLeft, levelsRight];
    for (let channel = 0; channel < 2; channel++) {
      let gain = this.currentGain[channel], index = this.delayCount;
      for (let i = 0; i < count; i++) {
        const value = f(this.outputVolume * delay[channel][index]);
        delay[channel][index] = f(this.inputVolume * buffers[channel][i]);
        if (++index > this.delayLength) index = 0;
        const level = levels[channel][i];
        const target = level >= 1 ? f(1 / f(1 + f(this.ratio * f(level - 1)))) : 1;
        gain = f(gain + f((target < gain ? this.attack : this.release) * f(target - gain)));
        buffers[channel][i] = f(value * gain);
      }
      this.currentGain[channel] = gain;
      if (channel === 1) this.delayCount = index;
    }
  }
}

class Reverb {
  constructor(synth) {
    this.synth = synth;
    const combLengths = [[1309, 1635, 1811, 1926], [1327, 1631, 1833, 1901]];
    const allLengths = [[220, 74], [205, 77]];
    this.combs = combLengths.map(row => row.map(length => new DelayLine(length)));
    this.allpasses = allLengths.map(row => row.map(length => new DelayLine(length)));
    this.combLow = [new Float32Array(4), new Float32Array(4)]; this.highPass = [0, 0];
    this.combGain = new Float32Array(4); this.allGain = new Float32Array(2);
  }
  reset() {
    for (const row of this.combs) for (const delay of row) delay.reset();
    for (const row of this.allpasses) for (const delay of row) delay.reset();
    this.combLow[0].fill(0); this.combLow[1].fill(0); this.highPass[0] = this.highPass[1] = 0;
  }
  set(params, offset = 0) {
    const combDefaults = [f(0.966384599), f(0.958186359), f(0.953783929), f(0.950933178)];
    const allDefaults = [f(0.994260075), f(0.998044717)];
    const time = f(this.synth.linearFrequency * f(f(64 / f(params[offset] + 1)) ** 2));
    for (let i = 0; i < 4; i++) this.combGain[i] = powf24(combDefaults[i], time);
    for (let i = 0; i < 2; i++) this.allGain[i] = powf24(allDefaults[i], time);
    this.damping = f(this.synth.linearFrequency * f(params[offset + 1] / 128));
    this.inputGain = f(params[offset + 3] / 128);
    const low = f(params[offset + 2] / 128);
    this.lowCut = f(this.synth.linearFrequency * f(f(low * low) * f(low * low)));
  }
  render(left, right, input, count) {
    const dest = [left, right];
    for (let i = 0; i < count; i++) {
      const source = f(f(input[i] * this.inputGain) + DC);
      for (let channel = 0; channel < 2; channel++) {
        let current = 0;
        for (let j = 0; j < 4; j++) {
          const delayed = f(this.combGain[j] * this.combs[channel][j].fetch());
          const next = (j & 1) ? f(delayed - source) : f(delayed + source);
          const low = f(this.combLow[channel][j] + f(this.damping * f(next - this.combLow[channel][j])));
          this.combLow[channel][j] = low; this.combs[channel][j].feed(low); current = f(current + low);
        }
        for (let j = 0; j < 2; j++) {
          const delayed = this.allpasses[channel][j].fetch();
          const next = f(current + f(this.allGain[j] * delayed));
          this.allpasses[channel][j].feed(next);
          current = f(delayed - f(this.allGain[j] * next));
        }
        this.highPass[channel] = f(this.highPass[channel] + f(this.lowCut * f(current - this.highPass[channel])));
        dest[channel][i] = f(dest[channel][i] + f(current - this.highPass[channel]));
      }
    }
  }
}

// Ronan is kept as a stateful insert even when a bank has no lyrics.  Its
// excitation/filter model here follows the public implementation's routing;
// phoneme-table interpolation is intentionally conservative so arbitrary v6
// speech banks remain audible without introducing a native dependency.
class Ronan {
  constructor(sampleRate, lyrics) {
    this.sampleRate = sampleRate; this.lyrics = lyrics; this.reset();
  }
  reset() { this.textIndex = 0; this.pitch = 1; this.frameRate = 3; this.waitOn = this.waitOff = false; this.phase = 0; this.noise = 0x12345678; this.formants = [0, 0, 0]; }
  noteOn() { this.waitOn = false; }
  noteOff() { this.waitOff = false; }
  controller(number, value) {
    if (number === 4) { if (value < 63) { this.textIndex = value; this.formants.fill(0); } else this.frameRate = value - 63; }
    else if (number === 5) this.pitch = 2 ** ((value - 64) / 128);
  }
  tick() {}
  process(left, right, count) {
    const text = this.lyrics[this.textIndex] || ' ';
    if (!text.trim()) { for (let i = 0; i < count; i++) right[i] = left[i]; return; }
    const vowel = /[aeiouy]/i.test(text) ? 1 : 0.45;
    const frequencies = [700, 1220, 2600];
    for (let i = 0; i < count; i++) {
      this.noise = (Math.imul(this.noise, 196314165) + 907633515) >>> 0;
      const n = f(bitsToFloat(((this.noise >>> 9) | 0x40000000) >>> 0) - 3);
      let source = f(left[i] * vowel + f(n * f(1 - vowel) * 0.08));
      let output = 0;
      for (let j = 0; j < 3; j++) {
        const coefficient = f(2 * sinf24(f(Math.PI * frequencies[j] * this.pitch / this.sampleRate)));
        this.formants[j] = f(this.formants[j] + f(coefficient * f(source - this.formants[j])));
        output = f(output + f(this.formants[j] * [0.7, -0.35, 0.18][j]));
      }
      left[i] = right[i] = output;
    }
  }
}

class ChannelStrip {
  constructor(synth, index) {
    this.synth = synth; this.index = index;
    this.dc1 = new DCFilter(synth.dcCoefficient); this.boost = new BassBoost(synth);
    this.distortion = new Distortion(synth); this.dc2 = new DCFilter(synth.dcCoefficient);
    this.chorus = new ModDelay(synth, 2048); this.compressor = new Compressor(synth);
    this.params = new Float32Array(CHANNEL_PARMS);
  }
  set(params) {
    this.params.set(params);
    this.auxAReceive = f(params[1] / 128); this.auxBReceive = f(params[2] / 128);
    this.auxASend = f(MIX_GAIN * f(params[3] / 128)); this.auxBSend = f(MIX_GAIN * f(params[4] / 128));
    this.gain = f(MIX_GAIN * f(params[0] / 128));
    this.aux1Gain = f(f(f(params[5] / 128) * MIX_GAIN) * this.gain);
    this.aux2Gain = f(f(f(params[6] / 128) * MIX_GAIN) * this.gain);
    this.routing = params[7] | 0;
    this.boost.set(params[8]); this.distortion.set(params, 9); this.chorus.set(params, 13); this.compressor.set(params, 20);
  }
  accumulate(destLeft, destRight, sourceLeft, sourceRight, count, gain) {
    if (!gain) return;
    for (let i = 0; i < count; i++) {
      destLeft[i] = f(destLeft[i] + f(gain * sourceLeft[i]));
      destRight[i] = f(destRight[i] + f(gain * sourceRight[i]));
    }
  }
  process(left, right, count, muted) {
    this.accumulate(left, right, this.synth.auxAL, this.synth.auxAR, count, this.auxAReceive);
    this.accumulate(left, right, this.synth.auxBL, this.synth.auxBR, count, this.auxBReceive);
    for (let i = 0; i < count; i++) { left[i] = this.dc1.step(left[i], 0); right[i] = this.dc1.step(right[i], 1); }
    this.compressor.render(left, right, count);
    this.boost.render(left, right, count);
    if (this.routing === 0) {
      this.distortion.renderStereo(left, right, count);
      for (let i = 0; i < count; i++) { left[i] = this.dc2.step(left[i], 0); right[i] = this.dc2.step(right[i], 1); }
      this.chorus.renderChannel(left, right, count);
    } else {
      this.chorus.renderChannel(left, right, count);
      this.distortion.renderStereo(left, right, count);
      for (let i = 0; i < count; i++) { left[i] = this.dc2.step(left[i], 0); right[i] = this.dc2.step(right[i], 1); }
    }
    let peak = 0;
    for (let i = 0; i < count; i++) peak = Math.max(peak, Math.abs(left[i]), Math.abs(right[i]));
    this.synth.channelPeaks[this.index] = Math.max(this.synth.channelPeaks[this.index], f(peak * this.gain));
    if (muted) return;
    for (let i = 0; i < count; i++) {
      const mono = f(left[i] + right[i]);
      this.synth.aux1[i] = f(this.synth.aux1[i] + f(this.aux1Gain * mono));
      this.synth.aux2[i] = f(this.synth.aux2[i] + f(this.aux2Gain * mono));
      this.synth.auxAL[i] = f(this.synth.auxAL[i] + f(this.auxASend * left[i]));
      this.synth.auxAR[i] = f(this.synth.auxAR[i] + f(this.auxASend * right[i]));
      this.synth.auxBL[i] = f(this.synth.auxBL[i] + f(this.auxBSend * left[i]));
      this.synth.auxBR[i] = f(this.synth.auxBR[i] + f(this.auxBSend * right[i]));
      this.synth.mixLeft[i] = f(this.synth.mixLeft[i] + f(this.gain * left[i]));
      this.synth.mixRight[i] = f(this.synth.mixRight[i] + f(this.gain * right[i]));
    }
  }
}

class V2Synth {
  constructor(songOrPatches, options = {}) {
    const song = songOrPatches && songOrPatches.patches ? songOrPatches : null;
    this.patches = song ? song.patches : songOrPatches;
    if (!Array.isArray(this.patches) || !this.patches.length) throw new TypeError('V2Synth requires a parsed V2 patch bank');
    this.sampleRate = options.sampleRate || 44100;
    this.samplesPerMs = f(this.sampleRate / 1000);
    const reciprocal = f(1 / this.sampleRate);
    this.oscBaseFrequency = f(f(261.6255653 * TWO31) * reciprocal);
    this.linearFrequency = f(44100 * reciprocal);
    this.dcCoefficient = f(1 - f(126 * reciprocal));
    this.frameSize = Math.round(128 * this.sampleRate / 44100);
    if (this.frameSize < 1 || this.frameSize > 280) throw new RangeError('V2 sample rate is outside the supported frame-size range');
    this.inverseFrame = f(1 / this.frameSize);
    const boostPhase = f(f(reciprocal * TWO_PI) * 150);
    this.boostCos = cosf24(boostPhase); this.boostSin = sinf24(boostPhase);
    this.voiceBuffer = new Float32Array(this.frameSize); this.voiceBuffer2 = new Float32Array(this.frameSize);
    this.levelLeft = new Float32Array(this.frameSize); this.levelRight = new Float32Array(this.frameSize);
    this.channelLeft = new Float32Array(this.frameSize); this.channelRight = new Float32Array(this.frameSize);
    this.mixLeft = new Float32Array(this.frameSize); this.mixRight = new Float32Array(this.frameSize);
    this.aux1 = new Float32Array(this.frameSize); this.aux2 = new Float32Array(this.frameSize);
    this.auxAL = new Float32Array(this.frameSize); this.auxAR = new Float32Array(this.frameSize);
    this.auxBL = new Float32Array(this.frameSize); this.auxBR = new Float32Array(this.frameSize);
    this.random = new V2Rand(1);
    this.voices = Array.from({ length: MAX_POLY }, (_, i) => new Voice(this, i));
    this.channels = Array.from({ length: CHANNELS }, () => ({ program: 0, controllers: new Uint8Array(7), voiceIndex: 0 }));
    for (const channel of this.channels) channel.controllers[6] = 127;
    this.channelStrips = Array.from({ length: CHANNELS }, (_, i) => new ChannelStrip(this, i));
    this.voiceChannel = new Int16Array(MAX_POLY); this.voiceChannel.fill(-1);
    this.allocation = new Uint32Array(MAX_POLY); this.currentAllocation = 0;
    this.reverb = new Reverb(this); this.delay = new ModDelay(this, 32768);
    this.masterDC = new DCFilter(this.dcCoefficient); this.masterCompressor = new Compressor(this);
    this.ronan = new Ronan(this.sampleRate, (song && song.lyrics) || options.lyrics || Array(256).fill(' '));
    this.channelPeaks = new Float32Array(CHANNELS); this.muteMask = 0;
    this.runningStatus = 0; this.frameRead = this.frameSize;
    this.lowState = [0, 0]; this.highState = [0, 0];
    this.setGlobals((song && song.globals) || options.globals || new Uint8Array(GLOBAL_PARMS));
  }
  setGlobals(globals) {
    if (globals.length < GLOBAL_PARMS) throw new Error('V2 globals block is truncated');
    this.globals = Uint8Array.from(globals.subarray(0, GLOBAL_PARMS));
    this.reverb.set(this.globals, 0); this.delay.set(this.globals, 4); this.masterCompressor.set(this.globals, 13);
    this.lowFrequency = f(f(f(this.globals[11] + 1) / 128) ** 2);
    this.highFrequency = f(f(f(this.globals[12] + 1) / 128) ** 2);
  }
  modulationSource(voice, channelIndex, source) {
    if (source === 0) return voice.velocity;
    if (source >= 1 && source <= 7) return this.channels[channelIndex].controllers[source - 1];
    if (source === 8 || source === 9) return voice.envelopes[source - 8].out;
    if (source === 10 || source === 11) return voice.lfos[source - 10].out;
    return f(2 * f(voice.note - 48));
  }
  setVoiceValues(index) {
    const channelIndex = this.voiceChannel[index]; if (channelIndex < 0) return;
    const channel = this.channels[channelIndex], patch = this.patches[channel.program] || this.patches[0], voice = this.voices[index];
    const params = voice.params;
    for (let i = 0; i < VOICE_PARMS; i++) params[i] = patch.voice[i];
    for (const mod of patch.mods) {
      if (mod.destination >= VOICE_PARMS) continue;
      const scale = f(f(mod.value - 64) / 64);
      params[mod.destination] = f(clamp(f(params[mod.destination] + f(scale * this.modulationSource(voice, channelIndex, mod.source))), 0, 128));
    }
    voice.set(params);
  }
  setChannelValues(index) {
    const channel = this.channels[index], patch = this.patches[channel.program] || this.patches[0];
    const voice = this.voices[channel.voiceIndex] || this.voices[0];
    const params = this.channelStrips[index].params;
    for (let i = 0; i < CHANNEL_PARMS; i++) params[i] = patch.channel[i];
    for (const mod of patch.mods) {
      const destination = mod.destination - VOICE_PARMS;
      if (destination < 0 || destination >= CHANNEL_PARMS) continue;
      const scale = f(f(mod.value - 64) / 64);
      params[destination] = f(clamp(f(params[destination] + f(scale * this.modulationSource(voice, index, mod.source))), 0, 128));
    }
    this.channelStrips[index].set(params);
  }
  allocateVoice(channelIndex, patch) {
    let poly = 0;
    for (let i = 0; i < MAX_POLY; i++) if (this.voiceChannel[i] === channelIndex) poly++;
    let selected = -1;
    if (!poly || poly < patch.maxPoly) {
      for (let i = 0; i < MAX_POLY; i++) if (this.voiceChannel[i] < 0) { selected = i; break; }
    }
    const sameChannel = poly && poly >= patch.maxPoly;
    if (selected < 0) {
      let oldest = this.currentAllocation >>> 0;
      for (let i = 0; i < MAX_POLY; i++) {
        if (sameChannel && this.voiceChannel[i] !== channelIndex) continue;
        if (!this.voices[i].gate && this.allocation[i] <= oldest) { oldest = this.allocation[i]; selected = i; }
      }
    }
    if (selected < 0) {
      let oldest = this.currentAllocation >>> 0;
      for (let i = 0; i < MAX_POLY; i++) {
        if (sameChannel && this.voiceChannel[i] !== channelIndex) continue;
        if (this.allocation[i] <= oldest) { oldest = this.allocation[i]; selected = i; }
      }
    }
    return selected < 0 ? 0 : selected;
  }
  noteOn(channelIndex, note, velocity) {
    if (!velocity) { this.noteOff(channelIndex, note); return; }
    if (channelIndex === 15) this.ronan.noteOn();
    const channel = this.channels[channelIndex], patch = this.patches[channel.program] || this.patches[0];
    const index = this.allocateVoice(channelIndex, patch);
    this.voiceChannel[index] = channelIndex; channel.voiceIndex = index;
    this.allocation[index] = this.currentAllocation++ >>> 0;
    this.setVoiceValues(index); this.voices[index].noteOn(note, velocity);
  }
  noteOff(channelIndex, note) {
    if (channelIndex === 15) this.ronan.noteOff();
    for (let i = 0; i < MAX_POLY; i++) {
      const voice = this.voices[i];
      if (this.voiceChannel[i] === channelIndex && voice.note === note && voice.gate) { voice.noteOff(); break; }
    }
  }
  programChange(channelIndex, program) {
    const channel = this.channels[channelIndex]; program &= 127;
    if (channel.program !== program) {
      channel.program = program;
      for (let i = 0; i < MAX_POLY; i++) if (this.voiceChannel[i] === channelIndex) this.voiceChannel[i] = -1;
    }
    channel.controllers.fill(0, 0, 6);
  }
  controller(channelIndex, number, value) {
    if (number >= 1 && number <= 7) {
      this.channels[channelIndex].controllers[number - 1] = value;
      if (channelIndex === 15) this.ronan.controller(number, value);
    } else if (number === 120) {
      for (let i = 0; i < MAX_POLY; i++) if (this.voiceChannel[i] === channelIndex) { this.voices[i].reset(true); this.voiceChannel[i] = -1; }
    } else if (number === 123) {
      for (let i = 0; i < MAX_POLY; i++) if (this.voiceChannel[i] === channelIndex) this.voices[i].noteOff();
    }
  }
  processMIDI(bytes) {
    let position = 0;
    while (position < bytes.length && bytes[position] !== 0xfd) {
      if (bytes[position] & 0x80) this.runningStatus = bytes[position++];
      if (this.runningStatus < 0x80) break;
      const channel = this.runningStatus & 15;
      switch ((this.runningStatus >>> 4) & 7) {
        case 0: this.noteOff(channel, bytes[position], bytes[position + 1]); position += 2; break;
        case 1: this.noteOn(channel, bytes[position], bytes[position + 1]); position += 2; break;
        case 2: position++; break;
        case 3: this.controller(channel, bytes[position], bytes[position + 1]); position += 2; break;
        case 4: this.programChange(channel, bytes[position++]); break;
        case 5: case 6: position += 2; break;
        case 7: position = bytes.length; break;
      }
    }
  }
  tick() {
    for (let i = 0; i < MAX_POLY; i++) {
      if (this.voiceChannel[i] < 0) continue;
      this.setVoiceValues(i);
      this.voices[i].tick();
      if (this.voices[i].envelopes[0].state === 0) this.voiceChannel[i] = -1;
    }
    for (let i = 0; i < CHANNELS; i++) this.setChannelValues(i);
    this.ronan.tick();
    this.renderFrame(this.frameSize);
    this.frameRead = 0;
  }
  renderFrame(count) {
    this.mixLeft.fill(0, 0, count); this.mixRight.fill(0, 0, count);
    this.aux1.fill(0, 0, count); this.aux2.fill(0, 0, count);
    this.auxAL.fill(0, 0, count); this.auxAR.fill(0, 0, count);
    this.auxBL.fill(0, 0, count); this.auxBR.fill(0, 0, count);
    for (let channel = 0; channel < CHANNELS; channel++) {
      let active = false;
      for (let i = 0; i < MAX_POLY; i++) if (this.voiceChannel[i] === channel) { active = true; break; }
      if (!active) continue;
      this.channelLeft.fill(0, 0, count); this.channelRight.fill(0, 0, count);
      for (let i = 0; i < MAX_POLY; i++) if (this.voiceChannel[i] === channel) this.voices[i].render(this.channelLeft, this.channelRight, count);
      if (channel === 15) this.ronan.process(this.channelLeft, this.channelRight, count);
      this.channelStrips[channel].process(this.channelLeft, this.channelRight, count, ((this.muteMask >>> channel) & 1) !== 0);
    }
    this.reverb.render(this.mixLeft, this.mixRight, this.aux1, count);
    this.delay.renderAux(this.mixLeft, this.mixRight, this.aux2, count);
    for (let i = 0; i < count; i++) {
      this.mixLeft[i] = this.masterDC.step(this.mixLeft[i], 0);
      this.mixRight[i] = this.masterDC.step(this.mixRight[i], 1);
      let left = f(this.mixLeft[i] - this.lowState[0]); this.lowState[0] = f(this.lowState[0] + f(this.lowFrequency * left));
      let right = f(this.mixRight[i] - this.lowState[1]); this.lowState[1] = f(this.lowState[1] + f(this.lowFrequency * right));
      if (this.highFrequency !== 1) {
        this.highState[0] = f(this.highState[0] + f(this.highFrequency * f(left - this.highState[0])));
        this.highState[1] = f(this.highState[1] + f(this.highFrequency * f(right - this.highState[1])));
        left = this.highState[0]; right = this.highState[1];
      }
      this.mixLeft[i] = left; this.mixRight[i] = right;
    }
    this.masterCompressor.render(this.mixLeft, this.mixRight, count);
  }
  render(out, frames, offsetFrames = 0, add = false) {
    if (!(out instanceof Float32Array)) throw new TypeError('V2Synth.render output must be Float32Array');
    if (frames === undefined) frames = out.length >>> 1;
    if (!Number.isSafeInteger(frames) || !Number.isSafeInteger(offsetFrames) || frames < 0 || offsetFrames < 0 || (offsetFrames + frames) * 2 > out.length) throw new RangeError('V2Synth.render output range is invalid');
    let written = 0;
    while (written < frames) {
      if (this.frameRead >= this.frameSize) this.tick();
      const amount = Math.min(frames - written, this.frameSize - this.frameRead);
      let output = (offsetFrames + written) * 2;
      for (let i = 0; i < amount; i++) {
        const left = this.mixLeft[this.frameRead + i], right = this.mixRight[this.frameRead + i];
        if (add) { out[output++] += left; out[output++] += right; }
        else { out[output++] = left; out[output++] = right; }
      }
      this.frameRead += amount; written += amount;
    }
    return out;
  }
  getPolyphony() {
    const result = new Uint8Array(CHANNELS + 1);
    for (let i = 0; i < MAX_POLY; i++) if (this.voiceChannel[i] >= 0) { result[this.voiceChannel[i]]++; result[CHANNELS]++; }
    return result;
  }
  readChannelPeaks(out = new Float32Array(CHANNELS)) { out.set(this.channelPeaks); this.channelPeaks.fill(0); return out; }
}

function eventToMidi(event) {
  switch (event.kind) {
    case 'program': return new Uint8Array([0xc0 | event.channel, event.values[0] & 127, 0xfd]);
    case 'controller': return new Uint8Array([0xb0 | event.channel, event.controller, event.values[0] & 127, 0xfd]);
    case 'pitchBend': return new Uint8Array([0xe0 | event.channel, event.values[0] & 127, event.values[1] & 127, 0xfd]);
    case 'note': return new Uint8Array([0x90 | event.channel, event.values[0] & 127, event.values[1] & 127, 0xfd]);
    default: return null;
  }
}

// Synth checkpoints are intentionally native JS object graphs rather than a
// JSON serialization: uint32 counters, float32 delay contents, negative zero,
// and NaN payloads all need to survive a seek byte-for-byte.  Most V2 patches
// leave many large compressor rings untouched, so represent an all-zero typed
// array with one shared marker.  The byte scan observes the underlying bits
// (and therefore does not mistake -0 for +0).
const ZERO_TYPED_STATE = Object.freeze({ zeroTypedState: true });
const SYNTH_STATE_SKIP = new Set([
  // Immutable/back references.
  'synth', 'patches', 'lyrics',
  // Per-frame scratch that is overwritten before it is read. mixLeft and
  // mixRight are deliberately not skipped: frameRead can point inside them.
  'voiceBuffer', 'voiceBuffer2', 'levelLeft', 'levelRight',
  'channelLeft', 'channelRight', 'aux1', 'aux2',
  'auxAL', 'auxAR', 'auxBL', 'auxBR',
]);

function captureSynthState(root) {
  let bytes = 0;
  const capture = value => {
    if (value === null || typeof value !== 'object') {
      bytes += 8;
      return value;
    }
    if (ArrayBuffer.isView(value)) {
      const raw = new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
      let nonzero = 0;
      for (let i = 0; i < raw.length; i++) nonzero |= raw[i];
      if (!nonzero) return ZERO_TYPED_STATE;
      bytes += value.byteLength;
      const copy = new value.constructor(value.length);
      new Uint8Array(copy.buffer, copy.byteOffset, copy.byteLength).set(raw);
      return copy;
    }
    if (Array.isArray(value)) {
      bytes += 16 + value.length * 8;
      return value.map(capture);
    }
    const state = {};
    const keys = Object.keys(value);
    bytes += 24 + keys.length * 8;
    for (const key of keys) if (!SYNTH_STATE_SKIP.has(key)) state[key] = capture(value[key]);
    return state;
  };
  return { state: capture(root), bytes };
}

function restoreSynthState(target, state) {
  if (ArrayBuffer.isView(target)) {
    if (state === ZERO_TYPED_STATE) target.fill(0);
    else {
      if (!ArrayBuffer.isView(state) || state.constructor !== target.constructor || state.length !== target.length) {
        throw new TypeError('Incompatible V2 synth checkpoint buffer');
      }
      new Uint8Array(target.buffer, target.byteOffset, target.byteLength).set(
        new Uint8Array(state.buffer, state.byteOffset, state.byteLength),
      );
    }
    return;
  }
  if (!target || !state || typeof target !== 'object' || typeof state !== 'object') {
    throw new TypeError('Incompatible V2 synth checkpoint state');
  }
  for (const key of Object.keys(state)) {
    const saved = state[key];
    if (saved === ZERO_TYPED_STATE) {
      if (!ArrayBuffer.isView(target[key])) throw new TypeError('Incompatible V2 zero-buffer checkpoint');
      target[key].fill(0);
    } else if (saved === null || typeof saved !== 'object') target[key] = saved;
    else restoreSynthState(target[key], saved);
  }
}

class V2Player {
  constructor(input, options = {}) {
    this.song = input && input.patches ? input : parseV2M(input);
    this.sampleRate = options.sampleRate || 44100;
    this.options = { sampleRate: this.sampleRate };
    const interval = options.checkpointIntervalSamples === undefined
      ? Math.round(this.sampleRate * Number(options.checkpointIntervalSeconds ?? 8))
      : Number(options.checkpointIntervalSamples);
    this.checkpointIntervalSamples = Number.isSafeInteger(interval) && interval > 0 ? interval : 0;
    const memoryBytes = Number(options.checkpointMemoryBytes ?? 64 * 1024 * 1024);
    this.checkpointMemoryBytes = Number.isFinite(memoryBytes) && memoryBytes > 0
      ? Math.floor(memoryBytes) : 0;
    this._checkpoints = [];
    this._checkpointBytes = 0;
    this.reset();
  }
  _resetPlayback() {
    this.synth = new V2Synth(this.song, this.options);
    this.eventIndex = 0;
    this.tickTime = 0;
    this.nextTick = this.song.events.length ? this.song.events[0].time : 0xffffffff;
    this.tempoSamples = 5000 * this.sampleRate;
    this.tempoNumerator = 4; this.tempoDenominator = 4; this.ticksPerQuarter = 8;
    this.sampleRemainder = 0;
    this.samplesToEvent = 0;
    this.samplePosition = 0;
    this.isPlaying = this.song.events.length > 0;
    this._scheduleNextCheckpoint();
    return this;
  }
  reset() {
    this.clearSeekCheckpoints();
    return this._resetPlayback();
  }
  clearSeekCheckpoints() {
    this._checkpoints.length = 0;
    this._checkpointBytes = 0;
    return this;
  }
  _scheduleNextCheckpoint() {
    const interval = this.checkpointIntervalSamples;
    this._nextCheckpointSample = interval && this.checkpointMemoryBytes
      ? (Math.floor(this.samplePosition / interval) + 1) * interval
      : Infinity;
  }
  _checkpointIndex(sample, upperBound = false) {
    let lo = 0, hi = this._checkpoints.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (this._checkpoints[mid].sample < sample || (upperBound && this._checkpoints[mid].sample === sample)) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }
  _discardCheckpointsFrom(sample) {
    const first = this._checkpointIndex(sample);
    for (let i = first; i < this._checkpoints.length; i++) this._checkpointBytes -= this._checkpoints[i].bytes;
    this._checkpoints.length = first;
  }
  _capturePlaybackSnapshot() {
    const captured = captureSynthState(this.synth);
    return Object.freeze({
      version: 2,
      song: this.song,
      sampleRate: this.sampleRate,
      samplePosition: this.samplePosition,
      eventIndex: this.eventIndex,
      tickTime: this.tickTime,
      nextTick: this.nextTick,
      tempoSamples: this.tempoSamples,
      tempoNumerator: this.tempoNumerator,
      tempoDenominator: this.tempoDenominator,
      ticksPerQuarter: this.ticksPerQuarter,
      sampleRemainder: this.sampleRemainder,
      samplesToEvent: this.samplesToEvent,
      isPlaying: this.isPlaying,
      synthState: captured.state,
      stateBytes: captured.bytes + 128,
    });
  }
  _restorePlaybackSnapshot(snapshot) {
    if (!snapshot || snapshot.version !== 2 || snapshot.song !== this.song || snapshot.sampleRate !== this.sampleRate) {
      throw new TypeError('Incompatible V2Player snapshot');
    }
    restoreSynthState(this.synth, snapshot.synthState);
    this.samplePosition = snapshot.samplePosition;
    this.eventIndex = snapshot.eventIndex;
    this.tickTime = snapshot.tickTime;
    this.nextTick = snapshot.nextTick;
    this.tempoSamples = snapshot.tempoSamples;
    this.tempoNumerator = snapshot.tempoNumerator;
    this.tempoDenominator = snapshot.tempoDenominator;
    this.ticksPerQuarter = snapshot.ticksPerQuarter;
    this.sampleRemainder = snapshot.sampleRemainder;
    this.samplesToEvent = snapshot.samplesToEvent;
    this.isPlaying = snapshot.isPlaying;
    this._scheduleNextCheckpoint();
    return this;
  }
  _storeCheckpoint() {
    if (!this.checkpointIntervalSamples || !this.checkpointMemoryBytes || !this.samplePosition) return;
    const at = this._checkpointIndex(this.samplePosition);
    if (this._checkpoints[at]?.sample === this.samplePosition) return;
    const snapshot = this._capturePlaybackSnapshot();
    const checkpoint = { sample: this.samplePosition, bytes: snapshot.stateBytes, snapshot };
    this._checkpoints.splice(at, 0, checkpoint);
    this._checkpointBytes += checkpoint.bytes;

    // Remove the least useful interior point.  Minimising the span between
    // its neighbours progressively thins old 8-second points to 16, 32, ...
    // instead of turning the cache into a recent-only FIFO.
    while (this._checkpointBytes > this.checkpointMemoryBytes && this._checkpoints.length) {
      let victim = 0;
      if (this._checkpoints.length > 1) {
        let bestSpan = Infinity;
        for (let i = 0; i < this._checkpoints.length - 1; i++) {
          const previous = i ? this._checkpoints[i - 1].sample : 0;
          const span = this._checkpoints[i + 1].sample - previous;
          if (span < bestSpan) { bestSpan = span; victim = i; }
        }
      }
      const [removed] = this._checkpoints.splice(victim, 1);
      this._checkpointBytes -= removed.bytes;
    }
  }
  checkpointStats() {
    const samples = this._checkpoints.map(checkpoint => checkpoint.sample);
    Object.freeze(samples);
    return Object.freeze({
      count: this._checkpoints.length,
      bytes: this._checkpointBytes,
      maxBytes: this.checkpointMemoryBytes,
      intervalSamples: this.checkpointIntervalSamples,
      samples,
    });
  }
  processTick() {
    if (!this.isPlaying) return;
    const time = this.nextTick;
    this.tickTime = time;
    while (this.eventIndex < this.song.events.length && this.song.events[this.eventIndex].time === time) {
      const event = this.song.events[this.eventIndex++];
      if (event.kind === 'global') {
        // v2seq stores tempo in 1/10000 usec units, then scales by SR/100.
        // The native sequencer uses integer `samplerate / 100` here.
        this.tempoSamples = event.tempo * Math.floor(this.sampleRate / 100);
        this.tempoNumerator = event.numerator; this.tempoDenominator = event.denominator; this.ticksPerQuarter = event.ticksPerQuarter;
      } else {
        const midi = eventToMidi(event); if (midi) this.synth.processMIDI(midi);
      }
    }
    if (this.eventIndex >= this.song.events.length) {
      this.nextTick = 0xffffffff; this.isPlaying = false; this.samplesToEvent = 0xffffffff;
      return;
    }
    this.nextTick = this.song.events[this.eventIndex].time;
    const delta = BigInt((this.nextTick - this.tickTime) >>> 0);
    const product = delta * BigInt(Math.trunc(this.tempoSamples));
    const divisor = BigInt(10000 * this.song.timediv);
    let quotient = Number(product / divisor) >>> 0;
    const remainder = Number(product % divisor) >>> 0;
    const old = this.sampleRemainder >>> 0;
    this.sampleRemainder = (old + remainder) >>> 0;
    if (this.sampleRemainder < old) quotient = (quotient + 1) >>> 0;
    this.samplesToEvent = quotient;
  }
  render(out, frames, offsetFrames = 0) {
    if (!(out instanceof Float32Array)) throw new TypeError('V2Player.render output must be Float32Array');
    if (frames === undefined) frames = out.length >>> 1;
    if (!Number.isSafeInteger(frames) || !Number.isSafeInteger(offsetFrames) || frames < 0 || offsetFrames < 0 || (offsetFrames + frames) * 2 > out.length) throw new RangeError('V2Player.render output range is invalid');
    let remaining = frames, offset = offsetFrames;
    while (remaining) {
      if (this.samplesToEvent === 0 && this.isPlaying) this.processTick();
      let amount = !this.isPlaying && this.samplesToEvent === 0xffffffff
        ? remaining // Like v2seq STOPPED, keep rendering release tails.
        : Math.min(remaining, this.samplesToEvent >>> 0);
      amount = Math.min(amount, this._nextCheckpointSample - this.samplePosition);
      if (amount) {
        this.synth.render(out, amount, offset);
        offset += amount; remaining -= amount; this.samplePosition += amount;
        if (this.samplesToEvent !== 0xffffffff) this.samplesToEvent = (this.samplesToEvent - amount) >>> 0;
      } else if (!this.isPlaying) {
        amount = Math.min(remaining, this._nextCheckpointSample - this.samplePosition);
        out.fill(0, offset * 2, (offset + amount) * 2);
        offset += amount; remaining -= amount; this.samplePosition += amount;
      }
      if (this.samplePosition === this._nextCheckpointSample) {
        // Canonical block-boundary state includes any events that lie exactly
        // on the checkpoint sample, just like the trailing-edge rule below.
        if (this.samplesToEvent === 0 && this.isPlaying) this.processTick();
        this._storeCheckpoint();
        this._scheduleNextCheckpoint();
      }
    }
    // Events exactly on the block's trailing edge have no samples before the
    // return, but applying them now keeps state and isPlaying invariant across
    // host callback sizes (and makes calcSongSamples() an exact stop boundary).
    if (this.samplesToEvent === 0 && this.isPlaying) this.processTick();
    return out;
  }
  renderFrames(frames) { const out = new Float32Array(frames * 2); return this.render(out, frames); }
  seekSamples(target) {
    target = Number(target);
    if (!Number.isFinite(target) || target > Number.MAX_SAFE_INTEGER) throw new RangeError('V2Player seek target must be a finite sample position');
    target = Math.max(0, Math.trunc(target));
    const upper = this._checkpointIndex(target, true);
    const checkpoint = upper ? this._checkpoints[upper - 1] : null;
    if (checkpoint && (target < this.samplePosition || checkpoint.sample > this.samplePosition)) {
      this._restorePlaybackSnapshot(checkpoint.snapshot);
    } else if (target < this.samplePosition) this._resetPlayback();
    const scratchFrames = 32768;
    const scratch = new Float32Array(scratchFrames * 2);
    while (this.samplePosition < target) {
      const amount = Math.min(scratchFrames, target - this.samplePosition);
      this.render(scratch, amount, 0);
    }
    return this.samplePosition;
  }
  seek(seconds) { return this.seekSamples(seconds * this.sampleRate) / this.sampleRate; }
  snapshot() { return this._capturePlaybackSnapshot(); }
  restore(snapshot) {
    if (!snapshot || !Number.isFinite(snapshot.samplePosition)) throw new TypeError('Invalid V2Player snapshot');
    if (snapshot.version === 2) {
      // A public snapshot may represent a deliberately modified synth branch;
      // checkpoints at and after it are no longer guaranteed descendants.
      this._discardCheckpointsFrom(snapshot.samplePosition);
      return this._restorePlaybackSnapshot(snapshot);
    }
    this.clearSeekCheckpoints(); this._resetPlayback(); this.seekSamples(snapshot.samplePosition); return this;
  }
  calcSongSamples() {
    let eventIndex = 0, time = 0, next = this.song.events.length ? this.song.events[0].time : 0xffffffff;
    let tempo = 5000 * this.sampleRate, remainder = 0, total = 0;
    while (eventIndex < this.song.events.length) {
      time = next;
      while (eventIndex < this.song.events.length && this.song.events[eventIndex].time === time) {
        const event = this.song.events[eventIndex++]; if (event.kind === 'global') tempo = event.tempo * Math.floor(this.sampleRate / 100);
      }
      if (eventIndex >= this.song.events.length) break;
      next = this.song.events[eventIndex].time;
      const product = BigInt((next - time) >>> 0) * BigInt(Math.trunc(tempo));
      const divisor = BigInt(10000 * this.song.timediv);
      let amount = Number(product / divisor) >>> 0;
      const rem = Number(product % divisor) >>> 0, old = remainder;
      remainder = (remainder + rem) >>> 0; if (remainder < old) amount = (amount + 1) >>> 0;
      total += amount;
    }
    return total;
  }
}

function createV2Player(input, options) { return new V2Player(input, options); }

const V2 = Object.freeze({
  parse: parseV2M,
  Synth: V2Synth,
  Player: V2Player,
  Rand: V2Rand,
  createPlayer: createV2Player,
});

export { parseV2M, V2Rand, V2Synth, V2Player, createV2Player, V2 };
