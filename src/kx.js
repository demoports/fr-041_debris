// Compact werkkzeug3 player-export reader. The byte order and compact scalar
// encodings mirror KDoc::Init in Farbrausch's released source.
import { OPC_BLOB, OPC_FLEXINPUT } from './abi.js';

const MAX_OP_ROOT = 16;
// The serialized convention has an eight-bit data-word count. Debris contains
// two intentional one-word packing overruns (18 fields for 17 data words), so
// 256 is both sufficient for the exact party data and a useful malformed-input
// ceiling before a packing string is multiplied by thousands of operations.
const MAX_PACKING_LENGTH = 256;
const MAX_DECODED_PARAMETER_SLOTS = 4 * 1024 * 1024;

const ANIM_PAYLOAD_SIZES = new Uint8Array([
  0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 16, 4, 4, 2, 0, 0, 0, 8, 2, 0,
  0, 0,
]);

const floatScratch = new DataView(new ArrayBuffer(4));

function bitsToFloat(bits) {
  floatScratch.setUint32(0, bits >>> 0, true);
  return floatScratch.getFloat32(0, true);
}

function align4(value) {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(`invalid byte length ${value}`);
  }
  const remainder = value % 4;
  return remainder ? value + 4 - remainder : value;
}

function inputCount(convention) {
  return (convention & 0x00000f00) >>> 8;
}

function linkCount(convention) {
  return (convention & 0x0000f000) >>> 12;
}

function stringCount(convention) {
  return (convention & 0x00070000) >>> 16;
}

function splineCount(convention) {
  return (convention & 0x00700000) >>> 20;
}

class Reader {
  constructor(bytes, offset = 0) {
    if (!Number.isSafeInteger(offset) || offset < 0 || offset > bytes.byteLength) {
      throw new RangeError(`invalid KX offset ${offset}`);
    }
    this.bytes = bytes;
    this.view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    this.pos = offset;
  }

  require(size, what = 'data') {
    if (!Number.isSafeInteger(size) || size < 0 || size > this.bytes.byteLength - this.pos) {
      throw new Error(
        `truncated ${what} at 0x${this.pos.toString(16)}: need ${size} bytes`,
      );
    }
  }

  skip(size, what = 'data') {
    this.require(size, what);
    const start = this.pos;
    this.pos += size;
    return start;
  }

  u8() {
    this.require(1);
    return this.bytes[this.pos++];
  }

  u16() {
    this.require(2);
    const value = this.view.getUint16(this.pos, true);
    this.pos += 2;
    return value;
  }

  s16() {
    this.require(2);
    const value = this.view.getInt16(this.pos, true);
    this.pos += 2;
    return value;
  }

  u24() {
    const a = this.u8();
    const b = this.u8();
    const c = this.u8();
    return a | (b << 8) | (c << 16);
  }

  u32() {
    this.require(4);
    const value = this.view.getUint32(this.pos, true);
    this.pos += 4;
    return value;
  }

  s32() {
    this.require(4);
    const value = this.view.getInt32(this.pos, true);
    this.pos += 4;
    return value;
  }

  compactShort() {
    let value = this.u8();
    if (value & 0x80) value = (value & 0x7f) | (this.u8() << 7);
    return value;
  }

  string(what = 'string', maxLength = Infinity) {
    const start = this.pos;
    while (this.pos < this.bytes.byteLength && this.bytes[this.pos] !== 0) {
      if (this.pos - start >= maxLength) {
        throw new Error(`${what} at 0x${start.toString(16)} exceeds ${maxLength} bytes`);
      }
      this.pos++;
    }
    if (this.pos === this.bytes.byteLength) {
      throw new Error(`unterminated ${what} at 0x${start.toString(16)}`);
    }
    let result = '';
    // Avoid one character-at-a-time concatenation and an unbounded spread call
    // for malformed long operator strings while preserving byte-for-code-unit
    // semantics (the native format is not UTF-8).
    for (let i = start; i < this.pos; i += 0x4000) {
      result += String.fromCharCode(...this.bytes.subarray(i, Math.min(this.pos, i + 0x4000)));
    }
    this.pos++;
    return result;
  }

  f16() {
    const first = this.u8();
    if (first === 0x00) return 0;
    if (first === 0x80) return 1;
    if (first === 0x01) return 0.5;
    if (first === 0x81) return 0.25; // exact released-reader behavior
    const value = (first << 8) | this.u8();
    const bits =
      ((value & 0x8000) << 16) |
      (((((value >>> 10) & 31) + 128 - 16) & 0xff) << 23) |
      ((value & 1023) << 13);
    return bitsToFloat(bits);
  }

  f24() {
    const first = this.u8();
    if (first === 0x00) return 0;
    if (first === 0x01) return 1;
    if (first === 0xff) return -1;
    const second = this.u16();
    return bitsToFloat(
      (first << 23) | ((second & 0x8000) << 16) | ((second & 0x7fff) << 8),
    );
  }
}

function parameter(reader, packing) {
  switch (packing.toLowerCase()) {
    case 'g': return reader.f16();
    case 'f': return reader.f24();
    case 'e': return reader.s16() / 4096;
    case 'i': return reader.s32();
    case 's': return reader.s16();
    case 'b': return reader.u8();
    case 'c': return reader.u32();
    case 'm': return reader.u24();
    default: return null;
  }
}

function animation(reader) {
  const start = reader.pos;
  for (;;) {
    const command = reader.u8();
    if (command >= 0x80) {
      reader.skip(1, 'animation store operand');
    } else {
      if (command >= ANIM_PAYLOAD_SIZES.length) {
        throw new Error(`unknown animation opcode 0x${command.toString(16)}`);
      }
      reader.skip(ANIM_PAYLOAD_SIZES[command], 'animation operand');
    }
    if (command === 1) return reader.bytes.subarray(start, reader.pos);
  }
}

function asBytes(input) {
  if (input instanceof Uint8Array) return input;
  if (input instanceof ArrayBuffer) return new Uint8Array(input);
  if (ArrayBuffer.isView(input)) {
    return new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
  }
  throw new TypeError('parseKX expects an ArrayBuffer or typed-array view');
}

function parseKX(input, offset = 0) {
  const bytes = asBytes(input);
  const reader = new Reader(bytes, offset);
  const start = offset;
  const sections = { header: 0 };

  const flags = reader.u32();
  if (flags & ~7) throw new Error(`unsupported KX flags 0x${flags.toString(16)}`);

  let name = null;
  if (flags & 1) {
    const nameStart = reader.pos;
    reader.skip(32, 'demo name');
    let length = 0;
    while (length < 32 && bytes[nameStart + length] !== 0) length++;
    name = String.fromCharCode(...bytes.subarray(nameStart, nameStart + length));
  }

  const songSize = reader.u32();
  const songOffset = reader.skip(align4(songSize), 'embedded song');
  sections.song = songOffset - start;

  let sampleOffset = null;
  let sampleSize = 0;
  if (flags & 2) {
    sampleSize = reader.u32();
    sampleOffset = reader.skip(align4(sampleSize), 'embedded samples');
    sections.samples = sampleOffset - start;
  }

  sections.document = reader.pos - start;
  const songBPMFixed = reader.u32();
  const songLength = reader.u32();
  const operationCount = reader.compactShort();
  const documentSplineCount = reader.compactShort();
  const roots = Array.from({ length: MAX_OP_ROOT }, () => reader.compactShort());

  const classes = [];
  for (;;) {
    const convention = reader.u32();
    if (convention === 0) break;
    if (classes.length === 128) throw new Error('more than 128 operator classes');
    classes.push({
      convention,
      id: reader.u16(),
      packing: reader.string('packing', MAX_PACKING_LENGTH),
    });
  }

  sections.graph = reader.pos - start;
  const operations = [];
  for (let opIndex = 0; opIndex < operationCount; opIndex++) {
    const typeByte = reader.u8();
    const classIndex = typeByte & 0x7f;
    const opClass = classes[classIndex];
    if (!opClass) throw new Error(`operator ${opIndex} has missing class ${classIndex}`);
    let count;
    if (typeByte & 0x80) count = 1;
    else if (opClass.convention & OPC_FLEXINPUT) count = reader.u8();
    else count = inputCount(opClass.convention);

    const inputs = [];
    for (let i = 0; i < count; i++) {
      const delta = typeByte & 0x80 ? 0 : reader.compactShort();
      const target = opIndex - 1 - delta;
      if (target < 0) throw new Error(`operator ${opIndex} has invalid input delta ${delta}`);
      inputs.push(target);
    }
    operations.push({
      id: opIndex,
      classIndex,
      classId: opClass.id,
      inputs,
      links: [],
      parameters: [],
      strings: [],
      splines: [],
      blobSize: 0,
      blob: null,
      animation: null,
    });
  }

  // Parameters are decoded into ordinary JavaScript arrays. A malicious class
  // table made entirely of no-byte '-' fields could otherwise amplify a small
  // graph into millions of array slots before the reader reaches another
  // bounds check. Real byte-consuming fields naturally stay below this ratio;
  // the exact party document uses 142,600 slots for 490,077 input bytes.
  const classOperationCounts = new Uint32Array(classes.length);
  for (const op of operations) classOperationCounts[op.classIndex]++;
  let decodedParameterSlots = 0;
  for (let classIndex = 0; classIndex < classes.length; classIndex++) {
    decodedParameterSlots += classOperationCounts[classIndex] * classes[classIndex].packing.length;
  }
  const parameterSlotBudget = Math.min(
    MAX_DECODED_PARAMETER_SLOTS,
    Math.max(4096, bytes.byteLength * 4),
  );
  if (decodedParameterSlots > parameterSlotBudget) {
    throw new Error(
      `decoded parameter table is too large: ${decodedParameterSlots} slots ` +
      `(limit ${parameterSlotBudget})`,
    );
  }

  for (const op of operations) {
    const count = linkCount(classes[op.classIndex].convention);
    for (let i = 0; i < count; i++) {
      const encoded = reader.compactShort();
      const target = encoded ? encoded - 1 : null;
      if (target !== null && target >= operationCount) throw new Error('invalid link target');
      op.links.push(target);
    }
  }

  sections.parameters = reader.pos - start;
  for (let classIndex = 0; classIndex < classes.length; classIndex++) {
    const opClass = classes[classIndex];
    for (const op of operations) {
      if (op.classIndex !== classIndex) continue;
      op.parameters = Array.from(opClass.packing, kind => parameter(reader, kind));
      const strings = stringCount(opClass.convention);
      for (let i = 0; i < strings; i++) op.strings.push(reader.string('operator string'));
      const splines = splineCount(opClass.convention);
      for (let i = 0; i < splines; i++) {
        const reference = reader.u16();
        const target = reference ? reference - 1 : null;
        if (target !== null && target >= documentSplineCount) {
          throw new Error(`operator ${op.id} has invalid spline target ${target}`);
        }
        op.splines.push(target);
      }
      if (opClass.convention & OPC_BLOB) op.blobSize = reader.u32();
    }
  }

  sections.animations = reader.pos - start;
  for (const op of operations) op.animation = animation(reader);

  sections.events = reader.pos - start;
  const eventCount = reader.compactShort();
  const events = [];
  for (let i = 0; i < eventCount; i++) {
    const operation = reader.compactShort();
    if (operation >= operationCount) throw new Error('event refers outside graph');
    const event = {
      operation,
      start: reader.s32(),
      end: reader.s32(),
      velocity: reader.f24(),
      modulation: reader.f24(),
      select: reader.u8(),
      scale: [reader.f24(), reader.f24(), reader.f24()],
      rotate: [reader.f24(), reader.f24(), reader.f24()],
      translate: [reader.f24(), reader.f24(), reader.f24()],
      color: reader.u32(),
      spline: (() => { const ref = reader.compactShort(); return ref ? ref - 1 : null; })(),
      startInterval: reader.f16(),
      endInterval: reader.f16(),
      flags: reader.u8(),
    };
    if (event.spline !== null && event.spline >= documentSplineCount) {
      throw new Error(`event ${i} has invalid spline target ${event.spline}`);
    }
    events.push(event);
  }

  sections.splines = reader.pos - start;
  const splines = [];
  for (let i = 0; i < documentSplineCount; i++) {
    const descriptor = reader.u8();
    const count = descriptor & 7;
    if (count > 4) throw new Error(`spline ${i} has ${count} channels`);
    const channels = [];
    for (let channel = 0; channel < count; channel++) {
      const keys = [];
      const keyCount = reader.compactShort();
      for (let key = 0; key < keyCount; key++) {
        keys.push({ time: reader.u8() / 255, value: reader.f16() });
      }
      channels.push(keys);
    }
    splines.push({ interpolation: descriptor >>> 3, channels });
  }

  sections.blobs = reader.pos - start;
  for (const op of operations) {
    if (!op.blobSize) continue;
    const blobOffset = reader.skip(op.blobSize, 'operator blob');
    op.blob = bytes.subarray(blobOffset, blobOffset + op.blobSize);
  }
  sections.end = reader.pos - start;

  return {
    bytes: bytes.subarray(start, reader.pos),
    flags,
    buzzTiming: Boolean(flags & 4),
    name,
    song: bytes.subarray(songOffset, songOffset + songSize),
    samples: sampleOffset === null
      ? new Uint8Array()
      : bytes.subarray(sampleOffset, sampleOffset + sampleSize),
    songBPMFixed,
    songBPM: songBPMFixed / 65536,
    songLength,
    roots,
    classes,
    operations,
    events,
    splines,
    sections,
  };
}

function summarizeKX(document) {
  const classCounts = new Uint32Array(document.classes.length);
  let animationBytes = 0;
  let blobBytes = 0;
  let blobCount = 0;
  for (const op of document.operations) {
    classCounts[op.classIndex]++;
    animationBytes += op.animation.byteLength;
    if (op.blobSize) {
      blobCount++;
      blobBytes += op.blobSize;
    }
  }
  return {
    size: document.bytes.byteLength,
    flags: document.flags,
    name: document.name,
    buzzTiming: document.buzzTiming,
    songSize: document.song.byteLength,
    sampleSize: document.samples.byteLength,
    songBPM: document.songBPM,
    songLength: document.songLength,
    operationCount: document.operations.length,
    eventCount: document.events.length,
    splineCount: document.splines.length,
    animationBytes,
    blobCount,
    blobBytes,
    sections: document.sections,
    classes: document.classes.map((opClass, index) => ({
      ...opClass,
      operationCount: classCounts[index],
    })),
  };
}

export { MAX_OP_ROOT, OPC_BLOB, OPC_FLEXINPUT, parseKX, summarizeKX };
