import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { parseKX, summarizeKX } from '../src/kx.js';

const bytes = await readFile(new URL('../assets/debris_party.kx', import.meta.url));
const document = parseKX(bytes);
const summary = summarizeKX(document);

assert.equal(summary.size, 490077);
assert.equal(summary.flags, 7);
assert.equal(summary.songSize, 175735);
assert.equal(summary.sampleSize, 6640);
assert.equal(summary.songBPM, 196);
assert.equal(summary.operationCount, 16478);
assert.equal(summary.eventCount, 95);
assert.equal(summary.splineCount, 11);
assert.equal(summary.classes.length, 90);
assert.equal(createHash('sha256').update(document.song).digest('hex'),
  '0c024b021f47b481b99a7fd785452bca78b95c4a69f9da2509c3517ecdfd55bd',
  'the KX audio boundary exposes the pinned embedded V2M payload');

const multiply = document.operations.filter(op => op.classId === 0x95 || op.classId === 0x12f);
assert.equal(multiply.length, 819);
assert.ok(multiply.every(op => op.parameters[17] === 0));

// Lock the authored consumers of the adaptive stencil-volume path. Material
// 1.1 requests a shadow pass through MultiFlags bit 2; Material 2.0 uses its
// main Flags bit 4. Runtime paint jobs are duplicated and culled per light.
const material11 = document.operations.filter(op => op.classId === 0xd0);
const material20 = document.operations.filter(op => op.classId === 0xd3);
assert.equal(material11.length, 101);
assert.equal(material20.length, 120);
assert.equal(material11.filter(op => (op.parameters[48] & 0x04) !== 0).length, 3);
assert.equal(material20.filter(op => (op.parameters[0] & 0x10) !== 0).length, 78);

// Lock the production post-processing inventory used by the source-backed
// renderer checks. Debris always takes glare's 512x256 filtered route and
// never enables color correction's diagnostic range-only mode.
const glare = document.operations.filter(op => op.classId === 0x74);
const colorCorrection = document.operations.filter(op => op.classId === 0x75);
assert.equal(glare.length, 16);
assert.ok(glare.every(op => op.parameters[1] === 0));
assert.equal(colorCorrection.length, 12);
assert.ok(colorCorrection.every(op => op.parameters[7] === 0));

// This widely reused 512x512 material texture deliberately reverses its mip
// threshold after level two. The renderer must preserve source-word mip
// generation rather than handing it to gl.generateMipmap.
const reverseMipTexture = document.operations[1767];
assert.equal(reverseMipTexture.classId, 0x25);
assert.deepEqual(reverseMipTexture.parameters, [1, 0, 0x12]);
assert.deepEqual(reverseMipTexture.inputs, [1766]);
const reverseMipMaterial = document.operations[1777];
assert.equal(reverseMipMaterial.classId, 0xd3);
assert.equal(reverseMipMaterial.links[0], 1767);
assert.equal(reverseMipMaterial.links[3], 1767);
assert.equal(reverseMipMaterial.parameters[8] & 7, 2);
assert.equal(reverseMipMaterial.parameters[11] & 7, 4);

function syntheticKX(options = {}) {
  const output = [];
  const u8 = value => output.push(value & 0xff);
  const u16 = value => { u8(value); u8(value >>> 8); };
  const u32 = value => { u16(value); u16(value >>> 16); };
  const compact = value => {
    if (value < 0 || value > 32767) throw new RangeError('synthetic compact short');
    u8((value & 0x7f) | (value > 0x7f ? 0x80 : 0));
    if (value > 0x7f) u8(value >>> 7);
  };
  const operationCount = options.operationCount ?? 1;
  const splineCount = options.splineCount ?? 0;
  const packing = options.packing ?? '';
  const convention = options.convention ?? 0x02000000;
  u32(0); // flags
  u32(0); // embedded song size
  u32(60 * 65536); // SongBPM
  u32(65536); // SongLength
  compact(operationCount);
  compact(splineCount);
  for (let i = 0; i < 16; i++) compact(operationCount); // deliberately null roots
  u32(convention);
  u16(900);
  for (let i = 0; i < packing.length; i++) u8(packing.charCodeAt(i));
  u8(0);
  u32(0); // class-table terminator
  for (let i = 0; i < operationCount; i++) u8(0); // class zero, no inputs
  if ((convention & 0x00700000) !== 0) {
    const references = (convention & 0x00700000) >>> 20;
    for (let op = 0; op < operationCount; op++) {
      for (let i = 0; i < references; i++) u16(options.operationSplineReference ?? 0);
    }
  }
  for (let i = 0; i < operationCount; i++) u8(1); // KA_END
  const hasEvent = options.eventSplineReference !== undefined;
  compact(hasEvent ? 1 : 0);
  if (hasEvent) {
    compact(0); // operation zero
    u32(0); u32(65536);
    u8(0); u8(0); // velocity, modulation (F24 zero shortcuts)
    u8(0); // select
    for (let i = 0; i < 9; i++) u8(0); // scale, rotate, translate
    u32(0xffffffff); // color
    compact(options.eventSplineReference);
    u8(0); u8(0x80); // start/end intervals
    u8(0); // flags
  }
  for (let i = 0; i < splineCount; i++) u8(0); // zero-channel spline
  return new Uint8Array(output);
}

const wrappedSongLength = new Uint8Array(64);
new DataView(wrappedSongLength.buffer).setUint32(4, 0xffffffff, true);
assert.throws(() => parseKX(wrappedSongLength), /truncated embedded song/,
  '32-bit padding must not wrap a malformed declared song size back to zero');
for (const offset of [-1, 0.5, Number.POSITIVE_INFINITY]) {
  assert.throws(() => parseKX(bytes, offset), /invalid KX offset/);
}
assert.throws(() => parseKX(syntheticKX({
  packing: 'x'.repeat(257),
  convention: 0x020000ff,
  operationCount: 0,
})), /packing .* exceeds 256 bytes/,
'a malformed packing string is rejected before it can amplify graph allocations');
assert.throws(() => parseKX(syntheticKX({
  packing: '-'.repeat(256),
  convention: 0x020000ff,
  operationCount: 20,
})), /decoded parameter table is too large/,
'no-byte packing fields cannot amplify a small document into an unbounded parameter table');
assert.throws(() => parseKX(syntheticKX({
  convention: 0x02100000,
  splineCount: 1,
  operationSplineReference: 2,
})), /operator 0 has invalid spline target 1/);
assert.throws(() => parseKX(syntheticKX({
  splineCount: 1,
  eventSplineReference: 2,
})), /event 0 has invalid spline target 1/);

console.log(JSON.stringify(summary, null, 2));
