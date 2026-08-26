import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';

import { Bitmap_Normals } from '../src/bitmap.js';
import { bitmapRGBA8 } from '../src/gl.js';
import { parseKX } from '../src/kx.js';
import { createOperatorHandlers } from '../src/operators.js';
import { Runtime } from '../src/runtime.js';

const document = parseKX(await readFile(
  new URL('../assets/debris_party.kx', import.meta.url),
));

// This is Debris' first complete authored normal-map path. The height map is
// genuinely coloured, so reading semantic red instead of native BGRA word 0
// (semantic blue) changes the result rather than passing accidentally through
// a grey generator.
assert.deepEqual(
  [document.operations[236].classId, document.operations[236].inputs],
  [0x24, [229, 235]],
);
assert.deepEqual(
  [document.operations[237].classId, document.operations[237].inputs,
    document.operations[237].parameters],
  [0x2e, [236], [0.75, 1]],
);
assert.deepEqual(
  [document.operations[245].classId, document.operations[245].inputs,
    document.operations[245].parameters],
  [0x24, [237, 244], [4]],
);
assert.deepEqual(
  [document.operations[246].classId, document.operations[246].inputs,
    document.operations[246].parameters],
  [0x25, [245], [5, 0, 0]],
);
assert.deepEqual(
  [document.operations[247].classId, document.operations[247].inputs],
  [0x30, [217, 246]],
);

const runtime = new Runtime(document, { handlers: createOperatorHandlers() });
const height = runtime.operations[236].precalc(runtime.environment);
let colouredPixels = 0;
for (let index = 0; index < height.data.length; index += 4) {
  if (height.data[index] !== height.data[index + 2]) colouredPixels++;
}
assert.equal(colouredPixels, 35725);
assert.equal(height.summary().hash, 2587439648);

// Swapping semantic R/B before the corrected blue-lane filter exactly
// simulates the former red-lane implementation and must not match production.
const redHeightSimulation = height.copy();
for (let index = 0; index < redHeightSimulation.data.length; index += 4) {
  const red = redHeightSimulation.data[index];
  redHeightSimulation.data[index] = redHeightSimulation.data[index + 2];
  redHeightSimulation.data[index + 2] = red;
}
assert.notEqual(
  Bitmap_Normals(redHeightSimulation, 0.75, 1).summary().hash,
  2457415038,
  'the production normal fixture distinguishes native blue height from red',
);

const normals = runtime.operations[237].precalc(runtime.environment);
assert.equal(normals.summary().hash, 2457415038);
assert.deepEqual(Array.from(normals.data.slice(0, 4)),
  [16213, 15988, 32761, 65535]);

// Merge mode 4 writes authored gloss into A without changing the X/Y/Z
// procedural normal words. Format then changes only upload policy.
const glossNormals = runtime.operations[245].precalc(runtime.environment);
assert.equal(glossNormals.summary().hash, 3460027563);
for (let index = 0; index < normals.data.length; index += 4) {
  assert.equal(glossNormals.data[index], normals.data[index]);
  assert.equal(glossNormals.data[index + 1], normals.data[index + 1]);
  assert.equal(glossNormals.data[index + 2], normals.data[index + 2]);
}
const signedNormals = runtime.operations[246].precalc(runtime.environment);
assert.equal(signedNormals.format, 5);
assert.equal(signedNormals.summary().hash, 3460027563);

// UpdateTexture(Q8W8V8U8) normalizes each raw X/Y/Z vector and writes words
// 2,1,0 as shader R,G,B. Existing GL unit fixtures pin the three basis axes;
// this digest pins the same swizzle on a complete 512x512 production map.
const upload = bitmapRGBA8(signedNormals);
assert.ok(upload instanceof Int8Array);
const uploadBytes = new Uint8Array(upload.buffer, upload.byteOffset, upload.byteLength);
assert.equal(createHash('sha256').update(uploadBytes).digest('hex'),
  'fa3af1a4c2ff79f8b29328b5e4187b5bade6621814b638e9fc9fe02ca70f2300');
assert.deepEqual(Array.from(upload.slice(0, 4)), [126, -3, -1, 7]);

// Bitmap_Bump consumes the pre-upload words in their original X/Y/Z order;
// it must not inherit the GPU-only 2,1,0 swizzle or any SNORM sign change.
const lit = runtime.operations[247].precalc(runtime.environment);
assert.equal(lit.summary().hash, 2928978459);
assert.deepEqual(Array.from(lit.data.slice(0, 4)), [269, 256, 310, 186]);

console.log('production normal -> gloss -> Q8W8V8U8/Bump pipeline passed');
