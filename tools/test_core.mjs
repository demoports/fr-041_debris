import assert from 'node:assert/strict';
import {
  MatrixStack,
  Random,
  f32ToBits,
  mat4EulerTurns,
  mat4Identity,
  mat4MulA,
  mat4SRT,
  mat4TransformPoint,
  sseSinCos,
} from '../src/core.js';

const random = new Random();
random.setSeed(340);
assert.deepEqual(
  [random.uint32(), random.uint32(), random.uint32(), random.uint32()],
  [0xe0d66cbe, 0x7bcd124b, 0x6fd0b046, 0xafdb0a35],
);

const identity = mat4Identity();
const srt = mat4SRT(new Float32Array([2, 3, 4, 0, 0, 0, 5, 6, 7]));
assert.deepEqual(Array.from(srt), [
  2, 0, 0, 0,
  0, 3, 0, 0,
  0, 0, 4, 0,
  5, 6, 7, 1,
]);

const rotation = mat4EulerTurns(new Float32Array([0.25, 0, 0]));
const rotated = mat4TransformPoint(rotation, new Float32Array([0, 1, 0, 0]));
assert.ok(Math.abs(rotated[0]) < 1e-6);
assert.ok(Math.abs(rotated[1]) < 1e-6);
// Released SSE_SinCos4 is a compact Remez approximation, so sin(pi/2) is
// intentionally not exactly one. Pin the assembly's float-lane results rather
// than silently replacing them with correctly-rounded host trigonometry.
assert.equal(f32ToBits(sseSinCos(0)[1]), 1065353197);
assert.deepEqual(Array.from(sseSinCos(Math.PI / 2), f32ToBits), [1065353197, 0]);
assert.deepEqual(Array.from(sseSinCos(-Math.PI / 2), f32ToBits),
  [3212836845, 3019898880]);
assert.equal(rotated[2], 0.9999977350234985);

assert.deepEqual(Array.from(mat4MulA(srt, identity)), Array.from(srt));
const stack = new MatrixStack();
stack.pushMul(srt);
assert.deepEqual(Array.from(stack.top), Array.from(srt));
stack.pop();
assert.deepEqual(Array.from(stack.top), Array.from(identity));

console.log('core numeric and matrix tests passed');
