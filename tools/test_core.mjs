import assert from 'node:assert/strict';
import {
  MatrixStack,
  Random,
  f32ToBits,
  mat4Direction,
  mat4EulerTurns,
  mat4Identity,
  mat4MulA,
  mat4SRT,
  mat4TransformPoint,
  sseSinCos,
  vec3Normalize,
  vec3NormalizeSafe,
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

// Released InitEulerPI2 loads its 2*pi multiplier from a dword constant before
// multiplying the authored float turns. Operation 3337's 0.2998046875-turn Y
// rotation distinguishes that path from multiplying by JavaScript's binary64
// Math.PI * 2 and narrowing only the product.
const productionRotation = mat4EulerTurns(new Float32Array([0, 0.2998046875, 0]));
assert.deepEqual(Array.from(productionRotation, f32ToBits), [
  0xbe9d9e74, 0x80000000, 0xbf739136, 0,
  0, 0x3f7fffda, 0x80000000, 0,
  0x3f739112, 0, 0xbe9d9e74, 0,
  0, 0, 0, 0x3f800000,
]);

assert.deepEqual(Array.from(mat4MulA(srt, identity)), Array.from(srt));

// UnitSafe3 compares squared length against 1e-20 and falls back to +X.
// Keep it separate from Unit3-style call sites, which still normalize this
// small but nonzero vector.
assert.deepEqual(Array.from(vec3Normalize([0, 1e-12, 0])), [0, 1, 0]);
assert.deepEqual(Array.from(vec3NormalizeSafe([0, 1e-12, 0])), [1, 0, 0]);
assert.deepEqual(Array.from(vec3NormalizeSafe([0, 1e-8, 0])), [0, 1, 0]);
assert.deepEqual(Array.from(mat4Direction([0, 1e-12, 0])), [
  0, 0, -1, 0,
  -0, 1, 0, 0,
  1, 0, 0, 0,
  0, 0, 0, 1,
]);
const stack = new MatrixStack();
stack.pushMul(srt);
assert.deepEqual(Array.from(stack.top), Array.from(srt));
stack.pop();
assert.deepEqual(Array.from(stack.top), Array.from(identity));

// Recycling remains opt-in because public callers may retain a pushed matrix
// after popping it. The default stack must never overwrite that reference.
{
  const defaultStack = new MatrixStack();
  const retained = defaultStack.push(srt);
  const retainedBits = new Uint32Array(retained.buffer, retained.byteOffset, 16).slice();
  defaultStack.pop();
  const replacement = defaultStack.pushIdentity();
  assert.notEqual(replacement, retained);
  assert.deepEqual(
    new Uint32Array(retained.buffer, retained.byteOffset, 16),
    retainedBits,
  );
}

// An opted-in stack keeps at most one reusable matrix per reached depth. All
// active levels remain distinct, and every operation retains its exact float
// bits when a recycled destination is used.
{
  const initial = mat4SRT(new Float32Array([
    1.25, -2.5, 0.75, 0.125, -0.25, 0.375, 4, -5, 6,
  ]));
  const operand = mat4SRT(new Float32Array([
    -0.5, 3, 1.5, -0.2, 0.3, -0.4, -7, 8, -9,
  ]));
  const expected = new MatrixStack(initial);
  const recycled = new MatrixStack(initial, { recycle: true });
  const bits = matrix => new Uint32Array(
    matrix.buffer, matrix.byteOffset, matrix.length,
  );
  const runSequence = () => {
    const expectedLevels = [
      expected.push(operand),
      expected.pushIdentity(),
      expected.pushMul(operand),
    ];
    const recycledLevels = [
      recycled.push(operand),
      recycled.pushIdentity(),
      recycled.pushMul(operand),
    ];
    for (let index = 0; index < expectedLevels.length; index++) {
      assert.deepEqual(bits(recycledLevels[index]), bits(expectedLevels[index]));
    }
    assert.equal(new Set(recycled.stack).size, recycled.depth,
      'no recycled matrix aliases another active stack level');
    return recycledLevels;
  };

  const firstLevels = runSequence();
  expected.reset();
  recycled.reset();
  assert.equal(recycled.depth, 1);
  const resetLevels = runSequence();
  for (let index = 0; index < firstLevels.length; index++) {
    assert.equal(resetLevels[index], firstLevels[index],
      'reset releases every inactive level for reuse at its previous depth');
  }

  expected.popAll();
  recycled.popAll();
  assert.deepEqual(bits(recycled.top), bits(mat4Identity()));
  const popAllLevels = runSequence();
  for (let index = 0; index < firstLevels.length; index++) {
    assert.equal(popAllLevels[index], firstLevels[index],
      'popAll releases every inactive level while restoring the root identity');
  }
}

console.log('core numeric and matrix tests passed');
