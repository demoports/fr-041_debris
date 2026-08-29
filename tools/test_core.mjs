import assert from 'node:assert/strict';
import {
  MatrixStack,
  Random,
  f32ToBits,
  mat4Direction,
  mat4EulerTurns,
  mat4Identity,
  mat4Mul,
  mat4Mul3,
  mat4MulA,
  mat4SRT,
  mat4TransformPoint,
  sFPow,
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

// Direct calls to released _types.cpp sFPow (0x8106d0), captured under the
// player's 0x103f x87 control word. These positive inputs each differ by one
// ULP from a narrowed Math.pow result and exercise both sides of that error.
const powInputs = new Float32Array(Uint32Array.from([0x3d34e1c3, 0x3d6238a1]).buffer);
assert.equal(sFPow(powInputs[0], 0.5), 0.2101442813873291);
assert.equal(sFPow(powInputs[1], 0.5), 0.23501017689704895);
assert.equal(f32ToBits(sFPow(powInputs[0], 0.5)), 0x3e573010);
assert.equal(f32ToBits(sFPow(powInputs[1], 0.5)), 0x3e70a682);
// FTST takes the zero branch before looking at the exponent.
assert.equal(Object.is(sFPow(0, 0), 0), true);
assert.equal(Object.is(sFPow(0, -1), 0), true);
assert.equal(Object.is(sFPow(-0, 0), -0), true);
assert.equal(Object.is(sFPow(-0, -1), -0), true);
assert.equal(sFPow(-1, 2), 0);
assert.equal(sFPow(Infinity, 2), 0);
assert.equal(sFPow(2, 0x7fffffff), 0.5);
assert.equal(sFPow(2, 0x7fffffff + 0.5), 0);

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

// Direct released-executable oracles for sMatrix::Mul4 (0x811810) and MulA
// (0x811a10), captured with the player's 0x103f x87 control word. Besides the
// float32 boundaries, these inputs distinguish the compiler's row-specific
// accumulation order.
const nativeMatrixA = new Float32Array([
  1.234567, -2.345678, 3.456789, -4.567891,
  5.678912, -6.789123, 7.891234, -8.912345,
  9.123456, -1.234568, 2.345679, -3.456791,
  4.567912, -5.678923, 6.789134, -7.891245,
]);
const nativeMatrixB = new Float32Array([
  -8.765432, 7.654321, -6.543219, 5.432198,
  -4.321987, 3.219876, -2.198765, 1.987654,
  -0.876543, 9.765432, -8.654321, 7.543219,
  -6.432198, 5.321987, -4.219876, 3.198765,
]);
assert.deepEqual(Array.from(mat4Mul(nativeMatrixB, nativeMatrixA), f32ToBits), [
  0x41cd5820, 0x4135800c, 0xc158f888, 0x41581fa3,
  0x41efc968, 0x424cf3b8, 0xc253a8eb, 0x42417c45,
  0xc259d389, 0x428cbc8f, 0xc27ac7f6, 0x4256f8de,
  0x41ea7e24, 0x4223ec18, 0xc22b6e16, 0x421dfba7,
]);
assert.deepEqual(Array.from(mat4MulA(nativeMatrixA, nativeMatrixB), f32ToBits), [
  0xc06daac2, 0x420e9db7, 0xc20358af, 0,
  0xc1dad22d, 0x42c556ba, 0xc2b50c42, 0,
  0xc29961f7, 0x42b187d2, 0xc29a90a4, 0,
  0xc1df077e, 0x42b0996c, 0xc2a0c142, 0x3f800000,
]);

// sMatrix::Mul3 is a different executable routine (0x8135e0) whose
// Scale3/AddScale3 stores preserve ordinary I,J,K accumulation order.
const matrixFromBits = values => new Float32Array(Uint32Array.from(values).buffer);
const nativeRotationA = matrixFromBits([
  0x3dfcd6de, 0xbf43f35c, 0x3eaaaaaa, 0,
  0xbf9e064b, 0x40161f97, 0xc05d3c08, 0,
  0x40922c2a, 0xc0b5b9a6, 0x40d9407f, 0,
  0, 0, 0, 0x3f800000,
]);
const nativeRotationB = matrixFromBits([
  0xbf6b13f0, 0x3f53ce21, 0xbf3c8851, 0,
  0x3fd2a140, 0xbfc6fe58, 0x3fbb5b70, 0,
  0xc017dc44, 0x40120ad0, 0xc00c2a9d, 0,
  0, 0, 0, 0x3f800000,
]);
assert.deepEqual(Array.from(mat4Mul3(nativeRotationA, nativeRotationB), f32ToBits), [
  0xc00a7cb0, 0x4003603f, 0xbff87dc6, 0,
  0x41532282, 0xc148e62d, 0x413e9d1a, 0,
  0xc1ed30d4, 0x41e0cd0e, 0xc1d45cc3, 0,
  0, 0, 0, 0x3f800000,
]);

// Two reachable party Scene edges expose MulA rounding in authored data.
const scene3208 = mat4SRT(new Float32Array([1, 1, 1, 0, 0.5, 0, 0, 0, 18.25]));
const scene3209 = mat4SRT(new Float32Array([
  1, 1, 1, 0, 0.25, 0, -23.125, 27.5, -39,
]));
assert.equal(f32ToBits(mat4MulA(scene3208, scene3209)[12]), 0xc09c0084);
const productionStack = new MatrixStack(scene3209);
productionStack.pushMul(scene3208);
assert.equal(f32ToBits(productionStack.top[12]), 0xc09c0084);
const scene10845 = mat4SRT(new Float32Array([
  1, 1, 1, 0, 0.25, 0, -16.5, 0, 0.75,
]));
const scene10846 = mat4SRT(new Float32Array([
  1, 1, 1, 0, 0.5, 0, -12, 3.75, 26,
]));
assert.equal(f32ToBits(mat4MulA(scene10845, scene10846)[12]), 0x408fffb0);

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
