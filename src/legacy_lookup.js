// Lookup textures created by sSystem_::MakeCubeNormalizer and
// sSystem_::MakeAttenuationVolume in the released Werkkzeug3 source.
//
// The returned bytes are in WebGL's logical RGBA order. The native
// D3DFMT_A8R8G8B8 allocation stored those same channels as BGRA bytes in
// memory, which is why MakeCubeNormalizer writes z, y, x, alpha there.

const f32 = Math.fround;

export const LEGACY_NORMALIZER_CUBE_SIZE = 64;
export const LEGACY_NORMALIZER_FACE_BYTE_LENGTH =
  LEGACY_NORMALIZER_CUBE_SIZE * LEGACY_NORMALIZER_CUBE_SIZE * 4;
export const LEGACY_NORMALIZER_CUBE_BYTE_LENGTH =
  LEGACY_NORMALIZER_FACE_BYTE_LENGTH * 6;

// D3DCUBEMAP_FACES and WebGL's six cube targets use this same order. Upload
// each consecutive face to gl.TEXTURE_CUBE_MAP_POSITIVE_X + faceIndex.
export const LEGACY_NORMALIZER_FACE_ORDER = Object.freeze([
  'positive-x',
  'negative-x',
  'positive-y',
  'negative-y',
  'positive-z',
  'negative-z',
]);

export const LEGACY_ATTENUATION_VOLUME_SIZE = 32;
export const LEGACY_ATTENUATION_VOLUME_BYTE_LENGTH =
  LEGACY_ATTENUATION_VOLUME_SIZE ** 3 * 4;

// First and second in-plane axes from the native faces[6][2] table. Their
// cross product is the outward face axis. Keeping the table literal also
// locks the D3D/WebGL face orientation instead of relying on a convention.
const CUBE_FACE_BASES = Object.freeze([
  Object.freeze([0, 0, -1, 0, 1, 0]),
  Object.freeze([0, 0, 1, 0, 1, 0]),
  Object.freeze([1, 0, 0, 0, 0, -1]),
  Object.freeze([1, 0, 0, 0, 0, 1]),
  Object.freeze([1, 0, 0, 0, 1, 0]),
  Object.freeze([-1, 0, 0, 0, 1, 0]),
]);

function multiply(a, b) {
  return f32(f32(a) * f32(b));
}

function add(a, b) {
  return f32(f32(a) + f32(b));
}

// sFtol uses x87 FISTP under the source's default round-to-nearest-even
// control word. All generator inputs are finite and non-negative.
function roundToNearestEven(value) {
  const lower = Math.floor(value);
  const fraction = value - lower;
  if (fraction < 0.5) return lower;
  if (fraction > 0.5) return lower + 1;
  return lower + (lower & 1);
}

function assertOutput(output, byteLength, label) {
  if (!(output instanceof Uint8Array) || output.byteLength !== byteLength) {
    throw new TypeError(`${label} output must be a ${byteLength}-byte Uint8Array`);
  }
  return output;
}

/**
 * Generate all six 64x64 normalizer faces in native face order.
 *
 * Layout is face-major, then row-major y/x, then logical RGBA. With a typed
 * array upload, leave UNPACK_FLIP_Y_WEBGL false: the native top-to-bottom D3D
 * rows line up with WebGL's cube-coordinate face orientation this way.
 */
export function makeLegacyCubeNormalizer(
    output = new Uint8Array(LEGACY_NORMALIZER_CUBE_BYTE_LENGTH)) {
  assertOutput(output, LEGACY_NORMALIZER_CUBE_BYTE_LENGTH,
    'legacy cube normalizer');
  const size = LEGACY_NORMALIZER_CUBE_SIZE;
  const half = f32(f32(size - 1) * 0.5);
  let offset = 0;

  for (const basis of CUBE_FACE_BASES) {
    const ux = basis[0], uy = basis[1], uz = basis[2];
    const vx = basis[3], vy = basis[4], vz = basis[5];
    const nx = f32(uy * vz - uz * vy);
    const ny = f32(uz * vx - ux * vz);
    const nz = f32(ux * vy - uy * vx);

    for (let y = 0; y < size; y++) {
      const vScale = f32(-y + half);
      for (let x = 0; x < size; x++) {
        const uScale = f32(x - half);
        let px = add(multiply(nx, half), multiply(ux, uScale));
        let py = add(multiply(ny, half), multiply(uy, uScale));
        let pz = add(multiply(nz, half), multiply(uz, uScale));
        px = add(px, multiply(vx, vScale));
        py = add(py, multiply(vy, vScale));
        pz = add(pz, multiply(vz, vScale));

        const lengthSquared = add(add(multiply(px, px), multiply(py, py)),
          multiply(pz, pz));
        const inverseLength = f32(1 / Math.sqrt(lengthSquared));
        px = multiply(px, inverseLength);
        py = multiply(py, inverseLength);
        pz = multiply(pz, inverseLength);

        output[offset++] = roundToNearestEven(add(128, multiply(px, 127)));
        output[offset++] = roundToNearestEven(add(128, multiply(py, 127)));
        output[offset++] = roundToNearestEven(add(128, multiply(pz, 127)));
        output[offset++] = 0;
      }
    }
  }
  return output;
}

export function legacyCubeNormalizerFace(cube, faceIndex) {
  assertOutput(cube, LEGACY_NORMALIZER_CUBE_BYTE_LENGTH,
    'legacy cube normalizer');
  if (!Number.isInteger(faceIndex) || faceIndex < 0 || faceIndex >= 6) {
    throw new RangeError('legacy cube face index must be an integer from 0 to 5');
  }
  const offset = faceIndex * LEGACY_NORMALIZER_FACE_BYTE_LENGTH;
  return cube.subarray(offset, offset + LEGACY_NORMALIZER_FACE_BYTE_LENGTH);
}

/**
 * Generate the native 32x32x32 radial attenuation volume.
 *
 * Layout is x-fastest, then y, then z, with identical logical RGBA bytes.
 * WebGL integration uses a single-level RGBA8 TEXTURE_3D with LINEAR filters,
 * CLAMP_TO_EDGE on S/T/R, and coordinates
 *   0.5 + 0.5 * (modelPosition - modelLightPosition) / lightRange.
 */
export function makeLegacyAttenuationVolume(
    output = new Uint8Array(LEGACY_ATTENUATION_VOLUME_BYTE_LENGTH)) {
  assertOutput(output, LEGACY_ATTENUATION_VOLUME_BYTE_LENGTH,
    'legacy attenuation volume');
  const size = LEGACY_ATTENUATION_VOLUME_SIZE;
  const scale = f32(2 / f32(size - 2));
  const middle = f32(size / 2);
  let offset = 0;

  for (let z = 0; z < size; z++) {
    const vz = multiply(f32(z - middle), scale);
    for (let y = 0; y < size; y++) {
      const vy = multiply(f32(y - middle), scale);
      for (let x = 0; x < size; x++) {
        const vx = multiply(f32(x - middle), scale);
        const radiusSquared = add(add(multiply(vx, vx), multiply(vy, vy)),
          multiply(vz, vz));
        const attenuation = Math.max(f32(1 - radiusSquared), 0);
        const byte = roundToNearestEven(multiply(attenuation, 255));
        output[offset++] = byte;
        output[offset++] = byte;
        output[offset++] = byte;
        output[offset++] = byte;
      }
    }
  }
  return output;
}
