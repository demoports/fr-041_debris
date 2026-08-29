// `_types.cpp`-level math and deterministic random numbers.

  const f32 = Math.fround;
  const scratch = new DataView(new ArrayBuffer(8));

  // The released player does not use the CRT trigonometric functions for
  // sMatrix::InitEuler. SSE_SinCos4 range-reduces four float lanes and then
  // evaluates this Remez polynomial with a rounded single-precision result
  // after every SSE instruction. Keeping the same rounding points is visible
  // in spline cameras and chained transforms, even though the approximation is
  // only about ten ULP away from Math.sin/Math.cos.
  const SSE_ONE_OVER_TWO_PI = f32(1 / f32(Math.PI * 2));
  const SSE_HALF_PI = f32(f32(Math.PI) * 0.5);
  const SSE_PI = f32(Math.PI);
  const SSE_TWO_PI = f32(Math.PI * 2);
  const SSE_SIN_C0 = f32(-1.854219680e-4);
  const SSE_SIN_C1 = f32(8.314273770e-3);
  const SSE_SIN_C2 = f32(-1.666585317e-1);
  const eulerSinCosScratch = new Float32Array(2);

  function mulF(a, b) { return f32(a * b); }
  function addF(a, b) { return f32(a + b); }
  function subF(a, b) { return f32(a - b); }

  function sseSinPolynomial(value) {
    const square = mulF(value, value);
    let polynomial = mulF(SSE_SIN_C0, square);
    polynomial = addF(polynomial, SSE_SIN_C1);
    polynomial = mulF(polynomial, square);
    polynomial = addF(polynomial, SSE_SIN_C2);
    polynomial = mulF(polynomial, square);
    polynomial = addF(polynomial, 1);
    return mulF(polynomial, value);
  }

  function sseSinCos(angle, out = new Float32Array(2)) {
    const input = f32(angle);
    const scaled = mulF(input, SSE_ONE_OVER_TWO_PI);
    // cvttps truncates, followed by subtracting one for every negative lane.
    // This deliberately maps exact negative multiples to 2*pi, as the source
    // assembly does, rather than using JavaScript's mathematical modulo.
    let period = f32(Math.trunc(scaled));
    period = subF(period, input < 0 ? 1 : 0);
    const reduced = subF(input, mulF(period, SSE_TWO_PI));

    const piDelta = subF(SSE_PI, reduced);
    const sineNegative = piDelta < 0 || Object.is(piDelta, -0);
    let quadrant = Math.abs(piDelta);
    const cosineNegative = quadrant < SSE_HALF_PI;
    quadrant = Math.min(quadrant, subF(SSE_PI, quadrant));

    let sine = sseSinPolynomial(quadrant);
    let cosine = sseSinPolynomial(subF(SSE_HALF_PI, quadrant));
    if (sineNegative) sine = -sine;
    if (cosineNegative) cosine = -cosine;
    out[0] = sine;
    out[1] = cosine;
    return out;
  }

  function f32ToBits(value) {
    scratch.setFloat32(0, value, true);
    return scratch.getUint32(0, true);
  }

  // Released _types.cpp sFPow is a handwritten x87 routine, not the CRT pow.
  // FYL2X and F2XM1 keep their transcendental result internally, while the
  // surrounding arithmetic observes the player's 24-bit x87 precision mode.
  // The FTST shortcut is especially visible: either signed zero is returned
  // unchanged for every exponent, including 0^0.
  function sFPow(base, exponent) {
    if (base === 0 || Number.isNaN(base)) return base;
    if (!(base > 0) || !Number.isFinite(base) || !Number.isFinite(exponent)) return 0;

    const logarithm = exponent * Math.log2(base);
    // FIST stores a signed dword. An out-of-range result becomes x87's integer
    // indefinite value, which the routine subsequently takes as underflow.
    if (!Number.isFinite(logarithm) ||
        logarithm < -0x80000000 - 0.5 || logarithm >= 0x7fffffff + 0.5) return 0;
    const lower = Math.floor(logarithm);
    const fractionFromLower = logarithm - lower;
    const integral = fractionFromLower < 0.5
      ? lower
      : fractionFromLower > 0.5
        ? lower + 1
        : (lower & 1) ? lower + 1 : lower;
    let scaleExponent = integral;
    if (integral >= 0x7fffc001) {
      // ADD EAX,0x3fff overflows for the top 16,383 valid FIST results. Its
      // flags bypass both range branches, so FLD reads the wrapped low 15 bits
      // as the handcrafted extended exponent (for example 2^INT_MAX -> 0.5).
      scaleExponent = ((integral + 0x3fff) & 0x7fff) - 0x3fff;
    } else {
      if (integral <= -0x3fff) return 0;
      if (integral >= 0x4000) return Infinity;
    }

    // FISUB and the add after F2XM1 each round at PC=24. Scaling by an exact
    // power of two changes only the exponent, so the returned Number retains
    // the same significand as the native sF64 return value.
    const fractional = f32(logarithm - integral);
    const mantissa = f32((Math.pow(2, fractional) - 1) + 1);
    return mantissa * Math.pow(2, scaleExponent);
  }

  function clamp(value, max, min) {
    return value >= max ? max : value <= min ? min : value;
  }

  function mulDiv(a, b, c) {
    return Math.trunc((a * b) / c) | 0;
  }

  function mulShift(a, b) {
    return Number((BigInt(a | 0) * BigInt(b | 0)) >> 16n) | 0;
  }

  function divShift(a, b) {
    return Number((BigInt(a | 0) << 16n) / BigInt(b | 0)) | 0;
  }

  // MSVCRT's two-step generator as used by sGetRnd in the released source.
  class Random {
    constructor(seed = 0x74382381) {
      this.seed = seed >>> 0;
    }

    setSeed(seed) {
      seed |= 0;
      // Keep the signed 32-bit overflow that occurs before the final division
      // in the original C expression.
      const seed121 = Math.imul(seed, 121) | 0;
      const mixed = (
        seed + Math.imul(seed, 17) + seed121 + Math.trunc(seed121 / 17)
      ) | 0;
      this.seed = mixed >>> 0;
      this.uint32();
      this.seed = (this.seed ^ mixed) >>> 0;
      this.uint32();
      this.seed = (this.seed ^ mixed) >>> 0;
      this.uint32();
      this.seed = (this.seed ^ mixed) >>> 0;
      this.uint32();
    }

    uint32() {
      const first = (Math.imul(this.seed, 0x343fd) + 0x269ec3) >>> 0;
      const second = (Math.imul(first, 0x343fd) + 0x269ec3) >>> 0;
      this.seed = second;
      return (((second >>> 10) & 0xffff) | ((first << 6) & 0xffff0000)) >>> 0;
    }

    int(max) {
      return max ? this.uint32() % max : 0;
    }

    float(max = 1) {
      return f32(((this.uint32() & 0x3fffffff) * max) / 0x40000000);
    }
  }

  function mat4Identity(out = new Float32Array(16)) {
    out.fill(0);
    out[0] = out[5] = out[10] = out[15] = 1;
    return out;
  }

  function mat4Copy(source, out = new Float32Array(16)) {
    out.set(source);
    return out;
  }

  // out = a * b in the original column-vector convention.
  function mat4Mul(a, b, out = new Float32Array(16)) {
    const t = out === a || out === b ? new Float32Array(16) : out;
    for (let column = 0; column < 4; column++) {
      const c = column * 4;
      const x = b[c], y = b[c + 1], z = b[c + 2], w = b[c + 3];
      // Released sMatrix::Mul4 keeps x87 at 24-bit precision and the compiler
      // schedules each row in this exact order. Preserve every rounded multiply
      // and add rather than evaluating one binary64 JavaScript dot product.
      t[c] = addF(
        addF(addF(mulF(a[4], y), mulF(a[12], w)), mulF(a[8], z)), mulF(a[0], x),
      );
      t[c + 1] = addF(
        addF(addF(mulF(a[1], x), mulF(a[5], y)), mulF(a[13], w)), mulF(a[9], z),
      );
      t[c + 2] = addF(
        addF(addF(mulF(a[6], y), mulF(a[14], w)), mulF(a[10], z)), mulF(a[2], x),
      );
      t[c + 3] = addF(
        addF(addF(mulF(a[15], w), mulF(a[11], z)), mulF(a[3], x)), mulF(a[7], y),
      );
    }
    if (t !== out) out.set(t);
    return out;
  }

  // sMatrix::MulA(a,b): despite its name/argument order this is b * a in
  // the column-vector representation shared by WebGL.
  function mat4MulA(a, b, out = new Float32Array(16)) {
    const t = out === a || out === b ? new Float32Array(16) : out;
    for (let column = 0; column < 4; column++) {
      const c = column * 4;
      const x = a[c], y = a[c + 1], z = a[c + 2];
      // Exact 0x811a10 player schedule: X accumulates K,J,I while Y and Z
      // accumulate I,K,J. The apparently unusual order is observable once the
      // x87 control word narrows every arithmetic instruction to 24 bits.
      t[c] = addF(addF(mulF(b[8], z), mulF(b[4], y)), mulF(b[0], x));
      t[c + 1] = addF(addF(mulF(b[1], x), mulF(b[9], z)), mulF(b[5], y));
      t[c + 2] = addF(addF(mulF(b[2], x), mulF(b[10], z)), mulF(b[6], y));
      t[c + 3] = 0;
    }
    t[12] = addF(t[12], b[12]);
    t[13] = addF(t[13], b[13]);
    t[14] = addF(t[14], b[14]);
    t[15] = 1;
    if (t !== out) out.set(t);
    return out;
  }

  // sMatrix::Mul3(a,b): b * a for the rotational 3x3 portion. The source uses
  // Scale3 followed by two AddScale3 calls, each of which stores a float32.
  function mat4Mul3(a, b, out = new Float32Array(16)) {
    const t = out === a || out === b ? new Float32Array(16) : out;
    mat4Identity(t);
    for (let column = 0; column < 3; column++) {
      const c = column * 4;
      const x = a[c], y = a[c + 1], z = a[c + 2];
      t[c] = addF(addF(mulF(b[0], x), mulF(b[4], y)), mulF(b[8], z));
      t[c + 1] = addF(addF(mulF(b[1], x), mulF(b[5], y)), mulF(b[9], z));
      t[c + 2] = addF(addF(mulF(b[2], x), mulF(b[6], y)), mulF(b[10], z));
      t[c + 3] = 0;
    }
    if (t !== out) out.set(t);
    return out;
  }

  // Matches sMatrix::InitEuler. Angles are radians and the basis vectors are
  // stored as the four columns i, j, k, l.
  function mat4Euler(x, y, z, out = new Float32Array(16)) {
    // Capture each pair before reusing the shared scratch. Euler evaluation is
    // synchronous, and avoiding three short-lived typed arrays here matters in
    // particle-heavy frames.
    sseSinCos(x, eulerSinCosScratch);
    const sx = eulerSinCosScratch[0], cx = eulerSinCosScratch[1];
    sseSinCos(y, eulerSinCosScratch);
    const sy = eulerSinCosScratch[0], cy = eulerSinCosScratch[1];
    sseSinCos(z, eulerSinCosScratch);
    const sz = eulerSinCosScratch[0], cz = eulerSinCosScratch[1];

    out[0] = mulF(cy, cz);
    out[1] = mulF(cy, sz);
    out[2] = f32(-sy);
    out[3] = 0;
    out[4] = addF(-mulF(cx, sz), mulF(mulF(sx, sy), cz));
    out[5] = addF(mulF(cx, cz), mulF(mulF(sx, sy), sz));
    out[6] = mulF(sx, cy);
    out[7] = 0;
    out[8] = addF(mulF(sx, sz), mulF(mulF(cx, sy), cz));
    out[9] = addF(-mulF(sx, cz), mulF(mulF(cx, sy), sz));
    out[10] = mulF(cx, cy);
    out[11] = 0;
    out[12] = out[13] = out[14] = 0;
    out[15] = 1;
    return out;
  }

  function mat4EulerTurns(turns, out = new Float32Array(16)) {
    const x = turns[0], y = turns[1], z = turns[2];
    if (Math.abs(x) < 1e-6 && Math.abs(y) < 1e-6 && Math.abs(z) < 1e-6) {
      return mat4Identity(out);
    }
    return mat4Euler(
      mulF(x, SSE_TWO_PI),
      mulF(y, SSE_TWO_PI),
      mulF(z, SSE_TWO_PI),
      out,
    );
  }

  function mat4SRT(srt, out = new Float32Array(16)) {
    mat4EulerTurns(srt.subarray ? srt.subarray(3, 6) : srt.slice(3, 6), out);
    const sx = srt[0], sy = srt[1], sz = srt[2];
    out[0] = f32(out[0] * sx);
    out[1] = f32(out[1] * sx);
    out[2] = f32(out[2] * sx);
    out[4] = f32(out[4] * sy);
    out[5] = f32(out[5] * sy);
    out[6] = f32(out[6] * sy);
    out[8] = f32(out[8] * sz);
    out[9] = f32(out[9] * sz);
    out[10] = f32(out[10] * sz);
    out[12] = f32(srt[6]);
    out[13] = f32(srt[7]);
    out[14] = f32(srt[8]);
    out[15] = 1;
    return out;
  }

  function mat4Direction(direction, out = new Float32Array(16)) {
    // sMatrix::InitDir uses UnitSafe3 for both basis vectors. Its epsilon is
    // applied to squared length, unlike the ordinary normalizer retained for
    // source call sites which use Unit3.
    const k = vec3NormalizeSafe(direction);
    const i = vec3NormalizeSafe(vec3Cross([0, 1, 0], k));
    const j = vec3Cross(k, i);
    out[0] = i[0]; out[1] = i[1]; out[2] = i[2]; out[3] = 0;
    out[4] = j[0]; out[5] = j[1]; out[6] = j[2]; out[7] = 0;
    out[8] = k[0]; out[9] = k[1]; out[10] = k[2]; out[11] = 0;
    out[12] = out[13] = out[14] = 0;
    out[15] = 1;
    return out;
  }

  class MatrixStack {
    constructor(initial, options = {}) {
      this.stack = [initial ? mat4Copy(initial) : mat4Identity()];
      // Public MatrixStack users may retain a matrix returned by push() after
      // popping it, so recycling is deliberately opt-in. Runtime traversal
      // snapshots deferred-job matrices before popping and can safely reuse
      // these short-lived stack levels.
      this._recycled = options?.recycle === true ? [] : null;
    }

    get top() {
      return this.stack[this.stack.length - 1];
    }

    get depth() {
      return this.stack.length;
    }

    _acquire() {
      return this._recycled?.pop() || new Float32Array(16);
    }

    _release(matrix) {
      if (this._recycled) this._recycled.push(matrix);
    }

    push(matrix) {
      const next = this._acquire();
      mat4Copy(matrix, next);
      this.stack.push(next);
      return this.top;
    }

    pushIdentity() {
      const next = this._acquire();
      mat4Identity(next);
      this.stack.push(next);
      return this.top;
    }

    pushMul(matrix) {
      // sMatrixStack::PushMul: MulA(matrix, top) == top * matrix.
      const next = this._acquire();
      mat4MulA(matrix, this.top, next);
      this.stack.push(next);
      return this.top;
    }

    duplicate() {
      return this.push(this.top);
    }

    pop() {
      if (this.stack.length > 1) this._release(this.stack.pop());
      return this.top;
    }

    reset() {
      if (this._recycled) {
        while (this.stack.length > 1) this._release(this.stack.pop());
      } else {
        this.stack.length = 1;
      }
      return this.top;
    }

    popAll() {
      this.reset();
      mat4Identity(this.stack[0]);
      return this.top;
    }

    summary() {
      return { depth: this.stack.length, top: Array.from(this.top) };
    }
  }

  function mat4TransformPoint(matrix, vector, out = new Float32Array(4)) {
    const x = vector[0], y = vector[1], z = vector[2];
    const w = vector.length > 3 ? vector[3] : 1;
    out[0] = f32(matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12] * w);
    out[1] = f32(matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13] * w);
    out[2] = f32(matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14] * w);
    out[3] = f32(matrix[3] * x + matrix[7] * y + matrix[11] * z + matrix[15] * w);
    return out;
  }

  function vec3Cross(a, b, out = new Float32Array(3)) {
    out[0] = f32(a[1] * b[2] - a[2] * b[1]);
    out[1] = f32(a[2] * b[0] - a[0] * b[2]);
    out[2] = f32(a[0] * b[1] - a[1] * b[0]);
    return out;
  }

  function vec3Normalize(vector, out = new Float32Array(3)) {
    const length = Math.sqrt(
      vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2],
    );
    if (length > 1e-20) {
      out[0] = f32(vector[0] / length);
      out[1] = f32(vector[1] / length);
      out[2] = f32(vector[2] / length);
    } else {
      out[0] = 1;
      out[1] = out[2] = 0;
    }
    return out;
  }

  function vec3NormalizeSafe(vector, out = new Float32Array(3)) {
    const lengthSquared =
      vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2];
    if (lengthSquared > 1e-20) {
      const inverseLength = 1 / Math.sqrt(lengthSquared);
      out[0] = f32(vector[0] * inverseLength);
      out[1] = f32(vector[1] * inverseLength);
      out[2] = f32(vector[2] * inverseLength);
    } else {
      out[0] = 1;
      out[1] = out[2] = 0;
    }
    return out;
  }

export {
  Random,
  MatrixStack,
  clamp,
  divShift,
  f32,
  f32ToBits,
  mat4Copy,
  mat4Direction,
  mat4Euler,
  mat4EulerTurns,
  mat4Identity,
  mat4Mul,
  mat4Mul3,
  mat4MulA,
  mat4SRT,
  mat4TransformPoint,
  mulDiv,
  mulShift,
  sFPow,
  sseSinCos,
  vec3Cross,
  vec3Normalize,
  vec3NormalizeSafe,
};
