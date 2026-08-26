// Quality-0 BC3/DXT5 encoder ported from Farbrausch's rygdxt implementation.
// The original is BSD-licensed; retain vendor/wz3/LICENSE.txt with this port.
// Production Debris never requests rygdxt's quality-1 dithered path.

const f32 = Math.fround;
const REFINE_WEIGHT1 = new Int32Array([3, 0, 2, 1]);
const REFINE_PRODUCTS = new Int32Array([0x090000, 0x000900, 0x040102, 0x010402]);

let codec = null;

function mul8Bit(a, b) {
  const t = a * b + 128;
  return (t + (t >> 8)) >> 8;
}

function createCodec() {
  const expand5 = new Uint8Array(32);
  const expand6 = new Uint8Array(64);
  const match5 = new Uint8Array(256 * 2);
  const match6 = new Uint8Array(256 * 2);
  for (let i = 0; i < 32; i++) expand5[i] = (i << 3) | (i >> 2);
  for (let i = 0; i < 64; i++) expand6[i] = (i << 2) | (i >> 4);

  const prepareMatchTable = (table, expand, size) => {
    for (let value = 0; value < 256; value++) {
      let bestError = 256;
      for (let minimum = 0; minimum < size; minimum++) {
        for (let maximum = 0; maximum < size; maximum++) {
          const error = Math.abs(
            expand[maximum] + mul8Bit(expand[minimum] - expand[maximum], 0x55) - value,
          );
          if (error < bestError) {
            table[value * 2] = maximum;
            table[value * 2 + 1] = minimum;
            bestError = error;
          }
        }
      }
    }
  };
  prepareMatchTable(match5, expand5, 32);
  prepareMatchTable(match6, expand6, 64);

  return {
    expand5, expand6, match5, match6,
    colors: new Int32Array(4 * 3),
    means: new Int32Array(3),
    minimums: new Int32Array(3),
    maximums: new Int32Array(3),
    covariance: new Int32Array(6),
    dots: new Int32Array(16),
    stops: new Int32Array(4),
    endpoints: new Int32Array(2),
    alphaPalette: new Int32Array(8),
  };
}

function getCodec() {
  return codec || (codec = createCodec());
}

function colorAs565(block, offset) {
  const state = getCodec();
  return (mul8Bit(block[offset + 2], 31) << 11) |
    (mul8Bit(block[offset + 1], 63) << 5) |
    mul8Bit(block[offset], 31);
}

function evaluateColors(color0, color1) {
  const state = getCodec();
  const colors = state.colors;
  colors[0] = state.expand5[color0 & 31];
  colors[1] = state.expand6[(color0 >>> 5) & 63];
  colors[2] = state.expand5[(color0 >>> 11) & 31];
  colors[3] = state.expand5[color1 & 31];
  colors[4] = state.expand6[(color1 >>> 5) & 63];
  colors[5] = state.expand5[(color1 >>> 11) & 31];
  for (let channel = 0; channel < 3; channel++) {
    colors[6 + channel] = colors[channel] +
      mul8Bit(colors[3 + channel] - colors[channel], 0x55);
    colors[9 + channel] = colors[channel] +
      mul8Bit(colors[3 + channel] - colors[channel], 0xaa);
  }
}

function optimizeColors(block) {
  const state = getCodec();
  const means = state.means;
  const minimums = state.minimums;
  const maximums = state.maximums;
  for (let channel = 0; channel < 3; channel++) {
    let mean = block[channel];
    let minimum = mean;
    let maximum = mean;
    for (let offset = channel + 4; offset < 64; offset += 4) {
      const value = block[offset];
      mean += value;
      minimum = Math.min(minimum, value);
      maximum = Math.max(maximum, value);
    }
    means[channel] = (mean + 8) >> 4;
    minimums[channel] = minimum;
    maximums[channel] = maximum;
  }

  const covariance = state.covariance;
  covariance.fill(0);
  for (let i = 0; i < 16; i++) {
    const offset = i * 4;
    const r = block[offset + 2] - means[2];
    const g = block[offset + 1] - means[1];
    const b = block[offset] - means[0];
    covariance[0] += r * r;
    covariance[1] += r * g;
    covariance[2] += r * b;
    covariance[3] += g * g;
    covariance[4] += g * b;
    covariance[5] += b * b;
  }

  const c0 = f32(covariance[0] / 255);
  const c1 = f32(covariance[1] / 255);
  const c2 = f32(covariance[2] / 255);
  const c3 = f32(covariance[3] / 255);
  const c4 = f32(covariance[4] / 255);
  const c5 = f32(covariance[5] / 255);
  let vr = f32(maximums[2] - minimums[2]);
  let vg = f32(maximums[1] - minimums[1]);
  let vb = f32(maximums[0] - minimums[0]);
  for (let iteration = 0; iteration < 4; iteration++) {
    const r = f32(f32(f32(vr * c0) + f32(vg * c1)) + f32(vb * c2));
    const g = f32(f32(f32(vr * c1) + f32(vg * c3)) + f32(vb * c4));
    const b = f32(f32(f32(vr * c2) + f32(vg * c4)) + f32(vb * c5));
    vr = r; vg = g; vb = b;
  }

  let magnitude = f32(Math.max(Math.abs(vr), Math.abs(vg), Math.abs(vb)));
  let axisR, axisG, axisB;
  if (magnitude < 4) {
    axisR = 148; axisG = 300; axisB = 58;
  } else {
    magnitude = f32(512 / magnitude);
    axisR = Math.trunc(f32(vr * magnitude));
    axisG = Math.trunc(f32(vg * magnitude));
    axisB = Math.trunc(f32(vb * magnitude));
  }

  let minimumDot = 0x7fffffff;
  let maximumDot = -0x7fffffff;
  let minimumOffset = 0;
  let maximumOffset = 0;
  for (let i = 0; i < 16; i++) {
    const offset = i * 4;
    const dot = block[offset + 2] * axisR + block[offset + 1] * axisG +
      block[offset] * axisB;
    if (dot < minimumDot) { minimumDot = dot; minimumOffset = offset; }
    if (dot > maximumDot) { maximumDot = dot; maximumOffset = offset; }
  }
  state.endpoints[0] = colorAs565(block, maximumOffset);
  state.endpoints[1] = colorAs565(block, minimumOffset);
}

function matchColors(block) {
  const state = getCodec();
  const colors = state.colors;
  const directionB = colors[0] - colors[3];
  const directionG = colors[1] - colors[4];
  const directionR = colors[2] - colors[5];
  const dots = state.dots;
  for (let i = 0; i < 16; i++) {
    const offset = i * 4;
    dots[i] = block[offset + 2] * directionR + block[offset + 1] * directionG +
      block[offset] * directionB;
  }
  const stops = state.stops;
  for (let i = 0; i < 4; i++) {
    const offset = i * 3;
    stops[i] = colors[offset + 2] * directionR + colors[offset + 1] * directionG +
      colors[offset] * directionB;
  }
  const color0Point = (stops[1] + stops[3]) >> 1;
  const halfPoint = (stops[3] + stops[2]) >> 1;
  const color3Point = (stops[2] + stops[0]) >> 1;
  let mask = 0;
  for (let i = 15; i >= 0; i--) {
    mask <<= 2;
    const dot = dots[i];
    if (dot < halfPoint) mask |= dot < color0Point ? 1 : 3;
    else mask |= dot < color3Point ? 2 : 0;
  }
  return mask >>> 0;
}

function refineColors(block, mask) {
  const state = getCodec();
  let accumulator = 0;
  let a1r = 0, a1g = 0, a1b = 0;
  let a2r = 0, a2g = 0, a2b = 0;
  let colorMask = mask >>> 0;
  for (let i = 0; i < 16; i++, colorMask >>>= 2) {
    const step = colorMask & 3;
    const weight = REFINE_WEIGHT1[step];
    const offset = i * 4;
    const r = block[offset + 2], g = block[offset + 1], b = block[offset];
    accumulator += REFINE_PRODUCTS[step];
    a1r += weight * r; a1g += weight * g; a1b += weight * b;
    a2r += r; a2g += g; a2b += b;
  }
  a2r = 3 * a2r - a1r;
  a2g = 3 * a2g - a1g;
  a2b = 3 * a2b - a1b;
  const xx = accumulator >> 16;
  const yy = (accumulator >> 8) & 0xff;
  const xy = accumulator & 0xff;
  if (!yy || !xx || xx * yy === xy * xy) return false;

  const determinant = xx * yy - xy * xy;
  const factorRB = f32(f32(93 / 255) / determinant);
  const factorG = f32(f32(factorRB * 63) / 31);
  const oldMaximum = state.endpoints[0];
  const oldMinimum = state.endpoints[1];
  state.endpoints[0] =
    (solveEndpoint(a1r * yy - a2r * xy, factorRB, 31) << 11) |
    (solveEndpoint(a1g * yy - a2g * xy, factorG, 63) << 5) |
    solveEndpoint(a1b * yy - a2b * xy, factorRB, 31);
  state.endpoints[1] =
    (solveEndpoint(a2r * xx - a1r * xy, factorRB, 31) << 11) |
    (solveEndpoint(a2g * xx - a1g * xy, factorG, 63) << 5) |
    solveEndpoint(a2b * xx - a1b * xy, factorRB, 31);
  return oldMaximum !== state.endpoints[0] || oldMinimum !== state.endpoints[1];
}

function solveEndpoint(numerator, factor, maximum) {
  return Math.max(0, Math.min(maximum,
    Math.trunc(f32(f32(numerator * factor) + 0.5)),
  ));
}

function write16(output, offset, value) {
  output[offset] = value & 0xff;
  output[offset + 1] = value >>> 8;
}

function write32(output, offset, value) {
  value >>>= 0;
  output[offset] = value & 0xff;
  output[offset + 1] = (value >>> 8) & 0xff;
  output[offset + 2] = (value >>> 16) & 0xff;
  output[offset + 3] = value >>> 24;
}

function compressAlphaBlock(block, output, offset) {
  let minimum = block[3];
  let maximum = minimum;
  for (let i = 1; i < 16; i++) {
    const alpha = block[i * 4 + 3];
    minimum = Math.min(minimum, alpha);
    maximum = Math.max(maximum, alpha);
  }
  output[offset++] = maximum;
  output[offset++] = minimum;
  const distance = maximum - minimum;
  const bias = minimum * 7 - (distance >> 1);
  const distance4 = distance * 4;
  const distance2 = distance * 2;
  let bits = 0;
  let mask = 0;
  for (let i = 0; i < 16; i++) {
    let alpha = block[i * 4 + 3] * 7 - bias;
    let t = (distance4 - alpha) >> 31;
    let index = t & 4;
    alpha -= distance4 & t;
    t = (distance2 - alpha) >> 31;
    index += t & 2;
    alpha -= distance2 & t;
    t = (distance - alpha) >> 31;
    index += t & 1;
    index = (-index) & 7;
    index ^= Number(2 > index);
    mask |= index << bits;
    bits += 3;
    if (bits >= 8) {
      output[offset++] = mask & 0xff;
      mask >>= 8;
      bits -= 8;
    }
  }
}

function compressColorBlock(block, output, offset) {
  const state = getCodec();
  let constant = true;
  for (let i = 1; i < 16 && constant; i++) {
    for (let channel = 0; channel < 4; channel++) {
      if (block[i * 4 + channel] !== block[channel]) { constant = false; break; }
    }
  }

  let mask;
  if (!constant) {
    optimizeColors(block);
    if (state.endpoints[0] !== state.endpoints[1]) {
      evaluateColors(state.endpoints[0], state.endpoints[1]);
      mask = matchColors(block);
    } else mask = 0;
    if (refineColors(block, mask)) {
      if (state.endpoints[0] !== state.endpoints[1]) {
        evaluateColors(state.endpoints[0], state.endpoints[1]);
        mask = matchColors(block);
      } else mask = 0;
    }
  } else {
    const r = block[2], g = block[1], b = block[0];
    mask = 0xaaaaaaaa;
    state.endpoints[0] = (state.match5[r * 2] << 11) |
      (state.match6[g * 2] << 5) | state.match5[b * 2];
    state.endpoints[1] = (state.match5[r * 2 + 1] << 11) |
      (state.match6[g * 2 + 1] << 5) | state.match5[b * 2 + 1];
  }

  let maximum = state.endpoints[0];
  let minimum = state.endpoints[1];
  if (maximum < minimum) {
    const swap = maximum; maximum = minimum; minimum = swap;
    mask = (mask ^ 0x55555555) >>> 0;
  }
  write16(output, offset, maximum);
  write16(output, offset + 2, minimum);
  write32(output, offset + 4, mask);
}

// Input pixels use the native little-endian Pixel byte order: B, G, R, A.
function compressDxt5Block(block, output = new Uint8Array(16), offset = 0) {
  if (!block || block.length < 64) throw new RangeError('DXT5 input block needs 16 BGRA pixels');
  if (!output || offset < 0 || offset + 16 > output.length) {
    throw new RangeError('DXT5 output block needs 16 writable bytes');
  }
  getCodec();
  compressAlphaBlock(block, output, offset);
  compressColorBlock(block, output, offset + 8);
  return output;
}

// Decode to a tightly packed 4x4 RGBA block. Integer interpolation follows
// rygdxt's own rounded RGB palette; this is the deterministic fallback when
// the browser cannot expose BC3 sampling directly.
function decodeDxt5Block(block, output = new Uint8Array(64), offset = 0) {
  if (!block || offset < 0 || offset + 16 > block.length) {
    throw new RangeError('DXT5 input needs 16 bytes');
  }
  if (!output || output.length < 64) throw new RangeError('DXT5 decode output needs 64 bytes');
  const state = getCodec();
  const alpha0 = block[offset];
  const alpha1 = block[offset + 1];
  const alpha = state.alphaPalette;
  alpha[0] = alpha0; alpha[1] = alpha1;
  if (alpha0 > alpha1) {
    for (let i = 1; i <= 6; i++) alpha[i + 1] = Math.floor(((7 - i) * alpha0 + i * alpha1 + 3) / 7);
  } else {
    for (let i = 1; i <= 4; i++) alpha[i + 1] = Math.floor(((5 - i) * alpha0 + i * alpha1 + 2) / 5);
    alpha[6] = 0; alpha[7] = 255;
  }
  const alphaLow = block[offset + 2] | (block[offset + 3] << 8) |
    (block[offset + 4] << 16);
  const alphaHigh = block[offset + 5] | (block[offset + 6] << 8) |
    (block[offset + 7] << 16);

  const color0 = block[offset + 8] | (block[offset + 9] << 8);
  const color1 = block[offset + 10] | (block[offset + 11] << 8);
  evaluateColors(color0, color1);
  const colors = state.colors;
  const colorMask = (block[offset + 12] | (block[offset + 13] << 8) |
    (block[offset + 14] << 16) | (block[offset + 15] << 24)) >>> 0;
  for (let i = 0; i < 16; i++) {
    const colorIndex = (colorMask >>> (i * 2)) & 3;
    const alphaBits = i < 8 ? alphaLow : alphaHigh;
    const alphaIndex = (alphaBits >>> ((i & 7) * 3)) & 7;
    const source = colorIndex * 3;
    const destination = i * 4;
    output[destination] = colors[source + 2];
    output[destination + 1] = colors[source + 1];
    output[destination + 2] = colors[source];
    output[destination + 3] = alpha[alphaIndex];
  }
  return output;
}

function dxt5LevelByteLength(width, height) {
  width |= 0; height |= 0;
  if (width < 1 || height < 1) return 0;
  return ((width + 3) >> 2) * ((height + 3) >> 2) * 16;
}

function dxt5MipChainByteLength(width, height, levelCount) {
  width |= 0; height |= 0; levelCount |= 0;
  let total = 0;
  for (let level = 0; level < levelCount && width > 0 && height > 0; level++) {
    total += dxt5LevelByteLength(width, height);
    if (width <= 1 || height <= 1) break;
    width >>= 1; height >>= 1;
  }
  return total;
}

function dxt5MipLevelCount(bitmap) {
  const width = bitmap?.width | 0, height = bitmap?.height | 0;
  if (width < 1 || height < 1) return 0;
  if ((width & (width - 1)) || (height & (height - 1))) return 1;
  const widthPower = 31 - Math.clz32(width);
  const heightPower = 31 - Math.clz32(height);
  const requested = (bitmap.texMipCount ?? bitmap.TexMipCount ?? 0) | 0;
  if (requested === 0) return Math.min(widthPower, heightPower) + 1;
  return Math.max(1, Math.min(requested, widthPower, heightPower));
}

function reduceMip(source, sourceStride, width, height, destination, destinationStride, filter) {
  const nextWidth = width >> 1;
  const nextHeight = height >> 1;
  for (let y = 0; y < nextHeight; y++) {
    const row0 = y * 2 * sourceStride;
    const row1 = row0 + sourceStride;
    for (let x = 0; x < nextWidth; x++) {
      const source0 = (row0 + x * 2) * 4;
      const source1 = (row1 + x * 2) * 4;
      const target = (y * destinationStride + x) * 4;
      for (let channel = 0; channel < 4; channel++) {
        destination[target + channel] = filter
          ? (source[source0 + channel] + source[source0 + 4 + channel] +
            source[source1 + channel] + source[source1 + 4 + channel] + 2) >>> 2
          : source[source0 + channel];
      }
    }
  }
}

// The callback is synchronous and receives a reused view. WebGL consumes the
// view during texImage2D/compressedTexImage2D; callers that retain it must copy.
function forEachDxt5Mip(bitmap, outputKind, callback) {
  if (!bitmap?.data || !bitmap.width || !bitmap.height || typeof callback !== 'function') return 0;
  if (outputKind !== 'bc3' && outputKind !== 'rgba8') {
    throw new TypeError("DXT5 output kind must be 'bc3' or 'rgba8'");
  }
  const levelCount = dxt5MipLevelCount(bitmap);
  if (!levelCount) return 0;
  const thresholdWord = (bitmap.texMipThreshold ?? bitmap.TexMipTresh ?? 0) | 0;
  if ((thresholdWord >> 6) !== 0) {
    throw new RangeError('only production quality-0 DXT5 compression is supported');
  }
  const reverseThreshold = Boolean(thresholdWord & 16);
  const threshold = thresholdWord & 15;
  const baseWidth = bitmap.width | 0;
  const baseHeight = bitmap.height | 0;
  const workspaceStride = baseWidth >> 1;
  const mipWorkspace = levelCount > 1
    ? new Uint16Array(workspaceStride * (baseHeight >> 1) * 4) : null;
  const levelOutput = outputKind === 'bc3'
    ? new Uint8Array(dxt5LevelByteLength(baseWidth, baseHeight))
    : new Uint8Array(baseWidth * baseHeight * 4);
  const inputBlock = new Uint8Array(64);
  const encodedBlock = new Uint8Array(16);
  const decodedBlock = outputKind === 'rgba8' ? new Uint8Array(64) : null;
  let source = bitmap.data;
  let sourceStride = baseWidth;
  let width = baseWidth;
  let height = baseHeight;
  let level = 0;

  for (;;) {
    const blockColumns = (width + 3) >> 2;
    let blockIndex = 0;
    for (let y = 0; y < height; y += 4) {
      for (let x = 0; x < width; x += 4, blockIndex++) {
        let destination = 0;
        for (let blockY = 0; blockY < 4; blockY++) {
          const sourceY = (y + blockY) & (height - 1);
          const row = sourceY * sourceStride;
          for (let blockX = 0; blockX < 4; blockX++) {
            const sourceX = (x + blockX) & (width - 1);
            const sourceOffset = (row + sourceX) * 4;
            // Preserve rygdxt's native Pixel byte layout while the browser
            // Bitmap storage remains logical RGBA words.
            inputBlock[destination++] = (source[sourceOffset + 2] >>> 7) & 0xff;
            inputBlock[destination++] = (source[sourceOffset + 1] >>> 7) & 0xff;
            inputBlock[destination++] = (source[sourceOffset] >>> 7) & 0xff;
            inputBlock[destination++] = (source[sourceOffset + 3] >>> 7) & 0xff;
          }
        }
        if (outputKind === 'bc3') {
          compressDxt5Block(inputBlock, levelOutput, blockIndex * 16);
        } else {
          compressDxt5Block(inputBlock, encodedBlock);
          decodeDxt5Block(encodedBlock, decodedBlock);
          for (let blockY = 0; blockY < 4 && y + blockY < height; blockY++) {
            for (let blockX = 0; blockX < 4 && x + blockX < width; blockX++) {
              const sourceOffset = (blockY * 4 + blockX) * 4;
              const target = ((y + blockY) * width + x + blockX) * 4;
              levelOutput[target] = decodedBlock[sourceOffset];
              levelOutput[target + 1] = decodedBlock[sourceOffset + 1];
              levelOutput[target + 2] = decodedBlock[sourceOffset + 2];
              levelOutput[target + 3] = decodedBlock[sourceOffset + 3];
            }
          }
        }
      }
    }
    const byteLength = outputKind === 'bc3'
      ? blockColumns * ((height + 3) >> 2) * 16 : width * height * 4;
    callback(level, width, height, levelOutput.subarray(0, byteLength));
    level++;
    if (width <= 1 || height <= 1 || level >= levelCount) break;
    let filter = threshold <= level - 1;
    if (reverseThreshold) filter = !filter;
    if (source === bitmap.data) {
      reduceMip(source, sourceStride, width, height,
        mipWorkspace, workspaceStride, filter);
      source = mipWorkspace;
      sourceStride = workspaceStride;
    } else {
      reduceMip(source, sourceStride, width, height,
        source, sourceStride, filter);
    }
    width >>= 1;
    height >>= 1;
  }
  return level;
}

export {
  compressDxt5Block,
  decodeDxt5Block,
  dxt5LevelByteLength,
  dxt5MipChainByteLength,
  dxt5MipLevelCount,
  forEachDxt5Mip,
};
