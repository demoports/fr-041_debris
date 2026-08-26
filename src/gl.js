// Small WebGL2 foundation shared by Debris' mesh, material and IPP renderers.
// The helpers are deliberately independent of the DOM so their numerical
// parts can be checked in Node and a lost context can be rebuilt cleanly.
import { f32, mat4Identity, mat4Mul } from './core.js';
import {
  dxt5MipChainByteLength,
  forEachDxt5Mip,
} from './dxt5.js';

function colorARGB(value, out = new Float32Array(4)) {
  value >>>= 0;
  out[0] = ((value >>> 16) & 255) / 255;
  out[1] = ((value >>> 8) & 255) / 255;
  out[2] = (value & 255) / 255;
  out[3] = (value >>> 24) / 255;
  return out;
}

function colorRGB(value, out = new Float32Array(3)) {
  value >>>= 0;
  out[0] = ((value >>> 16) & 255) / 255;
  out[1] = ((value >>> 8) & 255) / 255;
  out[2] = (value & 255) / 255;
  return out;
}

// General column-major inverse. Scene matrices are normally affine, but the
// full form also makes this useful for view-projection and texture matrices.
function mat4Inverse(a, out = new Float32Array(16)) {
  const a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3];
  const a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
  const a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11];
  const a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];
  const b00 = a00 * a11 - a01 * a10;
  const b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11;
  const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30;
  const b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31;
  const b11 = a22 * a33 - a23 * a32;
  let determinant = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (!Number.isFinite(determinant) || Math.abs(determinant) < 1e-30) return null;
  determinant = 1 / determinant;
  out[0] = f32((a11 * b11 - a12 * b10 + a13 * b09) * determinant);
  out[1] = f32((a02 * b10 - a01 * b11 - a03 * b09) * determinant);
  out[2] = f32((a31 * b05 - a32 * b04 + a33 * b03) * determinant);
  out[3] = f32((a22 * b04 - a21 * b05 - a23 * b03) * determinant);
  out[4] = f32((a12 * b08 - a10 * b11 - a13 * b07) * determinant);
  out[5] = f32((a00 * b11 - a02 * b08 + a03 * b07) * determinant);
  out[6] = f32((a32 * b02 - a30 * b05 - a33 * b01) * determinant);
  out[7] = f32((a20 * b05 - a22 * b02 + a23 * b01) * determinant);
  out[8] = f32((a10 * b10 - a11 * b08 + a13 * b06) * determinant);
  out[9] = f32((a01 * b08 - a00 * b10 - a03 * b06) * determinant);
  out[10] = f32((a30 * b04 - a31 * b02 + a33 * b00) * determinant);
  out[11] = f32((a21 * b02 - a20 * b04 - a23 * b00) * determinant);
  out[12] = f32((a11 * b07 - a10 * b09 - a12 * b06) * determinant);
  out[13] = f32((a00 * b09 - a01 * b07 + a02 * b06) * determinant);
  out[14] = f32((a31 * b01 - a30 * b03 - a32 * b00) * determinant);
  out[15] = f32((a20 * b03 - a21 * b01 + a22 * b00) * determinant);
  return out;
}

// WZ3's camera looks down its local +Z axis and deliberately uses q=1 rather
// than FarClip/(FarClip-NearClip), giving perspective an infinite far plane.
// WebGL's clip volume is -w..w, so native z'=z-near becomes z'=z-2*near here.
function legacyProjection(camera, out = new Float32Array(16)) {
  const near = Math.max(1e-5, Math.abs(camera.nearClip || 0.125));
  out.fill(0);
  out[0] = f32(camera.zoomX || 1);
  out[5] = f32(camera.zoomY || 1);
  out[8] = f32(camera.centerX || 0);
  out[9] = f32(camera.centerY || 0);
  out[10] = 1;
  out[11] = 1;
  out[14] = f32(-2 * near);
  // The released D3D9 MakeProjectionMatrix also writes -1/ViewportX and
  // +1/ViewportY into the projection translation. Those are its rasterizer
  // half-pixel compensation, not authored camera offsets: D3D9 samples at
  // integer pixel centers, while WebGL samples at half-integers. Leaving
  // out[12]/out[13] at zero is therefore the API conversion. Reintroducing
  // the native terms would apply the half-pixel correction twice.
  return out;
}

function cameraMatrices(camera) {
  const view = mat4Inverse(camera.cameraSpace) || mat4Identity();
  const projection = legacyProjection(camera);
  return { view, projection, viewProjection: mat4Mul(projection, view) };
}

function normalMatrix3(model, out = new Float32Array(9)) {
  const inverse = mat4Inverse(model);
  if (!inverse) {
    out.fill(0); out[0] = out[4] = out[8] = 1;
    return out;
  }
  // transpose(inverse(model)), packed column-major as mat3.
  out[0] = inverse[0]; out[1] = inverse[4]; out[2] = inverse[8];
  out[3] = inverse[1]; out[4] = inverse[5]; out[5] = inverse[9];
  out[6] = inverse[2]; out[7] = inverse[6]; out[8] = inverse[10];
  return out;
}

function bitmapLevelRGBA8(source, width, height, signedNormal = false) {
  if (!source || !width || !height) return null;
  // D3D9 Q8W8V8U8 is the signed-normalized counterpart of ordinary RGBA8.
  // Keep its bytes signed all the way to WebGL's RGBA8_SNORM upload; biasing
  // them into RGBA8_UNORM cannot reproduce filtering involving -128 exactly.
  const result = signedNormal
    ? new Int8Array(width * height * 4)
    : new Uint8Array(width * height * 4);
  // UpdateTexture's unsigned paths take bits 7..14 verbatim. In particular,
  // this is not a normalized 0..0x7fff conversion: intermediate GenBitmap
  // values on opposite sides of a 128-word boundary must reach different
  // bytes, and wrapped 0xffff alpha still becomes 255. Dispatch once per
  // level rather than testing signedNormal for every pixel.
  if (!signedNormal) {
    for (let i = 0; i < result.length; i += 4) {
      result[i] = (source[i] >>> 7) & 0xff;
      result[i + 1] = (source[i + 1] >>> 7) & 0xff;
      result[i + 2] = (source[i + 2] >>> 7) & 0xff;
      result[i + 3] = (source[i + 3] >>> 7) & 0xff;
    }
    return result;
  }
  // UpdateTexture's Q8W8V8U8 path casts the result of a float32 multiply to
  // sInt. Keep that final rounding before truncation: using JavaScript's
  // double product changes a byte for vectors that lie on a quantizer edge.
  const signedChannel = value => Math.max(-128,
    Math.min(127, Math.trunc(f32(value * 127))));
  // Q8W8V8U8 alpha is quantized independently rather than normalized with
  // XYZ. Int8Array assignment performs the source's signed-byte wrap for
  // words above 0x7fff, preserving both authored -128 and +127 endpoints.
  const signedWordChannel = value => (value - 0x4000) >> 7;
  for (let i = 0; i < result.length; i += 4) {
    // UpdateTexture(sTF_Q8W8V8U8) writes GenBitmap's words 2,1,0,3 into
    // the native U,V,W,Q bytes. D3D therefore exposes Bitmap_Normals'
    // logical (vz,vy,vx) as shader RGB. Material11/20 intentionally use
    // that ordering because their tangent basis is N,B,T rather than T,B,N.
    // The non-intro player also calls UnitSafe3 before its signed-byte
    // conversion. Store those same trunc(normal*127) values as signed bytes;
    // WebGL's SNORM sampler performs the native signed normalization before
    // filtering, including the duplicated -128/-127 value at -1.
    let x = source[i + 2] - 0x4000;
    let y = source[i + 1] - 0x4000;
    let z = source[i] - 0x4000;
    // sVector::UnitSafe3 stores Dot3, the reciprocal-length scale and the
    // three scaled components as sF32. Spell out those rounding points
    // instead of normalizing through Math.hypot's all-double path.
    const lengthSquared = f32(f32(f32(x * x) + f32(y * y)) + f32(z * z));
    if (lengthSquared > 1e-20) {
      const inverseLength = f32(1 / Math.sqrt(lengthSquared));
      x = f32(x * inverseLength);
      y = f32(y * inverseLength);
      z = f32(z * inverseLength);
    } else {
      x = 1; y = 0; z = 0;
    }
    result[i] = signedChannel(x);
    result[i + 1] = signedChannel(y);
    result[i + 2] = signedChannel(z);
    // Alpha is not part of UnitSafe3. Native Q8W8V8U8 exposes it as signed
    // Q, which M20 consumes as the authored gloss/specular multiplier.
    result[i + 3] = signedWordChannel(source[i + 3]);
  }
  return result;
}

function validBitmapUpload(bitmap) {
  if (!bitmap || typeof bitmap !== 'object') return false;
  const width = bitmap.width, height = bitmap.height, data = bitmap.data;
  if (!Number.isSafeInteger(width) || width < 1 ||
      !Number.isSafeInteger(height) || height < 1 ||
      !(data instanceof Uint16Array)) return false;
  const wordLength = width * height * 4;
  return Number.isSafeInteger(wordLength) && data.length === wordLength;
}

function bitmapRGBA8(bitmap) {
  if (!validBitmapUpload(bitmap)) return null;
  return bitmapLevelRGBA8(bitmap.data, bitmap.width, bitmap.height,
    (bitmap.format ?? bitmap.Format) === 5);
}

function nativeMipLevelCount(bitmap) {
  const width = bitmap?.width | 0, height = bitmap?.height | 0;
  if (width < 1 || height < 1) return 0;
  // AddTexture forces non-power-of-two textures to one level. Generated
  // Debris textures are power-of-two, but retaining this rule also keeps a
  // manually uploaded WebGL texture complete.
  if ((width & (width - 1)) || (height & (height - 1))) return 1;
  const widthPower = 31 - Math.clz32(width);
  const heightPower = 31 - Math.clz32(height);
  const requested = (bitmap.texMipCount ?? bitmap.TexMipCount ?? 0) | 0;
  // A zero D3D level count allocates the complete chain. Explicit counts are
  // clamped by AddTexture to sGetPower2 of the shorter dimension.
  if (requested === 0) return Math.min(widthPower, heightPower) + 1;
  return Math.max(1, Math.min(requested, widthPower, heightPower));
}

function rgba8MipChainByteLength(width, height, levelCount) {
  let total = 0;
  for (let level = 0; level < levelCount && width > 0 && height > 0; level++) {
    total += width * height * 4;
    if (width <= 1 || height <= 1) break;
    width >>= 1; height >>= 1;
  }
  return total;
}

// UpdateTexture does not ask D3D to derive mipmaps from already quantized
// texels. It box-filters the original 16-bit GenBitmap words, then converts
// every level to the requested hardware format. For Q8W8V8U8 that conversion
// also repeats the swizzle and UnitSafe3 normalization. This distinction is
// especially important for Material20's single bump-map path, but it also
// preserves color detail and TexMipTresh's authored point-pick levels.
//
// Invoke the callback synchronously and release each converted/upload level
// before allocating its successor. Only the current 16-bit source and one
// quarter-sized successor are retained; the complete mip chain is never kept.
function forEachSourceWordMip(bitmap, signedNormal, callback) {
  if (!validBitmapUpload(bitmap) || typeof callback !== 'function') return 0;
  let source = bitmap.data;
  const baseWidth = bitmap.width | 0, baseHeight = bitmap.height | 0;
  let width = baseWidth, height = baseHeight, sourceStride = baseWidth;
  const levelCount = nativeMipLevelCount(bitmap);
  if (!levelCount) return 0;
  const mipThresholdWord = (bitmap.texMipThreshold ?? bitmap.TexMipTresh ?? 0) | 0;
  const reverseThreshold = Boolean(mipThresholdWord & 16);
  const mipThreshold = mipThresholdWord & 15;
  // As in the DXT path, one quarter-sized workspace is enough for every
  // successor. Later levels compact toward its start in place, avoiding a
  // fresh Uint16Array (and later GC) for every mip.
  const mipWorkspace = levelCount > 1
    ? new Uint16Array((baseWidth >> 1) * (baseHeight >> 1) * 4)
    : null;
  let level = 0;

  for (;;) {
    let pixels = bitmapLevelRGBA8(source, width, height, signedNormal);
    callback(level, width, height, pixels);
    // Drop the converted bytes before allocating the next 16-bit level. The
    // GL call above has consumed them synchronously.
    pixels = null;
    level++;
    if (width <= 1 || height <= 1 || level >= levelCount) break;

    const nextWidth = width >> 1, nextHeight = height >> 1;
    let filter = mipThreshold <= level - 1;
    if (reverseThreshold) filter = !filter;
    for (let y = 0; y < nextHeight; y++) {
      const row0 = y * 2 * sourceStride;
      const row1 = row0 + sourceStride;
      for (let x = 0; x < nextWidth; x++) {
        const source0 = (row0 + x * 2) * 4;
        const source1 = (row1 + x * 2) * 4;
        const destination = (y * nextWidth + x) * 4;
        if (filter) {
          mipWorkspace[destination] = (source[source0] + source[source0 + 4] +
            source[source1] + source[source1 + 4] + 2) >>> 2;
          mipWorkspace[destination + 1] = (source[source0 + 1] + source[source0 + 5] +
            source[source1 + 1] + source[source1 + 5] + 2) >>> 2;
          mipWorkspace[destination + 2] = (source[source0 + 2] + source[source0 + 6] +
            source[source1 + 2] + source[source1 + 6] + 2) >>> 2;
          mipWorkspace[destination + 3] = (source[source0 + 3] + source[source0 + 7] +
            source[source1 + 3] + source[source1 + 7] + 2) >>> 2;
        } else {
          mipWorkspace[destination] = source[source0];
          mipWorkspace[destination + 1] = source[source0 + 1];
          mipWorkspace[destination + 2] = source[source0 + 2];
          mipWorkspace[destination + 3] = source[source0 + 3];
        }
      }
    }
    source = mipWorkspace;
    sourceStride = nextWidth;
    width = nextWidth;
    height = nextHeight;
  }
  return level;
}

function forEachBitmapMip(bitmap, callback) {
  return forEachSourceWordMip(bitmap, false, callback);
}

function forEachSignedNormalMip(bitmap, callback) {
  return forEachSourceWordMip(bitmap, true, callback);
}

function numberedSource(source) {
  return source.split('\n').map((line, index) => `${String(index + 1).padStart(3)}: ${line}`).join('\n');
}

function compileShader(gl, type, source, label = 'shader') {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader) || 'unknown shader compiler error';
    gl.deleteShader(shader);
    throw new Error(`${label}: ${log}\n${numberedSource(source)}`);
  }
  return shader;
}

function linkProgram(gl, vertexSource, fragmentSource, label = 'program') {
  const vertex = compileShader(gl, gl.VERTEX_SHADER, vertexSource, `${label} vertex shader`);
  const fragment = compileShader(gl, gl.FRAGMENT_SHADER, fragmentSource, `${label} fragment shader`);
  const program = gl.createProgram();
  gl.attachShader(program, vertex);
  gl.attachShader(program, fragment);
  gl.linkProgram(program);
  gl.deleteShader(vertex);
  gl.deleteShader(fragment);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(program) || 'unknown program linker error';
    gl.deleteProgram(program);
    throw new Error(`${label}: ${log}`);
  }
  return program;
}

function uniformLocations(gl, program, names) {
  const result = Object.create(null);
  for (const name of names) result[name] = gl.getUniformLocation(program, name);
  return result;
}

class TextureCache {
  constructor(gl, options = {}) {
    this.gl = gl;
    this.anisotropy = gl.getExtension?.('EXT_texture_filter_anisotropic') ||
      gl.getExtension?.('WEBKIT_EXT_texture_filter_anisotropic') ||
      gl.getExtension?.('MOZ_EXT_texture_filter_anisotropic') || null;
    this.maxAnisotropy = this.anisotropy
      ? Math.max(1, Number(gl.getParameter(this.anisotropy.MAX_TEXTURE_MAX_ANISOTROPY_EXT)) || 1)
      : 1;
    const dxt5Mode = String(options.dxt5Mode || 'auto').toLowerCase();
    if (dxt5Mode !== 'auto' && dxt5Mode !== 's3tc' && dxt5Mode !== 'rgba8') {
      throw new RangeError("dxt5Mode must be 'auto', 's3tc', or 'rgba8'");
    }
    this.dxt5Mode = dxt5Mode;
    this.s3tc = dxt5Mode === 'rgba8' ? null
      : gl.getExtension?.('WEBGL_compressed_texture_s3tc') || null;
    if (dxt5Mode === 's3tc' && !this.s3tc) {
      throw new Error("dxt5Mode 's3tc' requires WEBGL_compressed_texture_s3tc");
    }
    this.entries = new WeakMap();
    this.textureHandles = new Set();
    // Texture sampling parameters are object state in WebGL. Materials bind the
    // same generated bitmap many times per frame, so remember what is already
    // installed instead of repeating four texParameteri calls (and an
    // anisotropy call) for every draw. A Map is appropriate here because
    // textureHandles already owns every handle until dispose().
    this.samplerStates = new Map();
    this.entryCount = 0;
    this.dxt5CompressedCount = 0;
    this.dxt5FallbackCount = 0;
    this.estimatedBytes = 0;
    this.fallback = null;
  }

  setSamplerParameter(texture, name, value, floating = false) {
    let state = this.samplerStates.get(texture);
    if (!state) this.samplerStates.set(texture, state = new Map());
    if (state.get(name) === value) return false;
    if (floating) this.gl.texParameterf(this.gl.TEXTURE_2D, name, value);
    else this.gl.texParameteri(this.gl.TEXTURE_2D, name, value);
    state.set(name, value);
    return true;
  }

  fallbackTexture() {
    if (this.fallback) return this.fallback;
    const gl = this.gl;
    const texture = gl.createTexture();
    this.textureHandles.add(texture);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
      new Uint8Array([255, 255, 255, 255]));
    this.setSamplerParameter(texture, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    this.setSamplerParameter(texture, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    this.fallback = texture;
    return texture;
  }

  retireBitmapEntry(bitmap, entry) {
    if (!entry) return false;
    this.entries.delete(bitmap);
    this.textureHandles.delete(entry.texture);
    this.samplerStates.delete(entry.texture);
    this.gl.deleteTexture(entry.texture);
    this.entryCount--;
    this.estimatedBytes -= entry.byteLength || 0;
    if (entry.storageKind === 'dxt5') this.dxt5CompressedCount--;
    else if (entry.storageKind === 'dxt5-rgba8') this.dxt5FallbackCount--;
    return true;
  }

  estimatedUploadBytes(bitmap) {
    if (!validBitmapUpload(bitmap)) return this.fallback ? 0 : 4;
    const mipLevels = nativeMipLevelCount(bitmap);
    const mipmapped = mipLevels > 1;
    const format = bitmap.format ?? bitmap.Format ?? 1;
    const mipThreshold = (bitmap.texMipThreshold ?? bitmap.TexMipTresh ?? 0) | 0;
    const storageKind = format === 9 ? (this.s3tc ? 'dxt5' : 'dxt5-rgba8') : 'rgba8';
    const entry = this.entries.get(bitmap);
    if (entry && entry.data === bitmap.data && entry.width === bitmap.width &&
      entry.height === bitmap.height && entry.mipmapped === mipmapped &&
      entry.mipLevels === mipLevels && entry.mipThreshold === mipThreshold &&
      entry.format === format && entry.storageKind === storageKind) return 0;
    return storageKind === 'dxt5'
      ? dxt5MipChainByteLength(bitmap.width, bitmap.height, mipLevels)
      : rgba8MipChainByteLength(bitmap.width, bitmap.height, mipLevels);
  }

  get(bitmap) {
    // Validate the complete source shape before allocating a WebGL object.
    // Released graph inputs keep their dimensions after ownership transfer but
    // expose an empty Uint16Array; those must bind the shared fallback rather
    // than leave an untracked, incomplete texture behind.
    let entry = bitmap && typeof bitmap === 'object' ? this.entries.get(bitmap) : null;
    if (!validBitmapUpload(bitmap)) {
      if (entry) this.retireBitmapEntry(bitmap, entry);
      return this.fallbackTexture();
    }
    const mipLevels = nativeMipLevelCount(bitmap);
    const mipmapped = mipLevels > 1;
    const format = bitmap.format ?? bitmap.Format ?? 1;
    const mipThreshold = (bitmap.texMipThreshold ?? bitmap.TexMipTresh ?? 0) | 0;
    const storageKind = format === 9 ? (this.s3tc ? 'dxt5' : 'dxt5-rgba8') : 'rgba8';
    // forEachDxt5Mip rejects non-production compression quality. Do that
    // preflight here as well, before creating a replacement GPU object.
    if (format === 9 && (mipThreshold >> 6) !== 0) {
      throw new RangeError('only production quality-0 DXT5 compression is supported');
    }
    if (entry && entry.data === bitmap.data && entry.width === bitmap.width &&
      entry.height === bitmap.height && entry.mipmapped === mipmapped &&
      entry.mipLevels === mipLevels && entry.mipThreshold === mipThreshold &&
      entry.format === format && entry.storageKind === storageKind) {
      return entry.texture;
    }
    const gl = this.gl;
    // Every non-hit is staged on fresh storage. Besides preventing stale upper
    // mip levels after structural changes, this keeps the last good cached
    // texture intact if conversion or a WebGL upload throws halfway through.
    // Immutable production hits return above without allocating anything.
    const texture = gl.createTexture();
    if (!texture) throw new Error('WebGL texture allocation failed');
    let byteLength;
    try {
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      if (format === 9) {
        if (storageKind === 'dxt5') {
          forEachDxt5Mip(bitmap, 'bc3', (level, width, height, pixels) => {
            gl.compressedTexImage2D(gl.TEXTURE_2D, level,
              this.s3tc.COMPRESSED_RGBA_S3TC_DXT5_EXT,
              width, height, 0, pixels);
          });
          byteLength = dxt5MipChainByteLength(bitmap.width, bitmap.height, mipLevels);
        } else {
          forEachDxt5Mip(bitmap, 'rgba8', (level, width, height, pixels) => {
            gl.texImage2D(gl.TEXTURE_2D, level, gl.RGBA8, width, height, 0,
              gl.RGBA, gl.UNSIGNED_BYTE, pixels);
          });
          byteLength = rgba8MipChainByteLength(bitmap.width, bitmap.height, mipLevels);
        }
      } else if (mipmapped && format !== 8) {
        const iterator = format === 5 ? forEachSignedNormalMip : forEachBitmapMip;
        const internalFormat = format === 5 ? gl.RGBA8_SNORM : gl.RGBA8;
        const type = format === 5 ? gl.BYTE : gl.UNSIGNED_BYTE;
        iterator(bitmap, (level, width, height, pixels) => {
          gl.texImage2D(gl.TEXTURE_2D, level, internalFormat, width, height, 0,
            gl.RGBA, type, pixels);
        });
      } else {
        const pixels = bitmapRGBA8(bitmap);
        gl.texImage2D(gl.TEXTURE_2D, 0,
          format === 5 ? gl.RGBA8_SNORM : gl.RGBA8,
          bitmap.width, bitmap.height, 0, gl.RGBA,
          format === 5 ? gl.BYTE : gl.UNSIGNED_BYTE, pixels);
        if (mipmapped) gl.generateMipmap(gl.TEXTURE_2D);
      }
      if (gl.TEXTURE_MAX_LEVEL !== undefined) {
        this.setSamplerParameter(texture, gl.TEXTURE_MAX_LEVEL, Math.max(0, mipLevels - 1));
      }
      this.setSamplerParameter(texture, gl.TEXTURE_MIN_FILTER,
        mipmapped ? gl.LINEAR_MIPMAP_LINEAR : gl.LINEAR);
      this.setSamplerParameter(texture, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      this.setSamplerParameter(texture, gl.TEXTURE_WRAP_S, gl.REPEAT);
      this.setSamplerParameter(texture, gl.TEXTURE_WRAP_T, gl.REPEAT);
      if (byteLength === undefined) {
        byteLength = rgba8MipChainByteLength(bitmap.width, bitmap.height, mipLevels);
      }
    } catch (error) {
      this.samplerStates.delete(texture);
      gl.deleteTexture(texture);
      throw error;
    }
    this.textureHandles.add(texture);
    if (entry) {
      this.textureHandles.delete(entry.texture);
      this.samplerStates.delete(entry.texture);
      gl.deleteTexture(entry.texture);
    }
    if (entry) {
      this.estimatedBytes -= entry.byteLength || 0;
      if (entry.storageKind === 'dxt5') this.dxt5CompressedCount--;
      else if (entry.storageKind === 'dxt5-rgba8') this.dxt5FallbackCount--;
    } else this.entryCount++;
    entry = {
      texture, data: bitmap.data, width: bitmap.width, height: bitmap.height,
      byteLength, mipmapped, mipLevels, mipThreshold, format, storageKind,
    };
    if (storageKind === 'dxt5') this.dxt5CompressedCount++;
    else if (storageKind === 'dxt5-rgba8') this.dxt5FallbackCount++;
    this.estimatedBytes += byteLength;
    this.entries.set(bitmap, entry);
    return texture;
  }

  resourceStats() {
    return {
      textures: this.entryCount + (this.fallback ? 1 : 0),
      bitmapTextures: this.entryCount,
      dxt5CompressedTextures: this.dxt5CompressedCount,
      dxt5FallbackTextures: this.dxt5FallbackCount,
      estimatedBytes: Math.ceil(this.estimatedBytes + (this.fallback ? 4 : 0)),
    };
  }

  bind(bitmap, unit = 0, options = {}) {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + unit);
    const texture = this.get(bitmap);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    if (options.clamp !== undefined) {
      const wrap = options.clamp ? gl.CLAMP_TO_EDGE : gl.REPEAT;
      this.setSamplerParameter(texture, gl.TEXTURE_WRAP_S, wrap);
      this.setSamplerParameter(texture, gl.TEXTURE_WRAP_T, wrap);
    }
    if (options.filterMode !== undefined) {
      const mode = options.filterMode & 7;
      const entry = bitmap && typeof bitmap === 'object' ? this.entries.get(bitmap) : null;
      const mipmapped = Boolean(entry?.mipmapped);
      const material11 = options.filterProfile === 'material11';
      // Material11 and Material20 use different native D3D filter tables.
      // In particular, mode 2 is trilinear in M11 but point-mip in M20.
      const point = material11 ? mode === 0 || mode === 4 || mode >= 6
        : mode === 0 || mode >= 5;
      let minFilter;
      if (point) {
        minFilter = material11 && mode === 4 && mipmapped
          ? gl.NEAREST_MIPMAP_NEAREST : gl.NEAREST;
      } else if (!mipmapped || mode === 1) {
        minFilter = gl.LINEAR;
      } else if ((material11 && mode === 5) || (!material11 && mode === 2)) {
        minFilter = gl.LINEAR_MIPMAP_NEAREST;
      } else {
        minFilter = gl.LINEAR_MIPMAP_LINEAR;
      }
      this.setSamplerParameter(texture, gl.TEXTURE_MAG_FILTER,
        point ? gl.NEAREST : gl.LINEAR);
      this.setSamplerParameter(texture, gl.TEXTURE_MIN_FILTER, minFilter);
      if (this.anisotropy) {
        const anisotropic = material11 ? mode === 3 : mode === 4;
        this.setSamplerParameter(texture, this.anisotropy.TEXTURE_MAX_ANISOTROPY_EXT,
          anisotropic ? Math.min(4, this.maxAnisotropy) : 1, true);
      }
    } else if (options.nearest !== undefined) {
      // Kept for non-material callers; released material samplers use the
      // complete D3D filter table through filterMode above.
      this.setSamplerParameter(texture, gl.TEXTURE_MAG_FILTER,
        options.nearest ? gl.NEAREST : gl.LINEAR);
    }
  }

  dispose() {
    for (const texture of this.textureHandles) this.gl.deleteTexture(texture);
    this.textureHandles.clear();
    this.fallback = null;
    this.entryCount = 0;
    this.dxt5CompressedCount = 0;
    this.dxt5FallbackCount = 0;
    this.estimatedBytes = 0;
    this.entries = new WeakMap();
    this.samplerStates.clear();
  }
}

class RenderTarget {
  constructor(gl, width = 1, height = 1, options = {}) {
    this.gl = gl;
    this.framebuffer = gl.createFramebuffer();
    this.color = gl.createTexture();
    this.hasDepth = options.depth !== false;
    this.depth = this.hasDepth ? gl.createRenderbuffer() : null;
    this.width = this.height = 0;
    this.depthStencil = options.depthStencil !== false;
    this.resize(width, height);
  }

  resize(width, height) {
    width = Math.max(1, width | 0); height = Math.max(1, height | 0);
    if (width === this.width && height === this.height) return false;
    const gl = this.gl;
    this.width = width; this.height = height;
    gl.bindTexture(gl.TEXTURE_2D, this.color);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.color, 0);
    if (this.hasDepth) {
      gl.bindRenderbuffer(gl.RENDERBUFFER, this.depth);
      gl.renderbufferStorage(gl.RENDERBUFFER,
        this.depthStencil ? gl.DEPTH24_STENCIL8 : gl.DEPTH_COMPONENT24, width, height);
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER,
        this.depthStencil ? gl.DEPTH_STENCIL_ATTACHMENT : gl.DEPTH_ATTACHMENT,
        gl.RENDERBUFFER, this.depth);
    }
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    if (status !== gl.FRAMEBUFFER_COMPLETE) throw new Error(`WebGL framebuffer incomplete (0x${status.toString(16)})`);
    return true;
  }

  bind() {
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.framebuffer);
    this.gl.viewport(0, 0, this.width, this.height);
  }

  estimatedBytes() {
    return this.width * this.height * (4 + (this.hasDepth ? 4 : 0));
  }

  dispose() {
    const gl = this.gl;
    if (this.depth) gl.deleteRenderbuffer(this.depth);
    gl.deleteTexture(this.color);
    gl.deleteFramebuffer(this.framebuffer);
  }
}

function createWebGL2(canvas, attributes = {}) {
  const gl = canvas.getContext('webgl2', {
    alpha: false, antialias: true, depth: true, stencil: true,
    powerPreference: 'high-performance', ...attributes,
  });
  if (!gl) throw new Error('Debris requires WebGL 2.0.');
  return gl;
}

export {
  RenderTarget,
  TextureCache,
  bitmapRGBA8,
  forEachBitmapMip,
  forEachSignedNormalMip,
  cameraMatrices,
  colorARGB,
  colorRGB,
  compileShader,
  createWebGL2,
  legacyProjection,
  linkProgram,
  mat4Inverse,
  normalMatrix3,
  uniformLocations,
};
