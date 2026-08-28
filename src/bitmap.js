import { Random } from './core.js';

  // genbitmap.cpp stores one pixel in an sU64.  Keeping the four words
  // separate is both considerably faster in JS and documents the browser
  // port's RGBA (rather than little-endian BGRA) convention.
  const MAX = 0x7fff;
  const HALF = 0x4000;
  const TAU = Math.PI * 2;
  const f32 = Math.fround;
  const imul = Math.imul;
  const trunc = Math.trunc;
  const glowTableCache = new Map();
  const GLOW_TABLE_CACHE_LIMIT = 32;
  // Runtime dispatch alone can supply this unforgeable token. Public
  // Bitmap_* calls cannot opt into destructive behavior accidentally.
  const RUNTIME_IN_PLACE = Symbol('runtime bitmap in-place');
  // Public adopt() intentionally leaves two observable Bitmap wrappers on one
  // typed array. Keep both wrappers permanently on the conservative path;
  // abandoned aliases cost one later copy but can never corrupt a cache.
  const publiclyAliasedBitmaps = new WeakSet();
  let bitmapDefaultFormat = 1;
  let bitmapTextureSizeOffset = 0;
  let bitmapRendererHook = null;

  function setBitmapDefaultFormat(value) {
    bitmapDefaultFormat = value == null ? 1 : value | 0;
  }

  function setBitmapTextureSizeOffset(value) {
    bitmapTextureSizeOffset = value | 0;
  }

  function setBitmapRendererHook(hook) {
    bitmapRendererHook = hook || null;
  }

  const BI = Object.freeze({
    ADD: 0, SUB: 1, MUL: 2, DIFF: 3, ALPHA: 4,
    MULCOL: 5, ADDCOL: 6, SUBCOL: 7, GRAY: 8, INVERT: 9,
    SCALECOL: 10, MERGE: 11, BRIGHTNESS: 12, SUBR: 13,
    MULMERGE: 14, SHARPEN: 15, HARDLIGHT: 16, OVER: 17,
    ADDSMOOTH: 18, MIN: 19, MAX: 20, RANGE: 21,
  });

  function clamp15(value) {
    value = trunc(value);
    return value < 0 ? 0 : value > MAX ? MAX : value;
  }

  // sFtol is an x87 fistp while the player has round-to-nearest-even active.
  // Keep the integer-indefinite result used by x87 for non-finite/overflowing
  // inputs as well, even though authored bitmap parameters stay in range.
  function roundEven(value) {
    if (!Number.isFinite(value) || value < -0x80000000 || value >= 0x80000000) return -0x80000000;
    const floor = Math.floor(value);
    const fraction = value - floor;
    if (fraction < 0.5) return floor | 0;
    if (fraction > 0.5) return (floor + 1) | 0;
    return (floor & 1 ? floor + 1 : floor) | 0;
  }

  function powerOfTwo(value) {
    return 31 - Math.clz32(value >>> 0);
  }

  function fixedMul(a, b) {
    // sMulShift uses the high part of a signed 64-bit product. Bitmap ranges
    // keep the product below JS's exact-integer limit; floor matches SAR for
    // negative products without paying BigInt's per-pixel cost.
    return Math.floor((a | 0) * (b | 0) / 65536) | 0;
  }

  function packedColor(color, out = new Uint16Array(4)) {
    color >>>= 0;
    out[0] = (((color >>> 16) & 255) * 257) >>> 1;
    out[1] = (((color >>> 8) & 255) * 257) >>> 1;
    out[2] = ((color & 255) * 257) >>> 1;
    out[3] = ((color >>> 24) * 257) >>> 1;
    return out;
  }

  function rgbaToPacked(r, g, b, a = 255) {
    return (((a & 255) << 24) | ((r & 255) << 16) |
      ((g & 255) << 8) | (b & 255)) >>> 0;
  }

  function signed16(value) { value &= 0xffff; return value & 0x8000 ? value - 0x10000 : value; }
  function saturateSigned16(value) { return value < -0x8000 ? -0x8000 : value > 0x7fff ? 0x7fff : value | 0; }
  function mulHighSigned16(a, b) { return (signed16(a) * signed16(b)) >> 16; }

  function fadeChannel(a, b, fade) {
    // Literal scalar form of Fade64: form two signed 16-bit weights from the
    // signed 32-bit fade, use pmulhw, add as words, then double as words. The
    // seemingly surprising one-LSB loss at both endpoints is intentional.
    const f = (fade | 0) >> 1;
    const weightB = saturateSigned16(f);
    const weightA = saturateSigned16(0x8000 - f);
    const sum = (mulHighSigned16(a, weightA) + mulHighSigned16(b, weightB)) & 0xffff;
    return (sum << 1) & 0xffff;
  }

  function fadePixel(out, oi, a, ai, b, bi, fade) {
    for (let c = 0; c < 4; c++) out[oi + c] = fadeChannel(a[ai + c], b[bi + c], fade);
  }

  function addScalePixel(out, oi, base, bi, scale, si, fade) {
    const weight = saturateSigned16((fade | 0) >> 1);
    for (let c = 0; c < 4; c++) {
      const value = signed16(base[bi + c]) + (mulHighSigned16(scale[si + c], weight) << 1);
      out[oi + c] = saturateSigned16(value) & 0xffff;
    }
  }

  function grayMMX(red, green, blue) {
    // BI_ALPHA/BI_GRAY do not evaluate (r + 2*g + b) / 4 as one
    // expression. The released MMX kernel first halves the two adjacent word
    // sums, adds those (with word wrap), and halves once more.
    const rg = ((red + green) & 0xffff) >>> 1;
    const gb = ((green + blue) & 0xffff) >>> 1;
    return ((rg + gb) & 0xffff) >>> 1;
  }

  function hashWords(words) {
    let hash = 0x811c9dc5;
    for (let i = 0; i < words.length; i++) {
      const v = words[i];
      hash = imul(hash ^ (v & 255), 0x01000193) >>> 0;
      hash = imul(hash ^ (v >>> 8), 0x01000193) >>> 0;
    }
    return hash >>> 0;
  }

  function hashBytes(bytes) {
    let hash = 0x811c9dc5;
    for (let i = 0; i < bytes.length; i++) hash = imul(hash ^ bytes[i], 0x01000193) >>> 0;
    return hash >>> 0;
  }

  function copyBitmapSettings(target, source) {
    target.format = target.Format = source.format;
    target.texMipCount = target.TexMipCount = source.texMipCount;
    target.texMipThreshold = target.TexMipTresh = source.texMipThreshold;
    target.stripped = target.Stripped = source.stripped;
    // Derived GPU objects never belong to a CPU-side bitmap copy.
    target.texture = target.Texture = null;
    if (source.deferredRender) target.deferredRender = { ...source.deferredRender };
    else delete target.deferredRender;
    return target;
  }

  class Bitmap {
    constructor(width, height, data = null) {
      if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1) {
        throw new RangeError('Bitmap dimensions must be positive integers');
      }
      this.width = this.XSize = width;
      this.height = this.YSize = height;
      this.size = this.Size = width * height;
      this.format = this.Format = bitmapDefaultFormat;
      this.texture = this.Texture = null;
      this.texMipCount = this.TexMipCount = 0;
      this.texMipThreshold = this.TexMipTresh = 0;
      this.stripped = this.Stripped = false;
      this.data = this.Data = data ? new Uint16Array(data) : new Uint16Array(this.size * 4);
      if (this.data.length !== this.size * 4) throw new RangeError('Bitmap data length mismatch');
    }

    copy() {
      return copyBitmapSettings(new Bitmap(this.width, this.height, this.data), this);
    }

    clone() { return this.copy(); }

    // Public callers can explicitly alias storage. Runtime graph ownership
    // uses the private transferBitmapStorage helper below, so an internal
    // one-owner move does not become indistinguishable from a public alias.
    adopt(source) {
      if (!(source instanceof Bitmap) || source === this) return this;
      publiclyAliasedBitmaps.add(this);
      publiclyAliasedBitmaps.add(source);
      return transferBitmapStorage(this, source);
    }

    releaseStorage() {
      this.data = this.Data = new Uint16Array();
      this.texture = this.Texture = null;
      delete this.deferredRender;
      this.released = true;
    }

    summary() {
      return {
        kind: 'Bitmap', width: this.width, height: this.height,
        format: this.format, mipCount: this.texMipCount,
        mipThreshold: this.texMipThreshold, hash: hashWords(this.data),
      };
    }
  }

  function blankBitmapCopy(source) {
    const expectedLength = source.width * source.height * 4;
    // Preserve the released/malformed-storage guard while avoiding the
    // source-to-destination pixel copy for complete-overwrite operators.
    if (!source.data || source.data.length !== expectedLength) {
      throw new RangeError('Bitmap data length mismatch');
    }
    return copyBitmapSettings(new Bitmap(source.width, source.height), source);
  }

  function transferBitmapStorage(target, source) {
    if (!(target instanceof Bitmap) || !(source instanceof Bitmap) || source === target) return target;
    target.width = target.XSize = source.width;
    target.height = target.YSize = source.height;
    target.size = target.Size = source.size;
    target.format = target.Format = source.format;
    target.texture = target.Texture = source.texture;
    target.texMipCount = target.TexMipCount = source.texMipCount;
    target.texMipThreshold = target.TexMipTresh = source.texMipThreshold;
    target.stripped = target.Stripped = source.stripped;
    target.data = target.Data = source.data;
    if (source.deferredRender) target.deferredRender = { ...source.deferredRender };
    else delete target.deferredRender;
    target.released = false;
    return target;
  }

  function newBitmap(xExponent, yExponent) {
    const dontScale = (xExponent & 0x80) !== 0;
    let xs = xExponent & 0x7f;
    let ys = yExponent | 0;
    if (!dontScale) {
      const offset = bitmapTextureSizeOffset;
      xs += offset;
      ys += offset;
    }
    xs = Math.max(0, Math.min(12, xs));
    ys = Math.max(0, Math.min(12, ys));
    return new Bitmap(1 << xs, 1 << ys);
  }

  function requireBitmap(value, name = 'bitmap') {
    if (!(value instanceof Bitmap)) throw new TypeError(`${name} is not a Bitmap`);
    return value;
  }

  class BilinearContext {
    constructor(bitmap, border = 0, width = bitmap?.width, height = bitmap?.height) {
      bitmap = requireBitmap(bitmap);
      if (!Number.isInteger(width) || !Number.isInteger(height) ||
          width < 1 || height < 1 || width * height !== bitmap.size) {
        throw new RangeError('BilinearContext dimensions must cover the bitmap data');
      }
      this.bitmap = bitmap;
      this.data = bitmap.data;
      this.width = width;
      this.height = height;
      this.xMask = width - 1;
      this.yMask = height - 1;
      this.xMax = (width * 0x10000 - 1) | 0;
      this.yMax = (height * 0x10000 - 1) | 0;
      // BilinearSetup names these bits "border", but their released behavior
      // is specifically clamp: with a bit clear it masks by (size<<16)-1 and
      // therefore wraps; with the bit set it leaves the coordinate intact and
      // clamps values outside the source extent.
      this.clampX = (border & 1) !== 0;
      this.clampY = (border & 2) !== 0;
    }

    coordinate(value, maximum, clamp) {
      value |= 0;
      if (!clamp) return value & maximum;
      return value < 0 ? 0 : value > maximum ? maximum : value;
    }

    point(u, v, out = new Uint16Array(4), offset = 0) {
      // PointFilter intentionally ignores BilinearSetup's border masks.
      const x = (u >> 16) & this.xMask;
      const y = (v >> 16) & this.yMask;
      const source = (y * this.width + x) * 4;
      out[offset] = this.data[source];
      out[offset + 1] = this.data[source + 1];
      out[offset + 2] = this.data[source + 2];
      out[offset + 3] = this.data[source + 3];
      return out;
    }

    sample(u, v, out = new Uint16Array(4), offset = 0) {
      u = this.coordinate(u, this.xMax, this.clampX);
      v = this.coordinate(v, this.yMax, this.clampY);
      const x0 = u >>> 16;
      const y0 = v >>> 16;
      const x1 = (x0 + 1) & this.xMask;
      const y1 = (y0 + 1) & this.yMask;
      // The released MMX kernel drops the low fraction bit before pmulhw.
      // Its word-delta interpolation is deliberately not the conventional
      // positive weighted sum: negative slopes round down and each axis can
      // therefore differ from real-valued bilinear interpolation by a word.
      const fu = (u & 0xffff) >>> 1;
      const fv = (v & 0xffff) >>> 1;
      const p00 = (y0 * this.width + x0) * 4;
      const p10 = (y0 * this.width + x1) * 4;
      const p01 = (y1 * this.width + x0) * 4;
      const p11 = (y1 * this.width + x1) * 4;
      for (let c = 0; c < 4; c++) {
        const top = this.data[p00 + c];
        const bottom = this.data[p01 + c];
        const topDelta = ((this.data[p10 + c] - top) << 16) >> 16;
        const bottomDelta = ((this.data[p11 + c] - bottom) << 16) >> 16;
        const h0 = (top + (((topDelta * fu) >> 16) << 1)) & 0xffff;
        const h1 = (bottom + (((bottomDelta * fu) >> 16) << 1)) & 0xffff;
        // paddw 0x8000 followed by psubusw 0x8000 clamps the rare negative
        // interpolation underflow without changing values in the 15-bit
        // bitmap range.
        const verticalDelta = ((h1 - h0) << 16) >> 16;
        const biased = (h0 + 0x8000 + (((verticalDelta * fv) >> 16) << 1)) & 0xffff;
        out[offset + c] = biased < 0x8000 ? 0 : biased - 0x8000;
      }
      return out;
    }
  }

  function bilinearFilter(bitmapOrContext, u, v, out, offset = 0, border = 0) {
    const context = bitmapOrContext instanceof BilinearContext
      ? bitmapOrContext : new BilinearContext(bitmapOrContext, border);
    return context.sample(u | 0, v | 0, out, offset);
  }

  function pointFilter(bitmapOrContext, u, v, out, offset = 0, border = 0) {
    const context = bitmapOrContext instanceof BilinearContext
      ? bitmapOrContext : new BilinearContext(bitmapOrContext, border);
    return context.point(u | 0, v | 0, out, offset);
  }

  function sampleUV(bitmap, u, v, border = 0, linear = true, out = new Uint16Array(4)) {
    const context = new BilinearContext(bitmap, border);
    const x = trunc(u * bitmap.width * 0x10000) | 0;
    const y = trunc(v * bitmap.height * 0x10000) | 0;
    return linear ? context.sample(x, y, out) : context.point(x, y, out);
  }

  function innerPixel(mode, destination, source, original, constant0, constant1, result = null) {
    const out = result || [0, 0, 0, 0];
    const s = source;
    const x = original;
    const d = destination;
    for (let c = 0; c < 4; c++) {
      switch (mode) {
        case BI.ADD: out[c] = saturateSigned16(signed16(x[c]) + signed16(s[c])) & 0xffff; break;
        case BI.SUB: out[c] = Math.max(0, x[c] - s[c]); break;
        case BI.MUL: out[c] = (mulHighSigned16(x[c], s[c]) << 1) & 0xffff; break;
        case BI.DIFF: out[c] = ((((x[c] - s[c]) & 0xffff) + MAX) & 0xffff) >>> 1; break;
        case BI.MULCOL: out[c] = (mulHighSigned16(x[c], constant0[c]) << 1) & 0xffff; break;
        case BI.ADDCOL: out[c] = saturateSigned16(signed16(x[c]) + signed16(constant0[c])) & 0xffff; break;
        case BI.SUBCOL: out[c] = Math.max(0, x[c] - constant0[c]); break;
        case BI.INVERT: out[c] = x[c] ^ MAX; break;
        case BI.SCALECOL: {
          // The source reconstructs the signed 16x16 product, then uses
          // psrld (logical), not psrad, before packssdw. A negative product
          // therefore becomes a large positive dword and saturates to 0x7fff.
          const product = imul(signed16(x[c]), signed16(constant0[c])) >>> 0;
          out[c] = Math.min(MAX, product >>> 11);
          break;
        }
        case BI.MERGE: {
          const inverse = saturateSigned16(MAX - signed16(d[c]));
          const sum = saturateSigned16(mulHighSigned16(d[c], s[c]) + mulHighSigned16(inverse, x[c]));
          out[c] = (sum << 1) & 0xffff;
          break;
        }
        case BI.SUBR: out[c] = Math.max(0, s[c] - x[c]); break;
        case BI.MULMERGE: {
          const multiplied = (mulHighSigned16(s[c], x[c]) << 1) & 0xffff;
          const inverse = saturateSigned16(MAX - signed16(d[c]));
          const sum = saturateSigned16(mulHighSigned16(d[c], multiplied) + mulHighSigned16(inverse, x[c]));
          out[c] = (sum << 1) & 0xffff;
          break;
        }
        case BI.SHARPEN: {
          const difference = Math.max(0, x[c] - d[c]);
          const scaled = (signed16(difference) * signed16(constant0[c])) >> 11;
          out[c] = Math.max(0, saturateSigned16(signed16(x[c]) + scaled)) & 0xffff;
          break;
        }
        case BI.HARDLIGHT: {
          const doubled = (s[c] << 1) & 0xffff;
          const signMask = signed16(doubled) < 0 ? 0xffff : 0;
          const magnitude = doubled & MAX;
          let value = (mulHighSigned16(x[c], magnitude) << 1) & 0xffff;
          value ^= signMask;
          value = (value + ((magnitude + x[c]) & signMask)) & 0xffff;
          out[c] = value;
          break;
        }
        case BI.OVER: {
          const delta = saturateSigned16(signed16(s[c]) - signed16(x[c]));
          const mixed = saturateSigned16(signed16(x[c]) + (mulHighSigned16(delta, s[3]) << 1));
          out[c] = Math.max(0, mixed) & 0xffff;
          break;
        }
        case BI.ADDSMOOTH: out[c] = ((mulHighSigned16(x[c] ^ MAX, s[c] ^ MAX) << 1) ^ MAX) & 0xffff; break;
        case BI.MIN: out[c] = signed16(x[c]) < signed16(s[c]) ? x[c] : s[c]; break;
        case BI.MAX: out[c] = signed16(x[c]) > signed16(s[c]) ? x[c] : s[c]; break;
        case BI.RANGE: {
          const difference = saturateSigned16(signed16(constant1[c]) - signed16(constant0[c]));
          const value = saturateSigned16(signed16(constant0[c]) + (mulHighSigned16(x[c], difference) << 1));
          out[c] = Math.max(0, value) & 0xffff;
          break;
        }
        default: out[c] = x[c]; break;
      }
    }
    if (mode === BI.ALPHA) {
      // pand mm4 clears the high bit of the three retained color words.
      out[0] = x[0] & MAX; out[1] = x[1] & MAX; out[2] = x[2] & MAX;
      out[3] = grayMMX(s[0], s[1], s[2]);
    } else if (mode === BI.GRAY) {
      const gray = grayMMX(x[0], x[1], x[2]);
      out[0] = out[1] = out[2] = gray; out[3] = MAX;
    } else if (mode === BI.BRIGHTNESS) {
      for (let c = 0; c < 4; c++) {
        const mask = signed16(s[c]) > 0x3fff ? 0x7fff : 0;
        const product = (mulHighSigned16(s[c] ^ mask, x[c] ^ mask) << 2) & 0xffff;
        out[c] = product ^ mask;
      }
    }
    return out;
  }

  function bitmapInner(destination, source, mode, original = destination, constants = null) {
    const count = destination.length >>> 2;
    const constant0 = constants?.[0] || (source.length === 4 ? source : [0, 0, 0, 0]);
    const constant1 = constants?.[1] || constant0;
    const sourceConstant = source.length === 4;
    const originalConstant = original.length === 4;

    // These scalar kernels cover nearly every debris Color/Merge call. Keeping
    // the mode branch outside the pixel loop avoids millions of temporary
    // record loads and per-channel switches while retaining the literal MMX
    // signed-word formulas used by innerPixel.
    if (mode === BI.ADD || mode === BI.SUB || mode === BI.MUL || mode === BI.DIFF) {
      for (let pixel = 0; pixel < count; pixel++) {
        const di = pixel * 4;
        const si = sourceConstant ? 0 : di;
        const oi = originalConstant ? 0 : di;
        for (let c = 0; c < 4; c++) {
          const s = source[si + c], x = original[oi + c];
          if (mode === BI.ADD) destination[di + c] = saturateSigned16(signed16(x) + signed16(s)) & 0xffff;
          else if (mode === BI.SUB) destination[di + c] = Math.max(0, x - s);
          else if (mode === BI.MUL) destination[di + c] = (mulHighSigned16(x, s) << 1) & 0xffff;
          else destination[di + c] = ((((x - s) & 0xffff) + MAX) & 0xffff) >>> 1;
        }
      }
      return destination;
    }

    if (mode === BI.MULCOL || mode === BI.ADDCOL || mode === BI.SUBCOL ||
        mode === BI.INVERT || mode === BI.SCALECOL) {
      for (let pixel = 0; pixel < count; pixel++) {
        const di = pixel * 4;
        const oi = originalConstant ? 0 : di;
        for (let c = 0; c < 4; c++) {
          const x = original[oi + c], color = constant0[c];
          if (mode === BI.MULCOL) destination[di + c] = (mulHighSigned16(x, color) << 1) & 0xffff;
          else if (mode === BI.ADDCOL) destination[di + c] = saturateSigned16(signed16(x) + signed16(color)) & 0xffff;
          else if (mode === BI.SUBCOL) destination[di + c] = Math.max(0, x - color);
          else if (mode === BI.INVERT) destination[di + c] = x ^ MAX;
          else {
            const product = imul(signed16(x), signed16(color)) >>> 0;
            destination[di + c] = Math.min(MAX, product >>> 11);
          }
        }
      }
      return destination;
    }

    if (mode === BI.ALPHA || mode === BI.GRAY) {
      for (let pixel = 0; pixel < count; pixel++) {
        const di = pixel * 4;
        const si = sourceConstant ? 0 : di;
        const oi = originalConstant ? 0 : di;
        if (mode === BI.ALPHA) {
          destination[di] = original[oi] & MAX;
          destination[di + 1] = original[oi + 1] & MAX;
          destination[di + 2] = original[oi + 2] & MAX;
          destination[di + 3] = grayMMX(source[si], source[si + 1], source[si + 2]);
        } else {
          const gray = grayMMX(original[oi], original[oi + 1], original[oi + 2]);
          destination[di] = destination[di + 1] = destination[di + 2] = gray;
          destination[di + 3] = MAX;
        }
      }
      return destination;
    }

    if (mode === BI.MERGE) {
      for (let pixel = 0; pixel < count; pixel++) {
        const di = pixel * 4;
        const si = sourceConstant ? 0 : di;
        const oi = originalConstant ? 0 : di;
        for (let c = 0; c < 4; c++) {
          // Read all three lanes before writing so the native pixel-local
          // aliasing used by Mask remains valid.
          const d = destination[di + c];
          const s = source[si + c];
          const x = original[oi + c];
          const inverse = saturateSigned16(MAX - signed16(d));
          const sum = saturateSigned16(
            mulHighSigned16(d, s) + mulHighSigned16(inverse, x),
          );
          destination[di + c] = (sum << 1) & 0xffff;
        }
      }
      return destination;
    }

    if (mode === BI.RANGE) {
      const differences = new Int32Array(4);
      for (let c = 0; c < 4; c++) {
        differences[c] = saturateSigned16(
          signed16(constant1[c]) - signed16(constant0[c]),
        );
      }
      for (let pixel = 0; pixel < count; pixel++) {
        const di = pixel * 4;
        const oi = originalConstant ? 0 : di;
        for (let c = 0; c < 4; c++) {
          const value = saturateSigned16(
            signed16(constant0[c]) +
            (mulHighSigned16(original[oi + c], differences[c]) << 1),
          );
          destination[di + c] = Math.max(0, value) & 0xffff;
        }
      }
      return destination;
    }

    const d = [0, 0, 0, 0], s = [0, 0, 0, 0], x = [0, 0, 0, 0], out = [0, 0, 0, 0];
    for (let i = 0; i < count; i++) {
      const di = i * 4;
      const si = source.length === 4 ? 0 : di;
      const oi = original.length === 4 ? 0 : di;
      d[0] = destination[di]; d[1] = destination[di + 1]; d[2] = destination[di + 2]; d[3] = destination[di + 3];
      s[0] = source[si]; s[1] = source[si + 1]; s[2] = source[si + 2]; s[3] = source[si + 3];
      x[0] = original[oi]; x[1] = original[oi + 1]; x[2] = original[oi + 2]; x[3] = original[oi + 3];
      innerPixel(mode, d, s, x, constant0, constant1, out);
      destination[di] = out[0]; destination[di + 1] = out[1]; destination[di + 2] = out[2]; destination[di + 3] = out[3];
    }
    return destination;
  }

  function Bitmap_Flat(xs, ys, color) {
    const bitmap = newBitmap(xs | 0, ys | 0);
    const pixel = packedColor(color);
    // One native fill replaces four scalar stores per pixel. Constructing the
    // word through a Uint16 view keeps the channel order correct on the host
    // instead of assuming little-endian byte layout.
    if (typeof BigUint64Array === 'function' && !(bitmap.data.byteOffset & 7)) {
      const pixelWord = new BigUint64Array(pixel.buffer, pixel.byteOffset, 1)[0];
      new BigUint64Array(
        bitmap.data.buffer, bitmap.data.byteOffset, bitmap.size,
      ).fill(pixelWord);
    } else {
      for (let i = 0; i < bitmap.data.length; i += 4) {
        bitmap.data[i] = pixel[0]; bitmap.data[i + 1] = pixel[1];
        bitmap.data[i + 2] = pixel[2]; bitmap.data[i + 3] = pixel[3];
      }
    }
    return bitmap;
  }

  function Bitmap_Format(input, format, count, threshold) {
    const bitmap = requireBitmap(input).copy();
    bitmap.format = bitmap.Format = format | 0;
    bitmap.texMipCount = bitmap.TexMipCount = count | 0;
    bitmap.texMipThreshold = bitmap.TexMipTresh = threshold | 0;
    return bitmap;
  }

  function Bitmap_Merge(mode, bitmaps, writable = null) {
    if (!bitmaps.length) return null;
    const first = requireBitmap(bitmaps[0]);
    // Validate the entire native vararg list before touching a transferred
    // first input. A late size mismatch must not leave its final-consumed
    // producer partially accumulated when Merge returns null.
    for (let index = 1; index < bitmaps.length && bitmaps[index]; index++) {
      const other = requireBitmap(bitmaps[index]);
      if (first.width !== other.width || first.height !== other.height) return null;
    }
    const bitmap = writable === RUNTIME_IN_PLACE ? first : first.copy();
    const modes = [BI.ADD, BI.SUB, BI.MUL, BI.DIFF, BI.ALPHA, BI.BRIGHTNESS,
      BI.HARDLIGHT, BI.OVER, BI.ADDSMOOTH, BI.MIN, BI.MAX];
    let original = first.data;
    for (let i = 1; i < bitmaps.length; i++) {
      // The vararg source treats the first null as an end marker. RotateMul
      // relies on this when a degenerate Rotate (zero scale) yields null.
      if (!bitmaps[i]) break;
      const other = requireBitmap(bitmaps[i]);
      bitmapInner(bitmap.data, other.data, modes[mode] ?? BI.ADD, original);
      // Bitmap_Inner reads and writes one complete pixel before advancing, so
      // native aliases the accumulated result from the second input onward.
      original = bitmap.data;
    }
    return bitmap;
  }

  function Bitmap_Color(input, mode, color, writable = null) {
    const source = requireBitmap(input);
    const bitmap = writable === RUNTIME_IN_PLACE ? source : blankBitmapCopy(source);
    const original = input.data;
    bitmapInner(bitmap.data, packedColor(color), (mode | 0) + BI.MULCOL, original);
    return bitmap;
  }

  function Bitmap_Range(input, mode, color0, color1, writable = null) {
    const sourceBitmap = requireBitmap(input);
    const bitmap = writable === RUNTIME_IN_PLACE ? sourceBitmap : blankBitmapCopy(sourceBitmap);
    let source = input.data;
    if (mode & 1) {
      bitmapInner(bitmap.data, bitmap.data, BI.GRAY, source);
      source = bitmap.data;
    }
    if (mode & 2) {
      bitmapInner(bitmap.data, bitmap.data, BI.INVERT, source);
      source = bitmap.data;
    }
    bitmapInner(bitmap.data, packedColor(color0), BI.RANGE, source,
      [packedColor(color0), packedColor(color1)]);
    return bitmap;
  }

  function glowTables(power, alpha15, oldFalloff) {
    const key = `${oldFalloff ? 1 : 0}:${power}:${alpha15}`;
    let tables = glowTableCache.get(key);
    if (tables) {
      // Refresh insertion order so a small cache covers the repeated authored
      // curves without retaining every parameter combination forever.
      glowTableCache.delete(key);
      glowTableCache.set(key, tables);
      return tables;
    }
    const gamma = new Int32Array(1025);
    const low = new Int32Array(32);
    const newFalloffExponent = f32(power * 2);
    for (let index = 0; index <= 1024; index++) {
      const coordinate = f32(index / 1024);
      const strength = oldFalloff
        ? f32(Math.pow(f32(1 - coordinate), power) * alpha15)
        : f32(f32(1 - Math.pow(coordinate, newFalloffExponent)) * alpha15);
      gamma[index] = clamp15(strength) * 2;
    }
    // Values close to zero have slopes too steep for the coarser table.
    // genbitmap.cpp therefore retains the first 32 samples at 1/32768.
    for (let index = 0; index < 32; index++) {
      const coordinate = f32(index / 32768);
      const strength = oldFalloff
        ? f32(Math.pow(f32(1 - coordinate), power) * alpha15)
        : f32(f32(1 - Math.pow(coordinate, newFalloffExponent)) * alpha15);
      low[index] = clamp15(strength) * 2;
    }
    tables = { gamma, low };
    if (glowTableCache.size >= GLOW_TABLE_CACHE_LIMIT) {
      glowTableCache.delete(glowTableCache.keys().next().value);
    }
    glowTableCache.set(key, tables);
    return tables;
  }

  function Bitmap_GlowRect(input, cx, cy, rx, ry, sx, sy, color, alpha, power, wrap, bug,
    writable = null) {
    const source = requireBitmap(input);
    const bitmap = writable === RUNTIME_IN_PLACE ? source : source.copy();
    cx = f32(cx); cy = f32(cy); rx = f32(rx); ry = f32(ry);
    sx = f32(sx); sy = f32(sy); alpha = f32(alpha); power = f32(power);
    const circular = (bug & 2) === 0;
    let curvePower = power === 0 ? f32(1 / 65536) : power;
    curvePower = f32(0.25 / curvePower);
    const col = packedColor(color);
    const alpha15 = f32(alpha * 32768);
    const { gamma: gammaTable, low: lowTable } = glowTables(
      curvePower, alpha15, (bug & 1) !== 0,
    );
    const coverageLimit = 1 - 1 / 32768;
    function paint(centerX, centerY, wrapMode) {
      if (wrapMode === 1) {
        if (f32(f32(centerX + rx) + sx) > 1) paint(f32(centerX - 1), centerY, 2);
        if (f32(f32(centerX - rx) - sx) < 0) paint(f32(centerX + 1), centerY, 2);
      }
      if (wrapMode === 1 || wrapMode === 2) {
        if (f32(f32(centerY + ry) + sy) > 1) paint(centerX, f32(centerY - 1), 0);
        if (f32(f32(centerY - ry) - sy) < 0) paint(centerX, f32(centerY + 1), 0);
      }
      const pcx = f32(centerX * bitmap.width), pcy = f32(centerY * bitmap.height);
      const psx = f32(sx * bitmap.width), psy = f32(sy * bitmap.height);
      const threshold = f32(1 / 65536);
      const prx0 = Math.max(f32(rx * bitmap.width), threshold);
      const pry0 = Math.max(f32(ry * bitmap.height), threshold);
      const prx = circular ? f32(1 / f32(prx0 * prx0)) : f32(1 / prx0);
      const pry = circular ? f32(1 / f32(pry0 * pry0)) : f32(1 / pry0);
      // The released generator evaluates pow only while constructing these
      // two tables. The former port evaluated it for every covered pixel,
      // which was both slower and numerically different from GetGamma's
      // authored 5-bit interpolation (GlowRect is used 870 times by Debris).
      // A one-pixel conservative pad lets us skip rows/columns that provably
      // cannot pass the unchanged native coverage test. This is especially
      // valuable for the many small highlights in the production graph.
      const radiusX = prx0 * (circular ? Math.sqrt(coverageLimit) : coverageLimit);
      const radiusY = pry0 * (circular ? Math.sqrt(coverageLimit) : coverageLimit);
      const xStart = Math.max(0, Math.floor(pcx - psx - radiusX) - 1);
      const xEnd = Math.min(bitmap.width, Math.ceil(pcx + psx + radiusX) + 2);
      const yStart = Math.max(0, Math.floor(pcy - psy - radiusY) - 1);
      const yEnd = Math.min(bitmap.height, Math.ceil(pcy + psy + radiusY) + 2);
      for (let y = yStart; y < yEnd; y++) {
        let fy = f32(Math.abs(y - pcy) - psy);
        if (fy < 0) fy = 0;
        fy = f32(fy * (circular ? f32(fy * pry) : pry));
        for (let x = xStart; x < xEnd; x++) {
          let fx = f32(Math.abs(x - pcx) - psx);
          if (fx < 0) fx = 0;
          const a = circular
            ? f32(f32(f32(fx * fx) * prx) + fy)
            : Math.max(f32(fx * prx), fy);
          if (a < coverageLimit) {
            const i = (y * bitmap.width + x) * 4;
            const gammaIndex = roundEven(f32(a * 32768));
            const fade = gammaIndex < 32
              ? lowTable[gammaIndex]
              : gammaLookup(gammaTable, gammaIndex);
            bitmap.data[i] = fadeChannel(bitmap.data[i], col[0], fade);
            bitmap.data[i + 1] = fadeChannel(bitmap.data[i + 1], col[1], fade);
            bitmap.data[i + 2] = fadeChannel(bitmap.data[i + 2], col[2], fade);
            bitmap.data[i + 3] = fadeChannel(bitmap.data[i + 3], col[3], fade);
          }
        }
      }
    }
    paint(cx, cy, wrap | 0);
    return bitmap;
  }

  function gammaLookup(table, value, shift = 5) {
    const index = value >>> shift;
    const fraction = value & ((1 << shift) - 1);
    const a = table[Math.min(index, table.length - 1)];
    const b = table[Math.min(index + 1, table.length - 1)];
    // GetGamma uses an arithmetic right shift. This matters for GlowRect's
    // descending table: division/truncation would round a negative delta in
    // the opposite direction by one unit.
    return a + (((b - a) * fraction) >> shift);
  }

  function Bitmap_HSCB(input, hue, saturation, contrast, brightness, writable = null) {
    const sourceBitmap = requireBitmap(input);
    const bitmap = writable === RUNTIME_IN_PLACE ? sourceBitmap : blankBitmapCopy(sourceBitmap);
    const table = new Int32Array(1026);
    hue = f32(hue); saturation = f32(saturation);
    contrast = f32(contrast); brightness = f32(brightness);
    const exponent = f32(contrast * contrast);
    for (let i = 0; i <= 1025; i++) {
      // The literal 0.01 is nominally double, but the released player keeps
      // x87's precision control at single: the base, pow result, and both
      // products are rounded to sF32 before the ordinary truncating store.
      const coordinate = f32(f32(i * 32 + 0.01) / 32768);
      const curved = f32(Math.pow(coordinate, exponent));
      table[i] = trunc(f32(f32(curved * 32768) * brightness));
    }
    let hueFixed = trunc(f32(f32(hue * 6) * 65536)) % (6 * 65536);
    const satFixed = trunc(f32(saturation * 65536));
    const adjust = hueFixed !== 0 || satFixed !== 65536;
    const source = sourceBitmap.data;
    for (let i = 0; i < bitmap.data.length; i += 4) {
      let r = gammaLookup(table, source[i]);
      let g = gammaLookup(table, source[i + 1]);
      let b = gammaLookup(table, source[i + 2]);
      if (adjust) {
        const maximum = Math.max(r, g, b), minimum = Math.min(r, g, b);
        let range = maximum - minimum;
        let h = 0;
        if (range) {
          if (r === maximum) h = 65536 + trunc((g - b) * 65536 / range);
          else if (g === maximum) h = 3 * 65536 + trunc((b - r) * 65536 / range);
          else h = 5 * 65536 + trunc((r - g) * 65536 / range);
        }
        h += hueFixed;
        if (h > 6 * 65536) h -= 6 * 65536;
        range = fixedMul(range, satFixed);
        const min2 = maximum - range;
        const rh = h & 131071;
        const m1 = min2 + (rh >= 65536 ? fixedMul(rh - 65536, range) : 0);
        const m2 = min2 + (rh < 65536 ? fixedMul(65536 - rh, range) : 0);
        if (h < 2 * 65536) { r = maximum; g = m1; b = m2; }
        else if (h < 4 * 65536) { r = m2; g = maximum; b = m1; }
        else { r = m1; g = m2; b = maximum; }
      }
      bitmap.data[i] = clamp15(r);
      bitmap.data[i + 1] = clamp15(g);
      bitmap.data[i + 2] = clamp15(b);
      bitmap.data[i + 3] = source[i + 3];
    }
    return bitmap;
  }

  function blurScaleAccumulator(accumulator, amplitudeLow, amplitudeHigh, amplitudeClip) {
    const shifted = accumulator >> 6;
    const low = shifted & 0xffff, high = shifted >>> 16;
    let output = accumulator > amplitudeClip ? 0xffff : 0;

    // BlurCore reconstructs the high half of a 32x32 multiply from three
    // 16-bit products. Each paddusw saturates independently; combining these
    // terms before clamping changes sharpen and signed-word results.
    output = Math.min(0xffff, output + (imul(high, amplitudeLow) & 0xffff));
    output = Math.min(0xffff, output + (imul(low, amplitudeHigh) & 0xffff));
    output = Math.min(0xffff, output + (imul(low, amplitudeLow) >>> 16));

    // paddw 0x8000; psubsw 0x8000 is the released unsigned-to-0x7fff clamp.
    return Math.min(MAX, output);
  }

  function blurCore(source, sourceOffset, destination, destinationOffset,
    kernelSize, resolution, f0, f1, amplitude, amplitudeClip) {
    const coordinateMask = resolution - 1;
    let entering = (-((kernelSize + 1) >> 1)) & coordinateMask;
    let leaving = entering;
    let sourceIndex = sourceOffset + leaving * 4;
    const edgeDifference = f1 - f0;

    // pmaddwd treats all four bitmap words as signed, including 0xffff alpha
    // words produced by Normals. paddd/psubd then wrap at signed 32 bits.
    let accumulator0 = imul((source[sourceIndex] << 16) >> 16, edgeDifference) | 0;
    let accumulator1 = imul((source[sourceIndex + 1] << 16) >> 16, edgeDifference) | 0;
    let accumulator2 = imul((source[sourceIndex + 2] << 16) >> 16, edgeDifference) | 0;
    let accumulator3 = imul((source[sourceIndex + 3] << 16) >> 16, edgeDifference) | 0;

    for (let index = 0; index < kernelSize; index++) {
      const next = (entering + 1) & coordinateMask;
      const first = sourceOffset + entering * 4, second = sourceOffset + next * 4;
      accumulator0 = (accumulator0 + imul((source[first] << 16) >> 16, f0) +
        imul((source[second] << 16) >> 16, f1)) | 0;
      accumulator1 = (accumulator1 + imul((source[first + 1] << 16) >> 16, f0) +
        imul((source[second + 1] << 16) >> 16, f1)) | 0;
      accumulator2 = (accumulator2 + imul((source[first + 2] << 16) >> 16, f0) +
        imul((source[second + 2] << 16) >> 16, f1)) | 0;
      accumulator3 = (accumulator3 + imul((source[first + 3] << 16) >> 16, f0) +
        imul((source[second + 3] << 16) >> 16, f1)) | 0;
      entering = next;
    }

    const amplitudeLow = amplitude & 0xffff, amplitudeHigh = amplitude >>> 16;
    for (let coordinate = 0; coordinate < resolution; coordinate++) {
      const enteringNext = (entering + 1) & coordinateMask;
      let first = sourceOffset + entering * 4, second = sourceOffset + enteringNext * 4;
      accumulator0 = (accumulator0 + imul((source[first] << 16) >> 16, f0) +
        imul((source[second] << 16) >> 16, f1)) | 0;
      accumulator1 = (accumulator1 + imul((source[first + 1] << 16) >> 16, f0) +
        imul((source[second + 1] << 16) >> 16, f1)) | 0;
      accumulator2 = (accumulator2 + imul((source[first + 2] << 16) >> 16, f0) +
        imul((source[second + 2] << 16) >> 16, f1)) | 0;
      accumulator3 = (accumulator3 + imul((source[first + 3] << 16) >> 16, f0) +
        imul((source[second + 3] << 16) >> 16, f1)) | 0;
      entering = enteringNext;

      const output = destinationOffset + coordinate * 4;
      destination[output] = blurScaleAccumulator(
        accumulator0, amplitudeLow, amplitudeHigh, amplitudeClip);
      destination[output + 1] = blurScaleAccumulator(
        accumulator1, amplitudeLow, amplitudeHigh, amplitudeClip);
      destination[output + 2] = blurScaleAccumulator(
        accumulator2, amplitudeLow, amplitudeHigh, amplitudeClip);
      destination[output + 3] = blurScaleAccumulator(
        accumulator3, amplitudeLow, amplitudeHigh, amplitudeClip);

      const leavingNext = (leaving + 1) & coordinateMask;
      first = sourceOffset + leavingNext * 4;
      second = sourceOffset + leaving * 4;
      accumulator0 = (accumulator0 - imul((source[first] << 16) >> 16, f0) -
        imul((source[second] << 16) >> 16, f1)) | 0;
      accumulator1 = (accumulator1 - imul((source[first + 1] << 16) >> 16, f0) -
        imul((source[second + 1] << 16) >> 16, f1)) | 0;
      accumulator2 = (accumulator2 - imul((source[first + 2] << 16) >> 16, f0) -
        imul((source[second + 2] << 16) >> 16, f1)) | 0;
      accumulator3 = (accumulator3 - imul((source[first + 3] << 16) >> 16, f0) -
        imul((source[second + 3] << 16) >> 16, f1)) | 0;
      leaving = leavingNext;
    }
  }

  function blurTranspose(source, destination, width, height) {
    // The MMX source transposes eight pixels at a time. Keep that traversal
    // for production bitmaps, while allowing the same mapping for tiny direct
    // API fixtures whose width is below eight.
    for (let blockX = 0; blockX < width; blockX += 8) {
      const blockWidth = Math.min(8, width - blockX);
      for (let y = 0; y < height; y++) {
        let sourceIndex = (y * width + blockX) * 4;
        for (let localX = 0; localX < blockWidth; localX++) {
          const destinationIndex = ((blockX + localX) * height + y) * 4;
          destination[destinationIndex] = source[sourceIndex];
          destination[destinationIndex + 1] = source[sourceIndex + 1];
          destination[destinationIndex + 2] = source[sourceIndex + 2];
          destination[destinationIndex + 3] = source[sourceIndex + 3];
          sourceIndex += 4;
        }
      }
    }
  }

  function Bitmap_Blur(input, flags, sx, sy, requestedAmplitude, writable = null) {
    const source = requireBitmap(input);
    const bitmap = writable === RUNTIME_IN_PLACE ? source : source.copy();
    const order = flags & 15;
    if (!order) return bitmap;

    const original = bitmap.data, scratch = new Uint16Array(original.length);
    let current = original, other = scratch;
    let width = bitmap.width, height = bitmap.height;
    // Function arguments are sF32 in the released code. sFtol then observes
    // sFloatFix's round-to-nearest-even mode after each single-precision
    // multiply; the fixed amplitude uses ordinary truncation toward zero.
    let size = roundEven(f32(f32(128 * f32(sx)) * width));
    const size2 = roundEven(f32(f32(128 * f32(sy)) * height));
    const fixedAmplitude = trunc(f32(f32(f32(requestedAmplitude) * 65536) * 64)) | 0;

    for (let axis = 0; axis < 2; axis++) {
      const f1 = trunc((size & 127) / 2), f0 = 64 - f1;
      let kernelSize = trunc(size / 128) * 2;
      if (flags & 0x10) kernelSize++;
      const divisor = Math.max(1, kernelSize * 64 + f1 * 2);
      const amplitude = trunc(fixedAmplitude / divisor) | 0;
      const amplitudeClip = amplitude > 128
        ? (trunc((4194304 * 65536) / amplitude) - 1) | 0
        : 0x7fffffff;

      const rowSource = current, rowScratch = other;
      for (let y = 0; y < height; y++) {
        const rowOffset = y * width * 4;
        let passSource = rowSource, passDestination = rowScratch;
        for (let pass = 0; pass < order; pass++) {
          blurCore(passSource, rowOffset, passDestination, rowOffset,
            kernelSize, width, f0, f1, amplitude, amplitudeClip);
          const swap = passSource; passSource = passDestination; passDestination = swap;
        }
      }

      const blurred = order & 1 ? rowScratch : rowSource;
      const transposed = blurred === original ? scratch : original;
      blurTranspose(blurred, transposed, width, height);
      current = transposed;
      other = blurred;
      const dimension = width; width = height; height = dimension;
      size = size2;
    }

    bitmap.data = bitmap.Data = current;
    return bitmap;
  }

  function Bitmap_Mask(maskInput, bitmapB, bitmapC, mode, writable = null) {
    const mask = requireBitmap(maskInput);
    const bitmap = writable === RUNTIME_IN_PLACE ? mask : blankBitmapCopy(mask);
    const b = requireBitmap(bitmapB), c = requireBitmap(bitmapC);
    if (bitmap.size !== b.size || bitmap.size !== c.size) return null;
    bitmapInner(bitmap.data, bitmap.data, BI.GRAY, mask.data);
    if (mode === 0) bitmapInner(bitmap.data, b.data, BI.MERGE, c.data);
    else if (mode === 1) {
      bitmapInner(bitmap.data, b.data, BI.MUL, bitmap.data);
      bitmapInner(bitmap.data, c.data, BI.ADD, bitmap.data);
    } else if (mode === 2) {
      bitmapInner(bitmap.data, b.data, BI.MUL, bitmap.data);
      bitmapInner(bitmap.data, c.data, BI.SUBR, bitmap.data);
    } else if (mode === 3) bitmapInner(bitmap.data, b.data, BI.MULMERGE, c.data);
    return bitmap;
  }

  function Bitmap_Rotate(input, angle, sx, sy, tx, ty, border, newWidth, newHeight) {
    angle = f32(angle); sx = f32(sx); sy = f32(sy);
    tx = f32(tx); ty = f32(ty);
    if (!sx || !sy) return null;
    const source = requireBitmap(input);
    const xs = source.width, ys = source.height;
    const offset = bitmapTextureSizeOffset;
    const txs = newWidth ? 1 << Math.max(0, newWidth - 1 + offset) : xs;
    const tys = newHeight ? 1 << Math.max(0, newHeight - 1 + offset) : ys;
    const bitmap = new Bitmap(txs, tys);
    bitmap.format = bitmap.Format = source.format;
    // Native replaces only Data on the existing GenBitmap, so format/mipmap
    // policy survives the transform.
    bitmap.texMipCount = bitmap.TexMipCount = source.texMipCount;
    bitmap.texMipThreshold = bitmap.TexMipTresh = source.texMipThreshold;
    bitmap.stripped = bitmap.Stripped = source.stripped;
    const a = f32(angle * f32(TAU));
    const cosine = f32(Math.cos(a)), sine = f32(Math.sin(a));
    const m00 = roundEven(f32(f32(f32(cosine * 0x10000) * xs) /
      f32(f32(sx) * txs)));
    const m01 = roundEven(f32(f32(f32(sine * 0x10000) * ys) /
      f32(f32(sx) * txs)));
    const m10 = roundEven(f32(f32(f32(-sine * 0x10000) * xs) /
      f32(f32(sy) * tys)));
    const m11 = roundEven(f32(f32(f32(cosine * 0x10000) * ys) /
      f32(f32(sy) * tys)));
    // The parenthesized half-offset is integer arithmetic in the C source.
    const centerU = trunc(((Math.imul(txs, m00) + Math.imul(tys, m10)) | 0) / 2);
    const centerV = trunc(((Math.imul(txs, m01) + Math.imul(tys, m11)) | 0) / 2);
    const m20 = roundEven(f32(f32(f32(f32(tx) * xs) * 0x10000) - centerU));
    const m21 = roundEven(f32(f32(f32(f32(ty) * ys) * 0x10000) - centerV));
    const sourceData = source.data, destination = bitmap.data;
    const xMask = xs - 1, yMask = ys - 1;
    const xMax = (xs * 0x10000 - 1) | 0, yMax = (ys * 0x10000 - 1) | 0;
    const clampX = (border & 1) !== 0, clampY = (border & 2) !== 0;
    const point = (border & 4) !== 0;
    for (let y = 0; y < tys; y++) {
      let u = (imul(y, m10) + m20) | 0;
      let v = (imul(y, m11) + m21) | 0;
      let offset4 = y * txs * 4;
      for (let x = 0; x < txs; x++) {
        if (point) {
          // PointFilter deliberately always wraps, independent of border bits.
          const sourceOffset = ((((v >> 16) & yMask) * xs + ((u >> 16) & xMask)) * 4);
          destination[offset4] = sourceData[sourceOffset];
          destination[offset4 + 1] = sourceData[sourceOffset + 1];
          destination[offset4 + 2] = sourceData[sourceOffset + 2];
          destination[offset4 + 3] = sourceData[sourceOffset + 3];
        } else {
          let sampleU = u | 0, sampleV = v | 0;
          sampleU = clampX ? (sampleU < 0 ? 0 : sampleU > xMax ? xMax : sampleU) : sampleU & xMax;
          sampleV = clampY ? (sampleV < 0 ? 0 : sampleV > yMax ? yMax : sampleV) : sampleV & yMax;
          const x0 = sampleU >>> 16, y0 = sampleV >>> 16;
          const x1 = (x0 + 1) & xMask, y1 = (y0 + 1) & yMask;
          const fu = (sampleU & 0xffff) >>> 1, fv = (sampleV & 0xffff) >>> 1;
          const p00 = (y0 * xs + x0) * 4, p10 = (y0 * xs + x1) * 4;
          const p01 = (y1 * xs + x0) * 4, p11 = (y1 * xs + x1) * 4;
          for (let c = 0; c < 4; c++) {
            const top = sourceData[p00 + c], bottom = sourceData[p01 + c];
            const topDelta = ((sourceData[p10 + c] - top) << 16) >> 16;
            const bottomDelta = ((sourceData[p11 + c] - bottom) << 16) >> 16;
            const h0 = (top + (((topDelta * fu) >> 16) << 1)) & 0xffff;
            const h1 = (bottom + (((bottomDelta * fu) >> 16) << 1)) & 0xffff;
            const verticalDelta = ((h1 - h0) << 16) >> 16;
            const biased = (h0 + 0x8000 + (((verticalDelta * fv) >> 16) << 1)) & 0xffff;
            destination[offset4 + c] = biased < 0x8000 ? 0 : biased - 0x8000;
          }
        }
        u = (u + m00) | 0; v = (v + m01) | 0;
        offset4 += 4;
      }
    }
    return bitmap;
  }

  function Bitmap_RotateMul(input, angle, sx, sy, tx, ty, border, color, mode, count, fade,
    writable = null) {
    if (count <= 0) return null;
    // Only the initial Color may consume the caller's primary input. The
    // temporaries created below are privately owned by RotateMul, matching the
    // native reference-count fast path, so their Color/Merge stages can update
    // them in place without weakening the public copy-on-write API.
    let bitmap = Bitmap_Color(input, 0, color, writable);
    let accumulated = bitmap.copy();
    angle = f32(angle); sx = f32(sx); sy = f32(sy);
    tx = f32(tx); ty = f32(ty);
    const signX = sx < 0 ? f32(-1) : f32(1), signY = sy < 0 ? f32(-1) : f32(1);
    sx = f32(sx * signX); sy = f32(sy * signY);
    let i = 1;
    if (mode & 16) {
      const span = (1 << count) - 1;
      angle = f32(angle / f32(1 << count));
      tx = f32(f32(f32(tx - 0.5) / f32(span)) + 0.5);
      ty = f32(f32(f32(ty - 0.5) / f32(span)) + 0.5);
      const inverseSpan = f32(1 / f32(span));
      sx = f32(Math.pow(sx, inverseSpan)); sy = f32(Math.pow(sy, inverseSpan));
    } else {
      const countFloat = f32(count);
      angle = f32(angle / countFloat);
      tx = f32(f32(f32(tx - 0.5) / countFloat) + 0.5);
      ty = f32(f32(f32(ty - 0.5) / countFloat) + 0.5);
      const inverseCount = f32(1 / countFloat);
      sx = f32(Math.pow(sx, inverseCount)); sy = f32(Math.pow(sy, inverseCount));
    }
    sx = f32(sx * signX); sy = f32(sy * signY);
    while (count-- > 0) {
      if (mode & 16) {
        if (fade !== 0xffffffff) {
          bitmap = Bitmap_Color(bitmap, 0, fade, RUNTIME_IN_PLACE);
          let a = (fade >>> 24) & 255, r = (fade >>> 16) & 255;
          let g = (fade >>> 8) & 255, b = fade & 255;
          a = a * a >>> 8; r = r * r >>> 8;
          g = g * g >>> 8; b = b * b >>> 8;
          fade = ((a << 24) | (r << 16) | (g << 8) | b) >>> 0;
        }
        bitmap = Bitmap_Rotate(bitmap, angle, sx, sy, tx, ty, border, 0, 0);
        bitmap = Bitmap_Merge(mode & 15, [bitmap, accumulated], RUNTIME_IN_PLACE);
        accumulated = bitmap.copy();
        angle = f32(angle * 2);
        tx = f32(f32(f32(tx - 0.5) * 2) + 0.5);
        ty = f32(f32(f32(ty - 0.5) * 2) + 0.5);
        sx = f32(sx * sx); sy = f32(sy * sy);
      } else {
        if (fade !== 0xffffffff) {
          bitmap = Bitmap_Color(bitmap, 0, fade, RUNTIME_IN_PLACE);
        }
        const iteration = f32(i);
        const rotated = Bitmap_Rotate(bitmap, f32(angle * iteration),
          f32(f32(f32(sx - 1) * iteration) + 1),
          f32(f32(f32(sy - 1) * iteration) + 1),
          f32(f32(f32(tx - 0.5) * iteration) + 0.5),
          f32(f32(f32(ty - 0.5) * iteration) + 0.5), border, 0, 0);
        accumulated = Bitmap_Merge(mode & 15, [accumulated, rotated], RUNTIME_IN_PLACE);
        i++;
      }
    }
    return accumulated;
  }

  function Bitmap_Distort(sampleInput, displacementInput, distance, border, writable = null) {
    const sample = requireBitmap(sampleInput), displacement = requireBitmap(displacementInput);
    if (sample.size !== displacement.size) return null;
    const bitmap = writable === RUNTIME_IN_PLACE ? displacement : blankBitmapCopy(displacement);
    // Native validates only Size and sets BilinearSetup's dimensions from the
    // writable displacement input. Equal-area, differently shaped sample
    // buffers are therefore deliberately reinterpreted with that shape.
    const context = new BilinearContext(sample, border, bitmap.width, bitmap.height);
    distance = f32(distance);
    const bumpX = trunc(f32(f32(distance * bitmap.width) * 4));
    const bumpY = trunc(f32(f32(distance * bitmap.height) * 4));
    for (let y = 0; y < bitmap.height; y++) for (let x = 0; x < bitmap.width; x++) {
      const i = (y * bitmap.width + x) * 4;
      const u = ((x << 16) + (displacement.data[i] - HALF) * bumpX) | 0;
      const v = ((y << 16) + (displacement.data[i + 1] - HALF) * bumpY) | 0;
      context.sample(u, v, bitmap.data, i);
    }
    return bitmap;
  }

  function Bitmap_Normals(input, distance, mode) {
    const source = requireBitmap(input), bitmap = blankBitmapCopy(source);
    const xs = source.width, ys = source.height;
    const dist = roundEven(f32(f32(distance) * 65536));
    const shiftX = powerOfTwo(xs), shiftY = powerOfTwo(ys);
    // Native GenBitmap words are BGRA and filterbump reads word zero, i.e. the
    // blue height channel. This port stores RGBA, so the equivalent word is 2.
    const heightAt = (x, y) => source.data[
      (((y & (ys - 1)) * xs + (x & (xs - 1))) * 4) + 2
    ];
    for (let y = 0; y < ys; y++) for (let x = 0; x < xs; x++) {
      let vx, vy;
      if (mode & 4) {
        vx = 4 * (heightAt(x - 1, y) - heightAt(x, y));
        vy = 4 * (heightAt(x, y - 1) - heightAt(x, y));
      } else {
        vx = heightAt(x - 2, y) + 3 * heightAt(x - 1, y) - 3 * heightAt(x, y) - heightAt(x + 1, y);
        vy = heightAt(x, y - 2) + 3 * heightAt(x, y - 1) - 3 * heightAt(x, y) - heightAt(x, y + 1);
      }
      // The released code performs a signed 32-bit multiply followed by SAR.
      // Math.imul preserves the wraparound and >> preserves the negative-floor
      // behavior which truncating JS division would lose.
      vx = clamp15((imul(vx, dist >> 4) >> (20 - shiftX)) + HALF) - HALF;
      vy = clamp15((imul(vy, dist >> 4) >> (20 - shiftY)) + HALF) - HALF;
      let vz = 0;
      if (mode & 1) {
        const z2 = 0x3fff * 0x3fff - vx * vx - vy * vy;
        // sFSqrt returns through the x87 stack with sFloatFix's single-
        // precision control before the ordinary truncating integer store.
        // For a one-unit slope sqrt(0x3fff^2-1) consequently rounds back to
        // exactly 0x3fff instead of truncating the double result to 0x3ffe.
        if (z2 > 0) vz = trunc(f32(Math.sqrt(z2)));
        else {
          // Native stores the inverse-length scale in sF32 and multiplies it
          // back into integer vx/vy through an sF32 expression before the
          // language-mandated truncating conversion to sInt.
          const scale = f32(0x3fff / Math.sqrt(vx * vx + vy * vy));
          vx = trunc(f32(vx * scale));
          vy = trunc(f32(vy * scale));
        }
      }
      if (mode & 2) { const swap = vx; vx = vy; vy = -swap; }
      const i = (y * xs + x) * 4;
      bitmap.data[i] = vx + HALF; bitmap.data[i + 1] = vy + HALF;
      bitmap.data[i + 2] = vz + HALF; bitmap.data[i + 3] = 0xffff;
    }
    return bitmap;
  }

  function Bitmap_Bump(input, normalsInput, subcode, px, py, pz, da, db,
    diffuseColor, ambientColor, outer, falloff, amp, specularColor, specPower, specAmp,
    writable = null) {
    if (!outer) return null;
    const source = requireBitmap(input), normals = normalsInput ? requireBitmap(normalsInput) : null;
    if (normals && (normals.width !== source.width || normals.height !== source.height)) return null;
    const bitmap = writable === RUNTIME_IN_PLACE ? source : blankBitmapCopy(source);
    const xs = source.width, ys = source.height;
    const diffuse = packedColor(diffuseColor), ambient = packedColor(ambientColor), specular = packedColor(specularColor);
    px = f32(f32(f32(f32(px) * xs) * 2) - trunc(xs / 2));
    py = f32(f32(f32(f32(py) * ys) * 2) - trunc(ys / 2));
    pz = f32(f32(pz) * xs);
    da = f32(f32(da) * f32(TAU)); db = f32(f32(db) * f32(Math.PI));
    const cosineB = f32(Math.cos(db));
    const sineA = f32(Math.sin(da)), cosineA = f32(Math.cos(da));
    const dx = f32(cosineB * sineA);
    const dy = f32(cosineB * cosineA);
    const dz = f32(Math.sin(db));
    if (subcode === 0) {
      px = f32(px - f32(f32(dx * pz) / dz));
      py = f32(py - f32(f32(dy * pz) / dz));
    }
    outer = f32(outer); falloff = f32(falloff); amp = f32(amp);
    specPower = f32(specPower);
    const scaledSpec = f32(f32(specAmp) * 65536);
    const directionalHalfInverse = subcode === 2 && scaledSpec
      ? f32(1 / (Math.sqrt(f32(f32(f32(dx * dx) + f32(dy * dy)) +
        f32(f32(dz + 1) * f32(dz + 1)))) || 1))
      : 0;
    // Bitmap_Bump overwrites all four words before each use. Reusing one
    // scratch pixel avoids hundreds of thousands of tiny typed-array
    // allocations on a production-sized texture without changing arithmetic.
    const lit = new Uint16Array(4);

    // Every authored Debris Bump is a directional light with a normal map.
    // Split that stable case out of the general point/spot loop: the native
    // source also keeps its light and halfway vectors outside the pixel walk,
    // and avoiding four invariant branches per pixel is material at 512².
    if (subcode === 2 && normals) {
      const sourceData = source.data, normalData = normals.data, output = bitmap.data;
      if (scaledSpec) {
        const hx = dx, hy = dy, hz = f32(dz + 1);
        for (let i = 0; i < sourceData.length; i += 4) {
          const nx = (normalData[i] - HALF) / HALF;
          const ny = (normalData[i + 1] - HALF) / HALF;
          const nz = (normalData[i + 2] - HALF) / HALF;
          const light = f32(f32(f32(dx * nx) + f32(dy * ny)) + f32(dz * nz));
          let specFactor = f32(f32(f32(hx * nx) + f32(hy * ny)) + f32(hz * nz));
          if (specFactor < 0) specFactor = 0;
          specFactor = f32(Math.pow(f32(specFactor * directionalHalfInverse), specPower));
          const strength = f32(light * amp);
          for (let c = 0; c < 4; c++) {
            const lighting = f32(ambient[c] + f32(diffuse[c] * strength));
            const value = f32(f32(sourceData[i + c] * lighting) / 0x8000);
            lit[c] = clamp15(roundEven(value));
          }
          addScalePixel(output, i, lit, 0, specular, 0,
            roundEven(f32(specFactor * scaledSpec)));
        }
      } else {
        for (let i = 0; i < sourceData.length; i += 4) {
          const nx = (normalData[i] - HALF) / HALF;
          const ny = (normalData[i + 1] - HALF) / HALF;
          const nz = (normalData[i + 2] - HALF) / HALF;
          const light = f32(f32(f32(dx * nx) + f32(dy * ny)) + f32(dz * nz));
          const strength = f32(light * amp);
          for (let c = 0; c < 4; c++) {
            const lighting = f32(ambient[c] + f32(diffuse[c] * strength));
            const value = f32(f32(sourceData[i + c] * lighting) / 0x8000);
            output[i + c] = clamp15(roundEven(value));
          }
        }
      }
      return bitmap;
    }

    for (let y = 0; y < ys; y++) for (let x = 0; x < xs; x++) {
      const i = (y * xs + x) * 4;
      let lx = dx, ly = dy, lz = dz;
      if (subcode !== 2) {
        lx = x - px; ly = y - py; lz = pz;
        lx = f32(lx); ly = f32(ly);
        const length2 = f32(f32(f32(lx * lx) + f32(ly * ly)) + f32(lz * lz));
        const inverse = f32(1 / (Math.sqrt(length2) || 1));
        lx = f32(lx * inverse); ly = f32(ly * inverse); lz = f32(lz * inverse);
      }
      let nx = 0, ny = 0, nz = 1;
      if (normals) {
        nx = (normals.data[i] - HALF) / HALF; ny = (normals.data[i + 1] - HALF) / HALF;
        nz = (normals.data[i + 2] - HALF) / HALF;
      }
      const light = f32(f32(f32(lx * nx) + f32(ly * ny)) + f32(lz * nz));
      let specFactor = 0;
      if (scaledSpec) {
        const hx = lx, hy = ly, hz = f32(lz + 1);
        const inverse = subcode === 2 ? directionalHalfInverse
          : f32(1 / (Math.sqrt(f32(f32(f32(hx * hx) + f32(hy * hy)) + f32(hz * hz))) || 1));
        specFactor = f32(f32(f32(hx * nx) + f32(hy * ny)) + f32(hz * nz));
        if (specFactor < 0) specFactor = 0;
        specFactor = f32(Math.pow(f32(specFactor * inverse), specPower));
      }
      let direction = 1;
      if (subcode === 0) {
        direction = f32(f32(f32(lx * dx) + f32(ly * dy)) + f32(lz * dz));
        direction = direction < outer ? 0
          : f32(Math.pow(f32(f32(direction - outer) / f32(1 - outer)), falloff));
      }
      const strength = f32(f32(direction * light) * amp);
      for (let c = 0; c < 4; c++) {
        const lighting = f32(ambient[c] + f32(diffuse[c] * strength));
        lit[c] = clamp15(roundEven(f32(f32(source.data[i + c] * lighting) / 0x8000)));
      }
      addScalePixel(bitmap.data, i, lit, 0, specular, 0,
        roundEven(f32(f32(direction * specFactor) * scaledSpec)));
    }
    return bitmap;
  }

  let perlinState = null;
  function getPerlinState() {
    if (perlinState) return perlinState;
    const random = new Random();
    random.setSeed(1);
    const order = new Array(256);
    for (let i = 0; i < 256; i++) order[i] = { key: random.int(0x10000), value: i };
    // The source uses a stable exchange sort (strict >), which matters when
    // two 16-bit keys collide.
    order.sort((a, b) => a.key - b.key);
    const permutation = new Uint8Array(512);
    for (let i = 0; i < 256; i++) permutation[i] = permutation[i + 256] = order[i].value;
    const gradients = new Float32Array(512);
    for (let i = 0; i < 256;) {
      const x = random.int(0x10000) - 0x8000;
      const y = random.int(0x10000) - 0x8000;
      if (x * x + y * y < 0x8000 * 0x8000) {
        gradients[i * 2] = f32(x / 32768);
        gradients[i * 2 + 1] = f32(y / 32768);
        i++;
      }
    }
    return (perlinState = { permutation, gradients });
  }

  function noise2WithTables(x, y, mask, seed, p, g) {
    mask &= 255; seed &= 255;
    const vx = (x >> 16) & mask, vy = (y >> 16) & mask;
    let tx = (x & 0xffff) / 65536, ty = (y & 0xffff) / 65536;
    const vy0 = p[vy ^ seed], vy1 = p[((vy + 1) & mask) ^ seed];
    const v00 = p[vx + vy0], v01 = p[((vx + 1) & mask) + vy0];
    const v10 = p[vx + vy1], v11 = p[((vx + 1) & mask) + vy1];
    const f00 = g[v00 * 2] * tx + g[v00 * 2 + 1] * ty;
    const f01 = g[v01 * 2] * (tx - 1) + g[v01 * 2 + 1] * ty;
    const f10 = g[v10 * 2] * tx + g[v10 * 2 + 1] * (ty - 1);
    const f11 = g[v11 * 2] * (tx - 1) + g[v11 * 2 + 1] * (ty - 1);
    tx = tx * tx * tx * (10 + tx * (6 * tx - 15));
    ty = ty * ty * ty * (10 + ty * (6 * ty - 15));
    const a = f00 + (f01 - f00) * tx, b = f10 + (f11 - f10) * tx;
    return a + (b - a) * ty;
  }

  function noise2(x, y, mask, seed) {
    const { permutation, gradients } = getPerlinState();
    return noise2WithTables(x, y, mask, seed, permutation, gradients);
  }

  function Bitmap_Perlin(xs, ys, frequency, octaves, fadeoff, seed, mode, amplitude, gamma, color0, color1) {
    const bitmap = newBitmap(xs | 0, ys | 0), c0 = packedColor(color0), c1 = packedColor(color1);
    const shiftX = 16 - powerOfTwo(bitmap.width), shiftY = 16 - powerOfTwo(bitmap.height);
    frequency |= 0; octaves |= 0; seed &= 255; mode &= 3;
    fadeoff = f32(fadeoff); amplitude = f32(amplitude); gamma = f32(gamma);
    const gammaTable = new Int32Array(1025);
    for (let i = 0; i < 1025; i++) {
      gammaTable[i] = clamp15(Math.pow(i / 1024, gamma) * 0x8000) * 2;
    }
    amplitude = f32(amplitude * ((mode & 1) ? 0x8000 : 0x4000));
    const amplitudeFixed = roundEven(amplitude);
    const offset = (mode & 1) ? 0 : HALF;

    // The released #if 1 generator resolves sine-shaped noise through this
    // fixed-point table rather than calling sin for every pixel and octave.
    const sine = mode & 2 ? new Int32Array(257) : null;
    if (sine) {
      const tau = f32(TAU);
      for (let x = 0; x < sine.length; x++) {
        const angle = f32(f32(tau * x) / 256);
        sine[x] = roundEven(f32(Math.sin(angle) * 0.5 * 65536));
      }
    }

    // One polynomial period is shared by every group and octave. Explicit
    // f32 stores reproduce the single-precision x87 mode selected by
    // sFloatFix; sFtol rounds each table entry to nearest-even.
    const polynomial = new Int32Array(bitmap.width >> frequency);
    for (let x = 0; x < polynomial.length; x++) {
      const coordinate = f32(x / polynomial.length);
      let value = f32(f32(coordinate * coordinate) * coordinate);
      const inner = f32(10 + f32(coordinate * f32(f32(6 * coordinate) - 15)));
      value = f32(f32(value * inner) * 16384);
      polynomial[x] = roundEven(value);
    }

    const { permutation, gradients } = getPerlinState();
    const row = new Int32Array(bitmap.width);
    let destination = 0;
    for (let y = 0; y < bitmap.height; y++) {
      row.fill(0);
      let octaveScale = f32(1);
      for (let octave = frequency; octave < frequency + octaves; octave++) {
        const grouped = shiftX + octave < 16;
        const groupSize = grouped
          ? Math.min(bitmap.width, 1 << (16 - shiftX - octave)) : 1;
        const groups = grouped
          ? bitmap.width >> (16 - shiftX - octave) : bitmap.width;
        const mask = ((1 << octave) - 1) & 255;
        const py = (y << (shiftY + octave)) | 0;
        const vy = (py >> 16) & mask;
        const deltaX = 1 << (shiftX + octave);
        const ty = f32((py & 0xffff) / 65536);
        let tyFade = f32(f32(ty * ty) * ty);
        tyFade = f32(tyFade * f32(10 + f32(ty * f32(f32(6 * ty) - 15))));
        const ty0 = f32(ty * f32(1 - tyFade));
        const ty1 = f32(f32(ty - 1) * tyFade);
        const vy0 = permutation[vy ^ seed];
        const vy1 = permutation[((vy + 1) & mask) ^ seed];
        const polynomialShift = octave - frequency;
        const scaleFixed = roundEven(f32(octaveScale * 16384));

        // When both axes land exactly on lattice points the gradient dot
        // product is zero. The native path skips the complete octave row.
        if (shiftX + octave < 16 || (py & 0xffff)) {
          let rowIndex = 0;
          for (let vx = 0; vx < groups; vx++) {
            const v00 = permutation[((vx + 0) & mask) + vy0];
            const v01 = permutation[((vx + 1) & mask) + vy0];
            const v10 = permutation[((vx + 0) & mask) + vy1];
            const v11 = permutation[((vx + 1) & mask) + vy1];
            const v00i = v00 * 2, v01i = v01 * 2, v10i = v10 * 2, v11i = v11 * 2;

            const horizontal0 = f32(gradients[v00i] +
              f32(f32(gradients[v10i] - gradients[v00i]) * tyFade));
            const horizontal1 = f32(gradients[v01i] +
              f32(f32(gradients[v11i] - gradients[v01i]) * tyFade));
            const vertical0 = f32(f32(gradients[v00i + 1] * ty0) +
              f32(gradients[v10i + 1] * ty1));
            const vertical1 = f32(f32(gradients[v01i + 1] * ty0) +
              f32(gradients[v11i + 1] * ty1));

            let fa = roundEven(f32(vertical0 * 65536));
            let fb = roundEven(f32(f32(vertical1 - horizontal1) * 65536));
            const faDelta = roundEven(f32(horizontal0 * deltaX));
            const fbDelta = roundEven(f32(horizontal1 * deltaX));

            for (let x = 0; x < groupSize; x++) {
              let noise = (fa + (imul((fb - fa) | 0,
                polynomial[x << polynomialShift]) >> 14)) | 0;
              if (mode === 1) noise = Math.abs(noise) | 0;
              else if (mode & 2) {
                if (mode === 3) noise &= 0x7fff;
                const index = (noise >> 8) & 255;
                noise = (sine[index] + (imul(sine[index + 1] - sine[index],
                  noise & 255) >> 8)) | 0;
              }
              row[rowIndex] = (row[rowIndex] + (imul(noise, scaleFixed) >> 14)) | 0;
              rowIndex++;
              fa = (fa + faDelta) | 0;
              fb = (fb + fbDelta) | 0;
            }
          }
        }
        octaveScale = f32(octaveScale * fadeoff);
      }
      for (let x = 0; x < bitmap.width; x++) {
        const raw = clamp15((fixedMul(row[x], amplitudeFixed) + offset) | 0);
        const fade = gammaLookup(gammaTable, raw, 5);
        fadePixel(bitmap.data, destination, c0, 0, c1, 0, fade);
        destination += 4;
      }
    }
    return bitmap;
  }

  function Bitmap_Cell(xs, ys, color0, color1, color2, count, seed, amplitude, gamma,
    mode, minimumDistance, percent, aspectParameter) {
    amplitude = f32(amplitude); gamma = f32(gamma);
    minimumDistance = f32(minimumDistance); aspectParameter = f32(aspectParameter);
    if (count < 1 || count > 256) return null;
    const bitmap = newBitmap(xs | 0, ys | 0), random = new Random();
    random.setSeed(seed | 0);
    const cells = Array.from({ length: count }, () => [random.int(0x4000), random.int(0x4000), random.int(0x4000)]);
    const minimumDistanceFixed = roundEven(f32(minimumDistance * 0x4000));
    const minDistance2 = minimumDistanceFixed * minimumDistanceFixed;
    for (let i = 1; i < cells.length;) {
      if ((mode & 2) && random.int(255) < percent) cells[i][2] = 0xffff;
      const px = (cells[i][0] & 0x3fff) - 0x2000, py = (cells[i][1] & 0x3fff) - 0x2000;
      let cut = false;
      for (let j = 0; j < i && !cut; j++) {
        const dx = ((cells[j][0] - px) & 0x3fff) - 0x2000;
        const dy = ((cells[j][1] - py) & 0x3fff) - 0x2000;
        cut = dx * dx + dy * dy < minDistance2;
      }
      if (cut) {
        // C++ decrements max and copies the former last cell into slot i. When
        // i already is the last slot, assigning after Array.pop() would grow
        // the JS array back and retry the rejected point forever.
        const replacement = cells.pop();
        if (i < cells.length) cells[i] = replacement;
      } else i++;
    }
    const shiftX = 14 - powerOfTwo(bitmap.width), shiftY = 14 - powerOfTwo(bitmap.height);
    const c0 = packedColor(color1), c1 = packedColor(color0), cb = packedColor(color2);
    // aspect, aspf and aspdiv are all sF32 in the released code.
    const aspect = f32(Math.pow(2, aspectParameter));
    const aspectFixed = aspect >= 1
      ? trunc(f32(65536 / f32(aspect * aspect)))
      : trunc(f32(f32(aspect * aspect) * 65536));
    const divisor = aspect >= 1 ? f32(aspect / 16384) : f32(1 / f32(16384 * aspect));
    // Mode 2 overwrites the complete temporary pixel on every iteration.
    // Keep one scratch allocation rather than one per generated pixel.
    const cellColor = mode & 2 ? new Uint16Array(4) : null;
    const tiled = bitmap.width >= 16 && bitmap.height >= 16 &&
      (bitmap.width & 15) === 0 && (bitmap.height & 15) === 0;
    if (!tiled) {
      // The released non-intro loop below assumes complete 16x16 tiles.
      // Retain the exhaustive implementation for tiny direct-API fixtures
      // (and for any future bitmap size which is not tile-aligned).
      for (let y = 0; y < bitmap.height; y++) for (let x = 0; x < bitmap.width; x++) {
        let px = x << shiftX, py = y << shiftY;
        if (aspect < 1) { const swap = px; px = py; py = swap; }
        let best = Infinity, second = Infinity, bestIndex = -1;
        for (let i = 0; i < cells.length; i++) {
          const cx = aspect < 1 ? cells[i][1] : cells[i][0];
          const cy = aspect < 1 ? cells[i][0] : cells[i][1];
          const dx = ((cx - px) & 0x3fff) - 0x2000;
          const dy = ((cy - py) & 0x3fff) - 0x2000;
          const distance = fixedMul(dx * dx, aspectFixed) + dy * dy;
          if (distance < best) { second = best; best = distance; bestIndex = i; }
          else if (distance > best && distance < second) second = distance;
        }
        // v0 and v1 are sF32 locals, so each assignment rounds to single
        // precision before the next step reads it (genbitmap.cpp:2881-2890).
        let value = f32(Math.sqrt(best) * divisor);
        if (mode & 1) {
          const v1 = f32(Math.sqrt(second) * divisor);
          value = f32(value + v1) > 0.00001 ? f32((v1 - value) / (v1 + value)) : 0;
        }
        let fade = clamp15(f32(Math.pow(f32(value * amplitude), gamma)) * 0x8000) * 2;
        if (mode & 4) fade = 0x10000 - fade;
        const offset4 = (y * bitmap.width + x) * 4;
        if (mode & 2) {
          if (cells[bestIndex][2] === 0xffff) cellColor.set(cb);
          else fadePixel(cellColor, 0, c0, 0, c1, 0, cells[bestIndex][2] * 4);
          fadePixel(bitmap.data, offset4, cellColor, 0, cb, 0, fade);
        } else fadePixel(bitmap.data, offset4, c0, 0, c1, 0, fade);
      }
      return bitmap;
    }

    // genbitmap.cpp's released non-intro path sorts cells for each 16x16
    // output tile by a conservative squared-distance lower bound. Once the
    // second-best exact distance is no greater than the next bound, no later
    // cell can affect the pixel. The strict insertion comparison is important:
    // equal-bound cells retain the order left by the preceding tile, matching
    // the source's stable insertion sort and its strict nearest-cell ties.
    const flipXY = aspect < 1;
    if (flipXY) {
      for (const cell of cells) {
        const swap = cell[0]; cell[0] = cell[1]; cell[1] = swap;
      }
    }
    const bounds = new Int32Array(cells.length);
    const tileSize = 16;
    for (let by = 0; by < bitmap.height; by += tileSize) {
      for (let bx = 0; bx < bitmap.width; bx += tileSize) {
        let px0 = bx << shiftX, px1 = (bx + tileSize - 1) << shiftX;
        let py0 = by << shiftY, py1 = (by + tileSize - 1) << shiftY;
        if (flipXY) {
          let swap = px0; px0 = py0; py0 = swap;
          swap = px1; px1 = py1; py1 = swap;
        }

        for (let i = 0; i < cells.length; i++) {
          const cell = cells[i];
          let dx = ((cell[0] - px0) & 0x3fff) - 0x2000;
          let dy = ((cell[0] - px1) & 0x3fff) - 0x2000;
          let bound = 0;
          if ((dx ^ dy) > 0) {
            dx = Math.min(Math.abs(dx), Math.abs(dy));
            bound = fixedMul(dx * dx, aspectFixed);
          }
          dx = ((cell[1] - py0) & 0x3fff) - 0x2000;
          dy = ((cell[1] - py1) & 0x3fff) - 0x2000;
          if ((dx ^ dy) > 0) {
            dy = Math.min(Math.abs(dx), Math.abs(dy));
            bound += dy * dy;
          }
          bounds[i] = bound;
        }

        for (let i = 1; i < cells.length; i++) {
          const cell = cells[i], bound = bounds[i];
          let j = i;
          while (j && bounds[j - 1] > bound) {
            cells[j] = cells[j - 1];
            bounds[j] = bounds[j - 1];
            j--;
          }
          cells[j] = cell;
          bounds[j] = bound;
        }

        for (let ty = 0; ty < tileSize; ty++) {
          const py = (by + ty) << shiftY;
          for (let tx = 0; tx < tileSize; tx++) {
            const px = (bx + tx) << shiftX;
            const x = flipXY ? py : px;
            const y = flipXY ? px : py;
            let best = 0x40000000, second = 0x40000000, bestIndex = -1;
            for (let i = 0; i < cells.length && second > bounds[i]; i++) {
              const cell = cells[i];
              const dx = ((cell[0] - x) & 0x3fff) - 0x2000;
              const dy = ((cell[1] - y) & 0x3fff) - 0x2000;
              const distance = fixedMul(dx * dx, aspectFixed) + dy * dy;
              if (distance < best) {
                second = best;
                best = distance;
                bestIndex = i;
              } else if (distance > best && distance < second) second = distance;
            }

            // Same sF32 narrowing as the scalar path above.
            let value = f32(Math.sqrt(best) * divisor);
            if (mode & 1) {
              const v1 = f32(Math.sqrt(second) * divisor);
              value = f32(value + v1) > 0.00001 ? f32((v1 - value) / (v1 + value)) : 0;
            }
            let fade = clamp15(f32(Math.pow(f32(value * amplitude), gamma)) * 0x8000) * 2;
            if (mode & 4) fade = 0x10000 - fade;
            const offset4 = ((by + ty) * bitmap.width + bx + tx) * 4;
            if (mode & 2) {
              if (cells[bestIndex][2] === 0xffff) cellColor.set(cb);
              else fadePixel(cellColor, 0, c0, 0, c1, 0, cells[bestIndex][2] * 4);
              fadePixel(bitmap.data, offset4, cellColor, 0, cb, 0, fade);
            } else fadePixel(bitmap.data, offset4, c0, 0, c1, 0, fade);
          }
        }
      }
    }
    return bitmap;
  }

  function Bitmap_Gradient(xs, ys, color0, color1, position, angle, length, mode) {
    const bitmap = newBitmap(xs | 0, ys | 0), c0 = packedColor(color0), c1 = packedColor(color1);
    const l = f32(32768 / f32(length));
    const radians = f32(f32(angle) * f32(TAU));
    const dx = roundEven(f32(f32(Math.cos(radians)) * l));
    const dy = roundEven(f32(f32(Math.sin(radians)) * l));
    const cdx = fixedMul(dx, trunc(0x10000 / bitmap.width));
    const cdy = (fixedMul(dy, trunc(0x10000 / bitmap.height)) -
      imul(bitmap.width, cdx)) | 0;
    // dx/2 and dy/2 are signed integer divisions in the source, not floating
    // halves. The final float-to-int assignment itself truncates.
    const halfDx = trunc(dx / 2), halfDy = trunc(dy / 2);
    let coordinate = trunc(f32(HALF - f32((halfDx + halfDy) * f32(f32(position) + 1))));
    for (let y = 0; y < bitmap.height; y++) {
      for (let x = 0; x < bitmap.width; x++) {
        let fade;
        if (mode === 1) {
          const phase = f32(f32(f32(clamp15(coordinate) * f32(TAU)) / 0x8000) + 0x2000);
          fade = roundEven(f32(f32(f32(Math.sin(phase)) * MAX) + MAX));
        } else if (mode === 2) {
          const phase = f32(f32(clamp15(coordinate) * f32(TAU)) / 0x10000);
          fade = roundEven(f32(f32(Math.sin(phase)) * 0xffff));
        }
        else fade = clamp15(coordinate) * 2;
        fadePixel(bitmap.data, (y * bitmap.width + x) * 4, c0, 0, c1, 0, fade);
        coordinate = (coordinate + cdx) | 0;
      }
      coordinate = (coordinate + cdy) | 0;
    }
    return bitmap;
  }

  function Bitmap_Sharpen(input, order, sx, sy, amplitude) {
    const original = requireBitmap(input);
    const bitmap = Bitmap_Blur(original, (order | 0) | 0x10, sx, sy, 1);
    const value = Math.max(-MAX, Math.min(MAX, trunc(f32(f32(amplitude) * 0x800))));
    const constant = new Int16Array([value, value, value, value]);
    bitmapInner(bitmap.data, constant, BI.SHARPEN, original.data);
    return bitmap;
  }

  function bitmapColorBalance(input, values, writable) {
    const source = requireBitmap(input);
    const bitmap = writable === RUNTIME_IN_PLACE ? source : blankBitmapCopy(source);
    const tables = [new Int32Array(257), new Int32Array(257), new Int32Array(257)];
    const scale = f32(100 / 255);
    for (let channel = 0; channel < 3; channel++) {
      const shadow = f32(values[channel]), midtone = f32(values[channel + 3]);
      const highlight = f32(values[channel + 6]);
      const exponentArgument = f32(f32(f32(shadow * 0.5) + midtone) + f32(highlight * 0.5));
      const exponent = f32(Math.pow(0.5, exponentArgument));
      const minimum = f32(-Math.min(shadow, 0) * scale);
      const maximum = f32(1 - f32(Math.max(highlight, 0) * scale));
      const multiplier = f32(1 / f32(maximum - minimum));
      for (let i = 0; i <= 256; i++) {
        const coordinate = f32(i / 256);
        const bounded = Math.max(minimum, Math.min(maximum, coordinate));
        const normalized = f32(f32(bounded - minimum) * multiplier);
        const mapped = f32(Math.pow(normalized, exponent));
        tables[channel][i] = clamp15(f32(mapped * 32768));
      }
    }
    for (let i = 0; i < bitmap.data.length; i += 4) {
      for (let c = 0; c < 3; c++) bitmap.data[i + c] = gammaLookup(tables[c], source.data[i + c], 7);
      bitmap.data[i + 3] = source.data[i + 3];
    }
    return bitmap;
  }

  function Bitmap_ColorBalance(input, ...values) {
    return bitmapColorBalance(input, values, null);
  }

  function Bitmap_Unwrap(input, mode) {
    const source = requireBitmap(input), bitmap = blankBitmapCopy(source);
    const context = new BilinearContext(source, (mode >> 4) & 3);
    const xScale = source.width * 0x10000, yScale = source.height * 0x10000;
    for (let y = 0; y < source.height; y++) {
      const fy = y / source.height, fyc = 0.5 - fy;
      for (let x = 0; x < source.width; x++) {
        const fx = x / source.width, fxc = fx - 0.5;
        let u = fx, v = fy;
        switch (mode & 3) {
          case 0: v = (1 - Math.sin(fx * TAU) * fy) * 0.5; u = (1 + Math.cos(fx * TAU) * fy) * 0.5; break;
          case 1: u = Math.atan2(fyc, fxc) / TAU; if (u < 0) u += 1; v = Math.sqrt(fxc * fxc + fyc * fyc) * 2; break;
          case 2:
            if (Math.abs(fyc) <= Math.abs(fxc)) { v = Math.abs(fxc) * 2; u = 0.25 - fy / (v * 2); if (fxc < 0) u = 0.75 - u; }
            else { v = Math.abs(fyc) * 2; u = 0.5 - fx / (v * 2); if (fyc < 0) u = 1 - u; }
            break;
        }
        context.sample(trunc(u * xScale), trunc(v * yScale), bitmap.data, (y * source.width + x) * 4);
      }
    }
    return bitmap;
  }

  function Bitmap_Bulge(input, warp) {
    const source = requireBitmap(input), bitmap = blankBitmapCopy(source), context = new BilinearContext(source, 0);
    const xScale = source.width * 0x10000, yScale = source.height * 0x10000;
    const invX = f32(1 / source.width), invY = f32(1 / source.height);
    warp = f32(warp);
    for (let y = 0; y < source.height; y++) for (let x = 0; x < source.width; x++) {
      const fx = f32(x * invX), fy = f32(y * invY);
      let u = f32(f32(fx - 0.5) * 2), v = f32(f32(0.5 - fy) * 2);
      const radius2 = f32(f32(u * u) + f32(v * v));
      if (radius2 <= 1) {
        const root = f32(Math.sqrt(f32(1 - radius2)));
        const denominator = f32(1 + f32(warp * root));
        const inverse = f32(1 / denominator);
        u = f32(u * inverse); v = f32(v * inverse);
      }
      u = f32(f32(1 + u) * 0.5); v = f32(f32(1 - v) * 0.5);
      context.sample(trunc(f32(u * xScale)), trunc(f32(v * yScale)),
        bitmap.data, (y * source.width + x) * 4);
    }
    return bitmap;
  }

  function Bitmap_Bricks(xs, ys, color0, color1, mortarColor, mortarX, mortarY,
    tilesX, tilesY, seed, heads, flags, side, colorBalance) {
    const bitmap = newBitmap(xs | 0, ys | 0), random = new Random();
    random.setSeed(seed | 0);
    const c0 = packedColor(color0), c1 = packedColor(color1), mortar = packedColor(mortarColor);
    const fugueX = trunc(f32(f32(mortarX) * 0x2000));
    const fugueY = trunc(f32(f32(mortarY) * 0x2000));
    const sideStep = trunc(f32(f32(side) * 0x4000));
    const multiply = 1 << ((flags >> 4) & 7);
    const baseX = tilesX | 0, baseY = tilesY | 0;
    if (baseX <= 0 || baseY <= 0) return bitmap;
    let cells = Array.from({ length: baseX * baseY }, () => ({ head: 0, color: null }));
    let head = 0;
    for (let y = 0; y < baseY; y++) {
      let trigger = 0;
      for (let x = 0; x < baseX; x++) {
        trigger--;
        const cell = cells[y * baseX + x];
        cell.head = head;
        if (head === 0) head = 1;
        else if (random.int(255) > heads || trigger >= 0) head = 0;
        else if (flags & 4) trigger = 2;
      }
      const first = cells[y * baseX], last = cells[y * baseX + baseX - 1];
      if (!first.head && !last.head) first.head = 1;
      if ((flags & 4) && baseX >= 3) {
        if (cells[y * baseX].head && cells[y * baseX + 1].head && cells[y * baseX + 2].head) cells[y * baseX + 1].head = 0;
        if (last.head && first.head && cells[y * baseX + 1].head) first.head = 0;
        if (cells[y * baseX + baseX - 2].head && last.head && first.head) last.head = 0;
      }
    }
    const expandedX = baseX * multiply, expandedY = baseY * multiply;
    const expanded = Array.from({ length: expandedX * expandedY }, () => ({ head: 0, color: null }));
    for (let yy = 0; yy < multiply; yy++) for (let y = 0; y < baseY; y++) {
      for (let xx = 0; xx < multiply; xx++) for (let x = 0; x < baseX; x++) {
        expanded[((yy * baseY + y) * expandedX) + xx * baseX + x].head = cells[y * baseX + x].head;
      }
    }
    cells = expanded; tilesX = expandedX; tilesY = expandedY;
    // sFPow returns through the x87 stack while sFloatFix has single
    // precision selected; the following multiply is single precision too.
    // Keeping both stores matters for a handful of authored brick colors
    // whose fixed fade lands immediately on opposite sides of an integer.
    colorBalance = f32(colorBalance);
    const randomColorFade = () => trunc(f32(
      f32(Math.pow(random.float(), colorBalance)) * 0x10000,
    ));
    let current = new Uint16Array(4);
    fadePixel(current, 0, c0, 0, c1, 0, randomColorFade());
    for (let y = 0; y < tilesY; y++) {
      for (let x = 0; x < tilesX; x++) {
        const cell = cells[y * tilesX + x];
        if (cell.head) {
          current = new Uint16Array(4);
          fadePixel(current, 0, c0, 0, c1, 0, randomColorFade());
        }
        cell.color = new Uint16Array(current);
      }
      if (!cells[y * tilesX].head) cells[y * tilesX].color = new Uint16Array(cells[y * tilesX + tilesX - 1].color);
    }
    const pixelX = trunc(0x4000 / bitmap.width) * tilesX;
    const pixelY = trunc(0x4000 / bitmap.height) * tilesY;
    for (let y = 0; y < bitmap.height; y++) {
      let fy = trunc(0x4000 / bitmap.height) * y * tilesY;
      const by = fy >> 14; fy &= 0x3fff;
      const sideOffset = (by * sideStep) & 0x7fff;
      for (let x = 0; x < bitmap.width; x++) {
        let fx = trunc(0x4000 / bitmap.width) * x * tilesX + sideOffset;
        let bx = fx >> 14; while (bx >= tilesX) bx -= tilesX; fx &= 0x3fff;
        let fade = 0x4000;
        const leftHead = cells[by * tilesX + bx].head;
        const rightHead = cells[by * tilesX + ((bx + 1) % tilesX)].head;
        if (flags & 8) {
          if (leftHead && fx < fugueX) fade = pixelX ? trunc(0x4000 * Math.max(0, fx - (fugueX - pixelX)) / pixelX) : 0;
          if (rightHead && 0x4000 - fx < fugueX) fade = pixelX ? trunc(0x4000 * Math.max(0, 0x4000 - fx - (fugueX - pixelX)) / pixelX) : 0;
          if (fy < fugueY) fade = Math.min(fade, pixelY ? trunc(0x4000 * Math.max(0, fy - (fugueY - pixelY)) / pixelY) : 0);
          if (0x4000 - fy < fugueY) fade = Math.min(fade, pixelY ? trunc(0x4000 * Math.max(0, 0x4000 - fy - (fugueY - pixelY)) / pixelY) : 0);
        } else {
          if (leftHead && fx < fugueX) fade = fugueX ? trunc(0x4000 * fx / fugueX) : fade;
          if (rightHead && 0x4000 - fx < fugueX) fade = fugueX ? trunc(0x4000 * (0x4000 - fx) / fugueX) : fade;
          if (fy < fugueY) fade = Math.min(fade, fugueY ? trunc(0x4000 * fy / fugueY) : fade);
          if (0x4000 - fy < fugueY) fade = Math.min(fade, fugueY ? trunc(0x4000 * (0x4000 - fy) / fugueY) : fade);
        }
        fadePixel(bitmap.data, (y * bitmap.width + x) * 4, mortar, 0,
          cells[by * tilesX + bx].color, 0, fade * 4);
      }
    }
    return bitmap;
  }

  let fontAdapter = null;

  // This is the only place in the bitmap module that knows about Canvas.
  // Tests and non-window runtimes inject a deterministic adapter instead.
  function canvasFontAdapter(request) {
    const width4 = request.width * 4, height4 = request.height * 4;
    let canvas;
    if (typeof OffscreenCanvas !== 'undefined') canvas = new OffscreenCanvas(width4, height4);
    else if (typeof document !== 'undefined') { canvas = document.createElement('canvas'); canvas.width = width4; canvas.height = height4; }
    else throw new Error('Bitmap_Text needs a font adapter outside a canvas-capable browser');
    const context = canvas.getContext('2d', { willReadFrequently: true });
    context.clearRect(0, 0, width4, height4);
    context.fillStyle = '#fff'; context.textBaseline = 'top';
    // Bitmap_Text calls FontBegin(pageX,pageY,font,pageX*height,
    // pageY*width,...). FontBegin's final dimensions are X then Y: the
    // `height` controls explicit glyph width, while `width` controls the
    // actual line/glyph height. Treating the former as Canvas' font size
    // reduced the production labels to roughly half their authored height.
    // FontBegin takes integer X/Y sizes.  The authored float expressions are
    // therefore truncated before CreateFont sees them, as are every later GDI
    // text coordinate; GDI reports TEXTMETRIC and measured extents as integer
    // pixels. Keeping fractional Canvas sizes or origins moves centered labels
    // by subpixels and changes their 4x
    // downsampled coverage even when the host supplies the same Arial face.
    const pixelSize = Math.max(1, Math.trunc(request.textWidth * height4));
    const requestedGlyphWidth = Math.trunc(request.textHeight * width4);
    // FontBegin substitutes Arial for an empty name before CreateFont.
    context.font = `${pixelSize}px ${request.font || 'Arial'}`;
    let horizontalScale = 1;
    if (requestedGlyphWidth > 0) {
      const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz';
      const averageWidth = context.measureText(alphabet).width / alphabet.length;
      if (averageWidth > 0) horizontalScale = requestedGlyphWidth / averageWidth;
    }
    const lines = String(request.text || '').split('\n');
    // CreateFont receives a negative character height, but FontBegin returns
    // TEXTMETRIC.tmHeight (character height plus internal leading). Canvas'
    // font bounding box is the direct metric when exposed. Arial's released
    // Windows metrics are 2288 units high for a 2048-unit em, which keeps the
    // authored UV strips aligned on implementations without those fields.
    const textMetrics = context.measureText('Mg');
    const metricHeight = Number(textMetrics.fontBoundingBoxAscent) +
      Number(textMetrics.fontBoundingBoxDescent);
    const nativeLineHeight = Math.max(1, Math.round(
      Number.isFinite(metricHeight) && metricHeight > 0
        ? metricHeight : pixelSize * (2288 / 2048),
    ));
    const lineAdvance = Math.trunc(nativeLineHeight * request.lineSkip);
    // Native adds two output pixels plus authored external spacing before
    // positioning text in the 4x antialias surface.
    const external = (Number(request.externalSpace) || 0) * 4 + 8;
    let y = Math.trunc(request.y * height4 + external);
    if (request.flags & 2) y -= Math.trunc(nativeLineHeight * lines.length / 2);
    for (const line of lines) {
      const measuredWidth = Math.round(context.measureText(line).width * horizontalScale);
      let x = Math.trunc(request.x * width4 + external);
      if (request.flags & 1) x -= Math.trunc(measuredWidth / 2);
      context.save();
      context.translate(x, 0);
      context.scale(horizontalScale, 1);
      context.fillText(line, 0, y);
      context.restore();
      y += lineAdvance;
    }
    const rgba = context.getImageData(0, 0, width4, height4).data;
    const coverage = new Uint8Array(request.width * request.height);
    for (let py = 0; py < request.height; py++) for (let px = 0; px < request.width; px++) {
      let sum = 0;
      for (let ay = 0; ay < 4; ay++) for (let ax = 0; ax < 4; ax++) {
        sum += rgba[((py * 4 + ay) * width4 + px * 4 + ax) * 4 + 3];
      }
      coverage[py * request.width + px] = sum >>> 4;
    }
    return { coverage, lineHeight: nativeLineHeight };
  }

  function setBitmapFontAdapter(adapter) { fontAdapter = adapter; }

  function readU16LE(bytes, offset) { return bytes[offset] | (bytes[offset + 1] << 8); }
  function readI16LE(bytes, offset) { const value = readU16LE(bytes, offset); return value & 0x8000 ? value - 0x10000 : value; }

  function Bitmap_Text(op, environment, input, x, y, width, height, color, flags,
    externalSpace, internalSpace, lineSkip, text, font, writable = null) {
    const source = requireBitmap(input);
    const bitmap = writable === RUNTIME_IN_PLACE ? source : blankBitmapCopy(source);
    const col = packedColor(color);
    const blob = op?.blob;
    const page = (flags & 0x70) >> 4;
    let coverage = null, metrics = null, lineHeight = 0;
    if ((flags & 0x80) && blob && blob.length >= 8 && blob[0] === 3 &&
      blob[1] === (page ? 1 : 0) && readU16LE(blob, 2) === bitmap.width && readU16LE(blob, 4) === bitmap.height) {
      lineHeight = readU16LE(blob, 6);
      let offset = 8;
      if (page) {
        metrics = new Array(256);
        for (let i = 0; i < 256; i++, offset += 10) metrics[i] = {
          x: readU16LE(blob, offset), y: readU16LE(blob, offset + 2),
          before: readI16LE(blob, offset + 4), width: readI16LE(blob, offset + 6), after: readI16LE(blob, offset + 8),
        };
      }
      coverage = new Uint8Array(bitmap.size);
      for (let i = 0; i < coverage.length && offset + i < blob.length; i++) {
        const five = blob[offset + i]; coverage[i] = (five << 5) | (five << 2) | (five >> 1);
      }
    } else {
      const adapter = fontAdapter || canvasFontAdapter;
      const result = adapter({
        width: bitmap.width, height: bitmap.height, x, y,
        textWidth: width, textHeight: height, color: color >>> 0, flags: flags >>> 0,
        externalSpace, internalSpace, lineSkip, text: text || '', font: font || '', page,
      });
      if (!result || result.coverage?.length !== bitmap.size) throw new Error('font adapter returned invalid coverage');
      coverage = result.coverage; metrics = result.metrics || null; lineHeight = result.lineHeight || 0;
    }
    for (let i = 0; i < bitmap.size; i++) fadePixel(bitmap.data, i * 4, source.data, i * 4, col, 0, coverage[i] | (coverage[i] << 8));
    if (page && environment) {
      environment.letters ||= [];
      if (metrics) {
        const aliasWidth = bitmap.width * 4, aliasHeight = bitmap.height * 4;
        const inside = internalSpace * aliasWidth;
        environment.letters[page] = metrics.map(metric => ({
          uv: [metric.x / aliasWidth, metric.y / aliasHeight,
            (metric.x + metric.width + inside * 2) / aliasWidth,
            (metric.y + lineHeight + inside * 2) / aliasHeight],
          preSpace: (metric.before - inside) / aliasWidth,
          width: (metric.before + metric.width + metric.after + inside * 2) / aliasWidth,
        }));
      }
    }
    return bitmap;
  }

  let importAdapter = null;
  function setBitmapImportAdapter(adapter) { importAdapter = adapter; }

  function decodeBase64(value) {
    if (typeof atob === 'function') {
      const binary = atob(value), bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      return bytes;
    }
    if (typeof Buffer !== 'undefined') return new Uint8Array(Buffer.from(value, 'base64'));
    throw new Error('No base64 decoder available');
  }

  // The released debris KX has exactly one Import: fvs32.jpg, a 661-byte
  // baseline JPEG whose decoded lower-power image is 16x16.  Keeping this
  // decoded oracle avoids making graph initialization asynchronous merely to
  // pass through createImageBitmap. Other imports use the injectable decoder.
  const DEBRIS_IMPORT_RGBA =
    'hE6w/7GO4v++wPn/ssfy/7LH9P+yx/T/ssf0/7LH9P+yx/T/ssf0/7TG9P+0xvT/ssf0/8C/+P+3jNv/jEul/7aN3P/Otv//k57e/4ek3v+IpeH/h6Tg/4ek4P+HpOD/hqPd/4mj3P+Ko9v/iqPb/4mj3v+Und7/4cX//7qM1//Sxu7/m5zS/0Ngrv83Yr7/NmO+/zZjvv83Yr7/N2O8/zljuf86Y7P/PmKs/z5isP86Yrr/QWCy/5ic2f/Gvu//y8zh/5Shzf88Yrf/MGPO/y9kzv8vZM7/MGPM/zJkxf82ZLr/NF6y/9j///83YsD/M2LK/zlhuf+QodX/xM3s/8vL4/+Uoc3/PWO4/zBjzv8wY87/MGPM/zJjyf82ZLr/O16g/9////83Y7z/M2PJ/zJjzP89Yrr/k6HQ/8jM5//Ly+P/laDN/z5hs/8zY8f/M2PH/zZiwf85Y7X/OF+k/+P///9BY6L/OWK6/zNjx/8zY8f/PWKz/5Wgzf/Ly+X/y8vj/5egyf9DYan/OmK6/zpiuf8+Yqz/Q16N/+n///9DY6D/PmKs/zxitf86Yrr/OWO5/0Fiqf+XoMn/ysrk/8zM5P+YoMT/SmCZ/+L///9DYqX/5v///wACI//t////RGKe/9////9DYaf/4P///93///9EYaP/l6DH/8vL5f/Ly+P/m6C9/wAAJf/u////AAEp/05hif8AASz/SWGR/wACOf9HYpn/AAEw/wABM//g////RWCj/5egx//Ly+X/y8vj/5ygu/8AACD/UmCD//D////t////S2GS/+f///9LYZL/7f///1Jhgv/t////QWGq/0Nhqf+XoMn/y8vl/8vL4/+boL7/UGKK/wABMP/p////SmKS/wABNf9KYpL/AAEn/1Fgf/8AAR7/6////zxitf89YLL/laDN/8vL5f/Ly+P/mKDE/0pgmf8AAkL/R2Ga/+n///9LYIv/7v////H////x////7f///0VhoP83Yr7/PGG5/5Kg0f/IzOn/y8vj/5Why/9AYa7/OmO1/wACQv9MYYz/9f////j////0////6////0Biqv83Yr7/M2PJ/zpiuv+QodX/xM3u/9LG7P+bnc7/QWGq/zpksP9EYp7/AAEr/wABFf8AABD/AAEb/0thkP8+Yq7/N2O8/zZkuv89Yav/lJ/V/8LB6/+0kND/49H//6u/8f+Fp9T/iqXU/46jzv+Ro8f/k6PF/5Gjyf+Oo9D/iqPb/4al3P+Dp9f/qcHv/93W//+Vfrb/glKe/5h7u/+xv+L/rczo/7HI8f+yx/T/s8j1/7LH9P+yx/T/ssf0/7LH9P+wyfH/rczo/67B4f+RgLb/eleX/w==';

  function bitmapFromRGBA(width, height, rgba) {
    const bitmap = new Bitmap(width, height);
    for (let i = 0; i < bitmap.size; i++) {
      const color = rgbaToPacked(rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2], rgba[i * 4 + 3]);
      bitmap.data.set(packedColor(color), i * 4);
    }
    return bitmap;
  }

  function Bitmap_Import(op, filename) {
    const blob = op?.blob;
    if (filename === 'fvs32.jpg' && blob?.length === 661 && hashBytes(blob) === 0x67c4d515) {
      return bitmapFromRGBA(16, 16, decodeBase64(DEBRIS_IMPORT_RGBA));
    }
    const adapter = importAdapter;
    if (!adapter) return null;
    const decoded = adapter(blob, filename, op);
    if (!decoded || !decoded.width || !decoded.height || !decoded.data) return null;
    // LoadBitmapCore crops to the next lower power of two without resampling.
    const width = 1 << powerOfTwo(decoded.width), height = 1 << powerOfTwo(decoded.height);
    if (decoded.data instanceof Uint16Array) {
      const result = new Bitmap(width, height);
      for (let y = 0; y < height; y++) result.data.set(decoded.data.subarray(y * decoded.width * 4, y * decoded.width * 4 + width * 4), y * width * 4);
      return result;
    }
    const cropped = new Uint8Array(width * height * 4);
    for (let y = 0; y < height; y++) cropped.set(decoded.data.subarray(y * decoded.width * 4, y * decoded.width * 4 + width * 4), y * width * 4);
    return bitmapFromRGBA(width, height, cropped);
  }

  function Bitmap_Render(record, input, xs, ys) {
    const hook = record.runtime?.bitmapRendererHook || bitmapRendererHook;
    if (hook) return hook({ ...record, input, widthExponent: xs, heightExponent: ys });
    const bitmap = newBitmap(xs | 0, ys | 0);
    bitmap.deferredRender = { input, op: record.op, widthExponent: xs | 0, heightExponent: ys | 0 };
    return bitmap;
  }

  // Exec_Bitmap_Render is deliberately empty in the released player. Render
  // is marked dynamic in the class table, though, so strict runtime dispatch
  // still needs an explicit exec handler if one of these operators is reached
  // through a dynamic input path.
  function Exec_Bitmap_Render() {}

  const runtimeConsumerCounts = new WeakMap();
  const runtimeConsumptionState = new WeakMap();

  function consumerCounts(runtime) {
    if (!runtime || !Array.isArray(runtime.operations)) return null;
    let counts = runtimeConsumerCounts.get(runtime);
    if (counts) return counts;
    counts = new Map();
    const add = op => {
      if (op != null) counts.set(op, (counts.get(op) || 0) + 1);
    };
    for (const op of runtime.operations) {
      for (const input of op.inputs || []) add(input);
      for (const link of op.links || []) add(link);
    }
    for (const root of runtime.roots || []) add(root);
    for (const event of runtime.events || []) add(event.op);
    runtimeConsumerCounts.set(runtime, counts);
    return counts;
  }

  function bitmapInputIsUniquelyConsumed(call, inputIndex) {
    const inputOp = call?.op?.inputs?.[inputIndex];
    const counts = consumerCounts(call?.runtime);
    if (inputOp == null || !counts || !(call.inputs?.[inputIndex] instanceof Bitmap)) return false;
    let state = runtimeConsumptionState.get(call.runtime);
    if (!state) {
      state = { remaining: new Map(counts), claimed: new WeakMap() };
      runtimeConsumptionState.set(call.runtime, state);
    }
    let claimed = state.claimed.get(call.op);
    if (!claimed) state.claimed.set(call.op, claimed = new Set());
    // Production bitmap precalc is an immutable, one-shot graph walk, which is
    // the same lifetime assumption as the existing post-result adoption. A
    // handler can nevertheless be invoked again by incremental tooling; its
    // old result may already alias this producer, so repeated calls
    // conservatively clone and never mutate that input a second time.
    if (claimed.has(inputIndex)) return false;
    claimed.add(inputIndex);
    state.remaining.set(inputOp, Math.max(0, (state.remaining.get(inputOp) || 0) - 1));
    return state.remaining.get(inputOp) === 0;
  }

  function claimBitmapCallInputs(call) {
    const owned = [];
    for (let index = 0; index < (call.inputs?.length || 0); index++) {
      const input = call.inputs[index];
      if (input instanceof Bitmap && bitmapInputIsUniquelyConsumed(call, index)) owned.push(input);
    }
    return [...new Set(owned)];
  }

  function bitmapInputCanBeReused(call, inputIndex, owned) {
    const candidate = call.inputs?.[inputIndex];
    if (!(candidate instanceof Bitmap) || !owned.includes(candidate)) return false;
    for (let index = 0; index < (call.inputs?.length || 0); index++) {
      if (index === inputIndex) continue;
      const other = call.inputs[index];
      // Distinct wrappers can still expose the same typed-array storage after
      // an earlier adoption. Treat either identity or backing-buffer aliasing
      // as a duplicate so an input read later in the operator stays immutable.
      if (other instanceof Bitmap &&
          (other === candidate || other.data === candidate.data ||
           other.data.buffer === candidate.data.buffer)) return false;
    }
    return !publiclyAliasedBitmaps.has(candidate);
  }

  function finishBitmapCall(call, result, preclaimedOwned = null) {
    const uniqueOwned = preclaimedOwned || claimBitmapCallInputs(call);
    if (!(result instanceof Bitmap)) {
      for (const input of uniqueOwned) input.releaseStorage();
      return result;
    }
    if (uniqueOwned.includes(result)) {
      for (const input of uniqueOwned) if (input !== result) input.releaseStorage();
      return result;
    }
    const adoption = uniqueOwned[0];
    for (let index = 1; index < uniqueOwned.length; index++) uniqueOwned[index].releaseStorage();
    return adoption ? transferBitmapStorage(adoption, result) : result;
  }

  function finishOwnedBitmapFormat(call) {
    const input = call.inputs?.[0];
    // Format changes metadata only. When this is the producer's final graph
    // consumer, native CheckBitmap keeps its storage in place; adopting a
    // freshly copied result afterward was needlessly copying an entire
    // terminal texture first. Claim through the same conservative ownership
    // tracker used by finishBitmapCall, including its incremental-recalc guard.
    if (input instanceof Bitmap && bitmapInputIsUniquelyConsumed(call, 0)) {
      const parameters = call.parameters || [];
      input.format = input.Format = parameters[0] | 0;
      input.texMipCount = input.TexMipCount = parameters[1] | 0;
      input.texMipThreshold = input.TexMipTresh = parameters[2] | 0;
      return input;
    }
    return finishBitmapCall(call, Bitmap_Format(
      input, call.parameters?.[0], call.parameters?.[1], call.parameters?.[2],
    ));
  }

  const handlers = {
    0x21: ({ parameters: p }) => Bitmap_Flat(p[0], p[1], p[2]),
    0x22: ({ parameters: p }) => Bitmap_Perlin(...p),
    0x23: ({ inputs, parameters: p }, writable) => Bitmap_Color(inputs[0], p[0], p[1], writable),
    0x24: ({ inputs, parameters: p }, writable) => Bitmap_Merge(p[0], inputs, writable),
    0x25: ({ inputs, parameters: p }) => Bitmap_Format(inputs[0], p[0], p[1], p[2]),
    0x27: ({ inputs, parameters: p }, writable) => Bitmap_GlowRect(inputs[0], ...p, writable),
    0x29: ({ inputs, parameters: p }, writable) =>
      Bitmap_Blur(inputs[0], p[0], p[1], p[2], p[3], writable),
    0x2a: ({ inputs, parameters: p }, writable) =>
      Bitmap_Mask(inputs[0], inputs[1], inputs[2], p[0], writable),
    0x2b: ({ inputs, parameters: p }, writable) => Bitmap_HSCB(inputs[0], ...p, writable),
    0x2c: ({ inputs, parameters: p }) => Bitmap_Rotate(inputs[0], ...p),
    0x2d: ({ inputs, parameters: p }, writable) =>
      Bitmap_Distort(inputs[0], inputs[1], p[0], p[1], writable),
    0x2e: ({ inputs, parameters: p }) => Bitmap_Normals(inputs[0], p[0], p[1]),
    0x30: ({ inputs, parameters: p }, writable) =>
      Bitmap_Bump(inputs[0], inputs[1], ...p, writable),
    0x31: ({ op, environment, inputs, parameters: p, strings }, writable) =>
      Bitmap_Text(op, environment, inputs[0], ...p, strings[0], strings[1], writable),
    0x32: ({ parameters: p }) => Bitmap_Cell(...p),
    0x34: ({ parameters: p }) => Bitmap_Gradient(...p),
    0x35: ({ inputs, parameters: p }, writable) => Bitmap_Range(inputs[0], ...p, writable),
    0x36: ({ inputs, parameters: p }, writable) => Bitmap_RotateMul(inputs[0], ...p, writable),
    0x38: ({ inputs, parameters: p }) => Bitmap_Sharpen(inputs[0], ...p),
    0x39: ({ inputs, parameters: p }, writable) => Bitmap_GlowRect(inputs[0], ...p, writable),
    0x3a: ({ op, strings }) => Bitmap_Import(op, strings[0]),
    0x3b: ({ inputs, parameters: p }, writable) => bitmapColorBalance(inputs[0], p, writable),
    0x3d: ({ parameters: p }) => Bitmap_Bricks(...p),
    0x3e: ({ inputs, parameters: p }) => Bitmap_Bulge(inputs[0], p[0]),
    0x3f: {
      init: record => Bitmap_Render(record, record.inputs[0], record.parameters[0], record.parameters[1]),
      exec: Exec_Bitmap_Render,
    },
  };

  const writableBitmapInputIndexes = new Map([
    [0x23, 0], [0x24, 0], [0x27, 0], [0x29, 0], [0x2a, 0], [0x2b, 0],
    [0x2d, 1], [0x30, 0], [0x31, 0], [0x35, 0], [0x36, 0], [0x39, 0],
    [0x3b, 0],
  ]);

  const bitmapHandlers = {};
  for (const [id, handler] of Object.entries(handlers)) {
    // Bitmap_Render's deferred record (and renderer hooks) may retain the
    // exact input object. Never adopt/release a Bitmap input across this seam;
    // the production convention normally supplies IPP, but the lifetime rule
    // must remain valid for hooks and adapters too.
    const numericId = Number(id);
    bitmapHandlers[id] = numericId === 0x3f
      ? handler
      : numericId === 0x25
        ? finishOwnedBitmapFormat
        : writableBitmapInputIndexes.has(numericId)
          ? call => {
            const owned = claimBitmapCallInputs(call);
            const writableIndex = writableBitmapInputIndexes.get(numericId);
            const writable = bitmapInputCanBeReused(call, writableIndex, owned)
              ? RUNTIME_IN_PLACE : null;
            return finishBitmapCall(call, handler(call, writable), owned);
          }
          : call => finishBitmapCall(call, handler(call));
  }

  export {
    BI,
    BilinearContext,
    Bitmap,
    Bitmap_Blur,
    Bitmap_Bricks,
    Bitmap_Bulge,
    Bitmap_Bump,
    Bitmap_Cell,
    Bitmap_Color,
    Bitmap_ColorBalance,
    Bitmap_Distort,
    Bitmap_Flat,
    Bitmap_Format,
    Bitmap_GlowRect,
    Bitmap_Gradient,
    Bitmap_HSCB,
    Bitmap_Import,
    Bitmap_Mask,
    Bitmap_Merge,
    Bitmap_Normals,
    Bitmap_Perlin,
    Bitmap_Range,
    Bitmap_Render,
    Bitmap_Rotate,
    Bitmap_RotateMul,
    Bitmap_Sharpen,
    Bitmap_Text,
    Bitmap_Unwrap,
    addScalePixel,
    bilinearFilter,
    bitmapInner,
    bitmapFromRGBA,
    canvasFontAdapter,
    clamp15,
    fadePixel,
    newBitmap,
    noise2,
    packedColor,
    pointFilter,
    rgbaToPacked,
    sampleUV,
    setBitmapFontAdapter,
    bitmapInputIsUniquelyConsumed,
    setBitmapImportAdapter,
    bitmapDefaultFormat,
    bitmapTextureSizeOffset,
    bitmapRendererHook,
    setBitmapDefaultFormat,
    setBitmapTextureSizeOffset,
    setBitmapRendererHook,
    bitmapHandlers,
  };
