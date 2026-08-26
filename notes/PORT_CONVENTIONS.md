# Debris JavaScript port conventions

## Runtime boundary

The shipping player is plain browser JavaScript, WebGL2, and WebAudio. It does
not execute the PE image and does not load WebAssembly. The
released C++ and exact unpacked executable are reference oracles only.

All browser files are native ES Modules with explicit named dependencies:

```js
import { Random } from './core.js';

function publicName() {
  return new Random(0);
}

export { publicName };
```

The page must be served over HTTP; `file://` is not a supported runtime
boundary for the module Worker, AudioWorklet, or embedded asset fetches. Source
modules must not create a `globalThis.Debris` compatibility namespace.

## Numeric rules

- Keep serialized integers unsigned with `>>> 0` where they represent colors,
  masks, IDs, or packed words; use `| 0` for signed 32-bit operations.
- Use `Math.imul` for every original 32-bit multiply.
- Store persistent C++ `sF32` fields in `Float32Array` or pass them through
  `Math.fround`. JavaScript doubles may be used for temporary x87 expressions,
  but preserve source operation order.
- Matrices are 16-value `Float32Array`s in original `sMatrix` order:
  `i, j, k, l`, four consecutive vectors each. This is column-major for
  WebGL. Vectors are arrays or typed-array views in `x,y,z,w` order.
- Procedural bitmaps store interleaved 16-bit `RGBA` channels in a
  `Uint16Array`, four entries per pixel. This is the source `sU64` pixel format
  without relying on slow JavaScript `BigInt` arithmetic.
- Preserve the original RNG seed and update order. Never use `Math.random` in
  reconstructed code.

## Graph and handlers

`parseKX` preserves every raw packed field. Runtime dispatch is keyed by the
exported class ID, never by the document-local class index. Each operator owns:

- `parameters`: raw unpacked values, including the ignored Multiply padding;
- `animParameters`: only the class convention's `dataWords` values;
- resolved input and link operator references;
- a precalculated `cache` object;
- dynamic instance state used by events/effects.

Each native source-family module exports a side-effect-free handler table.
`src/operators.js` composes those tables in released `KHandlers` order and
throws on duplicate class IDs; `Runtime` receives the resulting `Map` through
its constructor. `Mesh_ToMin` lives in `src/mesh_to_min.js`, the explicit
GenMesh/GenMinMesh boundary, so the two geometry families do not import one
another. An init handler receives one call record:

```js
{
  runtime, environment, op,
  inputs, links, parameters, strings, splines
}
```

The arrays are already in the exact `KOp::Call` order. A dynamic `exec`
handler receives the same shape with current animated parameters. Handler
names and exact source locations are in `notes/debris_js_dispatch.json`.

The `0x95` and `0x12f` packing strings contain 18 fields despite a 17-word
convention. Parse all 18 fields to maintain stream alignment but expose only
words 0–16 to handlers; word 17 is an ignored historical alias artifact.

The KX reader treats the embedded bytes as untrusted structure even though the
shipping payload is pinned: caller offsets and padded 32-bit lengths must stay
inside the supplied view, operator/event spline references must stay inside the
declared table, and unterminated or oversized packing strings are rejected. A
packing string may contain at most 256 fields (the eight-bit native data-word
domain plus the one released Debris overrun), and the decoded parameter-slot
table is capped at both four times the input byte length and 4,194,304 slots.
These checks happen before per-operation parameter arrays are allocated; they
do not alter any field in the exact party export.

## Object shapes

Use readable JavaScript classes (`Bitmap`, `Mesh`, `MinMesh`, `Material`,
`Scene`, `Spline`) and retain the original field names in comments where they
differ. Avoid emulating pointers or reference counts. Copy-on-write operators
must preserve their input when called directly. During the one-shot immutable
bitmap graph walk, the handler layer may instead transfer a verified
final-consumed input and mutate its storage, mirroring `CheckBitmap`'s native
reference-count fast path. Shared inputs, wrappers aliased through public
`Bitmap.adopt()`, physical aliases co-present in one call, repeated calls, and
dynamic Bitmap_Render records remain on the conservative copy path.

Large homogeneous buffers use typed arrays. Topology under active mutation may
use arrays of records first, then compile to typed GPU buffers at `prepare()`.
Every precalculated object exposes a deterministic `summary()` suitable for
hashing and differential tests.

## Rendering and time

- Audio sample position is the visual master clock.
- WebGL2 draw state is collected into explicit jobs before rendering, matching
  the original scene/engine split.
- D3D9 state and shader behavior are translated deliberately; do not substitute
  generic Three.js materials.
- Fixed-time debug rendering uses `?t=SECONDS`; `?w`, `?h`, and `?dpr` control
  deterministic render size.
- Seeking backwards restores a runtime snapshot and advances simulation from
  it. Moving only the audio cursor is incorrect for particles and instance
  memory.

## Validation

Validation seams, in order: KX parse; animation/event state; each generated
bitmap; each generated mesh; scene/job stream; intermediate render targets;
final framebuffer; V2 PCM. Tests should report the class ID/operator ID and
first divergent field or pixel, not only a whole-file hash.
