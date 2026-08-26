# Native source to ES Module map

The browser graph follows the released werkkzeug3 player boundaries where they
remain meaningful without C++ headers, Direct3D, or reference counting. The
composition authority is `vendor/wz3/player_demo/demo_oplist.cpp`; its
`KHandlers` table becomes the checked `Map` built by `src/operators.js`.

| Released source owner | Browser module | Port boundary |
|---|---|---|
| `kdoc.cpp`, `genblobspline.cpp` | `src/runtime.js` | KOp execution, events, animation VM, runtime splines, and BlobSpline evaluation |
| `genbitmap.cpp` | `src/bitmap.js` | Procedural bitmap objects and all 25 used bitmap handlers |
| `genmaterial.cpp`, `materials/material11.*`, `material20.*` | `src/material.js`, `src/renderer.js` | Material operator records are separate from their WebGL pass translation |
| `genmesh.cpp` | `src/mesh.js` | Mutable old-GenMesh topology and its 21 directly used operators |
| `genmesh.cpp::Mesh_ToMin` | `src/mesh_to_min.js` | Explicit old-GenMesh to GenMinMesh ownership bridge (`0xb5`) |
| `genminmesh.cpp` | `src/minmesh.js` | Compact topology, Font3D, animation, preparation, and 27 handlers |
| `genscene.cpp` | `src/scene.js` | Scene records, multiply/transform/light/particle execution, and frame jobs |
| `geneffectdebris.cpp`, `geneffectipp.cpp` | `src/effects.js`, `src/overlay.js` | Geometry/post effects and the IPP graph nodes consumed by the renderer |
| `engine.cpp`, Direct3D material code | `src/gl.js`, `src/renderer.js` | WebGL resources, material passes, lighting, instancing, shadows, and IPP execution |
| `_start.cpp::UpdateTexture`, `_rygdxt.cpp` | `src/gl.js`, `src/dxt5.js` | Source-word mip construction, signed Q8W8V8U8 upload, and exact quality-0 BC3 encoding/decoding |
| `_start.cpp::MakeCubeNormalizer`, `_start.cpp::MakeAttenuationVolume` | `src/legacy_lookup.js`, `src/renderer.js` | Byte-exact runtime Material11 lookup generation is isolated from WebGL cube/volume ownership |
| `_types.*` | `src/core.js` | Float32 math, matrices, RNG, and matrix-stack primitives |
| compact K reader and class table | `src/abi.js`, `src/classes.js`, `src/kx.js`, `src/data.js` | Serialized constants, generated class metadata, bounds-checked parsing, and embedded payload loading |
| V2 player/synth | `src/v2.js`, `src/audio_core.js`, `src/audio.js` | Deterministic DSP is separated from Web Audio scheduling |
| loader thread / sound output | `src/audio_worker_core.js`, `src/audio_worker.js`, `src/audio_worklet_core.js`, `src/audio_worklet.js` | Testable protocol cores plus static module Worker/AudioWorklet entries |
| player lifecycle | `src/app.js` | Precalc, cache ownership, timeline clock, rendering, seeking, and disposal |

There are two intentional aggregations. `runtime.js` keeps KOp control flow and
the spline objects it executes together, while `renderer.js` keeps the native
engine/material/IPP draw-state translation together because WebGL pass state is
one resource owner. Conversely, `Mesh_ToMin` is split out: placing it in either
geometry family would create an import cycle that does not exist in the native
linker table. The two generated Material11 lookup tables and the DXT5 codec are
also split into pure modules so their released C algorithms can be checked by
byte/hash oracles without constructing a renderer or production graph.

Every module uses named imports/exports and has no registration side effect.
`src/operators.js` is the only composition point; it currently owns 101 unique
class IDs: all 90 IDs in the embedded production plus 11 released-source
generic operators. It rejects duplicate ownership before a `Runtime` is
constructed.
