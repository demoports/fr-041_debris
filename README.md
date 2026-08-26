# fr-041: debris. (WebGL port)

A plain-JavaScript, WebGL2, and Web Audio port of Farbrausch's **fr-041:
debris.**, released at Breakpoint 2007. The browser evaluates the original
compact werkkzeug3 production, generates its textures and geometry, executes
its timeline, and synthesizes both embedded V2M soundtracks.

This is a source-level browser reimplementation, not an emulator. It does not
execute the Windows program and contains no WebAssembly, framework, bundler, or
pregenerated production resources.

Original credits: **fiver2** — visuals and direction; **chaos / ryg** — code;
**ronny** — soundtrack; **kb** — synthesizer; **wayfinder** — sound effects;
**giZMo / tron / fried**.

## Port design

| Area | Browser implementation |
|---|---|
| Compact production | Bounds-checked parser for the exact party KX, including its operator graph, animation bytecode, events, splines, samples, and main song |
| Runtime | Operator evaluation, animation VM, Buzz-compatible timing, event execution, instances, snapshots, and stateful seeking |
| Content generation | The GenBitmap, GenMesh, GenMinMesh, material, scene, overlay, and Debris effect operations used by the production |
| Rendering | WebGL2 textures, geometry, materials, lighting, shadows, instancing, viewports, water, ribbons, and post-processing |
| Sound | Direct-JavaScript V2 synthesis, loader and main-song playback, output-rate conversion, pause, and checkpointed seeking |

The code is a native ES Module graph with named imports and exports. Its module
boundaries follow the released werkkzeug3 source where those boundaries remain
meaningful without C++ headers, Direct3D, or native reference counting.
[`notes/ES_MODULE_MAP.md`](notes/ES_MODULE_MAP.md) records the correspondence.

Textures and geometry are generated locally from the compact graph at their
authored dimensions. Intermediate caches are discarded or transferred when
their final consumer permits it; no generated texture or geometry cache is
downloaded. The production's 2:1 viewport is preserved, with letterboxing or
pillarboxing when the display has a different aspect ratio.

The Direct3D 9 renderer is translated into explicit WebGL2 material passes and
resource ownership. The port preserves the original clockwise front-face
convention, material culling, generated lookup textures, quality-0 DXT5 path,
lighting, shadow volumes, and glare/color-correction graph.

V2 sequencing and DSP run at the production's fixed 44.1 kHz rate. The audible
production sample is the visual master clock. During precalculation, a module
Worker synthesizes the forward-only loader tune and feeds an AudioWorklet
through a bounded transferable queue, allowing texture and geometry generation
to continue without blocking audio delivery. The loader is disposed before the
seekable main-song synthesizer is created.

Font3D geometry uses deterministic metric-compatible Arimo and Gelasio subsets
and a plain-JavaScript GLU tessellator port. This avoids host-font rasterization
differences while keeping the geometry procedural.

## Repository structure

| Path | Purpose |
|---|---|
| `index.html` | Launcher and sole browser entry point |
| `assets/` | Exact compact production, loader V2M, attribution, and launcher image |
| `src/` | Parser, runtime, generators, renderer, effects, audio, and application lifecycle |
| `notes/` | Reconstruction record, source mapping, conventions, and generated production inventories |
| `tools/` | Extraction, generation, deterministic tests, native oracles, and guarded browser validation |
| `vendor/` | Pinned reference source and license material |

## Validation

The low-memory subsystem suite runs each Node process with an independent heap
limit:

```sh
for test in tools/test_*.mjs; do node --max-old-space-size=256 "$test"; done
python3 tools/test_kx.py
node --max-old-space-size=192 tools/audit_font3d_canvas.mjs
```

The tests cover compact-data parsing, every production operator class, bitmap
and geometry generators, material and renderer state, camera and event timing,
audio synthesis and seeking, Worker/AudioWorklet ownership, module boundaries,
and deterministic native-oracle hashes.

Full browser precalculation uses substantial memory. `tools/browser_smoke.mjs`
serializes Chrome runs, monitors aggregate process-tree RSS, applies bounded
render dimensions and V8 limits, and terminates the browser at its safety
ceiling. Production browser probes must be run sequentially, with only one
precalculation active at a time.

The detailed evidence and reproducibility data live in:

- [`notes/RECONSTRUCTION.md`](notes/RECONSTRUCTION.md) — executable unpacking,
  exact payloads, hashes, source discrepancies, audio oracles, and timing.
- [`notes/PORT_CONVENTIONS.md`](notes/PORT_CONVENTIONS.md) — numeric,
  coordinate, texture, matrix, and rendering conventions.
- [`notes/ES_MODULE_MAP.md`](notes/ES_MODULE_MAP.md) — released C/C++ ownership
  mapped to browser modules.
- [`notes/debris_class_map.json`](notes/debris_class_map.json) — every compact
  class mapped to its released handler and pinned source location.

## Fidelity boundaries

- Format-9 textures use a plain-JavaScript port of the released quality-0 DXT5
  encoder. Browsers with S3TC upload the BC3 chain directly; other browsers
  decode the same blocks to RGBA8 before upload.
- Font3D is deterministic, but the open Arimo/Gelasio outlines are substitutes
  for proprietary XP-era Arial/Georgia glyph points. The separate bitmap-text
  path still obtains its fonts from browser Canvas and can vary by platform.
- Both embedded V2M songs follow the pinned V2Redux sequencer and DSP. If a
  browser cannot create a 44.1 kHz AudioContext, only the final stereo stream is
  resampled; production timing remains at 44.1 kHz.
- Direct3D 9 shader and fixed-function behavior is expressed in WebGL2 GLSL.
  Filtering, floating-point precision, and shared-edge rasterization can vary
  across browsers, drivers, and ANGLE backends.

The fidelity target is the released production's data and source behavior, not
execution of its x86 code or universal pixel identity across graphics drivers.

## Provenance and license

The exact party KX and loader tune were recovered from the supplied
`fr-041_debris.exe` and checked against Farbrausch's later source release. The
engine reference is [`farbrausch/fr_public`](https://github.com/farbrausch/fr_public)
tag `original`, commit `3e333cf720be7f3cdc311bbe98939a70e7d60edb`.

Farbrausch's werkkzeug3 source is BSD-licensed; its notice is retained in
[`vendor/wz3/LICENSE.txt`](vendor/wz3/LICENSE.txt). The Debris project and music
are CC BY 3.0; see [`assets/ATTRIBUTION.md`](assets/ATTRIBUTION.md). The V2Redux
reference is public domain, `libtess.js` retains the SGI Free Software License B
2.0, and the deterministic font subsets retain the SIL Open Font License 1.1.
Their notices and pinned-source records are kept under `vendor/`.

This port was made with Codex 5.6 Sol Ultra.
