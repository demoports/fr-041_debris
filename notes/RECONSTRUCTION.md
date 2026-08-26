# fr-041: debris — reconstruction notes

## Inputs

The original executable supplied for this port is the authority for runtime
behaviour:

| file | bytes | SHA-256 |
|---|---:|---|
| `fr-041_debris.exe` | 181,248 | `38bb8656c63946ece05680a10a71b660cc47dab09a5d5ad82dd1a4befc2cbeb5` |

It is a 32-bit PE packed into one `kkrunchy` section. The PE image base is
`0x790000`, the entry RVA is `0x2b8d0`, and the image reserves `0x11f0000`
bytes. Its static import table contains only `LoadLibraryA` and
`GetProcAddress`; the real imports are reconstructed by the depacker.

Farbrausch later published the werkkzeug3 code and Debris production data in
[`farbrausch/fr_public`](https://github.com/farbrausch/fr_public). The relevant
snapshot is the annotated tag `original` (tag object
`510a2b3f845d65e2633e4d7880d43e13e7716ae3`, peeled commit
`3e333cf720be7f3cdc311bbe98939a70e7d60edb`). The engine is BSD-licensed; the
files under `werkkzeug3/data/debris` are CC BY 3.0. This source is used to name
and understand recovered structures, but correspondence is checked against the
supplied binary.

The published repository's release executable is also 181,248 bytes but is not
identical (`b4d678bfc0498c7afd35cab8f109b8cae1e39663ad69a0301923ed99e245103d`).
Its entry RVA is `0x2b8bb`. Both executables use the same private `kkrunch_p7`
decompressor family, but the local stub is not byte-for-byte equal to the
checked-in alpha 3 source. In particular it uses nine context models, 40
inputs, no APM, and 1,056 weights. The checked-in source remains an algorithm
map; Unicorn executes the local stub itself to make unpacking authoritative.

## Exact unpacking

`tools/unpack_stage1.py` executes the packed arithmetic decoder until the
stage-two filter. The exact first-stage output is 12,444,766 bytes with SHA-256
`b1ae6cb5435040a134c81d2edaf9547688e2ae13643203b85fadde74b90eb310`.

`tools/unpack_stage2.py` supplies deterministic stubs only for
`LoadLibraryA`/`GetProcAddress`, records 92 dynamic imports from seven DLLs,
undoes the executable-code filters, and stops before the original entrypoint at
`0x83a4a0`. The resulting image begins at the original base `0x800000`, is
`0xba6000` bytes long, and has SHA-256
`ea2994abf417c29b15b4bdaa65b4448beb348a83ec990fd8781c916970d3a343`.

## Exact compact project

The original player passes the compact KX document at virtual address
`0x843684` to `KDoc::Init`. `tools/kx.py` is a bounds-checked transcription of
that released reader, and `tools/extract_kx.py` uses it to determine the end of
the embedded document without a guessed length.

The extracted `assets/debris_party.kx` is 490,077 bytes with SHA-256
`13fdab5e812d6698334434cb8dada652910c14d824a09f0bdd47c606de464377`.
It contains 16,478 operators in 90 exported classes, 95 events, 11 splines,
20,453 bytes of animation bytecode, 157 blobs, the 175,735-byte main V2 song,
and a 6,640-byte sample bank. It enables the original Buzz timing correction.
The complete section and class inventory is generated in
`notes/debris_party_kx.json`.

The released `debris.kx` is six bytes longer and is not the party data. The two
exports have identical embedded audio, class table, graph, links, strings,
spline references, animation programs, events, splines, blob sizes, and blob
contents. The real Multiply `extrude` parameter is word 16 and is zero in both
documents. The released editor packing string accidentally contains an 18th
field for a 17-word convention: export therefore reads one word beyond
`DataEdit`, aliasing `DataAnim[0]`. The party export captured zero there; the
published compatibility export captured the precalculated X scale (normally
one, with three customized values). `KOp::Calc` overwrites that word and both
players call byte-identical `Mesh_Multiply` code with only 17 parameters, so
this is ignored padding rather than a semantic scene change. The observable
project difference is the party document's Buzz-timing flag. The browser still
consumes the exact extracted party document.

`notes/debris_class_map.json` joins all 90 party class IDs and operation counts
to their released init/exec handlers, calling conventions, packing strings,
and pinned source locations. Its compact companion drives the plain-JavaScript
class registry that replaces the original x86 stack-call trampoline.

## Exact loader and frame timing

The compact document ends at `0x8bb0e1`. The following 2,889 bytes are the
loader tune and are byte-identical to the released
`debrisfx_beta_edited.v2m` (SHA-256
`d0afcd84ac58991d6b0f8505161be7212b70200a084574eeb8584e4862bcadb5`).
The party player starts it during procedural precalculation, disconnects and
destroys that player, then starts the main song.

The exact PAINT path reads the current audio sample, applies the Buzz divisor
`floor(60*44100/(196*8)) = 1687`, initializes the frame environment, adds the
active timeline events, executes the sole Demo root, and releases frame-local
state. The browser uses the audible WebAudio sample clock as the authority for
this same path.

The embedded main song is V2M format version 6. The public-domain V2Redux
implementation pinned in `vendor/v2redux/UPSTREAM.md` is the readable oracle
for the direct JavaScript synth port. Before the no-Wasm boundary was fixed,
temporary native and WebAssembly builds of that same C++ V2Redux core produced
identical first-second float PCM. “Both targets” referred to those two C++
builds, not to the then-incomplete direct-JavaScript port. An untracked native
audit capture and the audited direct-JavaScript render matched byte-for-byte at
hash `02546c706545c60b079b71e7701b0e87b9921e574bac3f299cff3221bc5cd337`.
Generated PCM remains under ignored `work/`; the ordinary clean-checkout suite
pins that digest, while `tools/oracles/v2_render_oracle.cpp` can regenerate the
raw capture for an optional word-by-word comparison.
The final streaming audit also compared both complete native sequences without
retaining those large PCM files. The party render is 18,934,066 stereo frames
(37,868,132 floats), SHA-256
`b8b1de3b74dbddde309dd3be71a71b13ac11bf0f6bd4056f5a236668236960de`;
the loader render is 2,559,552 stereo frames (5,119,104 floats), SHA-256
`0baeb08456856f4f1defd38d7df023fd3757eb4b9a0d061d973ea5392dad55bb`.
JavaScript matched every float word through both sequence boundaries.
The former JavaScript expectation
`7dd101464ae607bda6333c433a3d1755a64e4154a3cac1bc47c2f26f0d73e277`
was a pre-audit DSP render, not another native oracle: against that 88,200-float
audit capture it differed in 86,814 words, first at word 1,072 (frame 536), with
maximum absolute error about `4.56e-5` and RMS error about `5.58e-6`.

## Published production data

The source release contains:

| file | bytes | purpose |
|---|---:|---|
| `debris_9241.k` | 4,694,564 | werkkzeug3 project/document |
| `debris_9241_c.k` | 4,694,576 | closely related project variant |
| `debris_ost.v2m` | 175,731 | main V2 synthesizer song |
| `debrisfx_beta_edited.v2m` | 2,889 | loader music |

The browser runtime must not depend on the Windows executable. Derived data may
be embedded with the required attribution and license notices.

## Fidelity rules

- The supplied executable wins wherever released source and binary disagree.
- Preserve original serialized IDs, evaluation order, random-number behaviour,
  floating-point precision, timing, and deliberate quirks where observable.
- Make audible audio time the master clock.
- Add deterministic fixed-time hooks and subsystem dumps before visual work is
  allowed to become difficult to compare.
- Treat unpacked bytes, project parsing, generated textures, generated meshes,
  scene/timeline state, V2 PCM, draw-call state, and render targets as separate
  validation seams.
