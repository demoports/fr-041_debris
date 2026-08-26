# Debris production data

`debris_party.kx` is the exact compact werkkzeug3 project export extracted from
the supplied scene.org party executable, `fr-041_debris.exe`. It contains the
production graph, animation/timeline data, procedural-generator blobs, V2
music, and sound-effect sample bank used by the browser reconstruction.

Farbrausch released the Debris project and music under the Creative Commons
Attribution 3.0 license:

> This directory contains the project and music files for fr-041: debris.
> All are released under a CC-BY 3.0 license.

License: <https://creativecommons.org/licenses/by/3.0/>

Original production credits are preserved in the launcher and
[`README.md`](../README.md). The released source is available from
<https://github.com/farbrausch/fr_public/tree/original/werkkzeug3/data/debris>.

`debris_loader.v2m` is the loader tune embedded immediately after the compact
project. It is byte-identical to the released `debrisfx_beta_edited.v2m`.

`launcher-background.jpg` is a compact export of the user-selected 2:1 WebGL
capture. It is used only behind the launcher; the depicted production content
is covered by the same Farbrausch CC BY 3.0 release and credits above.

## Deterministic Font3D substitutes

`src/font3d_glyphs.js` contains modified lowercase glyph subsets of Arimo
Regular 1.33 and Gelasio Regular 1.008. Those open fonts are horizontally
metric-compatible with Arial and Georgia respectively and replace the original
player's proprietary system-font dependency without shipping pregenerated
geometry. Both subsets are licensed under the SIL Open Font License 1.1; source
metadata, copyright notices, and the complete license are in
`vendor/font3d/LICENSE.txt`.
