# V2 Redux provenance

This directory vendors the self-contained audio-engine portion of
[`spheenik/v2redux`](https://github.com/spheenik/v2redux) at commit
`5d3157b7c312fbfe2f82430dd3e819594cf3205b`.

Only the library sources needed to render V2M version 6 are included. They are
the readable behavioral reference for the direct JavaScript parser, sequencer,
and synthesizer under `src`; the browser does not compile or execute these C++
files. Upstream's code and the original Farbrausch V2 engine are public domain;
the complete notices are preserved in `LICENSE`.

Debris's embedded song identifies as native V2M version 6. The vendored port's
version-6 path is independently validated by upstream against the original
period engine and is deterministic across x86-64 and ARM64.
