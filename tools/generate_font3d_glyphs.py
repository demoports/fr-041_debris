#!/usr/bin/env python3
"""Generate the small deterministic Font3D outline table used by Debris.

This is a maintainer tool, not a runtime dependency. It requires fontTools and
the two pinned OFL-licensed source TTFs described by --help. The output is an
ES module written to stdout so it can be reviewed before replacing
src/font3d_glyphs.js.
"""

from argparse import ArgumentParser
from hashlib import sha256
from json import dumps
from pathlib import Path

from fontTools.ttLib import TTFont


# Reference bounds were measured with HarfBuzz from the Microsoft Core Fonts
# for the Web Arial 2.82/Georgia 2.05 packages and cross-checked against the
# matching 5.x system faces. Only the numeric bounds are retained here.
FAMILIES = (
    {
        "key": "arial",
        "source": "Arimo Regular 1.33",
        "commit": "4a6255f269916ae7ad3fc2706b0935e7621396b8",
        "argument": "arimo",
        "characters": ".adefhlmnoprstuvw",
        "units_per_em": 2048,
        # CreateFontA receives a positive height. Windows maps that to the
        # Arial character-cell height (usWinAscent + usWinDescent), not to the
        # TrueType em square.
        "units_per_cell": 2288,
        # CreateFontA selects these integer pixels-per-em values through
        # Arial 2.82's VDMX table for the two heights authored by Debris.
        "ppem_by_logical_height": {64: 55, 128: 114},
        # Reusable bounds only, not Microsoft outline points. Arial 2.82 and
        # 5.01 have identical bounds for this production subset.
        "reference_bounds": {
            ".": [186, 0, 391, 205],
            "a": [74, -24, 1052, 1086],
            "d": [70, -24, 991, 1466],
            "e": [75, -24, 1054, 1086],
            "f": [19, 0, 640, 1491],
            "h": [135, 0, 1000, 1466],
            "l": [131, 0, 311, 1466],
            "m": [135, 0, 1574, 1086],
            "n": [135, 0, 998, 1086],
            "o": [68, -24, 1063, 1086],
            "p": [135, -407, 1057, 1086],
            "r": [133, 0, 710, 1086],
            "s": [63, -24, 945, 1086],
            "t": [36, -14, 554, 1433],
            "u": [131, -24, 992, 1062],
            "v": [26, 0, 1000, 1062],
            "w": [6, 0, 1463, 1062],
        },
        "sha256": "41b22bc8f0b51f932825d37bc55b5eb6ba67dfe599a626e4aff2b43b624f9f8c",
    },
    {
        "key": "georgia",
        "source": "Gelasio Regular 1.008",
        "commit": "7ab20e7e5c42791e603b9ee3201a0b49849cfdb2",
        "argument": "gelasio",
        "characters": " .abcdefghiklmnopqrstuvwxy",
        "units_per_em": 2048,
        "units_per_cell": 2327,
        # Georgia 2.05's VDMX table has two consecutive 128-cell records;
        # Windows/Wine select the first one, at 112 ppem.
        "ppem_by_logical_height": {128: 112},
        # Georgia 2.05 and 5.00 likewise retain these subset bounds.
        "reference_bounds": {
            ".": [141, -20, 415, 252],
            "a": [80, -25, 1006, 1014],
            "b": [-16, -53, 1073, 1548],
            "c": [72, -26, 925, 1014],
            "d": [72, -28, 1152, 1548],
            "e": [72, -26, 948, 1014],
            "f": [24, 0, 806, 1549],
            "g": [42, -444, 1003, 1014],
            "h": [6, 0, 1143, 1548],
            "i": [43, 0, 551, 1515],
            "k": [6, 0, 1117, 1548],
            "l": [2, 0, 534, 1548],
            "m": [57, 0, 1761, 1013],
            "n": [61, 0, 1161, 1013],
            "o": [70, -30, 1033, 1014],
            "p": [33, -444, 1097, 1013],
            "q": [72, -444, 1154, 1040],
            "r": [61, 0, 827, 1010],
            "s": [73, -26, 808, 1015],
            "t": [13, -21, 681, 1289],
            "u": [35, -25, 1129, 998],
            "v": [-19, -9, 1046, 986],
            "w": [-5, -7, 1541, 986],
            "x": [12, 0, 1022, 986],
            "y": [-19, -444, 1038, 986],
        },
        "sha256": "48c797fbe0e07c48a18cb962e7bdfa23f19618327dddf54093265328dc9eb39d",
    },
)


def glyph_record(font, character):
    cmap = font.getBestCmap()
    glyph_name = cmap[ord(character)]
    advance, _left_side_bearing = font["hmtx"][glyph_name]
    glyph = font["glyf"][glyph_name]
    coordinates, endpoints, flags = glyph.getCoordinates(font["glyf"])
    contours = []
    first = 0
    for endpoint in endpoints:
        contour = []
        for index in range(first, endpoint + 1):
            x, y = coordinates[index]
            contour.extend((int(x), int(y), int(bool(flags[index] & 1))))
        contours.append(contour)
        first = endpoint + 1
    return [int(advance), contours]


def quoted(value):
    return dumps(value, ensure_ascii=True, separators=(",", ":"))


def load_font(path, description):
    digest = sha256(path.read_bytes()).hexdigest()
    if digest != description["sha256"]:
        raise ValueError(
            f"{path} has SHA-256 {digest}; expected {description['sha256']}"
        )
    font = TTFont(path, lazy=False)
    if font["head"].unitsPerEm != description["units_per_em"]:
        raise ValueError(f"{path} has unexpected {font['head'].unitsPerEm} units/em")
    return font


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--arimo", type=Path, required=True,
                        help="Arimo-Regular.ttf from googlefonts/Arimo commit 4a6255f")
    parser.add_argument("--gelasio", type=Path, required=True,
                        help="Gelasio-Regular.ttf from SorkinType/Gelasio commit 7ab20e7")
    args = parser.parse_args()

    print("// Generated by tools/generate_font3d_glyphs.py. Do not edit by hand.")
    print("// The compact subsets are OFL-1.1 Font Software; see vendor/font3d/LICENSE.txt.")
    print("const FONT3D_FAMILIES = {")
    for family_index, description in enumerate(FAMILIES):
        font = load_font(getattr(args, description["argument"]), description)
        expected_bounds = set(description["characters"]) - {" "}
        if set(description["reference_bounds"]) != expected_bounds:
            raise ValueError(f"{description['key']} reference bounds do not cover its outline subset")
        print(f"  {quoted(description['key'])}: {{")
        print(f"    source: {quoted(description['source'])},")
        print(f"    sourceCommit: {quoted(description['commit'])},")
        print(f"    sourceSHA256: {quoted(description['sha256'])},")
        print(f"    unitsPerEm: {description['units_per_em']},")
        print(f"    unitsPerCell: {description['units_per_cell']},")
        print(f"    ppemByLogicalHeight: {quoted(description['ppem_by_logical_height'])},")
        print(f"    referenceBounds: {quoted(description['reference_bounds'])},")
        print("    glyphs: {")
        for index, character in enumerate(description["characters"]):
            suffix = "," if index + 1 < len(description["characters"]) else ""
            print(f"      {quoted(character)}: {quoted(glyph_record(font, character))}{suffix}")
        print("    }")
        suffix = "," if family_index + 1 < len(FAMILIES) else ""
        print(f"  }}{suffix}")
        font.close()
    print("};")
    print("")
    print("export { FONT3D_FAMILIES };")


if __name__ == "__main__":
    main()
