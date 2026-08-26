#!/usr/bin/env python3
"""Extract the exact self-delimited KX document from an unpacked image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kx import parse_kx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--image-base", type=lambda value: int(value, 0), default=0x800000)
    location = parser.add_mutually_exclusive_group()
    location.add_argument("--kx-va", type=lambda value: int(value, 0), default=0x843684)
    location.add_argument("--offset", type=lambda value: int(value, 0))
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--song-output", type=Path)
    parser.add_argument("--samples-output", type=Path)
    args = parser.parse_args()

    image = args.image.read_bytes()
    offset = args.offset if args.offset is not None else args.kx_va - args.image_base
    if offset < 0 or offset >= len(image):
        parser.error(f"KX offset {offset:#x} is outside the {len(image):#x}-byte image")

    document = parse_kx(image, offset)
    summary = document.summary()
    summary["image"] = str(args.image)
    summary["image_base"] = f"{args.image_base:#x}"
    summary["image_offset"] = f"{offset:#x}"
    summary["virtual_address"] = f"{args.image_base + offset:#x}"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(document.data)
    summary["output"] = str(args.output)

    if args.song_output:
        args.song_output.parent.mkdir(parents=True, exist_ok=True)
        args.song_output.write_bytes(document.song_data)
        summary["song"]["output"] = str(args.song_output)
    if args.samples_output:
        args.samples_output.parent.mkdir(parents=True, exist_ok=True)
        args.samples_output.write_bytes(document.sample_data)
        summary["samples"]["output"] = str(args.samples_output)
    if args.metadata:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2) + "\n", end="")


if __name__ == "__main__":
    main()
