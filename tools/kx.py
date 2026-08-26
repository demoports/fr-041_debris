#!/usr/bin/env python3
"""Read the compact werkkzeug3 player export used by fr-041: debris.

The layout is a direct, bounds-checked transcription of ``KDoc::Init`` from
Farbrausch's released werkkzeug3 source.  It deliberately does not depend on
the editor project format or any Windows code, which makes it suitable both
for extracting the party build's embedded document and for producing fixtures
for the browser reader.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MAX_OP_ROOT = 16
OPC_FLEXINPUT = 0x80000000
OPC_BLOB = 0x00080000

# Payload bytes following the command byte. This is the CmdSize table in
# kdoc.cpp. KA_LOG2/KA_POW2 were added after the table and have no operands.
ANIM_PAYLOAD_SIZES = (
    0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 16, 4, 4, 2, 0, 0, 0, 8, 2, 0,
    0, 0,
)
KA_END = 0x01


class KxFormatError(ValueError):
    """The byte stream cannot be a valid compact KX document."""


class Cursor:
    def __init__(self, data: bytes | memoryview, offset: int = 0) -> None:
        self.data = memoryview(data)
        self.pos = offset

    def require(self, size: int, what: str = "data") -> None:
        if size < 0 or self.pos + size > len(self.data):
            raise KxFormatError(
                f"truncated {what} at {self.pos:#x}: need {size:#x} bytes, "
                f"have {max(0, len(self.data) - self.pos):#x}"
            )

    def take(self, size: int, what: str = "data") -> bytes:
        self.require(size, what)
        start = self.pos
        self.pos += size
        return bytes(self.data[start : self.pos])

    def skip(self, size: int, what: str = "data") -> int:
        self.require(size, what)
        start = self.pos
        self.pos += size
        return start

    def u8(self) -> int:
        self.require(1)
        value = self.data[self.pos]
        self.pos += 1
        return value

    def u16(self) -> int:
        return struct.unpack("<H", self.take(2))[0]

    def s16(self) -> int:
        return struct.unpack("<h", self.take(2))[0]

    def u24(self) -> int:
        raw = self.take(3)
        return raw[0] | (raw[1] << 8) | (raw[2] << 16)

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def s32(self) -> int:
        return struct.unpack("<i", self.take(4))[0]

    def compact_short(self) -> int:
        value = self.u8()
        if value & 0x80:
            value = (value & 0x7F) | (self.u8() << 7)
        return value

    def cstring(self, what: str = "string") -> str:
        start = self.pos
        while self.pos < len(self.data) and self.data[self.pos] != 0:
            self.pos += 1
        if self.pos == len(self.data):
            raise KxFormatError(f"unterminated {what} at {start:#x}")
        raw = bytes(self.data[start : self.pos])
        self.pos += 1
        return raw.decode("latin-1")

    def f16(self) -> float:
        first = self.u8()
        if first == 0x00:
            return 0.0
        if first == 0x80:
            return 1.0
        if first == 0x01:
            return 0.5
        if first == 0x81:
            # This surprising value is what the released reader returns.
            return 0.25
        value = (first << 8) | self.u8()
        bits = (
            ((value & 0x8000) << 16)
            | (((((value >> 10) & 31) + 128 - 16) & 0xFF) << 23)
            | ((value & 1023) << 13)
        )
        return struct.unpack("<f", struct.pack("<I", bits))[0]

    def f24(self) -> float:
        first = self.u8()
        if first == 0x00:
            return 0.0
        if first == 0x01:
            return 1.0
        if first == 0xFF:
            return -1.0
        second = self.u16()
        bits = (first << 23) | ((second & 0x8000) << 16) | ((second & 0x7FFF) << 8)
        return struct.unpack("<f", struct.pack("<I", bits))[0]


def align4(value: int) -> int:
    return (value + 3) & ~3


def input_count(convention: int) -> int:
    return (convention & 0x00000F00) >> 8


def link_count(convention: int) -> int:
    return (convention & 0x0000F000) >> 12


def string_count(convention: int) -> int:
    return (convention & 0x00070000) >> 16


def spline_count(convention: int) -> int:
    return (convention & 0x00700000) >> 20


def finite_or_string(value: float) -> float | str:
    if math.isfinite(value):
        return value
    return repr(value)


@dataclass
class KxDocument:
    source: memoryview
    start: int
    end: int
    flags: int
    name: str | None
    song_offset: int
    song_size: int
    sample_offset: int | None
    sample_size: int
    song_bpm_fixed: int
    song_length: int
    roots: list[int]
    classes: list[dict[str, Any]]
    operations: list[dict[str, Any]]
    events: list[dict[str, Any]]
    splines: list[dict[str, Any]]
    sections: dict[str, int]

    @property
    def data(self) -> bytes:
        return bytes(self.source[self.start : self.end])

    @property
    def song_data(self) -> bytes:
        return bytes(self.source[self.song_offset : self.song_offset + self.song_size])

    @property
    def sample_data(self) -> bytes:
        if self.sample_offset is None:
            return b""
        return bytes(self.source[self.sample_offset : self.sample_offset + self.sample_size])

    def summary(self) -> dict[str, Any]:
        class_counts = [0] * len(self.classes)
        animation_bytes = 0
        blob_bytes = 0
        blob_count = 0
        for op in self.operations:
            class_counts[op["class_index"]] += 1
            animation_bytes += op["animation_size"]
            if op["blob_size"]:
                blob_count += 1
                blob_bytes += op["blob_size"]

        classes = []
        for index, cls in enumerate(self.classes):
            classes.append(
                {
                    "index": index,
                    "id": cls["id"],
                    "id_hex": f"0x{cls['id']:04x}",
                    "convention_hex": f"0x{cls['convention']:08x}",
                    "packing": cls["packing"],
                    "operation_count": class_counts[index],
                }
            )

        event_starts = [event["start"] for event in self.events]
        event_ends = [event["end"] for event in self.events]
        spline_channels = sum(len(spline["channels"]) for spline in self.splines)
        spline_keys = sum(
            len(channel)
            for spline in self.splines
            for channel in spline["channels"]
        )
        relative_sections = {
            name: offset - self.start for name, offset in self.sections.items()
        }

        return {
            "format": "werkkzeug3 compact KX player export",
            "size": self.end - self.start,
            "sha256": hashlib.sha256(self.data).hexdigest(),
            "flags": self.flags,
            "has_name": bool(self.flags & 1),
            "has_samples": bool(self.flags & 2),
            "buzz_timing": bool(self.flags & 4),
            "name": self.name,
            "song": {
                "offset": self.song_offset - self.start,
                "size": self.song_size,
                "sha256": hashlib.sha256(self.song_data).hexdigest(),
            },
            "samples": {
                "offset": None
                if self.sample_offset is None
                else self.sample_offset - self.start,
                "size": self.sample_size,
                "sha256": hashlib.sha256(self.sample_data).hexdigest()
                if self.sample_offset is not None
                else None,
            },
            "song_bpm_fixed": self.song_bpm_fixed,
            "song_bpm": self.song_bpm_fixed / 65536.0,
            "song_length": self.song_length,
            "operation_count": len(self.operations),
            "event_count": len(self.events),
            "spline_count": len(self.splines),
            "spline_channel_count": spline_channels,
            "spline_key_count": spline_keys,
            "animation_bytes": animation_bytes,
            "blob_count": blob_count,
            "blob_bytes": blob_bytes,
            "event_start_min": min(event_starts) if event_starts else None,
            "event_end_max": max(event_ends) if event_ends else None,
            "roots": self.roots,
            "sections": relative_sections,
            "classes": classes,
        }


def parse_parameter(cursor: Cursor, packing: str) -> int | float | None:
    kind = packing.lower()
    if kind == "g":
        return cursor.f16()
    if kind == "f":
        return cursor.f24()
    if kind == "e":
        return cursor.s16() / 4096.0
    if kind == "i":
        return cursor.s32()
    if kind == "s":
        return cursor.s16()
    if kind == "b":
        return cursor.u8()
    if kind == "c":
        return cursor.u32()
    if kind == "m":
        return cursor.u24()
    # KDoc::Init ignores non-storage characters in a packing string.
    return None


def parse_animation(cursor: Cursor) -> bytes:
    start = cursor.pos
    while True:
        command = cursor.u8()
        if command >= 0x80:
            cursor.skip(1, "animation store operand")
        else:
            if command >= len(ANIM_PAYLOAD_SIZES):
                raise KxFormatError(
                    f"unknown animation opcode {command:#x} at {cursor.pos - 1:#x}"
                )
            cursor.skip(ANIM_PAYLOAD_SIZES[command], "animation operand")
        if command == KA_END:
            return bytes(cursor.data[start : cursor.pos])


def parse_kx(data: bytes | memoryview, offset: int = 0) -> KxDocument:
    """Parse one document and return its exact self-delimited byte range."""

    source = memoryview(data)
    cursor = Cursor(source, offset)
    start = offset
    sections: dict[str, int] = {"header": start}

    flags = cursor.u32()
    if flags & ~7:
        raise KxFormatError(f"unsupported KX flags {flags:#x}")

    name = None
    if flags & 1:
        raw_name = cursor.take(32, "demo name")
        name = raw_name.split(b"\0", 1)[0].decode("latin-1")

    song_size = cursor.u32()
    song_offset = cursor.skip(align4(song_size), "embedded song")
    sections["song"] = song_offset

    sample_offset: int | None = None
    sample_size = 0
    if flags & 2:
        sample_size = cursor.u32()
        sample_offset = cursor.skip(align4(sample_size), "embedded samples")
        sections["samples"] = sample_offset

    sections["document"] = cursor.pos
    song_bpm_fixed = cursor.u32()
    song_length = cursor.u32()
    operation_count = cursor.compact_short()
    document_spline_count = cursor.compact_short()
    roots = [cursor.compact_short() for _ in range(MAX_OP_ROOT)]

    classes: list[dict[str, Any]] = []
    while True:
        convention = cursor.u32()
        if convention == 0:
            break
        class_id = cursor.u16()
        packing = cursor.cstring("operator packing")
        if len(classes) == 128:
            raise KxFormatError("more than 128 exported operator classes")
        classes.append(
            {"convention": convention, "id": class_id, "packing": packing}
        )
    sections["graph"] = cursor.pos

    operations: list[dict[str, Any]] = []
    for op_index in range(operation_count):
        type_byte = cursor.u8()
        class_index = type_byte & 0x7F
        if class_index >= len(classes):
            raise KxFormatError(
                f"operator {op_index} uses missing class index {class_index}"
            )
        convention = classes[class_index]["convention"]
        if type_byte & 0x80:
            count = 1
        elif convention & OPC_FLEXINPUT:
            count = cursor.u8()
        else:
            count = input_count(convention)

        inputs = []
        for _ in range(count):
            delta = 0 if type_byte & 0x80 else cursor.compact_short()
            target = op_index - 1 - delta
            if target < 0:
                raise KxFormatError(
                    f"operator {op_index} has invalid input delta {delta}"
                )
            inputs.append(target)
        operations.append(
            {
                "class_index": class_index,
                "class_id": classes[class_index]["id"],
                "inputs": inputs,
                "links": [],
                "parameters": [],
                "parameter_offset": None,
                "parameter_size": 0,
                "strings": [],
                "splines": [],
                "blob_size": 0,
                "blob_offset": None,
                "animation_offset": None,
                "animation_size": 0,
            }
        )

    for operation in operations:
        convention = classes[operation["class_index"]]["convention"]
        for _ in range(link_count(convention)):
            encoded = cursor.compact_short()
            target = encoded - 1 if encoded else None
            if target is not None and target >= operation_count:
                raise KxFormatError(f"link target {target} is outside the graph")
            operation["links"].append(target)

    sections["parameters"] = cursor.pos
    for class_index, cls in enumerate(classes):
        convention = cls["convention"]
        for operation in operations:
            if operation["class_index"] != class_index:
                continue
            parameter_start = cursor.pos
            operation["parameter_offset"] = parameter_start - start
            operation["parameters"] = [
                finite_or_string(value) if isinstance(value, float) else value
                for value in (
                    parse_parameter(cursor, packing) for packing in cls["packing"]
                )
            ]
            operation["strings"] = [
                cursor.cstring("operator string")
                for _ in range(string_count(convention))
            ]
            operation["splines"] = [
                (reference - 1 if reference else None)
                for reference in (cursor.u16() for _ in range(spline_count(convention)))
            ]
            if convention & OPC_BLOB:
                operation["blob_size"] = cursor.u32()
            operation["parameter_size"] = cursor.pos - parameter_start

    sections["animations"] = cursor.pos
    for operation in operations:
        operation["animation_offset"] = cursor.pos - start
        animation = parse_animation(cursor)
        operation["animation_size"] = len(animation)

    sections["events"] = cursor.pos
    event_count = cursor.compact_short()
    events: list[dict[str, Any]] = []
    for _ in range(event_count):
        op = cursor.compact_short()
        if op >= operation_count:
            raise KxFormatError(f"event operator {op} is outside the graph")
        start_time = cursor.s32()
        end_time = cursor.s32()
        velocity = cursor.f24()
        modulation = cursor.f24()
        select = cursor.u8()
        scale = [cursor.f24() for _ in range(3)]
        rotate = [cursor.f24() for _ in range(3)]
        translate = [cursor.f24() for _ in range(3)]
        color = cursor.u32()
        spline_ref = cursor.compact_short()
        event_spline = spline_ref - 1 if spline_ref else None
        start_interval = cursor.f16()
        end_interval = cursor.f16()
        event_flags = cursor.u8()
        events.append(
            {
                "operation": op,
                "start": start_time,
                "end": end_time,
                "velocity": finite_or_string(velocity),
                "modulation": finite_or_string(modulation),
                "select": select,
                "scale": [finite_or_string(value) for value in scale],
                "rotate": [finite_or_string(value) for value in rotate],
                "translate": [finite_or_string(value) for value in translate],
                "color": color,
                "spline": event_spline,
                "start_interval": finite_or_string(start_interval),
                "end_interval": finite_or_string(end_interval),
                "flags": event_flags,
            }
        )

    sections["splines"] = cursor.pos
    splines: list[dict[str, Any]] = []
    for _ in range(document_spline_count):
        descriptor = cursor.u8()
        count = descriptor & 7
        interpolation = descriptor >> 3
        if count > 4:
            raise KxFormatError(f"spline has {count} channels")
        channels = []
        for _ in range(count):
            key_count = cursor.compact_short()
            keys = []
            for _ in range(key_count):
                time = cursor.u8() / 255.0
                value = cursor.f16()
                keys.append([time, finite_or_string(value)])
            channels.append(keys)
        splines.append(
            {"interpolation": interpolation, "channels": channels}
        )

    sections["blobs"] = cursor.pos
    for operation in operations:
        size = operation["blob_size"]
        if size:
            operation["blob_offset"] = cursor.pos - start
            cursor.skip(size, "operator blob")

    sections["end"] = cursor.pos
    return KxDocument(
        source=source,
        start=start,
        end=cursor.pos,
        flags=flags,
        name=name,
        song_offset=song_offset,
        song_size=song_size,
        sample_offset=sample_offset,
        sample_size=sample_size,
        song_bpm_fixed=song_bpm_fixed,
        song_length=song_length,
        roots=roots,
        classes=classes,
        operations=operations,
        events=events,
        splines=splines,
        sections=sections,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--offset", type=lambda value: int(value, 0), default=0)
    parser.add_argument("--full", action="store_true", help="include graph/event details")
    args = parser.parse_args()

    document = parse_kx(args.input.read_bytes(), args.offset)
    result = document.summary()
    if args.full:
        result["operations"] = document.operations
        result["events"] = document.events
        result["splines"] = document.splines
    print(json.dumps(result, indent=2) + "\n", end="")


if __name__ == "__main__":
    main()
