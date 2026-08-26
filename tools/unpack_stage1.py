#!/usr/bin/env python3
"""Run the first kkrunchy depacker stage and dump its output.

The packed Debris PE uses a self-contained arithmetic decoder before it calls
any operating-system function. Running that exact stub in Unicorn avoids
reimplementing a packer revision that is newer than the public-domain CCA
depacker present in the later fr_public source tree.

Only the PE image and a scratch stack are mapped. Emulation stops at the first
instruction of the freshly decoded second-stage image, before import loading or
any demo code can run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import pefile
from unicorn import Uc, UcError, UC_ARCH_X86, UC_HOOK_MEM_INVALID, UC_MODE_32
from unicorn.x86_const import UC_X86_REG_EIP, UC_X86_REG_ESP


PAGE = 0x1000
STACK_BASE = 0x3000000
STACK_SIZE = 0x100000


def align_up(value: int, alignment: int = PAGE) -> int:
    return (value + alignment - 1) & -alignment


def u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def locate_parameters(pe: pefile.PE, data: bytes) -> dict[str, int]:
    """Recover patched constants from the beginning of the stage-one stub."""

    base = pe.OPTIONAL_HEADER.ImageBase
    entry_rva = pe.OPTIONAL_HEADER.AddressOfEntryPoint
    entry_off = pe.get_offset_from_rva(entry_rva)
    code = data[entry_off : entry_off + 0x80]

    # Debris begins:
    #   bd WORK
    #   c7 45 00 SOURCE
    #   b8 DEST
    #   89 45 04; 89 45 54; 50
    #   c7 45 10 OUTPUT_COUNT
    if not (
        code[0] == 0xBD
        and code[5:8] == b"\xc7\x45\x00"
        and code[12] == 0xB8
        and code[17:24] == b"\x89\x45\x04\x89\x45\x54\x50"
        and code[24:27] == b"\xc7\x45\x10"
    ):
        raise ValueError("unrecognised kkrunchy stage-one entry stub")

    work = u32(code, 1)
    source = u32(code, 8)
    destination = u32(code, 13)
    output_count = u32(code, 27)

    if not (base <= source < base + pe.OPTIONAL_HEADER.SizeOfHeaders):
        raise ValueError(f"compressed source {source:#x} is not in the mapped PE headers")
    if not (base <= destination < work < base + pe.OPTIONAL_HEADER.SizeOfImage):
        raise ValueError("decoded destination/work area is outside the PE image")
    if destination + output_count > work:
        raise ValueError("decoded output overlaps the depacker work area")

    return {
        "image_base": base,
        "entry_rva": entry_rva,
        "entry_va": base + entry_rva,
        "entry_file_offset": entry_off,
        "work_va": work,
        "source_va": source,
        "source_file_offset": source - base,
        "destination_va": destination,
        "output_count": output_count,
    }


def map_pe(uc: Uc, pe: pefile.PE, data: bytes) -> None:
    base = pe.OPTIONAL_HEADER.ImageBase
    size = align_up(pe.OPTIONAL_HEADER.SizeOfImage)
    uc.mem_map(base, size)
    uc.mem_write(base, data[: pe.OPTIONAL_HEADER.SizeOfHeaders])

    for section in pe.sections:
        raw_size = section.SizeOfRawData
        if not raw_size:
            continue
        raw = data[section.PointerToRawData : section.PointerToRawData + raw_size]
        uc.mem_write(base + section.VirtualAddress, raw)


def unpack(input_path: Path, output_path: Path, metadata_path: Path | None) -> dict[str, object]:
    data = input_path.read_bytes()
    pe = pefile.PE(data=data, fast_load=True)
    params = locate_parameters(pe, data)

    uc = Uc(UC_ARCH_X86, UC_MODE_32)
    map_pe(uc, pe, data)
    uc.mem_map(STACK_BASE, STACK_SIZE)
    uc.reg_write(UC_X86_REG_ESP, STACK_BASE + STACK_SIZE - 0x100)

    def on_invalid(
        mu: Uc, access: int, address: int, size: int, value: int, user_data: object
    ) -> bool:
        eip = mu.reg_read(UC_X86_REG_EIP)
        raise RuntimeError(
            f"invalid memory access type={access} address={address:#x} size={size} at eip={eip:#x}"
        )

    destination = int(params["destination_va"])
    uc.hook_add(UC_HOOK_MEM_INVALID, on_invalid)

    try:
        uc.emu_start(int(params["entry_va"]), destination)
    except UcError as exc:
        eip = uc.reg_read(UC_X86_REG_EIP)
        raise RuntimeError(f"Unicorn failed at eip={eip:#x}: {exc}") from exc

    # The second argument to emu_start is an exclusive stop address. Unicorn
    # therefore returns immediately before executing the decoded stage-two
    # entrypoint.
    if uc.reg_read(UC_X86_REG_EIP) != destination:
        raise RuntimeError(
            f"emulation stopped at {uc.reg_read(UC_X86_REG_EIP):#x}, expected stage two at {destination:#x}"
        )

    output_count = int(params["output_count"])
    decoded = bytes(uc.mem_read(destination, output_count))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(decoded)

    result: dict[str, object] = {
        "input": str(input_path),
        "input_size": len(data),
        "input_sha256": hashlib.sha256(data).hexdigest(),
        **{key: f"{value:#x}" for key, value in params.items()},
        "decoded_size": len(decoded),
        "decoded_sha256": hashlib.sha256(decoded).hexdigest(),
        "output": str(output_path),
    }
    if metadata_path:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args()

    result = unpack(args.input, args.output, args.metadata)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
