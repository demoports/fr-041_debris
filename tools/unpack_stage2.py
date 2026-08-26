#!/usr/bin/env python3
"""Run kkrunchy's disassembly unfilter and reconstruct the original image.

Input is the stage-one dump produced by ``unpack_stage1.py``. The only Windows
calls made by stage two are LoadLibraryA and GetProcAddress; this tool replaces
them with deterministic stubs, records the complete dynamic import table, and
stops immediately before the original program entrypoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from pathlib import Path

import pefile
from unicorn import Uc, UcError, UC_ARCH_X86, UC_HOOK_CODE, UC_HOOK_MEM_INVALID, UC_MODE_32
from unicorn.x86_const import (
    UC_X86_REG_EAX,
    UC_X86_REG_EBP,
    UC_X86_REG_EBX,
    UC_X86_REG_EDI,
    UC_X86_REG_EIP,
    UC_X86_REG_ESI,
    UC_X86_REG_ESP,
)

from unpack_stage1 import PAGE, STACK_BASE, STACK_SIZE, align_up, locate_parameters, map_pe


FAKE_API_BASE = 0x2000000
FAKE_LOAD_LIBRARY = FAKE_API_BASE
FAKE_GET_PROC = FAKE_API_BASE + 0x10


def u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def s32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<i", data, offset)[0]


def read_c_string(mu: Uc, address: int, limit: int = 1024) -> str:
    out = bytearray()
    for i in range(limit):
        value = mu.mem_read(address + i, 1)[0]
        if value == 0:
            return out.decode("ascii", errors="replace")
        out.append(value)
    raise RuntimeError(f"unterminated string at {address:#x}")


def locate_stage2_parameters(
    pe: pefile.PE, packed: bytes, stage1: bytes, stage1_params: dict[str, int]
) -> dict[str, int]:
    entry_off = int(stage1_params["entry_file_offset"])
    entry = packed[entry_off : entry_off + 0x500]
    transition = re.search(
        rb"\x0f\x77\x58\x68(.{4})\x68(.{4})\x50\xbe(.{4})\xbb(.{4})\xc3",
        entry,
        re.DOTALL,
    )
    if not transition:
        raise ValueError("could not locate the stage-one to stage-two transition")

    original_base, patch_value, stage2_input, loader_iat = struct.unpack(
        "<IIII", b"".join(transition.groups())
    )
    stage2_va = int(stage1_params["destination_va"])

    if stage1[:6] != b"\x53\xad\x56\x01\xc6\xbf":
        raise ValueError("unrecognised stage-two prologue")
    code_va = u32(stage1, 6)

    jump = re.search(rb"\x5d\xe9(.{4})", stage1[:0x100], re.DOTALL)
    if not jump:
        raise ValueError("could not locate the stage-two entrypoint jump")
    jump_offset = jump.start() + 1
    original_entry = stage2_va + jump_offset + 5 + s32(stage1, jump_offset + 1)

    stage2_input_offset = stage2_input - stage2_va
    if not (0 <= stage2_input_offset <= len(stage1) - 4):
        raise ValueError("stage-two input pointer is outside the decoded data")
    import_size = u32(stage1, stage2_input_offset)

    output_end = stage2_va + int(stage1_params["output_count"])
    if code_va < original_base or output_end <= original_base:
        raise ValueError("invalid reconstructed image bounds")

    return {
        "stage2_va": stage2_va,
        "stage2_input_va": stage2_input,
        "stage2_input_offset": stage2_input_offset,
        "import_size": import_size,
        "loader_iat_va": loader_iat,
        "code_va": code_va,
        "original_base": original_base,
        "patch_value": patch_value,
        "original_entry_va": original_entry,
        "original_entry_rva": original_entry - original_base,
        "original_image_size": output_end - original_base,
    }


def reconstruct(
    packed_path: Path,
    stage1_path: Path,
    output_path: Path,
    imports_path: Path,
    metadata_path: Path | None,
) -> dict[str, object]:
    packed = packed_path.read_bytes()
    stage1 = stage1_path.read_bytes()
    pe = pefile.PE(data=packed, fast_load=True)
    stage1_params = locate_parameters(pe, packed)
    if len(stage1) != int(stage1_params["output_count"]):
        raise ValueError(
            f"stage-one dump is {len(stage1):#x} bytes; expected {int(stage1_params['output_count']):#x}"
        )
    params = locate_stage2_parameters(pe, packed, stage1, stage1_params)

    mu = Uc(UC_ARCH_X86, UC_MODE_32)
    map_pe(mu, pe, packed)
    mu.mem_write(int(params["stage2_va"]), stage1)
    mu.mem_map(STACK_BASE, STACK_SIZE)
    mu.mem_map(FAKE_API_BASE, PAGE)
    mu.mem_write(FAKE_API_BASE, b"\xc3" * PAGE)

    stack = STACK_BASE + STACK_SIZE - 0x100
    mu.mem_write(
        stack,
        struct.pack("<II", int(params["patch_value"]), int(params["original_base"])),
    )
    mu.reg_write(UC_X86_REG_ESP, stack)
    mu.reg_write(UC_X86_REG_EBP, int(stage1_params["work_va"]))
    mu.reg_write(UC_X86_REG_ESI, int(params["stage2_input_va"]))
    mu.reg_write(UC_X86_REG_EBX, int(params["loader_iat_va"]))
    mu.reg_write(UC_X86_REG_EAX, int(params["stage2_va"]))

    mu.mem_write(
        int(params["loader_iat_va"]),
        struct.pack("<II", FAKE_LOAD_LIBRARY, FAKE_GET_PROC),
    )

    libraries: list[dict[str, object]] = []
    imports: list[dict[str, object]] = []
    handles: dict[int, str] = {}

    def return_stdcall(uc: Uc, argument_bytes: int, eax: int) -> None:
        esp = uc.reg_read(UC_X86_REG_ESP)
        return_address = u32(bytes(uc.mem_read(esp, 4)), 0)
        uc.reg_write(UC_X86_REG_EAX, eax)
        uc.reg_write(UC_X86_REG_ESP, esp + 4 + argument_bytes)
        uc.reg_write(UC_X86_REG_EIP, return_address)

    def on_load_library(uc: Uc, address: int, size: int, user_data: object) -> None:
        esp = uc.reg_read(UC_X86_REG_ESP)
        name_ptr = u32(bytes(uc.mem_read(esp + 4, 4)), 0)
        name = read_c_string(uc, name_ptr)
        handle = 0x40000000 + len(libraries) * 0x10000
        libraries.append({"name": name, "handle": f"{handle:#x}"})
        handles[handle] = name
        return_stdcall(uc, 4, handle)

    def on_get_proc(uc: Uc, address: int, size: int, user_data: object) -> None:
        esp = uc.reg_read(UC_X86_REG_ESP)
        handle, proc = struct.unpack("<II", bytes(uc.mem_read(esp + 4, 8)))
        if proc < 0x10000:
            name: str | int = proc
        else:
            name = read_c_string(uc, proc)
        fake_address = 0x50000000 + len(imports) * 0x10
        imports.append(
            {
                "library": handles.get(handle, f"handle_{handle:#x}"),
                "name": name,
                "iat_va": f"{uc.reg_read(UC_X86_REG_EDI):#x}",
                "fake_address": f"{fake_address:#x}",
            }
        )
        return_stdcall(uc, 8, fake_address)

    def on_invalid(
        uc: Uc, access: int, address: int, size: int, value: int, user_data: object
    ) -> bool:
        raise RuntimeError(
            f"invalid memory access type={access} address={address:#x} size={size} "
            f"at eip={uc.reg_read(UC_X86_REG_EIP):#x}"
        )

    mu.hook_add(UC_HOOK_CODE, on_load_library, begin=FAKE_LOAD_LIBRARY, end=FAKE_LOAD_LIBRARY)
    mu.hook_add(UC_HOOK_CODE, on_get_proc, begin=FAKE_GET_PROC, end=FAKE_GET_PROC)
    mu.hook_add(UC_HOOK_MEM_INVALID, on_invalid)

    try:
        mu.emu_start(int(params["stage2_va"]), int(params["original_entry_va"]))
    except UcError as exc:
        raise RuntimeError(f"stage two failed at {mu.reg_read(UC_X86_REG_EIP):#x}: {exc}") from exc

    if mu.reg_read(UC_X86_REG_EIP) != int(params["original_entry_va"]):
        raise RuntimeError("stage two stopped before the original entrypoint")

    original_base = int(params["original_base"])
    original_size = int(params["original_image_size"])
    image = bytes(mu.mem_read(original_base, original_size))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(image)
    imports_path.parent.mkdir(parents=True, exist_ok=True)
    imports_path.write_text(json.dumps({"libraries": libraries, "imports": imports}, indent=2) + "\n")

    result: dict[str, object] = {
        "packed_input": str(packed_path),
        "stage1_input": str(stage1_path),
        **{key: f"{value:#x}" for key, value in params.items()},
        "library_count": len(libraries),
        "import_count": len(imports),
        "image_sha256": hashlib.sha256(image).hexdigest(),
        "output": str(output_path),
        "imports_output": str(imports_path),
    }
    if metadata_path:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("packed", type=Path)
    parser.add_argument("stage1", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--imports", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args()

    result = reconstruct(args.packed, args.stage1, args.output, args.imports, args.metadata)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
