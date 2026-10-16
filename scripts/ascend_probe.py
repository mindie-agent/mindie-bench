#!/usr/bin/env python3
"""Read physical NPU ownership through the installed vendor DCMI ABI.

The operator must validate this probe against npu-smi and the actual allocation.
No textual npu-smi parser or percentage-to-memory estimate is used.
"""

import argparse
import ctypes as ct
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


class HBM(ct.Structure):
    _fields_ = [
        ("memory_size", ct.c_ulonglong),
        ("freq", ct.c_uint),
        ("memory_usage", ct.c_ulonglong),
        ("temp", ct.c_int),
        ("bandwidth", ct.c_uint),
    ]


class Process(ct.Structure):
    _fields_ = [("pid", ct.c_int), ("used_mb", ct.c_ulong)]


class Chip(ct.Structure):
    _fields_ = [
        ("type", ct.c_char * 32),
        ("name", ct.c_char * 32),
        ("version", ct.c_char * 32),
        ("cores", ct.c_uint),
    ]


def call(function, *arguments):
    code = function(*arguments)
    if code != 0:
        raise RuntimeError(f"{function.__name__} returned {code}; probe is unavailable")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--dcmi", default="/usr/local/dcmi/libdcmi.so")
    parser.add_argument(
        "--cann-version", default="/usr/local/Ascend/ascend-toolkit/latest/compiler/version.info"
    )
    args = parser.parse_args()
    lib = ct.CDLL(args.dcmi)
    call(lib.dcmi_init)
    driver = ct.create_string_buffer(256)
    call(lib.dcmi_get_driver_version, driver, ct.c_uint(len(driver)))
    result = {}
    for physical in map(int, args.devices.split(",")):
        card, chip_id = ct.c_int(), ct.c_int()
        call(
            lib.dcmi_get_card_id_device_id_from_phyid,
            ct.byref(card),
            ct.byref(chip_id),
            ct.c_uint(physical),
        )
        hbm, chip = HBM(), Chip()
        call(lib.dcmi_get_device_hbm_info, card, chip_id, ct.byref(hbm))
        call(lib.dcmi_get_device_chip_info, card, chip_id, ct.byref(chip))
        processes, count = (Process * 1024)(), ct.c_int(1024)
        call(lib.dcmi_get_device_resource_info, card, chip_id, processes, ct.byref(count))
        if not 0 <= count.value <= 1024:
            raise RuntimeError("invalid DCMI process inventory size")
        result[str(physical)] = {
            "used_mb": int(hbm.memory_usage),
            "capacity_mb": int(hbm.memory_size),
            "pids": [processes[i].pid for i in range(count.value)],
            "model": chip.name.decode(),
            "physical_id": physical,
            "card_id": card.value,
            "chip_id": chip_id.value,
        }
    result["_environment"] = {
        "image_digest": args.image_digest,
        "cann": Path(args.cann_version).read_text().strip(),
        "torch_npu": importlib.metadata.version("torch_npu"),
        "driver": driver.value.decode(),
        "python": sys.version,
        "kernel": platform.release(),
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
