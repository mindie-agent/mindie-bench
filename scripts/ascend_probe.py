#!/usr/bin/env python3
# [Role] Reads physical NPU ownership through the installed vendor DCMI ABI.
# [Invariants] Fail closed: missing or uncertain evidence is never reported as success.
# [Timeouts & Locks] Callers bound external waits; no unbounded process or lock wait is owned here.
# [Side-Effect Policy] Writes only explicitly owned artifacts; no network or host user-directory mutation.

"""Read physical NPU ownership through the installed vendor DCMI ABI.

The operator must validate this probe against npu-smi and the actual allocation.
No textual npu-smi parser or percentage-to-memory estimate is used.
"""

import argparse
import ctypes as ct
import importlib.metadata
import json
import os
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


class ChipV2(ct.Structure):
    _fields_ = [
        ("type", ct.c_char * 32),
        ("name", ct.c_char * 32),
        ("version", ct.c_char * 32),
        ("cores", ct.c_uint),
        ("npu_name", ct.c_char * 32),
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
    driver = ct.create_string_buffer(256)
    init_code = lib.dcmi_init()
    use_v2 = False
    if init_code == -8255 and hasattr(lib, "dcmiv2_init"):
        call(lib.dcmiv2_init)
        use_v2 = True
        call(lib.dcmiv2_get_driver_version, driver, ct.c_uint(len(driver)))
    elif init_code != 0:
        raise RuntimeError(f"dcmi_init returned {init_code}; probe is unavailable")
    else:
        call(lib.dcmi_get_driver_version, driver, ct.c_uint(len(driver)))
    result = {}
    for physical in map(int, args.devices.split(",")):
        hbm = HBM()
        processes, count = (Process * 1024)(), ct.c_int(1024)
        if use_v2:
            dev_id = ct.c_uint()
            call(lib.dcmiv2_get_dev_id_by_chip_phy_id, ct.c_int(physical), ct.byref(dev_id))
            logic_id = int(dev_id.value)
            chip = ChipV2()
            call(lib.dcmiv2_get_device_hbm_info, ct.c_int(logic_id), ct.byref(hbm))
            call(lib.dcmiv2_get_device_chip_info, ct.c_int(logic_id), ct.byref(chip))
            call(
                lib.dcmiv2_get_device_proc_mem_info,
                ct.c_int(logic_id),
                processes,
                ct.byref(count),
            )
            card_val, chip_val = logic_id, 0
        else:
            card, chip_id = ct.c_int(), ct.c_int()
            call(
                lib.dcmi_get_card_id_device_id_from_phyid,
                ct.byref(card),
                ct.byref(chip_id),
                ct.c_uint(physical),
            )
            chip = Chip()
            call(lib.dcmi_get_device_hbm_info, card, chip_id, ct.byref(hbm))
            call(lib.dcmi_get_device_chip_info, card, chip_id, ct.byref(chip))
            call(lib.dcmi_get_device_resource_info, card, chip_id, processes, ct.byref(count))
            card_val, chip_val = card.value, chip_id.value
        if not 0 <= count.value <= 1024:
            raise RuntimeError("invalid DCMI process inventory size")
        result[str(physical)] = {
            "used_mb": int(hbm.memory_usage),
            "capacity_mb": int(hbm.memory_size),
            "pids": [processes[i].pid for i in range(count.value)],
            "model": chip.name.decode(),
            "physical_id": physical,
            "card_id": card_val,
            "chip_id": chip_val,
        }
    result["_environment"] = {
        "image_digest": args.image_digest,
        "cann": Path(args.cann_version).read_text().strip(),
        "torch_npu": importlib.metadata.version("torch_npu"),
        "driver": driver.value.decode(),
        "python": sys.version,
        "kernel": platform.release(),
    }
    print(json.dumps(result), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
