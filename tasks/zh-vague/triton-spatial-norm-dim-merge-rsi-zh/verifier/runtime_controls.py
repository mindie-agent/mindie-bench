"""Capture executed tensor entrypoints, then replay only their small controls.

[Role] Adapt candidate ablation interfaces through actual calls and tensor results.
[Invariants] Names/signatures only bind calls; independent tensor oracles decide success.
[Timeouts & Locks] At most 16 controls per entry and 32 entries; caller bounds process.
[Side-Effect Policy] Temporarily wraps candidate module functions; restores them on exit.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import inspect
import json
import types

import torch


class UnsupportedCPUExecution(NotImplementedError):
    pass


def _small(value, budget):
    budget[0] -= 1
    if budget[0] < 0:
        raise ValueError("control metadata exceeds the bounded portable representation")
    if isinstance(value, str) and len(value) > 512:
        raise ValueError("control string exceeds the portable bound")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _small(getattr(value, field.name), budget) for field in dataclasses.fields(value)
        }
    if isinstance(value, (tuple, list)):
        return [_small(v, budget) for v in value]
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {k: _small(v, budget) for k, v in value.items()}
    raise ValueError("tensor/state objects cannot be retained as ablation controls")


def output_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, tuple) and value and isinstance(value[0], torch.Tensor):
        return value[0]
    return None


class Control:
    def __init__(self, fn, tail, kwargs, description, phase, family):
        self.fn = fn
        self.tail = copy.deepcopy(tail)
        self.kwargs = copy.deepcopy(kwargs)
        self.description = description
        self.is_default = phase == "default"
        self.family = family

    def replay(self, inputs, core):
        args = [*inputs, *copy.deepcopy(self.tail)]
        kwargs = copy.deepcopy(self.kwargs)
        if self.family == "spatial":
            if len(args) > 3:
                args[3] = core["num_groups"]
            if len(args) > 4:
                args[4] = core["eps"]
            for key in ("num_groups", "groups", "eps"):
                if key in kwargs:
                    kwargs[key] = core["num_groups"] if key == "groups" else core[key]
        else:
            kwargs = {**core, **kwargs}
        return self.fn(*args, **kwargs)


class CallCapture:
    def __init__(self, module, family, core_keys=()):
        self.module = module
        self.family = family
        self.core_keys = set(core_keys)
        self.phase = "default"
        self.entries = {}
        self.originals = {}

    def __enter__(self):
        for name, value in vars(self.module).copy().items():
            if (
                isinstance(value, types.FunctionType)
                and value.__module__ == self.module.__name__
                and name != "run_ablation_suite"
            ):
                self.originals[name] = value
                setattr(self.module, name, self._wrap(name, value))
        return self

    def __exit__(self, *exc):
        for name, value in self.originals.items():
            setattr(self.module, name, value)

    def _wrap(self, name, fn):
        signature = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            result = None
            try:
                result = fn(*args, **kwargs)
                return result
            finally:
                self._record(name, fn, signature, args, kwargs, result)

        return wrapper

    def _record(self, name, fn, signature, args, kwargs, result):
        # Normalize positional and keyword calls without changing the callable.
        # Signature shape never establishes a behavioral verdict.
        bound = signature.bind(*args, **kwargs)
        args, kwargs = bound.args, bound.kwargs
        count = 3 if self.family == "spatial" else 16
        if len(args) < count or not all(isinstance(v, torch.Tensor) for v in args[:count]):
            return_record = False
        elif self.family == "spatial":
            return_record = args[0].ndim == 4 and args[1].ndim == args[2].ndim == 1
            if len(args) > 3 and not isinstance(args[3], int):
                return_record = False
            if len(args) > 4 and not isinstance(args[4], (int, float)):
                return_record = False
        else:
            return_record = args[0].ndim == 2
        tensor = output_tensor(result)
        valid_output = (
            return_record
            and tensor is not None
            and (self.family != "spatial" or tensor.shape == args[0].shape)
        )
        if return_record and (valid_output or name in self.entries):
            tail = args[count:]
            options = {k: v for k, v in kwargs.items() if k not in self.core_keys}
            try:
                description = _small({"args": tail, "kwargs": options}, [128])
            except ValueError:
                return
            key = json.dumps(description, sort_keys=True)
            if name not in self.entries and len(self.entries) >= 32:
                return
            records = self.entries.setdefault(name, {})
            if key in records:
                records[key].is_default |= self.phase == "default"
            elif len(records) < 16 and len(self.entries) <= 32:
                records[key] = Control(fn, tail, options, description, self.phase, self.family)


def invoke_suite(module, family, inputs, core, sparse_names=()):
    """Bind CPU inputs where supported; this introspection never awards a score."""
    fn = module.run_ablation_suite
    signature = inspect.signature(fn)
    if family == "spatial":
        attempts = [(tuple(inputs), core), ((), {"device": "cpu"}), ((), {})]
    else:
        payload = {**dict(zip(sparse_names, inputs)), **core}
        attempts = [((), {"inputs": payload}), ((), {"device": "cpu"}), ((), {})]
    for args, kwargs in attempts:
        try:
            signature.bind(*args, **kwargs)
        except TypeError:
            continue
        return fn(*args, **kwargs)
    raise UnsupportedCPUExecution("Ablation suite has no supported CPU input binding")
