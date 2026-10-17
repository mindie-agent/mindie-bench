# Imports follow selection of the owned helper/source boundary.
# ruff: noqa: E402
"""Independent tensor contracts; replay source-pattern results are never a grade."""

import hashlib
import json
import math
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent))

import torch
from replay_cpu_boundary import load_sampler


def check(workspace: Path):
    binding = json.loads((Path(__file__).parent / "runtime-binding.json").read_text())
    if hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest() != binding["python"]:
        raise RuntimeError("CPU interpreter identity changed")
    if torch.__version__ != binding["torch"]:
        raise RuntimeError("CPU tensor runtime version changed")
    sampler, state, gathers = load_sampler(workspace)
    count = 0
    for dtype in [torch.float32, torch.float64]:
        for width in [8, 17, 31]:
            values = [
                [math.sin(column * 1.7 + row) + column / 9 for column in range(width)]
                for row in range(2)
            ]
            logits = torch.tensor(values, dtype=dtype)
            for limit, selected, expected_count in [
                (4096, width, width),
                (width + 1, width, width),
                (3, 3, 3),
                (None, None, width),
                (2, None, width),
            ]:
                k = None if selected is None else torch.full((2,), selected)
                result = sampler.apply_top_k_top_p(logits.clone(), k, None, top_k=limit)
                assert isinstance(result, tuple) and len(result) == 2, (
                    "reduce sampling must return both candidate tensors"
                )
                candidates, indices = result
                assert candidates.shape == indices.shape == (2, expected_count)
                assert indices.dtype == torch.int64
                for row in range(2):
                    ids = indices[row].tolist()
                    assert len(set(ids)) == expected_count
                    assert all(0 <= index < width for index in ids)
                    actual = candidates[row].tolist()
                    original = logits[row].tolist()
                    assert all(
                        math.isclose(value, original[index], rel_tol=1e-7, abs_tol=1e-7)
                        for value, index in zip(actual, ids, strict=True)
                    ), "candidate values must correspond to their actual source indices"
                    wanted = sorted(range(width), key=lambda index: original[index], reverse=True)[
                        :expected_count
                    ]
                    assert set(ids) == set(wanted), "candidate set lost a valid high-logit token"
                count += 1
            # Independently check that top-p filtering preserves the high-probability
            # candidate and discards the negligible tail; no copied gold algorithm.
            skewed = torch.full((2, width), -20.0, dtype=dtype)
            skewed[:, 2] = 20.0
            candidates, indices = sampler.apply_top_k_top_p(
                skewed, None, torch.full((2,), 0.6), top_k=4096
            )
            assert torch.equal(indices[torch.isfinite(candidates)], torch.tensor([2, 2]))
            count += 1
    state.enable_reduce_sample = False
    ordinary = torch.tensor([[0.5, -1.0, 2.0]])
    assert torch.equal(sampler.apply_top_k_top_p(ordinary.clone(), None, None), ordinary)
    assert len(gathers) >= count * 2
    return count + 1


try:
    cases = check(Path(os.environ.get("MINDIE_BENCH_WORKSPACE", os.getcwd())).resolve())
except AssertionError as exc:
    print("CPU shape/candidate contract failed: " + str(exc), file=sys.stderr)
    print(json.dumps({"reward": 0.0}))
    raise SystemExit(1)
except RuntimeError as exc:
    if "selected index k out of range" not in str(exc).casefold():
        raise
    print("Historical top-k shape contract failed: " + str(exc), file=sys.stderr)
    print(json.dumps({"reward": 0.0}))
    raise SystemExit(1)
except TypeError as exc:
    if "topk" not in str(exc).casefold() or "none" not in str(exc).casefold():
        raise
    print("Historical unfiltered candidate contract failed: " + str(exc), file=sys.stderr)
    print(json.dumps({"reward": 0.0}))
    raise SystemExit(1)
print(json.dumps({"cpu_contract_cases": cases}))
print(json.dumps({"reward": 1.0}))
