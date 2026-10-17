# ruff: noqa: E402
# Platform registration and worker patches must precede dependent module imports.
import json
import os
import pathlib
import runpy
import sys
import time

if "RANK" not in os.environ:
    launcher = pathlib.Path(__file__).with_name("test_idle_dp_npu.py")
    sys.argv = [str(launcher), str(pathlib.Path(__file__).resolve()), *sys.argv[1:]]
    runpy.run_path(str(launcher), run_name="__main__")
    raise SystemExit(0)

import vllm_ascend

vllm_ascend.__path__.append("/vllm-workspace/vllm-ascend/vllm_ascend")
rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
os.environ.update(
    VLLM_DP_RANK=str(rank),
    VLLM_DP_RANK_LOCAL=str(local_rank),
    VLLM_DP_SIZE="2",
    VLLM_DP_MASTER_IP="127.0.0.1",
    VLLM_DP_MASTER_PORT=os.environ["MINDIE_BENCH_CONTRACT_DP_PORT"],
    VLLM_USE_V2_MODEL_RUNNER="0",
    VLLM_NO_USAGE_STATS="1",
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
)
import torch
import torch_npu  # noqa: F401 - register the real NPU backend.
from vllm.platforms import current_platform

current_platform.pre_register_and_update()
from vllm_ascend.utils import adapt_patch

adapt_patch()
from vllm.config import set_current_vllm_config
from vllm.engine.arg_utils import EngineArgs
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig
from vllm_ascend.worker.worker import NPUWorker

modelpath = pathlib.Path(os.environ["MINDIE_BENCH_CONTRACT_ROOT"]) / "tiny-kimi-k3"
if rank == 0:
    modelpath.mkdir(exist_ok=True)
    conf = KimiLinearConfig(
        vocab_size=256,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=128,
        num_experts=16,
        num_experts_per_token=2,
        moe_intermediate_size=128,
        num_shared_experts=1,
        use_grouped_topk=False,
        max_position_embeddings=128,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        mla_use_nope=True,
        linear_attn_config={
            "kda_layers": [1],
            "full_attn_layers": [2],
            "num_heads": 2,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": True,
        },
    )
    conf.architectures = ["KimiK3ForCausalLM"]
    conf.save_pretrained(str(modelpath))
    (modelpath / "ready").write_text("ready\n")
else:
    deadline = time.monotonic() + 30
    while not (modelpath / "ready").exists():
        if time.monotonic() > deadline:
            raise RuntimeError("rank0 config not ready")
        time.sleep(0.05)
print(
    json.dumps(
        {
            "stage": "imports",
            "rank": rank,
            "npu_count": torch.npu.device_count(),
        }
    ),
    flush=True,
)
args = EngineArgs(
    model=str(modelpath),
    tokenizer=str(modelpath),
    skip_tokenizer_init=True,
    load_format="dummy",
    dtype="bfloat16",
    max_model_len=64,
    max_num_seqs=2,
    max_num_batched_tokens=16,
    gpu_memory_utilization=0.1,
    num_gpu_blocks_override=8,
    tensor_parallel_size=1,
    data_parallel_size=2,
    data_parallel_size_local=1,
    data_parallel_rank=rank,
    distributed_executor_backend="external_launcher",
    enforce_eager=True,
    enable_prefix_caching=False,
    async_scheduling=False,
    additional_config={"enable_cpu_binding": False, "weight_nz_mode": 0},
    compilation_config={"mode": 0, "cudagraph_mode": "NONE"},
)
config = args.create_engine_config()
print(
    json.dumps(
        {
            "stage": "config",
            "rank": rank,
            "world_size": config.parallel_config.world_size,
            "dp": config.parallel_config.data_parallel_size,
            "tp": config.parallel_config.tensor_parallel_size,
            "mrv2": config.use_v2_model_runner,
        }
    ),
    flush=True,
)
worker = None
verdict = True
business_completed = False
try:
    with set_current_vllm_config(config):
        worker = NPUWorker(
            config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method="env://",
            is_driver_worker=True,
        )
        worker.init_device()
        print(
            json.dumps(
                {
                    "stage": "worker_init",
                    "rank": rank,
                    "device": str(worker.device),
                }
            ),
            flush=True,
        )
        worker.load_model()
        print(
            json.dumps(
                {
                    "stage": "model_loaded",
                    "rank": rank,
                    "model": type(worker.model_runner.model).__name__,
                    "layers": len(worker.model_runner.model.model.layers),
                }
            ),
            flush=True,
        )
        from vllm.v1.core.kv_cache_utils import get_kv_cache_configs

        spec = worker.get_kv_cache_spec()
        kv = get_kv_cache_configs(config, [spec], [16 * 1024 * 1024])[0]
        worker.initialize_from_config(kv)
        print(
            json.dumps(
                {
                    "stage": "cache_initialized",
                    "rank": rank,
                    "num_blocks": kv.num_blocks,
                    "groups": [
                        {"layers": g.layer_names, "spec": type(g.kv_cache_spec).__name__}
                        for g in kv.kv_cache_groups
                    ],
                }
            ),
            flush=True,
        )
        from vllm.config import CUDAGraphMode
        from vllm.distributed import get_dp_group
        from vllm.v1.core.sched.output import SchedulerOutput

        runner = worker.model_runner
        original_step = runner._dummy_run
        cases = json.loads(os.environ["MINDIE_BENCH_CONTRACT_CASES"])
        checks = 0
        for case_id, (stale_id, token_id, conv_value, ssm_value) in enumerate(cases):
            for gid in range(len(kv.kv_cache_groups)):
                runner.input_batch.block_table[gid].add_row([stale_id], 0)
            runner.input_batch.block_table.commit_block_table(1)
            runner.input_ids.gpu[:1].fill_(token_id)
            runner.positions[:1].zero_()
            gdn = config.compilation_config.static_forward_context["model.layers.0.self_attn"]
            conv, ssm = gdn.kv_cache
            if case_id == 0:
                print(
                    json.dumps(
                        {
                            "stage": "actual_state_fixture",
                            "rank": rank,
                            "layer_class": type(gdn).__name__,
                            "conv_shape": list(conv.shape),
                            "ssm_shape": list(ssm.shape),
                        }
                    ),
                    flush=True,
                )
            with torch.inference_mode():
                conv.fill_(conv_value)
                ssm.fill_(ssm_value)
            before_conv = conv.cpu().clone()
            before_ssm = ssm.cpu().clone()
            invoked = False

            def _eager_step(*args, **kwargs):
                global invoked
                invoked = True
                kwargs["force_attention"] = True
                return original_step(*args, **kwargs)

            if rank == 1:
                runner._dummy_run = _eager_step
                runner.execute_model(SchedulerOutput.make_empty())
            else:
                original_step(1, force_attention=True, cudagraph_runtime_mode=CUDAGraphMode.NONE)
            torch.npu.synchronize()
            after_conv = conv.cpu()
            after_ssm = ssm.cpu()
            conv_changed = int(torch.count_nonzero(after_conv != before_conv))
            ssm_changed = int(torch.count_nonzero(after_ssm != before_ssm))
            untouched = bool(
                torch.equal(after_conv[4], before_conv[4]) and torch.equal(after_ssm[4], before_ssm[4])
            )
            passed = (
                (conv_changed == 0 and ssm_changed == 0 and invoked)
                if rank == 1
                else (conv_changed > 0 and ssm_changed > 0)
            )
            passed = passed and untouched
            collective = torch.tensor([rank + 1], device=worker.device, dtype=torch.int32)
            torch.distributed.all_reduce(collective, group=get_dp_group().device_group)
            torch.npu.synchronize()
            hccl_sum = int(collective.item())
            passed = passed and hccl_sum == 3
            record = {
                "stage": "state_result",
                "case": case_id,
                "stale_block_id": stale_id,
                "input_token": token_id,
                "rank": rank,
                "role": "idle" if rank == 1 else "active",
                "conv_changed_elements": conv_changed,
                "ssm_changed_elements": ssm_changed,
                "unrelated_state_row4_unchanged": untouched,
                "hccl_rank_sum": hccl_sum,
                "pass": passed,
            }
            results = [None, None]
            torch.distributed.all_gather_object(results, record, group=get_dp_group().cpu_group)
            if rank == 0:
                for item in results:
                    os.write(1, (json.dumps(item) + "\n").encode())
            verdict = verdict and all(item["pass"] for item in results)
            checks += len(results)
        business_completed = True
        if rank == 0:
            os.write(
                1,
                (
                    json.dumps(
                        {
                            "reward": 1.0 if verdict else 0.0,
                            "case_count": checks,
                            "status": "passed" if verdict else "idle_state_corruption",
                        }
                    )
                    + "\n"
                ).encode(),
            )
finally:
    primary = sys.exception()
    cleanup_errors = []

    def cleanup_step(name, action):
        try:
            action()
        except BaseException as exc:
            cleanup_errors.append((name, exc))

    if worker is not None:
        cleanup_step("worker_shutdown", worker.shutdown)
    from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel

    cleanup_step("model_parallel", destroy_model_parallel)
    cleanup_step("distributed_environment", destroy_distributed_environment)
    if torch.distributed.is_initialized():
        cleanup_step("process_group", torch.distributed.destroy_process_group)
    cleanup_step("npu_empty_cache", torch.npu.empty_cache)
    print(
        json.dumps(
            {
                "stage": "cleanup",
                "rank": rank,
                "errors": [
                    {"stage": name, "type": type(exc).__name__, "message": str(exc)}
                    for name, exc in cleanup_errors
                ],
            }
        ),
        flush=True,
    )
    receipt = {
        "rank": rank,
        "business_completed": business_completed,
        "passed": verdict,
        "case_count": len(json.loads(os.environ["MINDIE_BENCH_CONTRACT_CASES"])),
        "primary_error_type": type(primary).__name__ if primary is not None else None,
        "cleanup_errors": [name for name, _ in cleanup_errors],
    }
    receipt_root = pathlib.Path(os.environ["MINDIE_BENCH_CONTRACT_ROOT"])
    temporary_receipt = receipt_root / f"rank-{rank}.pending.json"
    temporary_receipt.write_text(json.dumps(receipt))
    temporary_receipt.replace(receipt_root / f"rank-{rank}.completed.json")
    if cleanup_errors:
        if primary is not None:
            for name, exc in cleanup_errors:
                primary.add_note(name + ": " + repr(exc))
        else:
            first = cleanup_errors[0][1]
            for name, exc in cleanup_errors[1:]:
                first.add_note(name + ": " + repr(exc))
            raise first
# Let both ranks acknowledge cleanup before torchrun observes a semantic failure.
if business_completed:
    deadline = time.monotonic() + 5
    while not all((receipt_root / f"rank-{index}.completed.json").is_file() for index in (0, 1)):
        if time.monotonic() > deadline:
            raise RuntimeError("peer cleanup acknowledgement missing")
        time.sleep(0.02)
if not verdict:
    if rank == 1:
        sys.stderr.write("AssertionError: GDN recurrent state mutated during idle DP step\n")
        sys.stderr.flush()
    raise SystemExit(1)

