# ruff: noqa: E402
# Register the real platform before importing modules that bind its operators.
import json
import os
import pathlib
import sys
import time

import vllm_ascend

vllm_ascend.__path__.append("/vllm-workspace/vllm-ascend/vllm_ascend")
rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
phase = os.environ.get("MINDIE_STALE_PRIVATE_PHASE", "candidate")
os.environ.update(
    VLLM_USE_V2_MODEL_RUNNER="0",
    VLLM_NO_USAGE_STATS="1",
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
)
import torch
import torch_npu
from vllm.platforms import current_platform

current_platform.pre_register_and_update()
from vllm_ascend.utils import adapt_patch

adapt_patch()
from transformers import Qwen3MoeConfig
from vllm.config import CUDAGraphMode, set_current_vllm_config
from vllm.engine.arg_utils import EngineArgs
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.worker.worker import NPUWorker

root = pathlib.Path(os.environ["PROBE_ROOT"])
modelpath = root / "tiny-qwen3moe"
if rank == 0:
    modelpath.mkdir(exist_ok=True)
    model = Qwen3MoeConfig(
        vocab_size=256,
        hidden_size=1024,
        intermediate_size=1024,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=128,
        moe_intermediate_size=512,
        num_experts=16,
        num_experts_per_tok=8,
        decoder_sparse_step=1,
        norm_topk_prob=True,
    )
    model.architectures = ["Qwen3MoeForCausalLM"]
    model.save_pretrained(modelpath)
    (modelpath / "ready").write_text("ready")
else:
    deadline = time.monotonic() + 30
    while not (modelpath / "ready").exists():
        if time.monotonic() > deadline:
            raise RuntimeError("model config not ready")
        time.sleep(0.05)
base_options = {"enable_cpu_binding": False, "weight_nz_mode": 0, "enable_fused_mc2": 1}
if phase == "reference":
    deployment_options = dict(base_options, enable_fused_mc2=0)
else:
    from deployment import configure

    deployment_options = configure(base_options)
    if isinstance(deployment_options, dict) and isinstance(deployment_options.get("enable_fused_mc2"), bool):
        deployment_options = dict(deployment_options, enable_fused_mc2=int(deployment_options["enable_fused_mc2"]))
args = EngineArgs(
    model=str(modelpath),
    tokenizer=str(modelpath),
    skip_tokenizer_init=True,
    load_format="dummy",
    dtype="bfloat16",
    max_model_len=128,
    max_num_seqs=4,
    max_num_batched_tokens=64,
    gpu_memory_utilization=0.1,
    num_gpu_blocks_override=8,
    tensor_parallel_size=2,
    data_parallel_size=1,
    enable_expert_parallel=True,
    distributed_executor_backend="external_launcher",
    seed=17,
    enforce_eager=True,
    enable_prefix_caching=False,
    async_scheduling=False,
    additional_config=deployment_options,
    compilation_config={"mode": 0, "cudagraph_mode": "NONE"},
)
config = args.create_engine_config()
worker = None
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
        worker.load_model()
        from vllm.v1.core.kv_cache_utils import get_kv_cache_configs

        spec = worker.get_kv_cache_spec()
        kv = get_kv_cache_configs(config, [spec], [16 * 1024 * 1024])[0]
        worker.initialize_from_config(kv)
        runner = worker.model_runner
        for gid in range(len(kv.kv_cache_groups)):
            runner.input_batch.block_table[gid].add_row([1], 0)
        runner.input_batch.block_table.commit_block_table(1)
        from vllm.distributed import get_ep_group, get_tp_group

        for case_id, (tokens, token_id) in enumerate([(32, 7), (64, 53)]):
            runner.input_ids.gpu[:tokens].fill_(token_id)
            runner.positions[:tokens].zero_()
            with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU],
                record_shapes=True,
                schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
            ) as profile:
                hidden, _ = runner._dummy_run(
                    tokens, force_attention=True, cudagraph_runtime_mode=CUDAGraphMode.NONE
                )
                torch.npu.synchronize()
                profile.step()
            output = hidden.float().cpu()
            trace = root / (phase + "-trace-" + str(case_id) + "-" + str(rank) + ".json")
            profile.export_chrome_trace(str(trace))
            events = json.loads(trace.read_text())
            keys = sorted({event.get("name", "") for event in events})
            interesting = [
                key
                for key in keys
                if any(
                    x in key.lower()
                    for x in [
                        "dispatch",
                        "combine",
                        "grouped",
                        "moe",
                        "alltoall",
                        "all_to_all",
                        "all_gather",
                    ]
                )
            ]
            collective = torch.tensor([rank + 1], device=worker.device, dtype=torch.int32)
            torch.distributed.all_reduce(collective)
            torch.npu.synchronize()
            path = root / f"{phase}-case-{case_id}-rank-{rank}"
            torch.save(output, path.with_suffix(".pt"))
            receipt = dict(
                phase=phase,
                case_id=case_id,
                rank=rank,
                model=type(runner.model).__name__,
                config_effective=get_ascend_config().enable_fused_mc2,
                legacy_env=os.environ.get("VLLM_ASCEND_ENABLE_FUSED_MC2"),
                ep_size=get_ep_group().world_size,
                tp_size=get_tp_group().world_size,
                hccl_rank_sum=int(collective.cpu().item()),
                actual_operator_keys=interesting,
                output_shape=list(output.shape),
                output_finite=bool(torch.isfinite(output).all()),
                output_norm=float(output.norm()),
            )
            path.with_suffix(".json").write_text(json.dumps(receipt))
            print(json.dumps(receipt), flush=True)

finally:
    primary = sys.exception()
    cleanup_errors = []

    def cleanup(name, function):
        try:
            function()
        except BaseException as error:
            cleanup_errors.append((name, error))

    if worker is not None:
        cleanup("worker", worker.shutdown)
    from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel

    cleanup("parallel", destroy_model_parallel)
    cleanup("distributed", destroy_distributed_environment)
    if torch.distributed.is_initialized():
        cleanup("process_group", torch.distributed.destroy_process_group)
    cleanup("empty_cache", torch.npu.empty_cache)
    if cleanup_errors:
        if primary is None:
            primary = cleanup_errors.pop(0)[1]
        for name, error in cleanup_errors:
            primary.add_note(name + ": " + repr(error))
        raise primary
