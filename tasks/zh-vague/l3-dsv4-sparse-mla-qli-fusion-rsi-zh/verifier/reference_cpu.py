# [Role] Independently verify CPU tensor and leave-one-out fixture contracts.
# [Invariants] Candidate metrics and source spelling never establish correctness or speed.
# [Timeouts & Locks] The caller bounds this process; no locks or device contexts.
# [Side-Effect Policy] Loads candidate Python in this process; emits a JSON verdict only.

"""CPU interface fixture, not a QLI/SFA algorithm or native kernel reference."""

from __future__ import annotations

from types import SimpleNamespace

import torch


def _apply_rope_slice(x_rope: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x_rope.shape[-1] // 2
    x1 = x_rope[..., :half]
    x2 = x_rope[..., half:]
    c = cos.unsqueeze(1)[..., :half] if cos.ndim == 2 else cos[..., :half]
    s = sin.unsqueeze(1)[..., :half] if sin.ndim == 2 else sin[..., :half]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


def _op_q_fused(
    qr_i8: torch.Tensor,
    qr_scale: torch.Tensor,
    rms_weight: torch.Tensor,
    w_q_absorbed: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float = 1e-6,
    n_heads: int = 16,
    nope_dim: int = 512,
    rope_dim: int = 64,
) -> torch.Tensor:
    qr_fp32 = qr_i8.to(torch.float32) * qr_scale.to(torch.float32).unsqueeze(-1)
    var = qr_fp32.pow(2).mean(dim=-1, keepdim=True)
    normed = qr_fp32 * torch.rsqrt(var + eps) * rms_weight.to(torch.float32)
    q_full = torch.matmul(normed, w_q_absorbed.to(torch.float32).t())
    t_len = qr_i8.shape[0]
    q_full = q_full.view(t_len, n_heads, nope_dim + rope_dim)
    q_nope = q_full[..., :nope_dim]
    q_rope = _apply_rope_slice(q_full[..., nope_dim:], cos.to(torch.float32), sin.to(torch.float32))
    out = torch.cat([q_nope, q_rope], dim=-1).contiguous()
    return out


def _op_qli(
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
    weights: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    *,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seqused_k: torch.Tensor,
    topk: int,
    compress_ratio: int,
    layout_query: str = "TND",
    layout_key: str = "PA_BSND",
) -> torch.Tensor:
    if q_idx.stride(-1) != 1:
        raise RuntimeError("q_idx last dimension must be contiguous (stride(-1) == 1)")
    if layout_query != "TND" or layout_key != "PA_BSND":
        raise RuntimeError(f"Unsupported layouts: {layout_query}, {layout_key}")
    num_reqs = seqused_k.shape[0]
    device = q_idx.device
    slots = torch.arange(topk, device=device, dtype=torch.int32).unsqueeze(0).expand(num_reqs, topk)
    valid_blocks = torch.div(
        seqused_k.to(torch.int32) + compress_ratio - 1, compress_ratio, rounding_mode="floor"
    ).unsqueeze(-1)
    topk_indices = torch.where(slots < valid_blocks, slots, torch.full_like(slots, -1))
    return topk_indices


def _op_sfa(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    block_table: torch.Tensor,
    seqused_k: torch.Tensor,
    max_seqlen_k: int,
    sparse_mode: int = 3,
) -> torch.Tensor:
    if not isinstance(seqused_k, torch.Tensor) or seqused_k.dtype != torch.int32:
        raise RuntimeError("seqused_k must be an int32 device tensor")
    if int((block_table < 0).sum().item()) > 0:
        raise RuntimeError(
            "AICore invalid GM address: negative block id (-1) reached npu_sparse_flash_attention"
        )
    scale = (
        seqused_k.to(torch.float32).unsqueeze(-1).unsqueeze(-1) / float(max(max_seqlen_k, 1))
    ).clamp(min=0.1, max=1.0)
    return (q * scale).contiguous()


def install_cpu_contract() -> None:
    """Bind explicit CPU fixture operations; never describe these as native ops."""
    torch.ops._C_ascend = SimpleNamespace(
        fused_q_dequant_rmsnorm_rope_split=_op_q_fused,
        npu_quant_lightning_indexer=_op_qli,
        npu_sparse_flash_attention=_op_sfa,
    )
