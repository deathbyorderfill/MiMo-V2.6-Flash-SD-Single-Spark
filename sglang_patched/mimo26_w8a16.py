"""opt3 tier1b: W8A16 linear for decode-size batches (rows <= 16): weights FP8 e4m3 + one fp32 scale per output row,
cast to the activation dtype in registers (never materialised), scale applied to the dot result. Half the bytes of the
BF16 cuBLAS GEMM on a bandwidth-bound part. Optional fused SwiGLU MLP (gate_up -> silu*mul -> down) and sigmoid gate.
Plain Triton grid (one program per block of output rows); no atomics, so results are replay-deterministic."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

MAX_ROWS = 16
_FP8_MAX = 448.0


def quantize_rows_fp8(w: torch.Tensor):
    wf = w.float()
    s = wf.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / _FP8_MAX
    q = (wf / s).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn).contiguous()
    return q, s.reshape(-1).to(torch.float32).contiguous()


@triton.jit
def _w8a16_linear_kernel(x_ptr, w_ptr, s_ptr, out_ptr, K, N, num_rows,
                         ROWS: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, ROWS)
    mask_m = offs_m < num_rows
    n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = n < N
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((ROWS, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        mask_k = k < K
        xt = tl.load(x_ptr + offs_m[:, None] * K + k[None, :], mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(w_ptr + n[:, None] * K + k[None, :], mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        acc = tl.dot(xt, tl.trans(w.to(x_ptr.dtype.element_ty)), acc)
    s = tl.load(s_ptr + n, mask=mask_n, other=0.0)
    out = acc * s[None, :]
    tl.store(out_ptr + offs_m[:, None] * N + n[None, :], out.to(out_ptr.dtype.element_ty),
             mask=mask_m[:, None] & mask_n[None, :])


def w8a16_linear(x: torch.Tensor, wq: torch.Tensor, s: torch.Tensor, block_n: int = 0, block_k: int = 0) -> torch.Tensor:
    """x [rows<=16, K] bf16 contiguous, wq [N, K] fp8, s [N] fp32 -> [rows, N]"""
    rows, k = x.shape
    n = wq.shape[0]
    out = torch.empty((rows, n), dtype=x.dtype, device=x.device)
    if not block_n:  # measured 2026-09-20 (test_w8a16.py, rows 5): wide-K (2560) 16x512, narrow-K (640) 64x128
        block_n, block_k = (16, 512) if k >= 2048 else (64, 128)
    if rows == 0:
        return out
    _w8a16_linear_kernel[(triton.cdiv(n, block_n),)](x, wq, s, out, k, n, rows,
                                                      ROWS=MAX_ROWS, BLOCK_N=block_n, BLOCK_K=block_k)
    return out
