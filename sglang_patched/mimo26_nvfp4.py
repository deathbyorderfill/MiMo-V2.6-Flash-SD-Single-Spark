"""mimo26: load-time NVFP4 W4A16 for MiMo's dense attention GEMMs (qkv_proj block-FP8, o_proj block-FP8 in the -o8
variant). Kernel/packing/dequant code is copied verbatim from opt3/nvfp4_dense (dense2_patch.py + nvfp4_dequant.py);
the only new piece is `_mimo26_dequant_block_fp8` + the hook, because MiMo's source weights are block-FP8, not bf16.
Env: SGLANG_MIMO_DENSE_NVFP4=qkv,o  (groups); SGLANG_DENSE_NVFP4_BIGM=bf16|fp8|off, _ROWS (default 1024);
     SGLANG_DENSE_NVFP4_CODES_DIR=<dir> for GPTQ sidecar codes (<module>.pt)."""
import logging, os
import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)



@triton.jit
def _nvfp4_e2m1(c):
    """4-bit code (sign, e1, e0, m) -> value: e==0 -> m*0.5, else (1 + m/2) * 2^(e-1)."""
    e = (c >> 1) & 3
    m = (c & 1).to(tl.float32)
    mag = tl.where(e == 0, 0.5 * m, (1.0 + 0.5 * m) * tl.exp2((e - 1).to(tl.float32)))
    return tl.where(((c >> 3) & 1) == 1, -mag, mag)


@triton.jit
def _nvfp4_dequant_kernel(
    codes_ptr, sf_ptr, out_ptr, rowscale_ptr, alpha_ptr,
    N, K,
    stride_cn, stride_sn, stride_on,
    OUT_FP8: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """codes [N, K/2] uint8 (low nibble = even k), sf [N, K/16] e4m3-as-uint8, alpha fp32 [1].
    OUT_FP8=False: out bf16 [N, K] = code * sf * alpha.
    OUT_FP8=True:  out e4m3 [N, K] = code * sf * alpha / rowscale[n] (rowscale chosen so the
    row maximum lands on 448; see _nvfp4_row_scale)."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_kb = pid_k * (BLOCK_K // 2) + tl.arange(0, BLOCK_K // 2)      # packed byte index
    offs_kg = pid_k * (BLOCK_K // 16) + tl.arange(0, BLOCK_K // 16)    # scale group index
    n_mask = offs_n < N
    packed = tl.load(codes_ptr + offs_n[:, None] * stride_cn + offs_kb[None, :],
                     mask=n_mask[:, None] & (offs_kb[None, :] < K // 2), other=0)
    sf_u8 = tl.load(sf_ptr + offs_n[:, None] * stride_sn + offs_kg[None, :],
                    mask=n_mask[:, None] & (offs_kg[None, :] < K // 16), other=0)
    sf = sf_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)          # [BN, BK/16]
    alpha = tl.load(alpha_ptr)
    # expand each scale over its 8 packed bytes: [BN, BK/16] -> [BN, BK/2]
    sf8 = tl.reshape(tl.broadcast_to(sf[:, :, None], (BLOCK_N, BLOCK_K // 16, 8)), (BLOCK_N, BLOCK_K // 2))
    lo = (packed & 0xF).to(tl.int32)
    hi = ((packed >> 4) & 0xF).to(tl.int32)
    v_lo = _nvfp4_e2m1(lo) * sf8 * alpha
    v_hi = _nvfp4_e2m1(hi) * sf8 * alpha
    if OUT_FP8:
        rs = tl.load(rowscale_ptr + offs_n, mask=n_mask, other=1.0)
        v_lo = v_lo / rs[:, None]
        v_hi = v_hi / rs[:, None]
    # interleave low/high back to k = 2i, 2i+1
    v = tl.reshape(tl.join(v_lo, v_hi), (BLOCK_N, BLOCK_K))
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    out_ptrs = out_ptr + offs_n[:, None] * stride_on + offs_k[None, :]
    o_mask = n_mask[:, None] & (offs_k[None, :] < K)
    if OUT_FP8:
        tl.store(out_ptrs, v.to(tl.float8e4nv), mask=o_mask)
    else:
        tl.store(out_ptrs, v.to(tl.bfloat16), mask=o_mask)


def _nvfp4_row_scale(sf_lin_u8, alpha, fp8_max=448.0):
    """Per-row fp32 scale for the fp8 scratch: the largest |value| in a row is at most
    6 * max_k(sf) * alpha (e2m1 max is 6), so dividing by that /448 never clips."""
    sf = sf_lin_u8.view(torch.float8_e4m3fn).float()
    return (sf.amax(dim=1) * 6.0 * float(alpha) / fp8_max).clamp(min=1e-30).contiguous()


def nvfp4_dequant(codes, sf_lin_u8, alpha, out_dtype, rowscale=None, out=None):
    """codes [N, K/2] uint8, sf_lin_u8 [N, K/16] uint8 (e4m3 bits, linear layout),
    alpha fp32 [1] tensor -> [N, K] bf16, or e4m3 scaled by rowscale [N] (fp32)."""
    N, K2 = codes.shape
    K = K2 * 2
    fp8 = out_dtype == torch.float8_e4m3fn
    if out is None:
        out = torch.empty(N, K, dtype=out_dtype, device=codes.device)
    if fp8:
        assert rowscale is not None
    BLOCK_N, BLOCK_K = 32, 256 if K % 256 == 0 else 128
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K))
    _nvfp4_dequant_kernel[grid](
        codes, sf_lin_u8, out, rowscale if fp8 else out, alpha,
        N, K, codes.stride(0), sf_lin_u8.stride(0), out.stride(0),
        OUT_FP8=fp8, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=4,
    )
    return out

_NVFP4_FP8_MAX, _NVFP4_E2M1_MAX = 448.0, 6.0


def _nvfp4_sf_layout(flashinfer):
    layout = getattr(flashinfer, "SfLayout", None)
    if layout is None:
        from flashinfer.fp4_quantization import SfLayout as layout
    return layout.layout_128x4



class _LoadTimeNvfp4LinearMethod:
    """quant_method replacement: weight / weight_scale / weight_alpha hold the
    backend-prepared tensors from flashinfer.prepare_bf16_fp4_weights."""

    def __init__(self, name, backend):
        self.name = name
        self.backend = backend

    def process_weights_after_loading(self, layer):
        return None

    def create_weights(self, *a, **k):
        raise RuntimeError("load-time NVFP4 method is installed after weights exist")

    def apply(self, layer, x, bias=None):
        import flashinfer

        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        if x2.dtype != torch.bfloat16:
            x2 = x2.to(torch.bfloat16)
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        if _NVFP4_BIGM != "off" and x2.shape[0] >= _NVFP4_BIGM_ROWS and hasattr(layer, "weight_codes"):
            if _NVFP4_BIGM == "bf16":
                w = nvfp4_dequant(layer.weight_codes, layer.weight_sf_lin, layer.weight_alpha, torch.bfloat16)
                out = torch.matmul(x2, w.t())
            else:
                from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear

                w = nvfp4_dequant(layer.weight_codes, layer.weight_sf_lin, layer.weight_alpha,
                                  torch.float8_e4m3fn, rowscale=layer.weight_rowscale)
                out = apply_fp8_linear(x2, w.t(), layer.weight_rowscale.reshape(-1, 1),
                                       input_scale=None, use_per_token_if_dynamic=True)
        else:
            out = flashinfer.mm_bf16_fp4(x2, layer.weight, layer.weight_scale, layer.weight_alpha,
                                         backend=self.backend)
        if bias is not None:
            out = out + bias
        return out.reshape(*shp[:-1], out.shape[-1])

    def embedding(self, layer, x):
        raise NotImplementedError



_NVFP4_SIDECAR_USED = []
_NVFP4_BIGM = os.environ.get("SGLANG_DENSE_NVFP4_BIGM", "bf16").strip().lower() or "bf16"
_NVFP4_BIGM_ROWS = int(os.environ.get("SGLANG_DENSE_NVFP4_BIGM_ROWS", "1024") or 1024)


def _nvfp4_pack_weight(w, backend, keep_linear=True):
    """bf16 [N, K] -> (prepared packed codes, prepared block scales, alpha) for mm_bf16_fp4,
    plus (codes [N, K/2] u8, linear block scales [N, K/16] u8, rowscale fp32 [N]) for the
    large-M dequant path when keep_linear."""
    import flashinfer

    N, K = w.shape
    wf = w.float()
    g = (wf.abs().amax() / (_NVFP4_FP8_MAX * _NVFP4_E2M1_MAX)).clamp(min=1e-30)
    del wf
    ginv = (1.0 / g).reshape(1).to(torch.float32)
    alpha = g.reshape(1).to(torch.float32).contiguous()
    q, sf = flashinfer.nvfp4_quantize(w.contiguous(), ginv, sfLayout=_nvfp4_sf_layout(flashinfer), do_shuffle=False)
    prepared = flashinfer.prepare_bf16_fp4_weights(q.view(torch.uint8), sf, alpha, backend=backend)
    if not keep_linear or _NVFP4_BIGM == "off":
        return prepared, None
    _, sf_lin = flashinfer.nvfp4_quantize(w.contiguous(), ginv, sfLayout=flashinfer.SfLayout.layout_linear, do_shuffle=False)
    codes = q.view(torch.uint8).reshape(N, K // 2)
    if codes.data_ptr() != prepared[0].data_ptr():
        codes = codes.contiguous()          # prepare() made its own copy: keep the canonical one too
    sf_lin = sf_lin.view(torch.uint8).reshape(N, K // 16).contiguous()
    rowscale = _nvfp4_row_scale(sf_lin, alpha.item())
    return prepared, (codes, sf_lin, rowscale)



_NVFP4_CODES_DIR = os.environ.get("SGLANG_DENSE_NVFP4_CODES_DIR", "").strip()


def _nvfp4_load_sidecar(name, N, K, backend):
    """Precomputed codes (e.g. GPTQ-rounded, gptq_nvfp4.py): <dir>/<module>.pt with
    codes u8 [N, K/2], sf_lin u8 [N, K/16] (e4m3 bits), alpha fp32 [1]. Returns the same
    tuple as _nvfp4_pack_weight or None."""
    import flashinfer

    if not _NVFP4_CODES_DIR:
        return None
    path = os.path.join(_NVFP4_CODES_DIR, name + ".pt")
    if not os.path.exists(path):
        return None
    d = torch.load(path, map_location="cuda")
    codes, sf_lin, alpha = d["codes"].contiguous(), d["sf_lin"].contiguous(), d["alpha"].float().reshape(1).contiguous()
    assert tuple(codes.shape) == (N, K // 2) and tuple(sf_lin.shape) == (N, K // 16), (name, codes.shape, sf_lin.shape)
    sf_sw = flashinfer.nvfp4_block_scale_interleave(sf_lin)
    prepared = flashinfer.prepare_bf16_fp4_weights(codes, sf_sw, alpha, backend=backend)
    if _NVFP4_BIGM == "off":
        return prepared, None
    return prepared, (codes, sf_lin, _nvfp4_row_scale(sf_lin, alpha.item()))


def _nvfp4_convert_module(name, mod, backend):
    w = mod.weight.data
    if w.dtype not in (torch.bfloat16, torch.float16):
        return 0
    N, K = w.shape
    if N % 16 or K % 16:
        return 0
    side = _nvfp4_load_sidecar(name, N, K, backend)
    if side is not None:
        (b_p, s_p, a_p), lin = side
        _NVFP4_SIDECAR_USED.append(name)
    else:
        (b_p, s_p, a_p), lin = _nvfp4_pack_weight(w.to(torch.bfloat16), backend)
    saved = w.numel() * w.element_size() - (b_p.numel() * b_p.element_size()
                                            + s_p.numel() * s_p.element_size())
    mod.weight = torch.nn.Parameter(b_p, requires_grad=False)
    mod.weight_scale = torch.nn.Parameter(s_p, requires_grad=False)
    mod.weight_alpha = torch.nn.Parameter(a_p, requires_grad=False)
    if lin is not None:
        codes, sf_lin, rowscale = lin
        mod.weight_codes = codes
        mod.weight_sf_lin = sf_lin
        mod.weight_rowscale = rowscale
        extra = sf_lin.numel() + rowscale.numel() * 4 + (0 if codes.data_ptr() == b_p.data_ptr() else codes.numel())
        saved -= extra
    mod.quant_method = _LoadTimeNvfp4LinearMethod(name, backend)
    return saved




def _mimo26_dequant_block_fp8(w, scale_inv, block=128):
    """block-FP8 [N, K] e4m3 + scale_inv [ceil(N/b), ceil(K/b)] fp32 -> bf16 [N, K]"""
    N, K = w.shape
    s = scale_inv.to(torch.float32).repeat_interleave(block, 0)[:N].repeat_interleave(block, 1)[:, :K]
    return (w.to(torch.float32) * s).to(torch.bfloat16)


def _mimo26_dense_to_nvfp4(model):
    spec = os.environ.get("SGLANG_MIMO_DENSE_NVFP4", "").strip()
    if not spec:
        return
    backend = os.environ.get("SGLANG_DENSE_NVFP4_BACKEND", "cute-dsl").strip() or "cute-dsl"
    groups = {g.strip() for g in spec.split(",") if g.strip()}
    suffixes = tuple(s for g, ss in {"qkv": ("self_attn.qkv_proj",), "o": ("self_attn.o_proj",)}.items() if g in groups for s in ss)
    converted = 0; saved = 0; skipped = []
    for name, mod in model.named_modules():
        if "mtp" in name or not any(name.endswith("." + s) for s in suffixes):
            continue
        w = getattr(mod, "weight", None)
        if w is None:
            continue
        if w.dtype == torch.float8_e4m3fn and hasattr(mod, "weight_scale_inv"):
            wb = _mimo26_dequant_block_fp8(w.data, mod.weight_scale_inv.data)
        elif w.dtype in (torch.bfloat16, torch.float16):
            wb = w.data.to(torch.bfloat16)
        else:
            skipped.append((name, str(w.dtype))); continue
        N, K = wb.shape
        if N % 16 or K % 16:
            skipped.append((name, "shape")); continue
        side = _nvfp4_load_sidecar(name, N, K, backend)
        if side is not None:
            (b_p, s_p, a_p), lin = side; _NVFP4_SIDECAR_USED.append(name)
        else:
            (b_p, s_p, a_p), lin = _nvfp4_pack_weight(wb, backend)
        before = w.numel() * w.element_size() + (mod.weight_scale_inv.numel() * 4 if hasattr(mod, "weight_scale_inv") else 0)
        mod.weight = torch.nn.Parameter(b_p, requires_grad=False)
        mod.weight_scale = torch.nn.Parameter(s_p, requires_grad=False)
        mod.weight_alpha = torch.nn.Parameter(a_p, requires_grad=False)
        if hasattr(mod, "weight_scale_inv"):
            mod.weight_scale_inv = None
        if lin is not None:
            mod.weight_codes, mod.weight_sf_lin, mod.weight_rowscale = lin
        mod.quant_method = _LoadTimeNvfp4LinearMethod(name, backend)
        converted += 1; saved += before - (b_p.numel() * b_p.element_size() + s_p.numel() * s_p.element_size())
        del wb
    torch.cuda.empty_cache()
    logger.info("mimo26 dense NVFP4 (%s): converted %d modules, saved %.2f GB, sidecar %d, skipped %s",
                spec, converted, saved / 1e9, len(_NVFP4_SIDECAR_USED), skipped[:4])


# ---- calibration capture (GPTQ inputs): SGLANG_MIMO_DENSE_CALIB_DIR=<dir> -- ported from opt3/nvfp4_dense stage 5.
# Dumps W/<module>.pt (the block-FP8 weight dequantised to bf16) at install, accumulates H = sum x^T x per module,
# and writes H.pt + H_DONE when <dir>/DUMP_NOW appears. Takes precedence over the NVFP4 conversion.
_CALIB = {"mods": {}}


class _CalibApply:
    def __init__(self, name, inner, K):
        self.name, self.inner = name, inner
        self.H = torch.zeros(K, K, dtype=torch.float32, device="cuda")
        self.n = 0

    def __getattr__(self, item):
        return getattr(self.inner, item)

    def apply(self, layer, x, bias=None):
        if not torch.cuda.is_current_stream_capturing():
            x2 = x.reshape(-1, x.shape[-1])
            if x2.shape[0] > 0:
                xf = x2.float()
                self.H.addmm_(xf.t(), xf)
                self.n += int(x2.shape[0])
        return self.inner.apply(layer, x, bias) if bias is not None else self.inner.apply(layer, x)


def _mimo26_install_dense_calib(model):
    d = os.environ.get("SGLANG_MIMO_DENSE_CALIB_DIR", "").strip()
    if not d:
        return False
    import threading, time as _time
    suffixes = ("self_attn.qkv_proj", "self_attn.o_proj")
    lr = os.environ.get("SGLANG_MIMO_DENSE_CALIB_LAYERS", "").strip()  # "a-b" inclusive; H is 268 MB per o_proj
    lo, hi = (int(x) for x in lr.split("-")) if lr else (0, 10 ** 6)
    os.makedirs(os.path.join(d, "W"), exist_ok=True)
    import re as _re
    for name, mod in model.named_modules():
        if "mtp" in name or not any(name.endswith("." + s) for s in suffixes):
            continue
        m = _re.search(r"layers\.(\d+)\.", name)
        if m is None or not (lo <= int(m.group(1)) <= hi):
            continue
        w = getattr(mod, "weight", None)
        if w is None or not hasattr(mod, "quant_method"):
            continue
        if w.dtype == torch.float8_e4m3fn and hasattr(mod, "weight_scale_inv"):
            wb = _mimo26_dequant_block_fp8(w.data, mod.weight_scale_inv.data)
        elif w.dtype in (torch.bfloat16, torch.float16):
            wb = w.data.to(torch.bfloat16)
        else:
            continue
        wp = os.path.join(d, "W", name + ".pt")
        if not os.path.exists(wp):
            torch.save({"weight": wb.detach().to("cpu")}, wp)
        del wb
        mod.quant_method = _CalibApply(name, mod.quant_method, w.shape[1])
        _CALIB["mods"][name] = mod
    logger.info("mimo26 dense-calib: capturing H for %d modules into %s (touch %s/DUMP_NOW to dump)", len(_CALIB["mods"]), d, d)

    def _watch():
        flag = os.path.join(d, "DUMP_NOW")
        while True:
            _time.sleep(5)
            if os.path.exists(flag):
                try:
                    torch.cuda.synchronize()
                    # one module at a time (268 MB each): a single 7.6 GB host copy tripped the MemAvailable watchdog
                    os.makedirs(os.path.join(d, "H"), exist_ok=True)
                    tot = 0
                    for n, m in _CALIB["mods"].items():
                        h = m.quant_method.H.detach().to("cpu")
                        torch.save({"H": h, "n": m.quant_method.n}, os.path.join(d, "H", n + ".pt"))
                        tot += m.quant_method.n; del h
                    open(os.path.join(d, "H_DONE"), "w").write(str(tot))
                    logger.info("mimo26 dense-calib: dumped H for %d modules (per-module files in H/)", len(_CALIB["mods"]))
                except Exception as exc:
                    logger.warning("mimo26 dense-calib: dump failed: %s", exc)
                try:
                    os.remove(flag)
                except OSError:
                    pass

    threading.Thread(target=_watch, daemon=True).start()
    return True
