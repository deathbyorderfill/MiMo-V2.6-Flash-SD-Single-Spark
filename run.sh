#!/usr/bin/env bash
# MiMo-V2.6-Flash-SD on ONE DGX Spark (GB10, 128 GB), served by sglang.
# Usage:  CKPT=/path/to/MiMo-V2.6-Flash-SD ./run.sh [extra sglang args]
# Env:    IMAGE  PORT (8936)  NAME (mimo-sd)  MEMFRAC (0.85)  CTX (262144)  MAXREQ (4)
#         STEPS (2) / DRAFT (3): speculative depth; STEPS=3 DRAFT=4 is faster on math, slower on prose.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
CKPT=${CKPT:?set CKPT=<dir with the MiMo-V2.6-Flash-SD files>}
IMAGE=${IMAGE:-lmsysorg/sglang:qwen38flashnext}
PORT=${PORT:-8936}; NAME=${NAME:-mimo-sd}; MEMFRAC=${MEMFRAC:-0.85}; CTX=${CTX:-262144}; MAXREQ=${MAXREQ:-4}
STEPS=${STEPS:-2}; DRAFT=${DRAFT:-3}
S=/sgl-workspace/sglang/python/sglang
[ -f "$CKPT/model.safetensors.index.json" ] || { echo "checkpoint not found at $CKPT"; exit 1; }
[ -d "$CKPT/dense_codes" ] || { echo "$CKPT/dense_codes missing"; exit 1; }
if pgrep -f 'sglang::scheduler' >/dev/null; then echo "an sglang scheduler is already running on this box"; exit 2; fi
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus all --network host --ipc=host --shm-size 32g \
  -e SGLANG_NVFP2_LM20=1 -e SGLANG_NVFP2_GROUP=32 -e SGLANG_ENABLE_SPLITKV_VERIFY=1 -e SGLANG_MIMO_LMHEAD_FP8=1 \
  -e SGLANG_MIMO_DENSE_NVFP4=qkv,o -e SGLANG_DENSE_NVFP4_CODES_DIR=/ckpt/dense_codes \
  -e SGLANG_MIMO_DRAFT_VOCAB=/ckpt/sidecars/draft_vocab_99.pt \
  -v "$CKPT":/ckpt:ro \
  -v "$HERE/sglang_patched/fused_moe_triton_kernels.py":$S/kernels/ops/moe/fused_moe_triton_kernels.py:ro \
  -v "$HERE/sglang_patched/fused_moe.py":$S/srt/layers/moe/moe_runner/triton_utils/fused_moe.py:ro \
  -v "$HERE/sglang_patched/triton.py":$S/srt/layers/moe/moe_runner/triton.py:ro \
  -v "$HERE/sglang_patched/modelopt_quant.py":$S/srt/layers/quantization/modelopt_quant.py:ro \
  -v "$HERE/sglang_patched/fp8.py":$S/srt/layers/quantization/fp8.py:ro \
  -v "$HERE/sglang_patched/mimo_v2.py":$S/srt/models/mimo_v2.py:ro \
  -v "$HERE/sglang_patched/mimo_v2_nextn.py":$S/srt/models/mimo_v2_nextn.py:ro \
  -v "$HERE/sglang_patched/mimo26_w8a16.py":$S/srt/models/mimo26_w8a16.py:ro \
  -v "$HERE/sglang_patched/mimo26_nvfp4.py":$S/srt/models/mimo26_nvfp4.py:ro \
  -v "$HERE/sglang_patched/swa_memory_pool.py":$S/srt/mem_cache/swa_memory_pool.py:ro \
  -v "$HERE/sglang_patched/triton_backend.py":$S/srt/layers/attention/triton_backend.py:ro \
  -v "$HERE/sglang_patched/verify_splitkv.py":$S/kernels/ops/attention/verify_splitkv.py:ro \
  "$IMAGE" \
  python3 -m sglang.launch_server --model-path /ckpt --trust-remote-code --skip-server-warmup \
  --moe-runner-backend triton --context-length "$CTX" --mem-fraction-static "$MEMFRAC" \
  --chunked-prefill-size 2048 --max-running-requests "$MAXREQ" --cuda-graph-max-bs "$MAXREQ" \
  --speculative-algorithm NEXTN --enable-multi-layer-eagle --speculative-num-steps "$STEPS" \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens "$DRAFT" \
  --kv-cache-dtype fp8_e4m3 --swa-full-tokens-ratio 0.1 \
  --host 0.0.0.0 --port "$PORT" --served-model-name mimo-sd "$@"
echo "launched $NAME on :$PORT (first load ~10 min; docker logs -f $NAME)"
