#!/bin/bash

set -x

WEIGHT_DIR=${1:-lingbot-world-v2-14b-causal-fast}
FRAME=${2:-361}
ASSETS_DIR=${3:-}

# 1.3B has 12 heads, so Ulysses must divide 12. The reference setting is
# 2 GPUs (ulysses_size=2), matching the 1.3B causal-ODE CP=1 recipe.
# 14B has 40 heads and uses 8 GPUs (ulysses_size=8).
WEIGHT_LOWER=$(basename "${WEIGHT_DIR}" | tr '[:upper:]' '[:lower:]')
if [[ "${WEIGHT_LOWER}" == *"1.3b"* || "${WEIGHT_LOWER}" == *"1p3b"* ]]; then
    TASK="i2v-1.3B"
    NPROC=2
    GPUS="0,1"
else
    TASK="i2v-A14B"
    NPROC=8
    GPUS="0,1,2,3,4,5,6,7"
fi

ASSETS_ARGS=()
if [[ -n "${ASSETS_DIR}" ]]; then
    ASSETS_ARGS+=(--assets_dir "${ASSETS_DIR}")
fi

# causal_fast (default) — distilled few-step model
CUDA_VISIBLE_DEVICES=${GPUS} torchrun --nproc_per_node=${NPROC} generate.py \
                           --task ${TASK} \
                           --size 480*832 \
                           --ckpt_dir ${WEIGHT_DIR} \
                           --image examples/03/image.jpg \
                           --action_path examples/03 \
                           --dit_fsdp \
                           --t5_fsdp \
                           --ulysses_size ${NPROC} \
                           --frame_num ${FRAME} \
                           --prompt "A serene lakeside scene with a lone tree standing in calm water, surrounded by distant snow-capped mountains under a bright blue sky with drifting white clouds — gentle ripples reflect the tree and sky, creating a tranquil, meditative atmosphere." \
                           --local_attn_size 18 \
                           --sink_size 6 \
                           "${ASSETS_ARGS[@]}"
