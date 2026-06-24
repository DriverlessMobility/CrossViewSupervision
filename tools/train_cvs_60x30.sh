#!/usr/bin/env bash
set -e

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
python tools/train.py \
    plugin/configs/cvs_60x30_24e.py \
    --no-validate \
    --deterministic \
    --work-dir work_dirs/cvs_60x30 \
    --cfg-options load_from=teachers/60x30_teacher.pth \
    "$@"
