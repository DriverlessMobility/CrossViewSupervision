#!/usr/bin/env bash
set -e

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
python tools/train.py \
    plugin/configs/cvs_100x50_24e.py \
    --no-validate \
    --deterministic \
    --work-dir work_dirs/cvs_100x50 \
    --cfg-options load_from=teachers/100x50_teacher.pth \
    "$@"
