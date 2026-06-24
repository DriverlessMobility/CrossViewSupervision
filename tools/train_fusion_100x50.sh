#!/usr/bin/env bash
set -e

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
python tools/train.py \
    plugin/configs/aid4ad_480_100x50_24e.py \
    --no-validate \
    --deterministic \
    --work-dir work_dirs/fusion_100x50 \
    "$@"
