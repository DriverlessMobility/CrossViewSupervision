#!/usr/bin/env bash
set -e

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
python tools/test.py \
    plugin/configs/nusc_newsplit_480_60x30_24e.py \
    work_dirs/cvs_60x30/latest.pth \
    --work-dir work_dirs/cvs_60x30 \
    --eval \
    "$@"
