#!/usr/bin/env bash
set -e

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
python tools/test.py \
    plugin/configs/nusc_newsplit_480_100x50_24e.py \
    work_dirs/cvs_100x50/latest.pth \
    --work-dir work_dirs/cvs_100x50 \
    --eval \
    "$@"
