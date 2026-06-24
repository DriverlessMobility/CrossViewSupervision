#!/usr/bin/env bash
set -euo pipefail

bash tools/train_fusion_100x50.sh

python tools/extract_aerial_teacher.py \
    --work-dir work_dirs/fusion_100x50 \
    --output teachers/100x50_teacher.pth

bash tools/train_cvs_100x50.sh

bash tools/test_cvs_100x50.sh
