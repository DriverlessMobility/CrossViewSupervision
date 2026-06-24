#!/usr/bin/env bash
set -euo pipefail

bash tools/train_fusion_60x30.sh

python tools/extract_aerial_teacher.py \
    --work-dir work_dirs/fusion_60x30 \
    --output teachers/60x30_teacher.pth

bash tools/train_cvs_60x30.sh

bash tools/test_cvs_60x30.sh
