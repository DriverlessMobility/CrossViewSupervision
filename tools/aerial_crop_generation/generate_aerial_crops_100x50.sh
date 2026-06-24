#!/usr/bin/env bash
# Ego-referenced aerial crops (100x50) for CVS fusion + CVS training.
# Run from the CrossViewSupervision repo root:
#   bash tools/aerial_crop_generation/generate_aerial_crops_100x50.sh
# Uses AID4AD's own annotation_files/; writes crops into this repo's datasets/ folder.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
AID4AD_TOOLS="${REPO_ROOT}/../AID4AD/AID4AD_tools"
OUT_DIR="${REPO_ROOT}/datasets/AID4AD_ego_referenced_100x50"

mkdir -p "${OUT_DIR}"

python3 "${AID4AD_TOOLS}/scripts/02_export_frames.py" \
    --annotation_pickle_path "${AID4AD_TOOLS}/annotation_files" \
    --basemap_path "${AID4AD_TOOLS}/../nuScenes/maps/basemap" \
    --satmap_path "${AID4AD_TOOLS}/SatImgTiles" \
    --offset_grid_dir "${AID4AD_TOOLS}/offset_grid_data" \
    --per_frame_output_path "${OUT_DIR}" \
    --splits train val \
    --reference_frame ego \
    --crop_size_meters 100 50 \
    --final_image_size_pixels 400 200
