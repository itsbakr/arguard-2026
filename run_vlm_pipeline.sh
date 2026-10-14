#!/bin/zsh
# Post-training pipeline: checkpoint -> predictions -> thresholds -> submission zips.
#   ./run_vlm_pipeline.sh v2
# Runs the two GPU prediction passes sequentially (same container image, warm cache),
# then everything else is local and takes seconds.
set -e
TAG="${1:-v2}"
cd "$(dirname "$0")"

echo "── predict dev (calibration split) ──"
ARGUARD_GPU=H100 ./.venv/bin/modal run src/modal_vlm.py::predict --tag "$TAG" --split dev

echo "── predict dev_test (submission target) ──"
ARGUARD_GPU=H100 ./.venv/bin/modal run src/modal_vlm.py::predict --tag "$TAG" --split dev_test

echo "── fit thresholds on dev, decode dev_test (rule F) ──"
./.venv/bin/python -m src.vlm_decode --tag "$TAG" --target dev_test

echo "── package ──"
./.venv/bin/python -m src.build_submissions --run-id "awn_vlm_${TAG}"

echo "── validate with the official checker ──"
mkdir -p /tmp/argcheck_vlm && rm -rf /tmp/argcheck_vlm/*
./.venv/bin/python -c "
import zipfile
for s in ('a1','a2'):
    zipfile.ZipFile(f'submissions/{s}_prediction.zip').extractall(f'/tmp/argcheck_vlm/{s}')
"
(cd starter-kit/taskA && ../../.venv/bin/python format_checker/format_checker.py --subtask a1 --predictions /tmp/argcheck_vlm/a1/prediction.tsv)
(cd starter-kit/taskA && ../../.venv/bin/python format_checker/format_checker.py --subtask a2 --predictions /tmp/argcheck_vlm/a2/prediction.jsonl)

echo
echo "READY TO UPLOAD:"
echo "  submissions/a1_prediction.zip -> codabench comp 16909"
echo "  submissions/a2_prediction.zip -> codabench comp 16910"
