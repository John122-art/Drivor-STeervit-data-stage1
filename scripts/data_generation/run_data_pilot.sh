#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
case "$MODE" in
    prepare) ENV_ROOT="${DRIVOR_ENV_ROOT:-/mnt/zhaozc_workspace/envs/drivorpython}"; SCRIPT=prepare_navsim.py ;;
    generate) ENV_ROOT="${QWEN_ENV_ROOT:-/mnt/zhaozc_workspace/envs/mot_vega}"; SCRIPT=generate_captions.py ;;
    grounding) ENV_ROOT="${DRIVOR_ENV_ROOT:-/mnt/zhaozc_workspace/envs/drivorpython}"; SCRIPT=build_grounding.py ;;
    test) ENV_ROOT="${DRIVOR_ENV_ROOT:-/mnt/zhaozc_workspace/envs/drivorpython}"; SCRIPT=test_data_pipeline.py ;;
    validate) ENV_ROOT="${DRIVOR_ENV_ROOT:-/mnt/zhaozc_workspace/envs/drivorpython}"; SCRIPT=validate_pilot.py ;;
    *) printf '%s\n' 'Usage: bash scripts/data_generation/run_data_pilot.sh {prepare|generate|grounding|test|validate} [arguments]'; exit 2 ;;
esac
set +u
if [[ "$MODE" == generate ]]; then source /usr/local/PPU_SDK/envsetup.sh; fi
source "$ENV_ROOT/bin/activate"
set -u
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/drivor_data_pilot_mpl}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
mkdir -p "$HF_HOME"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/mnt/zhaozc_workspace/project/LQH/DrivoR-main-copy/dataset/maps}"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/nuplan-devkit${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"
exec "$ENV_ROOT/bin/python" "scripts/data_generation/$SCRIPT" "$@"
