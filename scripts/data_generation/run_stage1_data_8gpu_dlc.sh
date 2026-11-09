#!/usr/bin/env bash
# Single DLC node; independent frozen Qwen worker on each device, as in the reference cache launcher.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-run}"
if [[ $# -gt 1 || ! "$MODE" =~ ^(run|--dry-run)$ ]]; then
    echo "Usage: bash $0 [--dry-run]" >&2
    exit 2
fi
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
SCENES_PER_RANK="${SCENES_PER_RANK:-1}"
DEEP_SCENES="${DEEP_SCENES:-20}"
SPLIT="${SPLIT:-navtrain}"
SUBSET="${SUBSET:-train}"
SEED="${SEED:-3407}"
DATA_ROOT="${DATA_ROOT:-/mnt/zhaozc_workspace/project/LQH/DrivoR-main-copy/dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/stage1_${SPLIT}_${SUBSET}_${NPROC_PER_NODE}gpu}"
QWEN_ENV_ROOT="${QWEN_ENV_ROOT:-/mnt/zhaozc_workspace/envs/mot_vega}"
DRIVOR_ENV_ROOT="${DRIVOR_ENV_ROOT:-/mnt/zhaozc_workspace/envs/drivorpython}"
for value in "$SCENES_PER_RANK" "$DEEP_SCENES" "$SEED"; do
    [[ "$value" =~ ^[0-9]+$ ]] || { echo "Scene counts/seed must be nonnegative integers" >&2; exit 2; }
done
[[ "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]] || { echo "NPROC_PER_NODE must be positive" >&2; exit 2; }
[[ "$SPLIT" =~ ^(navtrain|navtest)$ && "$SUBSET" =~ ^(train|val|all)$ ]] || exit 2
if [[ "$SPLIT" == navtest && "$SUBSET" != all ]]; then echo "navtest requires SUBSET=all" >&2; exit 2; fi
cd "$REPO_ROOT"
command=("$QWEN_ENV_ROOT/bin/python" -m torch.distributed.run --standalone --nnodes=1
    "--nproc_per_node=$NPROC_PER_NODE" "--log-dir=$OUTPUT_DIR/launcher_logs" --tee=3
    scripts/data_generation/stage1_worker.py --output "$OUTPUT_DIR" --data-root "$DATA_ROOT"
    --split "$SPLIT" --subset "$SUBSET" --scenes "$SCENES_PER_RANK" --seed "$SEED"
    --deep-scenes "$DEEP_SCENES" --prompt "$REPO_ROOT/scripts/data_generation/prompts/stage1_grounding_v1.txt"
    --drivor-python "$DRIVOR_ENV_ROOT/bin/python" --qwen-python "$QWEN_ENV_ROOT/bin/python")
if [[ "$MODE" == --dry-run ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
set +u
source /usr/local/PPU_SDK/envsetup.sh
source "$QWEN_ENV_ROOT/bin/activate"
set -u
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/nuplan-devkit${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/drivor_stage1_mpl}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$DATA_ROOT/maps}"
mkdir -p "$HF_HOME"
"$QWEN_ENV_ROOT/bin/python" -c 'import sys,torch; n=int(sys.argv[1]); assert torch.cuda.is_available() and torch.cuda.device_count()>=n, f"Need {n} visible devices, found {torch.cuda.device_count()}"' "$NPROC_PER_NODE"
"${command[@]}"
"$DRIVOR_ENV_ROOT/bin/python" -B scripts/data_generation/merge_stage1_shards.py --output "$OUTPUT_DIR" --world-size "$NPROC_PER_NODE"
