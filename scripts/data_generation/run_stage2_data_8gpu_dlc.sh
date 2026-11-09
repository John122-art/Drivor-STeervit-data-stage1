#!/usr/bin/env bash
# Offline current-image + navigation descriptions, one independent Qwen VLM per device.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-run}"
if [[ $# -gt 1 || ! "$MODE" =~ ^(run|--dry-run)$ ]]; then
    echo "Usage: bash $0 [--dry-run]" >&2; exit 2
fi
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
SCENES_PER_RANK="${SCENES_PER_RANK:-1}"
DEEP_SCENES="${DEEP_SCENES:-20}"
SEED="${SEED:-3407}"
DATA_SPLITS="${DATA_SPLITS:-train,val,test}"
DATA_ROOT="${DATA_ROOT:-/mnt/zhaozc_workspace/project/LQH/DrivoR-main-copy/dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/stage2_navigation_no_rewrite_v2_8gpu_pilot}"
QWEN_ENV_ROOT="${QWEN_ENV_ROOT:-/mnt/zhaozc_workspace/envs/mot_vega}"
DRIVOR_ENV_ROOT="${DRIVOR_ENV_ROOT:-/mnt/zhaozc_workspace/envs/drivorpython}"
PROMPT_FILE="$REPO_ROOT/scripts/data_generation/prompts/scene_navigation_v2.txt"
for value in "$SCENES_PER_RANK" "$DEEP_SCENES" "$SEED"; do
    [[ "$value" =~ ^[0-9]+$ ]] || { echo "Counts/seed must be nonnegative integers" >&2; exit 2; }
done
[[ "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "$DATA_SPLITS" =~ ^(train|val|test)(,(train|val|test))*$ ]] || { echo "Use DATA_SPLITS=train,val,test or a subset" >&2; exit 2; }
IFS=, read -r -a partitions <<< "$DATA_SPLITS"
declare -A seen=()
for partition in "${partitions[@]}"; do
    [[ ! ${seen[$partition]+yes} ]] || { echo "Duplicate partition: $partition" >&2; exit 2; }
    seen[$partition]=1
done
cd "$REPO_ROOT"
if [[ "$MODE" != --dry-run ]]; then
    set +u
    source /usr/local/PPU_SDK/envsetup.sh
    source "$QWEN_ENV_ROOT/bin/activate"
    set -u
    export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/nuplan-devkit${PYTHONPATH:+:$PYTHONPATH}"
    export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
    export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/drivor_stage2_mpl}"
    export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$DATA_ROOT/maps}"
    mkdir -p "$HF_HOME"
    "$QWEN_ENV_ROOT/bin/python" -c 'import sys,torch; n=int(sys.argv[1]); assert torch.cuda.is_available() and torch.cuda.device_count()>=n, f"Need {n} visible devices, found {torch.cuda.device_count()}"' "$NPROC_PER_NODE"
fi
for partition in "${partitions[@]}"; do
    command=("$QWEN_ENV_ROOT/bin/python" -m torch.distributed.run --standalone --nnodes=1
        "--nproc_per_node=$NPROC_PER_NODE" "--log-dir=$OUTPUT_DIR/$partition/launcher_logs" --tee=3
        scripts/data_generation/stage2/worker.py --output "$OUTPUT_DIR" --partition "$partition"
        --data-root "$DATA_ROOT" --scenes "$SCENES_PER_RANK" --seed "$SEED" --deep-scenes "$DEEP_SCENES"
        --prompt "$PROMPT_FILE" --drivor-python "$DRIVOR_ENV_ROOT/bin/python" --qwen-python "$QWEN_ENV_ROOT/bin/python")
    merge=("$DRIVOR_ENV_ROOT/bin/python" -B scripts/data_generation/stage2/merge.py
        --output "$OUTPUT_DIR" --partition "$partition" --world-size "$NPROC_PER_NODE")
    if [[ "$MODE" == --dry-run ]]; then
        printf '%q ' "${command[@]}"; printf '\n'
        printf '%q ' "${merge[@]}"; printf '\n'
    else
        "${command[@]}"
        "${merge[@]}"
    fi
done
