#!/bin/bash
LIST_FILE=$1

# Fix HF Cache permissions
export HF_HOME="/home/users/kagyemang-dua/hf_cache"
# UPDATED: Now using GPUs 3 and 2
export CUDA_VISIBLE_DEVICES=3,2

mkdir -p logs

while read game; do
    [ -z "$game" ] && continue
    echo "[$(date)] Starting $game..."
    
    # 2-hour limit. If it deadlocks, Linux kills it (Exit 124) and moves on.
    timeout 2h clem run -m Qwen3.5-9B -g "$game" 2>&1 | tee "logs/${game}.log"
    
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 124 ]; then
        echo "[$(date)] TIMEOUT: $game stalled and was forcefully skipped."
    elif [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date)] SUCCESS: $game completed."
    else
        echo "[$(date)] ERROR: $game failed (Exit: $EXIT_CODE)."
    fi
done < "$LIST_FILE"
