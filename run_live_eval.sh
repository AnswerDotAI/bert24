#!/bin/bash

# Set up logging
LOG_FILE="$HOME/bert_live_eval.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check if a process is running
is_process_running() {
    pgrep -f "$1" > /dev/null
}

# Function to run a single evaluation job
run_single_eval() {
    local checkpoint_path="$1"
    log "Starting evaluation for checkpoint: $checkpoint_path"
    
    if python run_evals_from_checkpoints.py \
        --checkpoints "$checkpoint_path" \
        --skip-semipro \
        --skip-eurlex \
        --skip-ultrafeedback \
        --parallel \
        --override-existing-symlinks \
        --track-run; then
        log "Evaluation completed successfully for $checkpoint_path"
    else
        log "Error: Evaluation failed for $checkpoint_path"
        return 1
    fi
}

# Function to run the evaluation jobs
run_eval_jobs() {
    local checkpoints=(
        "/home/rb/bert24_checkpoints/bert24-large-v2"
    )
    
    for checkpoint in "${checkpoints[@]}"; do
        if ! run_single_eval "$checkpoint"; then
            log "Aborting further evaluations due to error"
            return 1
        fi
    done
    
}

# Trap Ctrl+C and exit gracefully
trap 'log "Received interrupt signal. Exiting..."; exit 0' INT

# Main loop
while true; do
    if is_process_running "run_evals_from_checkpoints.py"; then
        log "Evaluation jobs are already running. Waiting for next check."
    else
        log "Evaluation jobs are not running. Starting them now."
        if run_eval_jobs; then
            log "All evaluation jobs completed successfully"
        else
            log "Error occurred during evaluation jobs"
        fi
    fi

    log "Sleeping for 30 mins before next check"
    sleep 1800
done

