#!/bin/bash

# Function to get the number of available GPUs
get_num_gpus() {
    nvidia-smi -L | wc -l
}

# Function to wait for a free GPU
wait_for_free_gpu() {
    while true; do
        for i in $(seq 0 $((NUM_GPUS-1))); do
            if ! ps aux | grep -v grep | grep -q "CUDA_VISIBLE_DEVICES=$i"; then
                echo $i
                return
            fi
        done
        sleep 10
    done
}

# Check if a path argument is provided
if [ $# -eq 0 ]; then
    echo "Please provide a path as an argument."
    exit 1
fi

PATH_ARG=$1
NUM_GPUS=$(get_num_gpus)
echo "Number of available GPUs: $NUM_GPUS"

# Generate eval configs for each checkpoint
for folder in "$PATH_ARG"/*; do
    if [ -d "$folder" ]; then
        python generate_eval_config_from_checkpoint.py --checkpoint "$folder"
    fi
done

# Launch ablation jobs
for config in "$PATH_ARG"/*_evaluation.yaml; do
    if [ -f "$config" ]; then
        GPU_ID=$(wait_for_free_gpu)
        echo "Launching job for $config on GPU $GPU_ID"
        CUDA_VISIBLE_DEVICES=$GPU_ID python ablation_eval.py "$config" &
    fi
done

# Wait for all background jobs to finish
wait

echo "All jobs completed."
