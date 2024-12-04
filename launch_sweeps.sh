#!/bin/bash

launch_sweep() {
    local session_name=$1
    local sweep_id=$2
    local gpu_id=$3
    
    tmux new-session -d -s "$session_name"
    
    tmux send-keys -t "$session_name" "conda activate bert24" C-m
    tmux send-keys -t "$session_name" "export CUDA_VISIBLE_DEVICES=$gpu_id" C-m
    tmux send-keys -t "$session_name" "export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" C-m
    # resume from a previous sweep
    # tmux send-keys -t "$session_name" "wandb sweep --resume bert24/bert24-llm-guardrails-eval-sweeps/$sweep_id" C-m
    tmux send-keys -t "$session_name" "wandb agent bert24/bert24-llm-guardrails-eval-sweeps/$sweep_id" C-m
}

# sweep IDs
declare -a sweeps=(
    # put sweep IDs here
    "0a7s2bn9"
    "fpjzcugi"
    "0vf65v4m"
    "jn91wiq3"
    "p54t97yq"
    "4rmqkiz2"
    "urcwlbbi"
    "ohyhy3f6"

    # "ugb3btv2"
    # "hjhgnwlz"
    # "sppfczuh"
    # "pqv2zd56"
    # "iux0qfn1"
    # "b5wg9kpm"
    # "3av6jlto"
    # "01yol1wo"
)


for i in "${!sweeps[@]}"; do
    sweep_id="${sweeps[$i]}"
    gpu_id=$i
    session_name="sweep_$((gpu_id+1))"

    launch_sweep "$session_name" "$sweep_id" "$gpu_id"
    
    echo "Launched sweep ${sweep_id} in tmux session ${session_name} on GPU ${gpu_id}"

    sleep 2
done