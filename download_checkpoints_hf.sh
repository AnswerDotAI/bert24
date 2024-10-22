#!/bin/bash

# Store the current directory
CURRENT_DIR=$(pwd)

# Create directory and download checkpoints
mkdir -p /home/rb/bert24-base-checkpoints && cd /home/rb/bert24-base-checkpoints
git init
git lfs install
git remote add origin https://huggingface.co/answerdotai/temp-bert-checkpoints
git sparse-checkout init
git sparse-checkout set "bert24-base-8k-data-engineering-100b-1srt-flat-lr" "bert24-base-8k-data-engineering-100b-1srt"
git pull origin main

# Return to original directory
cd "$CURRENT_DIR"