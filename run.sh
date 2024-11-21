#!/bin/bash

set -e

python run_evals.py \
    --config longcontext/checkpoints/bert24-base-8k-data-engineering-100b-1srt-flat-lr_evaluation.yaml \
    --checkpoints longcontext/checkpoints \
    --run-all-yamls \
    --parallel