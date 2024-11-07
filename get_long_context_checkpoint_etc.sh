MODEL=answerdotai/temp-bert-checkpoints
CHECKPOINT=bert24-base-8k-data-engineering-100b-1srt-flat-lr
FILE=ep1-ba21195-rank0.pt

mkdir -p ./checkpoints
# download checkpoint files
huggingface-cli download answerdotai/temp-bert-checkpoints bert24-base-8k-data-engineering-100b-1srt-flat-lr/ep1-ba21195-rank0.pt --local-dir ./checkpoints/

# download the "training yaml file"
TRAINING_YAML_FILE=bert24-base-data-eng-100b-1srt.yaml
huggingface-cli download $MODEL $TRAINING_YAML_FILE --local-dir ./





