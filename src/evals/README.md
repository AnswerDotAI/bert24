# Ablation Evals

## Generate config
### Create cofig by specifying checkpoint & config path
```
python generate_eval_config_from_checkpoint.py \
--checkpoint /path/to/checkpoint/folder \
--train_config /path/to/config.yaml
```
## Create config from the matching wandb run
```
python generate_eval_config_from_checkpoint.py \
--checkpoint /path/to/checkpoint/folder \
--wandb_project bert24-data-ablations
```

## Create config from the matching wandb run & add wandb tracking
```
python generate_eval_config_from_checkpoint.py \
--checkpoint /path/to/checkpoint/folder \
--wandb_project bert24-data-ablations \
--track_run
```

## Launch the ablation job
```bash
python ablation_eval.py yamls/ablations/checkpoint_name.yaml
```

## Launch abalations for sub-directories of a given path
- Each subdir needs to contain a checkpoint named "latest-rank0.pt"
- Config file should be stored together with the checkpoint (<sub_dir_name>.yaml)
- If not, the script will try to find a matching wandb run in `bert24/bert24` project.
- If the above fails, then the job will be skipped.

```bash
./example_eval_checkpoints.sh /home/shared/data-ablations/checkpoints
```