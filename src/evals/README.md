# Ablation Evals

## Generate config

Run `python generate_eval_config_from_checkpoint.py --help` for all options.

### Create config by specifying checkpoint & config path
```
python generate_eval_config_from_checkpoint.py \
--checkpoint /path/to/checkpoint/folder \
--train_config /path/to/config.yaml
```

### Create config from the matching wandb run & add wandb tracking
```
python generate_eval_config_from_checkpoint.py \
--checkpoint /path/to/checkpoint/folder \
--wandb_entity entity_name \
--wandb_project project_name \
--track_run
```

### Create a config and skip the MNLI eval

You can skip any number of evals by adding `--skip_<eval_name>` for each eval you want to skip.

```
python generate_eval_config_from_checkpoint.py \
--checkpoint /path/to/checkpoint/folder \
--wandb_entity entity_name \
--wandb_project project_name \
--track_run \
--skip_mnli
```

## Launch a single ablation job
```bash
python eval.py yamls/ablations/checkpoint_name.yaml
```

## Automatically generate eval configs for multiple checkpoints and run evals on multiple GPUs

`run_evals_from_checkpoints.py` can be used to automatically generate configs from the latest checkpoints in a given directory, and run all evals on all available GPUs.

Run `python run_evals_from_checkpoints.py --help` for all options. All options from `generate_eval_config_from_checkpoint.py` are also available.

The logic for this script is:
- Each subdir in `--checkpoints` is scanned for model checkpoints.
    - If a checkpoint/symlink named "latest-rank0.pt" does not exist, a symlink to the latest checkpoint will be created.
    - If checkpoint/symlink exists, the script will use that checkpoint.
    - If you pass `--overwrite_existing_symlinks`, the script will create a new symlink to the latest checkpoint and use it.
- Config files should be stored together with the checkpoint (<sub_dir_name>_evaluation.yaml)
- If not, the script will try to find a matching wandb run in `wandb_entity`/`wandb_project` project and autogen a config.
- If the above fails, then the job will be skipped.

```
python run_evals_from_checkpoints.py \
--checkpoints /home/shared/data-ablations/checkpoints \
--wandb_entity entity_name \
--wandb_project project_name \
--track_run
```

