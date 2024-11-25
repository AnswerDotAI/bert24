# How to Run Evaluations

This document explains how to run fine-tuning evaluations for pre-trained models using the scripts `run_evals.py` and `generate_eval_config.py`. These scripts assume you have a pre-trained or finetuned Composer `FlexBert`checkpoint to evaluate.

## 1. Optionally Login to Hugging Face

First, make sure you are logged into Hugging Face, or provide a Hugging Face token to the `hub_token` argument:

```bash
huggingface-cli login
```

Follow the prompts to enter your authentication token.

## 2. Run Evaluations

### Option 1: Run Evaluations for All Checkpoints using `run_evals.py`

You can use the `run_evals.py` script to run evaluations for all checkpoints in a directory.

First, view the available arguments:
```bash
python run_evals.py --help
```

To simplify the process, create a YAML configuration file (e.g., `run_evals_args.yaml`) with the required argument values. For example:

```yaml
# Checkpoint & Config Paths
checkpoints: checkpoints
train_config: path/to/training_config.yaml # optional, uses default config if not provided and a wandb run isn't specified

# Model Options
model_size: base # default FlexBert model config to use

# Hugging Face Download
hub_repo: {org}/{repo}
hub_token: {your_hf_token} # needed if downloading from a private/gated repo and `huggingface-cli login` wasn't used
hub_files: {checkpoint_files} # optional limit to only download specific repo files or directories

# Evaluation Tasks
tasks:
    - mnli
    - sst2
    - cola
    - mrpc

# Task Settings
parallel: false
seeds:
    - 42
    - 314
    - 1234

# Weights & Biases (logging & config downloading)
wandb_run: ${your_pretraining_run_name} # these two options are only needed to download a non-default pretraining config
wandb_project: ${your_pretraining_wandb_project}

track_run: true # set these options to track the evaluation run in W&B
wandb_entity: ${your_wandb_entity}
track_run_project: ${your_evaluation_wandb_project}

# GPU Options (which GPUs to use)
gpu_ids:
    - 0
    - 1
```

Replace the placeholders with your specific values:

- `{parallel}`: Set to `true` to run evaluations on one checkpoint in parallel. Note that this can randomly error out.
- `{training_config.yaml}`: Path to your optional training configuration file if not using the default config.
- `{org}/{repo}`: Hugging Face Hub repository ID (e.g., `your_org/your_repo` where the Composer checkpoints are stored).
- `{your_hf_token}`: Your Hugging Face authentication token.
- `your_wandb_entity`: Your Weights & Biases entity (username or team name).
- `your_wandb_run_name`: The name of the Weights & Biases run containing the training configuration.
- `your_evaluation_wandb_project`: The name of your Weights & Biases evaluation project to log the eval runs to.

To run the script, use:

```bash
python run_evals.py --config run_evals_args.yaml
```

This will:

- **Download checkpoints** from the specified Hugging Face repository (if `hub_repo` is provided).
- **Generate evaluation configurations** for the specified tasks.
- **Run evaluations** in parallel on the specified GPUs.

### Option 2: Run Evaluation for a Specific Checkpoint using `generate_eval_config.py` and `eval.py`

If you want to run evaluation for a specific checkpoint, you can use `generate_eval_config.py` to generate the evaluation configuration, and then run `eval.py`.

#### Step 1: Generate the Evaluation Configuration

```bash
python generate_eval_config.py \
  --checkpoint path/to/checkpoint \
  --output-dir configs \
  --model-size base \
  --rope-theta 10000.0 \
  --tasks mnli sst2 \
  --wandb-entity ${your_wandb_entity} \
  --wandb-project ${your_wandb_project} \
  --wandb-run ${your_wandb_run_name} \
  --track-run \
  --track-run-project ${your_wandb_project}
```

Replace the placeholders accordingly:

- `path/to/checkpoint`: Path to your specific checkpoint file or directory.
- `configs`: Directory where the generated configuration file will be saved.
- `mnli sst2`: List of tasks you want to evaluate.
- `your_wandb_entity`, `your_wandb_project`, `your_wandb_run_name`: Your Weights & Biases details.

This command will generate a configuration YAML file in the `configs` directory.

#### Step 2: Run the Evaluation

```bash
python eval.py configs/generated_config.yaml
```

Replace `configs/generated_config.yaml` with the actual path to the generated configuration file.

## Tips & Tricks

1. **Building Evaluation Configurations for Single Tasks**

   If you want to build a fine-tuning evaluation configuration YAML for a single task, you can use `generate_eval_config.py` with the `--tasks` option to specify the task(s).

   For example:

       python generate_eval_config.py \
         --checkpoint path/to/checkpoint \
         --output-dir configs \
         --tasks mnli

   Then, run the evaluation with `eval.py`:

       python eval.py configs/generated_config.yaml

2. **Monitoring GPU Usage**

   Install `nvitop` to monitor GPU usage more effectively:

       pip install nvitop
       nvitop

   This provides a more useful and user-friendly interface than `nvidia-smi`.

## Additional Notes

- **Parallel Evaluations**: When running evaluations in parallel, you can specify the GPU IDs to use with the `--gpu-ids` option or in the YAML configuration file.

- **Configurable Options**: Both `run_evals.py` and `generate_eval_config.py` support various options to fine-tune the evaluation process. Use `--help` with these scripts to see all available options.

      python run_evals.py --help
      python generate_eval_config.py --help

- **Using Configuration Files**: You can use YAML configuration files to specify arguments for the scripts, which can simplify command-line usage. Command-line options will override options specified in the configuration file.

- **Hugging Face Hub Integration**: If you have your checkpoints stored in a private repository on Hugging Face Hub, ensure you have access by logging in via `huggingface-cli login` and providing your token.

- **Loading Training Configurations**: If you have a training configuration file or a Weights & Biases run containing the training configuration, you can provide it using the `--train-config` or `--wandb-run` options to ensure consistency between training and evaluation.

## Optional: Manual Checkpoint Download

If you prefer to manually download checkpoints instead of using the automatic download feature in `run_evals.py`, you can use `huggingface-cli`:

Replace `{org}`, `{repo}`, and `{checkpoint_folder}` with the appropriate organization, repository, and checkpoint folder names.

### Example Command:

```bash
huggingface-cli download {org}/{repo} --include "{checkpoint_folder}/*" --local-dir checkpoints
huggingface-cli download {org}/{repo} --include "{checkpoint_folder}" --local-dir checkpoints
```

### Notes:
- If there are multiple Composer checkpoints, use the latest one (usually starts with "ep-1").
- This manual download is optional since `run_evals.py` can automatically download checkpoints when you specify the `hub_repo`, `hub_folder`, and `hub_token` arguments.

---

This README reflects the latest updates in the scripts `run_evals.py` and `generate_eval_config.py`. Be sure to review the scripts and their help messages for the most current information.