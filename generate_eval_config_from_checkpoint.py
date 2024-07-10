import os
from pathlib import Path
import argparse
from collections import OrderedDict
import yaml
import wandb

parser = argparse.ArgumentParser(description="Convert configuration for model evaluation.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to a model checkpoint")
parser.add_argument(
    "--train_config",
    type=str,
    help="Path to a .yaml file containing training configuration",
)
parser.add_argument("--wandb_run", type=str, help="Name of a wandb run")
parser.add_argument("--wandb_project", type=str, default="bert24")
parser.add_argument("--output_dir", type=str, default=".eval_configs/")

args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)


def ordered_yaml_dump(data, stream=None, Dumper=yaml.Dumper, **kwds):
    class OrderedDumper(Dumper):
        pass

    def _dict_representer(dumper, data):
        return dumper.represent_mapping(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, data.items())

    OrderedDumper.add_representer(OrderedDict, _dict_representer)
    return yaml.dump(data, stream, OrderedDumper, **kwds)


def safe_get(dict_obj, key, default=None):
    return dict_obj.get(key, default)


def get_wandb_config(run_name):
    api = wandb.Api()
    runs = api.runs("bert24/bert24")
    target_run = next((run for run in runs if run.name == run_name), None)

    if target_run:
        # Download the config.yaml file
        file_name = "config.yaml"
        target_run.file(file_name).download(replace=True)

        # Load the config file
        config_path = Path(file_name)
        with config_path.open() as f:
            config = yaml.safe_load(f)

        # Flatten the nested structure and remove 'desc' and 'value' keys
        def flatten_config(cfg):
            flattened = {}
            for key, value in cfg.items():
                if isinstance(value, dict):
                    if "desc" in value and "value" in value and value["desc"] is None:
                        flattened[key] = value["value"]
                    else:
                        flattened[key] = flatten_config(value)
                else:
                    flattened[key] = value
            return flattened

        config = flatten_config(config)

        # Remove the _wandb key from the config
        if "_wandb" in config:
            del config["_wandb"]

        # Ensure all required keys are present
        required_keys = [
            "data_local",
            "data_remote",
            "max_seq_len",
            "tokenizer_name",
            "mlm_probability",
            "run_name",
            "model",
            "train_loader",
            "eval_loader",
            "scheduler",
            "optimizer",
            "algorithms",
            "max_duration",
            "eval_interval",
            "global_train_batch_size",
            "global_eval_batch_size",
            "seed",
            "device_eval_batch_size",
            "device_train_microbatch_size",
            "precision",
            "progress_bar",
            "log_to_console",
            "console_log_interval",
            "callbacks",
            "loggers",
            "save_interval",
            "save_num_checkpoints_to_keep",
            "save_folder",
        ]

        for key in required_keys:
            if key not in config:
                config[key] = None  # or some default value

        # Remove all keys not in required_keys
        config = {key: config[key] for key in required_keys if key in config}

        print(f"Successfully loaded config from run: {run_name}")

        return config
    else:
        print(f"Run '{run_name}' not found in project 'bert24'")

    # Finish the wandb run
    wandb.finish()


# Read the input YAML file

input_config = None

if "pt" in args.checkpoint:
    ckpt = args.checkpoint
    ckpt_path = args.checkpoint.split("/")[:-1]
else:
    ckpt = args.checkpoint + "/latest-rank0.pt"
    ckpt_path = args.checkpoint

if args.train_config:
    with open(args.train_config, "r") as file:
        input_config = yaml.safe_load(file)
else:
    # Specify the run name
    print("Attemptong to find config file within checkpoint folder...")
    yaml_file = ckpt + ".yaml"
    yaml_file_alt = ckpt + "/" + ckpt + ".yaml"

    if os.path.exists(yaml_file):
        with open(yaml_file, "r") as file:
            input_config = yaml.safe_load(file)
    elif os.path.exists(yaml_file_alt):
        with open(yaml_file_alt, "r") as file:
            input_config = yaml.safe_load(file)
    else:
        if args.wandb_run:
            run_name = args.wandb_run
        else:
            print("___ No train config specified and no wandb run specified ___")
            print("We will attempt to load the config from a wandb run named the same as the checkpoint provided.")
            print("If this fails, please specify a train config or a wandb run!")
            run_name = ckpt_path

        input_config = get_wandb_config(run_name)

if input_config is None:
    raise ValueError(
        "Could not find a config for the provided checkpoint. Please provide a wandb run name or a config file."
    )
# Create the new configuration OrderedDict
new_config = OrderedDict()

print(safe_get(input_config, "run_name", ckpt_path))

new_config["parallel"] = True
new_config["base_run_name"] = safe_get(input_config, "run_name", ckpt_path) + "_evaluation"
new_config["default_seed"] = 19
new_config["precision"] = safe_get(input_config, "precision")
new_config["tokenizer_name"] = safe_get(input_config, "tokenizer_name")

model_config = OrderedDict()
model_config["name"] = safe_get(input_config, "model", {}).get("name")
model_config["use_pretrained"] = True
model_config["pretrained_model_name"] = "${tokenizer_name}"
model_config["tokenizer_name"] = "${tokenizer_name}"

model_config_inner = OrderedDict()
for key in input_config.get("model", {}).get("model_config", {}).keys():
    model_config_inner[key] = safe_get(input_config, "model", {}).get("model_config", {}).get(key)
model_config_inner["use_fa2"] = True
model_config_inner["head_class_norm"] = None
model_config_inner["head_class_act"] = "tanh"

if model_config_inner:
    model_config["model_config"] = model_config_inner

if model_config:
    new_config["model"] = model_config

new_config["starting_checkpoint_load_path"] = ckpt
new_config["local_pretrain_checkpoint_folder"] = ckpt_path + "/"
new_config["save_finetune_checkpoint_prefix"] = "./finetuned-checkpoints"
new_config["save_finetune_checkpoint_folder"] = "${save_finetune_checkpoint_prefix}/${base_run_name}"

callbacks = OrderedDict()
callbacks["lr_monitor"] = {}
callbacks["speed_monitor"] = {}
if callbacks:
    new_config["callbacks"] = callbacks

scheduler = OrderedDict()
scheduler["name"] = "linear_decay_with_warmup"
scheduler["t_warmup"] = "0.06dur"
scheduler["alpha_f"] = 0.0
if scheduler:
    new_config["scheduler"] = scheduler

tasks = OrderedDict()
mnli = OrderedDict()


mmlu_amateur_semipro = OrderedDict()
mmlu_amateur_semipro["seeds"] = [23, 42]
mmlu_amateur_semipro["trainer_kwargs"] = {"save_num_checkpoints_to_keep": 0}
tasks["mmlu_amateur_semipro"] = mmlu_amateur_semipro

mmlu_rookie_reserve = OrderedDict()
mmlu_rookie_reserve["seeds"] = [23, 42]
mmlu_rookie_reserve["trainer_kwargs"] = {"save_num_checkpoints_to_keep": 0}
tasks["mmlu_rookie_reserve"] = mmlu_rookie_reserve

eurlex = OrderedDict()
eurlex["seeds"] = [23, 42, 6033]
eurlex["trainer_kwargs"] = {"save_num_checkpoints_to_keep": 0}
eurlex["model_config"] = {"problem_type": "multi_label_classification"}
tasks["eurlex"] = eurlex

mnli["seeds"] = [23]
mnli["trainer_kwargs"] = {"save_num_checkpoints_to_keep": 1}
tasks["mnli"] = mnli

boolq = OrderedDict()
boolq["seeds"] = [23, 42, 6033]
boolq["trainer_kwargs"] = {"save_num_checkpoints_to_keep": 0}
tasks["boolq"] = boolq

wic = OrderedDict()
wic["seeds"] = [23, 42, 6033]
wic["trainer_kwargs"] = {"save_num_checkpoints_to_keep": 0}
tasks["wic"] = wic


new_config["tasks"] = tasks

# Write the new configuration to a YAML file
output_filename = f"{args.output_dir}/{ckpt_path}_evaluation.yaml"
with open(output_filename, "w") as file:
    ordered_yaml_dump(new_config, file, default_flow_style=False)

print(f"Configuration converted and saved to {output_filename}")
