import os
from pathlib import Path
from collections import OrderedDict
import yaml
import wandb
import typer
from typer import Option
from typing import Annotated, List, Optional


app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]}, pretty_exceptions_show_locals=False)


def ordered_yaml_dump(data, stream=None, Dumper=yaml.Dumper, **kwds):
    class OrderedDumper(Dumper):
        pass

    def _dict_representer(dumper, data):
        return dumper.represent_mapping(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, data.items())

    OrderedDumper.add_representer(OrderedDict, _dict_representer)
    return yaml.dump(data, stream, OrderedDumper, **kwds)


def safe_get(dict_obj, key, default=None):
    return dict_obj.get(key, default)


def get_wandb_config(run_name, entity_name, project_name):
    api = wandb.Api()
    runs = api.runs(f"{entity_name}/{project_name}")
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

        print(f"   Successfully loaded config from run: {run_name}")

        return config
    else:
        print(f"Run '{run_name}' not found in project '{entity_name}/{project_name}'")

    # Finish the wandb run
    wandb.finish()


# fmt: off
@app.command()
def main(
    checkpoint: Annotated[Path, Option(help="Path to a model checkpoint", show_default=False, rich_help_panel="Checkpoint & Config Paths")],
    output_dir: Annotated[Path, Option(help="Output directory for the generated config", rich_help_panel="Checkpoint & Config Paths")] = Path("./yamls/ablations"),
    train_config: Annotated[Optional[Path], Option(help="Path to a .yaml file containing training configuration", rich_help_panel="Checkpoint & Config Paths")] = None,
    wandb_run: Annotated[Optional[str], Option(help="wandb run containing the training configuration", rich_help_panel="W&B")] = None,
    wandb_project: Annotated[Optional[str], Option(help="wandb project for the run", rich_help_panel="W&B")] = None,
    wandb_entity: Annotated[Optional[str], Option(help="wandb entity for the project", rich_help_panel="W&B")] = None,
    track_run: Annotated[bool, Option("--track-run", help="Track the eval run with wandb", rich_help_panel="W&B")] = False,
    pooling_type: Annotated[Optional[str], Option(help="Pooling type for the classification head", show_default=False, rich_help_panel="Model Options")] = None,
    head_class_act: Annotated[Optional[str], Option(help="Classification head activation function. Defaults to hidden_act if set, then tanh", show_default=False, rich_help_panel="Model Options")] = None,
    head_class_norm: Annotated[Optional[str], Option(help="Classification head normalization function", show_default=False, rich_help_panel="Model Options")] = None,
    head_class_dropout: Annotated[float, Option(help="Classification head dropout rate", rich_help_panel="Model Options")] = 0.0,
    skip_semipro: Annotated[bool, Option("--skip-semipro", help="Skip the MlMMLU-Amateur/Semipro eval", rich_help_panel="Skip Tasks")] = False,
    skip_reserve: Annotated[bool, Option("--skip-reserve", help="Skip the MlMMLU-Rookie/Reserve eval", rich_help_panel="Skip Tasks")] = False,
    skip_eurlex: Annotated[bool, Option("--skip-eurlex", help="Skip the EurLex eval", rich_help_panel="Skip Tasks")] = False,
    skip_mnli: Annotated[bool, Option("--skip-mnli", help="Skip the MNLI eval", rich_help_panel="Skip Tasks")] = False,
    skip_boolq: Annotated[bool, Option("--skip-boolq", help="Skip the BoolQ eval", rich_help_panel="Skip Tasks")] = False,
    skip_wic: Annotated[bool, Option("--skip-wic", help="Skip the WIC eval", rich_help_panel="Skip Tasks")] = False,
    skip_ultrafeedback: Annotated[bool, Option("--skip-ultrafeedback", help="Skip the UltraFeedback eval", rich_help_panel="Skip Tasks")] = False,
    seeds: Annotated[List[int], Option(help="List of seeds to use for the eval", rich_help_panel="Task Settings")] = [1618, 42, 6033, 3145],
    parallel: Annotated[bool, Option("--parallel/--single", help="Run the evals in parallel on multiple GPUs or one GPU", rich_help_panel="Task Settings")] = True,
):
# fmt: on
    # Read the input YAML file
    os.makedirs(output_dir, exist_ok=True)
    input_config = None

    if "pt" in str(checkpoint):
        ckpt = checkpoint.name  # checkpoint
        ckpt_path = str(checkpoint.parent)
    else:
        ckpt = "latest-rank0.pt"
        ckpt_path = str(checkpoint).rstrip("/")
    ckpt_id = ckpt_path.split("/")[-1]

    if train_config:
        with open(train_config, "r") as file:
            input_config = yaml.safe_load(file)
    else:
        # Specify the run name
        print("Attempting to find config file within checkpoint folder...")
        yaml_file = ckpt_path + ".yaml"
        yaml_file_alt = ckpt_path + "/" + ckpt_id + ".yaml"

        if os.path.exists(yaml_file):
            with open(yaml_file, "r") as file:
                input_config = yaml.safe_load(file)
        elif os.path.exists(yaml_file_alt):
            with open(yaml_file_alt, "r") as file:
                input_config = yaml.safe_load(file)
        else:
            if wandb_run:
                run_name = wandb_run
            else:
                print("   No train config specified and no wandb run specified")
                print("   We will attempt to load the config from a wandb run named the same as the checkpoint provided.")
                print("   If this fails, please specify a train config or a wandb run!")
                run_name = ckpt_id  # ckpt_path

            input_config = get_wandb_config(run_name, wandb_entity, wandb_project)

    if input_config is None:
        raise ValueError(
            "Could not find a config for the provided checkpoint. Please provide a wandb run name or a config file."
        )
    # Create the new configuration OrderedDict
    new_config = OrderedDict()

    print(f"Config found for run: {safe_get(input_config, 'run_name', ckpt_path)}")

    new_config["parallel"] = parallel
    new_config["base_run_name"] = safe_get(input_config, "run_name", ckpt_path) + "_evaluation"
    new_config["default_seed"] = 19
    new_config["precision"] = safe_get(input_config, "precision")
    new_config["tokenizer_name"] = safe_get(input_config, "tokenizer_name")

    model_config = OrderedDict()
    model_config["name"] = safe_get(input_config, "model", {}).get("name")
    model_config["use_pretrained"] = True
    model_config["pretrained_model_name"] = safe_get(input_config, "model", {}).get("pretrained_model_name")
    model_config["tokenizer_name"] = "${tokenizer_name}"

    model_config_inner = OrderedDict()
    for key in input_config.get("model", {}).get("model_config", {}).keys():
        model_config_inner[key] = safe_get(input_config, "model", {}).get("model_config", {}).get(key)
    model_config_inner["use_fa2"] = True
    model_config_inner["deterministic_fa2"] = True

    if head_class_norm:
        model_config_inner["head_class_norm"] = head_class_norm
    if head_class_dropout > 0:
        model_config_inner["head_class_dropout"] = head_class_dropout
    if head_class_act:
        model_config_inner["head_class_act"] = head_class_act
    else:
        model_config_inner["head_class_act"] = safe_get(model_config_inner, "hidden_act", "tanh")
    if pooling_type:
        model_config_inner["pooling_type"] = pooling_type

    if model_config_inner:
        model_config["model_config"] = model_config_inner

    if model_config:
        new_config["model"] = model_config

    new_config["starting_checkpoint_load_path"] = ckpt
    new_config["local_pretrain_checkpoint_folder"] = ckpt_path  # + "/"
    new_config["save_finetune_checkpoint_prefix"] = "./finetuned-checkpoints"
    new_config["save_finetune_checkpoint_folder"] = "${save_finetune_checkpoint_prefix}/${base_run_name}"

    loggers = OrderedDict()
    wandb_config = OrderedDict()
    wandb_config["project"] = f"{wandb_project}-evals"
    wandb_config["entity"] = wandb_entity
    loggers["wandb"] = wandb_config
    if track_run:
        new_config["loggers"] = loggers

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

    if not skip_semipro:
        mlmmlu_amateur_semipro = OrderedDict()
        mlmmlu_amateur_semipro["seeds"] = seeds[:4]
        mlmmlu_amateur_semipro["trainer_kwargs"] = {"save_num_checkpoints_to_keep": 0}
        tasks["mlmmlu_amateur_semipro"] = mlmmlu_amateur_semipro

    if not skip_reserve:
        mlmmlu_rookie_reserve = OrderedDict()
        mlmmlu_rookie_reserve["seeds"] = seeds[:4]
        mlmmlu_rookie_reserve["trainer_kwargs"] = {"save_num_checkpoints_to_keep": 0}
        tasks["mlmmlu_rookie_reserve"] = mlmmlu_rookie_reserve

    if not skip_eurlex:
        eurlex = OrderedDict()
        eurlex["seeds"] = seeds[:2]
        eurlex["trainer_kwargs"] = {"save_num_checkpoints_to_keep": 0}
        eurlex["model_config"] = {"problem_type": "multi_label_classification"}
        tasks["eurlex"] = eurlex

    if not skip_mnli:
        mnli = OrderedDict()
        mnli["seeds"] = [seeds[0]]
        mnli["trainer_kwargs"] = {"save_num_checkpoints_to_keep": 1}
        tasks["mnli"] = mnli

    if not skip_boolq:
        boolq = OrderedDict()
        boolq["seeds"] = seeds[:3]
        boolq["trainer_kwargs"] = {"save_num_checkpoints_to_keep": 0}
        tasks["boolq"] = boolq

    if not skip_wic:
        wic = OrderedDict()
        wic["seeds"] = seeds[:3]
        wic["trainer_kwargs"] = {"save_num_checkpoints_to_keep": 0}
        tasks["wic"] = wic

    if not skip_ultrafeedback:
        ultrafeedback = OrderedDict()
        ultrafeedback["seeds"] = seeds[:2]
        ultrafeedback["trainer_kwargs"] = {"save_num_checkpoints_to_keep": 0}
        tasks["ultrafeedback"] = ultrafeedback

    new_config["tasks"] = tasks

    # Write the new configuration to a YAML file
    output_filename = f"{output_dir}/{ckpt_id}_evaluation.yaml"
    with open(output_filename, "w") as file:
        ordered_yaml_dump(new_config, file, default_flow_style=False)

    print(f"Configuration converted and saved to {output_filename}\n")


if __name__ == "__main__":
    app()
