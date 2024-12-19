# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

import warnings
from collections import OrderedDict
from enum import Enum
from pathlib import Path
from typing import Annotated, List, Optional

import typer
import wandb
import yaml
from typer import Option

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    from eval import TASK_NAME_TO_CLASS


# Create TaskName enum dynamically from TASK_NAME_TO_CLASS keys
TaskName = Enum("TaskName", {name: name for name in TASK_NAME_TO_CLASS.keys()}, type=str)


app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]}, pretty_exceptions_show_locals=False)


# from maxb2: https://github.com/tiangolo/typer/issues/86#issuecomment-996374166
def conf_callback(ctx: typer.Context, param: typer.CallbackParam, config: Optional[str] = None):
    if config is not None:
        typer.echo(f"Loading config file: {config}\n")
        try:
            with open(config, "r") as f:  # Load config file
                conf = yaml.safe_load(f)
            ctx.default_map = ctx.default_map or {}  # Initialize the default map
            ctx.default_map.update(conf)  # Merge the config dict into default_map
        except Exception as ex:
            raise typer.BadParameter(str(ex))
    return config


class ModelSize(str, Enum):
    BASE = "base"
    LARGE = "large"


def get_model_defaults(model_size: ModelSize):
    # Define default model configurations for base and large sizes
    default_model_configs = {
        "base": {
            "num_hidden_layers": 22,
            "hidden_size": 768,
            "intermediate_size": 1152,
            "num_attention_heads": 12,  # to have head size of 64
        },
        "large": {
            "num_hidden_layers": 28,
            "hidden_size": 1024,
            "intermediate_size": 2624,
            "num_attention_heads": 16,
        },
    }

    # Select the default model config based on the model_size argument
    model_config = default_model_configs[model_size.value]

    # Additional default configurations common to both sizes
    default_model_config_common = {
        "vocab_size": 50368,
        "init_method": "full_megatron",
        "attention_layer": "rope",
        "attention_probs_dropout_prob": 0.0,
        "attn_out_bias": False,
        "attn_out_dropout_prob": 0.1,
        "attn_qkv_bias": False,
        "bert_layer": "prenorm",
        "embed_dropout_prob": 0.0,
        "embed_norm": True,
        "final_norm": True,
        "skip_first_prenorm": True,
        "embedding_layer": "sans_pos",
        "loss_function": "fa_cross_entropy",
        "loss_kwargs": {"reduction": "mean"},
        "mlp_dropout_prob": 0.0,
        "mlp_in_bias": False,
        "mlp_layer": "glu",
        "mlp_out_bias": False,
        "normalization": "layernorm",
        "norm_kwargs": {"eps": 1e-5, "bias": False},
        "hidden_act": "gelu",
        "head_pred_act": "gelu",
        "activation_function": "gelu",  # better safe than sorry
        "padding": "unpadded",
        "rotary_emb_dim": None,
        "rotary_emb_scale_base": None,
        "rotary_emb_interleaved": False,
        "local_attn_rotary_emb_base": 10000.0,
        "local_attn_rotary_emb_dim": None,
        "allow_embedding_resizing": True,
        "sliding_window": 128,
        "global_attn_every_n_layers": 3,
        "unpad_embeddings": True,
        "compile_model": True,
        "use_fa2": True,
        "deterministic_fa2": True,
    }
    model_config.update(default_model_config_common)
    return model_config


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


@app.command()
def main(
    checkpoint: Annotated[Path, Option(help="Path to a model checkpoint", show_default=False, rich_help_panel="Checkpoint & Config Paths")],
    output_dir: Annotated[Path, Option(help="Output directory for the generated config", rich_help_panel="Checkpoint & Config Paths")] = Path("./yamls/ablations"),
    train_config: Annotated[Optional[Path], Option(help="Path to a .yaml file containing training configuration. If one is not provided, will attempt to load the config from a wandb run or use defaults.", rich_help_panel="Checkpoint & Config Paths")] = None,
    model_size: Annotated[ModelSize, Option("--model-size", help="Model to use for default model config: 'base' or 'large'", rich_help_panel="Checkpoint & Config Paths")] = ModelSize.BASE,
    rope_theta: Annotated[Optional[float], Option("--rope-theta", help="Value for `rotary_emb_base` in the model configuration. If not provided, defaults to pretraining value of 10000.0", rich_help_panel="Checkpoint & Config Paths")] = None,
    use_dir_name: Annotated[bool, Option("--use-dir-name", help="Use the checkpoint's parent dirname as the eval base_run_name", rich_help_panel="Checkpoint & Config Paths")] = False,
    tasks: Annotated[Optional[List[TaskName]], Option(help="List of tasks to include in the evaluation. Default is all tasks.", rich_help_panel="Eval Tasks", case_sensitive=False, show_default=False)] = None, # type: ignore
    wandb_run: Annotated[Optional[str], Option(help="wandb run containing the training configuration", rich_help_panel="Weights & Biases")] = None,
    wandb_project: Annotated[Optional[str], Option(help="wandb project for the run", rich_help_panel="Weights & Biases")] = None,
    wandb_entity: Annotated[Optional[str], Option(help="wandb entity for the project", rich_help_panel="Weights & Biases")] = None,
    track_run: Annotated[bool, Option("--track-run", help="Track the eval run with wandb", rich_help_panel="Weights & Biases")] = False,
    track_run_project: Annotated[Optional[str], Option(help="wandb project for tracking the run", rich_help_panel="Weights & Biases")] = None,
    pooling_type: Annotated[Optional[str], Option(help="Pooling type for the classification head", show_default=False, rich_help_panel="Model Options")] = None,
    head_class_act: Annotated[Optional[str], Option(help="Classification head activation function. Defaults to hidden_act if set, then tanh", show_default=False, rich_help_panel="Model Options")] = None,
    head_class_norm: Annotated[Optional[str], Option(help="Classification head normalization function", show_default=False, rich_help_panel="Model Options")] = None,
    head_class_dropout: Annotated[float, Option(help="Classification head dropout rate", rich_help_panel="Model Options")] = 0.0,
    fast_ultrafeedback: Annotated[bool, Option("--fast-ultrafeedback", help="Use a shorter sequence length (1536) for the UltraFeedback eval", rich_help_panel="Task Settings")] = False,
    seeds: Annotated[List[int], Option(help="List of seeds to use for the eval", rich_help_panel="Task Settings")] = [1618, 42, 6033, 3145],
    parallel: Annotated[bool, Option("--parallel/--single", help="Run the evals in parallel on multiple GPUs or one GPU. Only use if evaluating a single checkpoint on multiple GPUs.", rich_help_panel="Task Settings")] = False,
    config: Annotated[Optional[Path], Option(callback=conf_callback, is_eager=True, help="Relative path to YAML config file for setting options. Passing CLI options will supersede config options.", case_sensitive=False, rich_help_panel="Options")] = None,
):  # fmt: skip
    # Read the input YAML file
    output_dir.mkdir(parents=True, exist_ok=True)
    input_config = None

    if checkpoint.is_file() and checkpoint.suffix == ".pt":
        ckpt = checkpoint.name  # checkpoint
        ckpt_path = checkpoint.parent
    elif checkpoint.is_dir():
        ckpts = list(checkpoint.glob("*.pt"))
        if len(ckpts) == 1:
            ckpt = ckpts[0].name
        elif len(ckpts) > 1:
            ckpt = "latest-rank0.pt"
        elif len(ckpts) == 0:
            raise ValueError(f"No checkpoint found in the provided directory: {checkpoint}")
        ckpt_path = checkpoint
    else:
        raise ValueError(f"Invalid checkpoint path provided: {checkpoint}")

    ckpt_id = ckpt_path.name

    if train_config:
        with train_config.open("r") as file:
            input_config = yaml.safe_load(file)
    else:
        # Specify the run name
        print("Attempting to find config file within checkpoint folder...")
        yaml_file = checkpoint.parent / f"{checkpoint.parent.name}.yaml"
        yaml_file_alt = ckpt_path / f"{ckpt_id}.yaml"
        print(yaml_file)

        if yaml_file.exists():
            with yaml_file.open("r") as file:
                input_config = yaml.safe_load(file)
        elif yaml_file_alt.exists():
            with yaml_file_alt.open("r") as file:
                input_config = yaml.safe_load(file)
        else:
            if wandb_run:
                run_name = wandb_run
            else:
                print("   No train config specified and no wandb run specified")
                print("   We will attempt to load the config from a wandb run named the same as the checkpoint provided or use the default model config.")  # fmt: skip
                run_name = ckpt_id
            try:
                input_config = get_wandb_config(run_name, wandb_entity, wandb_project)
            except Exception as e:
                print(f"   No valid wandb config found: {e}")
                input_config = {}

    if input_config is None:
        raise ValueError("Could not find a config for the provided checkpoint. Please provide a wandb run name or a config file.")  # fmt: skip
    # Create the new configuration OrderedDict
    new_config = OrderedDict()

    if input_config.get("run_name", None):
        print(f"Config found for run: {safe_get(input_config, 'run_name', ckpt_path.name)}")
    else:
        print(f"No configs found, using default {model_size.value} model config with theta={rope_theta if rope_theta is not None else 10000.0} for checkpoint: {ckpt_path.name}")  # fmt: skip

    new_config["parallel"] = parallel

    if use_dir_name:
        base_run_name = ckpt_path.name
    else:
        base_run_name = safe_get(input_config, "run_name", ckpt_path.name)
    new_config["base_run_name"] = base_run_name

    new_config["default_seed"] = 19
    new_config["precision"] = safe_get(input_config, "precision")
    new_config["tokenizer_name"] = safe_get(input_config, "tokenizer_name", "bclavie/olmo_bert_template")

    model_config = OrderedDict()
    model_config["name"] = safe_get(input_config, "model", {}).get("name", "flex_bert")
    model_config["use_pretrained"] = True
    model_config["pretrained_model_name"] = safe_get(input_config, "model", {}).get("pretrained_model_name", "bert-base-uncased")  # fmt: skip
    model_config["tokenizer_name"] = "${tokenizer_name}"

    # Get the default model config for the given model size
    default_model_config_inner = get_model_defaults(model_size)

    # If rope_theta is provided, set rotary_emb_base to rope_theta; otherwise, use default
    if rope_theta is not None:
        default_model_config_inner["rotary_emb_base"] = rope_theta
    else:
        default_model_config_inner["rotary_emb_base"] = 10000.0

    # Build model_config_inner by taking values from input_config or using defaults
    model_config_inner = {}
    input_model_config = safe_get(input_config, "model", {}).get("model_config", {})
    for key in default_model_config_inner.keys():
        model_config_inner[key] = input_model_config.get(key, default_model_config_inner[key])

    # Additional model configurations based on arguments
    if head_class_norm is not None:
        model_config_inner["head_class_norm"] = head_class_norm
    if head_class_dropout > 0:
        model_config_inner["head_class_dropout"] = head_class_dropout
    if head_class_act is not None:
        model_config_inner["head_class_act"] = head_class_act
    else:
        model_config_inner["head_class_act"] = safe_get(model_config_inner, "hidden_act", "tanh")
    if pooling_type is not None:
        model_config_inner["pooling_type"] = pooling_type

    model_config["model_config"] = model_config_inner

    new_config["model"] = model_config

    new_config["starting_checkpoint_load_path"] = ckpt
    new_config["local_pretrain_checkpoint_folder"] = str(ckpt_path)
    new_config["save_finetune_checkpoint_prefix"] = "./finetuned-checkpoints"
    new_config["save_finetune_checkpoint_folder"] = "${save_finetune_checkpoint_prefix}/${base_run_name}"

    loggers = OrderedDict()

    if track_run:
        wandb_config = OrderedDict()
        assert wandb_entity is not None, "set wandb entity"
        assert track_run_project is not None, "set wandb project for tracking"
        wandb_config["project"] = track_run_project
        wandb_config["entity"] = wandb_entity
        loggers["wandb"] = wandb_config
        new_config["loggers"] = loggers

    callbacks = OrderedDict()
    callbacks["lr_monitor"] = {}
    callbacks["speed_monitor"] = {}
    if callbacks:
        new_config["callbacks"] = callbacks

    scheduler = OrderedDict()
    scheduler["name"] = "linear_decay_with_warmup"
    scheduler["t_warmup"] = "0.1dur"
    scheduler["alpha_f"] = 0.0
    if scheduler:
        new_config["scheduler"] = scheduler

    # Build the task configurations based on the provided tasks
    tasks_dict = OrderedDict()
    all_tasks = [task.value for task in TaskName]
    tasks_list = [task.value for task in tasks] if tasks else all_tasks

    for task_name in tasks_list:
        task_config = OrderedDict()
        if task_name == "mlmmlu_amateur_semipro":
            task_config["seeds"] = seeds[:4]
            task_config["trainer_kwargs"] = {"save_num_checkpoints_to_keep": 0}

        elif task_name == "mlmmlu_rookie_reserve":
            task_config["seeds"] = seeds[:3]
            task_config["trainer_kwargs"] = {"save_num_checkpoints_to_keep": 0}

        elif task_name == "eurlex":
            task_config["seeds"] = seeds[:2]
            task_config["trainer_kwargs"] = {"save_num_checkpoints_to_keep": 0}
            task_config["model_config"] = {"problem_type": "multi_label_classification"}

        elif task_name == "mnli":
            task_config["seeds"] = seeds[:3]
            task_config["trainer_kwargs"] = {"save_num_checkpoints_to_keep": 1, "max_duration": "2ep"}

        elif task_name == "boolq":
            task_config["seeds"] = seeds[:3]
            task_config["trainer_kwargs"] = {"save_num_checkpoints_to_keep": 0, "max_duration": "4ep"}

        elif task_name == "wic":
            task_config["seeds"] = seeds[:3]
            task_config["trainer_kwargs"] = {"save_num_checkpoints_to_keep": 0, "max_duration": "2ep"}

        elif task_name == "ultrafeedback":
            task_config["seeds"] = seeds[:2]
            task_config["trainer_kwargs"] = {
                "save_num_checkpoints_to_keep": 0,
                "max_duration": "1ep",
                "max_sequence_length": 1536 if fast_ultrafeedback else 2048,
            }

        elif task_name == "triviamcqa":
            task_config["seeds"] = seeds[:1]
            task_config["trainer_kwargs"] = {"save_num_checkpoints_to_keep": 0}

        else:
            print(
                f"Warning: Task '{task_name}' doesn't have eval_config defaults. Using task defaults with three seeds."
            )
            task_config["seeds"] = seeds[:3]
        tasks_dict[task_name] = task_config

    new_config["tasks"] = tasks_dict

    # Write the new configuration to a YAML file
    output_filename = output_dir / f"{ckpt_id}_evaluation.yaml"
    with output_filename.open("w") as file:
        ordered_yaml_dump(new_config, file, default_flow_style=False)

    print(f"Configuration converted and saved to {output_filename}\n")


if __name__ == "__main__":
    app()
