# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import warnings
from pathlib import Path
from typing import Optional, cast

import torch
from torch import nn

from src.bert_layers.configuration_bert import FlexBertConfig
from src.bert_layers.model import init_mlm_model_from_pretrained

# Add folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from composer import Evaluator, Trainer, algorithms
from composer.callbacks import LRMonitor, MemoryMonitor, OptimizerMonitor, RuntimeEstimator, SpeedMonitor
from composer.core import DataSpec
from composer.loggers import WandBLogger
from composer.optim import DecoupledAdamW
from composer.optim.scheduler import (
    ConstantWithWarmupScheduler,
    CosineAnnealingWithWarmupScheduler,
    LinearWithWarmupScheduler,
)
from composer.utils import dist, reproducibility
from composer.utils.checkpoint import _ensure_valid_checkpoint
from omegaconf import DictConfig, OmegaConf
from omegaconf import OmegaConf as om
from torch.optim import AdamW

import src.flex_bert as flex_bert_module
import src.hf_bert as hf_bert_module
import src.mosaic_bert as mosaic_bert_module
import src.text_data as text_data_module
from src.algorithms.rope_schedule import FlexBertRopeSchedule
from src.callbacks.dataloader_speed import DataloaderSpeedMonitor
from src.callbacks.log_grad_norm import LogGradNorm
from src.callbacks.packing_efficiency import PackingEfficency
from src.callbacks.scheduled_gc import ScheduledGarbageCollector
from src.scheduler import CosineInverseSqrtScheduler, OneMinusSqrtScheduler, WarmupStableDecayScheduler
from src.sequence_packer import get_num_samples_in_packed_batch, split_packed_batch


def update_batch_size_info(cfg: DictConfig):
    global_batch_size, device_microbatch_size = cfg.global_train_batch_size, cfg.device_train_microbatch_size
    if global_batch_size % dist.get_world_size() != 0:
        raise ValueError(
            f"Global batch size {global_batch_size} is not divisible by {dist.get_world_size()} "
            "as a result, the batch size would be truncated, please adjust `global_batch_size` "
            f"to be divisible by world size, {dist.get_world_size()}."
        )
    device_train_batch_size = global_batch_size // dist.get_world_size()
    if isinstance(device_microbatch_size, int):
        if device_microbatch_size > device_train_batch_size:
            print(
                f"WARNING: device_train_microbatch_size > device_train_batch_size, "
                f"will be reduced from {device_microbatch_size} -> {device_train_batch_size}."
            )
            device_microbatch_size = device_train_batch_size
    cfg.n_gpus = dist.get_world_size()
    cfg.device_train_batch_size = device_train_batch_size
    cfg.device_train_microbatch_size = device_microbatch_size

    # Safely set `device_eval_microbatch_size` if not provided by user
    if "device_eval_microbatch_size" not in cfg:
        if cfg.device_train_microbatch_size == "auto":
            cfg.device_eval_microbatch_size = 1
        else:
            cfg.device_eval_microbatch_size = cfg.device_train_microbatch_size

    global_eval_batch_size, device_eval_microbatch_size = (
        cfg.get("global_eval_batch_size", global_batch_size),
        cfg.device_eval_microbatch_size,
    )
    device_eval_batch_size = global_eval_batch_size // dist.get_world_size()
    if isinstance(device_eval_microbatch_size, int):
        if device_eval_microbatch_size > device_eval_microbatch_size:
            print(
                f"WARNING: device_eval_microbatch_size > device_eval_batch_size, "
                f"will be reduced from {device_eval_microbatch_size} -> {device_eval_batch_size}."
            )
            device_eval_microbatch_size = device_eval_batch_size
    cfg.device_eval_batch_size = device_eval_batch_size
    cfg.device_eval_microbatch_size = device_eval_microbatch_size
    return cfg


# from timm: https://github.com/huggingface/pytorch-image-models/blob/main/timm/optim/optim_factory.py
# Copyright 2019 Ross Wightman, Apache-2.0 License
def param_groups_weight_decay(model: nn.Module, weight_decay=1e-5, no_weight_decay_list=()):
    no_weight_decay_list = set(no_weight_decay_list)
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [{"params": no_decay, "weight_decay": 0.0}, {"params": decay, "weight_decay": weight_decay}]


def log_config(cfg: DictConfig):
    print(om.to_yaml(cfg))
    if "wandb" in cfg.get("loggers", {}):
        try:
            import wandb
        except ImportError as e:
            raise e
        if wandb.run:
            wandb.config.update(om.to_container(cfg, resolve=True))


def build_algorithm(name, kwargs):
    if name == "gradient_clipping":
        return algorithms.GradientClipping(**kwargs)
    elif name == "alibi":
        return algorithms.Alibi(**kwargs)
    elif name == "gated_linear_units":
        return algorithms.GatedLinearUnits(**kwargs)
    elif name == "ema":
        return algorithms.EMA(
            half_life=kwargs.get("half_life", "1000ba"),
            smoothing=kwargs.get("smoothing", None),
            ema_start=kwargs.get("ema_start", "0.0dur"),
            update_interval=kwargs.get("update_interval", None),
        )
    elif name == "rope_schedule":
        return FlexBertRopeSchedule(
            min_rope_theta=kwargs.get("min_rope_theta", 10_000),
            max_rope_theta=kwargs.get("max_rope_theta", 80_000),
            warmup_tokens=kwargs.get("warmup_tokens", 25_000_000),
            rope_theta_increment=kwargs.get("rope_theta_increment", 10_000),
            batch_log_interval=kwargs.get("batch_log_interval", 10),
            increment_theta_immediately=kwargs.get("increment_theta_immediately", False),
        )
    else:
        raise ValueError(f"Not sure how to build algorithm: {name}")


def build_callback(name, kwargs):
    if name == "lr_monitor":
        return LRMonitor()
    elif name == "memory_monitor":
        return MemoryMonitor()
    elif name == "speed_monitor":
        return SpeedMonitor(
            window_size=kwargs.get("window_size", 1), gpu_flops_available=kwargs.get("gpu_flops_available", None)
        )
    elif name == "runtime_estimator":
        return RuntimeEstimator()
    elif name == "optimizer_monitor":
        return OptimizerMonitor(
            log_optimizer_metrics=kwargs.get("log_optimizer_metrics", True),
        )
    elif name == "scheduled_gc":
        return ScheduledGarbageCollector(batch_interval=kwargs.get("batch_interval", 100_000))
    elif name == "log_grad_norm":
        return LogGradNorm(
            log_optimizer_metrics=kwargs.get("log_optimizer_metrics", True),
            batch_log_interval=kwargs.get("batch_log_interval", 10),
        )
    elif name == "dataloader_speed":
        return DataloaderSpeedMonitor()
    elif name == "packing_efficiency":
        return PackingEfficency(log_interval=kwargs.get("log_interval", 10))
    else:
        raise ValueError(f"Not sure how to build callback: {name}")


def build_logger(name, kwargs):
    if name == "wandb":
        return WandBLogger(**kwargs)
    else:
        raise ValueError(f"Not sure how to build logger: {name}")


def build_scheduler(cfg):
    if cfg.name == "constant_with_warmup":
        return ConstantWithWarmupScheduler(t_warmup=cfg.t_warmup)
    elif cfg.name == "cosine_with_warmup":
        return CosineAnnealingWithWarmupScheduler(t_warmup=cfg.t_warmup, alpha_f=cfg.alpha_f)
    elif cfg.name == "linear_decay_with_warmup":
        return LinearWithWarmupScheduler(t_warmup=cfg.t_warmup, alpha_f=cfg.alpha_f)
    elif cfg.name == "warmup_stable_decay":
        return WarmupStableDecayScheduler(
            t_warmup=cfg.t_warmup, alpha_f=cfg.alpha_f, t_decay=cfg.get("t_decay", "0.1dur")
        )
    elif cfg.name == "cosine_inverse_sqrt":
        return CosineInverseSqrtScheduler(
            t_warmup=cfg.t_warmup,
            t_cooldown=cfg.t_cooldown,
            t_cosine=cfg.get("t_cosine", "0.25dur"),
            alpha_f=cfg.alpha_f,
            alpha_s=cfg.get("alpha_s", 0.0),
            warmup_schedule=cfg.get("warmup_schedule", "linear"),
            cooldown_schedule=cfg.get("cooldown_schedule", "linear"),
        )
    elif cfg.name == "one_minus_sqrt":
        return OneMinusSqrtScheduler(t_decay=cfg.t_decay, t_max=cfg.t_max, alpha_f=cfg.alpha_f)
    else:
        raise ValueError(f"Not sure how to build scheduler: {cfg.name}")


def build_optimizer(cfg, model):
    if cfg.get("filter_bias_norm_wd", False):
        params = param_groups_weight_decay(model, weight_decay=cfg.weight_decay)
    else:
        params = model.parameters()

    if cfg.name == "decoupled_adamw":
        return DecoupledAdamW(params, lr=cfg.lr, betas=list(cfg.betas), eps=cfg.eps, weight_decay=cfg.weight_decay)
    elif cfg.name == "adamw":
        print(
            "INFO: You might want to increase the weight decay because in AdamW it is scaled by the lr."
            f" Default weight decay is ``1e-2`` -> {cfg.weight_decay}. Default lr is `lr=1e-3` -> {cfg.lr}."
        )
        return AdamW(params, lr=cfg.lr, betas=list(cfg.betas), eps=cfg.eps, weight_decay=cfg.weight_decay)
    elif cfg.name == "stableadamw":
        try:
            if cfg.get("log_grad_norm", False):
                from src.optimizer import StableAdamW
            else:
                from optimi import StableAdamW
        except ImportError:
            raise ImportError("Install `pip install torch-optimi` to use the StableAdamW optimizer.")

        print(
            "INFO: You might want to increase the weight decay because in StableAdamW it is scaled by the lr."
            f" Default weight decay is ``1e-2`` -> {cfg.weight_decay}. Default lr is `lr=1e-3` -> {cfg.lr}."
        )
        return StableAdamW(params, lr=cfg.lr, betas=list(cfg.betas), eps=cfg.eps, weight_decay=cfg.weight_decay)
    elif cfg.name == "decoupled_stableadamw":
        try:
            if cfg.get("log_grad_norm", False):
                from src.optimizer import StableAdamW
            else:
                from optimi import StableAdamW
        except ImportError:
            raise ImportError("Install `pip install torch-optimi` to use the StableAdamW optimizer.")

        return StableAdamW(
            params,
            lr=cfg.lr,
            betas=list(cfg.betas),
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
            decouple_lr=True,
        )
    else:
        raise ValueError(f"Not sure how to build optimizer: {cfg.name}")


def get_num_tokens_in_batch_unpadded(batch: dict):
    return batch["attention_mask"].sum().item()


def build_dataloader(
    cfg,
    tokenizer,
    device_batch_size,
    count_padding_tokens=True,
    device_microbatch_size: int | None = None,
):
    split_batch_fn = None
    num_samples_in_batch_fn = None
    num_tokens_in_batch_fn = None

    if cfg.name == "text":
        data_loader = text_data_module.build_text_dataloader(
            cfg,
            tokenizer,
            device_batch_size,
            device_microbatch_size=device_microbatch_size,
        )
    else:
        raise ValueError(f"Not sure how to build dataloader with config: {cfg}")

    if not count_padding_tokens:
        num_tokens_in_batch_fn = get_num_tokens_in_batch_unpadded
    if cfg.get("sequence_packing", False):
        split_batch_fn = split_packed_batch
        num_samples_in_batch_fn = get_num_samples_in_packed_batch

    data_loader = DataSpec(
        data_loader,
        get_num_tokens_in_batch=num_tokens_in_batch_fn,
        split_batch=split_batch_fn,
        get_num_samples_in_batch=num_samples_in_batch_fn,
    )
    return data_loader


def build_model(cfg: DictConfig):
    if cfg.name == "hf_bert":
        return hf_bert_module.create_hf_bert_mlm(
            pretrained_model_name=cfg.pretrained_model_name,
            use_pretrained=cfg.get("use_pretrained", None),
            model_config=cfg.get("model_config", None),
            tokenizer_name=cfg.get("tokenizer_name", None),
            gradient_checkpointing=cfg.get("gradient_checkpointing", None),
        )
    elif cfg.name == "mosaic_bert":
        return mosaic_bert_module.create_mosaic_bert_mlm(
            pretrained_model_name=cfg.pretrained_model_name,
            pretrained_checkpoint=cfg.get("pretrained_checkpoint", None),
            model_config=cfg.get("model_config", None),
            tokenizer_name=cfg.get("tokenizer_name", None),
            gradient_checkpointing=cfg.get("gradient_checkpointing", None),
        )
    elif cfg.name == "flex_bert":
        return flex_bert_module.create_flex_bert_mlm(
            pretrained_model_name=cfg.pretrained_model_name,
            pretrained_checkpoint=cfg.get("pretrained_checkpoint", None),
            model_config=cfg.get("model_config", None),
            tokenizer_name=cfg.get("tokenizer_name", None),
            gradient_checkpointing=cfg.get("gradient_checkpointing", None),
            recompute_metric_loss=cfg.get("recompute_metric_loss", False),
            disable_train_metrics=cfg.get("disable_train_metrics", False),
        )
    else:
        raise ValueError(f"Not sure how to build model with name={cfg.name}")


def init_from_checkpoint(cfg: DictConfig, new_model: nn.Module):
    print(f"Initializing model from checkpoint {cfg.checkpoint_run_name}")
    checkpoint_cfg = Path(cfg.checkpoint_cfg)
    assert checkpoint_cfg.exists(), f"Checkpoint config {checkpoint_cfg} does not exist"
    pretrained_cfg = om.load(checkpoint_cfg)

    pretrained_model = build_model(pretrained_cfg.model)
    n_params = sum(p.numel() for p in pretrained_model.parameters())

    checkpoint_filepath = Path(cfg.checkpoint_load_path) / f"{cfg.checkpoint_run_name}" / "latest-rank0.pt"
    assert checkpoint_filepath.exists(), f"Checkpoint {checkpoint_filepath} does not exist"
    state = torch.load(_ensure_valid_checkpoint(checkpoint_filepath), map_location="cpu")

    state_dict = state.get("state", {})
    model_state = state_dict.get("model", {})
    assert len(model_state) > 0, "Model state is empty, please check the checkpoint and checkpoint path"

    pretrained_model.load_state_dict(model_state)

    if isinstance(pretrained_cfg.model.model_config, DictConfig):
        model_config = OmegaConf.to_container(pretrained_cfg.model.model_config, resolve=True)
    pretrained_config = FlexBertConfig.from_pretrained(pretrained_cfg.model.pretrained_model_name, **model_config)

    init_mlm_model_from_pretrained(
        config=pretrained_config,
        pretrained_model=pretrained_model.model,
        new_model=new_model.model,
        mode=cfg.get("mode", "tile_weights_from_middle"),
    )
    print(f"Initialized model from checkpoint {cfg.checkpoint_run_name} with {n_params=:.4e} parameters")


def main(cfg: DictConfig, return_trainer: bool = False, do_train: bool = True) -> Optional[Trainer]:
    print("Training using config: ")
    print(om.to_yaml(cfg))
    reproducibility.seed_all(cfg.seed)

    # Get batch size info
    cfg = update_batch_size_info(cfg)

    # Build Model
    print("Initializing model...")
    model = build_model(cfg.model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{n_params=:.4e}")

    if cfg.get("init_from_checkpoint", None) is not None:
        init_from_checkpoint(cfg.init_from_checkpoint, model)

    # Dataloaders
    print("Building train loader...")
    train_loader = build_dataloader(
        cfg=cfg.train_loader,
        tokenizer=model.tokenizer,
        device_batch_size=cfg.global_train_batch_size // dist.get_world_size(),
        count_padding_tokens=cfg.get("count_padding_tokens", True),
        device_microbatch_size=cfg.device_train_microbatch_size,
    )
    if cfg.get("eval_loader", None) is not None:
        print("Building eval loader...")
        global_eval_batch_size = cfg.get("global_eval_batch_size", cfg.global_train_batch_size)
        eval_loader = build_dataloader(
            cfg=cfg.eval_loader,
            tokenizer=model.tokenizer,
            device_batch_size=cfg.get("device_eval_batch_size", global_eval_batch_size // dist.get_world_size()),
        )
        eval_evaluator = Evaluator(
            label="eval",
            dataloader=eval_loader,
            device_eval_microbatch_size=cfg.get("device_eval_microbatch_size", None),
        )
    else:
        eval_evaluator = None

    # Optimizer
    optimizer = build_optimizer(cfg.optimizer, model)

    # Scheduler
    scheduler = build_scheduler(cfg.scheduler)

    # Loggers
    loggers = [build_logger(name, logger_cfg) for name, logger_cfg in cfg.get("loggers", {}).items()]

    # Callbacks
    callbacks = [build_callback(name, callback_cfg) for name, callback_cfg in cfg.get("callbacks", {}).items()]

    # Algorithms
    if (
        cfg.get("algorithms", {}).get("gradient_clipping", {}).get("clipping_threshold", 0) > 0
    ) and "stableadamw" in cfg.get("optimizer", {}).get("name", "adamw"):
        warnings.warn(
            f"The StableAdamW optimizer replaces gradient clipping. "
            f"Set {cfg['algorithms']['gradient_clipping']['clipping_threshold']=} to 0.0"
        )

    algorithms = [build_algorithm(name, algorithm_cfg) for name, algorithm_cfg in cfg.get("algorithms", {}).items()]

    if cfg.get("run_name") is None:
        cfg.run_name = os.environ.get("COMPOSER_RUN_NAME", "bert")

    # Build the Trainer
    trainer = Trainer(
        run_name=cfg.run_name,
        seed=cfg.seed,
        model=model,
        algorithms=algorithms,
        train_dataloader=train_loader,
        eval_dataloader=eval_evaluator,
        train_subset_num_batches=cfg.get("train_subset_num_batches", -1),
        eval_subset_num_batches=cfg.get("eval_subset_num_batches", -1),
        optimizers=optimizer,
        schedulers=scheduler,
        max_duration=cfg.max_duration,
        eval_interval=cfg.eval_interval,
        progress_bar=cfg.progress_bar,
        log_to_console=cfg.log_to_console,
        console_log_interval=cfg.console_log_interval,
        loggers=loggers,
        callbacks=callbacks,
        precision=cfg.precision,
        device=cfg.get("device", None),
        device_train_microbatch_size=cfg.get("device_train_microbatch_size", "auto"),
        save_folder=cfg.get("save_folder", None),
        save_interval=cfg.get("save_interval", "1000ba"),
        save_num_checkpoints_to_keep=cfg.get("save_num_checkpoints_to_keep", -1),
        save_overwrite=cfg.get("save_overwrite", False),
        load_path=cfg.get("load_path", None),
        load_weights_only=cfg.get("load_weights_only", False),
        python_log_level=cfg.get("python_log_level", None),
        autoresume=cfg.get("autoresume", None),
        fsdp_config=cfg.get("fsdp_config", None),
        compile_config=cfg.get("compile_config", None),
    )

    print("Logging config...")
    log_config(cfg)

    if do_train:
        print("Starting training...")
        # this section is intended to use when resuming from a checkpoint where one wants to change
        # the learning rate and weight deacy. It's only been tested with the warmup_stable_decay scheduler
        if cfg.get("restart_override", False):
            print("Overriding checkpoint's scheduler & optimizer LR & WD, and train microbatch size with config options")  # fmt: skip
            if cfg.scheduler.name not in ["constant_with_warmup", "warmup_stable_decay"]:
                print("Rescaling current step LR by ratio of new LR to old LR. This may require scaling the scheduler's alpha_f")  # fmt: skip
                for param_group in trainer.state.optimizers[0].param_groups:
                    lr_ratio = cfg.optimizer.lr / param_group["lr"]
                    param_group["lr"] = cfg.optimizer.lr
                    param_group["weight_decay"] = cfg.optimizer.weight_decay if param_group["weight_decay"] > 0 else 0.0
                for scheduler in trainer.state.schedulers:
                    for i in range(len(scheduler.base_lrs)):
                        scheduler.base_lrs[i] *= lr_ratio
                    for i in range(len(scheduler._last_lr)):
                        scheduler._last_lr[i] *= lr_ratio
            else:
                for param_group in trainer.state.optimizers[0].param_groups:
                    param_group["lr"] = cfg.optimizer.lr
                    param_group["weight_decay"] = cfg.optimizer.weight_decay if param_group["weight_decay"] > 0 else 0.0
                for scheduler in trainer.state.schedulers:
                    for i in range(len(scheduler.base_lrs)):
                        scheduler.base_lrs[i] = cfg.optimizer.lr
                    for i in range(len(scheduler._last_lr)):
                        scheduler._last_lr[i] = cfg.optimizer.lr
            trainer.fit(
                device_train_microbatch_size=cfg.get("device_train_microbatch_size", "auto"),
                reset_time=cfg.get("reset_time", False),
            )
        else:
            trainer.fit(reset_time=cfg.get("reset_time", False))

    if return_trainer:
        return trainer


if __name__ == "__main__":
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open("yamls/defaults.yaml") as f:
        default_cfg = om.load(f)
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(default_cfg, yaml_cfg, cli_cfg)
    cfg = cast(DictConfig, cfg)  # for type checking
    main(cfg)
