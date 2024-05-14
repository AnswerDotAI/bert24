# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""A starter script for fine-tuning a BERT model on your own dataset."""

import os
import sys
from typing import Optional, cast

# Add folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import src.glue.data as data_module
import src.hf_bert as hf_bert_module
import src.mosaic_bert as mosaic_bert_module
import transformers
from composer import Trainer, algorithms
from composer.callbacks import (HealthChecker, LRMonitor, MemoryMonitor,
                                OptimizerMonitor, RuntimeEstimator,
                                SpeedMonitor)
from composer.core.types import Dataset
from composer.loggers import WandBLogger
from composer.optim import DecoupledAdamW
from composer.optim.scheduler import (ConstantWithWarmupScheduler,
                                      CosineAnnealingWithWarmupScheduler,
                                      LinearWithWarmupScheduler)
from composer.utils import dist, reproducibility
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from torch.utils.data import DataLoader


def update_batch_size_info(cfg: DictConfig):
    global_batch_size, device_microbatch_size = cfg.global_train_batch_size, cfg.device_train_microbatch_size
    if global_batch_size % dist.get_world_size() != 0:
        raise ValueError(
            f'Global batch size {global_batch_size} is not divisible by {dist.get_world_size()} '
            'as a result, the batch size would be truncated, please adjust `global_batch_size` '
            f'to be divisible by world size, {dist.get_world_size()}.')
    device_train_batch_size = global_batch_size // dist.get_world_size()
    if isinstance(device_microbatch_size, int):
        if device_microbatch_size > device_train_batch_size:
            print(
                f'WARNING: device_train_microbatch_size > device_train_batch_size, '
                f'will be reduced from {device_microbatch_size} -> {device_train_batch_size}.'
            )
            device_microbatch_size = device_train_batch_size
    cfg.n_gpus = dist.get_world_size()
    cfg.device_train_batch_size = device_train_batch_size
    cfg.device_train_microbatch_size = device_microbatch_size
    # Safely set `device_eval_batch_size` if not provided by user
    if 'device_eval_batch_size' not in cfg:
        if cfg.device_train_microbatch_size == 'auto':
            cfg.device_eval_batch_size = 1
        else:
            cfg.device_eval_batch_size = cfg.device_train_microbatch_size
    return cfg


def log_config(cfg: DictConfig):
    print(om.to_yaml(cfg))
    if 'wandb' in cfg.get('loggers', {}):
        try:
            import wandb
        except ImportError as e:
            raise e
        if wandb.run:
            wandb.config.update(om.to_container(cfg, resolve=True))


def build_algorithm(name, kwargs):
    if name == 'gradient_clipping':
        return algorithms.GradientClipping(**kwargs)
    elif name == 'alibi':
        return algorithms.Alibi(**kwargs)
    elif name == 'fused_layernorm':
        return algorithms.FusedLayerNorm(**kwargs)
    elif name == 'gated_linear_units':
        return algorithms.GatedLinearUnits(**kwargs)
    elif name == 'low_precision_layernorm':
        return algorithms.LowPrecisionLayerNorm(**kwargs)
    else:
        raise ValueError(f'Not sure how to build algorithm: {name}')


def build_callback(name, kwargs):
    if name == 'lr_monitor':
        return LRMonitor()
    elif name == 'memory_monitor':
        return MemoryMonitor()
    elif name == 'speed_monitor':
        return SpeedMonitor(window_size=kwargs.get('window_size', 1),
                            gpu_flops_available=kwargs.get(
                                'gpu_flops_available', None))
    elif name == 'runtime_estimator':
        return RuntimeEstimator()
    elif name == 'optimizer_monitor':
        return OptimizerMonitor(log_optimizer_metrics=kwargs.get(
            'log_optimizer_metrics', True),)
    elif name == 'health_checker':
        return HealthChecker(**kwargs)
    else:
        raise ValueError(f'Not sure how to build callback: {name}')


def build_logger(name, kwargs):
    if name == 'wandb':
        return WandBLogger(**kwargs)
    else:
        raise ValueError(f'Not sure how to build logger: {name}')


def build_scheduler(cfg):
    if cfg.name == 'constant_with_warmup':
        return ConstantWithWarmupScheduler(t_warmup=cfg.t_warmup)
    elif cfg.name == 'cosine_with_warmup':
        return CosineAnnealingWithWarmupScheduler(t_warmup=cfg.t_warmup,
                                                  alpha_f=cfg.alpha_f)
    elif cfg.name == 'linear_decay_with_warmup':
        return LinearWithWarmupScheduler(t_warmup=cfg.t_warmup,
                                         alpha_f=cfg.alpha_f)
    else:
        raise ValueError(f'Not sure how to build scheduler: {cfg.name}')


def build_optimizer(cfg, model):
    if cfg.name == 'decoupled_adamw':
        return DecoupledAdamW(model.parameters(),
                              lr=cfg.lr,
                              betas=cfg.betas,
                              eps=cfg.eps,
                              weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f'Not sure how to build optimizer: {cfg.name}')


def build_my_dataloader(cfg: DictConfig, device_batch_size: int):
    """Create a dataloader for classification.

    **Modify this function to train on your own dataset!**

    This function is provided as a starter code to simplify fine-tuning a BERT
    classifier on your dataset. We'll use the dataset for QNLI (one of the
    GLUE tasks) as a demonstration.

    Args:
        cfg (DictConfig): An omegaconf config that houses all the configuration
            variables needed to instruct dataset/dataloader creation.
        device_batch_size (int): The size of the batches that the dataloader
            should produce.

    Returns:
        dataloader: A dataloader set up for use of the Composer Trainer.
    """
    # As a demonstration, we're using the QNLI dataset from the GLUE suite
    # of tasks.
    #
    # Note: We create our dataset using the `data_module.create_glue_dataset` utility
    #   defined in `./src/glue/data.py`. If you inspect that code, you'll see
    #   that we're taking some extra steps so that our dataset yields examples
    #   that follow a particular format. In particular, the raw text is
    #   tokenized and some of the data columns are removed. The result is that
    #   each example is a dictionary with the following:
    #
    #     - 'input_ids': the tokenized raw text
    #     - 'label': the target class that the text belongs to
    #     - 'attention_mask': a list of 1s and 0s to indicate padding
    #
    # When you set up your own dataset, it should handle tokenization to yield
    # examples with a similar structure!
    #
    # REPLACE THIS WITH YOUR OWN DATASET:
    dataset = data_module.create_glue_dataset(
        task='qnli',
        split=cfg.split,
        tokenizer_name=cfg.tokenizer_name,
        max_seq_length=cfg.max_seq_len,
    )

    dataset = cast(Dataset, dataset)
    dataloader = DataLoader(
        dataset,
        # As an alternative to formatting the examples inside the dataloader,
        # you can write a custom data collator to do that instead.
        collate_fn=transformers.default_data_collator,
        batch_size=device_batch_size,
        sampler=dist.get_sampler(dataset,
                                 drop_last=cfg.drop_last,
                                 shuffle=cfg.shuffle),
        num_workers=cfg.num_workers,
        pin_memory=cfg.get('pin_memory', True),
        prefetch_factor=cfg.get('prefetch_factor', 2),
        persistent_workers=cfg.get('persistent_workers', True),
        timeout=cfg.get('timeout', 0),
    )

    return dataloader


def build_model(cfg: DictConfig):
    # Note: cfg.num_labels should match the number of classes in your dataset!
    if cfg.name == 'hf_bert':
        return hf_bert_module.create_hf_bert_classification(
            num_labels=cfg.num_labels,
            pretrained_model_name=cfg.pretrained_model_name,
            use_pretrained=cfg.get('use_pretrained', False),
            model_config=cfg.get('model_config'),
            tokenizer_name=cfg.get('tokenizer_name'),
            gradient_checkpointing=cfg.get('gradient_checkpointing'))
    elif cfg.name == 'mosaic_bert':
        return mosaic_bert_module.create_mosaic_bert_classification(
            num_labels=cfg.num_labels,
            pretrained_model_name=cfg.pretrained_model_name,
            pretrained_checkpoint=cfg.get('pretrained_checkpoint'),
            model_config=cfg.get('model_config'),
            tokenizer_name=cfg.get('tokenizer_name'),
            gradient_checkpointing=cfg.get('gradient_checkpointing'))
    else:
        raise ValueError(f'Not sure how to build model with name={cfg.name}')


def main(cfg: DictConfig,
         return_trainer: bool = False,
         do_train: bool = True) -> Optional[Trainer]:
    print('Training using config: ')
    print(om.to_yaml(cfg))
    reproducibility.seed_all(cfg.seed)

    # Get batch size info
    cfg = update_batch_size_info(cfg)

    # Build Model
    print('Initializing model...')
    model = build_model(cfg.model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'{n_params=:.4e}')

    # Dataloaders
    print('Building train loader...')
    train_loader = build_my_dataloader(
        cfg.train_loader,
        cfg.global_train_batch_size // dist.get_world_size(),
    )
    print('Building eval loader...')
    global_eval_batch_size = cfg.get('global_eval_batch_size',
                                     cfg.global_train_batch_size)
    eval_loader = build_my_dataloader(
        cfg.eval_loader,
        global_eval_batch_size // dist.get_world_size(),
    )

    # Optimizer
    optimizer = build_optimizer(cfg.optimizer, model)

    # Scheduler
    scheduler = build_scheduler(cfg.scheduler)

    # Loggers
    loggers = [
        build_logger(name, logger_cfg)
        for name, logger_cfg in cfg.get('loggers', {}).items()
    ]

    # Callbacks
    callbacks = [
        build_callback(name, callback_cfg)
        for name, callback_cfg in cfg.get('callbacks', {}).items()
    ]

    # Algorithms
    algorithms = [
        build_algorithm(name, algorithm_cfg)
        for name, algorithm_cfg in cfg.get('algorithms', {}).items()
    ]

    if cfg.get('run_name') is None:
        cfg.run_name = os.environ.get('COMPOSER_RUN_NAME',
                                      'sequence-classification')

    # Build the Trainer
    trainer = Trainer(
        run_name=cfg.run_name,
        seed=cfg.seed,
        model=model,
        algorithms=algorithms,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        train_subset_num_batches=cfg.get('train_subset_num_batches', -1),
        eval_subset_num_batches=cfg.get('eval_subset_num_batches', -1),
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
        device=cfg.get('device'),
        device_train_microbatch_size=cfg.get('device_train_microbatch_size',
                                             'auto'),
        save_folder=cfg.get('save_folder'),
        save_interval=cfg.get('save_interval', '1000ba'),
        save_num_checkpoints_to_keep=cfg.get('save_num_checkpoints_to_keep',
                                             -1),
        save_overwrite=cfg.get('save_overwrite', False),
        load_path=cfg.get('load_path'),
        load_weights_only=True,
    )

    print('Logging config...')
    log_config(cfg)

    if do_train:
        print('Starting training...')
        trainer.fit()

    if return_trainer:
        return trainer


if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)
    cfg = cast(DictConfig, cfg)  # for type checking
    main(cfg)
