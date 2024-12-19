# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import copy
import gc
import multiprocessing as mp
import os
import sys
import re
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor as Pool
from multiprocessing.managers import DictProxy, SyncManager
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from urllib.parse import urlparse
from composer.optim import DecoupledAdamW
from torch.optim import AdamW

from main import param_groups_weight_decay

# Add folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import numpy as np
import omegaconf as om
import src.evals.glue_jobs as glue_jobs_module
import src.evals.misc_jobs as misc_jobs_module
import src.evals.superglue_jobs as superglue_jobs_module
import src.hf_bert as hf_bert_module
import src.mosaic_bert as mosaic_bert_module
import src.flex_bert as flex_bert_module
import torch
from composer import algorithms
from composer.callbacks import (
    LRMonitor,
    MemoryMonitor,
    OptimizerMonitor,
    RuntimeEstimator,
    SpeedMonitor,
)
from composer.loggers import WandBLogger
from composer.optim.scheduler import (
    ConstantWithWarmupScheduler,
    CosineAnnealingWithWarmupScheduler,
    LinearWithWarmupScheduler,
)
from src.scheduler import WarmupStableDecayScheduler
from composer.utils import reproducibility
from composer.utils.file_helpers import get_file
from composer.utils.object_store import S3ObjectStore
from omegaconf import DictConfig

TASK_NAME_TO_CLASS = {
    "mnli": glue_jobs_module.MNLIJob,
    "rte": glue_jobs_module.RTEJob,
    "mrpc": glue_jobs_module.MRPCJob,
    "qnli": glue_jobs_module.QNLIJob,
    "qqp": glue_jobs_module.QQPJob,
    "sst2": glue_jobs_module.SST2Job,
    "stsb": glue_jobs_module.STSBJob,
    "cola": glue_jobs_module.COLAJob,
    "boolq": superglue_jobs_module.BoolQJob,
    "cb": superglue_jobs_module.CBJob,
    "copa": superglue_jobs_module.COPAJob,
    "multirc": superglue_jobs_module.MultiRCJob,
    "wic": superglue_jobs_module.WiCJob,
    "swag": misc_jobs_module.SWAGJob,
    "eurlex": misc_jobs_module.EurlexJob,
    "ultrafeedback": misc_jobs_module.UltrafeedbackJob,
    "mlmmlu_amateur_semipro": misc_jobs_module.MLMMLUAmateurSemipro,
    "mlmmlu_rookie_reserve": misc_jobs_module.MLMMLUReserveRookie,
}

GLUE_TASKS = {"mnli", "rte", "mrpc", "qnli", "qqp", "sst2", "stsb", "cola"}
SUPERGLUE_TASKS = {"boolq", "cb", "copa", "multirc", "rte", "wic"}


def build_algorithm(name, kwargs):
    if name == "gradient_clipping":
        return algorithms.GradientClipping(**kwargs)
    elif name == "alibi":
        return algorithms.Alibi(**kwargs)
    elif name == "gated_linear_units":
        return algorithms.GatedLinearUnits(**kwargs)
    else:
        raise ValueError(f"Not sure how to build algorithm: {name}")


def build_callback(name, kwargs):
    if name == "lr_monitor":
        return LRMonitor()
    elif name == "memory_monitor":
        return MemoryMonitor()
    elif name == "speed_monitor":
        return SpeedMonitor(
            window_size=kwargs.get("window_size", 1),
            gpu_flops_available=kwargs.get("gpu_flops_available", None),
        )
    elif name == "runtime_estimator":
        return RuntimeEstimator()
    elif name == "optimizer_monitor":
        return OptimizerMonitor(
            log_optimizer_metrics=kwargs.get("log_optimizer_metrics", True),
        )
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
        return WarmupStableDecayScheduler(t_warmup=cfg.t_warmup, alpha_f=cfg.alpha_f)
    else:
        raise ValueError(f"Not sure how to build scheduler: {cfg.name}")


def build_optimizer(cfg, model):
    if cfg is None:
        return None

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


def build_model(cfg: DictConfig, num_labels: int, multiple_choice: bool = False, **kwargs):
    if cfg.name == "hf_bert":
        return hf_bert_module.create_hf_bert_classification(
            num_labels=num_labels,
            pretrained_model_name=cfg.pretrained_model_name,
            use_pretrained=cfg.get("use_pretrained", False),
            model_config=cfg.get("model_config", None),
            tokenizer_name=cfg.get("tokenizer_name", None),
            gradient_checkpointing=cfg.get("gradient_checkpointing", None),
            multiple_choice=multiple_choice,
            **kwargs,
        )
    elif cfg.name == "mosaic_bert":
        return mosaic_bert_module.create_mosaic_bert_classification(
            num_labels=num_labels,
            pretrained_model_name=cfg.pretrained_model_name,
            pretrained_checkpoint=cfg.get("pretrained_checkpoint", None),
            model_config=cfg.get("model_config", None),
            tokenizer_name=cfg.get("tokenizer_name", None),
            gradient_checkpointing=cfg.get("gradient_checkpointing", None),
            multiple_choice=multiple_choice,
            **kwargs,
        )
    elif cfg.name == "flex_bert":
        return flex_bert_module.create_flex_bert_classification(
            num_labels=num_labels,
            pretrained_model_name=cfg.pretrained_model_name,
            pretrained_checkpoint=cfg.get("pretrained_checkpoint", None),
            model_config=cfg.get("model_config", None),
            tokenizer_name=cfg.get("tokenizer_name", None),
            gradient_checkpointing=cfg.get("gradient_checkpointing", None),
            multiple_choice=multiple_choice,
            **kwargs,
        )
    else:
        raise ValueError(f"Not sure how to build model with name={cfg.name}")


def get_values_from_path(path: str, separator: str = "/") -> Dict[str, str]:
    """Parses out information from a path/string that looks like.

    ...<separator>key=value<separator...
    """
    dict_output = {}
    underscore_split = path.split(separator)
    for item in underscore_split:
        if "=" not in item:
            continue

        key, value = item.split("=")
        dict_output[key] = value
    return dict_output


def get_checkpoint_name_from_path(path: str) -> str:
    """To go from checkpoint name to path, replace | with /"""
    return path.lstrip("/").replace("/", "|")


def download_starting_checkpoint(starting_checkpoint_load_path: str, local_pretrain_checkpoints_folder: str) -> str:
    """Downloads the pretrained checkpoints to start from.

    Currently only supports S3 and URLs
    """
    load_object_store = None
    parsed_path = urlparse(starting_checkpoint_load_path)
    if parsed_path.scheme == "s3":
        load_object_store = S3ObjectStore(bucket=parsed_path.netloc)

    download_path = parsed_path.path if parsed_path.scheme == "s3" else starting_checkpoint_load_path
    os.makedirs(local_pretrain_checkpoints_folder, exist_ok=True)
    local_path = os.path.join(
        local_pretrain_checkpoints_folder,
        get_checkpoint_name_from_path(parsed_path.path),
    )
    if not os.path.exists(local_path):
        get_file(
            destination=local_path,
            path=download_path.lstrip("/"),
            object_store=load_object_store,
            progress_bar=True,
        )

    return local_path


def _setup_gpu_queue(num_gpus: int, manager: SyncManager):
    """Returns a queue with [0, 1, ..

    num_gpus].
    """
    gpu_queue = manager.Queue(num_gpus)
    for gpu_id in range(num_gpus):
        gpu_queue.put(gpu_id)
    return gpu_queue


def create_job_configs(
    main_config: om.DictConfig,
    tasks_to_run: Set[str],
    pretrained_checkpoint_path: Optional[str],
):
    configs = []
    for task_name, task_config in main_config.tasks.items():
        if main_config.get("base_run_name") is None:
            main_config.base_run_name = os.environ.get("COMPOSER_RUN_NAME", "glue")
        if task_name not in tasks_to_run:
            continue
        for task_seed in task_config.get("seeds", [main_config.default_seed]):
            run_name = f"{main_config.base_run_name}_task={task_name}_seed={str(task_seed)}"
            logger_configs = copy.deepcopy(main_config.get("loggers", {}))
            for logger_name, logger_config in logger_configs.items():
                if logger_name == "wandb":
                    # allow user set groups, otherwise set group to run name
                    if "group" not in logger_config:
                        logger_config["group"] = f"{main_config.base_run_name}_{task_name}"
                    logger_config["name"] = run_name

            # Create a copy of model config to avoid modifying the main_config
            model_kwargs = copy.deepcopy(main_config.model)
            if "model_config" not in model_kwargs:
                model_kwargs.model_config = {}
            # update with task specific model config
            model_kwargs.model_config.update(task_config.get("model_config", {}))

            task_seed_config = om.OmegaConf.create(
                {
                    "task": task_name,
                    "job_name": run_name,
                    "seed": task_seed,
                    "model": model_kwargs,
                    "tokenizer_name": main_config.tokenizer_name,
                    "scheduler": main_config.scheduler,
                    "optimizer": task_config.get("optimizer", main_config.get("optimizer", None)),
                    "load_path": pretrained_checkpoint_path,
                    "save_folder": os.path.join(
                        main_config.save_finetune_checkpoint_folder,
                        f"task={task_name}",
                        f"seed={task_seed}",
                    ),
                    "loggers": logger_configs,
                    "callbacks": main_config.get("callbacks", {}),
                    "algorithms": main_config.get("algorithms", {}),
                    "precision": main_config.get("precision", None),
                    "trainer_kwargs": task_config.trainer_kwargs,
                }
            )
            configs.append(task_seed_config)

    return configs


def run_job_worker(
    config: om.DictConfig,
    gpu_queue: Optional[mp.Queue] = None,
    process_to_gpu: Optional[DictProxy] = None,
) -> Any:
    """Instantiates the job object and runs it."""
    # need to set seed before model initialization for determinism
    reproducibility.configure_deterministic_mode()
    reproducibility.seed_all(config.seed)
    task_cls = TASK_NAME_TO_CLASS[config.task]

    model = build_model(
        config.model,
        num_labels=task_cls.num_labels,
        multiple_choice=task_cls.multiple_choice,
        custom_eval_metrics=task_cls.custom_eval_metrics,
    )

    instantiated_job = task_cls(
        job_name=config.job_name,
        seed=config.seed,
        model=model,
        tokenizer_name=config.tokenizer_name,
        scheduler=build_scheduler(config.scheduler),
        optimizer=build_optimizer(config.optimizer, model),
        load_path=config.load_path,
        save_folder=config.save_folder,
        loggers=[build_logger(name, logger_config) for name, logger_config in config.get("loggers", {}).items()],
        callbacks=[
            build_callback(name, callback_config) for name, callback_config in config.get("callbacks", {}).items()
        ],
        algorithms=[
            build_algorithm(name, algorithm_config) for name, algorithm_config in config.get("algorithms", {}).items()
        ],
        precision=config.precision,
        **config.trainer_kwargs,
    )
    results = instantiated_job.run(gpu_queue, process_to_gpu, config)

    # Extract W&B run ID from the logger
    results["wandb_name"] = None
    results["wandb_project"] = None
    results["wandb_entity"] = None

    if results["loggers"] is None:
        results["loggers"] = []
    for logger in results["loggers"]:
        if isinstance(logger, WandBLogger):
            results["wandb_run_url"] = logger.run_url
            break

    # Clean up: delete the job so that the optimizer and anything else on the gpu gets deleted
    del instantiated_job
    torch.cuda.empty_cache()
    gc.collect()
    return results


def run_jobs_parallel(configs: Sequence[om.DictConfig]) -> Dict[str, Any]:
    """Runs a list of jobs (passed in as Hydra configs) across GPUs.

    Returns a dictionary mapping job name to the result and original config
    Each job's results is a dict of:

    * 'checkpoints': list of saved_checkpoints, if any,
    * 'metrics': nested dict of results, accessed by
                 dataset and metric name, e.g.
                 ``metrics['glue_mnli']['MulticlassAccuracy']``.
    * 'job_name': The job name, helpful for keeping track of results during multiprocessing
    """
    num_gpus = torch.cuda.device_count()
    mp.set_start_method("spawn", force=True)
    torch.multiprocessing.set_start_method("spawn", force=True)
    results = []

    with mp.Manager() as manager:
        # workers get gpu ids from this queue
        # to set the GPU to run on
        gpu_queue = _setup_gpu_queue(num_gpus, manager)
        process_to_gpu = manager.dict()

        ctx = mp.get_context("spawn")
        with Pool(max_workers=min(num_gpus, len(configs)), mp_context=ctx) as pool:
            results = pool.map(
                run_job_worker,
                [config for config in configs],
                [gpu_queue for _ in configs],
                [process_to_gpu for _ in configs],
            )

    job_name_to_config = {config.job_name: config for config in configs}
    finished_results = {}
    for result in results:
        job_name = result["job_name"]
        finished_results[job_name] = {
            "result": result,
            "config": job_name_to_config[job_name],
        }

    return finished_results


def run_jobs_serial(configs) -> Dict[str, Any]:
    """Runs the jobs serially, rather than in parallel.

    Useful for debugging
    """
    results = {}
    for config in configs:
        result = run_job_worker(config)
        results[config.job_name] = {"result": result, "config": config}
    return results


def format_job_name(job_name: str) -> str:
    """Formats the job name for pretty printing."""
    dict_output = get_values_from_path(job_name, separator="_")
    return f'{dict_output["task"].upper()}(seed={dict_output["seed"]})'


def _print_table(results: Dict[str, Dict[str, Any]]):
    """Pretty prints a table given a results dictionary."""
    header = "{job_name:50}| {eval_task:25}| {name:27}|"
    hyphen_count = 50 + 25 + 27 + 11
    row_format = header + " {value:.2f}"
    print("\nCollected Job Results: \n")
    print("-" * hyphen_count)
    print(header.format(job_name="Job", eval_task="Dataset", name="Metric"))
    print("-" * hyphen_count)
    for job_name, result in results.items():
        for eval_task, eval_results in result["result"]["metrics"].items():
            for name, metric in eval_results.items():
                print(
                    row_format.format(
                        job_name=format_job_name(job_name),
                        eval_task=eval_task,
                        name=name,
                        value=metric * 100,
                    )
                )
    print("-" * hyphen_count)
    print("\n")


def _print_averaged_glue_results(glue_results: List[Tuple[str, float]]) -> None:
    """Pretty prints a table of glue results averaged across seeds."""
    header = "{job_name:50}|"
    hyphen_count = 50 + 11
    row_format = header + " {value:.2f}"
    print("\nCollected Job Results: \n")
    print("-" * hyphen_count)
    print(header.format(job_name="Task"))
    print("-" * hyphen_count)
    for task_name, result in glue_results:
        print(
            row_format.format(
                job_name=f"{task_name.upper()}",
                value=result,
            )
        )
    print("-" * hyphen_count)
    print("\n")


def train(config: om.DictConfig) -> None:
    """Main training logic.

    Args:
        config (DictConfig): Configuration composed by OmegaConf
    """
    # these subtasks require the parent task to have been run
    round_2_task_names = config.get(
        "round_2_task_names",
        {
            "mnli": {"rte", "mrpc", "stsb"},
            "swag": {"copa"},
        },
    )

    start_time = time.time()

    # Initial default seed
    reproducibility.seed_all(config.default_seed)

    # Quiet down WandB
    os.environ["WANDB_SILENT"] = "true"

    # Set tokenizer parallelism
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Confirm GPUs if parallel=True
    if config.parallel:
        assert (
            torch.cuda.device_count() > 0
        ), "Can only use parallel mode if GPUs are available. Please set parallel=False."

    # Downloads the starting checkpoint ahead of time so that
    # the different tasks don't all try to download it at the same time
    if config.get("starting_checkpoint_load_path", None):
        local_pretrain_checkpoint_path = download_starting_checkpoint(
            config.starting_checkpoint_load_path,
            config.local_pretrain_checkpoint_folder,
        )
    else:
        local_pretrain_checkpoint_path = None

    # Builds round 1 configs and runs them by first filtering out all round 2 tasks
    if round_2_task_names:
        round_2_tasks = [task for tasks in round_2_task_names.values() for task in tasks]
    else:
        round_2_tasks = []
    round_1_task_names = [task for task in TASK_NAME_TO_CLASS.keys() if task not in round_2_tasks]

    round_1_job_configs = create_job_configs(config, round_1_task_names, local_pretrain_checkpoint_path)

    round_1_results = {}
    if len(round_1_job_configs) > 0:
        if config.parallel:
            round_1_results = run_jobs_parallel(round_1_job_configs)
        else:
            round_1_results = run_jobs_serial(round_1_job_configs)

    # Builds up the information needed to run the second round, starting from the MNLI checkpoints
    checkpoint_paths = {}
    for job_name, output_dict in round_1_results.items():
        job_results = output_dict["result"]
        job_values = get_values_from_path(job_name, separator="_")
        task_name = job_values["task"]

        if task_name in checkpoint_paths:
            continue
        elif len(job_results["checkpoints"]) == 0:
            continue

        checkpoint_paths[task_name] = job_results["checkpoints"][-1]

    # Builds round 2 configs and runs them
    round_2_job_configs = []
    for dependent_task_name in round_2_task_names:
        starting_checkpoint_path = (
            checkpoint_paths[dependent_task_name]
            if dependent_task_name in checkpoint_paths
            else local_pretrain_checkpoint_path
        )
        round_2_job_configs.extend(
            create_job_configs(
                config,
                round_2_task_names[dependent_task_name],
                starting_checkpoint_path,
            )
        )

    round_2_results = {}
    if len(round_2_job_configs) > 0:
        if config.parallel:
            round_2_results = run_jobs_parallel(round_2_job_configs)
        else:
            round_2_results = run_jobs_serial(round_2_job_configs)

    end_time = time.time()

    print("-" * 30)
    print(f"Training completed in {(end_time-start_time):.2f} seconds")
    print("-" * 30)

    # Join the results and pretty print them
    all_results = {}
    all_results.update(round_1_results)
    all_results.update(round_2_results)
    _print_table(all_results)

    # Average the GLUE results across seeds and pretty print them
    task_metrics: Dict[str, List[float]] = defaultdict(list)
    task_to_run_infos: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for job_name, output_dict in all_results.items():
        result = output_dict["result"]
        job_values = get_values_from_path(job_name, separator="_")
        task_name = job_values["task"]

        # Collect W&B run information per task
        run_url = result.get("wandb_run_url")
        if run_url:
            task_to_run_infos[task_name].append({"job_name": job_name, "run_url": run_url})

        # Collect metrics per task
        for _, eval_results in result["metrics"].items():
            for _, metric in eval_results.items():
                task_metrics[task_name].append(metric * 100)

    # Compute average metrics per task
    results_mean: Dict[str, float] = {task_name: float(np.mean(values)) for task_name, values in task_metrics.items()}

    overall_glue = []
    overall_superglue = []
    overall_other = []

    for task_name, average_metric in results_mean.items():
        # Classify tasks into GLUE, SuperGLUE, or other
        if task_name in GLUE_TASKS:
            overall_glue.append(average_metric)
        elif task_name in SUPERGLUE_TASKS:
            overall_superglue.append(average_metric)
        else:
            overall_other.append(average_metric)

        # Update W&B runs with average metrics
        for run_info in task_to_run_infos.get(task_name, []):
            match = re.search(r"([^/]+)/([^/]+)/runs/([^/]+)", run_info["run_url"])
            if match:
                import wandb

                api = wandb.Api()
                run = api.run(f"{match.group(1)}/{match.group(2)}/{match.group(3)}")

                # Update the run's summary with the average metric
                run.summary[f"average_{task_name}"] = average_metric
                run.update()

    if len(overall_other) > 0:
        other_results_mean = {k: v for k, v in results_mean.items() if k not in GLUE_TASKS.union(SUPERGLUE_TASKS)}
        _print_averaged_glue_results([(key, value) for key, value in other_results_mean.items()])

    if len(overall_glue) > 0:
        glue_results_mean = {
            **{k: v for k, v in results_mean.items() if k in GLUE_TASKS},
            "glue": float(np.mean(overall_glue)),
        }
        _print_averaged_glue_results([(key, value) for key, value in glue_results_mean.items()])

    if len(overall_superglue) > 0:
        superglue_results_mean = {
            **{k: v for k, v in results_mean.items() if k in SUPERGLUE_TASKS},
            "superglue": float(np.mean(overall_superglue)),
        }
        _print_averaged_glue_results([(key, value) for key, value in superglue_results_mean.items()])


if __name__ == "__main__":
    yaml_path, args_list = sys.argv[1], sys.argv[2:]

    with open(yaml_path) as f:
        yaml_cfg = om.OmegaConf.load(f)

    cli_cfg = om.OmegaConf.from_cli(args_list)
    cfg = om.OmegaConf.merge(yaml_cfg, cli_cfg)

    if cfg.model.name == "mosaic_bert":
        with open("yamls/defaults.yaml") as f:
            default_cfg = om.OmegaConf.load(f)
        cfg = om.OmegaConf.merge(cfg, default_cfg)

    assert isinstance(cfg, om.DictConfig)
    train(cfg)
