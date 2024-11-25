# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

# """Contains GLUE job objects for the simple_glue_trainer."""
import atexit
import copy
import gc
import multiprocessing as mp
import os
import sys
from multiprocessing import managers
from typing import Any, Dict, List, Optional, Union, cast

import torch
import transformers
from composer import ComposerModel
from composer.core import Callback
from composer.core.types import Dataset
from composer.devices import Device, DeviceGPU
from composer.loggers import LoggerDestination
from composer.optim import ComposerScheduler
from composer.trainer.trainer import Trainer
from composer.utils import dist, reproducibility
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from torch.optim import Optimizer
from torch.utils.data import DataLoader

# Add glue folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.realpath(__file__)))


def multiple_choice_collate_fn(features):
    label_name = "label" if "label" in features[0].keys() else "labels"
    labels = [feature.pop(label_name) for feature in features]
    batch_size = len(features)
    num_choices = len(features[0]["input_ids"])
    flattened_features = [
        [{k: v[i] for k, v in feature.items() if isinstance(v, list)} for i in range(num_choices)]
        for feature in features
    ]
    flattened_features = sum(flattened_features, [])

    batch = {"input_ids": [], "attention_mask": [], "token_type_ids": []}

    for feature in flattened_features:
        for k, v in feature.items():
            batch[k].append(v)

    batch = {k: torch.tensor(v).view(batch_size, num_choices, -1) for k, v in batch.items()}
    batch["labels"] = torch.tensor(labels, dtype=torch.int64)
    return batch


def build_dataloader(dataset, collate_fn=None, **kwargs):
    dataset = cast(Dataset, dataset)

    return DataLoader(
        dataset=dataset,
        sampler=dist.get_sampler(dataset, drop_last=False, shuffle=True),
        collate_fn=(transformers.default_data_collator if collate_fn is None else collate_fn),
        **kwargs,
    )


Metrics = Dict[str, Dict[str, Any]]


def reset_trainer(trainer: Trainer, garbage_collect: bool = False):
    """Cleans up memory usage left by trainer."""
    trainer.close()
    # Unregister engine from atexit to remove ref
    atexit.unregister(trainer.engine._close)
    # Close potentially persistent dataloader workers
    loader = trainer.state.train_dataloader
    if loader and loader._iterator is not None:  # type: ignore
        loader._iterator._shutdown_workers()  # type: ignore
    # Explicitly delete attributes of state as otherwise gc.collect() doesn't free memory
    for key in list(trainer.state.__dict__.keys()):
        delattr(trainer.state, key)
    # Delete the rest of trainer attributes
    for key in list(trainer.__dict__.keys()):
        delattr(trainer, key)
    if garbage_collect:
        gc.collect()
        torch.cuda.empty_cache()


def log_config(cfg: DictConfig):
    if "wandb" in cfg.get("loggers", {}):
        try:
            import wandb
        except ImportError as e:
            raise e
        if wandb.run:
            wandb.config.update(om.to_container(cfg, resolve=True))


class FineTuneJob:
    """Encapsulates a fine-tuning job.

    Tasks should subclass FineTuneJob and implement the
    get_trainer() method.

    Args:
        name (str, optional): job name. Defaults to the class name.
        load_path (str, optional): path to load checkpoints. Default: None
        save_folder (str, optional): path to save checkpoints. Default: None
        kwargs (dict, optional): additional arguments passed available to the Trainer.
    """

    def __init__(
        self,
        job_name: Optional[str] = None,
        load_path: Optional[str] = None,
        save_folder: Optional[str] = None,
        seed: int = 42,
        **kwargs,
    ):
        reproducibility.seed_all(seed)
        self._job_name = job_name
        self.seed = seed
        self.load_path = load_path
        self.save_folder = save_folder
        self.kwargs = kwargs

    def get_trainer(self, device: Optional[Union[str, Device]]) -> Trainer:
        """Returns the trainer for the job."""
        raise NotImplementedError

    def print_metrics(self, metrics: Metrics):
        """Prints fine-tuning results."""
        job_name = self.job_name

        print(f"Results for {job_name}:")
        print("-" * (12 + len(job_name)))
        for eval, metric in metrics.items():
            for metric_name, value in metric.items():
                print(f"{eval}: {metric_name}, {value*100:.2f}")
        print("-" * (12 + len(job_name)))

    @property
    def job_name(self) -> str:
        """Job name, defaults to class name."""
        if self._job_name is not None:
            return self._job_name
        return self.__class__.__name__

    def run(
        self,
        gpu_queue: Optional[mp.Queue] = None,
        process_to_gpu: Optional[managers.DictProxy] = None,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Trains the model, optionally pulling a GPU id from the queue.

        Returns:
            A dict with keys:
            * 'checkpoints': list of saved_checkpoints, if any,
            * 'metrics': nested dict of results, accessed by
                        dataset and metric name, e.g.
                        ``metrics['glue_mnli']['MulticlassAccuracy']``.
        """
        if gpu_queue is None:
            if torch.cuda.device_count() > 0:
                gpu_id = 0
                device = DeviceGPU(gpu_id)
            else:
                gpu_id = None
                device = "cpu"
        else:
            current_pid = os.getpid()
            assert process_to_gpu is not None
            if current_pid in process_to_gpu:
                gpu_id = process_to_gpu[current_pid]
            else:
                gpu_id = gpu_queue.get()
                process_to_gpu[current_pid] = gpu_id
            device = DeviceGPU(gpu_id)

        print(f"Running {self.job_name} on GPU {gpu_id}")

        trainer = self.get_trainer(device=device)

        if cfg is not None:
            print("Logging config...")
            log_config(cfg)

        trainer.fit()

        collected_metrics: Dict[str, Dict[str, Any]] = {}
        for eval_name, metrics in trainer.state.eval_metrics.items():
            for name, metric in metrics.items():
                if hasattr(metric, "compute_final"):
                    result = metric.compute_final()
                else:
                    result = metric.compute()

                if isinstance(result, dict):
                    collected_metrics[eval_name] = result
                else:
                    collected_metrics[eval_name] = {name: metric.compute().cpu().numpy()}

        saved_checkpoints = copy.copy(trainer.saved_checkpoints)
        try:
            loggers = copy.copy(trainer.logger.destinations)
        except AttributeError:
            loggers = None

        reset_trainer(trainer, garbage_collect=True)

        self.print_metrics(collected_metrics)

        output = {
            "checkpoints": saved_checkpoints,
            "metrics": collected_metrics,
            "job_name": self.job_name,
            "loggers": loggers,
        }

        return output


class ClassificationJob(FineTuneJob):
    custom_eval_metrics = []
    multiple_choice = False

    def __init__(
        self,
        model: ComposerModel,
        tokenizer_name: str,
        job_name: Optional[str] = None,
        seed: int = 42,
        task_name: Optional[str] = None,
        eval_interval: str = "1000ba",
        scheduler: Optional[ComposerScheduler] = None,
        optimizer: Optional[Optimizer] = None,
        max_sequence_length: Optional[int] = 256,
        max_duration: Optional[str] = "3ep",
        batch_size: Optional[int] = 32,
        load_path: Optional[str] = None,
        save_folder: Optional[str] = None,
        loggers: Optional[List[LoggerDestination]] = None,
        callbacks: Optional[List[Callback]] = None,
        precision: Optional[str] = None,
        device_train_microbatch_size: Optional[int] = None,
        **kwargs,
    ):
        if task_name is None:
            raise ValueError(
                "ClassificationJob should not be instantiated directly. Please instantiate a specific glue job type instead (e.g. MNLIJob)."
            )
        super().__init__(job_name, load_path, save_folder, seed, **kwargs)

        self.task_name = task_name

        self.eval_interval = eval_interval
        self.tokenizer_name = tokenizer_name
        self.model = model

        self.scheduler = scheduler
        self.optimizer = optimizer

        self.max_sequence_length = max_sequence_length
        self.max_duration = max_duration
        self.batch_size = batch_size
        self.loggers = loggers
        self.callbacks = callbacks
        self.precision = precision
        self.device_train_microbatch_size = device_train_microbatch_size

        # These will be set by the subclasses for specific GLUE tasks
        self.train_dataloader = None
        self.evaluators = None

    def get_trainer(self, device: Optional[Union[Device, str]] = None):
        if self.device_train_microbatch_size is None:
            if torch.cuda.device_count() > 0:
                self.device_train_microbatch_size = "auto"

        return Trainer(
            model=self.model,
            optimizers=self.optimizer,
            schedulers=self.scheduler,
            train_dataloader=self.train_dataloader,
            eval_dataloader=self.evaluators,
            eval_interval=self.eval_interval,
            load_path=self.load_path,
            save_folder=self.save_folder,
            max_duration=self.max_duration,
            seed=self.seed,
            device_train_microbatch_size=self.device_train_microbatch_size,
            load_weights_only=True,
            load_strict_model_weights=False,
            loggers=self.loggers,
            callbacks=self.callbacks,
            python_log_level="ERROR",
            run_name=self.job_name,
            load_ignore_keys=["state/model/model.classifier*"],
            precision=self.precision,
            device=device,
            progress_bar=True,
            log_to_console=False,
            **self.kwargs,
        )
