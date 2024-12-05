# Copyright 2024 BERT24 authors
# SPDX-License-Identifier: Apache-2.0

# """Contains GLUE job objects for the simple_glue_trainer."""
import os
import sys
from typing import List, Optional, Tuple
from multiprocessing import cpu_count
import torch

# Add glue folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from composer import ComposerModel
from composer.core import Callback
from composer.core.evaluator import Evaluator
from composer.loggers import LoggerDestination
from composer.optim import ComposerScheduler, DecoupledAdamW
from torch.optim import Optimizer
from src.evals.data import create_glue_dataset
from src.evals.finetuning_jobs import build_dataloader, ClassificationJob


class MNLIJob(ClassificationJob):
    """MNLI."""

    num_labels = 3

    def __init__(
        self,
        model: ComposerModel,
        tokenizer_name: str,
        job_name: Optional[str] = None,
        seed: int = 42,
        eval_interval: str = "2300ba",
        scheduler: Optional[ComposerScheduler] = None,
        optimizer: Optional[Optimizer] = None,
        max_sequence_length: Optional[int] = 256,
        max_duration: Optional[str] = "2ep",
        batch_size: Optional[int] = 64,
        load_path: Optional[str] = None,
        save_folder: Optional[str] = None,
        loggers: Optional[List[LoggerDestination]] = None,
        callbacks: Optional[List[Callback]] = None,
        precision: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            tokenizer_name=tokenizer_name,
            job_name=job_name,
            seed=seed,
            task_name="mnli",
            eval_interval=eval_interval,
            scheduler=scheduler,
            optimizer=optimizer,
            max_sequence_length=max_sequence_length,
            max_duration=max_duration,
            batch_size=batch_size,
            load_path=load_path,
            save_folder=save_folder,
            loggers=loggers,
            callbacks=callbacks,
            precision=precision,
            **kwargs,
        )

        if optimizer is None:
            self.optimizer = DecoupledAdamW(
                self.model.parameters(),
                lr=5.0e-05,
                betas=(0.9, 0.98),
                eps=1.0e-06,
                weight_decay=1.0e-06,
            )

        dataset_kwargs = {
            "task": self.task_name,
            "tokenizer_name": self.tokenizer_name,
            "max_seq_length": self.max_sequence_length,
        }

        dataloader_kwargs = {
            "batch_size": self.batch_size,
            "num_workers": min(8, cpu_count() // torch.cuda.device_count()),
            "drop_last": False,
        }
        train_dataset = create_glue_dataset(split="train", **dataset_kwargs)
        self.train_dataloader = build_dataloader(train_dataset, **dataloader_kwargs)
        mnli_eval_dataset = create_glue_dataset(split="validation_matched", **dataset_kwargs)
        mnli_eval_mismatched_dataset = create_glue_dataset(split="validation_mismatched", **dataset_kwargs)
        mnli_evaluator = Evaluator(
            label="glue_mnli",
            dataloader=build_dataloader(mnli_eval_dataset, **dataloader_kwargs),
            metric_names=["MulticlassAccuracy"],
        )
        mnli_evaluator_mismatched = Evaluator(
            label="glue_mnli_mismatched",
            dataloader=build_dataloader(mnli_eval_mismatched_dataset, **dataloader_kwargs),
            metric_names=["MulticlassAccuracy"],
        )
        self.evaluators = [mnli_evaluator, mnli_evaluator_mismatched]


class RTEJob(ClassificationJob):
    """RTE."""

    num_labels = 2

    def __init__(
        self,
        model: ComposerModel,
        tokenizer_name: str,
        job_name: Optional[str] = None,
        seed: int = 42,
        eval_interval: str = "100ba",
        scheduler: Optional[ComposerScheduler] = None,
        optimizer: Optional[Optimizer] = None,
        max_sequence_length: Optional[int] = 256,
        max_duration: Optional[str] = "3ep",
        batch_size: Optional[int] = 16,
        load_path: Optional[str] = None,
        save_folder: Optional[str] = None,
        loggers: Optional[List[LoggerDestination]] = None,
        callbacks: Optional[List[Callback]] = None,
        precision: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            tokenizer_name=tokenizer_name,
            job_name=job_name,
            seed=seed,
            task_name="rte",
            eval_interval=eval_interval,
            scheduler=scheduler,
            optimizer=optimizer,
            max_sequence_length=max_sequence_length,
            max_duration=max_duration,
            batch_size=batch_size,
            load_path=load_path,
            save_folder=save_folder,
            loggers=loggers,
            callbacks=callbacks,
            precision=precision,
            **kwargs,
        )

        if optimizer is None:
            self.optimizer = DecoupledAdamW(
                self.model.parameters(),
                lr=1.0e-5,
                betas=(0.9, 0.98),
                eps=1.0e-06,
                weight_decay=1.0e-5,
            )

        dataset_kwargs = {
            "task": self.task_name,
            "tokenizer_name": self.tokenizer_name,
            "max_seq_length": self.max_sequence_length,
        }

        dataloader_kwargs = {
            "batch_size": self.batch_size,
            "num_workers": min(8, cpu_count() // torch.cuda.device_count()),
            "drop_last": False,
        }
        train_dataset = create_glue_dataset(split="train", **dataset_kwargs)
        self.train_dataloader = build_dataloader(train_dataset, **dataloader_kwargs)
        rte_eval_dataset = create_glue_dataset(split="validation", **dataset_kwargs)
        rte_evaluator = Evaluator(
            label="glue_rte",
            dataloader=build_dataloader(rte_eval_dataset, **dataloader_kwargs),
            metric_names=["MulticlassAccuracy"],
        )
        self.evaluators = [rte_evaluator]


class QQPJob(ClassificationJob):
    """QQP."""

    num_labels = 2

    def __init__(
        self,
        model: ComposerModel,
        tokenizer_name: str,
        job_name: Optional[str] = None,
        seed: int = 42,
        eval_interval: str = "2000ba",
        scheduler: Optional[ComposerScheduler] = None,
        optimizer: Optional[Optimizer] = None,
        max_sequence_length: Optional[int] = 256,
        max_duration: Optional[str] = "5ep",
        batch_size: Optional[int] = 16,
        load_path: Optional[str] = None,
        save_folder: Optional[str] = None,
        loggers: Optional[List[LoggerDestination]] = None,
        callbacks: Optional[List[Callback]] = None,
        precision: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            tokenizer_name=tokenizer_name,
            job_name=job_name,
            seed=seed,
            task_name="qqp",
            eval_interval=eval_interval,
            scheduler=scheduler,
            optimizer=optimizer,
            max_sequence_length=max_sequence_length,
            max_duration=max_duration,
            batch_size=batch_size,
            load_path=load_path,
            save_folder=save_folder,
            loggers=loggers,
            callbacks=callbacks,
            precision=precision,
            **kwargs,
        )

        if optimizer is None:
            self.optimizer = DecoupledAdamW(
                self.model.parameters(),
                lr=3.0e-5,
                betas=(0.9, 0.98),
                eps=1.0e-06,
                weight_decay=3.0e-6,
            )

        dataset_kwargs = {
            "task": self.task_name,
            "tokenizer_name": self.tokenizer_name,
            "max_seq_length": self.max_sequence_length,
        }

        dataloader_kwargs = {
            "batch_size": self.batch_size,
            "num_workers": min(8, cpu_count() // torch.cuda.device_count()),
            "drop_last": False,
        }
        train_dataset = create_glue_dataset(split="train", **dataset_kwargs)
        self.train_dataloader = build_dataloader(train_dataset, **dataloader_kwargs)
        qqp_eval_dataset = create_glue_dataset(split="validation", **dataset_kwargs)
        qqp_evaluator = Evaluator(
            label="glue_qqp",
            dataloader=build_dataloader(qqp_eval_dataset, **dataloader_kwargs),
            metric_names=["MulticlassAccuracy", "BinaryF1Score"],
        )
        self.evaluators = [qqp_evaluator]


class COLAJob(ClassificationJob):
    """COLA."""

    num_labels = 2

    def __init__(
        self,
        model: ComposerModel,
        tokenizer_name: str,
        job_name: Optional[str] = None,
        seed: int = 42,
        eval_interval: str = "250ba",
        scheduler: Optional[ComposerScheduler] = None,
        optimizer: Optional[Optimizer] = None,
        max_sequence_length: Optional[int] = 256,
        max_duration: Optional[str] = "10ep",
        batch_size: Optional[int] = 32,
        load_path: Optional[str] = None,
        save_folder: Optional[str] = None,
        loggers: Optional[List[LoggerDestination]] = None,
        callbacks: Optional[List[Callback]] = None,
        precision: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            tokenizer_name=tokenizer_name,
            job_name=job_name,
            seed=seed,
            task_name="cola",
            eval_interval=eval_interval,
            scheduler=scheduler,
            optimizer=optimizer,
            max_sequence_length=max_sequence_length,
            max_duration=max_duration,
            batch_size=batch_size,
            load_path=load_path,
            save_folder=save_folder,
            loggers=loggers,
            callbacks=callbacks,
            precision=precision,
            **kwargs,
        )

        if optimizer is None:
            self.optimizer = DecoupledAdamW(
                self.model.parameters(),
                lr=5.0e-5,
                betas=(0.9, 0.98),
                eps=1.0e-06,
                weight_decay=5.0e-6,
            )

        dataset_kwargs = {
            "task": self.task_name,
            "tokenizer_name": self.tokenizer_name,
            "max_seq_length": self.max_sequence_length,
        }

        dataloader_kwargs = {
            "batch_size": self.batch_size,
            "num_workers": min(8, cpu_count() // torch.cuda.device_count()),
            "drop_last": False,
        }
        train_dataset = create_glue_dataset(split="train", **dataset_kwargs)
        self.train_dataloader = build_dataloader(train_dataset, **dataloader_kwargs)
        cola_eval_dataset = create_glue_dataset(split="validation", **dataset_kwargs)
        cola_evaluator = Evaluator(
            label="glue_cola",
            dataloader=build_dataloader(cola_eval_dataset, **dataloader_kwargs),
            metric_names=["MatthewsCorrCoef"],
        )
        self.evaluators = [cola_evaluator]


class MRPCJob(ClassificationJob):
    """MRPC."""

    num_labels = 2

    def __init__(
        self,
        model: ComposerModel,
        tokenizer_name: str,
        job_name: Optional[str] = None,
        seed: int = 42,
        eval_interval: str = "100ba",
        scheduler: Optional[ComposerScheduler] = None,
        optimizer: Optional[Optimizer] = None,
        max_sequence_length: Optional[int] = 256,
        max_duration: Optional[str] = "10ep",
        batch_size: Optional[int] = 32,
        load_path: Optional[str] = None,
        save_folder: Optional[str] = None,
        loggers: Optional[List[LoggerDestination]] = None,
        callbacks: Optional[List[Callback]] = None,
        precision: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            tokenizer_name=tokenizer_name,
            job_name=job_name,
            seed=seed,
            task_name="mrpc",
            eval_interval=eval_interval,
            scheduler=scheduler,
            optimizer=optimizer,
            max_sequence_length=max_sequence_length,
            max_duration=max_duration,
            batch_size=batch_size,
            load_path=load_path,
            save_folder=save_folder,
            loggers=loggers,
            callbacks=callbacks,
            precision=precision,
            **kwargs,
        )

        if optimizer is None:
            self.optimizer = DecoupledAdamW(
                self.model.parameters(),
                lr=8.0e-5,
                betas=(0.9, 0.98),
                eps=1.0e-06,
                weight_decay=8.0e-6,
            )

        dataset_kwargs = {
            "task": self.task_name,
            "tokenizer_name": self.tokenizer_name,
            "max_seq_length": self.max_sequence_length,
        }

        dataloader_kwargs = {
            "batch_size": self.batch_size,
            "num_workers": min(8, cpu_count() // torch.cuda.device_count()),
            "drop_last": False,
        }
        train_dataset = create_glue_dataset(split="train", **dataset_kwargs)
        self.train_dataloader = build_dataloader(train_dataset, **dataloader_kwargs)
        mrpc_eval_dataset = create_glue_dataset(split="validation", **dataset_kwargs)
        mrpc_evaluator = Evaluator(
            label="glue_mrpc",
            dataloader=build_dataloader(mrpc_eval_dataset, **dataloader_kwargs),
            metric_names=["MulticlassAccuracy", "BinaryF1Score"],
        )
        self.evaluators = [mrpc_evaluator]


class QNLIJob(ClassificationJob):
    """QNLI."""

    num_labels = 2

    def __init__(
        self,
        model: ComposerModel,
        tokenizer_name: str,
        job_name: Optional[str] = None,
        seed: int = 42,
        eval_interval: str = "1000ba",
        scheduler: Optional[ComposerScheduler] = None,
        optimizer: Optional[Optimizer] = None,
        max_sequence_length: Optional[int] = 256,
        max_duration: Optional[str] = "10ep",
        batch_size: Optional[int] = 16,
        load_path: Optional[str] = None,
        save_folder: Optional[str] = None,
        loggers: Optional[List[LoggerDestination]] = None,
        callbacks: Optional[List[Callback]] = None,
        precision: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            tokenizer_name=tokenizer_name,
            job_name=job_name,
            seed=seed,
            task_name="qnli",
            eval_interval=eval_interval,
            scheduler=scheduler,
            optimizer=optimizer,
            max_sequence_length=max_sequence_length,
            max_duration=max_duration,
            batch_size=batch_size,
            load_path=load_path,
            save_folder=save_folder,
            loggers=loggers,
            callbacks=callbacks,
            precision=precision,
            **kwargs,
        )

        if optimizer is None:
            self.optimizer = DecoupledAdamW(
                self.model.parameters(),
                lr=1.0e-5,
                betas=(0.9, 0.98),
                eps=1.0e-06,
                weight_decay=1.0e-6,
            )

        dataset_kwargs = {
            "task": self.task_name,
            "tokenizer_name": self.tokenizer_name,
            "max_seq_length": self.max_sequence_length,
        }

        dataloader_kwargs = {
            "batch_size": self.batch_size,
            "num_workers": min(8, cpu_count() // torch.cuda.device_count()),
            "drop_last": False,
        }
        train_dataset = create_glue_dataset(split="train", **dataset_kwargs)
        self.train_dataloader = build_dataloader(train_dataset, **dataloader_kwargs)
        qnli_eval_dataset = create_glue_dataset(split="validation", **dataset_kwargs)
        qnli_evaluator = Evaluator(
            label="glue_qnli",
            dataloader=build_dataloader(qnli_eval_dataset, **dataloader_kwargs),
            metric_names=["MulticlassAccuracy"],
        )
        self.evaluators = [qnli_evaluator]


class SST2Job(ClassificationJob):
    """SST2."""

    num_labels = 2

    def __init__(
        self,
        model: ComposerModel,
        tokenizer_name: str,
        job_name: Optional[str] = None,
        seed: int = 42,
        eval_interval: str = "500ba",
        scheduler: Optional[ComposerScheduler] = None,
        optimizer: Optional[Optimizer] = None,
        max_sequence_length: Optional[int] = 256,
        max_duration: Optional[str] = "3ep",
        batch_size: Optional[int] = 16,
        load_path: Optional[str] = None,
        save_folder: Optional[str] = None,
        loggers: Optional[List[LoggerDestination]] = None,
        callbacks: Optional[List[Callback]] = None,
        precision: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            tokenizer_name=tokenizer_name,
            job_name=job_name,
            seed=seed,
            task_name="sst2",
            eval_interval=eval_interval,
            scheduler=scheduler,
            optimizer=optimizer,
            max_sequence_length=max_sequence_length,
            max_duration=max_duration,
            batch_size=batch_size,
            load_path=load_path,
            save_folder=save_folder,
            loggers=loggers,
            callbacks=callbacks,
            precision=precision,
            **kwargs,
        )

        if optimizer is None:
            self.optimizer = DecoupledAdamW(
                self.model.parameters(),
                lr=3.0e-5,
                betas=(0.9, 0.98),
                eps=1.0e-06,
                weight_decay=3.0e-6,
            )

        dataset_kwargs = {
            "task": self.task_name,
            "tokenizer_name": self.tokenizer_name,
            "max_seq_length": self.max_sequence_length,
        }

        dataloader_kwargs = {
            "batch_size": self.batch_size,
            "num_workers": min(8, cpu_count() // torch.cuda.device_count()),
            "drop_last": False,
        }
        train_dataset = create_glue_dataset(split="train", **dataset_kwargs)
        self.train_dataloader = build_dataloader(train_dataset, **dataloader_kwargs)
        sst2_eval_dataset = create_glue_dataset(split="validation", **dataset_kwargs)
        sst2_evaluator = Evaluator(
            label="glue_sst2",
            dataloader=build_dataloader(sst2_eval_dataset, **dataloader_kwargs),
            metric_names=["MulticlassAccuracy"],
        )
        self.evaluators = [sst2_evaluator]


class STSBJob(ClassificationJob):
    """STSB."""

    num_labels = 1

    def __init__(
        self,
        model: ComposerModel,
        tokenizer_name: str,
        job_name: Optional[str] = None,
        seed: int = 42,
        eval_interval: str = "200ba",
        scheduler: Optional[ComposerScheduler] = None,
        optimizer: Optional[Optimizer] = None,
        max_sequence_length: Optional[int] = 256,
        max_duration: Optional[str] = "10ep",
        batch_size: Optional[int] = 32,
        load_path: Optional[str] = None,
        save_folder: Optional[str] = None,
        loggers: Optional[List[LoggerDestination]] = None,
        callbacks: Optional[List[Callback]] = None,
        precision: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            tokenizer_name=tokenizer_name,
            job_name=job_name,
            seed=seed,
            task_name="stsb",
            eval_interval=eval_interval,
            scheduler=scheduler,
            optimizer=optimizer,
            max_sequence_length=max_sequence_length,
            max_duration=max_duration,
            batch_size=batch_size,
            load_path=load_path,
            save_folder=save_folder,
            loggers=loggers,
            callbacks=callbacks,
            precision=precision,
            **kwargs,
        )

        if optimizer is None:
            self.optimizer = DecoupledAdamW(
                self.model.parameters(),
                lr=3.0e-5,
                betas=(0.9, 0.98),
                eps=1.0e-06,
                weight_decay=3.0e-6,
            )

        dataset_kwargs = {
            "task": self.task_name,
            "tokenizer_name": self.tokenizer_name,
            "max_seq_length": self.max_sequence_length,
        }

        dataloader_kwargs = {
            "batch_size": self.batch_size,
            "num_workers": min(8, cpu_count() // torch.cuda.device_count()),
            "drop_last": False,
        }
        train_dataset = create_glue_dataset(split="train", **dataset_kwargs)
        self.train_dataloader = build_dataloader(train_dataset, **dataloader_kwargs)
        stsb_eval_dataset = create_glue_dataset(split="validation", **dataset_kwargs)
        stsb_evaluator = Evaluator(
            label="glue_stsb",
            dataloader=build_dataloader(stsb_eval_dataset, **dataloader_kwargs),
            metric_names=["SpearmanCorrCoef"],
        )
        self.evaluators = [stsb_evaluator]

        # Hardcoded for STSB due to a bug (Can be removed once torchmetrics fixes https://github.com/Lightning-AI/metrics/issues/1294)
        self.precision = "fp32"
