# Copyright 2024 BERT24 authors
# SPDX-License-Identifier: Apache-2.0

# """Contains SuperGLUE job objects for the simple_glue_trainer."""
import os
import sys
from typing import List, Optional

# Add glue folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from composer import ComposerModel
from composer.core import Callback
from composer.core.evaluator import Evaluator
from composer.loggers import LoggerDestination
from composer.optim import ComposerScheduler, DecoupledAdamW
from src.evals.data import create_superglue_dataset
from src.evals.finetuning_jobs import (
    build_dataloader,
    multiple_choice_collate_fn,
    ClassificationJob,
)


class BoolQJob(ClassificationJob):
    """BoolQ."""

    num_labels = 2

    def __init__(
        self,
        model: ComposerModel,
        tokenizer_name: str,
        job_name: Optional[str] = None,
        seed: int = 42,
        eval_interval: str = "100ba",
        scheduler: Optional[ComposerScheduler] = None,
        max_sequence_length: Optional[int] = 256,
        max_duration: Optional[str] = "5ep",
        batch_size: Optional[int] = 48,
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
            task_name="boolq",
            eval_interval=eval_interval,
            scheduler=scheduler,
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

        self.optimizer = DecoupledAdamW(
            self.model.parameters(),
            lr=5.0e-5,
            betas=(0.9, 0.98),
            eps=1.0e-06,
            weight_decay=5.0e-06,
        )

        dataset_kwargs = {
            "task": self.task_name,
            "tokenizer_name": self.tokenizer_name,
            "max_seq_length": self.max_sequence_length,
        }

        dataloader_kwargs = {
            "batch_size": self.batch_size,
            "num_workers": 0,
            "shuffle": False,
            "drop_last": False,
        }
        train_dataset = create_superglue_dataset(split="train", **dataset_kwargs)
        self.train_dataloader = build_dataloader(train_dataset, **dataloader_kwargs)
        boolq_eval_dataset = create_superglue_dataset(
            split="validation", **dataset_kwargs
        )
        boolq_evaluator = Evaluator(
            label="superglue_boolq",
            dataloader=build_dataloader(boolq_eval_dataset, **dataloader_kwargs),
            metric_names=["MulticlassAccuracy"],
        )
        self.evaluators = [boolq_evaluator]


class CBJob(ClassificationJob):
    """CB."""

    num_labels = 3

    def __init__(
        self,
        model: ComposerModel,
        tokenizer_name: str,
        job_name: Optional[str] = None,
        seed: int = 42,
        eval_interval: str = "20ba",
        scheduler: Optional[ComposerScheduler] = None,
        max_sequence_length: Optional[int] = 256,
        max_duration: Optional[str] = "30ep",
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
            task_name="cb",
            eval_interval=eval_interval,
            scheduler=scheduler,
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
            "num_workers": 0,
            "shuffle": False,
            "drop_last": False,
        }
        train_dataset = create_superglue_dataset(split="train", **dataset_kwargs)
        self.train_dataloader = build_dataloader(train_dataset, **dataloader_kwargs)
        cb_eval_dataset = create_superglue_dataset(split="validation", **dataset_kwargs)
        cb_evaluator = Evaluator(
            label="superglue_cb",
            dataloader=build_dataloader(cb_eval_dataset, **dataloader_kwargs),
            metric_names=["MulticlassAccuracy", "BinaryF1Score"],
        )
        self.evaluators = [cb_evaluator]


class COPAJob(ClassificationJob):
    """COPA."""

    model_type = "multiple_choice"
    num_labels = 2

    def __init__(
        self,
        model: ComposerModel,
        tokenizer_name: str,
        job_name: Optional[str] = None,
        seed: int = 42,
        eval_interval: str = "50ba",
        scheduler: Optional[ComposerScheduler] = None,
        max_sequence_length: Optional[int] = 256,
        max_duration: Optional[str] = "30ep",
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
            task_name="copa",
            eval_interval=eval_interval,
            scheduler=scheduler,
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

        self.optimizer = DecoupledAdamW(
            self.model.parameters(),
            lr=1.0e-5,
            betas=(0.9, 0.98),
            eps=1.0e-6,
            weight_decay=5.0e-06,
        )

        def tokenize_fn_factory(tokenizer, max_seq_length):
            def tokenize_fn(inp):
                first_sentences = [[context] * 2 for context in inp["premise"]]
                second_sentences = [
                    [inp["choice1"][i], inp["choice2"][i]]
                    for i in range(len(inp["choice1"]))
                ]

                first_sentences = sum(first_sentences, [])
                second_sentences = sum(second_sentences, [])

                tokenized_examples = tokenizer(
                    first_sentences,
                    second_sentences,
                    padding="max_length",
                    max_length=max_seq_length,
                    truncation=True,
                )

                return {
                    k: [v[i : i + 2] for i in range(0, len(v), 2)]
                    for k, v in tokenized_examples.items()
                }

            return tokenize_fn

        dataset_kwargs = {
            "task": self.task_name,
            "tokenizer_name": self.tokenizer_name,
            "max_seq_length": self.max_sequence_length,
            "tokenize_fn_factory": tokenize_fn_factory,
        }

        dataloader_kwargs = {
            "batch_size": self.batch_size,
            "num_workers": 0,
            "shuffle": False,
            "drop_last": False,
        }

        train_dataset = create_superglue_dataset(split="train", **dataset_kwargs)
        self.train_dataloader = build_dataloader(
            train_dataset, collate_fn=multiple_choice_collate_fn, **dataloader_kwargs
        )
        copa_eval_dataset = create_superglue_dataset(
            split="validation", **dataset_kwargs
        )
        copa_evaluator = Evaluator(
            label="superglue_copa",
            dataloader=build_dataloader(
                copa_eval_dataset,
                collate_fn=multiple_choice_collate_fn,
                **dataloader_kwargs,
            ),
            metric_names=["MulticlassAccuracy"],
        )
        self.evaluators = [copa_evaluator]


class WiCJob(ClassificationJob):
    """WiC."""

    num_labels = 2

    def __init__(
        self,
        model: ComposerModel,
        tokenizer_name: str,
        job_name: Optional[str] = None,
        seed: int = 42,
        eval_interval: str = "300ba",
        scheduler: Optional[ComposerScheduler] = None,
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
            task_name="wic",
            eval_interval=eval_interval,
            scheduler=scheduler,
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
            "num_workers": 0,
            "shuffle": False,
            "drop_last": False,
        }
        train_dataset = create_superglue_dataset(split="train", **dataset_kwargs)
        self.train_dataloader = build_dataloader(train_dataset, **dataloader_kwargs)
        wic_eval_dataset = create_superglue_dataset(
            split="validation", **dataset_kwargs
        )
        wic_evaluator = Evaluator(
            label="superglue_wic",
            dataloader=build_dataloader(wic_eval_dataset, **dataloader_kwargs),
            metric_names=["MulticlassAccuracy"],
        )
        self.evaluators = [wic_evaluator]
