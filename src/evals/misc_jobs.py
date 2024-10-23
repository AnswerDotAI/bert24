# Copyright 2024 BERT24 authors
# SPDX-License-Identifier: Apache-2.0

# """Contains SuperGLUE job objects for the simple_glue_trainer."""
import os
import sys
from itertools import chain
from multiprocessing import cpu_count
from typing import List, Optional

import torch

# Add glue folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from composer import ComposerModel
from composer.core import Callback
from composer.core.evaluator import Evaluator
from composer.loggers import LoggerDestination
from composer.optim import ComposerScheduler, DecoupledAdamW
from torch.optim import Optimizer
from torchmetrics.classification import MulticlassAUROC, MultilabelF1Score

from src.evals.data import (
    create_eurlex_dataset,
    create_mlmmlu_dataset,
    create_swag_dataset,
    create_ultrafeedback_dataset,
)
from src.evals.finetuning_jobs import (
    ClassificationJob,
    build_dataloader,
    multiple_choice_collate_fn,
)


class SWAGJob(ClassificationJob):
    """SWAG."""

    multiple_choice = True
    num_labels = 4

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
        max_duration: Optional[str] = "1ep",
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
            task_name="swag",
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
                eps=1.0e-6,
                weight_decay=5.0e-06,
        )

        def tokenize_fn_factory(tokenizer, max_seq_length):
            def tokenize_fn(inp):
                ending_names = ["ending0", "ending1", "ending2", "ending3"]
                first_sentences = [[context] * 4 for context in inp["sent1"]]
                question_headers = inp["sent2"]
                second_sentences = [
                    [f"{header} {inp[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
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
                return {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}

            return tokenize_fn

        dataset_kwargs = {
            "task": self.task_name,
            "tokenizer_name": self.tokenizer_name,
            "max_seq_length": self.max_sequence_length,
            "tokenize_fn_factory": tokenize_fn_factory,
        }

        dataloader_kwargs = {
            "batch_size": self.batch_size,
            "num_workers": min(8, cpu_count() // torch.cuda.device_count()),
            "drop_last": False,
        }

        train_dataset = create_swag_dataset(split="train", **dataset_kwargs)
        self.train_dataloader = build_dataloader(
            train_dataset, collate_fn=multiple_choice_collate_fn, **dataloader_kwargs
        )
        swag_eval_dataset = create_swag_dataset(split="validation", **dataset_kwargs)
        swag_evaluator = Evaluator(
            label="superglue_swag",
            dataloader=build_dataloader(
                swag_eval_dataset,
                collate_fn=multiple_choice_collate_fn,
                **dataloader_kwargs,
            ),
            metric_names=["MulticlassAccuracy"],
        )
        self.evaluators = [swag_evaluator]


class EurlexMultilabelF1Score(MultilabelF1Score):
    def __init__(self):
        super().__init__(num_labels=100, average="micro", threshold=0.5)


class EurlexJob(ClassificationJob):
    """Eurlex multi-label classification."""

    custom_eval_metrics = [EurlexMultilabelF1Score]
    num_labels = 100

    def __init__(
        self,
        model: ComposerModel,
        tokenizer_name: str,
        job_name: Optional[str] = None,
        seed: int = 42,
        eval_interval: str = "1600ba",
        scheduler: Optional[ComposerScheduler] = None,
        optimizer: Optional[Optimizer] = None,
        max_sequence_length: Optional[int] = 512,
        max_duration: Optional[str] = "4ep",
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
            task_name="coastalcph/lex_glue",
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
                weight_decay=1.0e-06,
        )

        def tokenize_fn_factory(tokenizer, max_seq_length):
            def tokenize_fn(inp):
                first_sentences = inp["text"]

                tokenized_examples = tokenizer(
                    first_sentences,
                    padding="max_length",
                    max_length=max_seq_length,
                    truncation=True,
                )

                return tokenized_examples

            return tokenize_fn

        dataset_kwargs = {
            "task": self.task_name,
            "tokenizer_name": self.tokenizer_name,
            "max_seq_length": self.max_sequence_length,
            "tokenize_fn_factory": tokenize_fn_factory,
        }

        dataloader_kwargs = {
            "batch_size": self.batch_size,
            "num_workers": min(8, cpu_count() // torch.cuda.device_count()),
            "drop_last": False,
        }

        eurlex_train_dataset = create_eurlex_dataset(split="train", **dataset_kwargs)
        eurlex_eval_dataset = create_eurlex_dataset(split="test", **dataset_kwargs)

        eurlex_train_dataset = eurlex_train_dataset.rename_column("labels", "eurovoc_concepts")
        eurlex_eval_dataset = eurlex_eval_dataset.rename_column("labels", "eurovoc_concepts")

        # process labels: eurovoc_concepts ---
        train_classes = sorted(list(set(chain(*eurlex_train_dataset["eurovoc_concepts"]))))
        class2id = {class_: id for id, class_ in enumerate(train_classes)}
        n_labels = len(train_classes)

        def generate_labels(example):
            concepts = set(example["eurovoc_concepts"]).intersection(
                set(train_classes)
            )  # not to introduce new concepts in validation
            labels = [0.0 for i in range(n_labels)]

            for label in concepts:
                label_id = class2id[label]
                labels[label_id] = 1.0
            example["labels"] = labels
            return example

        eurlex_train_dataset = eurlex_train_dataset.map(generate_labels, remove_columns=["eurovoc_concepts"])
        eurlex_eval_dataset = eurlex_eval_dataset.map(generate_labels, remove_columns=["eurovoc_concepts"])

        self.train_dataloader = build_dataloader(eurlex_train_dataset, **dataloader_kwargs)

        eurlex_evaluator = Evaluator(
            label="long_context_eurlex",
            dataloader=build_dataloader(eurlex_eval_dataset, **dataloader_kwargs),
            metric_names=["EurlexMultilabelF1Score"],
        )

        self.evaluators = [eurlex_evaluator]


class UltrafeedbackAUROC(MulticlassAUROC):
    def __init__(self):
        super().__init__(num_classes=2)


class UltrafeedbackJob(ClassificationJob):
    """ultrafeedback binary classification."""

    custom_eval_metrics = [UltrafeedbackAUROC]
    num_labels = 2

    def __init__(
        self,
        model: ComposerModel,
        tokenizer_name: str,
        job_name: Optional[str] = None,
        seed: int = 42,
        eval_interval: str = "300ba",
        scheduler: Optional[ComposerScheduler] = None,
        optimizer: Optional[Optimizer] = None,
        max_sequence_length: Optional[int] = 2048,
        max_duration: Optional[str] = "3ep",
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
            task_name="rbiswasfc/ultrafeedback-binary-classification",
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
                weight_decay=1.0e-06,
        )

        def tokenize_fn_factory(tokenizer, max_seq_length):
            def tokenize_fn(inp):
                first_sentences = [
                    f"{prompt} {tokenizer.sep_token} {response_a}"
                    for prompt, response_a in zip(inp["prompt"], inp["response_a"])
                ]

                second_sentences = inp["response_b"]

                tokenized_examples = tokenizer(
                    first_sentences,
                    second_sentences,
                    padding="max_length",
                    max_length=max_seq_length,
                    truncation=True,
                )
                return tokenized_examples

            return tokenize_fn

        dataset_kwargs = {
            "task": self.task_name,
            "tokenizer_name": self.tokenizer_name,
            "max_seq_length": self.max_sequence_length,
            "tokenize_fn_factory": tokenize_fn_factory,
        }

        dataloader_kwargs = {
            "batch_size": self.batch_size,
            "num_workers": min(8, cpu_count() // torch.cuda.device_count()),
            "drop_last": False,
        }
        train_dataset = create_ultrafeedback_dataset(split="train", **dataset_kwargs)

        self.train_dataloader = build_dataloader(train_dataset, **dataloader_kwargs)
        ultrafeedback_eval_dataset = create_ultrafeedback_dataset(split="test", **dataset_kwargs)
        ultrafeedback_evaluator = Evaluator(
            label="long_context_ultrafeedback",
            dataloader=build_dataloader(ultrafeedback_eval_dataset, **dataloader_kwargs),
            metric_names=["UltrafeedbackAUROC"],
        )
        self.evaluators = [ultrafeedback_evaluator]


class MLMMLUAmateurSemipro(ClassificationJob):
    """MLMMLU for Amateur & Semipro"""

    multiple_choice = True
    num_labels = 10

    def __init__(
        self,
        model: ComposerModel,
        tokenizer_name: str,
        job_name: Optional[str] = None,
        seed: int = 42,
        eval_interval: str = "100ba",
        scheduler: Optional[ComposerScheduler] = None,
        optimizer: Optional[Optimizer] = None,
        max_sequence_length: Optional[int] = 384,
        max_duration: Optional[str] = "2ep",
        batch_size: Optional[int] = 32,
        device_train_microbatch_size: Optional[int] = 1,
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
            task_name="answerdotai/MLMMLU",
            eval_interval=eval_interval,
            scheduler=scheduler,
            optimizer=optimizer,
            max_sequence_length=max_sequence_length,
            max_duration=max_duration,
            batch_size=batch_size,
            device_train_microbatch_size=device_train_microbatch_size,
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
                lr=2.0e-5,
                betas=(0.9, 0.98),
                eps=1.0e-6,
                weight_decay=5.0e-06,
            )

        def tokenize_fn_factory(tokenizer, max_seq_length):
            def tokenize_fn(inp):
                default_option = "NA"
                choice_col = "options"
                num_options = 10

                first_sentences = [[question] * num_options for question in inp["question"]]
                second_sentences = [
                    option_list + [default_option] * (num_options - len(option_list)) for option_list in inp[choice_col]
                ]

                first_sentences = list(chain(*first_sentences))
                second_sentences = list(chain(*second_sentences))

                tokenized_examples = tokenizer(
                    first_sentences,
                    second_sentences,
                    padding="max_length",
                    max_length=max_seq_length,
                    truncation=True,
                )

                return {
                    k: [v[i : i + num_options] for i in range(0, len(v), num_options)]
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
            "num_workers": min(8, cpu_count() // torch.cuda.device_count()),
            "drop_last": False,
        }

        train_dataset = create_mlmmlu_dataset(split="train", dataset_subset="Amateur", **dataset_kwargs)
        amateur_eval_dataset = create_mlmmlu_dataset(split="test", dataset_subset="Amateur", **dataset_kwargs)
        semipro_eval_dataset = create_mlmmlu_dataset(split="test", dataset_subset="Semipro", **dataset_kwargs)

        train_dataset = train_dataset.rename_column("answer_index", "labels")
        amateur_eval_dataset = amateur_eval_dataset.rename_column("answer_index", "labels")
        semipro_eval_dataset = semipro_eval_dataset.rename_column("answer_index", "labels")

        self.train_dataloader = build_dataloader(
            train_dataset, collate_fn=multiple_choice_collate_fn, **dataloader_kwargs
        )

        amateur_evaluator = Evaluator(
            label="mlmmlu_amateur",
            dataloader=build_dataloader(
                amateur_eval_dataset,
                collate_fn=multiple_choice_collate_fn,
                **dataloader_kwargs,
            ),
            metric_names=["MulticlassAccuracy"],
        )

        semipro_evaluator = Evaluator(
            label="mlmmlu_semipro",
            dataloader=build_dataloader(
                semipro_eval_dataset,
                collate_fn=multiple_choice_collate_fn,
                **dataloader_kwargs,
            ),
            metric_names=["MulticlassAccuracy"],
        )

        self.evaluators = [amateur_evaluator, semipro_evaluator]


class MLMMLUReserveRookie(ClassificationJob):
    """MLMMLU for Reserve & Rookie"""

    multiple_choice = True
    num_labels = 4

    def __init__(
        self,
        model: ComposerModel,
        tokenizer_name: str,
        job_name: Optional[str] = None,
        seed: int = 42,
        eval_interval: str = "200ba",
        scheduler: Optional[ComposerScheduler] = None,
        optimizer: Optional[Optimizer] = None,
        max_sequence_length: Optional[int] = 384,
        max_duration: Optional[str] = "3ep",
        batch_size: Optional[int] = 32,
        device_train_microbatch_size: Optional[int] = 1,
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
            task_name="answerdotai/MLMMLU",
            eval_interval=eval_interval,
            scheduler=scheduler,
            optimizer=optimizer,
            max_sequence_length=max_sequence_length,
            max_duration=max_duration,
            batch_size=batch_size,
            device_train_microbatch_size=device_train_microbatch_size,
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
                eps=1.0e-6,
                weight_decay=5.0e-06,
        )

        def tokenize_fn_factory(tokenizer, max_seq_length):
            def tokenize_fn(inp):
                default_option = "NA"
                choice_col = "choices"
                num_options = 4

                first_sentences = [[question] * num_options for question in inp["question"]]
                second_sentences = [
                    option_list + [default_option] * (num_options - len(option_list)) for option_list in inp[choice_col]
                ]

                first_sentences = list(chain(*first_sentences))
                second_sentences = list(chain(*second_sentences))

                tokenized_examples = tokenizer(
                    first_sentences,
                    second_sentences,
                    padding="max_length",
                    max_length=max_seq_length,
                    truncation=True,
                )

                return {
                    k: [v[i : i + num_options] for i in range(0, len(v), num_options)]
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
            "num_workers": min(8, cpu_count() // torch.cuda.device_count()),
            "drop_last": False,
        }

        train_dataset = create_mlmmlu_dataset(split="train", dataset_subset="Rookie", **dataset_kwargs)
        rookie_eval_dataset = create_mlmmlu_dataset(split="test", dataset_subset="Rookie", **dataset_kwargs)
        reserve_eval_dataset = create_mlmmlu_dataset(split="test", dataset_subset="Reserve", **dataset_kwargs)

        train_dataset = train_dataset.rename_column("answer", "labels")
        rookie_eval_dataset = rookie_eval_dataset.rename_column("answer", "labels")
        reserve_eval_dataset = reserve_eval_dataset.rename_column("answer", "labels")

        self.train_dataloader = build_dataloader(
            train_dataset, collate_fn=multiple_choice_collate_fn, **dataloader_kwargs
        )

        rookie_evaluator = Evaluator(
            label="mlmmlu_rookie",
            dataloader=build_dataloader(
                rookie_eval_dataset,
                collate_fn=multiple_choice_collate_fn,
                **dataloader_kwargs,
            ),
            metric_names=["MulticlassAccuracy"],
        )

        reserve_evaluator = Evaluator(
            label="mlmmlu_reserve",
            dataloader=build_dataloader(
                reserve_eval_dataset,
                collate_fn=multiple_choice_collate_fn,
                **dataloader_kwargs,
            ),
            metric_names=["MulticlassAccuracy"],
        )

        self.evaluators = [rookie_evaluator, reserve_evaluator]
