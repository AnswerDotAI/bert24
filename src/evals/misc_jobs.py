# Copyright 2024 BERT24 authors
# SPDX-License-Identifier: Apache-2.0

# """Contains SuperGLUE job objects for the simple_glue_trainer."""
import os
import sys
from itertools import chain
from typing import List, Optional

# Add glue folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from composer import ComposerModel
from composer.core import Callback
from composer.core.evaluator import Evaluator
from composer.loggers import LoggerDestination
from composer.optim import ComposerScheduler, DecoupledAdamW
from torchmetrics.classification import MultilabelF1Score
from src.evals.data import create_quality_dataset, create_swag_dataset, create_ultrafeedback_dataset, create_eurlex_dataset
from src.evals.finetuning_jobs import (
    build_dataloader,
    multiple_choice_collate_fn,
    ClassificationJob,
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
                ending_names = ["ending0", "ending1", "ending2", "ending3"]
                first_sentences = [[context] * 4 for context in inp["sent1"]]
                question_headers = inp["sent2"]
                second_sentences = [
                    [f"{header} {inp[end][i]}" for end in ending_names]
                    for i, header in enumerate(question_headers)
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
                    k: [v[i : i + 4] for i in range(0, len(v), 4)]
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
            "shuffle": True,
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


class QualityJob(ClassificationJob):
    """Scrolls/Quality MCQ"""

    multiple_choice = True
    num_labels = 4

    def __init__(
        self,
        model: ComposerModel,
        tokenizer_name: str,
        job_name: Optional[str] = None,
        seed: int = 42,
        eval_interval: str = "300ba",
        scheduler: Optional[ComposerScheduler] = None,
        max_sequence_length: Optional[int] = 4096,
        max_duration: Optional[str] = "1ep",
        batch_size: Optional[int] = 8,
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
            task_name="rbiswasfc/scrolls-quality-mcq",  # TODO: upload to answer.ai org
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
            eps=1.0e-6,
            weight_decay=5.0e-06,
        )

        def tokenize_fn_factory(tokenizer, max_seq_length):
            def tokenize_fn(inp):
                option_names = ["A", "B", "C", "D"]
                question_headers = inp["question"]

                first_sentences = [
                    [f"{header} {inp[end][i]}" for end in option_names] for i, header in enumerate(question_headers)
                ]

                second_sentences = [[context] * 4 for context in inp["context"]]

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
            "num_workers": 0,
            "shuffle": False,
            "drop_last": False,
        }

        train_dataset = create_quality_dataset(split="train", **dataset_kwargs)
        self.train_dataloader = build_dataloader(
            train_dataset, collate_fn=multiple_choice_collate_fn, **dataloader_kwargs
        )
        quality_eval_dataset = create_quality_dataset(split="test", **dataset_kwargs)
        quality_evaluator = Evaluator(
            label="long_context_quality",
            dataloader=build_dataloader(
                quality_eval_dataset,
                collate_fn=multiple_choice_collate_fn,
                **dataloader_kwargs,
            ),
            metric_names=["MulticlassAccuracy"],
        )
        self.evaluators = [quality_evaluator]



class UltrafeedbackJob(ClassificationJob):
    """ultrafeedback binary classification."""

    num_labels = 2

    def __init__(
        self,
        model: ComposerModel,
        tokenizer_name: str,
        job_name: Optional[str] = None,
        seed: int = 42,
        eval_interval: str = "200ba",
        scheduler: Optional[ComposerScheduler] = None,
        max_sequence_length: Optional[int] = 2048,
        max_duration: Optional[str] = "1ep",
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
            weight_decay=1.0e-06,
        )

        def tokenize_fn_factory(tokenizer, max_seq_length):
            def tokenize_fn(inp):
                first_sentences = [
                    f"{prompt} {tokenizer.sep_token} {response_a}" for prompt, response_a in zip(inp['prompt'], inp['response_a'])
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
            "num_workers": 0,
            "shuffle": False,
            "drop_last": False,
        }
        train_dataset = create_ultrafeedback_dataset(split="train", **dataset_kwargs)
        self.train_dataloader = build_dataloader(train_dataset, **dataloader_kwargs)
        ultrafeedback_eval_dataset = create_ultrafeedback_dataset(
            split="test", **dataset_kwargs
        )
        ultrafeedback_evaluator = Evaluator(
            label="long_context_ultrafeedback",
            dataloader=build_dataloader(ultrafeedback_eval_dataset, **dataloader_kwargs),
            metric_names=["MulticlassAccuracy"],
        )
        self.evaluators = [ultrafeedback_evaluator]


class EurlexMultilabelF1Score(MultilabelF1Score):
    def __init__(self):
        super().__init__(num_labels=4271)

class EurlexJob(ClassificationJob):
    """Eurlex multi-label classification."""

    custom_eval_metrics = [EurlexMultilabelF1Score]
    num_labels = 4271

    def __init__(
        self,
        model: ComposerModel,
        tokenizer_name: str,
        job_name: Optional[str] = None,
        seed: int = 42,
        eval_interval: str = "350ba",
        scheduler: Optional[ComposerScheduler] = None,
        max_sequence_length: Optional[int] = 2048,
        max_duration: Optional[str] = "1ep",
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
            task_name="pietrolesci/eurlex-57k",
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
            "num_workers": 0,
            "shuffle": False,
            "drop_last": False,
        }

        eurlex_train_dataset = create_eurlex_dataset(split="train", **dataset_kwargs)
        eurlex_eval_dataset = create_eurlex_dataset(split="validation", **dataset_kwargs)
        eurlex_test_dataset = create_eurlex_dataset(split="test", **dataset_kwargs)

        # process labels: eurovoc_concepts ---
        classes = sorted(list(set(chain(
            *eurlex_train_dataset['eurovoc_concepts'], 
            *eurlex_eval_dataset['eurovoc_concepts'],
            *eurlex_test_dataset['eurovoc_concepts']
            )
        )))

        class2id = {class_:id for id, class_ in enumerate(classes)}

        def generate_labels(example):
            concepts = example['eurovoc_concepts']
            labels = [0. for i in range(len(classes))]

            for label in concepts:
                label_id = class2id[label]
                labels[label_id] = 1.
            example['labels'] = labels
            return example
        
        eurlex_train_dataset = eurlex_train_dataset.map(generate_labels, remove_columns='eurovoc_concepts')
        eurlex_eval_dataset = eurlex_eval_dataset.map(generate_labels, remove_columns='eurovoc_concepts')

        self.train_dataloader = build_dataloader(eurlex_train_dataset, **dataloader_kwargs)

        eurlex_evaluator = Evaluator(
            label="long_context_eurlex",
            dataloader=build_dataloader(eurlex_eval_dataset, **dataloader_kwargs),
            metric_names=["EurlexMultilabelF1Score"], # ["MulticlassAccuracy"],
        )

        self.evaluators = [eurlex_evaluator]