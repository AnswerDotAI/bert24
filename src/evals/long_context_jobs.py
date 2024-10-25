# Copyright 2024 BERT24 authors
# SPDX-License-Identifier: Apache-2.0

# """Contains GLUE job objects for the simple_glue_trainer."""
import os
import sys
import json
from typing import List, Optional
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
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from composer.metrics.nlp import MaskedAccuracy

class MCQADataset(Dataset):
    """
    dicts with keys: ['question_id', 'question', 'context', 'qd_prompt', 'options', 'answer', 'answer_index']
    """
    def __init__(self,path_to_json):
        super().__init__()
        with open(path_to_json,'r') as f:
            self.items = json.load(f)
        self.tokenizer = AutoTokenizer.from_pretrained("bclavie/olmo_bert_template")
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        (prompt,answer) = self.items[idx]['qd_prompt'],self.items[idx]['answer']
        input_ids = self.tokenizer.encode(prompt + " ",add_special_tokens=False) + [self.tokenizer.mask_token_id]
        labels = self.tokenizer.encode(answer,add_special_tokens=False)
        return {"input_ids":input_ids,"labels":labels}

class MultipleChoiceMaskedAccuracy(MaskedAccuracy):
    def __init__(self, ANSWER_IDS, ignore_index: int = -100, dist_sync_on_step: bool = False):
        """
        ANSWER_IDS : dims of [multichoice_count]
        """
        super().__init__(ignore_index, dist_sync_on_step)
        self.ANSWER_IDS = ANSWER_IDS

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # predictions is a batch x num_classes tensor, take the argmax to get class indices
        preds = torch.argmax(preds, dim=-1)
        assert preds.shape == target.shape

        # mask out the padded indices
        mask = (target != self.ignore_index)
        masked_target = target[mask]
        masked_preds = preds[mask]

        self.correct += torch.sum(masked_preds == masked_target)
        self.total += mask.sum()


class TriviaQAJob(ClassificationJob):
    """TriviaQA."""

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
        max_duration: Optional[str] = "3ep",
        batch_size: Optional[int] = 48,
        load_path: Optional[str] = None,
        save_folder: Optional[str] = None,
        loggers: Optional[List[LoggerDestination]] = None,
        callbacks: Optional[List[Callback]] = None,
        precision: Optional[str] = None,
        opt_default_name: str = "decoupled_adamw",
        opt_lr: float = 1.0e-5,
        opt_betas: Tuple[float, float] = (0.9, 0.98),
        opt_eps: float = 1.0e-6,
        opt_weight_decay: float = 1.0e-5,
        **kwargs,
    ):
        super().__init__(
            model=model,
            tokenizer_name=tokenizer_name,
            job_name=job_name,
            seed=seed,
            task_name="triviaqa",
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
