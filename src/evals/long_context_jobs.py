# Copyright 2024 BERT24 authors
# SPDX-License-Identifier: Apache-2.0

# """Contains GLUE job objects for the simple_glue_trainer."""
import os
import sys
import json
from typing import List, Optional
from multiprocessing import cpu_count
import torch
from typing import Tuple

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
from datasets import load_dataset
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch

"""
Q: if for each question, we want to create 5 training examples, how is that expressed?

Background: There is a pipeline going JSON->MCQADataset->OtherThing(=Trainer?)-> training event at training time.

Possible answers:

1. 5 items in the JSON which is loaded into MCQADataset
2. 1 item in the JSON, but the MCQADataset processes it to express 5 items via __geitem__
3. 1 item in the JSON, 1 item in the MCQADataset, but some other process (the Trainer?) expresses it as 5 items before trainingt time.

QQ: How does an existing multiple-choice QA dataset already express this?

Wayde believes re typical HF dataset practice:
- each HF Dataset item is a single question
- 
"""

# class MCQADatasetModified(Dataset):
#     """
#     dicts with keys: ['question_id', 'question', 'context', 'qd_prompt', 'options', 'answer', 'answer_index']
#     """
#     def __init__(self,path_to_json):
#         super().__init__()
#         with open(path_to_json,'r') as f:
#             self.items = json.load(f)
#         self.tokenizer = AutoTokenizer.from_pretrained("bclavie/olmo_bert_template")
#     def __len__(self): return len(self.items)
#     def __getitem__(self, idx):
#         (prompt, choices, answer_idx) = self.items[idx]['qd_prompt'], self.items[idx]["options"], self.items[idx]['answer_index']
#         input_ids = self.tokenizer.encode([prompt]*len(choices), choices)
#         labels = answer_idx # list of token ids expressing the correct answer for one example
#         return {"input_ids":input_ids,"labels":labels}

# class MCQADataset(Dataset):
#     """
#     dicts with keys: ['question_id', 'question', 'context', 'qd_prompt', 'options', 'answer', 'answer_index']
#     """
#     def __init__(self,path_to_json):
#         super().__init__()
#         with open(path_to_json,'r') as f:
#             self.items = json.load(f)
#         self.tokenizer = AutoTokenizer.from_pretrained("bclavie/olmo_bert_template")
#     def __len__(self): return len(self.items)
#     def __getitem__(self, idx):
#         (prompt, answer) = self.items[idx]['qd_prompt'],self.items[idx]['answer']
#         input_ids = self.tokenizer.encode(prompt + " ",add_special_tokens=False) + [self.tokenizer.mask_token_id]
#         labels = self.tokenizer.encode(answer,add_special_tokens=False) # list of token ids expressing the correct answer for one example
#         return {"input_ids":input_ids,"labels":labels}

# class MultipleChoiceMaskedAccuracy(MaskedAccuracy):
#     def __init__(self, ANSWER_IDS, ignore_index: int = -100, dist_sync_on_step: bool = False):
#         """
#         ANSWER_IDS : dims of [multichoice_count]
#         """
#         super().__init__(ignore_index, dist_sync_on_step)
#         self.ANSWER_IDS = ANSWER_IDS

#     def update(self, preds: torch.Tensor, target: torch.Tensor):
#         # predictions is a batch x num_classes tensor, take the argmax to get class indices
#         preds = torch.argmax(preds, dim=-1)
#         assert preds.shape == target.shape

#         # mask out the padded indices
#         mask = (target != self.ignore_index)
#         masked_target = target[mask]
#         masked_preds = preds[mask]

#         self.correct += torch.sum(masked_preds == masked_target)
#         self.total += mask.sum()




@dataclass
class DataCollatorForMultipleChoice:
    """
    Produces the traditional MCQA batch data format, which is as follows:
    - inputs have dimensions: (batch_size x num_choices x seq_length)
    - labels have dimensions: (batch_size x 1) where each label is the index of the correct answer for that example.

    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


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

        # grab longcontext dataset
        ds = load_dataset("json", data_files="triviamcqa.json")
        # assert: ds is a dict with keys 'train' and 'validation', 
        #   where ds['train'][0] has keys ['question_id', 'question', 'context', 'qd_prompt', 'options', 'answer', 'answer_index']
        train_ds = ds["train"]
        val_ds = ds["validation"]

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        train_ds_enc = self.preprocess(train_ds)
        val_ds_enc = self.preprocess(val_ds)
        # assert: train_ds_enc enriches train_ds with additional per-item keys: ['input_ids', 'attention_mask', 'labels', 'token_type_ids']

        # dataset_kwargs = {
        #     "task": self.task_name,
        #     "tokenizer_name": self.tokenizer_name,
        #     "max_seq_length": self.max_sequence_length,
        # }

        dataloader_kwargs = {
            "batch_size": self.batch_size,
            "num_workers": min(8, cpu_count() // torch.cuda.device_count()),
            "drop_last": False,
        }
       
        self.train_dataloader = build_dataloader(train_ds_enc, collate_fn=DataCollatorForMultipleChoice, **dataloader_kwargs)

        evaluator = Evaluator(
            label="lc_trivia_mcqa",
            dataloader=build_dataloader(val_ds_enc, collate_fn=DataCollatorForMultipleChoice, **dataloader_kwargs),
            metric_names=["MulticlassAccuracy"],
        )

        self.evaluators = [evaluator]

    def preprocess(self, example):
        question = example["qd_prompt"]
        choices = example["options"]
        correct_answer_idex = example["answer_index"]

        inputs = self.tokenizer([question] * len(choices), choices, truncation=True)

        inputs["labels"] = correct_answer_idex
        return inputs