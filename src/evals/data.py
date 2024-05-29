# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import logging

from composer.utils import MissingConditionalImportError, dist

_glue_task_column_names = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
}

_superglue_task_column_names = {
    "boolq": ("question", "passage"),
    "cb": ("premise", "hypothesis"),
    # "copa": #("sentence1", "sentence2"), # ['premise', 'choice1', 'choice2', 'question
    # "multirc": ("paragraph", "sentence"), #  paragraph question answer
    # "record": ("question1", "question2"), ['passage', 'query', 'entities', 'entity_spans', 'answers', 'idx'
    "rte": ("premise", "hypothesis"),
    "wic": (
        "sentence1",
        "sentence2",
    ),  #'word','sentence1'  'sentence2',  'start1',  'start2',  'end1',  'end2',
    # "wsc": ("sentence1", "sentence2"), #'text','span1_index',  'span2_index',  'span1_text',  'span2_text',
    # "wsc.fixed": ("sentence1", "sentence2"), #'text','span1_index',  'span2_index',  'span1_text',  'span2_text',
    # "axb": ("sentence1", "sentence2"),
    # "axg": ("premise", "hypothesis"),
}

log = logging.getLogger(__name__)


def create_eval_dataset(
    task: str,
    tokenizer_name: str,
    split: str,
    max_seq_length: int = 256,
    max_retries: int = 10,
    num_workers: int = 0,
    dataset_name: str = "glue",
    task_column_names: dict = _glue_task_column_names,
):
    try:
        import datasets
        import transformers
    except ImportError as e:
        raise MissingConditionalImportError(
            extra_deps_group="nlp", conda_package="transformers"
        ) from e

    if task not in task_column_names:
        raise ValueError(f"task ({task}) must be one of {task_column_names.keys()}")

    if (max_seq_length % 8) != 0:
        log.warning(
            "For performance, a max_seq_length as a multiple of 8 is recommended."
        )

    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)  # type: ignore (thirdparty)

    log.info(f"Loading {task.upper()} on rank {dist.get_global_rank()}")
    download_config = datasets.DownloadConfig(max_retries=max_retries)
    dataset = datasets.load_dataset(
        dataset_name,
        task,
        split=split,
        download_config=download_config,
    )

    log.info(f"Starting tokenization by preprocessing over {num_workers} threads!")
    text_column_names = task_column_names[task]

    def tokenize_function(inp):
        # truncates sentences to max_length or pads them to max_length

        first_half = inp[text_column_names[0]]
        second_half = inp[text_column_names[1]] if text_column_names[1] in inp else None
        # a [SEP] is added between first_half and second_half
        return tokenizer(
            text=first_half,
            text_pair=second_half,
            padding="max_length",
            max_length=max_seq_length,
            truncation=True,
        )

    columns_to_remove = ["idx"] + [i for i in text_column_names if i is not None]

    assert isinstance(dataset, datasets.Dataset)
    safe_name = tokenizer_name.replace("/", ",")
    dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=None if num_workers == 0 else num_workers,
        batch_size=1000,
        remove_columns=columns_to_remove,
        load_from_cache_file=True,
    )
    return dataset


def create_glue_dataset(**kwargs):
    return create_eval_dataset(
        **kwargs, dataset_name="glue", task_column_names=_glue_task_column_names
    )


def create_superglue_dataset(**kwargs):
    return create_eval_dataset(
        **kwargs,
        dataset_name="aps/super_glue",
        task_column_names=_superglue_task_column_names,
    )
