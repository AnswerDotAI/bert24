# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2024 OLMo authors
# SPDX-License-Identifier: Apache-2.0

"""Build a StreamingTextDataset dataset and dataloader for training."""

import logging
import math
import os
import sys
import json
from itertools import islice
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Union

import numpy as np
import torch
import transformers
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from streaming import Stream, StreamingDataset
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from streaming.base.format import reader_from_json
from streaming.base.spanner import Spanner
from composer.utils import dist

from transformers.tokenization_utils_base import BatchEncoding

# Add src folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from src.sequence_packer import BufferedIterable, GreedyBestFitSequencePacker

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

logger = logging.getLogger(__name__)


# Subclass DistributedSampler to use PCG64DXSM for shuffling
class DistributedSamplerPCG64DXSM(DistributedSampler):
    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            # use numpy's RNG PCG64DXSM instead of torch.randperm
            rng = np.random.Generator(np.random.PCG64DXSM(self.seed + self.epoch))
            indices = rng.permutation(len(self.dataset)).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


def build_tokenizer(
    om_tokenizer_config: DictConfig,
) -> Tokenizer:
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    resolved_om_tokenizer_config = om.to_container(om_tokenizer_config, resolve=True)
    tokenizer_kwargs = resolved_om_tokenizer_config.get(  # type: ignore
        "kwargs", {}
    )
    tokenizer_name = resolved_om_tokenizer_config["name"]  # type: ignore
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)

    # HuggingFace does not respect the model_max_length kwarg, and overrides it with
    # min(kwargs['model_max_length'], original_config['model_max_length']), so we
    # explicitly set it here
    tokenizer.model_max_length = tokenizer_kwargs.get(
        "model_max_length",
        int(1e30),
    )

    return tokenizer


class StreamingTextDataset(StreamingDataset):
    """Generic text dataset using MosaicML's StreamingDataset.

    Args:
        tokenizer (Tokenizer): HuggingFace tokenizer to
            tokenize samples.
        max_seq_len (int): The max sequence length of each sample.
        streams (Sequence[Stream], optional): One or more Streams to stream/cache samples from,
            which may be upsampled or downsampled. StreamingDataset uses either ``streams`` or
            ``remote``/``local``. Defaults to ``None``.
        remote (str, optional): Remote path or directory to download the dataset from. If ``None``,
            its data must exist locally. StreamingDataset uses either ``streams`` or
            ``remote``/``local``. Defaults to ``None``.
        local (str, optional): Local working directory to download shards to. This is where shards
            are cached while they are being used. Uses a temp directory if not set.
            StreamingDataset uses either ``streams`` or ``remote``/``local``. Defaults to ``None``.
        split (str, optional): Which dataset split to use, if any. If provided, we stream from/to
            the ``split`` subdirs of  ``remote`` and ``local``. Defaults to ``None``.
        download_retry (int): Number of download re-attempts before giving up. Defaults to ``2``.
        download_timeout (float): Number of seconds to wait for a shard to download before raising
            an exception. Defaults to ``60``.
        validate_hash (str, optional): Optional hash or checksum algorithm to use to validate
            shards. Defaults to ``None``.
        keep_zip (bool): Whether to keep or delete the compressed form when decompressing
            downloaded shards. If ``False``, keep iff remote is local or no remote. Defaults to
            `False``.
        epoch_size (int, optional): Provide this field iff you are weighting sub-datasets
            proportionally. Defaults to ``None``.
        predownload (int, optional): Target number of samples ahead to download the shards of while
            iterating. Defaults to ``100_000``.
        partition_algo (str): Which partitioning algorithm to use. Defaults to ``orig``.
        num_canonical_nodes (int, optional): Canonical number of nodes for shuffling with
            resumption. Defaults to ``None``, which is interpreted as the number of nodes of the
            initial run.
        batch_size (int, optional): Batch size of its DataLoader, which affects how the dataset is
            partitioned over the workers. Defaults to ``None``.
        shuffle (bool): Whether to iterate over the samples in randomized order. Defaults to
            ``False``.
        shuffle_algo (str): Which shuffling algorithm to use. Defaults to ``py1s``.
        shuffle_seed (int): Seed for Deterministic data shuffling. Defaults to ``9176``.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        max_seq_len: int,
        streams: Optional[Sequence[Stream]] = None,
        remote: Optional[str] = None,
        local: Optional[str] = None,
        split: Optional[str] = None,
        download_retry: int = 2,
        download_timeout: float = 60,
        validate_hash: Optional[str] = None,
        keep_zip: bool = False,
        epoch_size: Optional[int] = None,
        predownload: int = 100_000,
        partition_algo: str = "orig",
        num_canonical_nodes: Optional[int] = None,
        batch_size: Optional[int] = None,
        shuffle: bool = False,
        shuffle_algo: str = "py1s",
        shuffle_seed: int = 9176,
        cache_limit: Optional[int] = None,
        **kwargs: Dict[str, Any],
    ):
        group_method = kwargs.pop("group_method", None)
        if group_method is not None:
            raise NotImplementedError(
                "group_method is deprecated and has been removed.\nTo "
                + "concatenate, use the --concat_tokens "
                + "argument when creating your MDS dataset with concat_c4.py"
            )

        if kwargs is not None and len(kwargs) > 0:
            raise ValueError(f"StreamingTextDataset() got an unexpected keyword argument: {kwargs}")

        if local is not None and (remote is None or (local == remote)):
            if os.path.isdir(local):
                contents = set(os.listdir(local))
                if split not in contents:
                    raise ValueError(f"local directory {local} does not contain split {split}")

        # Build Dataset
        super().__init__(
            streams=streams,
            remote=remote,
            local=local,
            split=split,
            download_retry=download_retry,
            download_timeout=download_timeout,
            validate_hash=validate_hash,
            keep_zip=keep_zip,
            epoch_size=epoch_size,
            predownload=predownload,
            partition_algo=partition_algo,
            num_canonical_nodes=num_canonical_nodes,
            batch_size=batch_size,
            shuffle=shuffle,
            shuffle_algo=shuffle_algo,
            shuffle_seed=shuffle_seed,
            cache_limit=cache_limit,
        )
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    # How to tokenize a text sample to a token sample
    def _tokenize(self, text_sample):
        if self.tokenizer._pad_token is None:
            # Some tokenizers (e.g. GPT2 tokenizer) have no padding token which causes bugs
            raise RuntimeError("If tokenizing on-the-fly, tokenizer must have a pad_token_id")

        return self.tokenizer(text_sample["text"], truncation=True, padding="max_length", max_length=self.max_seq_len)

    def _read_binary_tokenized_sample(self, sample: BatchEncoding):
        seq_len = sample["len"] if "len" in sample else len(sample["input_ids"])

        input_ids = np.frombuffer(sample["input_ids"], dtype=np.int64).copy()
        if "attention_mask" in sample:
            attention_mask = np.frombuffer(sample["attention_mask"], dtype=np.int64).copy()
        else:
            attention_mask = np.ones_like(input_ids)

        # calculate padding
        pad_len = self.max_seq_len - seq_len

        # pad or truncate input_ids and attention_mask
        if pad_len > 0:
            input_ids = np.pad(input_ids, (0, pad_len), constant_values=self.tokenizer.pad_token_id)
            attention_mask = np.pad(attention_mask, (0, pad_len), constant_values=0)
        elif pad_len < 0:
            input_ids = input_ids[: self.max_seq_len]
            attention_mask = attention_mask[: self.max_seq_len]

        token_type_ids = np.zeros(self.max_seq_len, dtype=np.int64)

        return BatchEncoding(
            data={
                "input_ids": input_ids.tolist(),
                "attention_mask": attention_mask.tolist(),
                "token_type_ids": token_type_ids.tolist(),
            },
            n_sequences=1,
        )

    # How to process a sample
    def __getitem__(self, idx: int) -> Union[Dict[str, Any], torch.Tensor]:
        sample = super().__getitem__(idx)
        if "text" in sample:
            token_sample = self._tokenize(sample)
        elif "input_ids" in sample:
            token_sample = self._read_binary_tokenized_sample(sample)
        else:
            raise RuntimeError("StreamingTextDataset needs samples to have a `text` or `input_ids` column")
        return token_sample


class ConcatenatedSequenceCollatorWrapper:
    """Collator wrapper to add sequence_id to batch."""

    def __init__(self, base_collator: Callable, eos_token_id: Optional[int] = None, bos_token_id: Optional[int] = None):
        self.base_collator = base_collator
        if (eos_token_id is None) and (bos_token_id is None):
            raise ValueError("Must supply a value for either eos_token_id or bos_token_id, but got None for both.")
        if (eos_token_id is not None) and (bos_token_id is not None):
            raise ValueError(
                "Cannot use *both* EOS and BOS tokens for detecting sequence boundaries. "
                + "Please supply `eos_token_id` if sequences end with an EOS token, or use "
                + "`bos_token_id` if sequences start with a BOS token."
            )
        if eos_token_id is None:
            self.split_token_id = bos_token_id
            self.bos_mode = True
        else:
            self.split_token_id = eos_token_id
            self.bos_mode = False

    def __call__(self, examples: List[Any]) -> Dict[str, torch.Tensor]:
        batch = self.base_collator(examples)
        batch["sequence_id"] = self.get_sequence_id_from_batch(batch)
        return batch

    def get_sequence_id_from_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        assert self.split_token_id is not None
        is_separator = torch.eq(batch["input_ids"], self.split_token_id)
        cumulative_sep = torch.cumsum(is_separator, dim=1).to(batch["input_ids"].dtype)
        # If separator token is bos, we're already done
        if self.bos_mode:
            return cumulative_sep

        # If separator token is eos, right shift 1 space
        left_zeros = cumulative_sep.new_zeros((cumulative_sep.shape[0], 1))
        return torch.cat([left_zeros, cumulative_sep[:, :-1]], dim=1)


def build_streaming_dataset(
    cfg: DictConfig,
    tokenizer: Tokenizer,
    device_batch_size: int,
):
    # build streams
    streams_dict = cfg.dataset.get("streams", None)
    streams = None
    if streams_dict is not None:
        streams = []
        for _, stream in streams_dict.items():
            streams.append(
                Stream(
                    remote=stream.get("remote", None) or cfg.dataset.get("remote", None),
                    local=stream.get("local", None) or cfg.dataset.get("local", None),
                    split=stream.get("split", None) or cfg.dataset.get("split", None),
                    proportion=stream.get("proportion", None),
                    repeat=stream.get("repeat", None),
                    choose=stream.get("choose", None),
                    download_retry=stream.get("download_retry", None) or cfg.dataset.get("download_retry", 2),
                    download_timeout=stream.get("download_timeout", None) or cfg.dataset.get("download_timeout", 60),
                    validate_hash=stream.get("validate_hash", None) or cfg.dataset.get("validate_hash", None),
                    keep_zip=stream.get("keep_zip", None) or cfg.dataset.get("keep_zip", False),
                )
            )

    # build dataset potentially with streams
    dataset = StreamingTextDataset(
        tokenizer=tokenizer,
        max_seq_len=cfg.dataset.max_seq_len,
        streams=streams,
        remote=cfg.dataset.get("remote", None),
        local=cfg.dataset.get("local", None),
        split=cfg.dataset.get("split", None),
        download_retry=cfg.dataset.get("download_retry", 2),
        download_timeout=cfg.dataset.get("download_timeout", 60),
        validate_hash=cfg.dataset.get("validate_hash", None),
        keep_zip=cfg.dataset.get("keep_zip", False),
        epoch_size=cfg.dataset.get("epoch_size", None),
        predownload=cfg.dataset.get("predownload", 100_000),
        partition_algo=cfg.dataset.get("partition_algo", "orig"),
        num_canonical_nodes=cfg.dataset.get("num_canonical_nodes", 128),
        batch_size=device_batch_size,
        shuffle=cfg.dataset.get("shuffle", False),
        shuffle_algo=cfg.dataset.get("shuffle_algo", "py1s"),
        shuffle_seed=cfg.dataset.get("shuffle_seed", 9176),
        cache_limit=cfg.dataset.get("cache_limit", None),
    )
    return dataset


def build_no_streaming_dataset(
    cfg: DictConfig,
    tokenizer: Tokenizer,
    pad_sequences: bool = True,
):
    return NoStreamingDataset(
        tokenizer=tokenizer,
        local=cfg.dataset.get("local", None),
        split=cfg.dataset.get("split", None),
        max_seq_len=cfg.dataset.max_seq_len,
        pad_sequences=pad_sequences,
    )


def build_text_dataloader(
    cfg: DictConfig,
    tokenizer: Tokenizer,
    device_batch_size: int,
    device_microbatch_size: int,
):
    assert cfg.name == "text", f"Tried to build text dataloader with cfg.name={cfg.name}"
    if cfg.dataset.get("group_method", None) is not None:
        raise NotImplementedError(
            "group_method is deprecated and has been removed.\nTo "
            + "concatenate, use the --concat_tokens "
            + "argument when creating your MDS dataset with convert_dataset.py"
        )

    if cfg.dataset.get("streaming", True):
        dataset = build_streaming_dataset(cfg, tokenizer, device_batch_size)
        sampler = None
    else:
        assert cfg.dataset.get("local", None) is not None, "Local path must be provided when not using streaming"
        # sequence packing should never use padded sequences, regular dataloaders may if tokenizing on the fly
        dataset = build_no_streaming_dataset(
            cfg, tokenizer=tokenizer, pad_sequences=not cfg.get("sequence_packing", False)
        )
        sampler = DistributedSamplerPCG64DXSM(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_global_rank(),
            shuffle=cfg.dataset.get("shuffle", False),
            seed=cfg.dataset.get("shuffle_seed", 9176),
            drop_last=cfg.drop_last,
        )

    mlm_probability = cfg.dataset.get("mlm_probability", None)
    # only use sequence packing if using the no_streaming_dataset
    if not cfg.dataset.get("streaming", True) and cfg.get("sequence_packing", False):
        dataloader = DataLoader(
            dataset,
            collate_fn=lambda x: x,
            batch_size=device_batch_size,
            drop_last=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.get("pin_memory", True),
            prefetch_factor=cfg.get("prefetch_factor", 2),
            persistent_workers=cfg.get("persistent_workers", True),
            timeout=cfg.get("timeout", 0),
            sampler=sampler,
        )
        sequence_packer = GreedyBestFitSequencePacker.from_composer(
            dataloader,
            batch_size=device_batch_size,
            micro_batch_size=device_microbatch_size,
            max_seq_len=cfg.dataset.max_seq_len,
            buffer_size=cfg.get("packing_buffer_size", 5 * device_batch_size),
            mask_token_id=tokenizer.mask_token_id,
            pad_token_id=tokenizer.pad_token_id,
            mask_prob=mlm_probability,
            seed=cfg.dataset.get("shuffle_seed", 42),
            batch_size_warmup_min_size=cfg.get("batch_size_warmup_min_size", None),
            batch_size_warmup_tokens=cfg.get("batch_size_warmup_tokens", None),
            world_size=dist.get_world_size(),
        )
        return BufferedIterable(sequence_packer, buffer_size=cfg.get("packing_prefetch_factor", 5))
    else:
        collate_fn = transformers.DataCollatorForLanguageModeling(
            tokenizer=dataset.tokenizer, mlm=mlm_probability is not None, mlm_probability=mlm_probability
        )

        eos_token_id = cfg.dataset.get("eos_token_id")
        bos_token_id = cfg.dataset.get("bos_token_id")
        if (eos_token_id is not None) or (bos_token_id is not None):
            # Note: Will raise an error if both are non-None
            collate_fn = ConcatenatedSequenceCollatorWrapper(
                base_collator=collate_fn, eos_token_id=eos_token_id, bos_token_id=bos_token_id
            )

        return DataLoader(
            dataset,
            collate_fn=collate_fn,
            batch_size=device_batch_size,
            drop_last=cfg.drop_last,
            num_workers=cfg.num_workers,
            pin_memory=cfg.get("pin_memory", True),
            prefetch_factor=cfg.get("prefetch_factor", 2),
            persistent_workers=cfg.get("persistent_workers", True),
            timeout=cfg.get("timeout", 0),
            sampler=sampler,
        )


class NoStreamingDataset(Dataset):
    """
    A dataset class that can read data with raw mds-format (mosaic streaming-format without compression)
    from local. In comparison with `StreamingTextDataset` that also can read data with mds-format from local,
    this class is slimmer, more efficient, and does not contain redundant code required for streaming.
    """

    def __init__(
        self,
        local: str,
        split: Optional[str],
        max_seq_len: int,
        tokenizer: Optional[Tokenizer] = None,
        pad_sequences: bool = True,
    ) -> None:
        super().__init__()
        if split is not None:
            split_path = os.path.join(local, split)
        else:
            split_path = local
        index_file_path = os.path.join(split_path, "index.json")
        obj = json.load(open(index_file_path))
        self.shards = []
        for info in obj["shards"]:
            shard = reader_from_json(local, split, info)
            raw_filename = os.path.join(shard.dirname, shard.split, shard.raw_data.basename)
            assert os.path.isfile(raw_filename), f"Raw file {raw_filename} does not exist"
            shard.validate(True)
            self.shards.append(shard)
        samples_per_shard = np.array([shard.samples for shard in self.shards], np.int64)
        self.len = samples_per_shard.sum()
        self.spanner = Spanner(samples_per_shard)
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.pad_sequences = pad_sequences

    def _tokenize(self, text_sample):
        assert self.tokenizer is not None, "Tokenizer required if data is not pretokenized"
        if self.tokenizer._pad_token is None:
            # Some tokenizers (e.g. GPT2 tokenizer) have no padding token which causes bugs
            raise RuntimeError("If tokenizing on-the-fly, tokenizer must have a pad_token_id")

        return self.tokenizer(
            text_sample["text"],
            truncation=True,
            padding="max_length" if self.pad_sequences else False,
            max_length=self.max_seq_len,
        )

    def __getitem__(self, index: int):
        shard_id, shard_sample_id = self.spanner[index]
        shard = self.shards[shard_id]
        sample = shard[shard_sample_id]
        if "input_ids" in sample:
            for k in list(sample.keys()):
                if isinstance(sample[k], np.ndarray):
                    if sample[k][0] != 50281:
                        sample[k] = np.insert(sample[k], 0, 50281)[: self.max_seq_len]
                    if sample[k][-1] != 50282:
                        sample[k] = sample[k][: self.max_seq_len - 1]
                        sample[k] = np.append(sample[k], 50282)
                    sample[k] = sample[k][: self.max_seq_len]
                else:
                    del sample[k]
            if "attention_mask" not in sample:
                sample["attention_mask"] = np.ones_like(sample["input_ids"])
            return sample
        elif "text" in sample:
            return self._tokenize(sample)
        else:
            RuntimeError("Data sample must contain a field with `input_ids` or `text`")

    def __len__(self):
        return self.len


# Helpful to test if your dataloader is working locally
# Run `python data.py  --local_path [local] [--remote_path remote, optional]` and verify that batches are printed out
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="the name of the tokenizer to use")
    parser.add_argument("--local_path", type=str, required=True, help="the path to the local copy of the dataset")
    parser.add_argument(
        "--remote_path", type=str, default=None, help="the path to the remote copy to stream from (optional)"
    )
    parser.add_argument("--split", type=str, default="val", help="which split of the dataset to use")
    parser.add_argument("--max_seq_len", type=int, default=32, help="max sequence length to test")

    args = parser.parse_args()

    if args.remote_path is not None:
        print(f"Reading {args.split} split from {args.local_path} <- streamed from <- {args.remote_path}")
    else:
        print(f"Reading {args.split} split from {args.local_path}")

    cfg = {
        "name": "text",
        "dataset": {
            "local": args.local_path,
            "remote": args.remote_path,
            "split": args.split,
            "shuffle": False,
            "max_seq_len": args.max_seq_len,
            "keep_zip": True,  # in case we need compressed files after testing
        },
        "drop_last": False,
        "num_workers": 4,
        "pin_memory": True,
    }
    cfg = om.create(cfg)
    device_batch_size = 2

    tokenizer_cfg = {"name": args.tokenizer, "kwargs": {}}
    tokenizer_cfg["kwargs"] = {"model_max_length": args.max_seq_len}
    tokenizer_cfg = om.create(tokenizer_cfg)
    tokenizer = build_tokenizer(tokenizer_cfg)

    loader = build_text_dataloader(cfg, tokenizer, device_batch_size)
    tokenizer = loader.dataset.tokenizer  # type: ignore
    for batch_ix, batch in enumerate(islice(loader, 5)):
        print("\n")
        print("#" * 20, f"Batch {batch_ix}", "#" * 20)
        for k, v in batch.items():
            print(k, v.shape, v.dtype)
        for sample_ix, token_sample in enumerate(batch["input_ids"]):
            print("-" * 20, f" Sample {sample_ix} ", "-" * 20)
            print(tokenizer.decode(token_sample))
