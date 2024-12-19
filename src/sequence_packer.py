# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Generic, Iterable, NamedTuple, Optional, TypeVar, Any, Union, Sequence
from composer.core.types import Batch

import numpy as np
import torch
from numba import njit


import math
from composer.core import Time


class BatchSizeWarmupScheduler:
    def __init__(
        self,
        min_batch_size: int,
        max_batch_size: int,
        warmup_tokens: Union[str, Time, int],
        world_size: int,
    ):
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size

        if isinstance(warmup_tokens, str):
            self.warmup_tokens = Time.from_timestring(warmup_tokens).value
        elif isinstance(warmup_tokens, Time):
            self.warmup_tokens = warmup_tokens.value
        else:
            self.warmup_tokens = warmup_tokens
        self.warmup_tokens = math.ceil(self.warmup_tokens / world_size)
        self._step_thresholds = self._calculate_step_thresholds()

    def _calculate_step_thresholds(self):
        total_batch_sizes = sum(range(self.min_batch_size, self.max_batch_size))
        steps_per_unit = self.warmup_tokens / total_batch_sizes

        thresholds = []
        cumsum = 0
        for batch_size in range(self.min_batch_size, self.max_batch_size):
            cumsum += batch_size
            steps = math.ceil(steps_per_unit * cumsum)
            thresholds.append(steps)
        return thresholds

    def __call__(self, current_step: int) -> int:
        if current_step >= self.warmup_tokens:
            return self.max_batch_size

        for i, threshold in enumerate(self._step_thresholds):
            if current_step < threshold:
                return self.min_batch_size + i

        # should never hit this, but just in case
        return self.max_batch_size


class SequencePackerBatchOutputTuple(NamedTuple):
    masked_pseqs: torch.Tensor
    labels: Optional[torch.Tensor]
    cu_seq_lens: list[torch.Tensor]
    max_cu_seq_len: list[torch.Tensor]


class SequencePacker(ABC):
    def __init__(
        self,
        # params defining the incoming batches of seqs
        src_iterable: Iterable[list[list[int]]],
        src_batch_size: int,
        src_max_seq_len: int,
        # params defining outgoing batches of pseqs
        out_batch_size: int,
        out_pseq_len: int,
        # params defining internal behavior
        buffer_size: int,
        pad_token_id: int = -1,
        mask_token_id: int = 0,
        ignore_token_id: int = -100,
        mask_prob: float = 0.3,
        seed=42,
        suppress_masking: bool = False,
        batch_size_warmup_min_size: Optional[int] = None,
        batch_size_warmup_tokens: Optional[Union[str, Time]] = None,
        world_size: int = 1,
    ):
        """
        Takes batches of unpacked, unpadded sequences (seqs) to batches of packed and padded sequences (pseqs).

        Every input batch must be a list[list[int]], a list of variable-length sequences of tokens.

        Every output batch is a tuple (masked_inputs:Tensor, labels:Tensor, seq_starts_and_end:list).

        It performs this streamwise, taking an iterable as the source of incoming batches, and
        presents itself as an iterable of outgoing batches.

        Args:
            src_iterable: An iterable (e.g., a DataLoader), whose iterator yields one incoming batch,
                        where a batch is a list of unpadded, variable-length Sequences of token
                        IDs. Since this only needs to be an Iterable, it could also be a generator object
                         like the result of `itertools.batched(dataset_list,batch_size))`

            src_batch_size:  This is the INCOMING batch size, the number of seqs in one batch yielded
                          from `src_iterable`'s iterator.

            src_max_seq_len: The maximum number of tokens in a seq within an incoming batch.

            out_batch_size: the number of pseqs (packed seqs) in one outgoing batch

            out_pseq_len: the number of tokens per packed seq, in every outgoing batch

            buffer_size: The maximum number of seqs which may be buffered internally.

            pad_token_id: The token ID used for padding the space which cannot be filled to reach out_pseq_len.

            mask_token_id: The token ID used for masking tokens in the input sequence.

            ignore_token_id: The token ID used to ignore tokens. Expected to be applied to every non-masked token, so the model only trains on predictions of masked tokens.

            suppress_masking: If True, the sequence packer will not perform masked language modeling.

            batch_size_warmup_min_size: If not None, the sequence packer will gradually increase the batch size from batch_size_warmup_min_size to out_batch_size over the course of the warmup_tokens.
                                    batch_size_warmup_min_size must be a multiple of micro_batch_size.

            batch_size_warmup_tokens: If not None, the sequence packer will gradually increase the batch size from batch_size_warmup_min_size to out_batch_size over the course of the warmup_tokens.

            world_size: The number of processes participating in this training run. batch_size_warmup_min_size is divided by this number.
        """
        assert buffer_size >= out_batch_size, f"required that {buffer_size=} >= {out_batch_size=}"
        self.src_dataloader_len = len(src_iterable)
        self.src_iterable = src_iterable
        self.src_batch_size = src_batch_size
        self.out_batch_size = out_batch_size
        self.out_pseq_len = out_pseq_len
        self.buffer_size = buffer_size
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.ignore_token_id = ignore_token_id
        self.mask_prob = mask_prob
        self.suppress_masking = suppress_masking
        # internals
        self.buffer = deque()  # internal buffer holds individual seqs, as tensors.
        # for stats to report packing efficiency.
        self._seqs_consumed = 0
        self._seqs_emitted = 0
        # Set random seed
        self.seed = seed
        self.epoch = -1
        self._token_count = 0
        self.batch_size_scheduler = None
        if batch_size_warmup_min_size is not None and batch_size_warmup_tokens is not None:
            self.batch_size_scheduler = BatchSizeWarmupScheduler(
                batch_size_warmup_min_size, out_batch_size, batch_size_warmup_tokens, world_size
            )
        else:
            self.batch_size_scheduler = None

    @property
    def seqs_emitted(self):
        "Number of seqs, incoming from src_iterable, which have been emitted in OUTGOING batches."
        return self._seqs_emitted

    @property
    def seqs_consumed(self):
        "Number of seqs, incoming from src_iterable, which have been consumed."
        return self._seqs_consumed

    def _reset_state(self):
        self.epoch += 1
        self.buffer.clear()
        self._seqs_consumed = 0
        self._seqs_emitted = 0
        self.np_rng = np.random.default_rng(self.epoch + self.seed)

        # Update the epoch for the sampler
        if isinstance(self.src_iterable, torch.utils.data.dataloader.DataLoader):
            if isinstance(self.src_iterable.sampler, torch.utils.data.distributed.DistributedSampler):
                self.src_iterable.sampler.set_epoch(self.epoch)

    def __iter__(self):
        self._reset_state()
        self.src_iterator = iter(self.src_iterable)
        return self._generate_batches()

    def __len__(self):
        # rather than estimate the packed length of the dataset, we rely on Composer's ability
        # to schedule training the using the number of batches or tokens instead of epochs.
        return None

    def _fill_buffer(self, max_items_to_add=float("inf")) -> int:
        """
        Refills the internal buffer.

        - max_items_to_add: an amount less than or equal to the number of items to add

        Returns: the number of items actually added.

        The default implementation of this simply extends to src.buffer, which is
        initialized as a list in __init__. Subclasses which want to use a different data
        structure for internal buffering should override this method and also add
        code in __init__ to initialize src.buffer appropriately.

        Any implementation of this MUST never place more than self.buffer_size items
        in the internal buffer.
        """
        items_added = 0
        # NOTE: this should be >=, kept as is to match model training code
        # TODO: change if training a new model
        while (self.buffer_size - len(self.buffer)) > self.src_batch_size:
            try:
                # if pulling another batch would fetch more than the requested max, stop
                if max_items_to_add < float("inf"):
                    if (items_added + self.src_batch_size) > max_items_to_add:
                        # print("Not adding, because of max_items_to_fetch")
                        break
                incoming_batch = next(self.src_iterator)
                assert (
                    len(incoming_batch) <= self.src_batch_size
                ), f"expected {len(incoming_batch)=} <= {self.src_batch_size=}"
                for item in incoming_batch:
                    if len(item["input_ids"]) > 0:  # ignore empty sequences
                        self.buffer.append(item["input_ids"])
                        items_added += 1
                        self._seqs_consumed += 1
            except StopIteration:
                break
        return items_added

    def _generate_batches(self):
        """
        Generates batches of packed sequences.

        The returned generator's iterator will always, when next() is called on it, either:
         - return a valid tuple batch (masked_batch, labels, cu_seq_lens,max_seq_lens)
         - raise StopIteration
        """
        while True:
            retval = self._create_batch()
            if retval is None:
                break
            batch, lst_cu_seq_lens = retval

            assert isinstance(retval, tuple), f"Unexpected {type(retval)=}"
            assert isinstance(retval[0], np.ndarray), f"Unexpected {type(retval[0])=}"
            assert isinstance(retval[1], list), f"Unexpected {type(retval[1])=}"

            cu_seq_lens = [torch.tensor(x, dtype=torch.int32) for x in lst_cu_seq_lens]
            max_seq_lens = [torch.max(x[1:] - x[:-1]).item() for x in cu_seq_lens]
            assert isinstance(cu_seq_lens, list), f"Unexpected {type(cu_seq_lens)=}"
            if self.suppress_masking:
                yieldval = {
                    "input_ids": torch.from_numpy(batch),
                    "labels": None,
                    "cu_seqlens": cu_seq_lens,
                    "max_seqlen": max_seq_lens,
                }
            else:
                (masked_batch, labels) = SequencePacker.mlm_masking(
                    batch, self.mask_prob, self.mask_token_id, self.pad_token_id, self.ignore_token_id, self.np_rng
                )
                yieldval = {
                    "input_ids": torch.from_numpy(masked_batch),
                    "labels": torch.from_numpy(labels),
                    "cu_seqlens": cu_seq_lens,
                    "max_seqlen": max_seq_lens,
                    "attention_mask": torch.from_numpy(np.where(batch == self.pad_token_id, 0, 1)),
                }
                self._token_count += yieldval["attention_mask"].sum().item()
            # # assert isinstance(yieldval[0], torch.Tensor), f"Unexpected {type(yieldval[0])=}"
            # if not self.suppress_masking:
            #     assert isinstance(yieldval[1], torch.Tensor), f"Unexpected {type(yieldval[1])=}"
            # assert isinstance(yieldval[2], list), f"Unexpected {type(yieldval[2])=}"
            # if yieldval[2]:
            #     assert isinstance(yieldval[2][0], torch.Tensor), f"Unexpected {type(yieldval[2][0])=}"
            yield yieldval

    @staticmethod
    def mlm_masking(
        seq: np.ndarray,
        mask_prob: float,
        mask_token: int,
        pad_token: int = -1,
        ignore_index: int = -100,
        np_rng=np.random.default_rng(),
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.

        This is exactly a numpy version of transformers' `DataCollatorForLanguageModeling.torch_mask_tokens`
        https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L827

        It performs masking in a way that produces on expectation the following masked inputs:
         - (1-mask_prob) of the original positions will be untouched.
         - mask_prob * 80%  of the original positions get replaced with a mask token
         - mask_prob * 10%  of the original positions get replaced with a random token
         - mask_prob * 10%  of the original positions also remain untouched.
        This generates the masked_inputs.

        It also generates a labels array, which has ignore tokens in the (1-mask_prob) positions

        These proportions are expectation values since the random transformation is performed
        independently per element. (This is why it is agnostic wrt shape.)

        Args:
          seq (np.ndarray): the input token IDs (e.g., a sequence, or batch of seqs)
          mask_prob (float): probability of initially masking a token, in the first "wave" of masking
          mask_token (int): token to use for masking
          ignore_index (int): the token indicating that position should be ignored during training. We call it `ignore_index` to conform to the API of the cross entropy loss function.

        Returns:
            tuple[np.array,np.array]: (masked_seq, labels)
                masked_seq: the input seq with some tokens replaced by `mask_token`
                labels: the original input seq with non-masked tokens replaced by `ignore_index`
        """
        # Create labels
        labels = np.where(seq == pad_token, ignore_index, seq)

        # Create a single mask
        rand = np_rng.random(seq.shape)

        # Partition the probability space appropriately using a single mask
        # 80% of the time, we mask the token
        mask_mask = rand < mask_prob * 0.8
        # 10% of the time, we replace the token with a random token
        random_mask = (rand >= mask_prob * 0.8) & (rand < mask_prob * 0.9)
        # 10% of the time, we keep the token the same
        keep_mask = (rand >= mask_prob * 0.9) & (rand < mask_prob)

        # We only compute loss over the tokens marked for masking
        labels = np.where(mask_mask | random_mask | keep_mask, labels, ignore_index)

        # Apply masking
        seq = np.where(mask_mask, mask_token, seq)

        # Apply random replacement
        random_words = np_rng.integers(0, np.max(seq) + 1, size=seq.shape)
        seq = np.where(random_mask, random_words, seq)

        return seq, labels

    @abstractmethod
    def _create_batch(self) -> Optional[tuple[np.ndarray, list[list[int]]]]:
        """
        Returns a batch of packed sequences with its cumulative seq length information.

        Or else, returns None if it cannot build a full outgoing batch.

        Must mutate self.buffer to remove the sequences that are packed into the batch.

        Returns:
            (out_batch,cumulative_seq_len):tuple[torch.tensor, list[list[int]]]
            where:
                - out_batch is a tensor of shape (out_batch_size, out_pseq_len);
                - cum_seq_lens is a list of lists, where the outer list is of len out_batch_size,
                    and each inner list is of varying length, and contains the start positions of
                    every seq in the pseq, and the end position of the last seq in the pseq. This end
                    position is necessary to communicate if any padding tokens were added.
        """
        pass


@njit
def find_best_fit(remaining_spaces, seq_len):
    valid_spaces = seq_len <= remaining_spaces
    if np.any(valid_spaces):
        valid_space_sizes = remaining_spaces[valid_spaces]
        best_fit_idx = np.argmin(valid_space_sizes)
        return np.arange(len(remaining_spaces))[valid_spaces][best_fit_idx]
    return -1


class GreedyBestFitSequencePacker(SequencePacker):
    @classmethod
    def from_composer(
        cls,
        src_iterable: Iterable[list[list[int]]],
        batch_size: int = 512,
        micro_batch_size: int = 32,
        max_seq_len: int = 1024,
        buffer_size: int = 5120,
        # token values
        pad_token_id: int = -1,
        mask_token_id: int = 0,
        ignore_token_id: int = -100,
        mask_prob: float = 0.3,
        # transform values
        seed=42,
        suppress_masking=False,
        batch_size_warmup_min_size: Optional[int] = None,
        batch_size_warmup_tokens: Optional[Union[str, Time]] = None,
        world_size: int = 1,
    ) -> "GreedyBestFitSequencePacker":
        if batch_size_warmup_min_size is not None:
            if batch_size_warmup_min_size % micro_batch_size != 0:
                raise ValueError(f"{batch_size_warmup_min_size=} must be a multiple of {micro_batch_size=}")
            batch_size_warmup_min_size = int(batch_size_warmup_min_size / micro_batch_size)
        return cls(
            # input shape
            src_iterable=src_iterable,
            src_batch_size=batch_size,
            src_max_seq_len=max_seq_len,
            # output shape
            out_batch_size=int(batch_size / micro_batch_size),
            out_pseq_len=int(micro_batch_size * max_seq_len),
            # internal
            buffer_size=buffer_size,
            # transformation
            pad_token_id=pad_token_id,
            mask_token_id=mask_token_id,
            ignore_token_id=ignore_token_id,
            mask_prob=mask_prob,
            seed=seed,
            suppress_masking=suppress_masking,
            batch_size_warmup_min_size=batch_size_warmup_min_size,
            batch_size_warmup_tokens=batch_size_warmup_tokens,
            world_size=world_size,
        )

    def _create_batch(self) -> Optional[tuple[np.ndarray, list[list[int]]]]:
        if self.batch_size_scheduler:
            self.out_batch_size = self.batch_size_scheduler(self._token_count)

        batch = np.full(
            (self.out_batch_size, self.out_pseq_len), self.pad_token_id, dtype=np.int64
        )  # the pseqs being constructed
        seq_counts = np.zeros(self.out_batch_size, dtype=np.int32)  # the count of seqs per pseq
        cum_seq_lens = [[0] for _ in range(self.out_batch_size)]
        remaining_spaces = np.full(
            (self.out_batch_size,), self.out_pseq_len, dtype=np.int32
        )  # the space remaining per pseq
        temp_buffer = []

        while True:
            # Check if buffer has more items, and if not replenish
            if not self.buffer:
                items_to_fetch = self.buffer_size - len(temp_buffer)
                items_added = self._fill_buffer(items_to_fetch)
                if items_added == 0:
                    break

            seq = self.buffer.popleft()
            seq_len = len(seq)

            # Find the best fit (smallest space that can accommodate the sequence)
            best_fit_idx = find_best_fit(remaining_spaces, seq_len)
            if best_fit_idx != -1:
                end_pos = self.out_pseq_len - remaining_spaces[best_fit_idx]
                batch[best_fit_idx, end_pos : end_pos + seq_len] = seq
                seq_counts[best_fit_idx] += 1
                remaining_spaces[best_fit_idx] -= seq_len
                cum_seq_lens[best_fit_idx].append(cum_seq_lens[best_fit_idx][-1] + seq_len)
            else:
                # Can't fit the sequence, save for next batch
                temp_buffer.append(seq)

        # Add any sequences we skipped back to the start of the buffer
        self.buffer.extendleft(temp_buffer)

        if np.all(seq_counts > 0):
            self._seqs_emitted += np.sum(seq_counts)
            for x in cum_seq_lens:
                if x[-1] != self.out_pseq_len:
                    x.append(self.out_pseq_len)
            return batch, cum_seq_lens
        else:
            # If we can't form a full batch, we return None to signal the end
            return None


T = TypeVar("T")


class BufferedIterable(Generic[T]):
    def __init__(self, iterable: Iterable[T], buffer_size: int):
        """
        Args:
          - iterable: an object which generates a fresh iterator on iter() and which implements len()
        """
        self.iterable = iterable
        self.buffer_size = buffer_size

    def __iter__(self):
        return BufferedIterator(self.iterable, self.buffer_size)


class BufferedIterator(Generic[T]):
    def __init__(self, iterable: Iterable[T], buffer_size: int):
        self.iterator = iter(iterable)
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.lock = threading.Lock()
        self.exhausted = False
        self.filler_thread = threading.Thread(target=self._background_fill, daemon=True)
        self.filler_thread.start()

    def _background_fill(self):
        # Fill up the buffer, whenever possible, in the background
        while not self.exhausted:
            if len(self.buffer) < self.buffer_size:
                try:
                    item = next(self.iterator)
                    with self.lock:
                        self.buffer.append(item)
                except StopIteration:
                    self.exhausted = True
                    break
            else:
                time.sleep(0.01)  # Sleep for a bit to avoid busy waiting

    def __iter__(self):
        return self

    def __next__(self) -> T:
        while True:
            if not self.buffer:
                if self.exhausted:
                    # We've exhausted the iterator and the buffer so we're done
                    raise StopIteration
                else:
                    # The buffer is empty but the iterator is not exhausted yet.
                    # Let's give the filler thread a chance to add items to the buffer
                    time.sleep(0.01)
            else:
                with self.lock:
                    return self.buffer.popleft()


def split_packed_batch(batch: Any, microbatch_size: Union[int, float], padding_tolerance=1.0) -> Sequence:
    # NOTE: Packed sequences are already packed into a microbatch size worth of tokens.
    # So to correctly return a microbatch worth of data, we will simply return each item (i.e. microbatch_size 1)

    num_items = batch["input_ids"].shape[0]
    split_inputs = [x.squeeze() for x in batch["input_ids"].split(1)]
    split_labels = [x.squeeze() for x in batch["labels"].split(1)]
    split_attention_masks = [x.squeeze() for x in batch["attention_mask"].split(1)]
    split_cu_seqlens = batch["cu_seqlens"]

    result = []
    for i in range(num_items):
        attention_mask = split_attention_masks[i]
        padding_amount = 1 - (attention_mask.sum() / len(attention_mask))

        if padding_amount > padding_tolerance:
            last_non_pad = attention_mask.nonzero().max()
            input_ids = split_inputs[i][: last_non_pad + 1]
            labels = split_labels[i][: last_non_pad + 1]
            cu_seqlens = split_cu_seqlens[i][:-1]
            attention_mask = attention_mask[: last_non_pad + 1]
        else:
            input_ids = split_inputs[i]
            labels = split_labels[i]
            cu_seqlens = split_cu_seqlens[i]

        result.append(
            {
                "input_ids": input_ids,
                "labels": labels,
                "cu_seqlens": cu_seqlens,
                "max_seqlen": batch["max_seqlen"][i],
                "attention_mask": attention_mask,
            }
        )

    assert all([x["input_ids"].shape[-1] == y["cu_seqlens"][-1] for x, y in zip(result, result)])
    return result


def get_num_samples_in_packed_batch(batch: Batch) -> int:
    # Number of sequences can be inferred from cu_seqlens arrays
    cu_seqlens = batch["cu_seqlens"]
    if isinstance(cu_seqlens, torch.Tensor):
        return cu_seqlens.size()[0] - 1
    elif isinstance(cu_seqlens, list):
        return sum([x.size()[0] - 1 for x in batch["cu_seqlens"]])
    else:
        raise TypeError('Expected a batch with a "cu_seqlens" key of type list or Tensor')
