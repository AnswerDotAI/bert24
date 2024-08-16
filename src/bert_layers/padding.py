import torch
from torch import Tensor
from typing import Optional, Tuple
import torch.nn.functional as F


def unpad_input(
    inputs: Tensor,
    attention_mask: Tensor,
    position_ids: Optional[Tensor] = None,
    labels: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, int, Optional[Tensor], Optional[Tensor]]:
    """
    Remove padding from input sequences.

    Args:
        inputs: (batch, seqlen, ...) or (batch, seqlen)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
        position_ids: (batch, seqlen), int, position ids
        labels: (batch, seqlen), int, labels

    Returns:
        unpadded_inputs: (total_nnz, ...), where total_nnz = number of tokens selected in attention_mask.
        indices: (total_nnz)
        cu_seqlens: (batch + 1), the cumulative sequence lengths
        max_seqlen_in_batch: int
        unpadded_position_ids: (total_nnz) or None
        unpadded_labels: (total_nnz) or None
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = int(seqlens_in_batch.max().item())
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))

    if inputs.dim() == 2:
        unpadded_inputs = inputs.flatten()[indices]
    else:
        batch, seqlen, *rest = inputs.shape
        shape = batch * seqlen
        unpadded_inputs = inputs.view(shape, *rest)[indices]

    unpadded_position_ids = position_ids.flatten()[indices] if position_ids is not None else None
    unpadded_labels = labels.flatten()[indices] if labels is not None else None

    return unpadded_inputs, indices, cu_seqlens, max_seqlen_in_batch, unpadded_position_ids, unpadded_labels


def pad_input(
    inputs: Tensor,
    indices: Tensor,
    batch: int,
    seqlen: int,
    labels: Optional[Tensor] = None,
    ignore_index: int = -100,
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Add padding to sequences.

    Args:
        inputs: (total_nnz, ...) or (total_nnz,), where total_nnz = number of tokens selected in attention_mask.
        indices: (total_nnz)
        batch: int, batch size
        seqlen: int, max sequence length
        position_ids: (total_nnz) or None
        labels: (total_nnz) or None

    Returns:
        padded_inputs: (batch, seqlen, ...) or (batch, seqlen)
        padded_labels: (batch, seqlen) or None
    """
    if inputs.dim() == 1:
        output = torch.zeros(batch * seqlen, dtype=inputs.dtype, device=inputs.device)
        output[indices] = inputs
        padded_inputs = output.view(batch, seqlen)
    else:
        _, *rest = inputs.shape
        output = torch.zeros(batch * seqlen, *rest, dtype=inputs.dtype, device=inputs.device)
        output[indices] = inputs
        padded_inputs = output.view(batch, seqlen, *rest)

    padded_labels = None
    if labels is not None:
        padded_labels = torch.full((batch * seqlen,), fill_value=ignore_index, dtype=labels.dtype, device=labels.device)
        padded_labels[indices] = labels
        padded_labels = padded_labels.view(batch, seqlen)

    return padded_inputs, padded_labels
