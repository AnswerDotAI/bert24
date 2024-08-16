import torch
import pytest
import sys
import os

# Add tests folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Add folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.bert_layers.padding import unpad_input, pad_input


@pytest.fixture
def sample_data():
    batch, seqlen, hidden_dim = 2, 4, 3
    inputs = torch.randn(batch, seqlen, hidden_dim)
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.int32)
    position_ids = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long)
    labels = torch.tensor([[1, 2, 3, -100], [4, 5, -100, -100]], dtype=torch.long)
    return inputs, attention_mask, position_ids, labels


def test_unpad_input(sample_data):
    inputs, attention_mask, position_ids, labels = sample_data
    unpadded_inputs, indices, cu_seqlens, max_seqlen, unpadded_position_ids, unpadded_labels = unpad_input(
        inputs, attention_mask, position_ids, labels
    )

    assert unpadded_inputs.shape == (5, 3)  # 5 valid tokens, hidden_dim = 3
    assert indices.tolist() == [0, 1, 2, 4, 5]
    assert cu_seqlens.tolist() == [0, 3, 5]
    assert max_seqlen == 3
    assert unpadded_position_ids.tolist() == [0, 1, 2, 0, 1]
    assert unpadded_labels.tolist() == [1, 2, 3, 4, 5]


def test_pad_input(sample_data):
    inputs, attention_mask, _, labels = sample_data
    unpadded_inputs, indices, _, _, _, unpadded_labels = unpad_input(inputs, attention_mask, labels=labels)

    padded_inputs, padded_labels = pad_input(unpadded_inputs, indices, batch=2, seqlen=4, labels=unpadded_labels)

    assert padded_inputs.shape == (2, 4, 3)
    assert torch.allclose(padded_inputs[attention_mask.bool()], unpadded_inputs)
    assert torch.all(padded_inputs[~attention_mask.bool()] == 0)
    assert torch.all(padded_labels[attention_mask.bool()] == unpadded_labels)
    assert torch.all(padded_labels[~attention_mask.bool()] == -100)


def test_roundtrip(sample_data):
    inputs, attention_mask, _, labels = sample_data
    unpadded_inputs, indices, _, _, _, unpadded_labels = unpad_input(inputs, attention_mask, labels=labels)
    padded_inputs, padded_labels = pad_input(unpadded_inputs, indices, batch=2, seqlen=4, labels=unpadded_labels)

    assert torch.allclose(inputs[attention_mask.bool()], padded_inputs[attention_mask.bool()])
    assert torch.all(labels == padded_labels)


def test_token_input():
    batch, seqlen, vocab_size = 2, 4, 1000
    token_ids = torch.randint(0, vocab_size, (batch, seqlen))
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.int32)

    unpadded_inputs, indices, _, _, _, _ = unpad_input(token_ids, attention_mask)

    assert unpadded_inputs.shape == (5,)  # 5 valid tokens
    assert unpadded_inputs.dtype == torch.long

    padded_inputs, _ = pad_input(unpadded_inputs, indices, batch=2, seqlen=4)

    assert padded_inputs.shape == (2, 4)
    assert padded_inputs.dtype == torch.long
    assert torch.all(padded_inputs[attention_mask.bool()] == unpadded_inputs)
    assert torch.all(padded_inputs[~attention_mask.bool()] == 0)


def test_2d_input():
    batch, seqlen = 2, 4
    inputs = torch.randn(batch, seqlen)
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.int32)

    unpadded_inputs, indices, cu_seqlens, max_seqlen, _, _ = unpad_input(inputs, attention_mask)

    assert unpadded_inputs.shape == (5,)  # 5 valid tokens
    assert indices.tolist() == [0, 1, 2, 4, 5]
    assert cu_seqlens.tolist() == [0, 3, 5]
    assert max_seqlen == 3

    padded_inputs, _ = pad_input(unpadded_inputs, indices, batch=2, seqlen=4)

    assert padded_inputs.shape == (2, 4)
    assert torch.allclose(padded_inputs[attention_mask.bool()], unpadded_inputs)
    assert torch.all(padded_inputs[~attention_mask.bool()] == 0)
