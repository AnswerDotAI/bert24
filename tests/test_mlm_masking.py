# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

import sys
import os
import pytest
import numpy as np

# Add tests folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Add folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sequence_packer import SequencePacker


import pytest
import numpy as np


@pytest.mark.parametrize("mask_prob", [0.1, 0.15, 0.3, 0.5])
def test_mlm_masking(mask_prob):
    # Test setup
    seq = np.arange(100_000)  # A sequence of 100,000 tokens
    mask_token = -1
    ignore_index = -100

    # Run the function
    masked_seq, labels = SequencePacker.mlm_masking(
        seq.copy(), mask_prob=mask_prob, mask_token=mask_token, ignore_index=ignore_index
    )

    # Test 1 and 2: Check if the output types and shapes are correct
    if not (isinstance(masked_seq, np.ndarray) and isinstance(labels, np.ndarray)):
        raise ValueError("Output types are not correct. Expected NumPy arrays.")
    if not (masked_seq.shape == labels.shape == seq.shape):
        raise ValueError("Output shapes are not correct.")

    # Test 3: Check 80-10-10 rule
    masked_indices = labels != ignore_index
    total_masked = np.sum(masked_indices)

    if total_masked > 0:
        replaced_by_mask = np.sum((masked_seq == mask_token) & masked_indices)
        replaced_by_random = np.sum((masked_seq != mask_token) & (masked_seq != seq) & masked_indices)
        kept_unchanged = np.sum((masked_seq == seq) & masked_indices)

        mask_ratio = replaced_by_mask / total_masked
        random_ratio = replaced_by_random / total_masked
        unchanged_ratio = kept_unchanged / total_masked

        if not 0.79 < mask_ratio < 0.81:
            raise ValueError(f"Mask token ratio ({mask_ratio:.4f}) is out of expected range [0.79, 0.81]")
        if not 0.09 < random_ratio < 0.11:
            raise ValueError(f"Random token ratio ({random_ratio:.4f}) is out of expected range [0.09, 0.11]")
        if not 0.09 < unchanged_ratio < 0.11:
            raise ValueError(f"Unchanged token ratio ({unchanged_ratio:.4f}) is out of expected range [0.09, 0.11]")

    # Test 4: Check overall masking probability
    actual_mask_prob = np.mean(masked_indices)
    if not mask_prob - 0.01 < actual_mask_prob < mask_prob + 0.01:
        raise ValueError(
            f"Actual masking probability ({actual_mask_prob:.4f}) is too far from requested probability ({mask_prob})"
        )