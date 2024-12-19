# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

import os
import sys

import pytest
import torch
import torch.nn as nn

# Add tests folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Add folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bert_layers.initialization import tile_weight, tile_linear, TileMode


tiling_test_data = [
    # Test 1: 2x2 to 4x4
    ((4, 4),
     [[1, 2],
      [3, 4]],
     [[4, 3, 4, 3],
      [2, 1, 2, 1],
      [4, 3, 4, 3],
      [2, 1, 2, 1]],
     [[1, 2, 1, 2],
      [3, 4, 3, 4],
      [1, 2, 1, 2],
      [3, 4, 3, 4]],
     [[0, 0, 0, 0],
      [0, 1, 2, 0],
      [0, 3, 4, 0],
      [0, 0, 0, 0]]),

    # Test 2: 3x3 to 5x5
    ((5, 5),
     [[1, 2, 3],
      [4, 5, 6],
      [7, 8, 9]],
     [[9, 7, 8, 9, 7],
      [3, 1, 2, 3, 1],
      [6, 4, 5, 6, 4],
      [9, 7, 8, 9, 7],
      [3, 1, 2, 3, 1]],
     [[1, 2, 3, 1, 2],
      [4, 5, 6, 4, 5],
      [7, 8, 9, 7, 8],
      [1, 2, 3, 1, 2],
      [4, 5, 6, 4, 5]],
     [[0, 0, 0, 0, 0],
      [0, 1, 2, 3, 0],
      [0, 4, 5, 6, 0],
      [0, 7, 8, 9, 0],
      [0, 0, 0, 0, 0]]),

    # Test 3: 2x3 to 6x7
    ((6, 7),
     [[1, 2, 3],
      [4, 5, 6]],
     [[2, 3, 1, 2, 3, 1, 2],
      [5, 6, 4, 5, 6, 4, 5],
      [2, 3, 1, 2, 3, 1, 2],
      [5, 6, 4, 5, 6, 4, 5],
      [2, 3, 1, 2, 3, 1, 2],
      [5, 6, 4, 5, 6, 4, 5]],
     [[1, 2, 3, 1, 2, 3, 1],
      [4, 5, 6, 4, 5, 6, 4],
      [1, 2, 3, 1, 2, 3, 1],
      [4, 5, 6, 4, 5, 6, 4],
      [1, 2, 3, 1, 2, 3, 1],
      [4, 5, 6, 4, 5, 6, 4]],
     [[0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0],
      [0, 0, 1, 2, 3, 0, 0],
      [0, 0, 4, 5, 6, 0, 0],
      [0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0]]),

    # Test 4: 4x2 to 7x5
    ((7, 5),
     [[1, 2],
      [3, 4],
      [5, 6],
      [7, 8]],
     [[8, 7, 8, 7, 8],
      [2, 1, 2, 1, 2],
      [4, 3, 4, 3, 4],
      [6, 5, 6, 5, 6],
      [8, 7, 8, 7, 8],
      [2, 1, 2, 1, 2],
      [4, 3, 4, 3, 4]],
     [[1, 2, 1, 2, 1],
      [3, 4, 3, 4, 3],
      [5, 6, 5, 6, 5],
      [7, 8, 7, 8, 7],
      [1, 2, 1, 2, 1],
      [3, 4, 3, 4, 3],
      [5, 6, 5, 6, 5]],
     [[0, 0, 0, 0, 0],
      [0, 1, 2, 0, 0],
      [0, 3, 4, 0, 0],
      [0, 5, 6, 0, 0],
      [0, 7, 8, 0, 0],
      [0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0]]),

    # Test 5: 1x3 to 4x8
    ((4, 8),
     [[1, 2, 3]],
     [[2, 3, 1, 2, 3, 1, 2, 3],
      [2, 3, 1, 2, 3, 1, 2, 3],
      [2, 3, 1, 2, 3, 1, 2, 3],
      [2, 3, 1, 2, 3, 1, 2, 3]],
     [[1, 2, 3, 1, 2, 3, 1, 2],
      [1, 2, 3, 1, 2, 3, 1, 2],
      [1, 2, 3, 1, 2, 3, 1, 2],
      [1, 2, 3, 1, 2, 3, 1, 2]],
     [[0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 1, 2, 3, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0]]),

    # Test 6: 1D tensor - 3 to 8
    ((8,),
     [1, 2, 3],
     [2, 3, 1, 2, 3, 1, 2, 3],
     [1, 2, 3, 1, 2, 3, 1, 2],
     [0, 0, 1, 2, 3, 0, 0, 0])
]  # fmt: skip


@pytest.mark.parametrize("new_size, input_tensor, expected_middle, expected_edge, expected_center", tiling_test_data)
def test_tile_weight(new_size, input_tensor, expected_middle, expected_edge, expected_center):
    input_tensor = torch.tensor(input_tensor)
    expected_middle = torch.tensor(expected_middle)
    expected_edge = torch.tensor(expected_edge)
    expected_center = torch.tensor(expected_center)
    output_tensor = torch.zeros(new_size)

    # Test tiling from middle
    result_middle = tile_weight(input_tensor, new_weights=output_tensor, mode=TileMode.tile_weights_from_middle)
    assert torch.all(
        result_middle.eq(expected_middle)
    ), f"Middle tiling failed for input {input_tensor.tolist()} to size {new_size}"

    # Test tiling from edge
    result_edge = tile_weight(input_tensor, new_weights=output_tensor, mode=TileMode.tile_weights_from_edge)
    assert torch.all(
        result_edge.eq(expected_edge)
    ), f"Edge tiling failed for input {input_tensor.tolist()} to size {new_size}"

    # Test center only
    result_center = tile_weight(input_tensor, new_weights=output_tensor, mode=TileMode.center_weights)
    assert torch.all(
        result_center.eq(expected_center)
    ), f"Center only failed for input {input_tensor.tolist()} to size {new_size}"


@pytest.mark.parametrize("new_size, input_tensor, expected_middle, expected_edge, expected_center", tiling_test_data)
def test_tile_linear(new_size, input_tensor, expected_middle, expected_edge, expected_center):
    input_tensor = torch.tensor(input_tensor, dtype=torch.float)
    expected_middle = torch.tensor(expected_middle, dtype=torch.float)
    expected_edge = torch.tensor(expected_edge, dtype=torch.float)
    expected_center = torch.tensor(expected_center, dtype=torch.float)

    if input_tensor.dim() == 1:
        # Handle 1D tensor as bias
        old_linear = nn.Linear(expected_middle.size(0), expected_middle.size(0), bias=True)
        new_linear = nn.Linear(expected_middle.size(0), expected_middle.size(0), bias=True)
        old_linear.bias = nn.Parameter(input_tensor)
    else:
        # Handle 2D tensor as weight
        old_linear = nn.Linear(input_tensor.size(1), input_tensor.size(0), bias=False)
        new_linear = nn.Linear(expected_middle.size(1), expected_middle.size(0), bias=False)
        old_linear.weight = nn.Parameter(input_tensor)

    # Test tiling from middle
    if input_tensor.dim() == 1:
        new_linear.bias = nn.Parameter(torch.zeros_like(expected_middle))
        tile_linear(old_linear, new_linear, linear_type="default", mode=TileMode.tile_weights_from_middle)
    else:
        new_linear.weight = nn.Parameter(torch.zeros_like(expected_middle))
        tile_linear(old_linear, new_linear, linear_type="default", mode=TileMode.tile_weights_from_middle)
    assert torch.allclose(
        new_linear.weight.data if input_tensor.dim() > 1 else new_linear.bias.data, expected_middle
    ), "Middle tiling failed for Linear layer"

    # Test tiling from edge
    if input_tensor.dim() == 1:
        new_linear.bias = nn.Parameter(torch.zeros_like(expected_edge))
        tile_linear(old_linear, new_linear, linear_type="default", mode=TileMode.tile_weights_from_edge)
    else:
        new_linear.weight = nn.Parameter(torch.zeros_like(expected_edge))
        tile_linear(old_linear, new_linear, linear_type="default", mode=TileMode.tile_weights_from_edge)
    assert torch.allclose(
        new_linear.weight.data if input_tensor.dim() > 1 else new_linear.bias.data, expected_edge
    ), "Edge tiling failed for Linear layer"

    # Test center only
    if input_tensor.dim() == 1:
        new_linear.bias = nn.Parameter(torch.zeros_like(expected_center))
        tile_linear(old_linear, new_linear, linear_type="default", mode=TileMode.center_weights)
    else:
        new_linear.weight = nn.Parameter(torch.zeros_like(expected_center))
        tile_linear(old_linear, new_linear, linear_type="default", mode=TileMode.center_weights)
    assert torch.allclose(
        new_linear.weight.data if input_tensor.dim() > 1 else new_linear.bias.data, expected_center
    ), "Center only failed for Linear layer"
