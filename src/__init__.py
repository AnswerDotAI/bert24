# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import torch

# Add src folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

# yapf: disable
from bert_layers import (BertEmbeddings, BertEncoder, BertForMaskedLM,
                         BertForSequenceClassification, BertGatedLinearUnitMLP,
                         BertLayer, BertLMPredictionHead, BertModel,
                         BertOnlyMLMHead, BertOnlyNSPHead, BertPooler,
                         BertPredictionHeadTransform, BertSelfOutput,
                         BertUnpadAttention, BertUnpadSelfAttention)
# yapf: enable
from bert_padding import (IndexFirstAxis, IndexPutFirstAxis, index_first_axis,
                          index_put_first_axis, pad_input, unpad_input,
                          unpad_input_only)
from configuration_bert import BertConfig

if torch.cuda.is_available():
    from flash_attn_triton import \
        flash_attn_func as flash_attn_func_bert # type: ignore
    from flash_attn_triton import \
        flash_attn_qkvpacked_func as flash_attn_qkvpacked_func_bert # type: ignore

from hf_bert import create_hf_bert_classification, create_hf_bert_mlm
from mosaic_bert import (create_mosaic_bert_classification,
                         create_mosaic_bert_mlm)

__all__ = [
    'BertConfig',
    'BertEmbeddings',
    'BertEncoder',
    'BertForMaskedLM',
    'BertForSequenceClassification',
    'BertGatedLinearUnitMLP',
    'BertLayer',
    'BertLMPredictionHead',
    'BertModel',
    'BertOnlyMLMHead',
    'BertOnlyNSPHead',
    'BertPooler',
    'BertPredictionHeadTransform',
    'BertSelfOutput',
    'BertUnpadAttention',
    'BertUnpadSelfAttention',
    'IndexFirstAxis',
    'IndexPutFirstAxis',
    'index_first_axis',
    'index_put_first_axis',
    'pad_input',
    'unpad_input',
    'unpad_input_only',
    'create_hf_bert_classification',
    'create_hf_bert_mlm',
    'create_mosaic_bert_classification',
    'create_mosaic_bert_mlm',

    # These are commented out because they only exist if CUDA is available
    # 'flash_attn_func_bert',
    # 'flash_attn_qkvpacked_func_bert'
]
