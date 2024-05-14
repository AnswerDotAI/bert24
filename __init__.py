# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import os
import sys

# Add folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

try:
    import torch
    # yapf: disable
    from src.bert_layers import (BertEmbeddings, BertEncoder, BertForMaskedLM,
                                 BertForSequenceClassification,
                                 BertGatedLinearUnitMLP, BertLayer,
                                 BertLMPredictionHead, BertModel,
                                 BertOnlyMLMHead, BertOnlyNSPHead, BertPooler,
                                 BertPredictionHeadTransform, BertSelfOutput,
                                 BertUnpadAttention, BertUnpadSelfAttention)
    # yapf: enable
    from src.bert_padding import (IndexFirstAxis, IndexPutFirstAxis,
                                  index_first_axis, index_put_first_axis,
                                  pad_input, unpad_input, unpad_input_only)
    if torch.cuda.is_available():
        from src.flash_attn_triton import \
            flash_attn_func as flash_attn_func_bert # type: ignore
        from src.flash_attn_triton import \
            flash_attn_qkvpacked_func as flash_attn_qkvpacked_func_bert # type: ignore
    from src.hf_bert import create_hf_bert_classification, create_hf_bert_mlm
    from src.mosaic_bert import (create_mosaic_bert_classification,
                                 create_mosaic_bert_mlm)
except ImportError as e:
    try:
        is_cuda_available = torch.cuda.is_available()  # type: ignore
    except:
        is_cuda_available = False

    reqs_file = 'requirements.txt' if is_cuda_available else 'requirements-cpu.txt'
    raise ImportError(
        f'Please make sure to pip install -r {reqs_file} to get the requirements for the BERT benchmark.'
    ) from e

__all__ = [
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
