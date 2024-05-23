# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

from transformers import BertConfig as TransformersBertConfig


class BertConfig(TransformersBertConfig):
    def __init__(
        self,
        alibi_starting_size: int = 512,
        normalization: str = "layernorm",
        attention_probs_dropout_prob: float = 0.0,
        **kwargs,
    ):
        """Configuration class for MosaicBert.

        Args:
            alibi_starting_size (int): Use `alibi_starting_size` to determine how large of an alibi tensor to
                create when initializing the model. You should be able to ignore this parameter in most cases.
                Defaults to 512.
            attention_probs_dropout_prob (float): By default, turn off attention dropout in MosaicBERT
                Note that the custom Triton Flash Attention with ALiBi implementation does not support droput.
                However, Flash Attention 2 supports ALiBi and dropout https://github.com/Dao-AILab/flash-attention
            embed_dropout_prob (float): Dropout probability for the embedding layer.
            attn_out_dropout_prob (float): Dropout probability for the attention output layer.
            mlp_dropout_prob (float): Dropout probability for the MLP layer.
        """
        super().__init__(attention_probs_dropout_prob=attention_probs_dropout_prob, **kwargs)
        self.alibi_starting_size = alibi_starting_size
        self.normalization = normalization


class FlexBertConfig(TransformersBertConfig):
    def __init__(
        self,
        activation_function: str = "silu",
        attention_layer: str = "base",
        attention_probs_dropout_prob: float = 0.0,
        attn_out_bias: bool = False,
        attn_out_dropout_prob: float = 0.0,
        attn_qkv_bias: bool = False,
        bert_layer: str = "prenorm",
        embed_dropout_prob: float = 0.0,
        embed_norm: bool = True,
        embedding_layer: str = "absolute",
        encoder_layer: str = "base",
        loss_function: str = "cross_entropy",
        loss_kwargs: dict = {},
        mlp_dropout_prob: float = 0.0,
        mlp_in_bias: bool = False,
        mlp_layer: str = "mlp",
        mlp_out_bias: bool = False,
        norm_kwargs: dict = {},
        normalization: str = "rmsnorm",
        padding: str = "unpadded",
        sparse_prediction: bool = False,
        **kwargs,
    ):
        super().__init__(attention_probs_dropout_prob=attention_probs_dropout_prob, **kwargs)
        self.activation_function = activation_function
        self.attention_layer = attention_layer
        self.attn_out_bias = attn_out_bias
        self.attn_out_dropout_prob = attn_out_dropout_prob
        self.attn_qkv_bias = attn_qkv_bias
        self.bert_layer = bert_layer
        self.embed_dropout_prob = embed_dropout_prob
        self.embed_norm = embed_norm
        self.embedding_layer = embedding_layer
        self.encoder_layer = encoder_layer
        self.loss_function = loss_function
        self.loss_kwargs = loss_kwargs
        self.mlp_dropout_prob = mlp_dropout_prob
        self.mlp_in_bias = mlp_in_bias
        self.mlp_layer = mlp_layer
        self.mlp_out_bias = mlp_out_bias
        self.norm_kwargs = norm_kwargs
        self.normalization = normalization
        self.padding = padding
        self.sparse_prediction = sparse_prediction


PADDING = ["unpadded", "padded"]


def maybe_add_padding(config: FlexBertConfig, config_option: str) -> str:
    if config.padding not in PADDING:
        raise ValueError(f"Invalid padding type: {config.padding}, must be one of {PADDING}")

    if not any(config_option.startswith(pad + "_") for pad in PADDING):
        config_option = f"{config.padding}_{config_option}"

    return config_option
