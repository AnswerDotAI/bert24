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
        decoder_bias: bool = True,
        embed_dropout_prob: float = 0.0,
        embed_norm: bool = True,
        embedding_layer: str = "absolute_pos",
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
        head_class_act: str = "silu",
        head_class_bias: bool = False,
        head_class_dropout: float = 0.0,
        head_class_norm: str = False,
        head_pred_act: str = "silu",
        head_pred_bias: bool = False,
        head_pred_dropout: float = 0.0,
        head_pred_norm: bool = True,
        pooling_type: str = "mean",
        rotary_emb_dim: int = 64,
        rotary_emb_base: float = 10000.0,
        rotary_emb_scale_base=None,
        rotary_emb_interleaved: bool = False,
        **kwargs,
    ):
        super().__init__(attention_probs_dropout_prob=attention_probs_dropout_prob, **kwargs)
        self.activation_function = activation_function
        self.attention_layer = attention_layer
        self.attn_out_bias = attn_out_bias
        self.attn_out_dropout_prob = attn_out_dropout_prob
        self.attn_qkv_bias = attn_qkv_bias
        self.bert_layer = bert_layer
        self.decoder_bias = decoder_bias
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
        self.head_class_act = head_class_act
        self.head_class_bias = head_class_bias
        self.head_class_dropout = head_class_dropout
        self.head_class_norm = head_class_norm
        self.head_pred_act = head_pred_act
        self.head_pred_bias = head_pred_bias
        self.head_pred_dropout = head_pred_dropout
        self.head_pred_norm = head_pred_norm
        self.pooling_type = pooling_type
        self.rotary_emb_dim = rotary_emb_dim
        self.rotary_emb_base = rotary_emb_base
        self.rotary_emb_scale_base = rotary_emb_scale_base
        self.rotary_emb_interleaved = rotary_emb_interleaved


PADDING = ["unpadded", "padded"]


def maybe_add_padding(config: FlexBertConfig, config_option: str) -> str:
    if config.padding not in PADDING:
        raise ValueError(f"Invalid padding type: {config.padding}, must be one of {PADDING}")

    if not any(config_option.startswith(pad + "_") for pad in PADDING):
        config_option = f"{config.padding}_{config_option}"

    return config_option
