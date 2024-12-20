# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import warnings

from transformers import BertConfig as TransformersBertConfig


class BertConfig(TransformersBertConfig):
    def __init__(
        self,
        alibi_starting_size: int = 512,
        normalization: str = "layernorm",
        attention_probs_dropout_prob: float = 0.0,
        head_pred_act: str = "gelu",
        deterministic_fa2: bool = False,
        allow_embedding_resizing: bool = False,
        **kwargs,
    ):
        """Configuration class for MosaicBert.

        Args:
            alibi_starting_size (int): Use `alibi_starting_size` to determine how large of an alibi tensor to
                create when initializing the model. You should be able to ignore this parameter in most cases.
                Defaults to 512.
            attention_probs_dropout_prob (float): By default, turn off attention dropout in MosaicBERT
                Note that the custom Triton Flash Attention with ALiBi implementation does not support dropout.
                However, Flash Attention 2 supports ALiBi and dropout https://github.com/Dao-AILab/flash-attention
            embed_dropout_prob (float): Dropout probability for the embedding layer.
            attn_out_dropout_prob (float): Dropout probability for the attention output layer.
            mlp_dropout_prob (float): Dropout probability for the MLP layer.
            allow_embedding_resizing (bool): Embeddings will be automatically resized when they are smaller than the tokenizer vocab size.
        """
        super().__init__(attention_probs_dropout_prob=attention_probs_dropout_prob, **kwargs)
        self.alibi_starting_size = alibi_starting_size
        self.normalization = normalization
        self.head_pred_act = head_pred_act
        self.deterministic_fa2 = deterministic_fa2
        self.allow_embedding_resizing = allow_embedding_resizing


class FlexBertConfig(TransformersBertConfig):
    def __init__(
        self,
        attention_layer: str = "base",
        attention_probs_dropout_prob: float = 0.0,
        attn_out_bias: bool = False,
        attn_out_dropout_prob: float = 0.0,
        attn_qkv_bias: bool = False,
        bert_layer: str = "prenorm",
        decoder_bias: bool = True,
        embed_dropout_prob: float = 0.0,
        embed_norm: bool = True,
        final_norm: bool = False,
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
        pooling_type: str = "cls",
        rotary_emb_dim: int | None = None,
        rotary_emb_base: float = 10000.0,
        rotary_emb_scale_base=None,
        rotary_emb_interleaved: bool = False,
        use_fa2: bool = True,
        use_sdpa_attn_mask: bool = False,
        allow_embedding_resizing: bool = False,
        init_method: str = "default",
        init_std: float = 0.02,
        init_cutoff_factor: float = 2.0,
        init_small_embedding: bool = False,
        initial_attention_layer: str | None = None,
        initial_bert_layer: str | None = None,
        initial_mlp_layer: str | None = None,
        num_initial_layers: int = 1,
        skip_first_prenorm: bool = False,
        deterministic_fa2: bool = False,
        sliding_window: int = -1,
        global_attn_every_n_layers: int = -1,
        local_attn_rotary_emb_base: float = -1,
        local_attn_rotary_emb_dim: int | None = None,
        unpad_embeddings: bool = False,
        pad_logits: bool = False,
        compile_model: bool = False,
        masked_prediction: bool = False,
        **kwargs,
    ):
        """
        Args:
            attention_layer (str): Attention layer type.
            attention_probs_dropout_prob (float): Dropout probability for attention probabilities.
            attn_out_bias (bool): use bias in attention output projection.
            attn_out_dropout_prob (float): Dropout probability for attention output.
            attn_qkv_bias (bool): use bias for query, key, value linear layer(s).
            bert_layer (str): BERT layer type.
            decoder_bias (bool): use bias in decoder linear layer.
            embed_dropout_prob (float): Dropout probability for embeddings.
            embed_norm (bool): Normalize embedding output.
            final_norm (bool): Add normalization after the final encoder layer and before head.
            embedding_layer (str): Embedding layer type.
            encoder_layer (str): Encoder layer type.
            loss_function (str): Loss function to use.
            loss_kwargs (dict): Keyword arguments for loss function.
            mlp_dropout_prob (float): Dropout probability for MLP layers.
            mlp_in_bias (bool): Use bias in MLP input linear layer.
            mlp_layer (str): MLP layer type.
            mlp_out_bias (bool): Use bias in MLP output linear layer.
            norm_kwargs (dict): Keyword arguments for normalization layers.
            normalization (str): Normalization type.
            padding (str): Unpad inputs. Best with `use_fa2=True`.
            head_class_act (str): Activation function for classification head.
            head_class_bias (bool): Use bias in classification head linear layer(s).
            head_class_dropout (float): Dropout probability for classification head.
            head_class_norm (str): Normalization type for classification head.
            head_pred_act (str): Activation function for prediction head.
            head_pred_bias (bool): Use bias in prediction head linear layer(s).
            head_pred_dropout (float): Dropout probability for prediction head.
            head_pred_norm (bool): Normalize prediction head output.
            pooling_type (str): Pooling type.
            rotary_emb_dim (int | None): Rotary embedding dimension.
            rotary_emb_base (float): Rotary embedding base.
            rotary_emb_scale_base (float): Rotary embedding scale base.
            rotary_emb_interleaved (bool): Use interleaved rotary embeddings.
            use_fa2 (bool): Use FlashAttention2. Requires flash_attn package.
            use_sdpa_attn_mask (bool): Pass a mask to SDPA. This will prevent SDPA from using the PyTorch FA2 kernel.
            allow_embedding_resizing (bool): Embeddings will be automatically resized when they are smaller than the tokenizer vocab size.
            init_method (str): Model layers initialization method.
            init_std (float): Standard deviation for initialization. Used for normal and full_megatron init.
            init_cutoff_factor (float): Cutoff factor for initialization. Used for normal and full_megatron init.
            init_small_embedding (bool): Initialize embeddings with RWKV small init.
            initial_attention_layer (str | None): Replace first `num_initial_layers` attention_layer instance with this layer.
            initial_bert_layer (str | None): Replace first `num_initial_layers` bert_layer instance with this layer.
            initial_mlp_layer (str | None): Replace first `num_initial_layers` mlp_layer instance with this layer.
            num_initial_layers (int): Number of initial layers to set via `initial_attention_layer`, `initial_bert_layer`, and `initial_mlp_layer`.
            skip_first_prenorm (bool): Skip pre-normalization for the first bert layer. Requires `embed_norm=True`.
            deterministic_fa2 (bool): Use Flash Attention 2 deterministic mode. This is slower then the default non-deterministic mode.
            sliding_window (int): Use sliding window attention with window size `n`. -1 to disable. Window size split between the left and right context. Only supports FA2.
            global_attn_every_n_layers (int): Use global attention every `n` layers and sliding window for the rest. -1 to disable.
            local_attn_rotary_emb_base (float): Rotary embedding base for local attention. -1 to disable and use `rotary_emb_base` for all layers.
            local_attn_rotary_emb_dim (int | None): Rotary embedding dimension for local attention. None to disable and use `rotary_emb_dim` for all layers.
            unpad_embeddings (bool): Unpad inputs before the embedding layer.
            pad_logits (bool): Pad logits after the calculating the loss.
            compile_model (bool): Compile the subset of the model which can be compiled.
            masked_prediction (bool): Use only pass the masked tokens through the final MLM layers
            **kwargs: Additional keyword arguments.
        """
        super().__init__(attention_probs_dropout_prob=attention_probs_dropout_prob, **kwargs)
        self.attention_layer = attention_layer
        self.attn_out_bias = attn_out_bias
        self.attn_out_dropout_prob = attn_out_dropout_prob
        self.attn_qkv_bias = attn_qkv_bias
        self.bert_layer = bert_layer
        self.decoder_bias = decoder_bias
        self.embed_dropout_prob = embed_dropout_prob
        self.embed_norm = embed_norm
        self.final_norm = final_norm
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
        self.use_fa2 = use_fa2
        self.use_sdpa_attn_mask = use_sdpa_attn_mask
        self.allow_embedding_resizing = allow_embedding_resizing
        self.init_method = init_method
        self.init_std = init_std
        self.init_cutoff_factor = init_cutoff_factor
        self.init_small_embedding = init_small_embedding
        self.initial_attention_layer = initial_attention_layer
        self.initial_bert_layer = initial_bert_layer
        self.initial_mlp_layer = initial_mlp_layer
        self.num_initial_layers = num_initial_layers
        self.skip_first_prenorm = skip_first_prenorm
        self.deterministic_fa2 = deterministic_fa2
        self.sliding_window = sliding_window
        self.global_attn_every_n_layers = global_attn_every_n_layers
        self.local_attn_rotary_emb_base = local_attn_rotary_emb_base
        self.local_attn_rotary_emb_dim = local_attn_rotary_emb_dim
        self.unpad_embeddings = unpad_embeddings
        self.pad_logits = pad_logits
        self.compile_model = compile_model
        self.masked_prediction = masked_prediction

        if loss_kwargs.get("return_z_loss", False):
            if loss_function != "fa_cross_entropy":
                raise ValueError("loss_function must be 'fa_cross_entropy' when return_z_loss is True")
            if loss_kwargs.get("lse_square_scale", 0) <= 0:
                raise ValueError(
                    "lse_square_scale must be passed to `loss_kwargs` and must be greater than 0 for z_loss"
                )
        if loss_kwargs.get("inplace_backward", False):
            self.loss_kwargs["inplace_backward"] = False
            warnings.warn("`inplace_backward=True` will cause incorrect metrics. Automatically setting to False.")

        if global_attn_every_n_layers > 0 and (self.num_hidden_layers - 1) % global_attn_every_n_layers != 0:
            raise ValueError(
                f"{global_attn_every_n_layers=} must be a divisor of one less than {self.num_hidden_layers=}"
            )

        if self.sliding_window != -1:
            if not self.use_fa2:
                raise ValueError("Sliding window attention is only supported with FlashAttention2")
            if self.sliding_window % 2 != 0 and self.sliding_window % 64 != 0:
                raise ValueError(
                    f"Sliding window must be an even number and divisible by 64: {self.sliding_window=} {self.sliding_window % 64} {self.sliding_window % 2}"
                )
        else:
            if self.global_attn_every_n_layers != -1:
                raise ValueError("global_attn_every_n_layers must be -1 when sliding_window is disabled")
            if self.local_attn_rotary_emb_base != -1:
                raise ValueError("local_attn_rotary_emb_base must be -1 when sliding_window is disabled")
            if self.local_attn_rotary_emb_dim is not None:
                raise ValueError("local_attn_rotary_emb_dim must be None when sliding_window is disabled")

        if self.unpad_embeddings and self.padding != "unpadded":
            warnings.warn(
                "`unpad_embeddings=True` requires `padding='unpadded'`. Automatically setting `padding='unpadded'`."
            )
            self.padding = "unpadded"
        if self.pad_logits and not self.unpad_embeddings:
            raise ValueError("`pad_logits=True` requires `unpad_embeddings=True`")
        if self.unpad_embeddings and self.embedding_layer == "absolute_pos":
            raise ValueError(f"{self.unpad_embeddings=} is incompatible with {self.embedding_layer=}")


PADDING = ["unpadded", "padded"]


def maybe_add_padding(config: FlexBertConfig, config_option: str) -> str:
    if config.padding not in PADDING:
        raise ValueError(f"Invalid padding type: {config.padding}, must be one of {PADDING}")

    if not any(config_option.startswith(pad + "_") for pad in PADDING):
        config_option = f"{config.padding}_{config_option}"

    return config_option
