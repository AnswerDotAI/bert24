# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

import enum
import time
from pathlib import Path
from typing import Annotated, List, Optional

import numpy as np
import pandas as pd
import pynvml
import torch
import typer
import yaml
from rich import print
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from torch.utils.data import DataLoader, TensorDataset
from typer import Option

from src.bert_layers.configuration_bert import FlexBertConfig
from src.bert_layers.model import FlexBertForMaskedLM, FlexBertForSequenceClassification

app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]}, pretty_exceptions_show_locals=False)


class ModelType(str, enum.Enum):
    mlm = "mlm"
    seqcls = "seqcls"


# from maxb2: https://github.com/tiangolo/typer/issues/86#issuecomment-996374166
def conf_callback(ctx: typer.Context, param: typer.CallbackParam, config: Optional[str] = None):
    if config is not None:
        typer.echo(f"Loading config file: {config}\n")
        try:
            with open(config, "r") as f:  # Load config file
                conf = yaml.safe_load(f)
            ctx.default_map = ctx.default_map or {}  # Initialize the default map
            ctx.default_map.update(conf)  # Merge the config dict into default_map
        except Exception as ex:
            raise typer.BadParameter(str(ex))
    return config


def get_model(
    hidden_size: int,
    num_hidden_layers: int,
    intermediate_size: float,
    parallel_attn: bool = True,
    vocab_size: int = 32768,
    model_type: ModelType = ModelType.mlm,
    sliding_window: int = -1,
    global_attn_every_n_layers: int = -1,
    normalization: str = "layernorm",
    compile_model: bool = True,
    masked_prediction: bool = False,
):
    config = FlexBertConfig(
        num_attention_heads=hidden_size // 64,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        attention_layer="rope_parallel" if parallel_attn else "rope",
        attention_probs_dropout_prob=0.0,
        attn_out_bias=False,
        attn_out_dropout_prob=0.0,
        attn_qkv_bias=False,
        bert_layer="parallel_prenorm" if parallel_attn else "prenorm",
        decoder_bias=False,
        embed_dropout_prob=0.0,
        embed_norm=True,
        final_norm=False,
        embedding_layer="sans_pos",
        encoder_layer="base",
        hidden_act="gelu",
        loss_function="fa_cross_entropy",
        loss_kwargs={"reduction": "mean"},
        mlp_dropout_prob=0.0,
        mlp_in_bias=False,
        mlp_layer="parallel_glu" if parallel_attn else "glu",
        mlp_out_bias=False,
        norm_kwargs={"eps": 1e-5},
        normalization=normalization,
        padding="padded",
        head_class_act="silu",
        head_class_bias=False,
        head_class_dropout=0.0,
        head_class_norm=False,
        head_pred_act="gelu",
        head_pred_bias=False,
        head_pred_dropout=0.0,
        head_pred_norm=True,
        pooling_type="cls",
        rotary_emb_dim=None,
        rotary_emb_base=10000.0,
        rotary_emb_scale_base=None,
        rotary_emb_interleaved=False,
        use_fa2=True,
        use_sdpa_attn_mask=False,
        allow_embedding_resizing=False,
        init_method="default",
        init_std=0.02,
        init_cutoff_factor=2.0,
        init_small_embedding=False,
        initial_attention_layer=None,
        initial_bert_layer=None,
        initial_mlp_layer=None,
        num_initial_layers=1,
        skip_first_prenorm=False,
        sliding_window=sliding_window,
        global_attn_every_n_layers=global_attn_every_n_layers,
        unpad_embeddings=True,
        pad_logits=False,
        compile_model=compile_model,
        masked_prediction=masked_prediction,
    )
    if model_type == ModelType.mlm:
        config.tie_word_embeddings = True
        return FlexBertForMaskedLM(config)
    elif model_type == ModelType.seqcls:
        config.num_labels = 5
        return FlexBertForSequenceClassification(config)
    else:
        raise ValueError(f"Invalid model type: {model_type}")


def get_gpu_power(gpu_idx=0):
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)  # Assuming we're using the first GPU
    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
    return power


def benchmark_training(model, dataloader, num_warmup_batches=10, gpu_idx=0):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    device = next(model.parameters()).device

    torch.cuda.reset_peak_memory_stats()

    power_readings = []
    max_allocated_memory = 0
    max_reserved_memory = 0

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
    ) as progress:
        warmup_task = progress.add_task("[yellow]Warmup", total=num_warmup_batches)
        for i, batch in enumerate(dataloader):
            if i >= num_warmup_batches:
                break
            input_ids, attention_mask, labels = [t.to(device) for t in batch]
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            progress.update(warmup_task, advance=1)

        train_task = progress.add_task("[green]Training", total=len(dataloader))
        total_time = 0
        epoch_start_time = time.time()
        for i, batch in enumerate(dataloader):
            input_ids, attention_mask, labels = [t.to(device) for t in batch]
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            progress.update(train_task, advance=1)
            if i % 10 == 0:
                power_readings.append(get_gpu_power(gpu_idx))
                max_allocated_memory = max(max_allocated_memory, torch.cuda.max_memory_allocated())
                max_reserved_memory = max(max_reserved_memory, torch.cuda.max_memory_reserved())
        epoch_end_time = time.time()
        total_time += epoch_end_time - epoch_start_time

    avg_epoch_time = total_time
    avg_power = np.mean(power_readings)
    max_power = np.max(power_readings)
    return avg_epoch_time, avg_power, max_power, max_allocated_memory, max_reserved_memory, loss.item()


def benchmark_inference(model, dataloader, num_warmup_batches=10, gpu_idx=0):
    model.eval()
    device = next(model.parameters()).device

    torch.cuda.reset_peak_memory_stats()

    power_readings = []
    max_allocated_memory = 0
    max_reserved_memory = 0

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
    ) as progress:
        warmup_task = progress.add_task("[yellow]Warmup", total=num_warmup_batches)
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_warmup_batches:
                    break
                input_ids, attention_mask, _ = [t.to(device) for t in batch]
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    _ = model(input_ids, attention_mask=attention_mask)
                progress.update(warmup_task, advance=1)

        inference_task = progress.add_task("[cyan]Inference", total=len(dataloader))
        total_time = 0
        with torch.no_grad():
            run_start_time = time.time()
            for i, batch in enumerate(dataloader):
                input_ids, attention_mask, _ = [t.to(device) for t in batch]
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    _ = model(input_ids, attention_mask=attention_mask)
                progress.update(inference_task, advance=1)
                if i % 10 == 0:
                    power_readings.append(get_gpu_power(gpu_idx))
                    max_allocated_memory = max(max_allocated_memory, torch.cuda.max_memory_allocated())
                    max_reserved_memory = max(max_reserved_memory, torch.cuda.max_memory_reserved())
            run_end_time = time.time()
            total_time += run_end_time - run_start_time

    avg_run_time = total_time
    avg_power = np.mean(power_readings)
    max_power = np.max(power_readings)
    return avg_run_time, avg_power, max_power, max_allocated_memory, max_reserved_memory


def create_dummy_data(num_samples, seq_length, vocab_size, model_type):
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_length))
    attention_mask = torch.ones((num_samples, seq_length))
    if model_type == ModelType.mlm:
        labels = torch.randint(0, vocab_size, (num_samples, seq_length))
        mask = torch.rand(num_samples, seq_length) < 0.7
        labels[mask] = -100
    elif model_type == ModelType.seqcls:
        labels = torch.randint(0, 5, (num_samples, 1))
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    return TensorDataset(input_ids, attention_mask, labels)


def tile_list_to_length(lst, length):
    if len(lst) == 1:
        return lst * length
    return lst


# fmt: off
@app.command()
def main(
    ctx: typer.Context,  # Typer Context to grab config for --verbose and passing to WandB
    hidden_sizes: Annotated[List[int], Option(help="List of hidden sizes", show_default=False)],
    num_hidden_layers: Annotated[List[int], Option(help="List of number of hidden layers", show_default=False)],
    intermediate_sizes: Annotated[List[int], Option(help="List of intermediate sizes", show_default=False)],
    parallel_attn: Annotated[List[bool], Option(is_flag=False, help="List of parallel attention flags", show_default=False)],
    sliding_window: Annotated[List[int], Option(help="Sliding window size. -1 to disable.")] = [-1],
    global_attn_every_n_layers: Annotated[List[int], Option(help="Use global attention every `n` layers and sliding window for the rest. -1 to disable.")] = [-1],
    normalization: Annotated[List[str], Option(help="Normalization type: layernorm or triton_layernorm")] = ["layernorm"],
    compile_model: Annotated[List[bool], Option(help="Compile model")] = [True],
    masked_prediction: Annotated[List[bool], Option(help="Only pass the masked tokens through the final MLM layers")] = [True],
    model_type: Annotated[List[ModelType], Option(help="Model type: MLM or Multiple Choice")] = [ModelType.mlm],
    vocab_size: Annotated[List[int], Option(help="Vocabulary size")] = [32768],
    num_samples: Annotated[int, Option(help="Number of samples")] = 1000,
    seq_length: Annotated[int, Option(help="Sequence length")] = 512,
    batch_size: Annotated[Optional[int], Option(help="Batch size (if not provided, will be set based on model size)")] = None,
    output_file: Annotated[str, Option(help="Output file name for results")] = "benchmark_results.md",
    sleep_time: Annotated[int, Option(help="Time to sleep between each model run")] = 25,
    print_model: Annotated[bool, Option(help="Print model")] = False,
    num_workers: Annotated[int, Option(help="Number of workers")] = 8,
    skip_inference: Annotated[bool, Option(help="Skip inference")] = False,
    gpu_idx: Annotated[int, Option(help="GPU index for power measurements")] = 0,
    config: Annotated[
        Optional[Path],
        Option(
            callback=conf_callback,
            is_eager=True,
            help="Relative path to YAML config file for setting options. Passing CLI options will supersede config options.",
            case_sensitive=False,
        ),
    ] = None,
):
# fmt: on
    pynvml.nvmlInit()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine the maximum length of the lists
    max_length = max(
        len(hidden_sizes),
        len(num_hidden_layers),
        len(intermediate_sizes),
        len(parallel_attn),
        len(vocab_size),
        len(model_type),
        len(sliding_window),
        len(global_attn_every_n_layers),
        len(normalization),
        len(compile_model),
        len(masked_prediction),
    )

    # Tile lists to match the maximum length
    hidden_sizes = tile_list_to_length(hidden_sizes, max_length)
    num_hidden_layers = tile_list_to_length(num_hidden_layers, max_length)
    intermediate_sizes = tile_list_to_length(intermediate_sizes, max_length)
    parallel_attn = tile_list_to_length(parallel_attn, max_length)
    vocab_size = tile_list_to_length(vocab_size, max_length)
    model_type = tile_list_to_length(model_type, max_length)
    sliding_window = tile_list_to_length(sliding_window, max_length)
    global_attn_every_n_layers = tile_list_to_length(global_attn_every_n_layers, max_length)
    normalization = tile_list_to_length(normalization, max_length)
    compile_model = tile_list_to_length(compile_model, max_length)
    masked_prediction = tile_list_to_length(masked_prediction, max_length)
    # Create configs from the input lists
    configs = [
        {
            "hidden_size": hs,
            "num_hidden_layers": nhl,
            "intermediate_size": ims,
            "parallel_attn": pa,
            "vocab_size": vs,
            "model_type": mt,
            "sliding_window": sw,
            "global_attn_every_n_layers": swel,
            "normalization": norm,
            "compile_model": cm,
            "masked_prediction": mp,
        }
        for hs, nhl, ims, pa, vs, mt, sw, swel, norm, cm, mp in zip(
            hidden_sizes,
            num_hidden_layers,
            intermediate_sizes,
            parallel_attn,
            vocab_size,
            model_type,
            sliding_window,
            global_attn_every_n_layers,
            normalization,
            compile_model,
            masked_prediction,
        )
    ]

    results = []

    for i, config_params in enumerate(configs):
        if batch_size is None:
            if config_params["hidden_size"] >= 1793:
                config_batch_size = 8
            elif config_params["hidden_size"] >= 1024:
                config_batch_size = 8
            else:
                config_batch_size = 32
        else:
            config_batch_size = batch_size

        # Create dummy dataset and dataloader
        torch.manual_seed(42)
        dataset = create_dummy_data(num_samples, seq_length, vocab_size[i], model_type[i])
        dataloader = DataLoader(
            dataset, batch_size=config_batch_size, shuffle=True, drop_last=True, num_workers=num_workers
        )

        print(f"\nBenchmarking model with config: {config_params}")
        model = get_model(**config_params).to(device)
        if print_model:
            print(model)

        num_params = model.get_number_parameters()

        print("Training benchmark:")
        train_run_time, avg_train_power, max_train_power, max_train_allocated_memory, max_train_reserved_memory, loss = (
            benchmark_training(model, dataloader, num_warmup_batches=25, gpu_idx=gpu_idx)
        )

        model = None
        torch.cuda.empty_cache()

        if not skip_inference:
            # allow gpu to cool off
            time.sleep(sleep_time)

            model = get_model(**config_params).to(device)
            print("\nInference benchmark:")
            infer_run_time, avg_infer_power, max_infer_power, max_infer_allocated_memory, max_infer_reserved_memory = (
                benchmark_inference(model, dataloader, num_warmup_batches=50, gpu_idx=gpu_idx)
            )

        # Calculate tokens per second
        tokens_per_sample = seq_length
        train_tokens_per_second = (num_samples * tokens_per_sample) / train_run_time
        infer_tokens_per_second = (num_samples * tokens_per_sample) / infer_run_time if not skip_inference else 0

        # Calculate tokens per second per million parameters
        train_tokens_per_second_per_million_params = train_tokens_per_second / (num_params / 1e6)
        infer_tokens_per_second_per_million_params = infer_tokens_per_second / (num_params / 1e6) if not skip_inference else 0

        # Store results
        if skip_inference:
            results.append(
                {
                    "Final Loss": f"{loss:.4f}",
                    "Num Params (M)": f"{num_params / 1e6:.2f}",
                    "Vocab Size": int(config_params["vocab_size"]),
                    "Hidden Size": int(config_params["hidden_size"]),
                    "Num Layers": int(config_params["num_hidden_layers"]),
                    "Interm Size": int(config_params["intermediate_size"]),
                    "Parallel Attn": config_params["parallel_attn"],
                    "Normalization": config_params["normalization"],
                    "Compile Model": config_params["compile_model"],
                    "Masked Prediction": config_params["masked_prediction"],
                    "Train Time (s)": f"{train_run_time:.2f}",
                    "Train Tok/s": f"{train_tokens_per_second:.2f}",
                    "Avg Train W": f"{avg_train_power:.2f}",
                    "Max Train W": f"{max_train_power:.2f}",
                    "Max Train GiB": f"{max_train_reserved_memory / (1024**3):.2f}",
                    "Train Tok/s/M Params": f"{train_tokens_per_second_per_million_params:.2f}",
                }
            )
        else:
            results.append(
                {
                    "Final Loss": f"{loss:.4f}",
                    "Num Params (M)": f"{num_params / 1e6:.2f}",
                    "Vocab Size": int(config_params["vocab_size"]),
                    "Hidden Size": int(config_params["hidden_size"]),
                    "Num Layers": int(config_params["num_hidden_layers"]),
                    "Interm Size": int(config_params["intermediate_size"]),
                    "Parallel Attn": config_params["parallel_attn"],
                    "Normalization": config_params["normalization"],
                    "Compile Model": config_params["compile_model"],
                    "Masked Prediction": config_params["masked_prediction"],
                    "Train Time (s)": f"{train_run_time:.2f}",
                    "Infer Time (s)": f"{infer_run_time:.2f}" ,
                    "Train Tok/s": f"{train_tokens_per_second:.2f}",
                    "Infer Tok/s": f"{infer_tokens_per_second:.2f}",
                    "Avg Train W": f"{avg_train_power:.2f}",
                    "Max Train W": f"{max_train_power:.2f}",
                    "Avg Infer W": f"{avg_infer_power:.2f}",
                    "Max Infer W": f"{max_infer_power:.2f}",
                    "Max Train GiB": f"{max_train_reserved_memory / (1024**3):.2f}",
                    "Max Infer GiB": f"{max_infer_reserved_memory / (1024**3):.2f}",
                    "Train Tok/s/M Params": f"{train_tokens_per_second_per_million_params:.2f}",
                    "Infer Tok/s/M Params": f"{infer_tokens_per_second_per_million_params:.2f}",
                }
            )

        # Print individual results (optional, you can remove this if you only want the table)
        print("\nResults:")
        for key, value in results[-1].items():
            print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

        model = None
        dataset = None
        dataloader = None
        torch.cuda.empty_cache()

        # allow gpu to cool off
        if i < len(configs) - 1:
            time.sleep(sleep_time)

    # Create and print results table using pandas
    print("\nResults Table:")
    df = pd.DataFrame(results)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.precision", 4)
    print(df.to_string(index=False))

    # Save results as markdown table
    markdown_table = df.to_markdown(index=False, floatfmt=".4f")
    with open(output_file, "w") as f:
        f.write("# Benchmark Results\n\n")
        f.write(markdown_table)

    print(f"\nResults saved as '{output_file}'")

    pynvml.nvmlShutdown()


if __name__ == "__main__":
    app()
