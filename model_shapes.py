# messy LLM generated code
from contextlib import contextmanager, nullcontext
import math
import sys
from typing import Annotated, Optional
import optuna
import torch
import typer

from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

from src.bert_layers.model import FlexBertConfig, FlexBertModel

app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]}, pretty_exceptions_show_locals=False)

# # Define the SM counts for different GPUs with weights
SM_COUNTS = {
    40: {"weight": 1.25, "name": "T4"},
    60: {"weight": 1.25, "name": "L4"},
    68: {"weight": 1, "name": "3080"},
    72: {"weight": 1.25, "name": "A10"},
    76: {"weight": 1, "name": "4080"},
    80: {"weight": 1.1, "name": "V100"},
    82: {"weight": 1, "name": "3090"},
    84: {"weight": 1, "name": "A6000"},
    108: {"weight": 1.1, "name": "A100"},
    128: {"weight": 1, "name": "4090"},
    142: {"weight": 1, "name": "6000 Ada"},
    144: {"weight": 1.1, "name": "H100"},
    20: {"weight": 0, "name": "4050M"},
}


def is_divisible_by_64(n):
    return n % 64 == 0


def is_divisible_by_256_128(n):
    return n % (256 * 128) == 0


def calculate_sm_tiling_score(name, x, y):
    results = {}
    blocks1 = math.ceil(x / 128) * math.ceil(y / 256)  # Total number of blocks
    blocks2 = math.ceil(y / 128) * math.ceil(x / 256)  # For both possible configs

    for sm_count, gpu in SM_COUNTS.items():
        full_waves, remaining_blocks, utilizations, scores = [], [], [], []
        for blocks in [blocks1, blocks2]:
            full_wave = blocks // sm_count
            remaining_block = blocks % sm_count

            if remaining_block == 0:
                utilization = 1.0  # Perfect utilization
            else:
                utilization = remaining_block / sm_count  # Utilization based on remainder

            score = utilization * gpu["weight"]  # Calculate final score, applying the GPU-specific weight

            full_waves.append(full_wave)
            remaining_blocks.append(remaining_block)
            utilizations.append(utilization)
            scores.append(score)

        avg_score = sum(scores) / len(scores)  # Calculate average score

        results[sm_count] = {
            "name": name,
            "x": x,
            "y": y,
            "blocks1": blocks1,
            "blocks2": blocks2,
            "full_waves1": full_waves[0],
            "full_waves2": full_waves[1],
            "remaining_blocks1": remaining_blocks[0],
            "remaining_blocks2": remaining_blocks[1],
            "utilization1": utilizations[0],
            "utilization2": utilizations[1],
            "score1": scores[0],
            "score2": scores[1],
            "avg_score": avg_score,
        }

    return results


def get_shapes(layer: torch.nn.Module):
    shapes = []
    if hasattr(layer.attn, "Wqkv"):
        shapes.append(("Wqkv", layer.attn.Wqkv.weight.shape))
    elif hasattr(layer, "Wqkvff"):
        shapes.append(("Wqkvff", layer.Wqkvff.weight.shape))
    shapes.append(("Attn Wo", layer.attn.Wo.weight.shape))
    if hasattr(layer.mlp, "Wi"):
        shapes.append(("FFN Wi", layer.mlp.Wi.weight.shape))
    shapes.append(("FFN Wo", layer.mlp.Wo.weight.shape))
    return tuple(shapes)


def calculate_score(model: torch.nn.Module, siglu_min: float = 2, siglu_max: int = 4):
    score = 0
    results = {}
    shapes = get_shapes(model.encoder.layers[0])

    for layer in shapes:
        name, (x, y) = layer
        if not is_divisible_by_64(x) or not is_divisible_by_64(y):
            return float("-inf"), results

        if not is_divisible_by_256_128(x * y):
            return float("-inf"), results

        if name == "Wqkvff" and not (siglu_min < (x - (y * 3)) / y < siglu_max):
            return float("-inf"), results

    # Check if using parallel attention (Wqkvff)
    if any(layer[0] == "Wqkvff" for layer in shapes):
        # Find Wqkvff, Attn Wo, and FFN Wi shapes
        wqkvff_shape = next(layer[1] for layer in shapes if layer[0] == "Wqkvff")
        attn_wo_shape = next(layer[1] for layer in shapes if layer[0] == "Attn Wo")
        ffn_wi_shape = next(layer[1] for layer in shapes if layer[0] == "FFN Wo")

        # Calculate parallel tiling score for Wqkvff
        wqkvff_scores = calculate_sm_tiling_score("Wqkvff", wqkvff_shape[0], wqkvff_shape[1])
        results["Wqkvff"] = wqkvff_scores
        score += sum(result["avg_score"] for result in wqkvff_scores.values())

        # Calculate parallel tiling score for Attn Wo and FFN Wi
        parallel_scores = calculate_sm_tiling_score("Parallel Wo", attn_wo_shape[0], attn_wo_shape[1] + ffn_wi_shape[1])
        results["Parallel Wo"] = parallel_scores
        score += sum(result["avg_score"] for result in parallel_scores.values()) * 0.5
        un_parallel_scores = calculate_sm_tiling_score("Attn Wo", attn_wo_shape[0], attn_wo_shape[1])
        results["Attn Wo"] = un_parallel_scores
        score += sum(result["avg_score"] for result in un_parallel_scores.values()) * 0.25
        un_parallel_scores = calculate_sm_tiling_score("FFN Wo", ffn_wi_shape[0], ffn_wi_shape[1])
        results["FFN Wo"] = un_parallel_scores
        score += sum(result["avg_score"] for result in un_parallel_scores.values()) * 0.25
    else:
        for layer in shapes:
            layer_scores = calculate_sm_tiling_score(layer[0], layer[1][0], layer[1][1])
            results[layer[0]] = layer_scores
            score += sum(result["avg_score"] for result in layer_scores.values())

    return score, results


def get_model(hidden_size: int, num_layers: int, intermediate_size: float, parallel_attn: bool = True):
    config = FlexBertConfig(
        num_attention_heads=hidden_size // 64,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        intermediate_size=intermediate_size,
        vocab_size=32768,
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
        loss_function="cross_entropy",
        loss_kwargs={"reduction": "mean"},
        mlp_dropout_prob=0.0,
        mlp_in_bias=False,
        mlp_layer="parallel_glu" if parallel_attn else "glu",
        mlp_out_bias=False,
        norm_kwargs={"eps": 1e-5},
        normalization="layernorm",
        padding="padded",
        head_class_act="silu",
        head_class_bias=False,
        head_class_dropout=0.0,
        head_class_norm=False,
        head_pred_act="silu",
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
    )
    with torch.device("meta"):
        model = FlexBertModel(config)
    return model


def calculate_total_trials(search_space):
    total = 1
    for param_values in search_space.values():
        total *= len(param_values)
    return total


def optimize_model_shape(
    target_params: int,
    hidden_size: int,
    parallel_attn: bool,
    min_layers: int,
    max_layers: int,
    top_configs: int,
    num_cpus: int,
    percent_off: float = 0.05,
    siglu_min: float = 2,
    siglu_max: int = 4,
):
    def objective(trial):
        obj_num_layers = trial.suggest_int("num_layers", min_layers, max_layers)
        obj_intermediate_size = trial.suggest_int("intermediate_size", hidden_size, (hidden_size * 5) + 64, step=64)

        model = get_model(hidden_size, obj_num_layers, obj_intermediate_size, parallel_attn)

        # Check parameter count first
        params = model.num_parameters()

        trial.set_user_attr("num_layers", obj_num_layers)
        trial.set_user_attr("intermediate_size", obj_intermediate_size)
        trial.set_user_attr("params", params)

        if abs(params - target_params) / target_params > percent_off:
            trial.set_user_attr("results", {})
            return float("-inf")

        score, results = calculate_score(model, siglu_min, siglu_max)
        trial.set_user_attr("results", results)

        return score

    search_space = {
        "num_layers": [num_layers for num_layers in range(min_layers, max_layers + 1) if num_layers % 2 == 0],
        "intermediate_size": [
            intermediate_size
            for intermediate_size in range(hidden_size, hidden_size * 6 + 64, 64)
            if siglu_min <= (intermediate_size * 2) / hidden_size <= siglu_max
        ],
    }

    total_trials = calculate_total_trials(search_space)
    print(f"Total number of grid search trials: {total_trials}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.GridSampler(search_space),
    )

    # Create Rich progress bar
    progress = Progress(
        TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn(), TimeRemainingColumn()
    )

    with progress:
        task = progress.add_task("[cyan]Optimizing...", total=total_trials)

        def update_progress(_study, _trial):
            progress.update(task, advance=1)

        study.optimize(objective, n_jobs=num_cpus, gc_after_trial=True, callbacks=[update_progress])

    seen_configs = set()
    unique_trials = []

    # optuna is returning duplicates despite the search space being unique, so we need to filter them out
    for trial in study.trials:
        config = (
            trial.value,
            trial.user_attrs["num_layers"],
            trial.user_attrs["intermediate_size"],
            trial.user_attrs["params"],
        )
        if config not in seen_configs:
            seen_configs.add(config)
            unique_trials.append(trial)

    top_trials = sorted(
        unique_trials,
        key=lambda t: (
            -t.value,  # Highest score first
            t.user_attrs["num_layers"],  # Lowest number of layers next
            t.user_attrs["params"],  # Lowest number of parameters last
        ),
    )[:top_configs]

    top_configs = []
    for trial in top_trials:
        top_configs.append(
            (
                trial.user_attrs["num_layers"],
                trial.user_attrs["intermediate_size"],
                trial.value,
                trial.user_attrs["params"],
                trial.user_attrs["results"],
            )
        )

    return top_configs


@contextmanager
def output_to_file_and_console(file_path):
    class FileAndConsoleWriter:
        def __init__(self, file):
            self.file = file
            self.stdout = sys.stdout

        def write(self, message):
            self.file.write(message)
            self.stdout.write(message)

        def flush(self):
            self.file.flush()
            self.stdout.flush()

    with open(file_path, "w") as f:
        writer = FileAndConsoleWriter(f)
        sys.stdout = writer
        try:
            yield
        finally:
            sys.stdout = writer.stdout


@app.command()
def main(
    num_cpus: Annotated[int, typer.Argument(help="Number of CPUs to use")],
    hidden_size: Annotated[int, typer.Option(help="Hidden size")] = 768,
    target_params: Annotated[int, typer.Option(help="Target number of parameters")] = 110_000_000,
    parallel_attn: Annotated[bool, typer.Option(help="Use parallel attention")] = True,
    min_layers: Annotated[int, typer.Option(help="Minimum number of layers")] = 10,
    max_layers: Annotated[int, typer.Option(help="Maximum number of layers")] = 18,
    siglu_min: Annotated[float, typer.Option(help="Minimum SiGLU ratio")] = 2,
    siglu_max: Annotated[float, typer.Option(help="Maximum SiGLU ratio")] = 4,
    percent_off: Annotated[float, typer.Option(help="Percent off from target parameters")] = 0.05,
    top_configs: Annotated[int, typer.Option(help="Number of top configurations to print")] = 10,
    output_file: Annotated[Optional[str], typer.Option(help="Output file path (optional)")] = None,
):
    configs = optimize_model_shape(
        target_params,
        hidden_size,
        parallel_attn,
        min_layers,
        max_layers,
        top_configs * 3,
        num_cpus,
        percent_off,
        siglu_min,
        siglu_max,
    )

    output_context = output_to_file_and_console(output_file) if output_file else nullcontext()

    with output_context:
        count = 1
        for _, (num_layers, intermediate_size, score, params, results) in enumerate(configs):
            if score > 0:
                print(f"\nConfiguration #{count}:")
                print(f"  num_hidden_layers={num_layers}")
                print(f"  intermediate_size={intermediate_size}")
                print(f"  SM tiling score: {score}")
                print(f"  Total parameters: {params:,}")

                # Add table of ratings for each layer
                print("\n    Summary Table:")
                print("                |  T4   |  L4   |  A10  | A100  | H100  | 3090  | 4090")
                print("    ------------+-------+-------+-------+-------+-------+-------+-------")
                weight_names = ["Wqkvff", "Parallel Wo"] if parallel_attn else ["Wqkv", "FFN Wi"]
                for weight_name in weight_names + ["Attn Wo", "FFN Wo"]:
                    row = f"    {weight_name:11}"
                    for gpu_name in ["T4", "L4", "A10", "A100", "H100", "3090", "4090"]:
                        sm_count = next(sm for sm, data in SM_COUNTS.items() if data["name"] == gpu_name)
                        rating = ""
                        for layer_name, layer_results in results.items():
                            if layer_name == weight_name:
                                utilization1 = layer_results[sm_count]["utilization1"]
                                utilization2 = layer_results[sm_count]["utilization2"]
                                utilization = max(utilization1, utilization2)
                                if utilization >= 0.85:
                                    rating = "Great"
                                elif utilization >= 0.7:
                                    rating = "Good"
                                elif utilization >= 0.45:
                                    rating = "Okay"
                                elif utilization >= 0.3:
                                    rating = "Poor"
                                else:
                                    rating = "Bad"
                                break
                        row += f" | {rating:5}"
                    print(row)
                print("\n")

                # Print tiling information for each layer using the results dict
                for layer_name, layer_results in results.items():
                    gpu = next(iter(layer_results.keys()))
                    x, y = layer_results[gpu]["x"], layer_results[gpu]["y"]
                    print(f"  Layer: {layer_name} (in: {y}, out: {x}):")
                    if layer_name == "Wqkvff":
                        print(f"  SiGLU ratio: {(x - (hidden_size * 3)) / hidden_size:.4f}")
                    total_score = 0
                    for sm_count, result in layer_results.items():
                        if result["blocks1"] != result["blocks2"]:
                            print(
                                f"    GPU {SM_COUNTS[sm_count]['name']:9} SMs {sm_count:3d}:  "
                                f"Blocks: {result['blocks1']:3d},  Full waves: {result['full_waves1']:2d},  Remaining blocks: {result['remaining_blocks1']:3d},  Utilization: {result['utilization1']:.4f},  Score: {result['score1']:.4f}\n"
                                f"                            "
                                f"Blocks: {result['blocks2']:3d},  Full waves: {result['full_waves2']:2d},  Remaining blocks: {result['remaining_blocks2']:3d},  Utilization: {result['utilization2']:.4f},  Score: {result['score2']:.4f}\n"
                                f"        "
                                f"Average score: {result['avg_score']:.4f}"
                            )
                        else:
                            print(
                                f"    GPU {SM_COUNTS[sm_count]['name']:9} SMs {sm_count:3d}:  "
                                f"Blocks: {result['blocks1']:3d},  Full waves: {result['full_waves1']:2d},  Remaining blocks: {result['remaining_blocks1']:3d},  Utilization: {result['utilization1']:.4f},  Score: {result['score1']:.4f}"
                            )
                        total_score += result["avg_score"]
                    print(f"    Total score: {total_score:.4f}")

                count += 1
            if count > top_configs:
                break


def debug_sm_tiling_score(x, y):
    print(f"\nTesting dimensions: {x} x {y}")
    results = calculate_sm_tiling_score(x, y)
    blocks = results[40]["blocks"]
    print(f"Total blocks: {blocks}")

    for sm_count, result in results.items():
        utilization = result["score"] / SM_COUNTS[sm_count]["weight"]
        print(
            f"{SM_COUNTS[sm_count]['name']:9} SM count {sm_count:3d}: "
            f"Full waves: {result['full_waves']:3d}, "
            f"Remaining blocks: {result['remaining_blocks']:3d}, "
            f"Utilization: {utilization:.4f}, "
            f"Weighted score: {result['score']:.4f}"
        )


if __name__ == "__main__":
    app()
