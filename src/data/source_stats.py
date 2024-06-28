import argparse
import huggingface_hub
from collections import Counter
from datasets import load_dataset, Dataset
import tqdm
import tempfile
from pathlib import Path
import multiprocessing
import re
import numpy as np
import pandas as pd
import math
import json
import os
from transformers import AutoTokenizer
from streaming import StreamingDataset
from streaming.base.util import clean_stale_shared_memory

from data_utils import ALL_REPOS, MDS_COLS_TEXT


NUM_PROC = int(math.ceil(0.35 * multiprocessing.cpu_count()))

model_name = "gpt2"

def main(out_fn, dataset_max_size):
    tokenizer = AutoTokenizer.from_pretrained(model_name, fast=True)

    # load all lines in out_fn
    percentiles_out_path = str(out_fn).replace(".csv", ".jsonl")
    cached_sources = set()
    if os.path.exists(percentiles_out_path):
        with open(str(out_fn).replace(".csv", ".jsonl"), "r") as f:
            for line in f:
                cached_sources.add(json.loads(line)["source"])


    percentiles = [1, 99] + list(range(0, 101, 5))
    print(f"Saving source-level token count percentiles to {str(out_fn).replace('.csv', '.jsonl')}")
    stats = []
    tokens_for_source = []
    current_repos_to_do = [item for item in ALL_REPOS if item.split("/")[-1] not in cached_sources]
    prev_src = current_repos_to_do[0].split("/")[-1]
    percentiles_out = open(percentiles_out_path, "a")

    for data_dir in tqdm.tqdm(current_repos_to_do):
        source = data_dir.split("/")[-1]
        print(f"Processing {source}... with data_dir {data_dir}")
        if source != prev_src:
            # add percentiles and reset
            tokens_np = np.array(tokens_for_source)
            percentile_stats_all = np.percentile(tokens_np, percentiles)
            percentile_stats = {
                "mean": np.mean(tokens_np),
                "std": np.std(tokens_np),
                "percentiles": {p: v for p, v in zip(percentiles, percentile_stats_all)}
            }
            tokens_for_source = []
            percentiles_out.write(json.dumps({prev_src: percentile_stats, "source": prev_src}) + "\n")
            percentiles_out.flush()
    
        prev_src = source

        with tempfile.TemporaryDirectory() as tmp_cache_dir:
            remote = f'hf://datasets/orionweller/{source}/'
            token_lens = []
            pool = []
            clean_stale_shared_memory()
            for idx, instance in tqdm.tqdm(enumerate(StreamingDataset(remote=remote, shuffle=False, split=None, batch_size=1, predownload=dataset_max_size))):
                pool.append(instance)
                if idx > dataset_max_size:
                    break
                if len(pool) > 1000:
                    hf_dataset = Dataset.from_list(pool)
                    try:
                        tokens = hf_dataset.map(
                            lambda row: {"num_tokens": tokenizer(row["text"]), "batched": True},
                            num_proc=NUM_PROC, remove_columns=MDS_COLS_TEXT.keys()
                        )["num_tokens"]
                    except Exception as e:
                        print(f"Error processing {source} at idx {idx}")
                        print(e)
                        tokens = hf_dataset.map(
                            lambda row: {"num_tokens": tokenizer(row["text"]), "batched": True},
                            num_proc=NUM_PROC, remove_columns=MDS_COLS_TEXT.keys()
                        )["num_tokens"]
                    token_lens.extend([len(item["input_ids"]) for item in tokens])
                    hf_dataset.cleanup_cache_files()
                    pool = []

            hf_dataset = Dataset.from_list(pool)
            tokens = hf_dataset.map(
                lambda row: {"num_tokens": tokenizer(row["text"]), "batched": True},
                num_proc=NUM_PROC, remove_columns=MDS_COLS_TEXT.keys()
            )["num_tokens"]
            token_lens.extend([len(item["input_ids"]) for item in tokens])

            tokens_for_source.extend(token_lens)

            # This is overkill, but just in case
            hf_dataset.cleanup_cache_files()

            stats.append({
                "source": source,
                "num_tokens": sum(token_lens)
            })

    # do the percentile calculation for the last source also
    tokens_np = np.array(tokens_for_source)
    percentile_stats_all = np.percentile(tokens_np, percentiles)
    percentile_stats = {
        "mean": np.mean(tokens_np),
        "std": np.std(tokens_np),
        "percentiles": {p: v for p, v in zip(percentiles, percentile_stats_all)}
    }
    percentiles_out.write(json.dumps({prev_src: percentile_stats}) + "\n")

    # now get the total stats
    stats = pd.DataFrame(stats)

    # Group by source and sum num_tokens
    stats = stats.groupby("source").sum().reset_index()

    # Add a column which shows the fractional contribution of each source
    stats["fraction"] = stats["num_tokens"] / stats["num_tokens"].sum()

    # Sort by fraction decreasing
    stats = stats.sort_values("fraction", ascending=False)

    print(f"Saving source-level token count statistics to {out_fn}")
    stats.to_csv(out_fn, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure token count per data source.")
    parser.add_argument("--out_fn", type=Path, default=Path(__file__).resolve().parent / "statistics" / "source_stats.csv", help="Output file for source stats.")
    parser.add_argument("--dataset_max_size", type=int, default=100000, help="Maximum number of instances to load at once.")
    args = parser.parse_args()
    
    main(args.out_fn, args.dataset_max_size)

    # example usage:
    #   python source_stats.py --dataset_max_size 100000
