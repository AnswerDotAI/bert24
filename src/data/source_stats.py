import argparse
import huggingface_hub
from collections import Counter
from datasets import load_dataset
from tqdm import tqdm
import tempfile
from pathlib import Path
import multiprocessing
import re

import numpy as np
import pandas as pd
import math
import json


from utils import ALL_REPOS


NUM_PROC = int(math.ceil(0.75 * multiprocessing.cpu_count()))


def main(dolma_hf_path, out_fn):
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B-hf")

    percentiles = [1, 99] + list(range(0, 101, 5))
    percentiles_out = open(str(out_fn).replace(".csv", ".jsonl"), "w")
    print(f"Saving source-level token count percentiles to {str(out_fn).replace('.csv', '.jsonl')}")
    stats = []
    tokens_for_source = []
    prev_src = ALL_REPOS[0].split("/")[-1]
    for data_dir in tqdm(ALL_REPOS):
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
            percentiles_out.write(json.dumps({prev_src: percentile_stats}) + "\n")
            percentiles_out.flush()
    
        prev_src = source

        with tempfile.TemporaryDirectory() as tmp_cache_dir:
            breakpoint()
            # TODO: make this use the StreamingDataset instead
            remote = f'hf://datasets/{source_hf_repo}/'
            dataset = StreamingDataset(remote=remote, shuffle=True, split=None, batch_size=1)
            
            # remove this section
            dataset = load_dataset(f"orionweller/{source}", data_dir=data_dir, split="train", cache_dir=tmp_cache_dir)

            # Return all column names in dataset
            remove_columns = dataset.column_names

            tokens = dataset.map(
                lambda row: {"num_tokens": tokenizer(row["text"]), "batched": True},
                num_proc=NUM_PROC, remove_columns=remove_columns
            )["num_tokens"]
            tokens_for_source.extend(tokens)

            # This is overkill, but just in case
            dataset.cleanup_cache_files()

            stats.append({
                "source": source,
                "num_tokens": sum(tokens),
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

    args = parser.parse_args()
    
    main(args.out_fn)
