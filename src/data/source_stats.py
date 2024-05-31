import argparse
import huggingface_hub
from collections import Counter
from datasets import load_dataset
from tqdm import tqdm
import tempfile
from pathlib import Path
import multiprocessing
import re

import pandas as pd
import math


DOLMA_HF_EXCLUDE_KEYS = [".gitattributes", "README.md"]
NUM_PROC = int(math.ceil(0.75 * multiprocessing.cpu_count()))


def simple_splitter(text: str) -> int:
    return len(re.split(r"\W+", text))


def main(dolma_hf_path, tokenizer, out_fn):
    data_dirs = list(map(lambda x: x.path, huggingface_hub.list_repo_tree(dolma_hf_path, repo_type="dataset")))
    data_dirs = list(filter(lambda x: x not in DOLMA_HF_EXCLUDE_KEYS, data_dirs))

    sources = Counter(["_".join(x.split("_")[:-1]) for x in data_dirs])

    print(sources.most_common())

    stats = []
    for data_dir in tqdm(data_dirs):
        source = "_".join(data_dir.split("_")[:-1])

        with tempfile.TemporaryDirectory() as tmp_cache_dir:
            dataset = load_dataset(dolma_hf_path, data_dir=data_dir, split="train", cache_dir=tmp_cache_dir)

            # Return all column names in dataset
            remove_columns = dataset.column_names

            num_tokens = sum(dataset.map(
                lambda row: {"num_tokens": tokenizer(row["text"]), "batched": True},
                num_proc=NUM_PROC, remove_columns=remove_columns
            )["num_tokens"])

            # This is overkill, but just in case
            dataset.cleanup_cache_files()

            stats.append({
                "source": source,
                "num_tokens": num_tokens,
                "shards": 1,
            })

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
    parser = argparse.ArgumentParser(description="Measure Dolma token count per data source.")

    parser.add_argument("--dolma_hf_path", type=str, default="orionweller/dolma_20_percent_sample", help="Path to Dolma data on HF.")
    parser.add_argument("--tokenizer", type=callable, default=simple_splitter, help="Tokenizer function to use.")
    parser.add_argument("--out_fn", type=Path, default=Path(__file__).resolve().parent / "source_stats.csv", help="Output file for source stats.")

    args = parser.parse_args()
    
    main(args.dolma_hf_path, args.tokenizer, args.out_fn)
