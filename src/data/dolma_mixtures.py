import argparse
import huggingface_hub
from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm
import tempfile
import yaml
import os
import math
import numpy as np
import multiprocessing
from pathlib import Path
import pandas as pd

from source_stats import simple_splitter, main as compute_corpus_stats


NUM_PROC = int(math.ceil(0.75 * multiprocessing.cpu_count()))
DOLMA_HF_EXCLUDE_KEYS = [".gitattributes", "README.md"]


def push_to_hub(dataset, repo_id, split, debug=False):
    if debug:
        print("Remove -debug if you'd like to push to Hub.")
    else:
        dataset.push_to_hub(repo_id, split=split)


def main(args):
    in_fn = args.config_dir / args.config_fn
    assert in_fn.exists(), f"Config file {in_fn} does not exist."
    with open(in_fn, 'r') as file:
        config = yaml.safe_load(file)
    
    hf_out_path = config["hf_out_path"]
    target_tokens = config["target_tokens"]

    api = huggingface_hub.HfApi()
    if api.repo_exists(repo_id=hf_out_path, repo_type="dataset"):
        # Ask user if they want to overwrite
        print(f"Repo {hf_out_path} already exists. Do you want to overwrite?")
        response = input("y/n: ")
        if response.lower() != "y":
            print("Exiting.")
            return
        print(f"Ok! Deleting {hf_out_path}.")
        api.delete_repo(repo_id=hf_out_path, repo_type="dataset")

    existing_cts = pd.read_csv(args.dolma_token_ct_fn)
    # Number of tokens for each source in existing source data
    src_cts = dict(zip(existing_cts["source"], existing_cts["num_tokens"]))

    dolma_tok_ct = sum(src_cts.values())
    print(f"Existing dataset has {dolma_tok_ct} tokens.")

    if target_tokens < 1.0:
        target_frac = target_tokens
        target_tokens = round(dolma_tok_ct * target_tokens)
        print(f"Requested {target_frac} of {dolma_tok_ct} existing tokens ~= {target_tokens} tokens.")

    # Apply coefficients to existing data with no adjustments and measure how close we are to target tokens
    if config["source_coefficient_type"] == "relative_proportion":
        sample_coefficients = {x["name"]: x["source_coefficient"] for x in config["sources"]}
        # Estimate how many tokens we'd have without adjusting sample_coefficients
        unadjusted_target_tokens = sum([v * src_cts[k] for k, v in sample_coefficients.items()])
        # Adjust sample_coefficients uniformly to ensure desired target tokens
        adjustment_ratio = target_tokens / unadjusted_target_tokens
        sample_coefficients = {k: adjustment_ratio * v for k, v in sample_coefficients.items()}
    else:
        assert config["source_coefficient_type"] == "target_fraction", f"Source coefficient type {config['source_coefficient_type']} not recognized."
        
        # Compute number of tokens needed to hit target
        src_toks_needed = {
            source["name"]: target_tokens * source["source_coefficient"] for source in config["sources"]
        }

        # Coefficients are the ratio of the number of tokens needed to hit target to the number of tokens we have (src_cts)
        sample_coefficients = {k: v / src_cts[k] for k, v in src_toks_needed.items()}

    print(f"Targeting {target_tokens} tokens with the following sampling fractions by source:")
    print("\n".join(f"\t- {k} -> {round(v, 4)}" for k, v in sample_coefficients.items()))

    data_dirs = list(map(lambda x: x.path, huggingface_hub.list_repo_tree(args.dolma_hf_path, repo_type="dataset")))
    data_dirs = list(filter(lambda x: x not in DOLMA_HF_EXCLUDE_KEYS, data_dirs))

    sources = [
        "_".join(x.split("_")[:-1]) for x in data_dirs
    ]

    sources_uniq = set(sources)
    assert sources_uniq == set(sample_coefficients.keys()) == set(src_cts.keys()), f"Sources in config file {set(sample_coefficients.keys())} must match those in HF datafiles {sources_uniq} and those in pre-computed token counts of the HF data ({set(src_cts.keys())})."

    # Initialize counters and aggregated shard info
    updated_stats = []
    curr_shards = []
    curr_size_gbs = 0.0
    save_ctr = 0

    for source, data_dir in tqdm(zip(sources, data_dirs)):
        frac_to_sample = sample_coefficients[source]

        if frac_to_sample == 0:
            updated_stats.append({
                "source": source,
                "num_tokens": 0,
                "shards": 0,
            })

            print(f"Excluding {data_dir}.")
            continue

        with tempfile.TemporaryDirectory() as tmp_cache_dir:
            dataset = load_dataset(args.dolma_hf_path, data_dir=data_dir, split="train", cache_dir=tmp_cache_dir)
            idxs = np.arange(len(dataset))
            target_n = int(len(dataset) * frac_to_sample)
            np.random.shuffle(idxs)

            if frac_to_sample > 1.0:
                # Repeat idxs to get more samples since we are upsampling
                idxs = np.tile(idxs, math.ceil(frac_to_sample))

            sampled_idxs = list(sorted(idxs[:target_n]))

            sampled_dataset = dataset.select(sampled_idxs)

            # For easy data provenance.
            sampled_dataset = sampled_dataset.add_column("original_shard_dir", [data_dir] * len(sampled_dataset))
            sampled_dataset = sampled_dataset.add_column("original_shard_idx", sampled_idxs)

            # Compute token count so we can return final mixture to make sure its close to desired.
            sampled_dataset = sampled_dataset.map(
                lambda batch: {"num_tokens": list(map(simple_splitter, batch["text"]))},
                batched=True,
                num_proc=NUM_PROC
            )

            updated_stats.append({
                "source": source,
                "num_tokens": sum(sampled_dataset["num_tokens"]),
                "shards": 1,
            })

            # if args.min_shard_size is not None:
            curr_size_gbs += sampled_dataset.data.nbytes / 1e9
            curr_shards.append(sampled_dataset)
            if curr_size_gbs > args.min_shard_size:
                shard_dir = f"shard_{save_ctr}"
                out_data = concatenate_datasets(curr_shards)
                print(f"Saving {len(out_data)} samples from {source} to {hf_out_path} -> {shard_dir}.")
                push_to_hub(out_data, repo_id=hf_out_path, split=shard_dir, debug=args.debug)

                save_ctr += 1
                curr_shards = []
                curr_size_gbs = 0.0

    if len(curr_shards) > 0:
        shard_dir = f"shard_{save_ctr}"
        out_data = concatenate_datasets(curr_shards)
        print(f"Saving {len(out_data)} samples from {source} to {hf_out_path} -> {shard_dir}.")
        push_to_hub(out_data, repo_id=hf_out_path, split=shard_dir, debug=args.debug)
        curr_shards = []
        curr_size_gbs = 0.0
        save_ctr += 1

    updated_stats = pd.DataFrame(updated_stats)

    # Group by source and sum num_tokens
    updated_stats = updated_stats.groupby("source").sum().reset_index()

    # Add a column which shows the fractional contribution of each source
    updated_stats["fraction"] = updated_stats["num_tokens"] / updated_stats["num_tokens"].sum()

    # Sort by fraction decreasing
    updated_stats = updated_stats.sort_values("fraction", ascending=False)

    updated_stats["original_num_tokens"] = updated_stats["source"].map(src_cts)
    updated_stats["percent_of_original"] = 100 * updated_stats["num_tokens"] / updated_stats["original_num_tokens"]

    out_stats_fn = in_fn.replace('.yaml', '_stats.csv')
    print(f"Saving updated source-level token count statistics to {out_stats_fn}")
    updated_stats.to_csv(out_stats_fn, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dolma mixtures according to mixture_configs.")

    parser.add_argument("--config_dir", type=Path, default=Path(__file__).resolve().parent / "configs", help="Output directory for mixtures.")
    parser.add_argument("--config_fn", type=str, default="stratified_20bn.yaml", help="Path to Yaml file containing mixture weights.")
    parser.add_argument("--dolma_hf_path", type=str, default="orionweller/dolma_20_percent_sample", help="Path to Dolma data on HF.")
    parser.add_argument("--dolma_token_ct_fn", type=Path, default=Path(__file__).resolve().parent / "source_stats.csv", help="Dolma token counts - precomputed.")

    parser.add_argument("--min_shard_size", default=10, type=float, help="Minimum shard size in Gigabytes.")

    parser.add_argument("-debug", default=False, action="store_true", help="Include to do a dry run without pushing to Hub.")

    args = parser.parse_args()

    if not os.path.exists(args.dolma_token_ct_fn):
        print(f"Token count file {args.dolma_token_ct_fn} does not exist. Computing and saving to {args.dolma_token_ct_fn} first.")

        compute_corpus_stats(
            args.dolma_hf_path, tokenizer=simple_splitter, out_fn=args.dolma_token_ct_fn
        )

    main(args)
