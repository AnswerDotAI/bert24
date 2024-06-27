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
import time
from transformers import AutoTokenizer

from source_stats import simple_splitter, main as compute_corpus_stats

from datasets.utils.logging import disable_progress_bar
disable_progress_bar()


DOLMA_TOTAL_TOKENS = 1715100000000 # 1.715 Trillion tokens


def relative_to_instance(args):
    assert os.path.isfile(args.config), f"Config file {in_fn} does not exist."
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    
    target_tokens = config["target_tokens"]
    token_adjustment_ratio = target_tokens / DOLMA_TOTAL_TOKENS

    # contains `tokens_per_instance`, `instance_proportions`, and `num_instances` for each source
    existing_cts = pd.read_csv(args.dolma_ground_truth)
    source2instances = dict(zip(existing_cts["source"], existing_cts["num_instances"]))

    # get the scaled amount of instances we need, based on the token adjustment ratio
    # NOTE: appx, since the number of tokens per instance is appx
    source2instances_scaled = {k: v * token_adjustment_ratio for k, v in source2instances.items()}

    # from the config file, get what relative weights we want
    sample_coefficients = {x["name"]: x["source_coefficient"] for x in config["sources"]}

    # multiply the relative weights by the number of instances
    final_proportions = {k: v * source2instances_scaled[k] for k, v in sample_coefficients.items()}
    print(f"Targeting {target_tokens} tokens with the following sampling fractions by source:")
    print("\n".join(f"\t- {k} -> {round(v, 4)}" for k, v in final_proportions.items()))

    breakpoint()
    # make a new config file with the instance numbers instead

    # write this out to the relative_configs folder that will be used for sampling (one dir up)
    out_fn = Path(args.config).parent / "relative_configs" / args.config.name
    out_fn.parent.mkdir(exist_ok=True)
    with open(out_fn, 'w') as file:
        yaml.dump(config, file)

    print(f"Written to {out_fn}")

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dolma_ground_truth", type=Path, required=True)
    args = parser.parse_args()

    relative_to_instance(args)