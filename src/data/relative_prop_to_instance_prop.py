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


from datasets.utils.logging import disable_progress_bar
disable_progress_bar()

TOTAL_TOKENS = 1000000000


def relative_to_instance(args):
    assert os.path.isfile(args.config), f"Config file {in_fn} does not exist."
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    
    target_tokens = config["target_tokens"]
    token_adjustment_ratio = target_tokens / TOTAL_TOKENS

    # contains `tokens_per_instance`, `instance_proportions`, and `num_instances` for each source
    existing_cts = pd.read_csv(args.ground_truth)
    source2instances = dict(zip(existing_cts["sources"], existing_cts["num_instances"]))

    # get the scaled amount of instances we need, based on the token adjustment ratio
    # NOTE: appx, since the number of tokens per instance is appx
    source2instances_scaled = {k: v * token_adjustment_ratio for k, v in source2instances.items()}

    # from the config file, get what relative weights we want
    sample_coefficients = {x["name"]: x["source_coefficient"] for x in config["sources"]}

    # multiply the relative weights by the number of instances
    final_proportions = {k: v * source2instances_scaled[k] for k, v in sample_coefficients.items()}
    print(f"Targeting {target_tokens} tokens with the following sampling fractions by source:")
    print("\n".join(f"\t- {k} -> {round(v, 4)}" for k, v in final_proportions.items()))

    # make a new config file with the instance numbers instead
    # write this out to the instance config folder that will be used for sampling (one dir up)
    out_fn = args.config.replace("relative", "instances")
    # replace sources with the `final_proportions` dict
    config["sources"] = [{"name": k, "num_instances": round(v)} for k, v in final_proportions.items()]
    with open(out_fn, 'w') as file:
        yaml.dump(config, file)
    print(f"New config file written to {out_fn}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ground_truth", type=str, default="statistics/ground_truth.csv")
    args = parser.parse_args()

    relative_to_instance(args)

    # example usage:
    #   python relative_prop_to_instance_prop.py --config configs/relative/stratified_20bn.yaml