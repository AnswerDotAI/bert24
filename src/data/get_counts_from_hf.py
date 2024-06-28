import tqdm
import argparse
import random
import json
import datasets
import requests
import math
import os
import gzip
import numpy as np
import multiprocessing
import huggingface_hub
import glob
import tempfile

from datasets import load_dataset, Dataset, DatasetDict, interleave_datasets
from streaming.base.util import _merge_index_from_root, merge_index
from transformers import set_seed, AutoTokenizer
from streaming import MDSWriter, StreamingDataset

from huggingface_hub import HfFileSystem
from data_utils import ALL_REPOS


def get_counts_for_repo(repo, args):
    # download the root index.json only
    files_in_repo = [item.path for item in huggingface_hub.list_repo_tree(repo, repo_type="dataset")]
    if "index.json" not in files_in_repo:
        # it must be in the main folder
        main_folder = None
        repo_name_folder = repo.split("/")[-1]
        for file in files_in_repo:
            if file not in [".gitattributes"] and file.count(".") == 0:
                main_folder = file
                break
        main_json = f"{main_folder}/index.json"
        print(f"Did not find a root index.json, using {main_json}")
    else:
        main_json = "index.json"

    with tempfile.TemporaryDirectory() as tmp_cache_dir:
        root_folder = huggingface_hub.snapshot_download(repo_id=repo, allow_patterns=main_json, repo_type="dataset", cache_dir=tmp_cache_dir)
        dataset = StreamingDataset(local=os.path.join(root_folder, main_json.replace("index.json", "")), shuffle=False, split=None, batch_size=1)
        dataset_size = len(dataset)

        base_dir = f"datasets/{repo}"
        fs = HfFileSystem()
        try:
            size_of_folder = fs.du(base_dir, total=True, maxdepth=None, withdirs=True)
        except Exception as e:
            print(f"Error: {e}. Sleeping for 60 seconds and trying again")
            import time
            time.sleep(60)
            size_of_folder = fs.du(base_dir, total=True, maxdepth=None, withdirs=True)

        return {"dataset": repo, "size": size_of_folder / 1e9, "instances": dataset_size}


def get_counts(args):
    # read in all that have been already processed
    if os.path.exists("dataset_info.jsonl"):
        with open("dataset_info.jsonl", "r") as f:
            processed_datasets = set([json.loads(line)["dataset"] for line in f])
    else:
        processed_datasets = set()

    output_f = open("dataset_info.jsonl", "a")
    for repo in tqdm.tqdm(args.repos):
        if repo in processed_datasets:
            print(f"Skipping {repo} since it's already processed")
            continue
        print(f"Getting counts for {repo}")
        output_dict = get_counts_for_repo(repo, args)
        output_f.write(json.dumps(output_dict) + "\n")
        # flush it
        output_f.flush()

    output_f.close()

    # read in the info and sum and print
    total_size = 0
    total_instances = 0
    with open("dataset_info.jsonl", "r") as f:
        for line in f:
            info = json.loads(line)
            total_size += info["size"]
            total_instances += info["instances"]

    print(f"Total size: {total_size} GB")
    print(f"Total instances: {total_instances}")
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repos", type=str, nargs="+", help="List of repos to get counts for", default=None)
    args = parser.parse_args()

    # if repos is None use the default ALL_REPOS
    if args.repos is None:
        args.repos = ALL_REPOS

    get_counts(args)

    # example usage:
    #   python get_counts_from_hf.py