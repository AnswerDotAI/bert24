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

from data_utils import MDS_COLS_TEXT


set_seed(11111111)

FILES_INFO = None



def push_to_hub_incrementally(repo_name, local_path):
    api = huggingface_hub.HfApi()
    # Upload all the content from the local folder to your remote Space.
    # By default, files are uploaded at the root of the repo
    print(f"Uploading {local_path} to {repo_name}/{local_path}")
    try:
        api.upload_folder(
            folder_path=local_path,
            repo_id=repo_name,
            path_in_repo=local_path,
            repo_type="dataset",
            multi_commits=True,
            multi_commits_verbose=True,
        )
    except Exception as e:
        print(e)
        import time
        time.sleep(30)
        print(f"Error uploading {local_path} to {repo_name}, trying again")
        api.upload_folder(
            folder_path=local_path,
            repo_id=repo_name,
            path_in_repo=local_path,
            repo_type="dataset",
            multi_commits=True,
            multi_commits_verbose=True,
        )
    
    os.system(f"rm -rf {local_path}")
    print(f"Pushed {local_path} to {repo_name}")
    

def sample_hf(upload_repo, repo_name, split_name, config_name):
    print(f"Sampling the data with repo {repo_name} and {split_name} and {config_name} and pushing to {upload_repo}...")

    if config_name is not None and split_name:
        dataset = load_dataset(repo_name, config_name, streaming=True)[split_name]
    elif config_name is not None:
        dataset = load_dataset(repo_name, config_name, streaming=True)
    elif split_name is not None:
        dataset = load_dataset(repo_name, streaming=True)[split_name]
    else:
        dataset = load_dataset(repo_name, streaming=True)


    try:
        files = list(huggingface_hub.list_repo_tree(upload_repo, repo_type="dataset"))
        files = [file.path for file in files]
    except huggingface_hub.utils._errors.RepositoryNotFoundError:
        # make the dataset if it doesn't exist
        api = huggingface_hub.HfApi()
        repo_url = api.create_repo(
            upload_repo,
            repo_type="dataset",
            exist_ok=False,
        )
        files = []

       
    if "data/index.json" not in files:
        config_name_dirsafe = config_name.replace("/", "-") if config_name is not None else "default"
        split_name_dirsafe = split_name.replace("/", "-") if split_name is not None else "default"
        tmp_cache_dir = f"{repo_name.replace('/', '-')}---{split_name_dirsafe}---{config_name_dirsafe}"
        if not os.path.isfile(os.path.join(tmp_cache_dir, "index.json")):
            print(f"Writing to MDS...")
            with MDSWriter(out=tmp_cache_dir, columns=MDS_COLS_TEXT, compression='zstd') as train_writer:
                for item in tqdm.tqdm(dataset):
                    train_writer.write(item)

        print(f"Pushing to HF...")
        dataset = StreamingDataset(local=tmp_cache_dir, shuffle=False, split=None, batch_size=1)
        num_instances = len(dataset)
        push_to_hub_incrementally(
            upload_repo,
            tmp_cache_dir
        )
    else:
        print(f"Using existing MDS written out")

    fs = HfFileSystem()
    size_of_folder = fs.du(f"datasets/{upload_repo}")
    with open("dataset_info.jsonl", "a") as f:
        f.write(json.dumps({"dataset": upload_repo, "split_name": split_name, "config_name": config_name, "size": size_of_folder / 1e9, "instances": num_instances}) + "\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--upload_repo", type=str, required=True)
    parser.add_argument("-r", "--repo_name", type=str, required=True)
    parser.add_argument("-s", "--repo_split", type=str, required=False)
    parser.add_argument("-c", "--repo_config", type=str, required=False)
    args = parser.parse_args()

    sample_hf(args.upload_repo, args.repo_name, args.repo_split, args.repo_config)

    # example usage:
    #   python hf_to_mds.py -r HF_DATASET -c CONFIG -s SPLIT -u HF_SAVE_PATH 
