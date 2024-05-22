from datasets import load_dataset, Dataset, DatasetDict
import datasets
from dolma_urls import DOLMA_URLS
import tqdm
import argparse
import random
import json
from transformers import set_seed
import requests
import os
import gzip
import numpy as np
import huggingface_hub

set_seed(11111111)


KEEP_COLUMNS = ["id", "text", "added", "created", "source"]
PARTITIONS_TO_KEEP_ALL = ["wiki"]


def keep_split_name(split_name):
    for partition in PARTITIONS_TO_KEEP_ALL:
        if partition in split_name:
            return True
    return False
    

def folder_exists_in_huggingface_dataset(repo_name, folder_name):
    files_info = huggingface_hub.list_repo_tree(repo_name, repo_type="dataset")
    for file_info in files_info:
        # Check if the file path starts with the folder name
        if file_info.path.startswith(folder_name):
            return True
    return False


def sample_data_from_url(url, sample_fraction=0.2):
    local_filename = url.split('/')[-1]

    # if the local filename exists due to a previous download, use it
    if not os.path.exists(local_filename) and not os.path.exists(local_filename.replace(".gz", "")):
        print(f"Downloading {url}")
        response = requests.head(url)
        total_size = int(response.headers.get('content-length', 0))
        local_filename = url.split('/')[-1]
        chunk_size = 1024 * 1024  
        # Download the file with a progress bar
        with requests.get(url, stream=True) as r, open(local_filename, 'wb') as f, tqdm.tqdm(
                total=total_size, unit='B', unit_scale=True, unit_divisor=1024, desc=local_filename
        ) as progress_bar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                progress_bar.update(len(chunk))

        # ungz the file from the command line as it's faster than python
        print(f"gunzip'ing {local_filename}")
        os.system(f"gunzip {local_filename}")

    sampled_data = []
    # load the file without the gz
    print(local_filename)
    print(f"Sampling {sample_fraction} of the data")
    with open(local_filename.replace(".gz", ""), "r") as fin:
        for i, line in enumerate(tqdm.tqdm(fin, desc="Sampling")):
            if random.random() < sample_fraction:
                inst = json.loads(line)
                inst = {k: inst[k] for k in KEEP_COLUMNS}
                assert len(inst) == len(KEEP_COLUMNS), f"Instance {i} has missing columns: {inst}"
                sampled_data.append(inst)

    return sampled_data


def push_to_hub_incrementally(repo_name, split_name, data):
    existing_dataset = Dataset.from_list(data)
    dataset_dict = DatasetDict({split_name: existing_dataset})

    # Save the dataset to a local directory
    local_path = os.path.join("tmp", split_name)
    os.makedirs(local_path, exist_ok=True)
    dataset_dict.save_to_disk(local_path, max_shard_size="5GB")

    # Push to hub with a specified folder
    api = huggingface_hub.HfApi()
    # Upload all the content from the local folder to your remote Space.
    # By default, files are uploaded at the root of the repo
    print(f"Uploading {local_path} to {repo_name}")
    api.upload_folder(
        folder_path=local_path,
        repo_id=repo_name,
        repo_type="dataset",
    )
    # upload the default readme from `readme.md`
    print(f"Uploading README.md to {repo_name}")
    api.upload_file(
        path_or_fileobj="README.md",
        path_in_repo="README.md",
        repo_id=repo_name,
        repo_type="dataset",
    )
    
    os.system(f"rm -rf tmp")
    print(f"Pushed {split_name} to {repo_name}")
    

def sample_dolma(percentage: float, repo_name: str, debug: bool = False):
    if debug: 
        urls = DOLMA_URLS[:10]
    else:
        urls = DOLMA_URLS

    for url_i, url in enumerate(tqdm.tqdm(urls, desc="URLs")):
        # if url already exists in the dataset, skip it
        split_name = url.replace("https://olmo-data.org/dolma-v1_7/", "").split("/")[-1].replace(".json.gz", "").replace("-", "_")
        try:
            if folder_exists_in_huggingface_dataset(repo_name, split_name):
                print(f"Split {split_name} already exists in the dataset")
                continue
        except huggingface_hub.utils._errors.RepositoryNotFoundError:
            assert url_i == 0, f"Couldn't find repo at {url_i}: {url}" # first pass, the repo doesn't exist
            # make the dataset if it doesn't exist
            # start by making a generic file dataset
            DatasetDict({"creation": Dataset.from_list([])}).push_to_hub(repo_name, private=False, token=None, max_shard_size="5GB")

        sampled_dataset = sample_data_from_url(url, sample_fraction=percentage if not keep_split_name(split_name) else 1.0)
        print(f"Pushing to split {split_name} with {len(sampled_dataset)} samples")

        push_to_hub_incrementally(
            repo_name=args.repo_name,
            split_name=split_name,
            data=sampled_dataset,
        )

        # remove the downloaded files, since we don't have too much space on the machine
        print(f"Removing {url.split('/')[-1].replace('.gz', '')}")
        os.system(f"rm {url.split('/')[-1].replace('.gz', '')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--percentage", type=float, default=0.2)
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-r", "--repo_name", type=str, required=True)
    args = parser.parse_args()

    sample_dolma(args.percentage, args.repo_name, args.debug)

    # example usage: python -u sample_from_dolma.py --repo_name orionweller/dolma_20_percent_sample --percentage 0.2 > dolma_sample.log 2>&1