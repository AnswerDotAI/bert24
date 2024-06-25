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

from dolma_urls import DOLMA_URLS


set_seed(11111111)

FILES_INFO = None

columns = {
        'text': 'str',
        'id': 'str'
        # we can ignore the other fields, they're not used in training
}


def folder_exists_in_huggingface_dataset(repo_name, folder_name):
    global FILES_INFO
    if FILES_INFO is None: # really only need to do this once, will help prevent 429's
        FILES_INFO = list(huggingface_hub.list_repo_tree(repo_name, repo_type="dataset"))
        print(len(FILES_INFO), "is the number of files in the dataset")

    for file_info in FILES_INFO:
        # print(file_info, file_info.path, folder_name)

        if os.path.isdir(file_info.path):
            print(f"Removing {file_info.path} since it's uploaded now")
            os.system(f"rm -rf {file_info.path}")

        # Check if the file path starts with the folder name
        if file_info.path == folder_name:
            return True

    return False


def download_data_from_url(url):
    local_filename = url.split('/')[-1]

    # if the local filename exists due to a previous download, use it
    if not os.path.exists(local_filename) and not os.path.exists(local_filename.replace(".gz", "")):
        print(f"Downloading {url}")
        os.system(f"wget {url}")
        # ungz the file from the command line as it's faster than python
        print(f"un-gunzip'ing {local_filename}")
        os.system(f"gunzip {local_filename}")

    # print(f"Reading in all of the data")
    return local_filename.replace(".gz", "")



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
        )
    
    os.system(f"rm -rf {local_path}")
    print(f"Pushed {local_path} to {repo_name}")
    

def sample_dolma(repo_name, debug, section):
    print(f"Sampling the data with repo {repo_name} and {args.section}")
    urls = [item for item in DOLMA_URLS if args.section in item]

    for url_i, url in enumerate(tqdm.tqdm(urls, desc="URLs")):
        # if url already exists in the dataset, skip it
        split_name = url.replace("https://olmo-data.org/dolma-v1_7/", "").split("/")[-1].replace(".json.gz", "").replace("-", "_")
        specific_name = split_name.split("/")[-1].split(".")[0]
        try:
            if folder_exists_in_huggingface_dataset(repo_name, split_name):
                print(f"Split {split_name} already exists in the dataset")
                continue
        except huggingface_hub.utils._errors.RepositoryNotFoundError:
            assert url_i == 0, f"Couldn't find repo at {url_i}: {url}" # first pass, the repo doesn't exist
            # make the dataset if it doesn't exist
            api = huggingface_hub.HfApi()
            repo_url = api.create_repo(
                repo_name,
                repo_type="dataset",
                exist_ok=False,
            )

        sampled_dataset_path = download_data_from_url(url)

        print(f"Writing to {split_name}")
        # if there exists an index.json file skip this step
        if not os.path.exists(f"{split_name}/index.json"):
            print(f"Creating {split_name}")
            with MDSWriter(out=split_name, columns=columns, compression='zstd') as train_writer:
                # if we're sampling everything, just return the entire dataset
                with open(sampled_dataset_path, "r") as fin:
                    for line in tqdm.tqdm(fin):
                        instance = json.loads(line)
                        train_writer.write(instance)
        else:
            print(f"Using existing MDS at {split_name}")


        # push the MDS to the hub
        # if there is not an index.json file, error and continue
        if not os.path.exists(f"{split_name}/index.json"):
            print(f"Error: {split_name}/index.json does not exist")
            continue

        print(f"Pushing...")
        push_to_hub_incrementally(
            args.repo_name,
            split_name
        )

        # remove the downloaded files, since we don't have too much space on the machine
        print(f"Removing {url.split('/')[-1].replace('.gz', '')}")
        os.system(f"rm {url.split('/')[-1].replace('.gz', '')}")

    # when all urls are done, add the final index.json by merging
    folders_in_dataset = huggingface_hub.list_repo_tree(args.repo_name, repo_type="dataset")
    folders_in_dataset = [folder.path for folder in folders_in_dataset if not folder.path.startswith("data")]

    # for each folder, download it and save it in a temp directory
    with tempfile.TemporaryDirectory() as tmp_cache_dir:
        # for folder in tqdm.tqdm(folders_in_dataset):
        #     if folder not in ["README.md", "data"]:
        #         try:
        #             huggingface_hub.snapshot_download(repo_id=args.repo_name, allow_patterns=f"{folder}/index.json", repo_type="dataset", cache_dir=tmp_cache_dir)
        #         except Exception as e:
        #             print(f"Failed to download {folder} with error: {e}")
        #             huggingface_hub.snapshot_download(repo_id=args.repo_name, allow_patterns=f"{folder}/index.json", repo_type="dataset", cache_dir=tmp_cache_dir)


        # # locate the root of the MDS files with glob, which is the folder above the index.json files
        # json_files = glob.glob(tmp_cache_dir + "/**/index.json", recursive=True)
        # root_folders = set([os.path.dirname(os.path.dirname(json_file)) for json_file in json_files])
        # assert len(root_folders) == 1, f"Expected one root folder, got {len(root_folders)}: {root_folders}"
        # root_folder = root_folders.pop()

        # now we have all the folders in the temp directory, we can combine them
        # use Mosiac's `merge_index` to combine the indexes
        all_files = huggingface_hub.list_repo_tree(args.repo_name, repo_type="dataset", recursive=True)
        string_files = [item.path for item in all_files]
        json_files = [item for item in string_files if item.endswith("index.json") and item != "index.json"]
        # if "index.json" not in string_files:
        #     print(f"Merging at root folder: {root_folder}")
        #     _merge_index_from_root(root_folder)

        #     # now push the new index.json file up to the hub
        #     print(f"Uploading to {args.repo_name}")
        #     api = huggingface_hub.HfApi()
        #     api.upload_file(
        #         path_or_fileobj=root_folder + "/index.json",
        #         path_in_repo="index.json",
        #         repo_id=args.repo_name,
        #         repo_type="dataset",
        #     )

        # now to not get confused, let's change the olds ones
        fs = HfFileSystem()
        base_dir = f"datasets/{args.repo_name}"
        # for every json file move it to index_old.json
        print(f"Moving old index.json files to index_old.json")
        for json_file in tqdm.tqdm(json_files):
            last_two_dirs = "/".join(json_file.split("/")[-2:])
            fs.mv(f"{base_dir}/{last_two_dirs}", f"{base_dir}/{last_two_dirs.replace('index.json', 'index_old.json')}")      

            
        # download the root folder only
        # huggingface_hub.snapshot_download(repo_id=args.repo_name, allow_patterns=f"index.json", repo_type="dataset", cache_dir=tmp_cache_dir)
        # print(f"Getting details about the dataset")
        # size_of_folder = fs.du(base_dir)
        # dataset = StreamingDataset(local=root_folder, shuffle=False, split=None, batch_size=1)
        # with open("dataset_info.jsonl", "a") as f:
        #     f.write(json.dumps({"dataset": args.repo_name, "size": size_of_folder / 1e9, "instances": len(dataset)}) + "\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-r", "--repo_name", type=str, required=True)
    parser.add_argument("-s", "--section", type=str, required=True)
    args = parser.parse_args()

    sample_dolma(args.repo_name, args.debug, args.section)

    # example usage: 
    # for falcon: python dolma_to_mds.py -s "falcon" -r "orionweller/refinedweb_mds_incremental"
