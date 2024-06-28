from streaming import MDSWriter
import huggingface_hub
from datasets import load_dataset, interleave_datasets
import argparse
import os
import tqdm
import yaml
import random
from transformers import AutoTokenizer, set_seed
import datasets
from streaming import StreamingDataset

from hf_to_mds import push_to_hub_incrementally
from data_utils import SOURCE_MAP, MDS_COLS_TOKENIZED, MDS_COLS_TEXT

set_seed(123456789)

TEST = False


def tokenize_and_write(writer, pool, tokenizer):
    global TEST
    total_tokens = 0
    pool_texts = [instance["text"] for instance in pool]
    if tokenizer is not None:
        texts_tokenized = tokenizer(pool_texts, truncation=False, return_tensors="np")

    for i, instance in enumerate(pool):
        instance_dict = {
            "text": instance["text"],
            "id": instance["id"],
        }
        if tokenizer is not None:
            instance_dict["input_ids"] = texts_tokenized["input_ids"][i].squeeze()
            instance_dict["attention_mask"] = texts_tokenized["attention_mask"][i].squeeze()
            del instance_dict["text"]
            total_tokens += len(instance_dict["input_ids"])

        if not TEST:
            print(instance_dict.keys())
            TEST = True

        writer.write(instance_dict)
    
    return total_tokens


def sample_dataset_from_config(args):
    assert os.path.isfile(args.config), f"Config file {in_fn} does not exist."
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    
    target_tokens = config["target_tokens"]
    sample_nums = {x["name"]: x["num_instances"] for x in config["sources"]}
    config_file_name = os.path.basename(args.config).split(".")[0]

    train_path = os.path.join(config_file_name, "train")
    validation_path = os.path.join(config_file_name, "validation")
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(validation_path, exist_ok=True)  

    if args.tokenizer is not None:
        print(f"Using tokenizer model {args.tokenizer}...")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    COLS = MDS_COLS_TOKENIZED if args.tokenizer is not None else MDS_COLS_TEXT
    with MDSWriter(out=os.path.join(config_file_name, "train"), columns=COLS, compression='zstd') as train_writer:
        with MDSWriter(out=os.path.join(config_file_name, "validation"), columns=COLS, compression='zstd') as validation_writer:
            for source in tqdm.tqdm(config["sources"], desc="Sources"):
                # pools are used to tokenize more than once, using args.tokenization_batch_size
                train_pool = []
                validation_pool = [] 

                source_name = source["name"]
                num_train = sample_nums[source_name]
                num_validation = max(1, round(num_train * args.validation_fraction))

                source_hf_repo = SOURCE_MAP[source_name]
                remote = f'hf://datasets/{source_hf_repo}/'
                dataset = StreamingDataset(remote=remote, shuffle=True, split=None, batch_size=1, cache_limit="50GB")

                for idx, instance in tqdm.tqdm(dataset):
                    if args.debug and idx > 100:
                        break

                    if idx < num_train:
                        train_pool.append(instance_dict)
                    else:
                        validation_pool.append(instance_dict)

                    if len(train_pool) > args.tokenization_batch_size:
                        num_tokens += tokenize_and_write(train_writer, train_pool, tokenizer)
                        train_pool = []

                    if len(validation_pool) > args.tokenization_batch_size:
                        tokenize_and_write(validation_writer, validation_pool, tokenizer)
                        validation_pool = []

            # any that didn't fit in the batch size
            if len(train_pool) > 0:
                num_tokens += tokenize_and_write(train_writer, train_pool, tokenizer)

            if len(validation_pool) > 0:
                tokenize_and_write(validation_writer, validation_pool, tokenizer)

    print(f"Finished writing with a total of {num_tokens} tokens.")
    # add the config file to the output directory with the total number of tokens
    with open(os.path.join(config_file_name, "config.yaml"), 'w') as file:
        config["total_tokens"] = num_tokens
        yaml.dump(config, file)

    # now push it to HF
    print(f"Pushing to HF...")
    upload_repo_path = f"orionweller/{config_file_name}"
    api = huggingface_hub.HfApi()
    repo_url = api.create_repo(
        upload_repo_path,
        repo_type="dataset",
        exist_ok=False,
    )
    push_to_hub_incrementally(
        upload_repo_path,
        config_file_name
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-t", "--tokenizer", type=str, required=False, default=None)
    parser.add_argument("-v", "--validation_fraction", type=float, default=0.01)
    parser.add_argument("-b", "--tokenization_batch_size", type=int, default=100000)
    parser.add_argument("-d", "--debug", action="store_true")
    args = parser.parse_args()

    sample_dataset_from_config(args)

    # example usage:
    #   python sample_dataset_from_config.py -c configs/instances/stratified_20bn.yaml -d