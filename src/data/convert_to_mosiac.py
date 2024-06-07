from streaming import MDSWriter
import huggingface_hub
from datasets import load_dataset
import argparse
import os
import tqdm


def convert_to_mosiac_format(args):

    # A dictionary mapping input fields to their data types
    columns = {
        'text': 'str',
        'id': 'str'
        # we can ignore the other fields, they're not used in training
    }

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with MDSWriter(out=args.output_dir, columns=columns, compression='zstd') as out:
        # loop over our data
        dataset = load_dataset(args.dataset, data_files={"train": "data/*"}, streaming=True)["train"]
        for idx, instance in tqdm.tqdm(enumerate(dataset)):
            out.write({
                "text": instance["text"],
                "id": instance["id"]
            })


    ### Example Usage
    ## Upload
    """
    az storage blob upload-batch --source PATH_TO_SRC --destination https://ACCOUNT_NAME.blob.core.windows.net/bert24usablations --auth-mode login --destination-path data/NAME_OF_DATA
    """
    ## Download and Usage
    """
    # be sure to export AZURE_ACCOUNT_NAME and AZURE_ACCOUNT_ACCESS_KEY
    from torch.utils.data import DataLoader
    from streaming import StreamingDataset

    # Remote path where full dataset is persistently stored
    # for this example, we'll use a single shard but otherwise you'd use the whole folder
    remote = 'azure://ACCOUNT_NAME.blob.core.windows.net/bert24usablations/data/dolma_20b_stratified/shard.01522.mds.zstd'

    # Local working dir where dataset is cached during operation
    local = '/tmp'

    # Create streaming dataset
    dataset = StreamingDataset(local=local, remote=remote, shuffle=True)

    # Let's see what is in sample #1337...
    sample = dataset[1337]
    text = sample['text']
    id = sample['id']

    # Create PyTorch DataLoader
    dataloader = DataLoader(dataset)
    """

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a dataset to the Mosaic Dataset Specification format.")
    parser.add_argument("-d", "--dataset", type=str, help="The dataset to convert.", default="answerdotai/dolma_20bn_stratified_sample")
    parser.add_argument("-o", "--output_dir", type=str, help="The directory to write the converted dataset to.", required=True)
    args = parser.parse_args()
    convert_to_mosiac_format(args)

