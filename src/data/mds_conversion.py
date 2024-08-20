"""
This script allows conversion of mds-data, such as
* Compressing or decompressing a dataset
* Removing unnecessary fields
* Adapting `input_ids` to a more appropriate dtype
"""

import argparse
import os
import json
import numpy as np
from streaming.base.format.mds.writer import MDSWriter
from streaming.base.format import reader_from_json
from streaming.base.compression import decompress
from tqdm import tqdm

def maybe_decompress_shard(shard, delete_zip: bool = False):
    """
    If shard does not have decompressed data,
    this function decompresses the shard
    """
    raw_filename = os.path.join(shard.dirname, shard.split, shard.raw_data.basename)
    if not os.path.isfile(raw_filename):
        zip_filename = os.path.join(shard.dirname, shard.split, shard.zip_data.basename)
        data = open(zip_filename, 'rb').read()
        data = decompress(shard.compression, data)
        tmp_filename = raw_filename + '.tmp'
        with open(tmp_filename, 'wb') as out:
            out.write(data)
        os.rename(tmp_filename, raw_filename)

    # Maybe remove compressed to save space.
    if shard.zip_data is not None and delete_zip:
        zip_filename = os.path.join(shard.dirname, shard.split, shard.zip_data.basename)
        if os.path.exists(zip_filename):
            os.remove(zip_filename)

def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser()
    
    # Define the arguments
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data file')
    parser.add_argument('--read_split', type=str, required=True, help='Data split to read data from')
    parser.add_argument('--write_split', type=str, default=None, help='Data split to write data to')
    parser.add_argument('--dtype', type=str, default=None, help='Data type to convert the values of input_ids to')
    parser.add_argument('--columns_to_keep', type=str, nargs='+', default=None, help='List of columns to keep, if None, all columns will be kept')
    parser.add_argument('--decompress', action='store_true', help='If data in read_split should be be decompressed. Necessary if there is only compressed data in read_split')
    parser.add_argument('--delete_zip', action='store_true', help='Whether the compressed files should be kept after decompression or not')
    parser.add_argument('--compression', type=str, default=None, help='Compression type to use for the data to write')
    
    # Parse the arguments
    args = parser.parse_args()

    # Verify that the data path exists
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data path {args.data_path} does not exist.")

    if not args.write_split:
        assert args.decompress and not args.dtype and not args.columns_to_keep, "Only decompression is allowed if no write_split has been specified"

    # Convert args.dtype string into actual np.dtype if given
    dtype = np.dtype(args.dtype) if args.dtype else None

    # Load index file
    split_path = os.path.join(args.data_path, args.read_split)
    index_file_path = os.path.join(split_path, "index.json")
    obj = json.load(open(index_file_path))

    # Load columns from first shard to know what columns to write, and adapt if columns_to_keep is specified
    columns_to_write = {col_name: col_enc for col_name, col_enc in zip(obj["shards"][0]["column_names"], obj["shards"][0]["column_encodings"])}
    assert "input_ids" in columns_to_write, f"The data in the read path must have `input_ids` in its columns. Its columns: {columns_to_write.keys()}"
    if args.columns_to_keep:
        # Verify that each column in columns_to_keep is valid
        for column in args.columns_to_keep:
            assert column in columns_to_write, f"The given column to keep {column} must exist in the data in {args.read_split}"
        columns_to_write = {col_name: col_encoding for col_name, col_encoding in columns_to_write.items() if col_name in args.columns_to_keep}

    # read all shards
    shards = []
    for info in tqdm(obj['shards'], desc="Reading shards"):
        shard = reader_from_json(args.data_path, args.read_split, info)
        maybe_decompress_shard(shard, args.delete_zip)
        shards.append(shard)

    # potentially filter/alter shards and write the new ones
    if args.write_split:
        with MDSWriter(
            columns=columns_to_write, out=os.path.join(args.data_path, args.write_split), compression=args.compression
        ) as out:
            for shard in tqdm(shards, desc="Writing shards"):
                for sample in shard:
                    if dtype:
                        assert np.all(sample["input_ids"]<=np.iinfo(dtype).max), f"value in sample[input_ids] must not exceed {dtype} max"
                        sample["input_ids"] = sample["input_ids"].astype(dtype)
                    out.write({k: sample[k] for k in columns_to_write.keys()})
    
if __name__ == "__main__":
    main()