# Training Data
This readme describes the training data and process used in BERT24.



## Re-Generating the Data
0. Install dependencies from the `requirements.txt` and `requirements-data.txt`
3. Turn a HF dataset into MDS via `hf_to_mds.py`.
6. Sample each dataset using the `sample_dataset_from_config.py TODO`


## Utilities

#### Gathering the size of HF MDS datasets
Thanks to the MDS format, it is simple and quick to get the number of instances and the size of the data. To do this, run `get_counts_from_hf.py` to get all counts, or use `get_counts_from_hf.py --repos "REPO_1 REPO_2 ... REPO_N"`. 

#### Calculating the tokens in a dataset
We can gather the number of total tokens and the tokens per instance using `python source_stats.py`. This is still in progress.  

## Config format
TODO: copy from old README
