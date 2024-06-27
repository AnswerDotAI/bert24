# Training Data
This readme describes the training data and process used in BERT24.

## Existing Data
The following steps were run to create the data found on HuggingFace in `orionweller/*_mds_incremental` for Dolma splits and `orionweller/DATASET_NAME_mds` for non-Dolma dataset. The sampled ablation datasets are in TODO.


## Re-Generating the Data
0. Install dependencies from the `requirements.txt` and `requirements-data.txt`
1. To turn each Dolma subset into MDS format and upload them to Huggingface, run `python dolma_to_mds.py -s <DATASET_TAG> -r <HF_REPO_NAME>`. You can process all Dolma subsets with `bash dolma_mds_all.sh`.
3. Do the ones for the non-Dolma gathering using `hf_to_mds.py`. An example is `python hf_to_mds.py -r HuggingFaceFW/fineweb-edu -c sample-10BT -s train -u orionweller/fineweb-edu-10B`.
4. Gather the ratio of tokens per dataset with `python source_stats.py TODO`
5. Convert the relative ratio of configs into ones with number of instances with `relative_prop_to_instance_prop.py` or TODO bash file. 
6. Sample each dataset using the `sample_dataset_from_config.py TODO`


## Utilities

#### Gathering the size of HF MDS datasets
Thanks to the MDS format, it is simple and quick to get the number of instances and the size of the data. To do this, run `get_counts_from_hf.py` to get all counts, or use `get_counts_from_hf.py --repos "REPO_1 REPO_2 ... REPO_N"`. 

#### Calculating the tokens in a dataset
We can gather the number of total tokens and the tokens per instance using `python source_stats.py`. This is still in progress.  


## Config format
TODO: copy from old README

## Statistics
Various statistics can be found in the [statistics](path_to_stats) folder, such as the number of tokens per instance and size and instances numbers of the subsets.