# Sampling Dolma

As of 5/28/24, the [Dolma dataset sample](https://huggingface.co/datasets/orionweller/dolma_20_percent_sample) has ~330bn tokens (see `source_stats.csv`).  This can be re-created by runnning:

```
python sample_from_dolma.py --repo_name orionweller/dolma_20_percent_sample --percentage 0.2
```

# Generating Mixtures

Then, to create different mixtures from this data, run

```
python dolma_mixtures.py --dolma_hf_path {{repo_name from above}} --config_fn {{.yaml file in --config_dir}}
```

## Config Docs

The config yaml file specifies how much of each dataset source to use, in addition to specifying an overall desired token count of the mixture. It follows the following structure:

```
sources:
  - name: cc_en_middle
    source_coefficient: 1
  - name: falcon
    source_coefficient: 1
  - name: cc_en_tail
    source_coefficient: 1
  - name: cc_en_head
    source_coefficient: 1
  - name: c4
    source_coefficient: 1
  - name: starcoder
    source_coefficient: 1
  - name: reddit
    source_coefficient: 1
  - name: pes2o
    source_coefficient: 1
  - name: arxiv
    source_coefficient: 1
  - name: wiki
    source_coefficient: 1
  - name: tulu_flan
    source_coefficient: 1
  - name: stackexchange
    source_coefficient: 1
  - name: cc_news
    source_coefficient: 1
  - name: open_web_math_train
    source_coefficient: 1
  - name: algebraic_stack_train
    source_coefficient: 1
  - name: books
    source_coefficient: 1
  - name: megawika
    source_coefficient: 1
source_coefficient_type: relative_proportion
target_tokens: 20000000000 # 20bn
hf_out_path: answerdotai/dolma_20bn_stratified_sample
```

For each source in the `--dolma_hf_path` dataset, you must specify a `source_coefficient`.

If `source_coefficient_type == "relative_proportion"`, `source_coefficient` specifies how much to up-weight / down-weight the source. `1` is a stratified sample of this data source, `2.0` doubles its weight relative to its original mixture. This coefficient must be `>= 0`.

If the `source_coefficient_type == "target_fraction"`, `source_coefficient` specifies what percentage of the final dataset the given source should be. In this case, **the `source_coefficient` must sum to 1 across the sources**. An example `target_fraction` is shown in `configs/example_target_fraction.yaml`.

`target_tokens` can be expressed as an absolute token count or as a fraction of the original (`--dolma_hf_path`), which is inferred if `target_tokens <= 1.0`.