# Long Context Evaluation

We are currently considering the following three benchmarks for the long context evaluation suite:

### EURLEX
Source dataset: [coastalcph/lex_glue](https://huggingface.co/datasets/coastalcph/lex_glue) (eurlex Subset)
Task: Multi-label classification (100 labels)
Size: 57k examples (55k train, 5k validation, 5k test)
Context length: Mean -> 595 tokens, Std -> 416 tokens, Max -> 5.3k tokens*
Domain: legislative documents
Metric: micro-f1

## Ultrafeedback
Source dataset: [rbiswasfc/ultrafeedback-binary-classification](https://huggingface.co/datasets/rbiswasfc/ultrafeedback-binary-classification) derived from [argilla/ultrafeedback-binarized-preferences-cleaned](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned)
Task: Binary classification
Size: 60k examples (48k train, 12k test)
Context length: Mean -> 820 tokens, Std -> 540 tokens, Max -> 6.1k tokens*
Domain: diverse set of queries
Metric: AUC

### SCROLLS (Quality Subset)
Source dataset: [rbiswasfc/scrolls-quality-mcq](https://huggingface.co/datasets/rbiswasfc/scrolls-quality-mcq) derived from [tau/sled](https://huggingface.co/datasets/tau/scrolls)
Task: MCQ with 4 choices
Size: 6.7k examples (2.5k train, 2.1k validation, 2.1k test)
Context length: Mean -> 7.4k tokens, Std -> 2.3k tokens, Max -> 11.6k tokens*
Domain: stories and articles sourced from Project Gutenberg, the Open American National Corpus etc.
Metric: Accuracy

*Context token stats are as per `bclavie/bert24_32k_tok_llama2` tokenizer.

# Run
Long context evaluation suite can be run using the following command from the root directory of the repository:
```bash
python glue.py yamls/baselines/deberta-v3-long-context.yaml
```

# Results
## Eurlex (micro-f1)
Training Time: ~200 mins

```
-----------------------------------------------------------------------------------------------------------------
Job                                               | Dataset                  | Metric                     |
-----------------------------------------------------------------------------------------------------------------
EURLEX(seed=461)                                  | long_context_eurlex      | EurlexMultilabelF1Score    | 72.43
EURLEX(seed=475)                                  | long_context_eurlex      | EurlexMultilabelF1Score    | 72.60
EURLEX(seed=480)                                  | long_context_eurlex      | EurlexMultilabelF1Score    | 71.88
-----------------------------------------------------------------------------------------------------------------



Collected Job Results: 

-------------------------------------------------------------
Task                                              |
-------------------------------------------------------------
EURLEX                                            | 72.30
-------------------------------------------------------------
```

## Ablation Evals

For hf-bert:
```bash
python ablation_eval.py yamls/ablations/hf-bert-ablation-eval.yaml
```

For flex-bert:
```bash
python ablation_eval.py yamls/ablations/flex-bert-ablation-eval.yaml
```

For mosaic-bert:
```bash
python ablation_eval.py yamls/ablations/mosaic-bert-ablation-eval.yaml
```

## Results

**flex-bert pre-trained (512_flex-bert-base-uncased_dolma_rope_postnorm_layernorm_geglu-1e3)**
```
-----------------------------------------------------------------------------------------------------------------
Job                                               | Dataset                  | Metric                     |
-----------------------------------------------------------------------------------------------------------------
MNLI(seed=23)                                     | glue_mnli_mismatched     | MulticlassAccuracy         | 84.53
EURLEX(seed=23)                                   | long_context_eurlex      | EurlexMultilabelF1Score    | 71.85
BOOLQ(seed=23)                                    | superglue_boolq          | MulticlassAccuracy         | 78.35
WIC(seed=23)                                      | superglue_wic            | MulticlassAccuracy         | 65.99
-----------------------------------------------------------------------------------------------------------------
```

**hf-bert**
```
-----------------------------------------------------------------------------------------------------------------
Job                                               | Dataset                  | Metric                     |
-----------------------------------------------------------------------------------------------------------------
MNLI(seed=23)                                     | glue_mnli_mismatched     | MulticlassAccuracy         | 85.24
EURLEX(seed=23)                                   | long_context_eurlex      | EurlexMultilabelF1Score    | 70.29
BOOLQ(seed=23)                                    | superglue_boolq          | MulticlassAccuracy         | 79.66
WIC(seed=23)                                      | superglue_wic            | MulticlassAccuracy         | 65.67
-----------------------------------------------------------------------------------------------------------------
```

WIC & BoolQ runs use the MNLI fine-tuned checkpoints.


## Hard Evals
For flex-bert:
```bash
python ablation_eval.py yamls/ablations/flex-bert-ablation-eval-hard.yaml
```

## MLMMLU
### Flex-bert:
```bash
python ablation_eval.py yamls/ablations/flex-bert-ablation-mlmmlu.yaml
```
```
-----------------------------------------------------------------------------------------------------------------
Job                                               | Dataset                  | Metric                     |
-----------------------------------------------------------------------------------------------------------------
MLMMLU(seed=24)                                   | mlmmlu_amateur           | MulticlassAccuracy         | 34.91
MLMMLU(seed=24)                                   | mlmmlu_semipro           | MulticlassAccuracy         | 27.70
MLMMLU(seed=61)                                   | mlmmlu_amateur           | MulticlassAccuracy         | 33.75
MLMMLU(seed=61)                                   | mlmmlu_semipro           | MulticlassAccuracy         | 26.75
MLMMLU(seed=24)                                   | mlmmlu_rookie            | MulticlassAccuracy         | 34.33
MLMMLU(seed=24)                                   | mlmmlu_reserve           | MulticlassAccuracy         | 30.56
MLMMLU(seed=61)                                   | mlmmlu_rookie            | MulticlassAccuracy         | 34.93
MLMMLU(seed=61)                                   | mlmmlu_reserve           | MulticlassAccuracy         | 31.35
-----------------------------------------------------------------------------------------------------------------
```

For hf-bert:
```bash
python ablation_eval.py yamls/ablations/hf-bert-ablation-mlmmlu.yaml
```
```
-----------------------------------------------------------------------------------------------------------------
Job                                               | Dataset                  | Metric                     |
-----------------------------------------------------------------------------------------------------------------
MLMMLU(seed=24)                                   | mlmmlu_amateur           | MulticlassAccuracy         | 37.85
MLMMLU(seed=24)                                   | mlmmlu_semipro           | MulticlassAccuracy         | 28.13
MLMMLU(seed=61)                                   | mlmmlu_amateur           | MulticlassAccuracy         | 35.45
2MLMMLU(seed=24)                                   | mlmmlu_rookie            | MulticlassAccuracy         | 45.73
MLMMLU(seed=24)                                   | mlmmlu_reserve           | MulticlassAccuracy         | 38.56
MLMMLU(seed=61)                                   | mlmmlu_rookie            | MulticlassAccuracy         | 48.37
MLMMLU(seed=61)                                   | mlmmlu_reserve           | MulticlassAccuracy         | 39.77
-----------------------------------------------------------------------------------------------------------------
```

For mosaic-bert:
```bash
python ablation_eval.py yamls/ablations/mosaic-bert-ablation-mlmmlu.yaml
```

```
-----------------------------------------------------------------------------------------------------------------
Job                                               | Dataset                  | Metric                     |
-----------------------------------------------------------------------------------------------------------------
MLMMLU(seed=24)                                   | mlmmlu_amateur           | MulticlassAccuracy         | 30.52
MLMMLU(seed=24)                                   | mlmmlu_semipro           | MulticlassAccuracy         | 22.91
MLMMLU(seed=61)                                   | mlmmlu_amateur           | MulticlassAccuracy         | 31.14
MLMMLU(seed=61)                                   | mlmmlu_semipro           | MulticlassAccuracy         | 23.29
MLMMLU(seed=24)                                   | mlmmlu_rookie            | MulticlassAccuracy         | 32.10
MLMMLU(seed=24)                                   | mlmmlu_reserve           | MulticlassAccuracy         | 29.13
MLMMLU(seed=61)                                   | mlmmlu_rookie            | MulticlassAccuracy         | 31.52
MLMMLU(seed=61)                                   | mlmmlu_reserve           | MulticlassAccuracy         | 28.86
-----------------------------------------------------------------------------------------------------------------
```



NB: There maybe slight differences in the observed scores for mosaic-bert, likely due to fa2.


# Generate config for ablations

## Create cofig by specifying checkpoint & config path
```
python generate_eval_config_from_checkpoint.py \
--checkpoint /home/shared/data-ablations/checkpoints/1024_mosaic-bert-base-uncased_dolma_1e-3_20bn_cc_high_quality \
--train_config /home/shared/data-ablations/configs/1024_mosaic-bert-base-uncased_20bn_cc_high_quality_1e-3.yaml \
```

## Create config from the matching wandb run
```
python generate_eval_config_from_checkpoint.py \
--checkpoint /home/shared/data-ablations/checkpoints/1024_mosaic-bert-base-uncased_dolma_1e-3_20bn_cc_high_quality \
--wandb_project bert24-data-ablations
```

## Create config from the matching wandb run & add wandb tracking
```
python generate_eval_config_from_checkpoint.py \
--checkpoint /home/shared/data-ablations/checkpoints/1024_mosaic-bert-base-uncased_dolma_1e-3_20bn_cc_high_quality \
--wandb_project bert24-data-ablations \
--track_run
```

## Launch the ablation job
```bash
python ablation_eval.py yamls/ablations/1024_mosaic-bert-base-uncased_dolma_1e-3_20bn_cc_high_quality_evaluation.yaml
```


## Launch abalations for sub-directories of a given path
- Each subdir needs to contain a checkpoint named "latest-rank0.pt"
- Config file should be stored together with the checkpoint (<sub_dir_name>.yaml)
- If not, the script will try to find a matching wandb run in `bert24/bert24` project.
- If the above fails, then the job will be skipped.

```bash
./example_eval_checkpoints.sh /home/shared/data-ablations/checkpoints
```