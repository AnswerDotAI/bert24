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
For flex-bert:
```bash
python ablation_eval.py yamls/ablations/flex-bert-ablation-mlmmlu.yaml
```

For hf-bert:
```bash
python ablation_eval.py yamls/ablations/hf-bert-ablation-mlmmlu.yaml
```