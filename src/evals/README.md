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
-----------------------------------------------------------------------------------------------------------------
Job                                               | Dataset                  | Metric                     |
-----------------------------------------------------------------------------------------------------------------
EURLEX(seed=461)                                  | long_context_eurlex      | EurlexMultilabelF1Score    | 66.66
EURLEX(seed=475)                                  | long_context_eurlex      | EurlexMultilabelF1Score    | 66.21
EURLEX(seed=480)                                  | long_context_eurlex      | EurlexMultilabelF1Score    | 66.21
-----------------------------------------------------------------------------------------------------------------



Collected Job Results: 

-------------------------------------------------------------
Task                                              |
-------------------------------------------------------------
EURLEX                                            | 66.36
-------------------------------------------------------------

Time: 73 mins