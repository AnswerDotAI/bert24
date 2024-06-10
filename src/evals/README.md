# Long Context Evaluation

We are currently considering the following three benchmarks for the long context evaluation suite:

### EURLEX
Source dataset: [pietrolesci/eurlex-57k](https://huggingface.co/datasets/pietrolesci/eurlex-57k)
Task: Multi-label classification (4271 labels)
Size: 57k examples (45k train, 6k validation, 6k test)
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