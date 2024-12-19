# ModernBERT retrieval boilerplates

In this folder, you can find different boilerplates to train and evaluate retrieval models using ModernBERT as the backbone, with [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) for single vector retrieval (DPR) and [PyLate](https://github.com/lightonai/pylate) for multi vector retrieval (ColBERT).

You can use ```train_st.py``` and ```train_pylate.py``` to train a single vector model using contrastive learning on [MS-MARCO with mined hard negatives](https://huggingface.co/datasets/sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1) and a multi vector model using knowledge distillation on [MS-MARCO with teacher weights from bge-reranker-v2-m3](https://huggingface.co/datasets/lightonai/ms-marco-en-bge) respectively. Alternatively, ```train_st_gooaq.py``` provides a training script for training a single vector model on the [GooAQ](https://huggingface.co/datasets/sentence-transformers/gooaq) question-answer dataset.

You can launch training on multiple GPUs by using ```accelerate launch --num_processes num_gpu train_st.py```

You can then run ```python evaluate_st.py``` or ```python evaluate_pylate.py``` to evaluate the trained models on BEIR datasets.