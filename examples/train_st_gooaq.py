# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

import argparse

from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import NanoBEIREvaluator
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

def main():
    # parse the lr & model name
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--model_name", type=str, default="answerdotai/ModernBERT-base")
    args = parser.parse_args()
    lr = args.lr
    model_name = args.model_name
    model_shortname = model_name.split("/")[-1]

    # 1. Load a model to finetune
    model = SentenceTransformer(model_name)

    # 2. Load a dataset to finetune on
    dataset = load_dataset("sentence-transformers/gooaq", split="train")
    dataset_dict = dataset.train_test_split(test_size=1_000, seed=12)
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["test"]

    # 3. Define a loss function
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=128)  # Increase mini_batch_size if you have enough VRAM

    run_name = f"{model_shortname}-gooaq-{lr}"
    # 4. (Optional) Specify training arguments
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=f"output/{model_shortname}/{run_name}",
        # Optional training parameters:
        num_train_epochs=1,
        per_device_train_batch_size=2048,
        per_device_eval_batch_size=2048,
        learning_rate=lr,
        warmup_ratio=0.05,
        fp16=False,  # Set to False if GPU can't handle FP16
        bf16=True,  # Set to True if GPU supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # (Cached)MultipleNegativesRankingLoss benefits from no duplicates
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        logging_steps=10,
        run_name=run_name,  # Used in `wandb`, `tensorboard`, `neptune`, etc. if installed
    )

    # 5. (Optional) Create an evaluator & evaluate the base model
    dev_evaluator = NanoBEIREvaluator(dataset_names=["NQ", "MSMARCO"])
    dev_evaluator(model)

    # 6. Create a trainer & train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=dev_evaluator,
    )
    trainer.train()

    # 7. (Optional) Evaluate the trained model on the evaluator after training
    dev_evaluator(model)

    # 8. Save the model
    model.save_pretrained(f"output/{model_shortname}/{run_name}/final")

    # 9. (Optional) Push it to the Hugging Face Hub
    model.push_to_hub(run_name, private=False)

if __name__ == "__main__":
    main()