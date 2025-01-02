# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0
from datasets import load_dataset
from pylate import evaluation, losses, models, utils
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)


def main():
    model_name = "answerdotai/ModernBERT-base"  # Choose the pre-trained model you want to use as base
    model_shortname = model_name.split("/")[-1]

    # Define model parameters for contrastive training
    batch_size = 64  # Larger batch size often improves results, but requires more memory
    lr = 3e-6
    num_train_epochs = 5  # Adjust based on your requirements

    # Set the run name for logging and output directory
    run_name = f"{model_shortname}-colbert-contrastive-{lr}"
    output_dir = f"output/{model_shortname}/{run_name}"

    # 1. Here we define our ColBERT model. If not a ColBERT model, will add a linear layer to the base encoder.
    model = models.ColBERT(model_name_or_path=model_name)
    # ModernBERT is compiled by default, so there is no need to call compile explicitly and it actually breaks the model
    # model = torch.compile(model)

    # Load dataset
    dataset = load_dataset("sentence-transformers/msmarco-bm25", "triplet", split="train")
    # Split the dataset (this dataset does not have a validation set, so we split the training set)
    splits = dataset.train_test_split(test_size=0.01)
    train_dataset = splits["train"]
    eval_dataset = splits["test"]

    # Define the loss function
    train_loss = losses.Contrastive(model=model)

    # Initialize the evaluator
    dev_evaluator = evaluation.ColBERTTripletEvaluator(
        anchors=eval_dataset["query"],
        positives=eval_dataset["positive"],
        negatives=eval_dataset["negative"],
    )

    # Configure the training arguments (e.g., batch size, evaluation strategy, logging steps)
    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=True,  # Set to True if you have a GPU that supports BF16
        run_name=run_name,  # Will be used in W&B if `wandb` is installed
        learning_rate=lr,
        logging_steps=100,
    )

    # Initialize the trainer for the contrastive training
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=train_loss,
        evaluator=dev_evaluator,
        data_collator=utils.ColBERTCollator(model.tokenize),
    )
    # Start the training process
    trainer.train()

    model.save_pretrained(f"{output_dir}/final")


if __name__ == "__main__":
    main()
