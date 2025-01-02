# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

from datasets import load_dataset
from pylate import losses, models, utils
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)


def main():
    # Load the datasets required for knowledge distillation (train, queries, documents)
    train = load_dataset(
        path="lightonai/ms-marco-en-bge",
        name="train",
    )

    queries = load_dataset(
        path="lightonai/ms-marco-en-bge",
        name="queries",
    )

    documents = load_dataset(
        path="lightonai/ms-marco-en-bge",
        name="documents",
    )

    # Set the transformation to load the documents/queries texts using the corresponding ids on the fly
    train.set_transform(
        utils.KDProcessing(queries=queries, documents=documents).transform,
    )

    # Define the base model, training parameters, and output directory
    num_train_epochs = 1
    lr = 8e-5
    batch_size = 16
    accum_steps = 1
    model_name = "answerdotai/ModernBERT-base"
    model_shortname = model_name.split("/")[-1]

    # Set the run name for logging and output directory
    run_name = f"{model_shortname}-colbert-KD-{lr}"
    output_dir = f"output/{model_shortname}/{run_name}"

    # Initialize the ColBERT model from the base model
    model = models.ColBERT(model_name_or_path=model_name)

    # Configure the training arguments (e.g., epochs, batch size, learning rate)
    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=True,  # Set to True if you have a GPU that supports BF16
        run_name=run_name,
        logging_steps=10,
        learning_rate=lr,
        gradient_accumulation_steps=accum_steps,
        warmup_ratio=0.05,
    )

    # Use the Distillation loss function for training
    train_loss = losses.Distillation(model=model)

    # Initialize the trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train,
        loss=train_loss,
        data_collator=utils.ColBERTCollator(tokenize_fn=model.tokenize),
    )

    # Start the training process
    trainer.train()

    model.save_pretrained(f"{output_dir}/final")


if __name__ == "__main__":
    main()
