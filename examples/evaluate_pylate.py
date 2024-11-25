import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--models", type=str, nargs="+", required=True, help="Models to use")
parser.add_argument("--lrs", type=str, nargs="+", default=["8e-5"], help="Learning rates to use")
parser.add_argument("--datasets", type=str, nargs="+", required=True, help="Datasets to use")
args = parser.parse_args()


from collections import defaultdict
from typing import Literal

import srsly
from pylate import evaluation, indexes, models, retrieve

VALID_DATASETS = ["mldr", "trec-covid", "fiqa", "scifact", "nfcorpus"]

model_names = args.models
lrs = [float(x) for x in args.lrs]
if args.datasets[0] == "all":
    eval_datasets = VALID_DATASETS
elif args.datasets[0] == "beir":
    eval_datasets = ["trec-covid", "fiqa", "scifact", "nfcorpus"]
elif args.datasets[0] == "nanobeir":
    raise NotImplementedError
elif args.datasets[0] == "long_context":
    eval_datasets = ["mldr"]
else:
    assert all([x in VALID_DATASETS for x in args.datasets])
    eval_datasets = args.datasets

ALL_RESULTS = defaultdict(lambda: defaultdict(dict))


def fetch_most_recent_checkpoint(model_path: str):
    from pathlib import Path

    output_dir = Path(model_path)
    if not output_dir.is_dir():
        return model_path

    checkpoints = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]

    if not checkpoints:
        return model_path

    latest_checkpoint = max(checkpoints, key=lambda d: int(d.name.split("-")[1]))
    print(f"Latest checkpoint: {latest_checkpoint}")
    return str(latest_checkpoint)


def eval_model(model_path: str, dataset: Literal["scifact", "nfcorpus"]):
    document_length = 8190
    if (
        any(x in model_path.lower() for x in ["deberta", "bert-base", "bert-large", "roberta"])
        and "mldr" not in dataset
    ):
        document_length = 510
    model = models.ColBERT(
        model_name_or_path=fetch_most_recent_checkpoint(model_path),
        document_length=document_length,
        trust_remote_code=True,
    )

    index = indexes.Voyager(
        index_name=f"new_evals_gpu{args.gpu}", override=True, M=200, ef_construction=500, ef_search=500
    )

    retriever = retrieve.ColBERT(index=index)

    # Download the SciFact dataset
    if "mldr" in dataset:
        documents, queries, qrels = evaluation.load_custom_dataset(
            path="/opt/home/bert24/pylate/data/mldr_subsample_15k",
            split="dev",
        )
    else:
        documents, queries, qrels = evaluation.load_beir(
            dataset_name=dataset,
            split="test",
        )

    batch_size = 500

    documents_embeddings = model.encode(
        sentences=[document["text"] for document in documents],
        batch_size=batch_size,
        is_query=False,
        show_progress_bar=True,
    )

    index.add_documents(
        documents_ids=[document["id"] for document in documents],
        documents_embeddings=documents_embeddings,
    )

    queries_embeddings = model.encode(
        sentences=queries,
        is_query=True,
        show_progress_bar=True,
        batch_size=16,
    )

    scores = retriever.retrieve(queries_embeddings=queries_embeddings, k=20, k_token=80, batch_size=2)

    evaluation_scores = evaluation.evaluate(
        scores=scores,
        qrels=qrels,
        queries=queries,
        metrics=["ndcg@10"],
    )
    return evaluation_scores


if __name__ == "__main__":
    for model_name in model_names:
        model_results = defaultdict(dict)
        model_shortname = model_name.split("/")[-1]
        for lr in lrs:
            # Set the run name for logging and output directory
            run_name = f"{model_shortname}-colbert-KD-{lr}"
            output_dir = f"output/{model_shortname}/{run_name}"

            for eval_dataset in eval_datasets:
                results = eval_model(output_dir, eval_dataset)
                print(f"{model_name} - {lr} - {eval_dataset}")
                print(results)
                print("-----------")
                model_results[lr][eval_dataset] = results
                ALL_RESULTS[model_name][lr][eval_dataset] = results
        srsly.write_json(f"output/{model_shortname}/{model_shortname}_results.json", model_results)
