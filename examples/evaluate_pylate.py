from collections import defaultdict

import srsly
from pylate import evaluation, indexes, models, retrieve


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


eval_datasets = ["fiqa", "scifact", "nfcorpus", "trec-covid"]
model_name = "answerdotai/ModernBERT-base"
model_shortname = model_name.split("/")[-1]
lr = 8e-5
model_results = defaultdict(dict)
run_name = f"{model_shortname}-colbert-KD-{lr}"
output_dir = f"output/{model_shortname}/{run_name}"
model = models.ColBERT(
    model_name_or_path=f"{output_dir}/final",
    document_length=510,
)
for dataset in eval_datasets:
    for eval_dataset in eval_datasets:
        index = indexes.Voyager(index_name=dataset, override=True, M=200, ef_construction=500, ef_search=500)

        retriever = retrieve.ColBERT(index=index)

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
        print(f"{model_name} - {lr} - {eval_dataset}")
        print(evaluation_scores)
        print("-----------")
        model_results[eval_dataset] = evaluation_scores
    srsly.write_json(f"output/{model_shortname}/{model_shortname}_results.json", model_results)
