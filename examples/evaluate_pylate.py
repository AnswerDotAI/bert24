# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

from collections import defaultdict

import srsly

from pylate import evaluation, indexes, models, retrieve

eval_datasets = ["scifact", "nfcorpus", "fiqa", "trec-covid"]
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

for eval_dataset in eval_datasets:
    index = indexes.Voyager(index_name=eval_dataset, override=True, M=200, ef_construction=500, ef_search=500)

    retriever = retrieve.ColBERT(index=index)

    documents, queries, qrels = evaluation.load_beir(
        dataset_name=eval_dataset,
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

    scores = retriever.retrieve(queries_embeddings=queries_embeddings, k=100)

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
