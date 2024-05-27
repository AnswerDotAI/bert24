from collections import defaultdict
from pathlib import Path
from colbert import Indexer, Searcher
from colbert.infra import ColBERTConfig
import ir_datasets
from tqdm import tqdm
from ranx import Run as ranx_run
from ranx import Qrels, evaluate


def build_colbert_index(
    dataset_name: str,
    model_name_or_path: str,
    checkpoint_path: str,
    collection: list[str],
    tmp_path: str,
):
    config = ColBERTConfig(
        nbits=8,
        root=str(Path(tmp_path) / f"benchmark_{model_name_or_path}"),
        overwrite=True,
        kmeans_niters=10,
        doc_maxlen=300,
    )
    indexer = Indexer(checkpoint=checkpoint_path, config=config)
    indexer.index(
        name=dataset_name,
        collection=collection,
        overwrite=True,
    )
    return True


def colbert_score(
    model_name_or_path: str,
    dataset_name: str,
    dataset: ir_datasets.Dataset,
    int2docid: dict[int, str],
    tmp_path: str,
    metric: str = "ndcg@10",
):
    qrels_dict = defaultdict(dict)
    for qrel in dataset.qrels_iter():
        qrels_dict[qrel.query_id][qrel.doc_id] = qrel.relevance

    qrels = Qrels(qrels_dict)

    qid_to_query = {}

    for query in dataset.queries_iter():
        qid_to_query[query.query_id] = query.text

    config = ColBERTConfig(
        nbits=8,
        ncells=8,
        ndocs=8192,
        root=str(Path(tmp_path) / f"benchmark_{model_name_or_path}"),
        centroid_score_threshold=0.3,
        doc_maxlen=300,
    )
    searcher = Searcher(index=dataset_name, config=config)
    run_dict = defaultdict(dict)
    for qid, query in tqdm(qid_to_query.items(), desc="Querying " + dataset_name):
        result = searcher.search(query, k=10)
        for i, r in enumerate(result[0]):
            run_dict[qid][int2docid[r]] = result[2][i]
    run = ranx_run(run_dict)
    return evaluate(qrels, run, metric)
