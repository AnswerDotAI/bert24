from pathlib import Path
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Trainer


def colbert_train(model_name_or_path: str, train_params: dict, n_gpu: int, data_path: str):
    with Run().context(RunConfig(nranks=n_gpu, experiment=model_name_or_path)):
        config = ColBERTConfig(doc_maxlen=300, **train_params)
        data_path = Path(data_path)

        trainer = Trainer(
            triples=str(data_path / "triples.train.colbert.jsonl"),
            queries=str(data_path / "queries.train.colbert.tsv"),
            collection=str(data_path / "corpus.train.colbert.tsv"),
            config=config,
        )

        return trainer.train()