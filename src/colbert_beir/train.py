from pathlib import Path
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Trainer


def colbert_train(model_name_or_path: str, train_params: dict, n_gpu: int, data_path: str):
    with Run().context(
        RunConfig(
            nranks=n_gpu,
            experiment=model_name_or_path,
            name=train_params["name"],
            root=train_params["root"],
        )
    ):
        config = ColBERTConfig(doc_maxlen=300, **train_params)
        print(config)
        data_path = Path(data_path)

        trainer = Trainer(
            triples=str(data_path / "triples.train.colbert.jsonl"),
            queries=str(data_path / "queries.train.colbert.tsv"),
            collection=str(data_path / "corpus.train.colbert.tsv"),
            config=config,
        )

        trainer.train(checkpoint=model_name_or_path)
        return f"{train_params['root']}/{model_name_or_path}/none/{train_params['name']}/checkpoints/colbert"
