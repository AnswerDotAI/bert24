import sys
import shutil

import huggingface_hub
import omegaconf as om
import ir_datasets

from src.colbert_beir import build_colbert_index, colbert_score, colbert_train


def _make_passage(doc):
    if hasattr(doc, "title"):
        return f"{doc.title}\n{doc.text}"
    else:
        return doc.text


if __name__ == "__main__":
    yaml_path, args_list = sys.argv[1], sys.argv[2:]

    with open(yaml_path) as f:
        yaml_cfg = om.OmegaConf.load(f)

    cli_cfg = om.OmegaConf.from_cli(args_list)
    cfg = om.OmegaConf.merge(yaml_cfg, cli_cfg)

    assert isinstance(cfg, om.DictConfig)

    huggingface_hub.snapshot_download(repo_id=cfg.train_dataset_id, repo_type="dataset", local_dir=cfg.tmp_dir)

    checkpoint = colbert_train(
        model_name_or_path=cfg.model_name_or_path,
        train_params=cfg.train_params,
        n_gpu=cfg.n_gpu,
        data_path=cfg.tmp_dir,
    )

    for dataset_name in cfg.eval_datasets:
        int2docid = {}
        docs = []
        ds_split = ""
        dataset = ir_datasets.load(dataset_name)

        for i, doc in enumerate(dataset.docs_iter()):
            int2docid[i] = doc.doc_id
            docs.append(_make_passage(doc))

        build_colbert_index(
            dataset_name=dataset_name,
            model_name_or_path=cfg.model_name_or_path,
            checkpoint_path=checkpoint,
            collection=docs,
            tmp_path=cfg.tmp_dir,
        )
        score = colbert_score(
            model_name_or_path=cfg.model_name_or_path,
            dataset_name=dataset_name,
            dataset=dataset,
            int2docid=int2docid,
            tmp_path=cfg.tmp_dir,
        )
        print(f"NDCG@10 for {dataset_name}: {score}")

        shutil.rmtree(cfg.tmp_dir, ignore_errors=True)
