# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

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
    try:
        yaml_path, args_list = sys.argv[1], sys.argv[2:]

        with open(yaml_path) as f:
            yaml_cfg = om.OmegaConf.load(f)

        cli_cfg = om.OmegaConf.from_cli(args_list)
        cfg = om.OmegaConf.merge(yaml_cfg, cli_cfg)

        assert isinstance(cfg, om.DictConfig)

        data_dir = f"{cfg.tmp_dir}/data"
        huggingface_hub.snapshot_download(repo_id=cfg.train_dataset_id, repo_type="dataset", local_dir=data_dir)

        if cfg.debug:
            import srsly

            triplets_path = f"{data_dir}/triples.train.colbert.jsonl"
            triplets = srsly.read_jsonl(triplets_path)
            downsampled_triplets = [triplet for i, triplet in enumerate(triplets) if i < 2000]
            srsly.write_jsonl(triplets_path, downsampled_triplets)

        model_name = cfg.model_name_or_path.split("/")[-1] if "/" in cfg.model_name_or_path else cfg.model_name_or_path
        model_name += "_colbert"

        train_params = cfg.train_params
        train_params["root"] = cfg.tmp_dir
        train_params["name"] = model_name

        checkpoint = colbert_train(
            model_name_or_path=cfg.model_name_or_path,
            train_params=train_params,
            n_gpu=cfg.n_gpu,
            data_path=data_dir,
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
    except Exception as e:
        print(f"Error: {e}")
    finally:
    # Clean up ColBERT artifacts
        shutil.rmtree("./experiments/default", ignore_errors=True)
        shutil.rmtree(cfg.tmp_dir, ignore_errors=True)
