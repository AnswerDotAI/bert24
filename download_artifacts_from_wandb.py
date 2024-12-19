# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

import argparse
import itertools
import os

import wandb


def get_model_artifacts(api, entity, project):
    runs = api.runs(f"{entity}/{project}")
    return list(itertools.chain(*[[(run, a) for a in run.logged_artifacts() if a.type == "model"] for run in runs]))


def get_base_folder(artifact_name):
    name = artifact_name.replace("checkpoint-", "")
    name = "-".join((name.split("-")[:-1]))
    return name


def get_ba(artifact_name):
    name = artifact_name.replace("checkpoint-", "")
    ba = name.split("-")[-1].split(":")[0]
    ba = int(ba.replace("ba", "").strip())
    return ba


def main(api, args):
    print("Fetching all model artifacts...")
    artifacts = get_model_artifacts(api, args.wandb_entity, args.wandb_project)
    print(f"Found {len(artifacts)} model artifacts.")

    for run, artifact in artifacts:
        print(f"Run: {run.name}")
        print(f"Artifact: {artifact.name}")

        base_dir = os.path.join(args.local_download_dir, get_base_folder(artifact.name))

        os.makedirs(base_dir, exist_ok=True)
        out_dir = os.path.join(base_dir, artifact.name)
        if os.path.exists(out_dir):
            print(f"Artifact already exists locally: {out_dir}")
            artifact.delete(delete_aliases=True)
            continue

        os.makedirs(out_dir, exist_ok=True)
        artifact.download(root=out_dir)

        meta_fn = os.path.join(out_dir, "metadata.json")
        meta = {
            "artifact_id": artifact.id,
            "artifact_name": artifact.name,
            "artifact_created_at": artifact.created_at,
            "artifact_updated_at": artifact.updated_at,
            "run_id": run.id,
            "run_name": run.name,
            "project": args.wandb_project,
            "entity": args.wandb_entity,
        }

        with open(meta_fn, "w") as fd:
            fd.write(wandb.util.json_dumps_safer(meta))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download WandB artifacts")
    parser.add_argument("--wandb_entity", help="WandB entity name")
    parser.add_argument("--wandb_project", help="WandB project name")
    parser.add_argument(
        "--local_download_dir",
        help="Local directory to download artifacts",
    )
    args = parser.parse_args()

    # Create download directory if it doesn't exist
    os.makedirs(args.local_download_dir, exist_ok=True)

    # Usage
    # crontab -e
    # 0 * * * * WANDB_API_KEY=<<api_key>> python download_artifacts_from_wandb.py >> <<log_file>> 2>&1
    api = wandb.Api(api_key=os.environ.get("WANDB_API_KEY"))
    main(api, args)
