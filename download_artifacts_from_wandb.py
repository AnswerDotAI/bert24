import itertools
import os

import wandb

###############
# WANDB CONFIGS
WANDB_ENTITY = "bert24"
WANDB_PROJECT = "bert24-base-in-run-evals"
LOCAL_DOWNLOAD_DIR = os.path.expanduser("~/bert24_checkpoints/")
################

# Create download directory if it doesn't exist
os.makedirs(LOCAL_DOWNLOAD_DIR, exist_ok=True)


def get_model_artifacts(api):
    runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}")
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


def main(api, ba_th):
    print("Fetching all model artifacts...")
    artifacts = get_model_artifacts(api)
    print(f"Found {len(artifacts)} model artifacts.")

    for run, artifact in artifacts:
        if "bert24-base-v2" not in artifact.name:
            # print(f"skipping {artifact.name}")
            continue

        print(f"Run: {run.name}")
        print(f"Artifact: {artifact.name}")
        ba = get_ba(artifact.name)
        print(f"ba = {ba}")

        if ba <= ba_th:
            print(f"skipping {artifact.name}")
            continue

        base_dir = os.path.join(LOCAL_DOWNLOAD_DIR, get_base_folder(artifact.name))

        os.makedirs(base_dir, exist_ok=True)
        out_dir = os.path.join(base_dir, artifact.name)
        if os.path.exists(out_dir):
            print(f"Artifact already exists locally: {out_dir}")
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
            "project": WANDB_PROJECT,
            "entity": WANDB_ENTITY,
        }

        with open(meta_fn, "w") as fd:
            fd.write(wandb.util.json_dumps_safer(meta))


if __name__ == "__main__":
    # Usage
    # crontab -e
    # 0 * * * * WANDB_API_KEY=api_key /opt/conda/envs/bert24/bin/python /home/rb/bert24/download_artifacts_from_wandb.py >> /home/rb/wandb_checkpoint_downloader.log 2>&1
    api = wandb.Api(api_key=os.environ.get("WANDB_API_KEY"))
    ba_skip_before = 0
    main(api, ba_skip_before)
