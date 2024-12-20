# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

import argparse
import re
import time
from datetime import datetime

import pandas as pd
import schedule
import wandb


def parse_model_string(s):
    pattern = r"(bert24-(base|large)(?:-v\d+)?(?:-\w+)?)-ba(\d+)_task=(\w+)(?:_\w+)?_seed=(\d+)"
    match = re.match(pattern, s)
    if match:
        full_model, size, batch, task, seed = match.groups()
        return {"model": full_model, "size": size, "batch": int(batch), "task": task, "seed": int(seed)}
    else:
        raise ValueError(f"Could not parse model string: {s}")


def init_run(args):
    # Initialize meta W&B run
    wandb.init(project=args.meta_project, name=f"{args.meta_run_name}")
    meta_run_id = wandb.run.id
    wandb.finish()
    print(f"Initialized meta run with ID: {meta_run_id}")
    return meta_run_id


def process_data(args):
    print(f"Starting data processing at {datetime.now()}")

    # Get runs from source eval project
    api = wandb.Api()
    runs = api.runs(f"{args.entity}/{args.source_project}")

    # Process data
    stats = []
    for run in runs:
        if run.state != "finished" or "task=" not in run.name:
            continue
        try:
            meta = parse_model_string(run.name)
        except ValueError:
            print(f"Skipping run with unparsable name: {run.name}")
            continue
        task = meta["task"]
        summary = run.summary

        for m in args.task2metric_dict[task]:
            val = summary.get(m)
            if val:
                stats.append({**meta, "metric": m, "score": val})

    # Aggregate stats
    stats_df = pd.DataFrame(stats)
    print(f"available models: {stats_df.model.unique().tolist()}")
    stats_df = stats_df[stats_df["model"] == args.model_name]

    grouped_df = stats_df.groupby(["model", "size", "batch", "task", "metric"])["score"].mean().reset_index()
    count_df = stats_df.groupby(["model", "size", "batch", "task", "metric"])["score"].count().reset_index()
    count_df.rename(columns={"score": "count"}, inplace=True)
    grouped_df = pd.merge(grouped_df, count_df, on=["model", "size", "batch", "task", "metric"])

    # Log metrics to W&B
    batch_ticks = sorted(grouped_df["batch"].unique().tolist())
    all_metrics = args.all_metrics  # sorted(grouped_df["metric"].unique().tolist())
    grouped_df = grouped_df[grouped_df["metric"].isin(all_metrics)]
    print(batch_ticks)

    with wandb.init(project=args.meta_project, job_type="eval", id=args.meta_run_id, resume="must") as run:
        for step in batch_ticks:
            # check if all metrics are computed for the current batch
            for metric in all_metrics:
                ex = grouped_df[(grouped_df["batch"] == step) & (grouped_df["metric"] == metric)]
                if len(ex) == 0 or ex["count"].values[0] < args.metric2num_seeds[metric]:
                    print(f"insufficient data for step={step} and metric={metric}")
                    print(f"Logged up to step < {step}")
                    return

            for metric in all_metrics:
                ex = grouped_df[(grouped_df["batch"] == step) & (grouped_df["metric"] == metric)]
                if len(ex) == 1:
                    if ex["count"].values[0] >= args.metric2num_seeds[metric]:
                        score = ex["score"].values[0]
                        run.log({metric: score}, step=step)

    print(f"Finished data processing at {datetime.now()}")


def main():
    parser = argparse.ArgumentParser(description="W&B Logging Script")
    parser.add_argument("--entity", type=str, default="bert24", help="W&B entity name")
    parser.add_argument("--meta-project", type=str, default="bert24-evals-meta", help="meta project name")
    parser.add_argument("--model-name", type=str, default="bert24-large-v2", help="Model name")
    parser.add_argument("--meta-run-id", type=str, help="ID of the meta run to update")
    parser.add_argument("--meta-run-name", type=str, default="bert24-large-v2-evals", help="Meta run name")

    parser.add_argument("--source-project", type=str, default="bert24-large-v2-evals", help="project for eval runs")
    parser.add_argument("--interval", type=int, default=60, help="Interval in minutes between data refresh")
    parser.add_argument("--init-meta", action="store_true", help="Initialize a new meta run")

    args = parser.parse_args()

    # metadata information ---
    args.task2metric_dict = {
        "mnli": ["metrics/glue_mnli/MulticlassAccuracy", "metrics/glue_mnli_mismatched/MulticlassAccuracy"],
        "ultrafeedback": ["metrics/long_context_ultrafeedback/UltrafeedbackAUROC"],
        "mlmmlu_rookie_reserve": [
            "metrics/mlmmlu_rookie/MulticlassAccuracy",
            "metrics/mlmmlu_reserve/MulticlassAccuracy",
        ],
        "wic": ["metrics/superglue_wic/MulticlassAccuracy"],
        "boolq": ["metrics/superglue_boolq/MulticlassAccuracy"],
    }

    args.metric2num_seeds = {
        "metrics/glue_mnli/MulticlassAccuracy": 3,
        "metrics/glue_mnli_mismatched/MulticlassAccuracy": 3,
        "metrics/mlmmlu_rookie/MulticlassAccuracy": 3,
        "metrics/mlmmlu_reserve/MulticlassAccuracy": 3,
        "metrics/superglue_wic/MulticlassAccuracy": 3,
        "metrics/superglue_boolq/MulticlassAccuracy": 3,
        "metrics/long_context_ultrafeedback/UltrafeedbackAUROC": 2,
    }

    args.all_metrics = [
        "metrics/glue_mnli/MulticlassAccuracy",
        "metrics/glue_mnli_mismatched/MulticlassAccuracy",
        # "metrics/mlmmlu_rookie/MulticlassAccuracy",
        # "metrics/mlmmlu_reserve/MulticlassAccuracy",
        "metrics/superglue_wic/MulticlassAccuracy",
        "metrics/superglue_boolq/MulticlassAccuracy",
    ]

    if args.init_meta:
        meta_run_id = init_run(args)
        print(f"Use this meta_run_id for future runs: {meta_run_id}")
        return

    if not args.meta_run_id:
        parser.error("--meta-run-id is required when not initializing a new meta run")

    schedule.every(args.interval).minutes.do(process_data, args)
    process_data(args)  # first run

    while True:
        try:
            schedule.run_pending()
            time.sleep(30)
        except KeyboardInterrupt:
            print("Scheduler stopped by user. Exiting...")
            break


if __name__ == "__main__":
    main()

## Usage
# python wandb_log_live_eval.py --init-meta --model-name <<model_name>> --meta-project <<project_name>> --meta-run-name "<<model_name>>-evals"
