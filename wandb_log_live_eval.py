import re
import time
from datetime import datetime, timezone

import pandas as pd
import schedule

import wandb

task2metric_dict = {
    "mnli": ["metrics/glue_mnli/MulticlassAccuracy", "metrics/glue_mnli_mismatched/MulticlassAccuracy"],
    "ultrafeedback": ["metrics/long_context_ultrafeedback/UltrafeedbackAUROC"],
    "mlmmlu_rookie_reserve": ["metrics/mlmmlu_rookie/MulticlassAccuracy", "metrics/mlmmlu_reserve/MulticlassAccuracy"],
    "wic": ["metrics/superglue_wic/MulticlassAccuracy"],
    "boolq": ["metrics/superglue_boolq/MulticlassAccuracy"],
}

metric2num_seeds = {
    "metrics/glue_mnli/MulticlassAccuracy": 3,
    "metrics/glue_mnli_mismatched/MulticlassAccuracy": 3,
    "metrics/mlmmlu_rookie/MulticlassAccuracy": 4,
    "metrics/mlmmlu_reserve/MulticlassAccuracy": 4,
    "metrics/superglue_wic/MulticlassAccuracy": 3,
    "metrics/superglue_boolq/MulticlassAccuracy": 3,
    "metrics/long_context_ultrafeedback/UltrafeedbackAUROC": 2,
}


def parse_model_string(s):
    pattern = r"(bert\d+-base-v2)-ba(\d+)_task=(\w+)(?:_\w+)?_seed=(\d+)"
    match = re.match(pattern, s)
    if match:
        run, batch, task, seed = match.groups()
        return {"run": run, "batch": int(batch), "task": task, "seed": int(seed)}
    else:
        raise ValueError(f"Could not parse model string: {s}")


def process_data():
    print(f"Starting data processing at {datetime.now()}")

    # Initialize first W&B run
    wandb.init(
        project="bert24-live-evals-combined-view",
        name=f"bert24-base-v2-{datetime.now(timezone.utc).strftime('%b %d, %H:%M')}",
    )
    eval_run_id = wandb.run.id
    wandb.finish()

    # Get runs
    api = wandb.Api()
    entity_name = "bert24"
    project_name = "bert24-base-v2-evals"
    runs = api.runs(f"{entity_name}/{project_name}")

    # Process data
    stats = []
    for run in runs:
        if run.state != "finished" or "task=" not in run.name:
            continue
        meta = parse_model_string(run.name)
        task = meta["task"]
        summary = run.summary

        for m in task2metric_dict[task]:
            val = summary.get(m)
            if val:
                stats.append({**meta, "metric": m, "score": val})

    # Aggregate stats
    stats_df = pd.DataFrame(stats)
    grouped_df = stats_df.groupby(["run", "batch", "task", "metric"])["score"].mean().reset_index()
    count_df = stats_df.groupby(["run", "batch", "task", "metric"])["score"].count().reset_index()
    count_df.rename(columns={"score": "count"}, inplace=True)
    grouped_df = pd.merge(grouped_df, count_df, on=["run", "batch", "task", "metric"])

    # Log metrics to W&B
    batch_ticks = sorted(grouped_df["batch"].unique().tolist())
    all_metrics = sorted(grouped_df["metric"].unique().tolist())

    with wandb.init(project="bert24-live-evals-combined-view", id=eval_run_id, resume="must") as run:
        for step in batch_ticks:
            for metric in all_metrics:
                ex = grouped_df[(grouped_df["batch"] == step) & (grouped_df["metric"] == metric)]
                if len(ex) == 1:  # at least one seed in completed
                    if ex["count"].values[0] == metric2num_seeds[metric]:  # make sure all seeds are completed
                        score = ex["score"].values[0]
                        run.log({metric: score}, step=step)

    print(f"Finished data processing at {datetime.now()}")


def main():
    schedule.every(30).minutes.do(process_data)
    process_data()  # first run

    while True:
        try:
            schedule.run_pending()
            time.sleep(30)
        except KeyboardInterrupt:
            print("Scheduler stopped by user. Exiting...")
            break


if __name__ == "__main__":
    main()
