# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

import mteb
from sentence_transformers import SentenceTransformer

model_name = "answerdotai/ModernBERT-base"
lr = 8e-5
model_shortname = model_name.split("/")[-1]
run_name = f"{model_shortname}-DPR-{lr}"
output_dir = f"output/{model_shortname}/{run_name}"
model = SentenceTransformer(f"{output_dir}/final")

task_names = ["SciFact", "NFCorpus", "FiQA2018", "TRECCOVID"]
tasks = mteb.get_tasks(tasks=task_names)
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder=f"results/{run_name}")
