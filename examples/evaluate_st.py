import mteb
from sentence_transformers import SentenceTransformer

model_name = "ModernBERT/bert24-base-v2-2ep-decay_100B-0.08-lr"
model = SentenceTransformer(model_name, trust_remote_code=True)

task_names = ["SciFact", "NFCorpus", "FiQA2018", "TRECCOVID"]
tasks = mteb.get_tasks(tasks=[task_names])
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder=f"results/{model_name}")
