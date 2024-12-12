import wandb
import yaml

"""
This script creates sweeps in the UI from a list of configs. The sweep IDs are written to a text file.
Those sweeps are then run with launch_sweeps.sh script.
"""


def create_sweep_for_config(config_path):
    parent_task = "guardrails"
    if "beavertails" in config_path:
        task = "beavertails"
    elif "wildjailbreak" in config_path:
        task = "wildjailbreak"
    elif "glue/" in config_path:
        parent_task = "glue"
        task = config_path.split("/glue/")[1].split("/")[0]
        print(f"found task {task} of parent task : {parent_task}")
    
    if parent_task == "glue":
        sweep_config = {
            "name": config_path.split('/')[-1].replace("-sweep", "").replace(".yaml", ""),
            "command": [
                "${env}",
                "${interpreter}",
                "${program}",
                config_path,
                "${args}"
            ],
            "method": "random",
            "metric": {
                "goal": "maximize",
                "name": f"metrics/{parent_task}_{task}/MulticlassAccuracy"
            },
            "parameters": {
                "device_train_microbatch_size": {"values": [64]},
                "task": {"values": [task]},
                "starting_cp": {"values": ["composer_checkpoint.pt"]},
                "learning_rate": {"values": [1e-5, 3e-5, 5e-5, 8e-5]},
                "max_duration": {"values": [1, 2, 3]},
                "weight_decay": {"values": [1e-5, 8e-6, 5e-6, 3e-6, 1e-6]}
            },
            "program": "eval.py",
            "run_cap": 60,
        }

        sweep_id = wandb.sweep(sweep_config, project="better_glue_sweeps", entity="bert24")
    else:
        sweep_config = {
            "name": config_path.split('/')[-1].replace("-sweep", "").replace(".yaml", ""),
            "command": [
                "${env}",
                "${interpreter}",
                "${program}",
                config_path,
                "${args}"
            ],
            "method": "random",
            "metric": {
                "goal": "maximize",
                "name": f"metrics/{task}/MulticlassAccuracy"
            },
            "parameters": {
                "device_train_microbatch_size": {"values": [64]},
                "learning_rate": {"values": [1e-5, 2e-5, 4e-5, 8e-5]},
                "max_duration": {"values": [1, 2, 3, 4]},
                "weight_decay": {"values": [1e-8, 1e-6, 1e-5]}
            },
            "program": "eval.py",
            "run_cap": 48,
        }

        sweep_id = wandb.sweep(sweep_config, project="bert24-llm-guardrails-eval-sweeps", entity="bert24")
    
    return sweep_id

# configs to run
config_files = [
    # configs to run here
    
]

all_sweep_ids = {}
for config_path in config_files:
    sweep_id = create_sweep_for_config(config_path)
    all_sweep_ids[config_path] = sweep_id

# save sweep IDs to a txt
with open('sweep_ids.txt', 'w') as f:
    for config_path, sweep_id in all_sweep_ids.items():
        f.write(f"{sweep_id}\n")
        f.write("\n")