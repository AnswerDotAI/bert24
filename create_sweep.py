import wandb
import yaml

def create_sweep_for_config(config_path):
    task = "beavertails" if "beavertails" in config_path else "wildjailbreak"
    
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
        "program": "eval.py"
    }

    sweep_id = wandb.sweep(sweep_config, project="bert24-llm-guardrails-eval-sweeps", entity="bert24")
    
    return sweep_id

# List of config files
config_files = [
    "configs/beavertails/bert-base-sweep-beavertails.yaml",
    "configs/beavertails/deberta-bert-base-sweep-beavertails.yaml",
    "configs/beavertails/gte-base-sweep-beavertails.yaml",
    "configs/beavertails/modern-bert-base-predecay-sweep-beavertails.yaml",
    "configs/beavertails/nomic-bert-base-sweep-beavertails.yaml",
    "configs/beavertails/roberta-bert-base-sweep-beavertails.yaml",
    "configs/large/wildjailbreak/bert-large.yaml",
    "configs/large/wildjailbreak/deberta-bert-large.yaml",
    
    # "configs/large/wildjailbreak/roberta-bert-large.yaml"
]

# Create sweeps for each config and save IDs
all_sweep_ids = {}
for config_path in config_files:
    sweep_id = create_sweep_for_config(config_path)
    all_sweep_ids[config_path] = sweep_id

# Save sweep IDs to file
with open('sweep_ids.txt', 'w') as f:
    for config_path, sweep_id in all_sweep_ids.items():
        f.write(f"{sweep_id}\n")
        f.write("\n")