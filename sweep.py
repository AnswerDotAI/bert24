import wandb
import subprocess
from pathlib import Path

# Constants
PROGRAM_PATH = "/opt/home/bert24/guardrails/bert24/eval.py"
CONFIG_PATH = "configs/modern-bert-base-sweep-wild-jail-break-early-stopping.yaml"
SWEEP_PROJECT = "bert24-llm-guardrails-eval-sweeps"
SWEEP_ENTITY = "bert24"
COMMAND_BASE = ["/usr/bin/env", "python", PROGRAM_PATH, CONFIG_PATH]
NUMBER_RUNS = 10
TASK = "beavertails"

# Sweep Configuration
SWEEP_CONFIG = {
    "program": PROGRAM_PATH,
    "command": ["${env}", "${interpreter}", "${program}", CONFIG_PATH, "${args}"],
    "method": "random",
    "metric": {"goal": "maximize", "name": f"metrics/{TASK}/MulticlassAccuracy"},
    "parameters": {
        "device_train_microbatch_size": {"values": [32, 64]},
        "learning_rate": {"values": [1e-5, 2e-5, 4e-5, 8e-5]},
        "weight_decay": {"values": [1e-8, 1e-6, 1e-5]},
        "max_duration": {"values": [1, 2, 3, 4]},
    },
}


def create_sweep_config():
    return SWEEP_CONFIG


def train():
    with wandb.init():
        config = wandb.config
        cmd = COMMAND_BASE + [
            f"--device_train_microbatch_size={config.device_train_microbatch_size}",
            f"--learning_rate={config.learning_rate}",
            f"--max_duration={config.max_duration}",
            f"--weight_decay={config.weight_decay}",
        ]

        try:
            subprocess.run(cmd, check=True, text=True)  # Logs directly to console
        except subprocess.CalledProcessError as e:
            print(f"Command failed with error: {e.stderr}")
            raise e


def main():
    wandb.login()
    sweep_config = create_sweep_config()
    sweep_id = wandb.sweep(sweep=sweep_config, project=SWEEP_PROJECT, entity=SWEEP_ENTITY)
    print(f"Created sweep with ID: {sweep_id}")
    wandb.agent(sweep_id, function=train, count=NUMBER_RUNS)


if __name__ == "__main__":
    main()
