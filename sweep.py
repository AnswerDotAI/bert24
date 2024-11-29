import argparse
import wandb
import subprocess

# Constants
PROGRAM_PATH = "/opt/home/bert24/guardrails/bert24/eval.py"
COMMAND_BASE = ["/usr/bin/env", "python", PROGRAM_PATH]


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run WandB sweeps.")
    parser.add_argument("--project", default="bert24-llm-guardrails-eval-sweeps", help="WandB project name")
    parser.add_argument("--entity", default="bert24", help="WandB entity name")
    parser.add_argument("--task", default="wildjailbreak", help="Task name for metrics")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs for the sweep")
    parser.add_argument("--method", choices=["random", "bayes"], default="random", help="Sweep method")
    parser.add_argument("--config_path", required=True, help="Path to the YAML config file")
    return parser.parse_args()


def create_sweep_config(task, method, config_path):
    """Create the WandB sweep configuration."""
    return {
        "program": PROGRAM_PATH,
        "command": ["${env}", "${interpreter}", "${program}", config_path, "${args}"],
        "method": method,
        "metric": {"goal": "maximize", "name": f"metrics/{task}/MulticlassAccuracy"},
        "parameters": {
            "device_train_microbatch_size": {"values": [64]},
            "learning_rate": {"values": [1e-5, 2e-5, 4e-5, 8e-5]},
            "weight_decay": {"values": [1e-8, 1e-6, 1e-5]},
            "max_duration": {"values": [1]},
        },
    }


def train(config_path):
    """Training function for WandB sweep."""
    with wandb.init():
        config = wandb.config
        cmd = COMMAND_BASE + [
            config_path,
            f"--device_train_microbatch_size={config.device_train_microbatch_size}",
            f"--learning_rate={config.learning_rate}",
            f"--max_duration={config.max_duration}",
            f"--weight_decay={config.weight_decay}",
        ]

        try:
            subprocess.run(cmd, check=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Command failed with error: {e.stderr}")
            raise e


def main():
    args = parse_args()

    wandb.login()
    sweep_config = create_sweep_config(task=args.task, method=args.method, config_path=args.config_path)
    sweep_id = wandb.sweep(sweep=sweep_config, project=args.project, entity=args.entity)
    print(f"Created sweep with ID: {sweep_id}")

    wandb.agent(sweep_id, function=lambda: train(args.config_path), count=args.runs)


if __name__ == "__main__":
    main()
