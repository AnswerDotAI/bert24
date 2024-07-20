import glob
import os
import re
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
import psutil
from typing import Annotated, List, Optional

import typer
import yaml
from typer import Exit, Option

app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]}, pretty_exceptions_show_locals=False)


# from maxb2: https://github.com/tiangolo/typer/issues/86#issuecomment-996374166
def conf_callback(ctx: typer.Context, param: typer.CallbackParam, config: Optional[str] = None):
    if config is not None:
        typer.echo(f"Loading config file: {config}\n")
        try:
            with open(config, "r") as f:  # Load config file
                conf = yaml.safe_load(f)
            ctx.default_map = ctx.default_map or {}  # Initialize the default map
            ctx.default_map.update(conf)  # Merge the config dict into default_map
        except Exception as ex:
            raise typer.BadParameter(str(ex))
    return config


# Global dictionary to keep track of GPUs with running jobs
gpus_in_use = {}
# Queue to keep track of GPUs that might be free
potentially_free_gpus = deque()
# Global list to keep track of all running processes
all_processes = []


def kill_process_tree(pid):
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            child.terminate()
        gone, still_alive = psutil.wait_procs(children, timeout=5)
        for p in still_alive:
            p.kill()
        parent.terminate()
        parent.wait(5)
    except psutil.NoSuchProcess:
        pass


def signal_handler(signum, frame):
    print("\nReceived termination signal. Cleaning up subprocesses...")
    for process in all_processes:
        if process.poll() is None:  # If the process is still running
            kill_process_tree(process.pid)

    print("Cleanup completed. Exiting.")
    os._exit(0)  # Force exit without running cleanup handlers


def get_gpu_memory_usage(gpu_id):
    """Get memory usage for a specific GPU."""
    try:
        output = (
            subprocess.check_output(
                f"nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader -i {gpu_id}", shell=True
            )
            .decode("utf-8")
            .strip()
        )
        return int(output)
    except subprocess.CalledProcessError:
        print(f"Failed to get memory usage for GPU {gpu_id}")
        return None


def get_free_gpu():
    """Check for free GPUs, prioritizing potentially free GPUs."""
    while potentially_free_gpus:
        gpu_id = potentially_free_gpus.popleft()
        if gpu_id not in gpus_in_use:
            memory_used = get_gpu_memory_usage(gpu_id)
            if memory_used is not None and memory_used < 100:
                return gpu_id

    # If no potentially free GPUs, check all GPUs
    try:
        gpu_output = subprocess.check_output(
            "nvidia-smi --query-gpu=index,memory.used --format=csv,nounits,noheader", shell=True
        ).decode("utf-8")
        for line in gpu_output.strip().split("\n"):
            gpu_id, memory_used = map(int, line.split(","))
            if memory_used < 100 and gpu_id not in gpus_in_use:
                return gpu_id
        return None
    except subprocess.CalledProcessError:
        print("Failed to execute nvidia-smi")
        return None


def launch_job(gpu_id: int, config_path: Path, quiet: bool = False):
    """Launch a job on a specified GPU and return the process."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    stdout = subprocess.DEVNULL if quiet else None
    process = subprocess.Popen(["python", "ablation_eval.py", config_path], env=env, stdout=stdout, stderr=stdout)
    gpus_in_use[gpu_id] = process
    all_processes.append(process)  # Add the process to the global list
    return process


def run_subprocess(cmd: str, quiet: bool = False):
    stdout = subprocess.DEVNULL if quiet else None
    process = subprocess.Popen(cmd, stdout=stdout, stderr=stdout)
    all_processes.append(process)  # Add the process to the global list
    process.wait()


def run_single_job(config_path: Path, quiet: bool = False):
    """Run a single job without GPU management."""
    print(f"Running job for {config_path}")
    stdout = subprocess.DEVNULL if quiet else None
    process = subprocess.Popen(["python", "ablation_eval.py", config_path], stdout=stdout, stderr=stdout)
    all_processes.append(process)  # Add the process to the global list
    process.wait()


def check_finished_jobs():
    """Check for finished jobs and free up their GPUs."""
    finished_gpus = []
    for gpu_id, process in gpus_in_use.items():
        if process.poll() is not None:  # Job has finished
            finished_gpus.append(gpu_id)

    for gpu_id in finished_gpus:
        print(f"Job on GPU {gpu_id} has finished. Marking GPU as potentially free.")
        del gpus_in_use[gpu_id]
        potentially_free_gpus.append(gpu_id)


def manage_jobs(config_directory: Path, quiet=False):
    """Manage the launching of jobs for each configuration file in the directory."""
    for config in config_directory.glob("*_evaluation.yaml"):
        while True:
            check_finished_jobs()
            gpu_id = get_free_gpu()
            if gpu_id is not None:
                print(f"\nLaunching job for {config} on GPU {gpu_id}\n")
                launch_job(gpu_id, config, quiet)
                break
            else:
                time.sleep(10)

    # Wait for all remaining jobs to finish
    while gpus_in_use:
        check_finished_jobs()
        time.sleep(10)


def create_symlink_for_newest_checkpoint(folder: Path, override_existing: bool = False):
    """Create a symlink to the newest checkpoint file if 'latest-rank0.pt' does not exist."""
    if folder.is_dir():
        pt_files = list(folder.glob("*.pt"))
        if not pt_files:
            print(f"   Warning: No .pt file found in {folder}.")
            return

        # Sort files based on epoch and batch numbers extracted from filenames
        def extract_numbers(filename: Path):
            if filename.is_symlink():
                return (0, 0)
            if filename.name == "latest-rank0.pt":
                return (0, 0)

            try:
                # Using regex to find patterns of 'ep' followed by digits and 'ba' followed by digits
                match = re.search(r"ep(\d+)-ba(\d+)", filename.stem)
                if match:
                    epoch, batch = map(int, match.groups())
                    return (epoch, batch)
                else:
                    raise ValueError(f"Filename does not match expected pattern: {filename}")
            except Exception as e:
                print(f"   Error extracting numbers from filename {filename}: {e}")
                return (0, 0)

        newest_file = max(pt_files, key=extract_numbers)

        symlink_path = folder / "latest-rank0.pt"
        if symlink_path.exists() and symlink_path.is_symlink():
            if symlink_path.resolve() == newest_file.resolve():
                print(f"   Existing symlink points to latest checkpoint: {newest_file.parent.name}/{newest_file.name}")
                return
            else:
                print(
                    f"   Warning: Existing symlink points to {symlink_path.parent.name}/{symlink_path.name}, "
                    f"but latest checkpoint is {newest_file.parent.name}/{newest_file.name}"
                )
                if not override_existing:
                    return

        symlink_path.symlink_to(newest_file.name)
        if override_existing:
            print(
                f"   Overwriting existing symlink with {symlink_path.parent.name}/{symlink_path.name} -> {newest_file.name}"
            )
        else:
            print(f"   Created new symlink {symlink_path.parent.name}/{symlink_path.name} -> {newest_file.name}")


def generate_eval_configs(
    checkpoints: Path,
    train_config: Optional[Path],
    wandb_run: Optional[str],
    wandb_project: Optional[str],
    wandb_entity: Optional[str],
    track_run: bool,
    pooling_type: Optional[str],
    head_class_act: Optional[str],
    head_class_norm: Optional[str],
    head_class_dropout: float,
    skip_semipro: bool,
    skip_reserve: bool,
    skip_eurlex: bool,
    skip_mnli: bool,
    skip_boolq: bool,
    skip_wic: bool,
    seeds: List[int],
    parallel: bool,
):
    """Generate evaluation configs for each checkpoint."""
    for folder in checkpoints.glob("*"):
        if folder.is_dir():
            cmd = [
                "python",
                "generate_eval_config_from_checkpoint.py",
                "--checkpoint",
                folder,
                "--output-dir",
                str(checkpoints),
            ]

            # Add optional arguments if they're provided
            if train_config:
                cmd.extend(["--train-config", str(train_config)])
            if wandb_run:
                cmd.extend(["--wandb-run", wandb_run])
            if wandb_project:
                cmd.extend(["--wandb-project", wandb_project])
            if wandb_entity:
                cmd.extend(["--wandb-entity", wandb_entity])
            if track_run:
                cmd.append("--track-run")

            # classification head options
            if pooling_type:
                cmd.extend(["--pooling-type", pooling_type])
            if head_class_act:
                cmd.extend(["--head-class-act", head_class_act])
            if head_class_norm:
                cmd.extend(["--head-class-norm", head_class_norm])
            if head_class_dropout > 0:
                cmd.extend(["--head-class-dropout", str(head_class_dropout)])

            # Add skip arguments
            if skip_semipro:
                cmd.append("--skip-semipro")
            if skip_reserve:
                cmd.append("--skip-reserve")
            if skip_eurlex:
                cmd.append("--skip-eurlex")
            if skip_mnli:
                cmd.append("--skip-mnli")
            if skip_boolq:
                cmd.append("--skip-boolq")
            if skip_wic:
                cmd.append("--skip-wic")

            for seed in seeds:
                cmd.extend(["--seeds", str(seed)])

            cmd.append("--parallel") if parallel else cmd.append("--single")

            # Run the config generation process without suppressing output
            run_subprocess(cmd)


# fmt: off
@app.command()
def main(
    checkpoints: Annotated[Path, Option(help="Path to the directory containing checkpoints", rich_help_panel="Checkpoint & Config Paths")],
    train_config: Annotated[Optional[Path], Option(help="Path to a single .yaml file containing training configuration", rich_help_panel="Checkpoint & Config Paths")] = None,
    skip_generation: Annotated[bool, Option("--skip-generation", help="Skip generation of evaluation configs", rich_help_panel="Config Options")] = False,
    wandb_run: Annotated[Optional[str], Option(help="wandb run containing the training configuration", rich_help_panel="W&B")] = None,
    wandb_project: Annotated[Optional[str], Option(help="wandb project for the run", rich_help_panel="W&B")] = None,
    wandb_entity: Annotated[Optional[str], Option(help="wandb entity for the project", rich_help_panel="W&B")] = None,
    track_run: Annotated[bool, Option("--track-run", help="Track the eval run with wandb", rich_help_panel="W&B")] = False,
    pooling_type: Annotated[Optional[str], Option(help="Pooling type for the classification head", show_default=False, rich_help_panel="Model Options")] = None,
    head_class_act: Annotated[Optional[str], Option(help="Classification head activation function", show_default=False, rich_help_panel="Model Options")] = None,
    head_class_norm: Annotated[Optional[str], Option(help="Classification head normalization function", show_default=False, rich_help_panel="Model Options")] = None,
    head_class_dropout: Annotated[float, Option(help="Classification head dropout rate", rich_help_panel="Model Options")] = 0.0,
    skip_semipro: Annotated[bool, Option("--skip-semipro", help="Skip the MlMMLU-Amateur/Semipro eval", rich_help_panel="Skip Tasks")] = False,
    skip_reserve: Annotated[bool, Option("--skip-reserve", help="Skip the MlMMLU-Rookie/Reserve eval", rich_help_panel="Skip Tasks")] = False,
    skip_eurlex: Annotated[bool, Option("--skip-eurlex", help="Skip the EurLex eval", rich_help_panel="Skip Tasks")] = False,
    skip_mnli: Annotated[bool, Option("--skip-mnli", help="Skip the MNLI eval", rich_help_panel="Skip Tasks")] = False,
    skip_boolq: Annotated[bool, Option("--skip-boolq", help="Skip the BoolQ eval", rich_help_panel="Skip Tasks")] = False,
    skip_wic: Annotated[bool, Option("--skip-wic", help="Skip the WIC eval", rich_help_panel="Skip Tasks")] = False,
    seeds: Annotated[List[int], Option(help="List of seeds to use for the eval", rich_help_panel="Task Settings")] = [23, 42, 6033],
    quiet: Annotated[bool, Option("-q", "--quiet", help="Suppress output from evaluation jobs", rich_help_panel="Config Options")] = False,
    overwrite_existing_symlinks: Annotated[bool, Option("--override-existing-symlinks", help="Overwrite existing symlinks to point to latest checkpoint", rich_help_panel="Config Options")] = False,
    parallel: Annotated[bool, Option("--parallel/--single", help="Run the evals in parallel on multiple GPUs or one GPU", rich_help_panel="Task Settings")] = False,
    config: Annotated[Optional[Path], Option(callback=conf_callback, is_eager=True, help="Relative path to YAML config file for setting options. Passing CLI options will supersede config options.", case_sensitive=False, rich_help_panel="Config Options")] = None,
):
# fmt: on
    print("Creating symlinks for latest checkpoints...")
    for folder in checkpoints.glob("*"):
        create_symlink_for_newest_checkpoint(folder, overwrite_existing_symlinks)
    print()
    if train_config:
        config_files = [train_config]
    elif not skip_generation:
        print("Generating evaluation configs...\n")
        generate_eval_configs(
            checkpoints=checkpoints,
            train_config=train_config,
            wandb_run=wandb_run,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            track_run=track_run,
            pooling_type=pooling_type,
            head_class_act=head_class_act,
            head_class_norm=head_class_norm,
            head_class_dropout=head_class_dropout,
            skip_semipro=skip_semipro,
            skip_reserve=skip_reserve,
            skip_eurlex=skip_eurlex,
            skip_mnli=skip_mnli,
            skip_boolq=skip_boolq,
            skip_wic=skip_wic,
            seeds=seeds,
            parallel=parallel,
        )
        config_files = list(checkpoints.glob("*_evaluation.yaml"))
    else:
        config_files = list(checkpoints.glob("*_evaluation.yaml"))

    if len(config_files) == 1:
        run_single_job(config_files[0], quiet)
    elif len(config_files) > 1:
        manage_jobs(checkpoints, quiet)
    else:
        print("No configuration files found in the specified directory.")
        raise Exit(code=1)

    print("All jobs completed.")


# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    try:
        app()
    finally:
        # Ensure all subprocesses are terminated when the script exits
        for process in all_processes:
            if process.poll() is None:
                process.terminate()
        for process in all_processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
