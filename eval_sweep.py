import argparse
import wandb
from ablation_eval import train
import omegaconf as om
from functools import partial


def update_config(config, base_cfg, task_name):
    
    new_task_config = om.OmegaConf.create({})
    new_task_config = om.OmegaConf.merge(new_task_config, base_cfg.tasks[task_name])
    
    new_task_config.trainer_kwargs.max_sequence_length = config["max_sequence_length"]
    new_task_config.trainer_kwargs.max_duration = f"{config['max_duration']}ep" # f"{config['max_duration']}ep", f"{config['max_duration']}ba"
    new_task_config.trainer_kwargs.batch_size = config['batch_size']
    new_task_config.trainer_kwargs.lr = config["lr"]
    new_task_config.trainer_kwargs.weight_decay = config["weight_decay"]
    new_task_config.trainer_kwargs.save_num_checkpoints_to_keep = 0 # no need to keep finetuned checkpoint

    # Update the base configuration with the new task configuration
    base_cfg.tasks = om.OmegaConf.create({task_name: new_task_config})

    return base_cfg

def train_with_one_hp(base_cfg, task_name, metric_name, config=None):
    with wandb.init(config=config):
        # If called by wandb.agent, as below, this config will be set by Sweep Controller
        config = wandb.config
        base_cfg = update_config(config, base_cfg, task_name)

        resutls = train(base_cfg)
        score = resutls[task_name] # random.random()
        wandb.log({metric_name: score})
        print("-" * 80)

def run_sweep(args, base_cfg):
    sweep_config = {'method': 'random'}
    metric = {'name': args.metric_name, 'goal': args.metric_goal}
    sweep_config['metric'] = metric
    parameters_dict = {
        'max_sequence_length': {'value': 256},
        'max_duration': {'values': [2, 3, 4]},
        'batch_size': {'values': [32, 64]},
        'lr': {'values': [1e-5, 2e-5, 4e-5, 8e-5]},
        'weight_decay': {'values': [1e-8, 1e-6]},
    }
    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(sweep=sweep_config, project=args.sweep_project_name)

    wandb.agent(sweep_id, partial(train_with_one_hp, base_cfg=base_cfg, task_name=args.task_name, metric_name=args.metric_name), count=args.run_count)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-config-path', type=str, required=True)
    parser.add_argument('--sweep-project-name', type=str, required=True)
    parser.add_argument('--task-name', type=str, default='mnli')
    parser.add_argument("--metric-name", type=str, required=True)
    parser.add_argument("--metric-goal", type=str, default='maximize')
    parser.add_argument('--run-count', type=int, default=32, help='number of runs in a sweep')

    args = parser.parse_args()

    with open(args.baseline_config_path) as f:
        base_cfg = om.OmegaConf.load(f)

    # just keep the task of interest
    task_name = args.task_name
    if task_name in base_cfg.tasks:
        base_cfg.tasks = om.OmegaConf.create({task_name: base_cfg.tasks[task_name]})
    else:
        raise ValueError(f"Task '{task_name}' not found in the configuration.")

    run_sweep(args,base_cfg)

# Usage: MNLI sweep
# python eval_sweep.py --baseline-config-path "/home/rb/temp-bert-checkpoints/bert24-base-v2-1ep-decay_100B/bert24-base-v2-1ep-decay_100B_evaluation.yaml" --sweep-project-name bert24-base-v2-1ep-decay-100B-sweep \
# --task-name mnli \
# --metric-name metrics/glue_mnli/MulticlassAccuracy \
# --run-count 32


# Usage: MNLI sweep
# python eval_sweep.py --baseline-config-path "/home/rb/bert24_checkpoints/bert24-large-decay-200B-1p-lr-linear/checkpoint-bert24-large-decay-200B-1p-lr-linear-ba39639:v0_evaluation.yaml" --sweep-project-name bert24-large-decay-200B-1p-lr-linear-sweep \
# --task-name mnli \
# --metric-name metrics/glue_mnli/MulticlassAccuracy \
# --run-count 32