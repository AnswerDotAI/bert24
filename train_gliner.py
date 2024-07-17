import argparse
import json
import os
import re
from transformers import AutoTokenizer
import omegaconf as om
from types import SimpleNamespace

from tqdm import tqdm
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_inverse_sqrt_schedule,
)
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from gliner import GLiNER
from gliner.modules.base import load_config_as_namespace
from gliner.modules.run_evaluation import get_for_all_path

from src.flex_bert import create_flex_bert_model



def build_model(
    cfg, num_labels: int, multiple_choice: bool = False, **kwargs
):
    # TODO
    # if cfg.name == "hf_bert":
    #     return hf_bert_module.create_hf_bert_classification(
    #         num_labels=num_labels,
    #         pretrained_model_name=cfg.pretrained_model_name,
    #         use_pretrained=cfg.get("use_pretrained", False),
    #         model_config=cfg.get("model_config", None),
    #         tokenizer_name=cfg.get("tokenizer_name", None),
    #         gradient_checkpointing=cfg.get("gradient_checkpointing", None),
    #         multiple_choice=multiple_choice,
    #         **kwargs,
    #     )
    # elif cfg.name == "mosaic_bert":
    #     return mosaic_bert_module.create_mosaic_bert_classification(
    #         num_labels=num_labels,
    #         pretrained_model_name=cfg.pretrained_model_name,
    #         pretrained_checkpoint=cfg.get("pretrained_checkpoint", None),
    #         model_config=cfg.get("model_config", None),
    #         tokenizer_name=cfg.get("tokenizer_name", None),
    #         gradient_checkpointing=cfg.get("gradient_checkpointing", None),
    #         multiple_choice=multiple_choice,
    #         **kwargs,
    #     )
    if cfg.name == "flex_bert":
        return create_flex_bert_model(
            pretrained_model_name=cfg.pretrained_model_name,
            pretrained_checkpoint=cfg.get("pretrained_checkpoint", None),
            model_config=cfg.get("model_config", None),
            tokenizer_name=cfg.get("tokenizer_name", None),
            gradient_checkpointing=cfg.get("gradient_checkpointing", None),
            **kwargs,
        )

def save_top_k_checkpoints(model: GLiNER, save_path: str, checkpoint: int, top_k: int = 5):
    """
    Save the top-k checkpoints (latest k checkpoints) of a model and tokenizer.

    Parameters:
        model (GLiNER): The model to save.
        save_path (str): The directory path to save the checkpoints.
        top_k (int): The number of top checkpoints to keep. Defaults to 5.
    """
    # Save the current model and tokenizer
    if isinstance(model, DDP):
        model.module.save_pretrained(os.path.join(save_path, str(checkpoint)))
    else:
        model.save_pretrained(os.path.join(save_path, str(checkpoint)))

    # List all files in the directory
    files = os.listdir(save_path)

    # Filter files to keep only the model checkpoints
    checkpoint_folders = [file for file in files if re.search(r'model_\d+', file)]

    # Sort checkpoint files by modification time (latest first)
    checkpoint_folders.sort(key=lambda x: os.path.getmtime(os.path.join(save_path, x)), reverse=True)

    # Keep only the top-k checkpoints
    for checkpoint_folder in checkpoint_folders[top_k:]:
        checkpoint_folder = os.path.join(save_path, checkpoint_folder)
        checkpoint_files = [os.path.join(checkpoint_folder, f) for f in os.listdir(checkpoint_folder)]
        for file in checkpoint_files:
            os.remove(file)
        os.rmdir(os.path.join(checkpoint_folder))


class Trainer:
    def __init__(self, raw_config, gliner_config, allow_distributed, device='cuda'):
        self.config = gliner_config
        self.raw_config = raw_config
        config = gliner_config
        self.lr_encoder = float(self.config.lr_encoder)
        self.lr_others = float(self.config.lr_others)
        self.weight_decay_encoder = float(self.config.weight_decay_encoder)
        self.weight_decay_others = float(self.config.weight_decay_other)

        self.device = device

        if self.config.prev_path != "none":  # fine-tuning mode
            raise NotImplementedError
            self.model_config = SimpleNamespace(
                loss_alpha=config.loss_alpha,
                loss_gamma=config.loss_gamma,
                loss_reduction=config.loss_reduction,
                max_types=config.max_types,
                shuffle_types=config.shuffle_types,
                random_drop=config.random_drop,
                max_neg_type_ratio=config.max_neg_type_ratio,
                max_len=config.max_len,
            )
        else:
            if self.raw_config is not None:
                print(raw_config)
                raw_model = create_flex_bert_model(pretrained_model_name=raw_config.model.pretrained_model_name,
                                                   model_config=raw_config.model.model_config,
                                                   tokenizer_name=raw_config.model.tokenizer_name,
                                                   gradient_checkpointing=raw_config.model.get('gradient_checkpointing', None),
                                                   pretrained_checkpoint=raw_config.starting_checkpoint_load_path)
                self.model_config = SimpleNamespace(
                custom_base_model=True,
                custom_model_object=raw_model,
                tokenizer_object=AutoTokenizer.from_pretrained(raw_config.tokenizer_name),
                model_name=raw_config.base_run_name,
                tokenizer_name=raw_config.tokenizer_name,
                name=config.name,
                max_width=config.max_width,
                hidden_size=config.hidden_size,
                dropout=config.dropout,
                fine_tune=config.fine_tune,
                subtoken_pooling=config.subtoken_pooling,
                span_mode=config.span_mode,
                loss_alpha=config.loss_alpha,
                loss_gamma=config.loss_gamma,
                loss_reduction=config.loss_reduction,
                max_types=config.max_types,
                shuffle_types=config.shuffle_types,
                random_drop=config.random_drop,
                max_neg_type_ratio=config.max_neg_type_ratio,
                    max_len=config.max_len,
                    )

            else:
                self.model_config = SimpleNamespace(
                    model_name=config.model_name,
                tokenizer_name=config.tokenizer_name,
                name=config.name,
                max_width=config.max_width,
                hidden_size=config.hidden_size,
                dropout=config.dropout,
                fine_tune=config.fine_tune,
                subtoken_pooling=config.subtoken_pooling,
                span_mode=config.span_mode,
                loss_alpha=config.loss_alpha,
                loss_gamma=config.loss_gamma,
                loss_reduction=config.loss_reduction,
                max_types=config.max_types,
                shuffle_types=config.shuffle_types,
                random_drop=config.random_drop,
                max_neg_type_ratio=config.max_neg_type_ratio,
                max_len=config.max_len,
                )

        self.allow_distributed = allow_distributed

    def setup_distributed(self, rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12356'
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    def cleanup_distributed(self):
        dist.destroy_process_group()

    def setup_model_and_optimizer(self, rank=None, device=None):
        if device is None:
            device = self.device
        if self.config.prev_path != "none":
            raise NotImplementedError
            model = GLiNER.from_pretrained(self.config.prev_path).to(device)

            # some parameters of model.config are not overwritten by the config file
            # other than these are overwritten
            keep_params = ['model_name', 'name', 'max_width', 'hidden_size', 'dropout', 'subtoken_pooling', 'span_mode',
                           "fine_tune"]

            for param in keep_params:
                original_value = getattr(model.config, param)
                setattr(self.model_config, param, original_value)

            model.config = self.model_config
        else:
            model = GLiNER(self.model_config).to(device)

        if rank is not None:
            model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
            optimizer = model.module.get_optimizer(self.lr_encoder, self.lr_others,
                                                   self.weight_decay_encoder, self.weight_decay_others,
                                                   freeze_token_rep=self.config.freeze_token_rep)
        else:
            optimizer = model.get_optimizer(self.lr_encoder, self.lr_others,
                                            self.weight_decay_encoder, self.weight_decay_others,
                                            freeze_token_rep=self.config.freeze_token_rep)
        return model, optimizer

    def train_dist(self, rank, world_size, dataset):
        # Init distributed process group
        self.setup_distributed(rank, world_size)

        device = f'cuda:{rank}'

        model, optimizer = self.setup_model_and_optimizer(rank, device=device)

        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)

        train_loader = model.module.create_dataloader(dataset, batch_size=self.config.train_batch_size, shuffle=False,
                                                      sampler=sampler)

        num_steps = self.config.num_steps // world_size

        self.train(model=model, optimizer=optimizer, train_loader=train_loader,
                   num_steps=num_steps, device=device, rank=rank)

        self.cleanup_distributed()

    def init_scheduler(self, scheduler_type, optimizer, num_warmup_steps, num_steps):
        if scheduler_type == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_steps
            )
        elif scheduler_type == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_steps
            )
        elif scheduler_type == "constant":
            scheduler = get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
            )
        elif scheduler_type == "polynomial":
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_steps
            )
        elif scheduler_type == "inverse_sqrt":
            scheduler = get_inverse_sqrt_schedule(
                optimizer,
                num_warmup_steps=num_warmup_steps,
            )
        else:
            raise ValueError(
                f"Invalid scheduler_type value: '{scheduler_type}' \n Supported scheduler types: 'cosine', 'linear', 'constant', 'polynomial', 'inverse_sqrt'"
            )
        return scheduler

    def train(self, model, optimizer, train_loader, num_steps, device='cuda', rank=None):
        model.train()
        pbar = tqdm(range(num_steps))

        warmup_ratio = self.config.warmup_ratio
        eval_every = self.config.eval_every
        save_total_limit = self.config.save_total_limit
        log_dir = self.config.log_dir
        val_data_dir = self.config.val_data_dir

        num_warmup_steps = int(num_steps * warmup_ratio) if warmup_ratio < 1 else int(warmup_ratio)

        scheduler = self.init_scheduler(self.config.scheduler_type, optimizer, num_warmup_steps, num_steps)
        iter_train_loader = iter(train_loader)
        scaler = torch.cuda.amp.GradScaler()

        for step in pbar:
            optimizer.zero_grad()

            try:
                x = next(iter_train_loader)
            except StopIteration:
                iter_train_loader = iter(train_loader)
                x = next(iter_train_loader)

            for k, v in x.items():
                if isinstance(v, torch.Tensor):
                    x[k] = v.to(device)

            # try:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                loss = model(x)

            if torch.isnan(loss).any():
                print("Warning: NaN loss detected")
                continue

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            # except Exception as e:
                # print(f"Error: {e}")
                # torch.cuda.empty_cache()
                # continue

            description = f"step: {step} | epoch: {step // len(train_loader)} | loss: {loss.item():.2f}"
            pbar.set_description(description)

            if (step + 1) % eval_every == 0:
                if rank is None or rank == 0:
                    checkpoint = f'model_{step + 1}'
                    save_top_k_checkpoints(model, log_dir, checkpoint, save_total_limit)
                    if val_data_dir != "none":
                        get_for_all_path(model, step, log_dir, val_data_dir)
                    model.train()

    def run(self):
        with open(self.config.train_data, 'r') as f:
            data = json.load(f)

        if torch.cuda.device_count() > 1 and self.allow_distributed:
            world_size = torch.cuda.device_count()
            mp.spawn(self.train_dist, args=(world_size, data), nprocs=world_size, join=True)
        else:
            model, optimizer = self.setup_model_and_optimizer()

            train_loader = model.create_dataloader(data, batch_size=self.config.train_batch_size, shuffle=True)

            self.train(model, optimizer, train_loader, num_steps=self.config.num_steps, device=self.device)


def create_parser():
    parser = argparse.ArgumentParser(description="Span-based NER")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument('--log_dir', type=str, default='logs', help='Path to the log directory')
    parser.add_argument('--allow_distributed', type=bool, default=False,
                        help='Whether to allow distributed training if there are more than one GPU available')
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    # Load the config file
    with open(args.config, 'r') as f:
        yaml_cfg = om.OmegaConf.load(f)

    # Create a DictConfig from the loaded YAML
    cfg = om.OmegaConf.create(yaml_cfg)

    # Extract the gliner_train configuration
    gliner_config = om.OmegaConf.to_container(cfg.tasks.gliner_train, resolve=True)

    # Convert the extracted config to a namespace for compatibility with existing code
    gliner_config = argparse.Namespace(**gliner_config)

    # Override log_dir from command line argument
    gliner_config.log_dir = args.log_dir


    trainer = Trainer(cfg, gliner_config, allow_distributed=args.allow_distributed,
                      device='cuda' if torch.cuda.is_available() else 'cpu')
    trainer.run()
