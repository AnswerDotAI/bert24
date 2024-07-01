import torch
import math

from composer.trainer import Trainer
from composer.trainer.trainer import _adjust_device_train_microbatch_size, _is_cuda_oom, log
import textwrap
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    TextIO,
    Tuple,
    Union,
)
import torch.distributed
import torch.utils.data
from torch.optim.lr_scheduler import LRScheduler
from composer.core import (
    Algorithm,
    AlgorithmPass,
    Callback,
    DataSpec,
    Evaluator,
    Event,
    Precision,
    State,
    Time
)
from composer.devices import Device
from composer.loggers import LoggerDestination
from composer.models import ComposerModel
from composer.optim import ComposerScheduler, ConstantWithWarmupScheduler
from composer.profiler import Profiler
from composer.trainer.dist_strategy import (
    DDPSyncStrategy,
)
from composer.utils import (
    ObjectStore,
    dist,
)

class CustomTrainer(Trainer):
    def __init__(
        self,
        *,
        # The Model
        model: ComposerModel,

        # Train Dataloader
        train_dataloader: Optional[Union[Iterable, DataSpec, Dict[str, Any]]] = None,
        train_dataloader_label: str = 'train',
        train_subset_num_batches: int = -1,
        spin_dataloaders: bool = True,

        # Stopping Condition
        max_duration: Optional[Union[int, str, Time]] = None,

        # Algorithms
        algorithms: Optional[Union[Algorithm, Sequence[Algorithm]]] = None,

        # Engine Pass Registration
        algorithm_passes: Optional[Union[AlgorithmPass,
                                         Tuple[AlgorithmPass, int],
                                         Sequence[Union[AlgorithmPass, Tuple[AlgorithmPass, int]]],
                                        ]] = None,

        # Optimizers and Scheduling
        optimizers: Optional[torch.optim.Optimizer] = None,
        schedulers: Optional[Union[ComposerScheduler,
                                   LRScheduler,
                                   Sequence[Union[ComposerScheduler,
                                                  LRScheduler,
                                                 ]],
                                  ]] = None,
        scale_schedule_ratio: float = 1.0,
        step_schedulers_every_batch: Optional[bool] = None,

        # Evaluators
        eval_dataloader: Optional[Union[Iterable, DataSpec, Evaluator, Sequence[Evaluator]]] = None,
        eval_interval: Union[int, str, Time, Callable[[State, Event], bool]] = 1,
        eval_subset_num_batches: int = -1,

        # Callbacks and Logging
        callbacks: Optional[Union[Callback, Sequence[Callback]]] = None,
        loggers: Optional[Union[LoggerDestination, Sequence[LoggerDestination]]] = None,
        run_name: Optional[str] = None,
        progress_bar: bool = True,
        log_to_console: bool = False,
        console_stream: Union[str, TextIO] = 'stderr',
        console_log_interval: Union[int, str, Time] = '1ba',
        log_traces: bool = False,
        auto_log_hparams: bool = False,

        # Load Checkpoint
        load_path: Optional[str] = None,
        load_object_store: Optional[Union[ObjectStore, LoggerDestination]] = None,
        load_weights_only: bool = False,
        load_strict_model_weights: bool = True,
        load_progress_bar: bool = True,
        load_ignore_keys: Optional[Union[List[str], Callable[[Dict], None]]] = None,
        load_exclude_algorithms: Optional[List[str]] = None,

        # Save Checkpoint
        save_folder: Optional[str] = None,
        save_filename: str = 'ep{epoch}-ba{batch}-rank{rank}.pt',
        save_latest_filename: Optional[str] = 'latest-rank{rank}.pt',
        save_overwrite: bool = False,
        save_interval: Union[str, int, Time, Callable[[State, Event], bool]] = '1ep',
        save_weights_only: bool = False,
        save_ignore_keys: Optional[Union[List[str], Callable[[Dict], None]]] = None,
        save_num_checkpoints_to_keep: int = -1,
        save_metrics: bool = False,

        # Graceful Resumption
        autoresume: bool = False,

        # DeepSpeed
        deepspeed_config: Optional[Dict[str, Any]] = None,
        fsdp_config: Optional[Dict[str, Any]] = None,
        fsdp_auto_wrap: bool = True,

        # System/Numerics
        device: Optional[Union[str, Device]] = None,
        precision: Optional[Union[str, Precision]] = None,
        precision_config: Optional[Dict[str, Any]] = None,
        device_train_microbatch_size: Optional[Union[int, float, str]] = None,

        # Reproducibility
        seed: Optional[int] = None,
        deterministic_mode: bool = False,

        # Distributed Training
        dist_timeout: float = 300.0,
        ddp_sync_strategy: Optional[Union[str, DDPSyncStrategy]] = None,

        # Profiling
        profiler: Optional[Profiler] = None,

        # Python logging
        python_log_level: Optional[str] = None,

        # compile config for PyTorch 2.0 or higher
        compile_config: Optional[Dict[str, Any]] = None,

        # batch rampup
        batch_rampup: Optional[Union[str, Time]] = None,
        initial_per_device_train_batch_size: Optional[int] = None
    ):
        super().__init__(model=model, train_dataloader=train_dataloader, train_dataloader_label=train_dataloader_label, train_subset_num_batches=train_subset_num_batches, spin_dataloaders=spin_dataloaders, max_duration=max_duration, algorithms=algorithms, algorithm_passes=algorithm_passes, optimizers=optimizers, schedulers=schedulers, scale_schedule_ratio=scale_schedule_ratio, step_schedulers_every_batch=step_schedulers_every_batch, eval_dataloader=eval_dataloader, eval_interval=eval_interval, eval_subset_num_batches=eval_subset_num_batches, callbacks=callbacks, loggers=loggers, run_name=run_name, progress_bar=progress_bar, log_to_console=log_to_console, console_stream=console_stream, console_log_interval=console_log_interval, log_traces=log_traces, auto_log_hparams=auto_log_hparams, load_path=load_path, load_object_store=load_object_store, load_weights_only=load_weights_only, load_strict_model_weights=load_strict_model_weights, load_progress_bar=load_progress_bar, load_ignore_keys=load_ignore_keys, load_exclude_algorithms=load_exclude_algorithms, save_folder=save_folder, save_filename=save_filename, save_latest_filename=save_latest_filename, save_overwrite=save_overwrite, save_interval=save_interval, save_weights_only=save_weights_only, save_ignore_keys=save_ignore_keys, save_num_checkpoints_to_keep=save_num_checkpoints_to_keep, save_metrics=save_metrics, autoresume=autoresume, deepspeed_config=deepspeed_config, fsdp_config=fsdp_config, fsdp_auto_wrap=fsdp_auto_wrap, device=device, precision=precision, precision_config=precision_config, device_train_microbatch_size=device_train_microbatch_size, seed=seed, deterministic_mode=deterministic_mode, dist_timeout=dist_timeout, ddp_sync_strategy=ddp_sync_strategy, profiler=profiler, python_log_level=python_log_level, compile_config=compile_config)
        if batch_rampup:
            self.batch_rampup_scheduler = ConstantWithWarmupScheduler(t_warmup=batch_rampup)
            if initial_per_device_train_batch_size is None:
                max_n_subbatches = self.state.train_dataloader.batch_size // self.state.device_train_microbatch_size
            else:
                assert self.state.train_dataloader.batch_size % initial_per_device_train_batch_size == 0, "Final batch size must be a multiple of initial batch size"
                max_n_subbatches = self.state.train_dataloader.batch_size // initial_per_device_train_batch_size
            self.n_rampup_steps = math.log(max_n_subbatches, 2)
            assert self.n_rampup_steps > 0, "The given batch can not be split into several subbatches for rampup"
        else:
            assert initial_per_device_train_batch_size is None, "initial_per_device_train_batch_size can't be set without specifying batch_rampup"
            self.batch_rampup_scheduler = None

    def _train_batch(self, use_grad_scaling: bool) -> Dict[str, torch.Tensor]:
        """Compute loss by training on a full batch of data.

        Adaptively change microbatch size if enabled to maximize GPU usage.

        Args:
            use_grad_scaling (bool): Enables gradient scaling.

        Returns:
            Dict[str, torch.Tensor]: a dictionary containing the total loss and individual losses if available.
        """
        assert self._train_data_spec is not None, 'The train data spec should be set on __init__ or fit()'

        # Cache the device batch, because `self.state.batch` gets overridden in microbatching loop.
        # Any in-place changes to a microbatch will be reflected in the device batch.
        device_batch = self.state.batch

        # Retry until we successfully complete training and return loss
        while True:
            # Reset train_metrics on every batch
            # Placing reset here ensures that if auto grad accum catches an OOM, incomplete metric state is cleared
            if self.state.train_metrics is not None:  # pyright: ignore[reportUnnecessaryComparison]
                for metric in self.state.train_metrics.values():
                    metric.reset()


            found_cuda_oom = 0  # int since bool BOR not supported on all torch.distributed backends
            try:
                assert self.state.scaler is not None
                assert self.state.device_train_microbatch_size is not None
                microbatches = self._train_data_spec.split_batch(device_batch, self.state.device_train_microbatch_size)
                # During batch rampup, we devide the batch into subbatches. After each subbatch, gradients are calculated and the weights are updated.
                # please note that the loss reported during rampup corresponds to the average of the loss of all subbatches, so there is only one reported step
                # in the logs per final batch size, even though there are n_subbatches gradient updates/steps occuring per final batch size.
                n_subbatches = self._get_batch_rampup_divider()
                n_microbatches_per_subbatch = len(microbatches) // n_subbatches
                loss_accumulator = 0
                self.logger.log_metrics({'trainer/batch_rampup_factor': n_subbatches})
                self.logger.log_metrics({'trainer/device_train_microbatch_size': self.state.device_train_microbatch_size})

                for i in range(n_subbatches):
                    total_loss_dict = {
                        'loss/train/total': self.state.device.tensor_to_device(torch.zeros(size=(1,))),
                        }
                    microbatches_subbatch = microbatches[i*n_microbatches_per_subbatch:(i+1)*n_microbatches_per_subbatch]

                    if self._use_closures():
                        for optimizer in self.state.optimizers:
                            if use_grad_scaling:
                                self.state.scaler.step(
                                    optimizer,
                                    closure=lambda loss_dict=total_loss_dict,
                                    **kwargs: self._train_microbatches(microbatches_subbatch, loss_dict, **kwargs),
                                )
                            else:
                                optimizer.step(
                                    closure=lambda loss_dict=total_loss_dict,
                                    **kwargs: self._train_microbatches(microbatches_subbatch, loss_dict, **kwargs).item(),
                                )
                    else:
                        self._train_microbatches(microbatches_subbatch, total_loss_dict)
                        if not self.state.deepspeed_enabled:
                            for optimizer in self.state.optimizers:
                                if use_grad_scaling:
                                    self.state.scaler.step(optimizer)
                                else:
                                    optimizer.step()
                    loss_accumulator +=  total_loss_dict["loss/train/total"] / n_subbatches
            except RuntimeError as e:
                if self.state.auto_microbatching and _is_cuda_oom(e):
                    log.debug((f"Rank {dist.get_global_rank()} OOM'd."))
                    found_cuda_oom = 1
                elif self.state.auto_microbatching and ('cuda' in str(e).lower() or 'c10' in str(e).lower()):
                    raise RuntimeError(
                        textwrap.dedent(
                            'Encountered non-addressable cuda error while using auto microbatching. '
                            'If this repeatedly occurs, set `device_train_microbatch_size` manually.',
                        ),
                    ) from e
                else:
                    raise

            if self.state.auto_microbatching:
                all_ranks_finished = False
                while not all_ranks_finished:
                    # Propagate across all ranks if any rank hit CUDA OOM
                    found_cuda_oom_tensor = self.state.device.tensor_to_device(
                        torch.tensor([found_cuda_oom], dtype=torch.uint8),
                    )
                    dist.all_reduce(found_cuda_oom_tensor, reduce_operation='MAX')
                    found_cuda_oom = found_cuda_oom_tensor.item()
                    # Check if any rank is still not done with the batch. This may happen if only a
                    # subset of ranks OOM, leaving some batches still in the forward pass
                    all_ranks_finished_tensor = self.state.device.tensor_to_device(torch.tensor([1], dtype=torch.uint8))
                    dist.all_reduce(all_ranks_finished_tensor, reduce_operation='MIN')
                    all_ranks_finished = all_ranks_finished_tensor.item() == 1
                if found_cuda_oom == 1:
                    _adjust_device_train_microbatch_size(self.state)
                    # Skip return and rerun after handling oom
                    continue
            # Log microbatch and return loss if we've completed without OOMing.
            assert self.state.device_train_microbatch_size is not None
            self.logger.log_metrics({'trainer/device_train_microbatch_size': self.state.device_train_microbatch_size})
            self.first_batch_complete = True
            total_loss_dict = {
                'loss/train/total': loss_accumulator,
            }
            return total_loss_dict
    
    def _get_batch_rampup_divider(self) -> int:
        """
        This function returns how many subbatches the batch should be divided into during batch ramup
        Example: If we have 3 rampup steps:
        During stage 0 (first 1/3 of rampup) each batch will be divided into 8 subbatches
        During stage 1 (second 1/3 of rampup) each batch will be divided into 4 subbatches
        During stage 2 (last 1/3 of rampup) each batch will be divided into 2 subbatches
        After warmup stages, we won't partition the batches into subbatches and thus there is 1 subbatch
        """
        if self.batch_rampup_scheduler:
            percentage_of_rampup = self.batch_rampup_scheduler(self.state)
            if percentage_of_rampup == 1:
                return 1
            else:
                rampup_step = min(int(self.n_rampup_steps * percentage_of_rampup), self.n_rampup_steps - 1)
                rampup_factor = 2 ** int(self.n_rampup_steps - rampup_step)
                return rampup_factor
        else:
            return 1
