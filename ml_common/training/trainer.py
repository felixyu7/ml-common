"""Generic trainer class for PyTorch models."""

import csv
import math
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Tuple
from torch.utils.data import DataLoader
from tqdm import tqdm
from contextlib import nullcontext


class Trainer:
    """
    Generic trainer for PyTorch models with mixed precision, checkpointing, and logging.

    Supports both W&B and CSV logging. Provides flexible loss and metric computation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        cfg: Dict[str, Any],
        loss_fn: Optional[Callable] = None,
        metric_fn: Optional[Callable] = None,
        batch_prep_fn: Optional[Callable] = None,
        use_wandb: bool = False
    ):
        """
        Initialize Trainer.

        Args:
            model: PyTorch model
            device: Device to train on
            cfg: Config dict with training_options and project_save_dir
            loss_fn: Loss function (preds, labels) -> loss
            metric_fn: Optional metric function (preds, labels) -> dict
            batch_prep_fn: Optional batch preparation (coords, features, labels) -> (coords, features, batch_ids, labels)
            use_wandb: Whether to use W&B logging
        """
        self.model = model
        self.device = device
        self.cfg = cfg
        self.use_wandb = use_wandb
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.batch_prep_fn = batch_prep_fn or self._default_batch_prep

        # Extract training options
        training_opts = cfg['training_options']
        self.epochs = training_opts['epochs']
        self.lr = training_opts['lr']
        self.weight_decay = training_opts['weight_decay']
        self.batch_size = training_opts['batch_size']
        self.precision = training_opts.get('precision', 'fp32')
        self.save_epochs = training_opts.get('save_epochs', 5)
        self.grad_clip = training_opts.get('grad_clip', 1.0)

        # Setup optimizer, mixed precision, and logging. The LR scheduler is built
        # in fit() once steps-per-epoch is known, since it steps per optimizer step.
        self.optimizer = self._setup_optimizer(training_opts)
        self.scheduler = None
        self._pending_scheduler_state = None
        self.warmup_steps_cfg = training_opts.get('warmup_steps', None)
        self.warmup_ratio = training_opts.get('warmup_ratio', 0.05)
        self.min_lr_ratio = training_opts.get('min_lr_ratio', 0.0)
        self.steps_per_epoch_cfg = training_opts.get('steps_per_epoch', None)
        self.scaler = self._setup_mixed_precision()
        self._setup_logging()

        # Training state
        self.current_epoch = 0
        self.current_step = 0

        # Best-checkpoint selection (default: minimize val_loss)
        self.best_metric_key = training_opts.get('best_metric_key', 'val_loss')
        self.best_metric_mode = training_opts.get('best_metric_mode', 'min')
        if self.best_metric_mode not in ('min', 'max'):
            raise ValueError(
                f"best_metric_mode must be 'min' or 'max', got '{self.best_metric_mode}'"
            )
        self.best_metric_value = float('inf') if self.best_metric_mode == 'min' else float('-inf')

    @staticmethod
    def _precision_to_dtype(precision: str) -> Optional[torch.dtype]:
        """Convert precision string to dtype. Only accepts 'fp16', 'bf16', 'fp32'."""
        precision = precision.lower()
        if precision == 'fp16':
            return torch.float16
        elif precision == 'bf16':
            return torch.bfloat16
        elif precision == 'fp32':
            return None  # No autocast for fp32
        else:
            raise ValueError(
                f"Invalid precision '{precision}'. Must be one of: 'fp16', 'bf16', 'fp32'"
            )

    def _setup_optimizer(self, training_opts: Dict[str, Any]) -> torch.optim.Optimizer:
        """AdamW with decoupled weight decay: 2D+ params (matrices, embeddings) are
        decayed; 1D params (RMSNorm gains, biases, LayerScale) are not — the
        standard transformer convention, which avoids shrinking normalization
        gains and bias terms."""
        decay, no_decay = [], []
        for _name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.ndim >= 2 else no_decay).append(p)
        param_groups = [
            {'params': decay, 'weight_decay': self.weight_decay},
            {'params': no_decay, 'weight_decay': 0.0},
        ]
        # fused: single multi-tensor kernel on CUDA (~1-2 ms/step at 8-layer scale)
        return torch.optim.AdamW(param_groups, lr=self.lr,
                                 fused=(self.device.type == 'cuda'))

    def _build_scheduler(self, steps_per_epoch: int) -> torch.optim.lr_scheduler.LambdaLR:
        """Per-step linear-warmup + cosine-decay schedule over the whole run.

        Stepped once per optimizer step (not per epoch), so warmup ramps smoothly
        and cosine is a continuous curve (cf. HF get_cosine_schedule_with_warmup).
        Warmup length is ``warmup_steps`` if set, else ``warmup_ratio`` of total
        steps; cosine decays from the peak LR to ``min_lr_ratio * peak``
        (default 0)."""
        total_steps = max(1, self.epochs * steps_per_epoch)
        if self.warmup_steps_cfg is not None:
            warmup_steps = int(self.warmup_steps_cfg)
        else:
            warmup_steps = int(self.warmup_ratio * total_steps)
        warmup_steps = max(0, min(warmup_steps, total_steps - 1))
        min_ratio = self.min_lr_ratio

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return float(step) / float(warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
            return min_ratio + (1.0 - min_ratio) * cosine

        print(
            f"LR schedule: {warmup_steps}/{total_steps} warmup steps "
            f"(ratio {warmup_steps / total_steps:.3f}), cosine to {min_ratio:g}*peak"
        )
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _setup_mixed_precision(self) -> Optional[torch.amp.GradScaler]:
        """Setup mixed precision training."""
        self.amp_device = 'cuda' if self.device.type == 'cuda' else 'cpu'
        self.amp_dtype = self._precision_to_dtype(self.precision)

        # Fall back to fp16 if bf16 not supported on CUDA
        if self.amp_device == 'cuda' and self.amp_dtype is torch.bfloat16:
            if not torch.cuda.is_bf16_supported():
                print('Warning: CUDA bf16 not supported. Falling back to fp16.')
                self.amp_dtype = torch.float16

        # Only use scaler for CUDA fp16
        if self.amp_device == 'cuda' and self.amp_dtype is torch.float16:
            return torch.amp.GradScaler(self.amp_device)
        return None

    def _setup_logging(self):
        """Setup logging directories and CSV writer."""
        self.save_dir = Path(self.cfg['project_save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.save_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)

        if not self.use_wandb:
            self.csv_file = self.save_dir / 'metrics.csv'
            self.csv_writer = None
            self.csv_file_handle = None

    def _get_autocast_context(self, precision: str = None):
        """Get autocast context manager for the given precision."""
        if precision is None:
            precision = self.precision

        dtype = self._precision_to_dtype(precision)
        device = 'cuda' if self.device.type == 'cuda' else 'cpu'

        # Fall back to fp16 if bf16 not supported
        if device == 'cuda' and dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
            dtype = torch.float16

        return torch.amp.autocast(device, dtype=dtype) if dtype is not None else nullcontext()

    def _default_batch_prep(
        self, coords_b: torch.Tensor, features_b: torch.Tensor, labels_b: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Default batch preparation: extract batch indices from coords."""
        assert coords_b.dim() == 2 and coords_b.size(-1) >= 4
        batch_ids = coords_b[:, 0].long()
        coords = coords_b[:, 1:4]
        feats = features_b if (features_b is not None and features_b.numel() > 0) else None
        return coords, feats, batch_ids, labels_b

    def _init_csv_writer(self, metrics: Dict[str, float]):
        """Initialize CSV writer with appropriate fields."""
        base_fields = ['epoch', 'step', 'train_loss', 'train_loss_epoch', 'val_loss', 'learning_rate']
        additional = [k for k in metrics.keys() if k not in base_fields]
        all_fields = base_fields + additional

        self.csv_file_handle = open(self.csv_file, 'w', newline='')
        self.csv_writer = csv.DictWriter(
            self.csv_file_handle, fieldnames=all_fields, extrasaction='ignore'
        )
        self.csv_writer.writeheader()

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to W&B or CSV."""
        if self.use_wandb:
            try:
                import wandb
                wandb.log(metrics, step=step)
            except ImportError:
                pass
        else:
            if self.csv_writer is None:
                self._init_csv_writer(metrics)

            row = {'epoch': self.current_epoch, 'step': step or 0, **metrics}
            self.csv_writer.writerow(row)
            self.csv_file_handle.flush()

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        if self.loss_fn is None:
            raise ValueError("loss_fn must be provided")

        self.model.train()
        running_loss = 0.0
        try:
            num_batches = len(train_loader)
        except TypeError:
            num_batches = None
        batches_seen = 0

        pbar = tqdm(
            train_loader,
            desc=f'Epoch {self.current_epoch+1}/{self.epochs}',
            leave=True,
            dynamic_ncols=True,
            total=num_batches,
            position=1,
        )

        for batch_idx, (coords, features, labels) in enumerate(pbar):
            coords = coords.to(self.device, non_blocking=True)
            features = features.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            coords, feats, batch_ids, labels = self.batch_prep_fn(coords, features, labels)
            batches_seen += 1

            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            with self._get_autocast_context():
                preds = self.model(coords, feats, batch_ids=batch_ids)
                loss = self.loss_fn(preds, labels)

            # Backward pass with gradient scaling
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            # Per-step LR schedule update (after the optimizer step)
            self.scheduler.step()

            # Accumulate on-device; loss.item() would sync the stream every
            # step, so the host only reads it back on the logging cadences.
            # .float() is essential: a bf16 running sum saturates at +-1024
            # (increments fall below the ulp), silently corrupting the average.
            running_loss = running_loss + loss.detach().float()
            if self.current_step % 20 == 0:
                pbar.set_postfix({
                    'Loss': f'{loss.item():.6f}',
                    'LR': f'{self.scheduler.get_last_lr()[0]:.2e}',
                    'Avg': f'{float(running_loss)/(batch_idx+1):.6f}'
                })
            if self.current_step % 50 == 0:
                metrics = {
                    'train_loss': loss.item(),
                    'learning_rate': self.scheduler.get_last_lr()[0]
                }
                if self.use_wandb and hasattr(self.loss_fn, 'current_weights'):
                    vmf_weight, _ = self.loss_fn.current_weights()
                    metrics['vmf_weight'] = float(vmf_weight)
                self.log_metrics(metrics, step=self.current_step)
            self.current_step += 1

        avg_loss = float(running_loss) / max(1, batches_seen)
        # Distinct key from the per-step instantaneous 'train_loss': logging the
        # epoch mean under the same key paints a spurious spike onto the
        # per-step chart at every epoch boundary.
        return {'train_loss_epoch': avg_loss}

    def _print_profiling_results(self, forward_times: list, total_time: float):
        """Print profiling statistics."""
        total_forward = sum(forward_times)
        avg_forward = total_forward / len(forward_times) if forward_times else 0

        print("\n--- Inference Profiling ---")
        print(f"Total runtime (incl. I/O): {total_time:.4f}s")
        print(f"Total forward time: {total_forward:.4f}s")
        if self.device.type == 'cuda':
            print(f"Avg forward time per batch: {avg_forward * 1000:.4f}ms")
            peak_mem_gb = torch.cuda.max_memory_allocated(self.device) / (1024**3)
            print(f"Peak CUDA memory: {peak_mem_gb:.4f}GB")
        else:
            std_forward = np.std(forward_times) if forward_times else 0
            print(f"Avg forward time per batch: {avg_forward * 1000:.4f}ms, std: {std_forward * 1000:.4f}ms")
        print("---------------------------\n")

    def validate(
        self, val_loader: DataLoader, save_predictions: bool = False, profile: bool = False
    ) -> Dict[str, float]:
        """Run validation."""
        if self.loss_fn is None:
            raise ValueError("loss_fn must be provided")

        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []
        forward_times = []

        if profile and self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)

        start_time = time.time()

        try:
            val_total = len(val_loader)
        except TypeError:
            val_total = None

        with torch.no_grad():
            val_pbar = tqdm(
                val_loader,
                desc='Validation',
                leave=True,
                dynamic_ncols=True,
                total=val_total,
                position=2,
            )
            for coords, features, labels in val_pbar:
                coords = coords.to(self.device, non_blocking=True)
                features = features.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                coords, feats, batch_ids, labels = self.batch_prep_fn(coords, features, labels)

                with self._get_autocast_context():
                    if profile:
                        t0 = time.time()
                    preds = self.model(coords, feats, batch_ids=batch_ids)
                    if profile:
                        if self.device.type == 'cuda':
                            torch.cuda.synchronize()
                        forward_times.append(time.time() - t0)

                    loss = self.loss_fn(preds, labels)
                loss_value = loss.item()
                total_loss += loss_value
                # fp32 accumulation: the best-checkpoint metric is computed from
                # these, and bf16 autocast outputs would quantize it (~0.4%).
                all_preds.append(preds.float().cpu())
                all_labels.append(labels.float().cpu())
                val_pbar.set_postfix({'Val Loss': f'{loss_value:.6f}'})

        if profile:
            self._print_profiling_results(forward_times, time.time() - start_time)

        batch_count = len(all_preds)
        avg_val_loss = total_loss / max(1, batch_count)

        metrics = {'val_loss': avg_val_loss}

        if batch_count == 0:
            if save_predictions:
                print('Warning: No validation batches processed; skipping prediction export.')
            return metrics

        preds_tensor = torch.cat(all_preds, dim=0)
        labels_tensor = torch.cat(all_labels, dim=0)

        if save_predictions:
            predictions_file = self.save_dir / 'results.npz'
            np.savez(predictions_file, predictions=preds_tensor.float().numpy(), labels=labels_tensor.float().numpy())
            print(f"Saved predictions to: {predictions_file}")

        if self.metric_fn is not None:
            task_metrics = self.metric_fn(preds_tensor, labels_tensor)
            metrics.update({f'val_{k}': v for k, v in task_metrics.items()})

        return metrics

    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False, is_final: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'step': self.current_step,
            'best_metric_value': self.best_metric_value,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'cfg': self.cfg
        }

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        # Regular checkpoint
        if self.current_epoch % self.save_epochs == 0 or is_final:
            if is_final:
                checkpoint_path = self.checkpoint_dir / f'final-checkpoint-{time.strftime("%Y%m%d-%H%M%S")}.pt'
            else:
                checkpoint_path = self.checkpoint_dir / f'epoch-{self.current_epoch:02d}-checkpoint-{time.strftime("%Y%m%d-%H%M%S")}.pt'

            torch.save(checkpoint, checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')

            if self.use_wandb:
                try:
                    import wandb
                    artifact_name = f"{wandb.run.name or wandb.run.id}-checkpoint"
                    artifact = wandb.Artifact(artifact_name, type='model', metadata={'epoch': self.current_epoch, **metrics})
                    artifact.add_file(str(checkpoint_path))
                    wandb.log_artifact(artifact)
                except Exception as e:
                    print(f"Could not save wandb artifact: {e}")

        # Best model
        if is_best:
            best_path = self.checkpoint_dir / 'best-checkpoint.pt'
            torch.save(checkpoint, best_path)
            print(f'Best model saved: {best_path}')

    def load_checkpoint(self, checkpoint_path: str, resume_training: bool = False):
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if resume_training:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Scheduler is built in fit() once steps-per-epoch is known; stash its
            # state there and apply it after construction.
            self._pending_scheduler_state = checkpoint.get('scheduler_state_dict')
            # The checkpoint's epoch is the just-FINISHED one: resume at the
            # next, and restore step/best so wandb steps stay monotonic and
            # best-checkpoint.pt can't be overwritten by a worse epoch.
            self.current_epoch = checkpoint['epoch'] + 1
            self.current_step = checkpoint.get('step', 0)
            if 'best_metric_value' in checkpoint:
                self.best_metric_value = checkpoint['best_metric_value']

            if self.scaler is not None and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        """Train model."""
        print(f"Starting training for {self.epochs} epochs...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Build the per-step LR scheduler now that steps-per-epoch is known.
        try:
            steps_per_epoch = len(train_loader)
        except TypeError:
            steps_per_epoch = self.steps_per_epoch_cfg
            if steps_per_epoch is None:
                raise ValueError(
                    "train_loader has no length; set training_options.steps_per_epoch "
                    "so the LR scheduler knows the total step count."
                )
        self.scheduler = self._build_scheduler(steps_per_epoch)
        if self._pending_scheduler_state is not None:
            self.scheduler.load_state_dict(self._pending_scheduler_state)
            self._pending_scheduler_state = None

        epoch_pbar = tqdm(
            range(self.current_epoch, self.epochs),
            desc='Training Progress',
            position=0,
            dynamic_ncols=True
        )

        for epoch in epoch_pbar:
            self.current_epoch = epoch
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            epoch_metrics = {**train_metrics, **val_metrics}
            # Surface a silent compile/packed fallback (throughput downgrade).
            epoch_metrics['encoder_compiled'] = float(
                getattr(self.model, 'encoder_compiled', True))
            self.log_metrics(epoch_metrics)

            current_val = epoch_metrics.get(self.best_metric_key)
            if current_val is None:
                is_best = False
            elif self.best_metric_mode == 'min':
                is_best = current_val < self.best_metric_value
            else:
                is_best = current_val > self.best_metric_value
            if is_best:
                self.best_metric_value = current_val

            self.save_checkpoint(epoch_metrics, is_best)
            epoch_pbar.set_postfix({
                'Train Loss': f'{train_metrics["train_loss_epoch"]:.6f}',
                'Val Loss': f'{val_metrics["val_loss"]:.6f}'
            })

        self.save_checkpoint({f'best_{self.best_metric_key}': self.best_metric_value}, is_final=True)
        self._log_best_checkpoint_artifact()
        print(f"Training completed! Best {self.best_metric_key} ({self.best_metric_mode}): {self.best_metric_value:.6f}")

    def _log_best_checkpoint_artifact(self):
        """Upload best-checkpoint.pt as a wandb artifact at end of training."""
        if not self.use_wandb:
            return
        best_path = self.checkpoint_dir / 'best-checkpoint.pt'
        if not best_path.exists():
            return
        try:
            import wandb
            artifact_name = f"{wandb.run.name or wandb.run.id}-best"
            artifact = wandb.Artifact(
                artifact_name,
                type='model',
                metadata={
                    'best_metric_key': self.best_metric_key,
                    'best_metric_mode': self.best_metric_mode,
                    'best_metric_value': self.best_metric_value,
                },
            )
            artifact.add_file(str(best_path))
            wandb.log_artifact(artifact, aliases=['best'])
        except Exception as e:
            print(f"Could not save best wandb artifact: {e}")

    def test(self, test_loader: DataLoader):
        """Run test evaluation."""
        print("Running test evaluation...")
        test_metrics = self.validate(test_loader, save_predictions=True, profile=True)

        print("Test Results:")
        for key, value in test_metrics.items():
            print(f"{key}: {value:.6f}")

        self.log_metrics(test_metrics)

        return test_metrics
