import time
import torch
import torch.distributed as dist
from lightning.pytorch import callbacks


class GlobalMeanBatchTimeCallback(callbacks.Callback):
    """Global mean batch time calculator.

    for each GPU, compute total_batch_duration and n_batch
    sum over all GPUs: global_total_batch_duration and global_n_batch
    global_mean_batch_time = global_total_batch_duration / global_n_batch

    e.g. 2 GPUs; for each GPU, 2 batches take 200ms
    global_mean_batch_time = (200ms + 200ms) / (2 + 2) = 100ms
    """

    def __init__(self, reset_per_epoch: bool = False):
        super().__init__()
        # all local: results on a single GPU
        self.local_train_batch_start_time = 0
        self.local_total_train_batch_duration = 0.0
        self.local_n_train_batch = 0
        self.local_val_batch_start_time = 0
        self.local_total_val_batch_duration = 0.0
        self.local_n_val_batch = 0
        self.reset_per_epoch = reset_per_epoch

    def on_train_batch_start(self, *args, **kwargs):
        self.local_train_batch_start_time = time.time()

    def on_train_batch_end(self, *args, **kwargs):
        duration = time.time() - self.local_train_batch_start_time
        self.local_total_train_batch_duration += duration
        self.local_n_train_batch += 1

    def on_train_epoch_end(self, trainer, pl_module):
        total_duration = self.local_total_train_batch_duration
        n_batch = self.local_n_train_batch
        if dist.is_available() and dist.is_initialized():
            # synchronize across all processes (all GPUs)
            total_duration_tensor = torch.tensor(
                [self.local_total_train_batch_duration, self.local_n_train_batch],
                dtype=torch.float,
                device=pl_module.device,
            )
            dist.all_reduce(total_duration_tensor, op=dist.ReduceOp.SUM)
            total_duration, n_batch = total_duration_tensor.tolist()

        # compute global average duration
        avg_duration = (total_duration / n_batch) if n_batch > 0 else 0.0

        if trainer.is_global_zero:
            e = pl_module.current_epoch
            te = trainer.max_epochs
            print(
                f"Epoch {e}/{te}: "
                f"global_mean_train_batch_time={avg_duration * 1000:.2f} ms"
            )

        if self.reset_per_epoch:
            self.local_total_train_batch_duration = 0.0
            self.local_n_train_batch = 0

    def on_validation_batch_start(self, *args, **kwargs):
        self.local_val_batch_start_time = time.time()

    def on_validation_batch_end(self, *args, **kwargs):
        duration = time.time() - self.local_val_batch_start_time
        self.local_total_val_batch_duration += duration
        self.local_n_val_batch += 1

    def on_validation_epoch_end(self, trainer, pl_module):
        # synchronize across all processes
        total_duration = self.local_total_val_batch_duration
        n_batch = self.local_n_val_batch
        if dist.is_available() and dist.is_initialized():
            total_duration_tensor = torch.tensor(
                [self.local_total_val_batch_duration, self.local_n_val_batch],
                dtype=torch.float32,
                device=pl_module.device,
            )
            dist.all_reduce(total_duration_tensor, op=dist.ReduceOp.SUM)
            total_duration, n_batch = total_duration_tensor.tolist()

        # compute global average duration
        avg_duration = (total_duration / n_batch) if n_batch > 0 else 0.0

        if trainer.is_global_zero:
            e = pl_module.current_epoch
            te = trainer.max_epochs
            print(
                f"Epoch {e}/{te}: "
                f"global_mean_val_batch_time={avg_duration * 1000:.2f} ms"
            )

        if self.reset_per_epoch:
            self.local_total_val_batch_duration = 0.0
            self.local_n_val_batch = 0

    def load_state_dict(self, sd):
        self.local_total_train_batch_duration = sd["local_total_train_batch_duration"]
        self.local_n_train_batch = sd["local_n_train_batch"]
        self.local_total_val_batch_duration = sd["local_total_val_batch_duration"]
        self.local_n_val_batch = sd["local_n_val_batch"]
        self.reset_per_epoch = sd["reset_per_epoch"]

    def state_dict(self):
        return {
            "local_total_val_batch_duration": self.local_total_val_batch_duration,
            "local_n_val_batch": self.local_n_val_batch,
            "local_total_train_batch_duration": self.local_total_train_batch_duration,
            "local_n_train_batch": self.local_n_train_batch,
            "reset_per_epoch": self.reset_per_epoch,
        }


class SamplePerSecondCallback(callbacks.Callback):
    """Sample per second calculator.

    For each GPU, compute the number of samples.
    Sum the number of samples across all GPUs to get global_total_samples.
    Use the epoch duration (same for all GPUs in synchronized training) as global_total_duration.
    SPS = global_total_samples / global_total_duration.

    e.g. 2 GPUs, 2 batches per GPU, 5 samples per batch; take 200ms in total
    sample_per_second = (2 * 2 * 5) / 200ms = 100 samples/s
    """

    def __init__(self, batch_size=None):
        super().__init__()
        # all local: results on a single GPU
        self.local_train_epoch_start_time = 0
        self.local_n_train_samples = 0
        self.local_val_epoch_start_time = 0
        self.local_n_val_samples = 0

        self.batch_size = batch_size  # explicitly set the batch size

    def on_train_epoch_start(self, *args, **kwargs):
        self.local_n_train_samples = 0
        self.local_train_epoch_start_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, *args, **kwargs):
        if self.batch_size is None:
            self.local_n_train_samples += batch[1].shape[0]  # batch size
        else:
            self.local_n_train_samples += self.batch_size

    def on_train_epoch_end(self, trainer, pl_module):
        duration = time.time() - self.local_train_epoch_start_time
        n = self.local_n_train_samples

        # synchronize across all processes (all GPUs)
        if dist.is_available() and dist.is_initialized():
            # only sum samples, not duration (since GPUs are synchronized)
            samples_tensor = torch.tensor(
                [n],
                dtype=torch.float,
                device=pl_module.device,
            )
            dist.all_reduce(samples_tensor, op=dist.ReduceOp.SUM)
            n = samples_tensor.item()
            # duration is the same across GPUs in synchronized training

        # compute samples per second
        sps = (n / duration) if duration > 0 else 0.0

        if trainer.is_global_zero:
            e = pl_module.current_epoch
            te = trainer.max_epochs
            print(f"Epoch {e}/{te}: train_samples_per_second={sps:.2f} samples/s")

    def on_validation_epoch_start(self, *args, **kwargs):
        self.local_n_val_samples = 0
        self.local_val_epoch_start_time = time.time()

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, *args, **kwargs
    ):
        if self.batch_size is None:
            self.local_n_val_samples += batch[1].shape[0]
        else:
            self.local_n_val_samples += self.batch_size

    def on_validation_epoch_end(self, trainer, pl_module):
        duration = time.time() - self.local_val_epoch_start_time
        n = self.local_n_val_samples

        # synchronize across all processes
        if dist.is_available() and dist.is_initialized():
            samples_tensor = torch.tensor(
                [n],
                dtype=torch.float,
                device=pl_module.device,
            )
            dist.all_reduce(samples_tensor, op=dist.ReduceOp.SUM)
            n = samples_tensor.item()
            # duration is the same across GPUs

        # compute samples per second
        sps = (n / duration) if duration > 0 else 0.0

        if trainer.is_global_zero:
            e = pl_module.current_epoch
            te = trainer.max_epochs
            print(f"Epoch {e}/{te}: val_samples_per_second={sps:.2f} samples/s")

    def load_state_dict(self, sd):
        self.local_n_train_samples = sd["local_n_train_samples"]
        self.local_n_val_samples = sd["local_n_val_samples"]
        self.batch_size = sd["batch_size"]

    def state_dict(self):
        return {
            "local_n_train_samples": self.local_n_train_samples,
            "local_n_val_samples": self.local_n_val_samples,
            "batch_size": self.batch_size,
        }


class PeakMemoryTillNowCallback(callbacks.Callback):
    def on_fit_start(self, trainer, pl_module):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            mem_stats = torch.cuda.memory_stats()
            peak_allocated = mem_stats["allocated_bytes.all.peak"] / (1024**2)
            peak_reserved = mem_stats["reserved_bytes.all.peak"] / (1024**2)
            if trainer.is_global_zero:
                print(
                    f"Before training: "
                    f"peak_allocated={peak_allocated} MB, "
                    f"peak_reserved={peak_reserved} MB",
                    flush=True,
                )

    def on_train_epoch_end(self, trainer, pl_module):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_stats = torch.cuda.memory_stats()
            peak_allocated = mem_stats["allocated_bytes.all.peak"] / (1024**2)
            peak_reserved = mem_stats["reserved_bytes.all.peak"] / (1024**2)

            if dist.is_available() and dist.is_initialized():
                peak_tensor = torch.tensor(
                    [peak_allocated, peak_reserved],
                    dtype=torch.float32,
                    device=pl_module.device,
                )
                dist.all_reduce(peak_tensor, op=dist.ReduceOp.MAX)
                peak_allocated, peak_reserved = peak_tensor.tolist()

            if trainer.is_global_zero:
                e = pl_module.current_epoch
                te = trainer.max_epochs
                print(
                    f"Epoch {e}/{te}: "
                    f"peak_allocated={peak_allocated} MB, "
                    f"peak_reserved={peak_reserved} MB",
                    flush=True,
                )
