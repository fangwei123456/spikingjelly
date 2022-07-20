import datetime
import os
import time
import warnings
from .tv_ref_classify import presets, transforms, utils
import torch
import torch.utils.data
import torchvision
from .tv_ref_classify.sampler import RASampler
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import sys
import argparse
from .. import functional


try:
    from torchvision import prototype
except ImportError:
    prototype = None

def set_deterministic(_seed_: int = 2020, disable_uda=False):
    random.seed(_seed_)
    np.random.seed(_seed_)
    torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
    torch.cuda.manual_seed_all(_seed_)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if disable_uda:
        pass
    else:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        # set a debug environment variable CUBLAS_WORKSPACE_CONFIG to ":16:8" (may limit overall performance) or ":4096:8" (will increase library footprint in GPU memory by approximately 24MiB).
        torch.use_deterministic_algorithms(True)



def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class Trainer:
    def cal_acc1_acc5(self, output, target):
        # define how to calculate acc1 and acc5
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        return acc1, acc5

    def preprocess_train_sample(self, args, x: torch.Tensor):
        # define how to process train sample before send it to model
        return x

    def preprocess_test_sample(self, args, x: torch.Tensor):
        # define how to process test sample before send it to model
        return x

    def process_model_output(self, args, y: torch.Tensor):
        # define how to process y = model(x)
        return y

    def train_one_epoch(self, model, criterion, optimizer, data_loader, device, epoch, args, model_ema=None, scaler=None):
        model.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
        metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

        header = f"Epoch: [{epoch}]"
        for i, (image, target) in enumerate(metric_logger.log_every(data_loader, -1, header)):
            start_time = time.time()
            image, target = image.to(device), target.to(device)
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                image = self.preprocess_train_sample(args, image)
                output = self.process_model_output(args, model(image))
                loss = criterion(output, target)

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                if args.clip_grad_norm is not None:
                    # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
            functional.reset_net(model)

            if model_ema and i % args.model_ema_steps == 0:
                model_ema.update_parameters(model)
                if epoch < args.lr_warmup_epochs:
                    # Reset ema buffer to keep copying weights during warmup period
                    model_ema.n_averaged.fill_(0)

            acc1, acc5 = self.cal_acc1_acc5(output, target)
            batch_size = target.shape[0]
            metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        train_loss, train_acc1, train_acc5 = metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg
        print(f'Train: train_acc1={train_acc1:.3f}, train_acc5={train_acc5:.3f}, train_loss={train_loss:.6f}, samples/s={metric_logger.meters["img/s"]}')
        return train_loss, train_acc1, train_acc5

    def evaluate(self, args, model, criterion, data_loader, device, log_suffix=""):
        model.eval()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = f"Test: {log_suffix}"

        num_processed_samples = 0
        start_time = time.time()
        with torch.inference_mode():
            for image, target in metric_logger.log_every(data_loader, -1, header):
                image = image.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                image = self.preprocess_test_sample(args, image)
                output = self.process_model_output(args, model(image))
                loss = criterion(output, target)

                acc1, acc5 = self.cal_acc1_acc5(output, target)
                # FIXME need to take into account that the datasets
                # could have been padded in distributed setup
                batch_size = target.shape[0]
                metric_logger.update(loss=loss.item())
                metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
                metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
                num_processed_samples += batch_size
                functional.reset_net(model)
        # gather the stats from all processes

        num_processed_samples = utils.reduce_across_processes(num_processed_samples)
        if (
            hasattr(data_loader.dataset, "__len__")
            and len(data_loader.dataset) != num_processed_samples
            and torch.distributed.get_rank() == 0
        ):
            # See FIXME above
            warnings.warn(
                f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
                "samples were used for the validation, which might bias the results. "
                "Try adjusting the batch size and / or the world size. "
                "Setting the world size to 1 is always a safe bet."
            )

        metric_logger.synchronize_between_processes()

        test_loss, test_acc1, test_acc5 = metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg
        print(f'Test: test_acc1={test_acc1:.3f}, test_acc5={test_acc5:.3f}, test_loss={test_loss:.6f}, samples/s={num_processed_samples / (time.time() - start_time):.3f}')
        return test_loss, test_acc1, test_acc5

    def _get_cache_path(self, filepath):
        import hashlib

        h = hashlib.sha1(filepath.encode()).hexdigest()
        cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
        cache_path = os.path.expanduser(cache_path)
        return cache_path


    def load_data(self, args):
        return self.load_ImageNet(args)

    def load_CIFAR10(self, args):
        # Data loading code
        print("Loading data")
        val_resize_size, val_crop_size, train_crop_size = args.val_resize_size, args.val_crop_size, args.train_crop_size
        interpolation = InterpolationMode(args.interpolation)

        print("Loading training data")
        st = time.time()
        auto_augment_policy = getattr(args, "auto_augment", None)
        random_erase_prob = getattr(args, "random_erase", 0.0)
        dataset = torchvision.datasets.CIFAR10(
            root=args.data_path,
            train=True,
            transform=presets.ClassificationPresetTrain(
                crop_size=train_crop_size,
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010),
                interpolation=interpolation,
                auto_augment_policy=auto_augment_policy,
                random_erase_prob=random_erase_prob,
            ),
        )

        print("Took", time.time() - st)

        print("Loading validation data")

        dataset_test = torchvision.datasets.CIFAR10(
            root=args.data_path,
            train=False,
            transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]),
        )

        print("Creating data loaders")
        loader_g = torch.Generator()
        loader_g.manual_seed(args.seed)

        if args.distributed:
            if hasattr(args, "ra_sampler") and args.ra_sampler:
                train_sampler = RASampler(dataset, shuffle=True, repetitions=args.ra_reps, seed=args.seed)
            else:
                train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, seed=args.seed)
            test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
        else:
            train_sampler = torch.utils.data.RandomSampler(dataset, generator=loader_g)
            test_sampler = torch.utils.data.SequentialSampler(dataset_test)


        return dataset, dataset_test, train_sampler, test_sampler

    def load_ImageNet(self, args):
        # Data loading code
        traindir = os.path.join(args.data_path, "train")
        valdir = os.path.join(args.data_path, "val")
        print("Loading data")
        val_resize_size, val_crop_size, train_crop_size = args.val_resize_size, args.val_crop_size, args.train_crop_size
        interpolation = InterpolationMode(args.interpolation)

        print("Loading training data")
        st = time.time()
        cache_path = self._get_cache_path(traindir)
        if args.cache_dataset and os.path.exists(cache_path):
            # Attention, as the transforms are also cached!
            print(f"Loading dataset_train from {cache_path}")
            dataset, _ = torch.load(cache_path)
        else:
            auto_augment_policy = getattr(args, "auto_augment", None)
            random_erase_prob = getattr(args, "random_erase", 0.0)
            dataset = torchvision.datasets.ImageFolder(
                traindir,
                presets.ClassificationPresetTrain(
                    crop_size=train_crop_size,
                    interpolation=interpolation,
                    auto_augment_policy=auto_augment_policy,
                    random_erase_prob=random_erase_prob,
                ),
            )
            if args.cache_dataset:
                print(f"Saving dataset_train to {cache_path}")
                utils.mkdir(os.path.dirname(cache_path))
                utils.save_on_master((dataset, traindir), cache_path)
        print("Took", time.time() - st)

        print("Loading validation data")
        cache_path = self._get_cache_path(valdir)
        if args.cache_dataset and os.path.exists(cache_path):
            # Attention, as the transforms are also cached!
            print(f"Loading dataset_test from {cache_path}")
            dataset_test, _ = torch.load(cache_path)
        else:
            if not args.prototype:
                preprocessing = presets.ClassificationPresetEval(
                    crop_size=val_crop_size, resize_size=val_resize_size, interpolation=interpolation
                )
            else:
                if args.weights:
                    weights = prototype.models.get_weight(args.weights)
                    preprocessing = weights.transforms()
                else:
                    preprocessing = prototype.transforms.ImageNetEval(
                        crop_size=val_crop_size, resize_size=val_resize_size, interpolation=interpolation
                    )

            dataset_test = torchvision.datasets.ImageFolder(
                valdir,
                preprocessing,
            )
            if args.cache_dataset:
                print(f"Saving dataset_test to {cache_path}")
                utils.mkdir(os.path.dirname(cache_path))
                utils.save_on_master((dataset_test, valdir), cache_path)

        print("Creating data loaders")
        loader_g = torch.Generator()
        loader_g.manual_seed(args.seed)

        if args.distributed:
            if hasattr(args, "ra_sampler") and args.ra_sampler:
                train_sampler = RASampler(dataset, shuffle=True, repetitions=args.ra_reps, seed=args.seed)
            else:
                train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, seed=args.seed)
            test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
        else:
            train_sampler = torch.utils.data.RandomSampler(dataset, generator=loader_g)
            test_sampler = torch.utils.data.SequentialSampler(dataset_test)

        return dataset, dataset_test, train_sampler, test_sampler

    def load_model(self, args, num_classes):
        raise NotImplementedError("Users should define this function to load model")

    def get_tb_logdir_name(self, args):
        tb_dir = f'{args.model}' \
                 f'_b{args.batch_size}' \
                 f'_e{args.epochs}' \
                 f'_{args.opt}' \
                 f'_lr{args.lr}' \
                 f'_wd{args.weight_decay}' \
                 f'_ls{args.label_smoothing}' \
                 f'_ma{args.mixup_alpha}' \
                 f'_ca{args.cutmix_alpha}' \
                 f'_sbn{1 if args.sync_bn else 0}' \
                 f'_ra{args.ra_reps if args.ra_sampler else 0}' \
                 f'_re{args.random_erase}' \
                 f'_aaug{args.auto_augment}' \
                 f'_size{args.train_crop_size}_{args.val_resize_size}_{args.val_crop_size}' \
                 f'_seed{args.seed}'
        return tb_dir


    def set_optimizer(self, args, parameters):
        opt_name = args.opt.lower()
        if opt_name.startswith("sgd"):
            optimizer = torch.optim.SGD(
                parameters,
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                nesterov="nesterov" in opt_name,
            )
        elif opt_name == "rmsprop":
            optimizer = torch.optim.RMSprop(
                parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
            )
        elif opt_name == "adamw":
            optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
        else:
            optimizer = None
        return optimizer

    def set_lr_scheduler(self, args, optimizer):
        args.lr_scheduler = args.lr_scheduler.lower()
        if args.lr_scheduler == "step":
            main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
        elif args.lr_scheduler == "cosa":
            main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - args.lr_warmup_epochs
            )
        elif args.lr_scheduler == "exp":
            main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
        else:
            main_lr_scheduler = None
        if args.lr_warmup_epochs > 0:
            if args.lr_warmup_method == "linear":
                warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
                )
            elif args.lr_warmup_method == "constant":
                warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                    optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
                )
            else:
                warmup_lr_scheduler = None
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
            )
        else:
            lr_scheduler = main_lr_scheduler

        return lr_scheduler

    def main(self, args):
        set_deterministic(args.seed, args.disable_uda)
        if args.prototype and prototype is None:
            raise ImportError("The prototype module couldn't be found. Please install the latest torchvision nightly.")
        if not args.prototype and args.weights:
            raise ValueError("The weights parameter works only in prototype mode. Please pass the --prototype argument.")
        if args.output_dir:
            utils.mkdir(args.output_dir)

        utils.init_distributed_mode(args)
        print(args)

        device = torch.device(args.device)

        dataset, dataset_test, train_sampler, test_sampler = self.load_data(args)

        collate_fn = None
        num_classes = len(dataset.classes)
        mixup_transforms = []
        if args.mixup_alpha > 0.0:
            if torch.__version__ >= torch.torch_version.TorchVersion('1.10.0'):
                pass
            else:
                # TODO implement a CrossEntropyLoss to support for probabilities for each class.
                raise NotImplementedError("CrossEntropyLoss in pytorch < 1.11.0 does not support for probabilities for each class."
                                          "Set mixup_alpha=0. to avoid such a problem or update your pytorch.")
            mixup_transforms.append(transforms.RandomMixup(num_classes, p=1.0, alpha=args.mixup_alpha))
        if args.cutmix_alpha > 0.0:
            mixup_transforms.append(transforms.RandomCutmix(num_classes, p=1.0, alpha=args.cutmix_alpha))
        if mixup_transforms:
            mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)
            collate_fn = lambda batch: mixupcutmix(*default_collate(batch))  # noqa: E731
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.workers,
            pin_memory=not args.disable_pinmemory,
            collate_fn=collate_fn,
            worker_init_fn=seed_worker
        )

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=not args.disable_pinmemory,
            worker_init_fn=seed_worker
        )

        print("Creating model")
        model = self.load_model(args, num_classes)
        model.to(device)
        print(model)

        if args.distributed and args.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

        if args.norm_weight_decay is None:
            parameters = model.parameters()
        else:
            param_groups = torchvision.ops._utils.split_normalization_params(model)
            wd_groups = [args.norm_weight_decay, args.weight_decay]
            parameters = [{"params": p, "weight_decay": w} for p, w in zip(param_groups, wd_groups) if p]

        optimizer = self.set_optimizer(args, parameters)

        if args.disable_amp:
            scaler = None
        else:
            scaler = torch.cuda.amp.GradScaler()

        lr_scheduler = self.set_lr_scheduler(args, optimizer)


        model_without_ddp = model
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module

        model_ema = None
        if args.model_ema:
            # Decay adjustment that aims to keep the decay independent from other hyper-parameters originally proposed at:
            # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
            #
            # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
            # We consider constant = Dataset_size for a given dataset/setup and ommit it. Thus:
            # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
            adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
            alpha = 1.0 - args.model_ema_decay
            alpha = min(1.0, alpha * adjust)
            model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)

        # 确定目录文件名

        tb_dir = self.get_tb_logdir_name(args)
        pt_dir = os.path.join(args.output_dir, 'pt', tb_dir)
        tb_dir = os.path.join(args.output_dir, tb_dir)
        if args.print_logdir:
            print(tb_dir)
            print(pt_dir)
            exit()
        if args.clean:
            if utils.is_main_process():
                if os.path.exists(tb_dir):
                    os.remove(tb_dir)
                if os.path.exists(pt_dir):
                    os.remove(pt_dir)
                print(f'remove {tb_dir} and {pt_dir}.')

        if utils.is_main_process():
            os.makedirs(tb_dir, exist_ok=args.resume is not None)
            os.makedirs(pt_dir, exist_ok=args.resume is not None)

        if args.resume is not None:
            if args.resume == 'latest':
                checkpoint = torch.load(os.path.join(pt_dir, 'checkpoint_latest.pth'), map_location="cpu")
            else:
                checkpoint = torch.load(args.resume, map_location="cpu")
            model_without_ddp.load_state_dict(checkpoint["model"])
            if not args.test_only:
                optimizer.load_state_dict(checkpoint["optimizer"])
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1
            if model_ema:
                model_ema.load_state_dict(checkpoint["model_ema"])
            if scaler:
                scaler.load_state_dict(checkpoint["scaler"])

            if utils.is_main_process():
                max_test_acc1 = checkpoint['max_test_acc1']
                if model_ema:
                    max_ema_test_acc1 = checkpoint['max_ema_test_acc1']

        if utils.is_main_process():
            tb_writer = SummaryWriter(tb_dir, purge_step=args.start_epoch)
            with open(os.path.join(tb_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
                args_txt.write(str(args))
                args_txt.write('\n')
                args_txt.write(' '.join(sys.argv))

            max_test_acc1 = -1.
            if model_ema:
                max_ema_test_acc1 = -1.


        if args.test_only:
            if model_ema:
                self.evaluate(args, model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
            else:
                self.evaluate(args, model, criterion, data_loader_test, device=device)
            return




        for epoch in range(args.start_epoch, args.epochs):
            start_time = time.time()
            if args.distributed:
                train_sampler.set_epoch(epoch)

            self.before_train_one_epoch(args, model, epoch)
            train_loss, train_acc1, train_acc5 = self.train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema, scaler)
            if utils.is_main_process():
                tb_writer.add_scalar('train_loss', train_loss, epoch)
                tb_writer.add_scalar('train_acc1', train_acc1, epoch)
                tb_writer.add_scalar('train_acc5', train_acc5, epoch)

            lr_scheduler.step()
            self.before_test_one_epoch(args, model, epoch)
            test_loss, test_acc1, test_acc5 = self.evaluate(args, model, criterion, data_loader_test, device=device)
            if utils.is_main_process():
                tb_writer.add_scalar('test_loss', test_loss, epoch)
                tb_writer.add_scalar('test_acc1', test_acc1, epoch)
                tb_writer.add_scalar('test_acc5', test_acc5, epoch)
            if model_ema:
                ema_test_loss, ema_test_acc1, ema_test_acc5 = self.evaluate(args, model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
                if utils.is_main_process():
                    tb_writer.add_scalar('ema_test_loss', ema_test_loss, epoch)
                    tb_writer.add_scalar('ema_test_acc1', ema_test_acc1, epoch)
                    tb_writer.add_scalar('ema_test_acc5', ema_test_acc5, epoch)

            if utils.is_main_process():
                save_max_test_acc1 = False
                save_max_ema_test_acc1 = False

                if test_acc1 > max_test_acc1:
                    max_test_acc1 = test_acc1
                    save_max_test_acc1 = True

                checkpoint = {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "args": args,
                    "max_test_acc1": max_test_acc1,
                }
                if model_ema:
                    if ema_test_acc1 > max_ema_test_acc1:
                        max_ema_test_acc1 = ema_test_acc1
                        save_max_ema_test_acc1 = True
                    checkpoint["model_ema"] = model_ema.state_dict()
                    checkpoint["max_ema_test_acc1"] = max_ema_test_acc1
                if scaler:
                    checkpoint["scaler"] = scaler.state_dict()

                utils.save_on_master(checkpoint, os.path.join(pt_dir, f"checkpoint_{epoch}.pth"))
                utils.save_on_master(checkpoint, os.path.join(pt_dir, "checkpoint_latest.pth"))
                if save_max_test_acc1:
                    utils.save_on_master(checkpoint, os.path.join(pt_dir, f"checkpoint_max_test_acc1.pth"))
                if model_ema and save_max_ema_test_acc1:
                    utils.save_on_master(checkpoint, os.path.join(pt_dir, f"checkpoint_max_ema_test_acc1.pth"))

                if utils.is_main_process() and epoch > 0:
                    os.remove(os.path.join(pt_dir, f"checkpoint_{epoch - 1}.pth"))
            print(f'escape time={(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')
            print(args)

    def before_test_one_epoch(self, args, model, epoch):
        pass

    def before_train_one_epoch(self, args, model, epoch):
        pass

    def get_args_parser(self, add_help=True):

        parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

        parser.add_argument("--data-path", default="/datasets01/imagenet_full_size/061417/", type=str, help="dataset path")
        parser.add_argument("--model", default="resnet18", type=str, help="model name")
        parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
        parser.add_argument(
            "-b", "--batch-size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
        )
        parser.add_argument("--epochs", default=90, type=int, metavar="N", help="number of total epochs to run")
        parser.add_argument(
            "-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)"
        )
        parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
        parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
        parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
        parser.add_argument(
            "--wd",
            "--weight-decay",
            default=0.,
            type=float,
            metavar="W",
            help="weight decay (default: 0.)",
            dest="weight_decay",
        )
        parser.add_argument(
            "--norm-weight-decay",
            default=None,
            type=float,
            help="weight decay for Normalization layers (default: None, same value as --wd)",
        )
        parser.add_argument(
            "--label-smoothing", default=0.1, type=float, help="label smoothing (default: 0.1)", dest="label_smoothing"
        )
        parser.add_argument("--mixup-alpha", default=0.2, type=float, help="mixup alpha (default: 0.2)")
        parser.add_argument("--cutmix-alpha", default=1.0, type=float, help="cutmix alpha (default: 1.0)")
        parser.add_argument("--lr-scheduler", default="cosa", type=str, help="the lr scheduler (default: cosa)")
        parser.add_argument("--lr-warmup-epochs", default=5, type=int, help="the number of epochs to warmup (default: 5)")
        parser.add_argument(
            "--lr-warmup-method", default="linear", type=str, help="the warmup method (default: linear)"
        )
        parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
        parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
        parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
        parser.add_argument("--output-dir", default="./logs", type=str, help="path to save outputs")
        parser.add_argument("--resume", default=None, type=str, help="path of checkpoint. If set to 'latest', it will try to load the latest checkpoint")
        parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
        parser.add_argument(
            "--cache-dataset",
            dest="cache_dataset",
            help="Cache the datasets for quicker initialization. It also serializes the transforms",
            action="store_true",
        )
        parser.add_argument(
            "--sync-bn",
            dest="sync_bn",
            help="Use sync batch norm",
            action="store_true",
        )
        parser.add_argument(
            "--test-only",
            dest="test_only",
            help="Only test the model",
            action="store_true",
        )
        parser.add_argument(
            "--pretrained",
            dest="pretrained",
            help="Use pre-trained models from the modelzoo",
            action="store_true",
        )
        parser.add_argument("--auto-augment", default='ta_wide', type=str, help="auto augment policy (default: ta_wide)")
        parser.add_argument("--random-erase", default=0.1, type=float, help="random erasing probability (default: 0.1)")

        # Mixed precision training parameters

        # distributed training parameters
        parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
        parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
        parser.add_argument(
            "--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters"
        )
        parser.add_argument(
            "--model-ema-steps",
            type=int,
            default=32,
            help="the number of iterations that controls how often to update the EMA model (default: 32)",
        )
        parser.add_argument(
            "--model-ema-decay",
            type=float,
            default=0.99998,
            help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
        )
        parser.add_argument(
            "--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
        )
        parser.add_argument(
            "--val-resize-size", default=232, type=int, help="the resize size used for validation (default: 232)"
        )
        parser.add_argument(
            "--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)"
        )
        parser.add_argument(
            "--train-crop-size", default=176, type=int, help="the random crop size used for training (default: 176)"
        )
        parser.add_argument("--clip-grad-norm", default=None, type=float, help="the maximum gradient norm (default None)")
        parser.add_argument("--ra-sampler", action="store_true", help="whether to use Repeated Augmentation in training")
        parser.add_argument(
            "--ra-reps", default=4, type=int, help="number of repetitions for Repeated Augmentation (default: 4)"
        )

        # Prototype models only
        parser.add_argument(
            "--prototype",
            dest="prototype",
            help="Use prototype model builders instead those from main area",
            action="store_true",
        )
        parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
        parser.add_argument("--seed", default=2020, type=int, help="the random seed")

        parser.add_argument("--print-logdir", action="store_true", help="print the dirs for tensorboard logs and pt files and exit")
        parser.add_argument("--clean", action="store_true", help="delete the dirs for tensorboard logs and pt files")
        parser.add_argument("--disable-pinmemory", action="store_true", help="not use pin memory in dataloader, which can help reduce memory consumption")
        parser.add_argument("--disable-amp", action="store_true",
                            help="not use automatic mixed precision training")
        parser.add_argument("--local_rank", type=int, help="args for DDP, which should not be set by user")
        parser.add_argument("--disable-uda", action="store_true",
                            help="not set 'torch.use_deterministic_algorithms(True)', which can avoid the error raised by some functions that do not have a deterministic implementation")


        return parser

if __name__ == "__main__":
    trainer = Trainer()
    args = trainer.get_args_parser().parse_args()
    trainer.main(args)