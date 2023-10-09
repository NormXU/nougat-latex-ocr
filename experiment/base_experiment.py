# -*- coding:utf-8 -*-
# create: 2021/7/2
import copy
import itertools
import json
import logging
import os
from contextlib import nullcontext

import munch
import torch
import torch.profiler
from accelerate import Accelerator
from torch import autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import mydatasets
from mydatasets import get_dataset
from base.common_util import get_absolute_file_path, merge_config, save_params
from base.driver import log_formatter, logger
from base.torch_utils.dl_util import get_optimizer, get_scheduler, get_scheduler2, seed_all, get_grad_norm
from base.torch_utils.torch_util import ModelEMA



class BaseExperiment(object):

    def __init__(self, config):
        config = self._init_config(config)
        self.experiment_name = config["name"]
        self.args = munch.munchify(config)
        self.init_device(config)
        self.init_random_seed(config)
        self.init_model(config)
        self.init_dataset(config)
        self.init_trainer_args(config)
        self.init_evaluator_args(config)
        self.prepare_accelerator()

    """
        Main Block
    """

    def evaluate(self, **kwargs):
        pass

    def train(self, **kwargs):
        pass

    def _step_forward(self, batch, is_train=True, eval_model=None, **kwargs):
        model = eval_model if is_train is False and eval_model is not None else self.model
        with self.precision_scope:
            output = model(input)
        return output

    def _step_backward(self, loss, **kwargs):
        # ADD grad norm和clip都不在这一步做
        if self.use_torch_amp:
            self.mixed_scaler.scale(loss).backward()
        else:
            if self.accelerator is not None:
                self.accelerator.backward(loss)
            else:
                loss = loss / self.args.trainer.grad_accumulate
                loss.backward()

    def _get_current_lr(self, ni, global_step=0, **kwargs):
        if self.args.trainer.scheduler_type == "scheduler2":
            current_lr = self.scheduler.get_update_values(global_step)[-1]
        else:
            current_lr = self.scheduler.get_last_lr()[-1]
        return current_lr

    def _step_optimizer(self, **kwargs):
        params_to_clip = (itertools.chain(self.model.parameters()))
        for param_group in self.optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = param_group["lr"] * param_group["lr_scale"]
        grad_norm = None
        if self.args.trainer.grad_clip is not None:
            if self.use_torch_amp:
                # Unscales the gradients of optimizer's assigned params in-place
                # called only after all gradients for that optimizer’s assigned parameters have been accumulated
                self.mixed_scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(params_to_clip, self.args.trainer.grad_clip)
                self.mixed_scaler.step(self.optimizer)
                # Updates the scale for next iteration.
                self.mixed_scaler.update()
            if self.accelerator:
                if self.accelerator.sync_gradients:
                    grad_norm = self.accelerator.clip_grad_norm_(params_to_clip, self.args.trainer.grad_clip)
                self.optimizer.step()
        if grad_norm is None:
            grad_norm = get_grad_norm(params_to_clip)
            self.optimizer.step()

        self.optimizer.zero_grad()
        if hasattr(self, "ema") and self.ema:
            self.ema.update(self.model)
        return grad_norm

    def _step_scheduler(self, global_step, **kwargs):
        if self.args.trainer.scheduler_type == "scheduler2":
            self.scheduler.step_update(global_step)
        else:
            self.scheduler.step()

    """
        Initialization Functions
    """

    def _init_config(self, config):
        if 'trainer' in config and config.get('phase', 'train') == 'train':
            trainer_args = config["trainer"]
            trainer_args['save_dir'] = get_absolute_file_path(trainer_args.get("save_dir"))
            os.makedirs(trainer_args['save_dir'], exist_ok=True)
            # save training yml for easier replication
            save_params(trainer_args['save_dir'], config)
            train_log_path = os.path.join(trainer_args['save_dir'], "{}.log".format(config['name']))
            file_handler = logging.FileHandler(train_log_path)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(log_formatter)
            logger.addHandler(file_handler)
        return config

    def init_device(self, config):
        if config['device'].get('allow_tf32', False):
            # Enable TF32 for faster training on Ampere GPUs,
            # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        # ADD RUN_ON_GPU_IDs=-1是cpu，多张默认走accelerator
        self.args.device = munch.munchify(config.get('device', {}))
        self.accelerator = None
        self.weight_dtype = torch.float32
        self.gradient_accumulate_scope = nullcontext
        self.precision_scope = nullcontext()
        self.use_torch_amp = False
        if os.environ.get("RUN_ON_GPU_IDs", 0) == str(-1):
            # load model with CPU
            self.args.device.device_id = torch.device("cpu")
            self.args.device.device_ids = [-1]
            self.args.device.is_master = True
            self.args.device.is_distributed = False
        else:
            # accelerator configuration
            if len(os.environ.get("RUN_ON_GPU_IDs", 0)) > 1:
                # If you define multiple visible GPU, I suppose you to use accelerator to do ddp training
                self.accelerator = Accelerator(
                    gradient_accumulation_steps=int(self.args.trainer.grad_accumulate),
                    mixed_precision=self.args.model.mixed_precision)
                self.args.device.device_id = self.accelerator.device
                self.args.device.device_ids = []
                if self.accelerator.mixed_precision == "fp16":
                    self.weight_dtype = torch.float16
                elif self.accelerator.mixed_precision == "bf16":
                    self.weight_dtype = torch.bfloat16
                self.gradient_accumulate_scope = self.accelerator.accumulate
                self.args.device.is_master = self.accelerator.is_main_process
                self.args.device.is_distributed = self.accelerator.num_processes > 1
            else:
                # USE one GPU specified by user w/o using accelerate
                device_id = os.environ.get("RUN_ON_GPU_IDs", 0)
                self.args.device.device_id = torch.device("cuda:{}".format(device_id))
                self.args.device.device_ids = [int(device_id)]
                torch.cuda.set_device(int(device_id))
                self.args.device.is_master = True
                self.args.device.is_distributed = False
                if self.args.model.mixed_precision in ["fp16", "bf16"]:
                    self.use_torch_amp = True
                    self.weight_dtype = torch.float16 if self.args.model.mixed_precision == "fp16" else torch.bfloat16
                    self.precision_scope = autocast(device_type="cuda", dtype=self.weight_dtype)
        logger.info("device:{}, is_master:{}, device_ids:{}, is_distributed:{}".format(
            self.args.device.device_id, self.args.device.is_master, self.args.device.device_ids,
            self.args.device.is_distributed))

    def init_model(self, config):
        pass

    def init_dataset(self, config):
        if 'datasets' in config and config.get('phase', 'train') != 'predict':
            dataset_args = config.get("datasets")
            train_data_loader_args = dataset_args.get("train")
            if config.get('phase', 'train') == 'train':
                self.train_dataset = get_dataset(train_data_loader_args['dataset'])
                self.train_data_loader = self._get_data_loader_from_dataset(self.train_dataset,
                                                                            train_data_loader_args,
                                                                            phase='train')
                logger.info("success init train data loader len:{} ".format(len(self.train_data_loader)))
            eval_data_loader_args = dataset_args.get("eval")
            merged_eval_data_loader_args = train_data_loader_args.copy()
            merge_config(eval_data_loader_args, merged_eval_data_loader_args)
            self.eval_dataset = get_dataset(merged_eval_data_loader_args['dataset'])
            self.eval_data_loader = self._get_data_loader_from_dataset(self.eval_dataset,
                                                                       merged_eval_data_loader_args,
                                                                       phase='eval')
            logger.info("success init eval data loader len:{}".format(len(self.eval_data_loader)))

    def init_random_seed(self, config):
        if 'random_seed' in config['trainer']:
            seed_all(config['trainer']['random_seed'])
        else:
            logger.warning("random seed is missing")

    def init_evaluator_args(self, config):
        if 'evaluator' in config and config.get('phase', 'train') != 'predict':
            evaluator_args = config["evaluator"]
            self.args.evaluator.save_dir = get_absolute_file_path(evaluator_args.get("save_dir"))
            os.makedirs(self.args.evaluator.save_dir, exist_ok=True)

    def init_trainer_args(self, config):
        if 'trainer' in config and config.get('phase', 'train') == 'train':
            trainer_args = config["trainer"]
            self._init_optimizer(trainer_args)
            self._init_scheduler(trainer_args)
            logger.info("current trainer  epochs:{}, train_dataset_len:{}, data_loader_len:{}".format(
                self.args.trainer.epochs, len(self.train_dataset), len(self.train_data_loader)))
            self.mixed_scaler = torch.cuda.amp.GradScaler(enabled=True) if self.use_torch_amp else None
            self.args.trainer.best_eval_result = -1
            self.args.trainer.best_model_path = ''
            self.ema = ModelEMA(self.model) if trainer_args['use_ema'] else None
            self.args.trainer.start_epoch = 0
            self.args.trainer.start_global_step = 0
            if self.args.trainer.resume_flag and 'model_path' in self.args.model and self.args.model.model_path is not None:
                # ADD resume
                resume_path = self.args.model.model_path.replace('.pth', '_resume.pth')
                if os.path.exists(resume_path):
                    resume_checkpoint = torch.load(resume_path)
                    self.optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
                    self.scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])
                    self.args.trainer.start_epoch = resume_checkpoint['epoch']
                    self.args.trainer.start_global_step = resume_checkpoint['global_step']
                else:
                    logger.warning("resume path {} doesn't exist: failed to resume!!".format(resume_path))

        if 'trainer' in config and config.get('phase', 'train') != 'predict':
            trainer_args = config["trainer"]
            # init tensorboard and log
            if "tensorboard_dir" in trainer_args and self.args.device.is_master:
                tensorboard_log_dir = get_absolute_file_path(trainer_args.get("tensorboard_dir"))
                os.makedirs(tensorboard_log_dir, exist_ok=True)
                self.writer = SummaryWriter(log_dir=tensorboard_log_dir, comment=self.experiment_name)
            else:
                self.writer = None

    def _init_optimizer(self, trainer_args, **kwargs):
        optimizer_args = trainer_args.get("optimizer")
        # ADD scale lr
        if optimizer_args["scale_lr"]:
            num_process = 1 if self.accelerator is None else self.accelerator.num_processes
            optimizer_args['lr'] = optimizer_args['lr'] * self.args.trainer.grad_accumulate * \
                                   self.train_data_loader.batch_size * num_process
        self.optimizer = get_optimizer(self.model, **optimizer_args)

    def _init_scheduler(self, trainer_args, **kwargs):
        scheduler_args = trainer_args.get("scheduler")
        self.args.trainer.scheduler_by_epoch = scheduler_args.get("scheduler_by_epoch", False)
        total_epoch_train_steps = len(self.train_data_loader)
        if scheduler_args["warmup_epochs"] > 0:
            warmup_steps = scheduler_args.get("warmup_epochs") * total_epoch_train_steps
        elif scheduler_args['warmup_steps'] > 0:
            warmup_steps = scheduler_args.get("warmup_steps")
        else:
            warmup_steps = 0
        self.args.trainer.scheduler.warmup_steps = warmup_steps
        num_training_steps = total_epoch_train_steps * self.args.trainer.epochs
        if self.accelerator is None:
            # accelerator will automatically take care of the grad accumulate in calculating total num_training steps,
            # or you need to calculate by yourself
            num_training_steps = num_training_steps // self.args.trainer.grad_accumulate
        if "scheduler_method" in scheduler_args and scheduler_args["scheduler_method"] == "get_scheduler2":
            self.scheduler = get_scheduler2(self.optimizer,
                                            num_training_steps=num_training_steps,
                                            num_warmup_steps=warmup_steps,
                                            **scheduler_args)
            self.args.trainer.scheduler_type = "scheduler2"
        else:
            self.scheduler = get_scheduler(self.optimizer,
                                           num_training_steps=num_training_steps,
                                           num_warmup_steps=warmup_steps,
                                           epochs=self.args.trainer.epochs,
                                           **scheduler_args)
            self.args.trainer.scheduler_type = "scheduler"

        logger.info(
            "success init optimizer and scheduler, optimizer:{}, scheduler:{}, scheduler_args:{}, warmup_steps:{},"
            "num_training_steps:{}, gradient_accumulator:{}".format(self.optimizer, self.scheduler, scheduler_args,
                                                                    warmup_steps, num_training_steps,
                                                                    self.args.trainer.grad_accumulate))

    """
        Tool Functions
    """

    def load_model(self, checkpoint_path, strict=True, **kwargs):
        if os.path.exists(checkpoint_path) and os.path.isfile(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            if 'model_state_dict' in state_dict:
                model_state_dict = state_dict['model_state_dict']
            else:
                model_state_dict = state_dict
            self.model.load_state_dict(model_state_dict, strict=strict)
            logger.info("success load model:{}".format(checkpoint_path))

    def save_model(self, checkpoint_path, **save_kwargs):
        if self.accelerator is not None:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            if self.args.trainer.resume_flag:
                save_kwargs.update({
                    'model_state_dict': unwrapped_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                })
                self.accelerator.save(save_kwargs, checkpoint_path.replace('.pth', '.ckpt'))
            else:
                self.accelerator.save(unwrapped_model.state_dict(), checkpoint_path)
        else:
            if self.args.model.quantization_type == 'quantization_aware_training':
                self.model.eval()
                model_int8 = torch.quantization.convert(self.model)
                torch.save(model_int8.state_dict(), checkpoint_path)
            else:
                if self.args.trainer.resume_flag:
                    save_kwargs.update({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                    })
                    torch.save(save_kwargs, checkpoint_path.replace('.pth', '.ckpt'))
                else:
                    torch.save(self.model.state_dict(), checkpoint_path)
        logger.info("model successfully saved to {}".format(checkpoint_path))

    def _get_data_loader_from_dataset(self, dataset, data_loader_args, phase="train"):
        num_workers = data_loader_args.get("num_workers", 0)
        batch_size = data_loader_args.get("batch_size", 1)
        if phase == "train":
            shuffle = data_loader_args.get("shuffle", True)
        else:
            shuffle = data_loader_args.get("shuffle", False)
        pin_memory = data_loader_args.get("shuffle", False)

        collate_fn_args = data_loader_args.get("collate_fn")
        if collate_fn_args.get("type") is None:
            collate_fn = None
        else:
            collate_fn_type = collate_fn_args.get("type")
            collate_fn = getattr(mydatasets, collate_fn_type)(batch_size=batch_size, **collate_fn_args)
        data_loader = DataLoader(dataset,
                                 shuffle=shuffle,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory,
                                 collate_fn=collate_fn,
                                 batch_size=batch_size)
        logger.info("use data loader with batch_size:{},num_workers:{}".format(batch_size, num_workers))

        return data_loader

    # initialize accelerator
    def prepare_accelerator(self):
        if self.accelerator is not None:
            self.model, self.optimizer, self.train_data_loader, self.scheduler = self.accelerator.prepare(
                self.model, self.optimizer, self.train_data_loader, self.scheduler)

    def _train_post_process(self):
        args = copy.deepcopy(self.args)
        args.model.model_path = args.trainer.best_model_path
        if 'base' in args:
            args.pop('base')
        args.device.pop('device_id')
        args.pop('trainer')
        args.phase = 'predict'
        save_params(self.args.trainer.save_dir, json.loads(json.dumps(args)), 'model_args.yaml')

    def _print_step_log(self, epoch, global_step, global_eval_step, loss_meter, norm_meter, batch_time, ni, **kwargs):
        current_lr = self._get_current_lr(ni, global_step)
        if self.args.device.is_master and self.args.trainer.print_freq > 0 and global_step % self.args.trainer.print_freq == 0:
            message = "experiment:{}; train, (epoch: {}, steps: {}, lr:{:e}, step_mean_loss:{}," \
                      " average_loss:{}), time, (train_step_time: {:.5f}s, train_average_time: {:.5f}s);" \
                      "(grad_norm_mean: {:.5f}, grad_norm_step: {:.5f})". \
                format(self.experiment_name, epoch, global_step, current_lr,
                       loss_meter.val, loss_meter.avg, batch_time.val, batch_time.avg, norm_meter.avg,
                       norm_meter.val)
            logger.info(message)
            if self.writer is not None:
                self.writer.add_scalar("{}_train/lr".format(self.experiment_name), current_lr, global_step)
                self.writer.add_scalar("{}_train/step_loss".format(self.experiment_name), loss_meter.val, global_step)
                self.writer.add_scalar("{}_train/average_loss".format(self.experiment_name), loss_meter.avg,
                                       global_step)
        if global_step > 0 and self.args.trainer.save_step_freq > 0 and global_step % self.args.trainer.save_step_freq == 0:
            message = "experiment:{}; eval, (epoch: {}, steps: {});".format(self.experiment_name, epoch, global_step)
            logger.info(message)
            result = self.evaluate(global_eval_step=global_eval_step)
            global_eval_step, acc = result['global_eval_step'], result['acc']
            # ADD is_master判断移到这里
            if (not self.args.trainer.save_best or (self.args.trainer.save_best
                                                    and acc > self.args.trainer.best_eval_result)) and self.args.device.is_master:
                checkpoint_name = "{}_epoch{}_step{}_lr{:e}_average_loss{:.5f}_acc{:.5f}.pth".format(
                    self.experiment_name, epoch, global_step, current_lr, loss_meter.avg, acc)
                checkpoint_path = os.path.join(self.args.trainer.save_dir, checkpoint_name)
                # ADD记得传epoch和global_step，resume才能存
                self.save_model(checkpoint_path, epoch=epoch, global_step=global_step, loss=loss_meter.val)
                if acc > self.args.trainer.best_eval_result:
                    self.args.trainer.best_eval_result = acc
                    self.args.trainer.best_model_path = checkpoint_path
        return global_eval_step

    def _print_epoch_log(self, epoch, global_step, global_eval_step, loss_meter, ni, **kwargs):
        current_lr = self._get_current_lr(ni, global_step)
        if self.args.trainer.save_epoch_freq > 0 and epoch % self.args.trainer.save_epoch_freq == 0:
            message = "experiment:{}; eval, (epoch: {}, steps: {});".format(self.experiment_name, epoch, global_step)
            logger.info(message)
            result = self.evaluate(global_eval_step=global_eval_step)
            global_eval_step, acc = result['global_eval_step'], result['acc']
            # ADD is_master判断移到这里
            if (not self.args.trainer.save_best or (self.args.trainer.save_best
                                                    and acc > self.args.trainer.best_eval_result)) and self.args.device.is_master:
                checkpoint_name = "{}_epoch{}_step{}_lr{:e}_average_loss{:.5f}_acc{:.5f}.pth".format(
                    self.experiment_name, epoch, global_step, current_lr, loss_meter.avg, acc)
                checkpoint_path = os.path.join(self.args.trainer.save_dir, checkpoint_name)
                # ADD记得传epoch和global_step，resume才能存
                self.save_model(checkpoint_path, epoch=epoch, global_step=global_step, loss=loss_meter.val)
                if acc > self.args.trainer.best_eval_result:
                    self.args.trainer.best_eval_result = acc
                    self.args.trainer.best_model_path = checkpoint_path
        return global_eval_step

    def _print_eval_log(self, global_step, loss_meter, eval_metric, **kwargs):
        evaluate_report = eval_metric.get_report()
        acc = evaluate_report["acc"]
        message = "experiment:{}; eval,global_step:{}, (step_mean_loss:{},average_loss:{:.5f},evaluate_report:{})".format(
            self.experiment_name, global_step, loss_meter.val, loss_meter.avg, evaluate_report)
        logger.info(message)
        if self.writer is not None:
            self.writer.add_scalar("{}_eval/step_loss".format(self.experiment_name), loss_meter.val, global_step)
            self.writer.add_scalar("{}_eval/average_loss".format(self.experiment_name), loss_meter.avg, global_step)
            self.writer.add_scalar("{}_eval/acc".format(self.experiment_name), acc, global_step)
        return acc
