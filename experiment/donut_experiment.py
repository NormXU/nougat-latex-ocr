# -*- coding:utf-8 -*-
# create: 2023/6/2
import time
import munch
import os
import torch
import mydatasets
from Levenshtein import distance
from torch.utils.data import DataLoader
from tqdm import tqdm
from mydatasets import get_dataset
from .base_experiment import BaseExperiment
from metrics import AverageMeter, TokenAccMetric
from base.driver import logger
from base.torch_utils.dl_util import get_optimizer
from base.common_util import get_absolute_file_path, merge_config
from transformers import AutoTokenizer, VisionEncoderDecoderModel, VisionEncoderDecoderConfig, NougatProcessor
from nougat_latex.image_processing_nougat import NougatImageProcessor


class DonutExperiment(BaseExperiment):

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
        edit_distance = AverageMeter()
        eval_metric = TokenAccMetric(pad_token_id=self.processor.tokenizer.pad_token_id,
                                     eos_token_id=self.processor.tokenizer.eos_token_id)
        for i, batch_data in tqdm(enumerate(self.eval_data_loader), total=len(self.eval_data_loader)):
            pixel_values = torch.stack([instance for instance in batch_data['pixel_values']])
            answers = batch_data['processed_parse']
            batch_size = pixel_values.shape[0]
            decoder_input_ids = torch.full((batch_size, 1), self.model.config.decoder_start_token_id)
            start = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    pixel_values.to(self.args.device.device_id),
                    decoder_input_ids=decoder_input_ids.to(self.args.device.device_id),
                    max_length=self.model.decoder.config.max_length,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=1,  # for beam search
                    bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                    return_dict_in_generate=True,
                )
            batch_time = time.time() - start
            logger.info("batch inference time:{} s".format(batch_time))
            eval_metric.add(outputs.sequences[:, 1:].cpu(), batch_data['labels'].cpu())
            pred_sequences = self.processor.tokenizer.batch_decode(outputs.sequences)
            for ans_seq, pred_seq in zip(answers, pred_sequences):
                pred_seq = pred_seq.replace(self.processor.tokenizer.bos_token, "") \
                    .replace(self.processor.tokenizer.eos_token, ""). \
                    replace(self.processor.tokenizer.pad_token, "")
                if len(ans_seq) > 0:
                    edit_distance.update(distance(pred_seq, ans_seq) / len(ans_seq))
        logger.info("evaluating...")
        logger.info("token_acc: {}; edit_dis: {}".format(eval_metric.mean(), edit_distance.avg))
        return {"token_acc": eval_metric.mean(), "edit_dis": edit_distance.avg}

    def train(self, **kwargs):
        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        norm_meter = AverageMeter()
        global_step = self.args.trainer.start_epoch * len(self.train_data_loader)
        global_eval_step = 0
        ni = 0
        for epoch in range(self.args.trainer.start_epoch, self.args.trainer.epochs):
            self.optimizer.zero_grad()
            for i, batch in enumerate(self.train_data_loader):
                if global_step < self.args.trainer.start_global_step:
                    global_step += 1
                    continue
                start = time.time()
                self.model.train()
                ni = i + len(self.train_data_loader) * epoch  # number integrated batches (since train start)
                with self.gradient_accumulate_scope(self.model):
                    result = self._step_forward(batch)
                    self._step_backward(result.loss)
                    if self.accelerator is not None or ((i + 1) % self.args.trainer.grad_accumulate
                                                        == 0) or ((i + 1) == len(self.train_data_loader)):
                        grad_norm = self._step_optimizer()
                        norm_meter.update(grad_norm)
                        if not self.args.trainer.scheduler_by_epoch:
                            self._step_scheduler(global_step)
                loss_meter.update(result['loss'].item(), self.args.datasets.train.batch_size)
                batch_time.update(time.time() - start)
                global_step += 1
                global_eval_step = self._print_step_log(epoch, global_step, global_eval_step, loss_meter, norm_meter,
                                                        batch_time, ni)
            if self.args.trainer.scheduler_by_epoch:
                self._step_scheduler(global_step)
            global_eval_step = self._print_epoch_log(epoch, global_step, global_eval_step, loss_meter, ni)
        self._train_post_process()
        if self.args.device.is_master:
            self.writer.close()

    def _step_forward(self, batch, is_train=True, eval_model=None, **kwargs):
        input_args_list = ['pixel_values', 'labels', 'decoder_input_ids']
        batch = {k: v.to(self.args.device.device_id) for k, v in batch.items() if k in input_args_list}
        # Runs the forward pass with auto-casting.
        with self.precision_scope:
            output = self.model(**batch)
        return output

    """
        Initialization Functions
    """

    def init_model(self, config):
        model_args = config["model"]
        processor_args = model_args['processor_args']
        pretrained_model_name_or_path = model_args["pretrained_model_name_or_path"]
        # initialize processor & image processor & tokenizer
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        image_processor = NougatImageProcessor(
            size={"height": model_args['image_size'][0], "width": model_args['image_size'][1]},
            **processor_args['img_processor_args'])
        self.processor = NougatProcessor(image_processor=image_processor,
                                         tokenizer=tokenizer)
        # model initialization
        config = VisionEncoderDecoderConfig.from_pretrained(pretrained_model_name_or_path)
        config.encoder.image_size = model_args['image_size']
        # during pre-training, a larger image size was used; for fine-tuning,
        # we update max_length of the decoder (for generation)
        config.decoder.max_length = model_args['max_length']
        model = VisionEncoderDecoderModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            config=config
        )
        logger.info("init weight from pretrained model:{}".format(pretrained_model_name_or_path))
        self.model = model
        self.model.to(self.args.device.device_id)
        if "model_path" in model_args and model_args['model_path'] is not None:
            model_path = get_absolute_file_path(model_args['model_path'])
            self.load_model(model_path, strict=model_args.get('load_strict', True))
        total = sum([param.nelement() for param in self.model.parameters()])
        logger.info("Number of parameter: %.2fM" % (total / 1e6))

    def _init_optimizer(self, trainer_args, **kwargs):
        optimizer_args = trainer_args.get("optimizer")
        if optimizer_args.get("scale_lr"):
            num_process = 1 if self.accelerator is None else self.accelerator.num_processes
            optimizer_args['lr'] = float(optimizer_args['lr']) * self.grad_accumulate * \
                                   self.train_data_loader.batch_size * num_process
        self.optimizer = get_optimizer(self.model, **optimizer_args)

    def init_dataset(self, config):
        if 'datasets' in config:
            dataset_args = config.get("datasets")
            train_data_loader_args = dataset_args.get("train")
            if config.get('phase', 'train') == 'train':
                train_data_loader_args['dataset'].update({
                    "processor": self.processor,
                    "max_length": config['model']['max_length'],
                    "phase": 'train',
                })
                if "cache_dir" not in train_data_loader_args['dataset']:
                    train_data_loader_args['dataset'].update({
                        "cache_dir": config['trainer']['save_dir']})
                self.train_dataset = get_dataset(train_data_loader_args['dataset'])
                self.train_data_loader = self._get_data_loader_from_dataset(self.train_dataset,
                                                                            train_data_loader_args,
                                                                            phase='train')
                logger.info("success init train data loader len:{} ".format(len(self.train_data_loader)))

            eval_data_loader_args = dataset_args.get("eval")
            eval_data_loader_args['dataset'].update({
                "processor": self.processor,
                "max_length": config['model']['max_length'],
                "phase": "evaluate"})
            merged_eval_data_loader_args = train_data_loader_args.copy()
            merge_config(eval_data_loader_args, merged_eval_data_loader_args)
            self.eval_dataset = get_dataset(merged_eval_data_loader_args['dataset'])
            self.eval_data_loader = self._get_data_loader_from_dataset(self.eval_dataset,
                                                                       merged_eval_data_loader_args,
                                                                       phase='eval')
            logger.info("success init eval data loader len:{}".format(len(self.eval_data_loader)))

            # set task start token & pad token for bart decoder;
            # Do NOT change it since you can only set the start_token after dataset initialization where special tokens
            # are added into vocab
            self.model.config.decoder_start_token_id = self.processor.tokenizer.bos_token_id
            self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id

    """
        Tool Functions
    """

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
            collate_fn = getattr(mydatasets, collate_fn_type)(batch_size=batch_size, processor=self.processor,
                                                              **collate_fn_args)
        data_loader = DataLoader(dataset,
                                 shuffle=shuffle,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory,
                                 collate_fn=collate_fn,
                                 batch_size=batch_size)
        logger.info("use data loader with batch_size:{},num_workers:{}".format(batch_size, num_workers))

        return data_loader

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
        if global_step > 0 and self.args.trainer.save_step_freq > 0 and self.args.device.is_master and global_step % self.args.trainer.save_step_freq == 0:
            message = "experiment:{}; eval, (epoch: {}, steps: {});".format(self.experiment_name, epoch, global_step)
            logger.info(message)
            result = self.evaluate(global_eval_step=global_eval_step)
            checkpoint_name = "{}_epoch{}_step{}_lr{:e}_avg_loss{:.5f}_token_acc{:.5f}_edit_dis{:.5f}.pth".format(
                self.experiment_name, epoch, global_step, current_lr,
                loss_meter.avg, result["token_acc"], result["edit_dis"])
            checkpoint_path = os.path.join(self.args.trainer.save_dir, checkpoint_name)
            self.save_model(checkpoint_path, epoch=epoch, global_step=global_step, loss=loss_meter.val)
        return global_eval_step

    def _print_epoch_log(self, epoch, global_step, global_eval_step, loss_meter, ni, **kwargs):
        current_lr = self._get_current_lr(ni, global_step)
        if self.args.trainer.save_epoch_freq > 0 and self.args.device.is_master and epoch % self.args.trainer.save_epoch_freq == 0:
            message = "experiment:{}; eval, (epoch: {}, steps: {});".format(self.experiment_name, epoch, global_step)
            logger.info(message)
            result = self.evaluate(global_eval_step=global_eval_step)
            checkpoint_name = "{}_epoch{}_step{}_lr{:e}_avg_loss{:.5f}_token_acc{:.5f}_edit_dis{:.5f}.pth".format(
                self.experiment_name, epoch, global_step, current_lr,
                loss_meter.avg, result["token_acc"], result["edit_dis"])
            checkpoint_path = os.path.join(self.args.trainer.save_dir, checkpoint_name)
            self.save_model(checkpoint_path, epoch=epoch, global_step=global_step, loss=loss_meter.val)
        return global_eval_step
