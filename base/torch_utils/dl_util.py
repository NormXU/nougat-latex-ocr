# -*- coding:utf-8 -*-
# create: 2021/6/17

import json
import math
import torch
import random
import numpy as np
from torch import nn
import torch.optim as optim
from base.driver import logger
from collections import OrderedDict
from torch.optim import lr_scheduler
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from transformers.optimization import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup, \
    get_linear_schedule_with_warmup
from base.torch_utils.scheduler_util import LinearLRScheduler, get_cosine_schedule_by_epochs, \
    get_stairs_schedule_with_warmup


def seed_all(random_seed):
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True


def print_network(net, verbose=False, name=""):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    if verbose:
        logger.info(net)
    if hasattr(net, 'flops'):
        flops = net.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")
    logger.info('network:{} Total number of parameters: {}'.format(name, num_params))


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def set_params_optimizer(model, keyword=None, keywords=None, weight_decay=0.0, lr=None):
    if keywords is None:
        keywords = []
    param_dict = OrderedDict()
    no_decay_param_names = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if keyword in name or check_keywords_in_name(name, keywords):
            param_dict[name] = {"weight_decay": weight_decay}
            if lr is not None:
                lr = float(lr)
                param_dict[name].update({"lr": lr})
        else:
            no_decay_param_names.append(name)
    return param_dict, no_decay_param_names


def get_optimizer(model,
                  optimizer_type="adam",
                  lr=0.001,
                  beta1=0.9,
                  beta2=0.999,
                  no_decay_keys=None,
                  weight_decay=0.0,
                  layer_decay=None,
                  eps=1e-8,
                  momentum=0,
                  params=None,
                  **kwargs):
    assigner = None
    if layer_decay is not None:
        if layer_decay < 1.0:
            num_layers = kwargs.get('num_layers')
            assigner = LayerDecayValueAssigner(list(layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))

    lr = float(lr)
    beta1, beta2 = float(beta1), float(beta2)
    weight_decay = float(weight_decay)
    momentum = float(momentum)
    eps = float(eps)
    freeze_params = kwargs.get('freeze_params', [])
    custom_lr_dict = kwargs.get('custom_lr_dict', {})
    for name, param in model.named_parameters():
        freeze_flag = False
        for freeze_param in freeze_params:
            if freeze_param in name:
                freeze_flag = True
                break
        if freeze_flag:
            print("name={} param.requires_grad = False".format(name))
            param.requires_grad = False

    if params is None:
        if weight_decay:
            skip = {}
            if no_decay_keys is not None:
                skip = no_decay_keys
            elif hasattr(model, 'no_weight_decay'):
                skip = model.no_weight_decay()
            param_configs = get_parameter_groups(model, custom_lr_dict, weight_decay, skip, assigner)
            weight_decay = 0.
        else:
            param_configs = model.parameters()
    else:
        param_configs = params
    if optimizer_type == "sgd":
        optimizer = optim.SGD(param_configs, momentum=momentum, nesterov=True, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "adam":
        optimizer = optim.Adam(param_configs, lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)
    elif optimizer_type == "adadelta":
        optimizer = optim.Adadelta(param_configs, lr=lr, eps=eps, weight_decay=weight_decay)
    elif optimizer_type == "rmsprob":
        optimizer = optim.RMSprop(param_configs, lr=lr, eps=eps, weight_decay=weight_decay, momentum=momentum)
    elif optimizer_type == "adamw":
        optimizer = optim.AdamW(param_configs, lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)
    elif optimizer_type == "adafactor":
        from transformers.optimization import Adafactor
        optimizer = Adafactor(param_configs, scale_parameter=False, relative_step=False, lr=lr,
                              weight_decay=weight_decay)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', optimizer_type)
    return optimizer


def get_scheduler(optimizer,
                  scheduler_type="linear",
                  num_warmup_steps=0,
                  num_training_steps=10000,
                  last_epoch=-1,
                  step_size=10,
                  gamma=0.1,
                  epochs=20,
                  **kwargs):
    gamma = float(gamma)
    if scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_training_steps,
                                                    last_epoch=last_epoch)
    elif scheduler_type == 'cosine_epoch':
        scheduler = get_cosine_schedule_by_epochs(optimizer, num_epochs=epochs, last_epoch=last_epoch)
    elif scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_training_steps,
                                                    last_epoch=last_epoch)
    elif scheduler_type == "stairs":
        logger.info("current use stair scheduler")
        scheduler = get_stairs_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_training_steps,
                                                    last_epoch=last_epoch,
                                                    **kwargs)
    elif scheduler_type == "step":
        step_size = int(step_size)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)
        """
        def exp_decay(epoch):
           initial_lrate = 0.1
           k = 0.1
           lrate = initial_lrate * exp(-k*t)
           return lrate
        """

    else:
        scheduler = get_constant_schedule_with_warmup(optimizer,
                                                      num_warmup_steps=num_warmup_steps,
                                                      last_epoch=last_epoch)
    return scheduler


def get_scheduler2(optimizer,
                   scheduler_type="cosine",
                   num_warmup_steps=0,
                   num_training_steps=10000,
                   decay_steps=1000,
                   decay_rate=0.1,
                   lr_min=5e-6,
                   warmup_lr=5e-7):
    lr_min = float(lr_min)
    warmup_lr = float(warmup_lr)
    decay_rate = float(decay_rate)
    if scheduler_type == "cosine":
        scheduler = CosineLRScheduler(optimizer,
                                      t_initial=num_training_steps,
                                      t_mul=1,
                                      lr_min=lr_min,
                                      warmup_lr_init=warmup_lr,
                                      cycle_limit=1,
                                      t_in_epochs=False)
    elif scheduler_type == "linear":
        scheduler = LinearLRScheduler(optimizer,
                                      t_initial=num_training_steps,
                                      lr_min_rate=0.01,
                                      warmup_lr_init=warmup_lr,
                                      warmup_t=num_warmup_steps,
                                      t_in_epochs=False)
    else:
        scheduler = StepLRScheduler(optimizer,
                                    decay_t=decay_steps,
                                    decay_rate=decay_rate,
                                    warmup_lr_init=warmup_lr,
                                    warmup_t=num_warmup_steps,
                                    t_in_epochs=False)
    return scheduler


def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def get_tensorboard_texts(label_texts):
    new_labels = []
    for label_text in label_texts:
        new_labels.append(label_text.replace("/", "//").replace("<", "/<").replace(">", "/>"))
    return "  \n".join(new_labels)


def get_parameter_groups(model, custom_lr_dict, weight_decay, skip_list=(), assigner=None):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias") or name.split('.')[-1] in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if assigner is not None:  # for layer decay
            layer_id = assigner.get_layer_id(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        custom_lr = None
        for key in custom_lr_dict.keys():
            if key in name:
                group_name = "{}.{}".format(group_name, key)
                custom_lr = custom_lr_dict[key]

        if group_name not in parameter_group_names:
            if assigner is not None:
                scale = assigner.get_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {"weight_decay": this_weight_decay, "params": [], "lr_scale": scale}
            parameter_group_vars[group_name] = {"weight_decay": this_weight_decay, "params": [], "lr_scale": scale}
            if custom_lr:
                parameter_group_names[group_name]["lr"] = custom_lr
                parameter_group_vars[group_name]["lr"] = custom_lr

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


class LayerDecayValueAssigner(object):

    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer(var_name, len(self.values))


def get_num_layer(var_name, num_max_layer):
    var_name = var_name.split('.', 1)[-1]
    if var_name.startswith("embeddings"):
        return 0
    elif var_name.startswith("encoder.layer"):
        layer_id = int(var_name.split('.')[2])
        return layer_id + 1
    else:
        return num_max_layer - 1
