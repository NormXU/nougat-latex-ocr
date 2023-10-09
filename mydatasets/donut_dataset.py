# -*- coding:utf-8 -*-
# create: @time: 6/6/23 10:30
import glob
import os
import os.path
import random
from os.path import join

import albumentations as alb
import cv2
import imagesize
import numpy as np
import torch
from PIL import Image, ImageFile
from base.driver import CACHE_ROOT
from nougat_latex.util import process_raw_latex_code
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True
nougut_train_transform = alb.Compose(
    [
        alb.RGBShift(r_shift_limit=15, g_shift_limit=15,
                     b_shift_limit=15, p=0.15),
        alb.Compose(
            [alb.ShiftScaleRotate(shift_limit=0.01, scale_limit=(-.15, 0), rotate_limit=1, border_mode=0,
                                  interpolation=3,
                                  value=[255, 255, 255], p=1),
             alb.GridDistortion(distort_limit=0.1, border_mode=0, interpolation=3, value=[255, 255, 255], p=.5)],
            p=.15),
        # alb.ElasticTransform(alpha=0.2, sigma=0.1, alpha_affine=0.1, p=.1),
        alb.RandomBrightnessContrast(.1, (-.2, 0), True, p=.2),
        alb.ImageCompression(95, p=.15),
        alb.GaussNoise(2, p=.2),
        # alb.GaussianBlur(p=.2)
    ]
)


class NougatDataset(Dataset):
    def __init__(
            self,
            data_root: list,
            equations,
            processor,
            max_length: int,
            phase: str = "train",
            max_dimensions=(1024, 512), min_dimensions=(32, 32),
            **kwargs
    ):
        super().__init__()
        self.processor = processor
        self.max_length = max_length
        self.phase = phase
        self.images = []

        all_images = [path for images in data_root for path in glob.glob(join(images, '*.png'))]
        for img in all_images:
            width, height = imagesize.get(img)
            if min_dimensions[0] <= width <= max_dimensions[0] and min_dimensions[1] <= height <= max_dimensions[1]:
                self.images.append(img)
        eqs = open(equations, 'r').read().split('\n')
        self.indices = [int(os.path.basename(img).split('.')[0]) for img in self.images]

        self.pairs = list()
        for i, im in tqdm(enumerate(self.images), total=len(self.images)):
            self.pairs.append((eqs[self.indices[i]], im))
        if phase == "train":
            self.transforms = nougut_train_transform
        else:
            self.transforms = None

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        batch = self.pairs[idx]
        eqs, img_path = batch
        eqs = process_raw_latex_code(eqs)  # remove unnecessary blank in the latex codes
        im = cv2.imread(img_path)
        tok = self.processor.tokenizer(eqs, return_token_type_ids=False)
        # check if sequence length is too long
        if self.max_length < len(tok.attention_mask) or im is None:
            random_index = random.randrange(self.__len__())
            return self.__getitem__(random_index)
        # augmentation
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if self.phase == "train":
            # sometimes convert to bitmask
            if np.random.random() < .04:
                im[im != 255] = 0
        if self.transforms is not None:
            im = self.transforms(image=im)['image']
        return tok, im, eqs


class NougatPadFixSizeCollectFn(object):
    def __init__(self, batch_size, processor, debug=False, maxH=None, **kwargs):
        self.debug = debug
        self.batch_size = batch_size
        self.pad_token_id = processor.tokenizer.pad_token_id
        self.processor = processor
        self.imgW = processor.image_processor.size['width']
        self.imgH = processor.image_processor.size['height']
        self.maxH = self.imgH // 2 if maxH is None else maxH

    def __call__(self, instances):
        batch = dict(labels=list(), mask=list(), processed_parse=list())
        img_batch = []
        for instance in instances:
            # labels
            batch['labels'].append(instance[0].input_ids[1:])
            # attention mask
            batch['mask'].append(instance[0].attention_mask[1:])
            batch['processed_parse'].append(instance[2])
            img_batch.append(to_pil_image(instance[1]))
        # input_ids
        batch['labels'] = pad_sequence([torch.LongTensor(x) for x in batch['labels']], batch_first=True,
                                       padding_value=self.pad_token_id)
        # attention_mask
        batch['mask'] = pad_sequence([torch.LongTensor(x) for x in batch['mask']], batch_first=True,
                                     padding_value=0)
        pixel_values = []
        # pixel_values
        max_w = self.imgW
        for i, img in enumerate(img_batch):
            if img.height < self.maxH:
                empty_img = Image.new("RGB", (self.imgW, self.imgH))
                target_w = max(1, int(img.width / img.height * self.maxH))
                if target_w > max_w:
                    target_w = max_w
                    target_h = max(1, int(max_w / img.width * img.height))
                else:
                    target_h = self.maxH
                img = img.resize((target_w, target_h))
                start_h = (self.imgH - target_h) // 2
                start_w = 0
                empty_img.paste(img, (start_w, start_h))
                img = empty_img
            pixel_values.append(self.processor(img, return_tensors="pt").pixel_values)
            if self.debug:
                cache_path = os.path.join(CACHE_ROOT, "{}.jpg".format(i))
                img.save(cache_path)
        batch['pixel_values'] = torch.cat(pixel_values)
        return batch
