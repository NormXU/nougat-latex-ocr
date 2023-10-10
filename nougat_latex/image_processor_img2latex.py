# -*- coding:utf-8 -*-
# create: @time: 9/22/23 16:11

from PIL import Image
from .image_processing_nougat import NougatImageProcessor

class NougatLaTexProcessor(NougatImageProcessor):

    def __init__(self, img_height=224, img_width=560, **kwargs):
        self.imgH = img_height
        self.imgW = img_width
        self.maxH = img_height // 2
        super(NougatLaTexProcessor, self).__init__(**kwargs)

    def __call__(self, images, **kwargs):
        images = self._rescale(images)
        return self.preprocess(images, **kwargs)

    def _rescale(self, images):
        if images.height < self.maxH:
            empty_img = Image.new("RGB", (self.imgW, self.imgH))
            target_w = max(1, int(images.width / images.height * self.maxH))
            if target_w > self.imgW:
                target_w = self.imgW
                target_h = max(1, int(self.imgW / images.width * images.height))
            else:
                target_h = self.maxH
            images = images.resize((target_w, target_h))
            start_h = (self.imgH - target_h) // 2
            start_w = 0
            empty_img.paste(images, (start_w, start_h))
            images = empty_img
        return images
