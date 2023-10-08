# -*- coding:utf-8 -*-
# email:xunuo@datagrand.com
# create: @time: 9/22/23 16:11

from PIL import Image


class NougatLaTexProcessor:

    def __init__(self, image_processor, img_height=224, img_width=560):
        self.imgH = img_height
        self.imgW = img_width
        self.maxH = img_height // 2
        self.image_processor = image_processor

    def __call__(self, images, **kwargs):
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
        return self.image_processor(images, return_tensors="pt", **kwargs).pixel_values
