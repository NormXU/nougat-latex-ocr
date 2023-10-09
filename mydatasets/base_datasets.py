# -*- coding:utf-8 -*-
# create: 2021/6/28

from torch.utils.data.dataset import Dataset
from base.driver import logger


class BaseDataset(Dataset):

    def __init__(self, data_root, name="", dataset_format="json", **kwargs):
        if isinstance(data_root, list):
            self.data_root = data_root
        else:
            self.data_root = [data_root]
        self.dataset_name = name
        self.dataset_format = dataset_format


class BaseImgDataset(Dataset):

    def __init__(self, data_root, name="", extensions=None, **kwargs):
        if isinstance(data_root, list):
            self.data_root = data_root
        else:
            self.data_root = [data_root]
        self.dataset_name = name
        if extensions is None:
            self.extensions = ["jpg", "png", "jpeg", "tif"]
        else:
            self.extensions = extensions
