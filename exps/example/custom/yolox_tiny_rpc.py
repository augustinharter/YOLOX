#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os
from yolox.data.dataloading import get_yolox_datadir

from yolox.exp import Exp as MyExp

# yolox tiny template
class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.375
        self.input_size = (416, 416)
        self.mosaic_scale = (0.5, 1.5)
        self.random_size = (10, 20)
        self.test_size = (416, 416)
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.enable_mixup = False

        # Define yourself dataset path
        #self.data_dir = "/home/augo/data/retail_product_checkout/"
        self.data_dir = os.path.join(get_yolox_datadir(), "retail_product_checkout")
        self.train_ann = "instances_test2019.json"
        self.train_name = "test2019"
        self.val_ann = "instances_val2019.json"
        self.val_name = "val2019"
        self.num_classes = 200

        self.max_epoch = 2
        self.data_num_workers = 1
        self.eval_interval = 1
