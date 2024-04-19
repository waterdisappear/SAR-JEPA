# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


from torch.utils.data import Dataset

class MyDataSet(Dataset):
    def __init__(self, image_list, label_list, transform=None):
        self.image_list = image_list
        self.label_list = label_list
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img = self.image_list[index]
        # img = cv2.imread(img, cv2.IMREAD_COLOR)
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

        img = Image.fromarray(img)
        label = self.label_list[index]
        if self.transform:
            img = self.transform(img)
        return img, label

from PIL import Image #导入PIL库
import numpy as np
import re
import cv2
def load_data(file_dir, transform):
    pic_list = []
    label_list = []
    for root, dirs, files in os.walk(file_dir):
        files = sorted(files)
        for file in files:
            jpeg_path = os.path.join(root, file)
            # jpeg = cv2.imread(jpeg_path, cv2.IMREAD_GRAYSCALE)
            # jpeg = Image.fromarray(cv2.cvtColor(jpeg,cv2.COLOR_BGR2RGB))
            # pic_list.append(jpeg)
            pic_list.append(jpeg_path)
            label_list.append(re.split('[/\\\]', jpeg_path)[-1])

    dataset = MyDataSet(pic_list, label_list, transform)
    return dataset