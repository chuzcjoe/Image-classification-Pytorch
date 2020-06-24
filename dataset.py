# -*-coding:utf-8 -*-
"""
    DataSet class
"""
import os
import glob
import torch
import numpy as np
from PIL import Image
from PIL import ImageFilter
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from xml_helper import *

torch.manual_seed(0)


def loadData(img_dir, xml_dir, input_size, batch_size, training=True):
    """

    :return:
    """
    # define transformation
    if training:
        transformations = transforms.Compose([transforms.Resize((input_size, input_size)),
                                              transforms.RandomRotation(degrees=90),
                                              transforms.RandomHorizontalFlip(p=0.5),
                                              transforms.RandomVerticalFlip(p=0.5),
                                              transforms.ColorJitter(brightness=1, contrast=1, saturation=0.5, hue=0.5),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        train_dataset = TrainDataSet(img_dir, xml_dir, transformations)
        print("Traning sampels:", train_dataset.length)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        return train_loader
    else:
        transformations = transforms.Compose([transforms.Resize((input_size, input_size)),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        test_dataset = TestDataSet(img_dir, xml_dir, transformations)

        # initialize train DataLoader
        data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        return data_loader


class TrainDataSet(Dataset):
    def __init__(self, img_dir, xml_dir, transform, image_mode="RGB"):
        self.imgs = glob.glob(img_dir+'/*.jpg')
        self.label2idx = {}
        self.length = len(self.imgs)
        self.image_mode = image_mode
        self.xml_dir = xml_dir
        self.transform = transform

        with open('names.txt','r') as f:
            lines = f.read().splitlines()

        for i, obj in enumerate(lines):
            self.label2idx[obj] = i 

    def __getitem__(self, index):

        img_path = self.imgs[index]
        basename = os.path.basename(img_path)

        # read image file
        img = Image.open(img_path)
        img = img.convert(self.image_mode)

        # get face bounding box
        bboxs, labels = parse_xml(os.path.join(self.xml_dir, basename.split(".")[0]+'.xml'))
        bbox = bboxs[0]
        label = labels[0]
        x_min, y_min, x_max, y_max = bbox

        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        h = max(img.size)
        l = min(img.size)

        #Add padding for large w:h/h:w images
        if h / l > 1.5:
            # long width side
            if h == img.size[0]:
                img = transforms.Pad((0, (h-l)//2))(img)
            # long height side
            elif h == img.size[1]:
                img = transforms.Pad(((h-l)//2, 0))(img)

        # Augmentation:Blur?
        if np.random.random_sample() < 0.5:
            img = img.filter(ImageFilter.BLUR)

        # Augmentation:Gray?
        if np.random.random_sample() < 0.5:
            img = img.convert('L').convert("RGB")

        # transform
        if self.transform:
            img = self.transform(img)

        # RGB2BGR
        img = img[np.array([2, 1, 0]), :, :]

        return img, self.label2idx[label]

    def __len__(self):
        return self.length


class TestDataSet(Dataset):
    def __init__(self, img_dir, xml_dir, transform, image_mode="RGB"):
        self.imgs = glob.glob(img_dir+'/*.jpg')
        self.label2idx = {}
        self.length = len(self.imgs)
        self.image_mode = image_mode
        self.xml_dir = xml_dir
        self.transform = transform

        with open('names.txt','r') as f:
            lines = f.read().splitlines()

        for i, obj in enumerate(lines):
            self.label2idx[obj] = i 

    def __getitem__(self, index):

        img_path = self.imgs[index]
        basename = os.path.basename(img_path)

        # read image file
        img = Image.open(img_path)
        img = img.convert(self.image_mode)

        # get face bounding box
        bboxs, labels = parse_xml(os.path.join(self.xml_dir, basename.split(".")[0]+'.xml'))
        bbox = bboxs[0]
        label = labels[0]
        x_min, y_min, x_max, y_max = bbox

        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        h = max(img.size)
        l = min(img.size)

        #Add padding
        if h / l > 1.5:
            if h == img.size[0]:
                img = transforms.Pad((0, (h-l)//2))(img)
            elif h == img.size[1]:
                img = transforms.Pad(((h-l)//2, 0))(img)

        # transform
        if self.transform:
            img = self.transform(img)

        # RGB2BGR
        img = img[np.array([2, 1, 0]), :, :]

        return img, self.label2idx[label]

    def __len__(self):
        return self.length
