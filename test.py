# -*- coding:utf-8 -*-
"""
    training HeadPoseNet
"""
import os
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from dataset import loadData
from net import ResNet
from tensorboardX import SummaryWriter
from torchvision.models.mobilenet import model_urls
import torchvision

gpu = '1'

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description="TriNet: Head Pose Estimation")
    parser.add_argument("--num_classes", dest="num_classes", help="classification numbers",
                        default=43, type=int)
    parser.add_argument('--snapshot', dest='snapshot', help='Name of model snapshot.',
                        default='', type=str)
    parser.add_argument("--batch_size", dest="batch_size", help="batch size",
                        default=64, type=int)
    parser.add_argument("--img_dir", dest="img_dir", help="directory path of train dataset",
                        default="", type=str)
    parser.add_argument("--xml_dir", dest="xml_dir", help="directory path of valid dataset",
                        default="", type=str)
    parser.add_argument("--input_size", dest="input_size", choices=[224, 192, 160, 128, 96], help="size of input images",
                        default=224, type=int)
    args = parser.parse_args()
    return args


def test(img_dir, xml_dir, input_size, batch_size, model, test_loader):
    
    test_acc = 0

    for i, (images, labels) in enumerate(test_loader):
        with torch.no_grad():

            images = images.cuda(0)
            labels = labels.cuda(0)

            outputs = model(images)

            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            test_acc += acc.item() * images.size(0)

    print("Testing Accuracy: {}".format(test_acc/len(test_loader.dataset)))




if __name__ == "__main__":
    args = parse_args()
    img_dir = args.img_dir
    xml_dir = args.xml_dir
    input_size = args.input_size
    batch_size = args.batch_size
    num_classes = args.num_classes
    snapshot = args.snapshot

    #model = ResNet(torchvision.models.resnet50(pretrained=False), num_classes)

    print("Loading weight......")
    #saved_state_dict = torch.load(snapshot)
    #model.load_state_dict(saved_state_dict)

    model = torch.load(snapshot)
    model.cuda(0)
    model.eval()

    test_loader = loadData(img_dir, xml_dir, input_size, batch_size, False)

    print("Start testing...")

    # run train function
    test(img_dir, xml_dir, input_size, batch_size, model, test_loader)
