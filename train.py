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
    parser.add_argument("--epochs", dest="epochs", help="Maximum number of training epochs.",
                        default=20, type=int)
    parser.add_argument("--num_classes", dest="num_classes", help="classification numbers",
                        default=43, type=int)
    parser.add_argument("--batch_size", dest="batch_size", help="batch size",
                        default=64, type=int)
    parser.add_argument("--lr_resnet", dest="lr_resnet", help="Base learning rate",
                        default=0.001, type=float)
    parser.add_argument("--img_dir", dest="img_dir", help="directory path of train dataset",
                        default="", type=str)
    parser.add_argument("--xml_dir", dest="xml_dir", help="directory path of valid dataset",
                        default="", type=str)
    parser.add_argument("--input_size", dest="input_size", choices=[224, 192, 160, 128, 96], help="size of input images",
                        default=224, type=int)
    args = parser.parse_args()
    return args


def train(img_dir, xml_dir, epochs, input_size, batch_size, num_classes):
    """
    params: 
          bins: number of bins for classification
          alpha: regression loss weight
          beta: ortho loss weight
    """ 
    # create model
    model = ResNet(torchvision.models.resnet50(pretrained=True), num_classes=num_classes)

    cls_criterion = nn.CrossEntropyLoss().cuda(1)

    softmax = nn.Softmax(dim=1).cuda(1)
    model.cuda(1)
    

    # initialize learning rate and step
    lr = 0.001
    step = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #load data
    train_data_loader = loadData(img_dir, xml_dir, input_size, batch_size, True)
    test_loader = loadData('../yolov3/data/test_imgs', '../yolov3/data/test_anns', 224, 8, False)

    #variables
    history = []
    best_acc = 0.0
    best_epoch = 0

    # start training
    for epoch in range(epochs):
        print("Epoch:", epoch)
        print("------------")

        # reduce lr by lr_decay factor for each epoch
        if epoch % 10 == 0:
            lr = lr * 0.9
        
        train_loss = 0.0
        train_acc = 0
        val_acc = 0

        model.train()

        for i, (images, labels) in enumerate(train_data_loader):
            if i % 10 == 0:
                print("batch: {}/{}".format(i, len(train_data_loader.dataset)//batch_size))
            images = images.cuda(1)
            labels = labels.cuda(1)

            # backward
            optimizer.zero_grad()
            outputs = model(images)

            loss = cls_criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() 

            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            train_acc += acc.item() * images.size(0)

        print("epoch: {:03d}, Training loss: {:.4f}, Accuracy: {:.4f}%".format(epoch+1, train_loss, train_acc/3096*100))

        #if (epoch+1) % 3 == 0:
        #    torch.save(model, 'models/'+'model_'+str(epoch+1)+'.pt')
        print("Start testing...")
        with torch.no_grad():
            model.eval()

            for j, (images, labels) in enumerate(test_loader):
                images = images.cuda(1)
                labels = labels.cuda(1)

                outputs = model(images)

                ret, preds = torch.max(outputs.data, 1)
                cnt = preds.eq(labels.data.view_as(preds))

                acc = torch.mean(cnt.type(torch.FloatTensor))
                val_acc += acc.item() * images.size(0)

            if val_acc > best_acc:
                print("correct testing samples:", val_acc)
                best_acc = val_acc
                torch.save(model, 'models/'+'model_'+str(epoch+1)+'.pt')





if __name__ == "__main__":
    args = parse_args()
    img_dir = args.img_dir
    xml_dir = args.xml_dir
    epochs = args.epochs
    input_size = args.input_size
    batch_size = args.batch_size
    num_classes = args.num_classes

    # run train function
    train(img_dir, xml_dir, epochs, input_size, batch_size, num_classes)
