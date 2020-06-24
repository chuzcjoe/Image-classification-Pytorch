# -*- coding:utf-8 -*-
"""
    Implementation of Pose Estimation with mobileNetV2
"""
import torch.nn as nn
import torch
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url



class ResNet(nn.Module):
    def __init__(self, features, num_classes):
       super(ResNet, self).__init__()

       self.features = features
       self.last_channel = 1000

       # building classifier
       self.fc = nn.Linear(self.last_channel, num_classes)

    def forward(self, x):
        x = self.features(x)
        #x = self.avgpool(x)
        #x = torch.flatten(x, 1)
        #x = self.classifier(x)

        output = self.fc(x)

        return output




