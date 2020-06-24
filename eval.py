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

idx2labels = {}
k = 1
softmax = torch.nn.Softmax(dim=1).cuda(1)

with open('names.txt','r') as f:
	objs = f.read().splitlines()

for i, obj in enumerate(objs):
	idx2labels[i] = obj


def test_img(model, img):
    with torch.no_grad():
        img = img.cuda(1)
        output = model(img)
        probs = softmax(output)
        
        ret, prediction = torch.max(output.data, 1)
        
        topk = np.array(torch.topk(output.data, k=k))[1]
        topk_p = np.array(torch.topk(probs.data, k=k))[0][0]

        #print(topk)
        #print(topk_p)
        
        #return idx2labels[prediction.item()], [idx2labels[int(t)] for t in topk[0]], topk_p.item()
        return idx2labels[prediction.item()], idx2labels[int(topk[0])], topk_p.item()
