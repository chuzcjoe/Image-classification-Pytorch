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
from eval import *


img_dir = "../yolov3/data/test_imgs"
xml_dir = "../yolov3/data/test_anns"

input_size = 224

transformations = transforms.Compose([transforms.Resize((input_size, input_size)),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

imgs = glob.glob(img_dir+'/*.jpg')

total = len(imgs)
cnt = 0

model = torch.load('./models/model_24.pt')
model.cuda(1)
model.eval()

for img in imgs:
    im = Image.open(img)
    im = im.convert("RGB")
    
    basename = os.path.basename(img)
    bboxs, labels = parse_xml(os.path.join(xml_dir, basename.split(".")[0]+'.xml'))

    bbox = bboxs[0]
    label = labels[0]
    x_min, y_min, x_max, y_max = bbox
    im = im.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

    h = max(im.size)
    l = min(im.size)

    #Add padding
    if h / l > 1.5:
        if h == im.size[0]:
            im = transforms.Pad((0, (h-l)//2))(im)
        elif h == im.size[1]:
            im = transforms.Pad(((h-l)//2, 0))(im)

	#save cropped image
    im.save(os.path.join('./cropped' ,basename.split(".")[0]+'_crop.jpg'))


    im = transformations(im)

    #BGR
    im = im[np.array([2,1,0]), :, :]
    im = im.unsqueeze(0)

    res, topk, topk_probs = test_img(model, im)
    if label in topk:
        cnt += 1

    print('-'*50)
    print(img)
    print("label:   ", label)
    print("prediction:  ", topk)
    print("probabilities: ", topk_probs)

print('='*50)
print('Testing accuracy: ', cnt / total)
