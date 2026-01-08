import cv2
import argparse
import random
from os.path import isfile

import torch
import torch.backends.cudnn as cudnn
import torch.optim
import yaml
from torch.optim import lr_scheduler
from logger import Logger
from torchvision import transforms
from nets.network_mit_unet import Net


CONFIGS = yaml.load(open("config.yml"), Loader=yaml.FullLoader)

img_path = "data/images/1_00217.jpg"

logger = Logger()
logger.info(CONFIGS)
torch.set_grad_enabled(False)
GPU_ID = CONFIGS["GPU_ID"]
seg_threshold = CONFIGS["SEG_THRESHOLD"]
pretrained_weight = CONFIGS["PRETRAINED_WEIGHT"]
phi = CONFIGS["PHI"]

def main():
    model = Net(phi=phi)
    model.cuda(device=GPU_ID)

    if isfile(pretrained_weight):
        state_dict = model.state_dict()
        model_dict = {}
        pretrain_dict = torch.load(pretrained_weight, map_location="cuda:0")
        pretrain_dict_items = pretrain_dict.items() if "state_dict" not in pretrain_dict else pretrain_dict["state_dict"].items()
        for k, v in pretrain_dict_items:
            k = "mit_unet" + k[9:]
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        model.load_state_dict(state_dict)
        print(f"Loading pretrained weight: '{pretrained_weight}' done.")
    else:
        logger.info("=> no pretrained weight found at '{}'".format(pretrained_weight))

    model.eval()
    image_origin = cv2.imread(img_path)
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转为Tensor格式
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化操作
    ])
    image = transform(image_origin)
    image = image.unsqueeze(0).cuda(device=GPU_ID)

    seg_maps = model(image)
    seg_maps = seg_maps.squeeze(0).squeeze(0)
    seg_maps = (torch.sigmoid(seg_maps) > seg_threshold).float().cpu().numpy()
    cv2.imshow("image", seg_maps*255)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
