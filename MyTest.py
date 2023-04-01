import os, argparse

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from scipy import misc

from torch import nn

from lib.BCMNet import BCMNet
from utils.dataloader import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=512, help='testing size')
parser.add_argument('--pth_path', type=str,
                    default='G:\CJG\BCMNet\checkpoints\BCMNet.pth')

for _data_name in ['CHAMELEON', 'COD10K', "MAS3K"]:
    data_path = './data/TestDataset/{}/'.format(_data_name)
    save_path = './results/BCMNet/{}/'.format(_data_name)
    opt = parser.parse_args()

    model = BCMNet(channel=64)

    model = nn.DataParallel(model).cuda()
    checkpoint = torch.load(opt.pth_path)
    model.load_state_dict(checkpoint['model'])
    model = nn.DataParallel(model).cuda()

    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/Imgs/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        P1, P2, P3, P4, P5, P6 = model(image)

        res = F.upsample(P3 + P4, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()

        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print("第{}张图像开始保存".format(i))
        cv2.imwrite(save_path + name, res * 255)
