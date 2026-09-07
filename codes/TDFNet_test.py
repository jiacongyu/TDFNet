import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
import cv2
from data import test_dataset_pano
import time

from models.TDFNet import ImgModel

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=512, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--fusion_strategy', type=str, default='geometry_only',
                    choices=['geometry_only', 'mixed'],
                    help='fusion strategy must match training: geometry_only for 400, mixed for 1150')
parser.add_argument('--model_path', type=str,
                    default=r'D:/study/qjxzxjc/TDFNet/weights/CSMA-Net_epoch_15(400).pth',
                    help='path to trained model')
opt = parser.parse_args()

# set device for test
if opt.gpu_id == '0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id == '1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')

print(f'Loading model with fusion_strategy: {opt.fusion_strategy}')
model = ImgModel(fusion_strategy=opt.fusion_strategy)
model.load_state_dict(torch.load(opt.model_path))
model.cuda()
model.eval()

# test
TIME = []
test_datasets = ['sod']
for dataset in test_datasets:
    save_path = './test_maps/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = r'D:/study/dataset/360-SOD/test/images/'
    gt_root = r'D:/study/dataset/360-SOD/test/labels/'
    test_loader = test_dataset_pano(image_root, gt_root, opt.testsize)
    for i in range(test_loader.size):
        image, gt, ER_img, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        ER_img = ER_img.cuda()
        time_s = time.time()
        res = model(image, ER_img)
        time_e = time.time()
        TIME.append(time_e - time_s)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('save img to: ', save_path + name)
        cv2.imwrite(save_path+name, res * 255)
print('Speed: %f FPS' % (462 / np.sum(TIME)))
print('Test Done!')