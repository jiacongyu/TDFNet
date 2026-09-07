import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--epoch', type=int, default=50, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=2, help='training batch size')
parser.add_argument('--trainsize', type=int, default=512, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
parser.add_argument('--dataset_size', type=int, default=850,
                    choices=[107, 400, 850, 4000],
                    help='dataset size for auto strategy selection (107/400/850/4000)')
parser.add_argument('--fusion_strategy', type=str, default='auto',
                    choices=['auto', 'geometry_only', 'mixed'],
                    help='fusion strategy: auto=infer from dataset_size, geometry_only=0 param, mixed=lightweight')

parser.add_argument('--tr_img_root', type=str,
    default=r'/home/zqq/桌面/YJC/360-SSOD/train/images/', help='training images')
parser.add_argument('--tr_gt_root', type=str,
    default=r'/home/zqq/桌面/YJC/360-SSOD/train/labels/', help='training gt')
parser.add_argument('--save_path', type=str,
    default=r'/home/zqq/桌面/YJC/CSMA-Net-main_3/weights/', help='save checkpoints & logs')

opt = parser.parse_args()