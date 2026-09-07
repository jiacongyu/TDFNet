import os
import torch
import numpy as np
from datetime import datetime
from data import get_loader
from utils import clip_gradient
import logging
import torch.backends.cudnn as cudnn
from options import opt
from utils import print_network
from utils import structure_loss
import cv2
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR

from models.TDFNet import ImgModel

# set the device for training
if opt.gpu_id == '0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id == '1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')
cudnn.benchmark = True


def get_fusion_strategy(dataset_size, strategy_choice):

    if strategy_choice != 'auto':
        return strategy_choice

    if dataset_size <= 500:
        return 'geometry_only'
    elif dataset_size <= 1000:
        return 'mixed'
    else:
        return 'mixed'


fusion_strategy = get_fusion_strategy(opt.dataset_size, opt.fusion_strategy)
print(f"Dataset size: {opt.dataset_size}, Fusion strategy: {fusion_strategy}")

model = ImgModel(fusion_strategy=fusion_strategy)
# print_network(model, 'TDFNet')

if opt.load is not None:
    model.load_state_dict(torch.load(opt.load))
    print('load model from ', opt.load)
model.cuda()

frozen_count = 0
total_count = 0
if fusion_strategy == 'mixed' and opt.dataset_size > 1000:
    for name, param in model.named_parameters():
        total_count += 1
        if 'fusion_module.weight_generator' in name:
            param.requires_grad = False
            frozen_count += 1
    if frozen_count > 0:
        print(f"【大数据集优化】冻结 fusion_module.weight_generator 的 {frozen_count} 个参数")
        print(f"                仅保留 alpha 可学习，保持温度系数10.0不变")
        print(f"                模型总参数: {total_count}, 冻结参数: {frozen_count}")


params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(params, opt.lr, weight_decay=1e-4)


warmup_epochs = 5 if opt.dataset_size <= 500 else 10
warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
main_scheduler = CosineAnnealingLR(optimizer, T_max=opt.epoch - warmup_epochs, eta_min=1e-6)

# set the path
tr_img_root = opt.tr_img_root
tr_gt_root = opt.tr_gt_root


save_path = os.path.join(opt.save_path, f'size{opt.dataset_size}_{fusion_strategy}/')
if not os.path.exists(save_path):
    os.makedirs(save_path)

# load data
print('load data...')
train_loader = get_loader(tr_img_root, tr_gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)

logging.basicConfig(filename=save_path + 'log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("TDFNet-Train")
logging.info("Config")
logging.info(f'epoch:{opt.epoch};lr:{opt.lr};batchsize:{opt.batchsize};trainsize:{opt.trainsize};'
             f'clip:{opt.clip};load:{opt.load};save_path:{save_path};'
             f'dataset_size:{opt.dataset_size};fusion_strategy:{fusion_strategy}')

step = 0
best_loss = float('inf')
best_epoch = 0


# train function
def train(train_loader, model, optimizer, epoch, save_path):
    global step, best_loss, best_epoch, warmup_epochs
    model.train()
    loss_all = 0
    epoch_step = 0

    try:
        for i, (images, gts, er_images, er_gts) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = images.cuda()
            gts = gts.cuda()
            er_images = er_images.cuda()
            er_gts = er_gts.cuda()

            preds, preds_aux, cube_gts = model(images, er_images, er_gts)
            loss = structure_loss(preds, gts) + structure_loss(preds_aux, cube_gts)

            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            step += 1
            epoch_step += 1
            loss_all += loss.data

            if i % 200 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data))
                logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                             format(epoch, opt.epoch, i, total_step, loss.data))

        loss_all /= epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))

        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            main_scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f'当前学习率: {current_lr:.2e}')
        logging.info(f'当前学习率: {current_lr:.2e}')

        if loss_all < best_loss:
            best_loss = loss_all
            best_epoch = epoch
            torch.save(model.state_dict(), save_path + 'TDFNet_best.pth')
            print(f'新 best model 保存 (epoch {epoch}): Loss {loss_all:.4f}')
            logging.info(f'新 best model 保存 (epoch {epoch}): Loss {loss_all:.4f}')

        if epoch % 5 == 0:
            torch.save(model.state_dict(), save_path + 'TDFNet_epoch_{}.pth'.format(epoch))

    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'TDFNet_epoch_{}.pth'.format(epoch + 1))
        print('save checkpoints successfully!')
        raise


if __name__ == '__main__':
    print("Start train...")
    for epoch in range(1, opt.epoch + 1):
        train(train_loader, model, optimizer, epoch, save_path)