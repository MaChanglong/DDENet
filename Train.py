import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR

import time
import ml_collections

from DDENet import DDENet

from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, AvgMeter
import torch.nn.functional as F
import numpy as np

def structure_loss(pred, mask):
    weit = 1 + 5 * \
        torch.abs(F.avg_pool2d(mask, kernel_size=31,
                  stride=1, padding=15) - mask)

    # print(pred.shape)
    # print(mask.shape)
    wbce = F.binary_cross_entropy_with_logits(pred, mask)
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)

    return (wbce + wiou).mean()

def adjust_learnrate(optims, lr):
    for param_groups in optims.param_groups:
        param_groups['lr'] = lr

def adjust_lr_d(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_rate

def test(model, path):

    data_path = path

    model.eval()
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, 352)
    b = 0.0

    print('[test_size]', test_loader.size)
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res3,res2, res1,res0 = model(image)
        res = res3

        res = F.interpolate(res,size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        input = res
        target = np.array(gt)

        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))

        intersection = (input_flat*target_flat)

        loss = (2 * intersection.sum() + smooth) / \
            (input.sum() + target.sum() + smooth)

        a = '{:.4f}'.format(loss)
        a = float(a)
        b = b + a


    return b / test_loader.size


def train(name, train_loader, model, optimizer, epoch, test_path):
    model.train()
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]
    loss_record0, loss_record1, loss_record2, loss_record3,loss_record4= AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.trainsize*rate/32)*32)
            if rate != 1:
                images = F.interpolate(images, size=(
                    trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.interpolate(gts, size=(
                    trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            lateral_map_3, lateral_map_2 , lateral_map_1 , lateral_map_0= model(images)
            # gts0 = F.interpolate(gts, size=(lateral_map_4.shape[3], lateral_map_4.shape[2]), mode='bilinear',
            #                      align_corners=False)
            gts1 = F.interpolate(gts, size=(lateral_map_3.shape[3], lateral_map_3.shape[2]), mode='bilinear', align_corners=False)
            gts2 = F.interpolate(gts, size=(lateral_map_2.shape[3], lateral_map_2.shape[2]), mode='bilinear', align_corners=False)
            gts3 = F.interpolate(gts, size=(lateral_map_1.shape[3], lateral_map_1.shape[2]), mode='bilinear', align_corners=False)
            gts4 = F.interpolate(gts, size=(lateral_map_0.shape[3], lateral_map_0.shape[2]), mode='bilinear', align_corners=False)

            # gts = F.interpolate(gts, size=(lateral_map.shape[3], lateral_map.shape[2]), mode='bilinear',align_corners=False)

            # ---- loss function ----
            loss3 = structure_loss(lateral_map_3, gts1)
            loss2 = structure_loss(lateral_map_2, gts2)
            loss1 = structure_loss(lateral_map_1, gts3)
            loss0 = structure_loss(lateral_map_0, gts4)

            loss = loss3+loss2+loss1+loss0
            # loss = structure_loss(lateral_map, gts)
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record3.update(loss3.data, opt.batchsize)
                loss_record2.update(loss2.data, opt.batchsize)
                loss_record1.update(loss1.data, opt.batchsize)
                loss_record0.update(loss0.data, opt.batchsize)

        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' lateral-3: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record3.show()))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=300, help='epoch number')

    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')

    parser.add_argument('--optimizer', type=str,
                        default='Adam', help='choosing optimizer Adam or SGD')

    parser.add_argument('--augmentation',
                        default=True, help='choose to do random flip rotation')

    parser.add_argument('--batchsize', type=int,
                        default=8, help='training batch size')

    parser.add_argument('--trainsize', type=int,
                        default=256, help='training dataset size')

    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')

    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')

    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')

    parser.add_argument('--train_path', type=str,
                        default='DDENet/TrainDataset/BUSI/train', help='path to train dataset')

    parser.add_argument('--test_path', type=str,
                        default=r'DDENet/TrainDataset/BUSI/val', help='path to testing BUSI dataset')

    parser.add_argument('--train_save', type=str,
                        default='DDENet_BUSI')
    parser.add_argument('--model_name', type=str,default='DDENet',
                        help='please input your model')
    parser.add_argument('--early_stop_patience', type=int, default=50,
                        help='Number of epochs to wait before early stopping')

    opt = parser.parse_args()

    # ---- build models ----
    # set ur gpu device, if you run by .sh, it is can comment.
    torch.cuda.set_device(0)

    if opt.model_name == 'DDENet':
        A = [DDENet(1)]

    else:
        raise TypeError('Please enter a valid name for the model name!')

    num = -1
    for i in range(len(A)):
        tmp = A[i].__class__.__name__
        if tmp == opt.model_name:
            num = i
            break

    if num == -1:
        raise RuntimeError('model Error!')

    model = A[num].cuda()
    name = model.__class__.__name__

    flname = opt.train_save
    print("train: ", name)
    fp = open('DDENet/logs/best-'+flname+'_dice.txt', 'w')
    fp.write('0\n')
    fp.close()

    params = model.parameters()

    if opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params, opt.lr)
    else:
        optimizer = torch.optim.SGD(
            params, opt.lr, weight_decay=1e-4, momentum=0.9)

    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize,
                              trainsize=opt.trainsize, augmentation=opt.augmentation)
    total_step = len(train_loader)


    print("#"*20, "Start Training", "#"*20)
    print("#"*20, "Train params", "#"*20)
    print("model: ", model.__class__.__name__)
    print("batch size: ", opt.batchsize)
    print("train size: ", opt.trainsize)
    print("init lr: ", opt.lr)
    print("train path : ", opt.train_path)
    print("test path : ", opt.test_path)
    print("#"*20, "Training", "#"*20)
    time.sleep(10)

    early_stop_counter = 0
    best_dice = 0.0
    save_path = 'snapshots/{}/'.format(opt.train_save)
    # save_path = 'snapshots1/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)

    lrs = CosineAnnealingLR(optimizer, T_max=300, eta_min=1e-5, verbose=True )
    for epoch in range(1, opt.epoch):
        torch.cuda.empty_cache()
        # lrs = opt.lr
        # if epoch > 120:
        #     lrs = opt.lr / 100
        # if epoch > 60:
        #     lrs = opt.lr / 10

        # adjust_learnrate(optimizer, lrs)

        print("epoch:", epoch)
        print("lr: ", optimizer.param_groups[0]['lr'])

        train(flname, train_loader, model, optimizer, epoch, opt.test_path)
        lrs.step()

        meandice = test(model, opt.test_path)

        fp = open(f'logs_second/log-{flname}_dice.txt', 'a')
        fp.write(f"{meandice}\n")
        fp.close()

        with open(f'logs_second/best-{flname}_dice.txt', 'r') as f:
            best_dice_file = float(f.read().strip() or 0)

        if meandice > best_dice_file:
            with open(f'logs_second/best-{flname}_dice.txt', 'w') as f:
                f.write(str(meandice))

            torch.save(model.state_dict(), os.path.join(save_path, f'{flname}.pth'))
            print(f'Saving new best model with dice: {meandice:.4f}')
            best_dice = meandice
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f'No improvement for {early_stop_counter} epochs')

            if early_stop_counter >= opt.early_stop_patience:
                print(f'Early stopping after {opt.early_stop_patience} epochs without improvement')
                break