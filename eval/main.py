import os
import argparse
import pandas as pd
from PIL import Image
import numpy as np
import cv2
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt as distance

from sklearn.metrics import f1_score
from utils import AverageMeter
from metrics import dice_coef, iou_score, get_F1, get_accuracy, get_recall, get_precision



def calculate_hd95(pred, gt, spacing=1.0):

    pred = np.atleast_1d(pred.astype(np.bool_))
    gt = np.atleast_1d(gt.astype(np.bool_))


    if pred.sum() == 0 or gt.sum() == 0:

        if pred.sum() == 0 and gt.sum() == 0:
            return 0.0
        return 100.0

    dst_border_gt = distance(np.logical_not(gt))
    dst_border_pred = distance(np.logical_not(pred))

    hd1 = dst_border_gt[pred]
    hd2 = dst_border_pred[gt]

    hd_all = np.concatenate([hd1, hd2])


    res = np.percentile(hd_all, 95)
    return res * spacing

def calculate_boundary_f1(pred, gt, tolerance=2.0):

    pred = (pred * 255).astype(np.uint8) if pred.max() <= 1 else pred.astype(np.uint8)
    gt = (gt * 255).astype(np.uint8) if gt.max() <= 1 else gt.astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    pred_border = cv2.subtract(pred, cv2.erode(pred, kernel))
    gt_border = cv2.subtract(gt, cv2.erode(gt, kernel))

    if np.count_nonzero(pred_border) == 0 or np.count_nonzero(gt_border) == 0:
        if np.count_nonzero(pred_border) == 0 and np.count_nonzero(gt_border) == 0:
            return 1.0
        return 0.0

    dt_gt = cv2.distanceTransform(255 - gt_border, cv2.DIST_L2, 5)
    dt_pred = cv2.distanceTransform(255 - pred_border, cv2.DIST_L2, 5)
    pred_border_indices = pred_border > 0
    distances_to_gt = dt_gt[pred_border_indices]
    tp_precision = np.sum(distances_to_gt <= tolerance)
    total_pred = np.sum(pred_border_indices)
    precision = tp_precision / (total_pred + 1e-8)
    gt_border_indices = gt_border > 0
    distances_to_pred = dt_pred[gt_border_indices]
    tp_recall = np.sum(distances_to_pred <= tolerance)
    total_gt = np.sum(gt_border_indices)
    recall = tp_recall / (total_gt + 1e-8)


    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return f1

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--testpath', type=str,
                       default="./Results/DDENet_BUSI/TrainDataset")
    parse.add_argument('--path', type=str, default="./TrainDataset/BUSI")

    args = parse.parse_args()

    for _dataname in ['test','STU']:  
        gtpath = os.path.join(args.path, _dataname, "masks")
        prepath = os.path.join(args.testpath, _dataname)


        if not os.path.exists(gtpath) or not os.path.exists(prepath):
            print(f"Path not found: {gtpath} or {prepath}")
            continue

        files = os.listdir(gtpath)
        lens = len(files)
        print("##" * 20)
        print(_dataname, "length: ", lens)
        print("##" * 20)

        
        iou = AverageMeter()
        dice = AverageMeter()
        f1 = AverageMeter()
        acc = AverageMeter()
        recall = AverageMeter()
        precision = AverageMeter()
        hd95_meter = AverageMeter()
        boundary_f1_meter = AverageMeter()

        for file in files:
            gtfile = os.path.join(gtpath, file)
            prefile = os.path.join(prepath, file)

            
            gt_img = Image.open(gtfile).convert('L')
            pre_img = Image.open(prefile).convert('L')

            gt_arr = np.asarray(gt_img)
            pre_arr = np.asarray(pre_img)

            h, w = gt_arr.shape[:2]

            
            gt_norm = gt_arr.reshape(h, w, 1) / 255.0
            pre_norm = pre_arr.reshape(h, w, 1) / 255.0

            
            iou.update(iou_score(pre_norm, gt_norm))
            dice.update(dice_coef(pre_norm, gt_norm))
            f1.update(get_F1(pre_norm, gt_norm))
            acc.update(get_accuracy(pre_norm, gt_norm))
            recall.update(get_recall(pre_norm, gt_norm))
            precision.update(get_precision(pre_norm, gt_norm))


            gt_binary = (gt_norm[:, :, 0] > 0.5).astype(int)
            pre_binary = (pre_norm[:, :, 0] > 0.5).astype(int)

            val_hd95 = calculate_hd95(pre_binary, gt_binary)
            hd95_meter.update(val_hd95)

        print("-" * 30)
        print("Results for:", _dataname)
        print(f"Acc: {acc.avg:.4f}")
        print(f"IoU: {iou.avg:.4f}")
        print(f"Dice: {dice.avg:.4f}")
        print(f"F1: {f1.avg:.4f}")
        print(f"Recall: {recall.avg:.4f}")
        print(f"Precision: {precision.avg:.4f}")
        print("-" * 30)
        print(f"HD95 (pixels): {hd95_meter.avg:.4f}")  # 越小越好
        print(f"Boundary F-score: {boundary_f1_meter.avg:.4f}")  # 越高越好
        print("-" * 30)
