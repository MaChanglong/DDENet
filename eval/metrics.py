import numpy as np
import torch
import torch.nn.functional as F

thsrhold=0.5
def iou_score(output, target):
    smooth = 1e-15

    # if torch.is_tensor(output):
    #     output = torch.sigmoid(output).data.cpu().numpy()
    # if torch.is_tensor(target):
    #     target = target.data.cpu().numpy()
    output_ = output >= thsrhold
    target_ = target >= thsrhold
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def dice_coef(output, target):
    smooth = 1e-15
    output_ = output > thsrhold
    output_ = output_.reshape(-1)
    output_ = output_.astype(np.uint8)
    target_ = target > thsrhold
    target_ = target_.reshape(-1)
    target_ = target_.astype(np.uint8)
    intersection = (output_ * target_).sum()
    return (2. * intersection + smooth) / \
        (output_.sum() + target_.sum() + smooth)


def get_accuracy(output,target):

    threshold = thsrhold
    tmp_out = (output >= threshold)
    tmp_out = tmp_out.astype(float)

    index = (tmp_out == target)
    index = index.astype(float)
    tmp_sum = np.sum(index)

    target_num = target.size

    return tmp_sum/target_num


def get_specificity(output,target):
    output = torch.sigmoid(output).data.cpu().numpy()
    # output = output.data.cpu().numpy()
    target = target.data.cpu().numpy()

    threshold = thsrhold
    thre_out = (output >= threshold)
    thre_out = thre_out.astype(float)

    tn_out = thre_out + target
    tn_out = (tn_out == 0)
    TN = np.sum(tn_out)

    Gt_neg = (target == 0)
    Gt_neg = np.sum(Gt_neg)

    return TN / (Gt_neg+1e-15)


def get_precision(output,target):

    threshold = thsrhold
    thre_out = (output >= threshold)
    thre_out = thre_out.astype(float)

    tp_out = thre_out + target
    tp_out = (tp_out==2)
    TP = np.sum(tp_out)

    Pre_pos = (thre_out==1)
    Pre_pos = np.sum(Pre_pos)

    return TP/(Pre_pos+1e-15)


def get_recall(output,target):

    threshold = thsrhold
    thre_out = (output >= threshold)
    thre_out = thre_out.astype(float)


    tp_out = thre_out + target
    tp_out = (tp_out == 2)
    TP = np.sum(tp_out)

    Gt_pos = (target == 1)
    Gt_pos = np.sum(Gt_pos)

    return TP /(Gt_pos+1e-15)

# F1
def get_F1(output,target):
    recall = get_recall(output,target)
    precision = get_precision(output,target)
    F1 = 2*recall*precision/(recall+precision+1e-15)

    return F1



if __name__ == "__main__":

    output = np.array([[[[0.4, 0.7, 0.3],
                       [1, 0.9, 0.8],
                       [0.3, 0.5, 0.1]]],
                       [[[0.5, 1, 1],
                         [1, 0.6, 1],
                         [1, 1, 0.7]]]
                       ])
    output = torch.from_numpy(output)
    target = np.array([[[[0, 0, 0],
                       [1, 1, 1],
                       [0, 0, 0]]],
                       [[[0, 0, 0],
                         [0, 1, 0],
                         [0, 0, 0]]]
                       ])
    target = torch.from_numpy(target)

    tnr = get_specificity(output,target)
    print(tnr)
