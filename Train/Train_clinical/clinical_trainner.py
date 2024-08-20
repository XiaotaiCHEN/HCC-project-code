# -*- coding: utf-8 -*-
# @Time    : 4-4-2024  15:15
# @Author  : Yi Chen
# @File    : clinical_trainner.py
# @Software: PyCharm
# @Describe: 
# -*- encoding:utf-8 -*-
# -*- coding: utf-8 -*-
# @Time    : 19-10-2023  12:53
# @Author  : Yi Chen
# @File    : Clincal_main_code.py
# @Software: PyCharm
# @Describe:
# -*- encoding:utf-8 -*-
import os

import monai.networks.nets
from monai import transforms
from monai.transforms import (
    Compose, Resize, RandRotate,RandFlip,
    ToTensor, RandGaussianNoise,OneOf
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
from barbar import Bar
from tensorboardX import SummaryWriter

from Train_all_in_April.Early_stop import EarlyStopping
from Train_all_in_April.Train_clinical.Clinical_dataloader import All_info_loader, load_all_clinical
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt


Patient_list = [(np.array([0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14,
                           15, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 31,
                           32, 33, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45,
                           46, 47, 48, 51, 52, 54, 55, 56, 57, 58, 59, 60, 61,
                           62, 63, 64, 65, 66, 68, 69, 71, 72, 73, 74, 75, 76,
                           77, 78, 79, 80, 83, 84, 85, 86, 87, 88, 89, 91, 92,
                           93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105,
                           106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
                           119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131,
                           132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144,
                           145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157,
                           158, 160, 162, 163, 164, 165, 166, 167, 168, 169, 170, 172, 173,
                           174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 185]),
                 np.array([7, 8, 16, 17, 26, 30, 40, 49, 50, 53, 67, 70, 81,
                           82, 90, 159, 161, 171, 184])),
                (np.array([2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                           16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29,
                           30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
                           43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                           56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
                           69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                           82, 83, 85, 86, 87, 89, 90, 91, 92, 93, 94, 95, 96,
                           97, 98, 99, 100, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                           111, 112, 113, 114, 115, 116, 119, 120, 121, 122, 123, 124, 125,
                           126, 127, 129, 130, 131, 133, 134, 135, 136, 137, 139, 140, 141,
                           142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 155,
                           157, 158, 159, 160, 161, 163, 164, 166, 167, 168, 169, 170, 171,
                           172, 173, 174, 175, 176, 177, 179, 180, 181, 182, 184]),
                 np.array([0, 1, 6, 23, 84, 88, 101, 117, 118, 128, 132, 138, 154,
                           156, 162, 165, 178, 183, 185])),
                (np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                           13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 26, 27,
                           28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41,
                           43, 44, 45, 46, 47, 49, 50, 51, 52, 53, 54, 55, 56,
                           57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                           70, 71, 72, 73, 74, 75, 77, 78, 79, 80, 81, 82, 83,
                           84, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 97, 98,
                           99, 101, 102, 103, 104, 106, 107, 108, 111, 112, 113, 114, 115,
                           116, 117, 118, 119, 120, 122, 123, 124, 125, 126, 127, 128, 129,
                           130, 131, 132, 133, 134, 135, 136, 138, 139, 140, 142, 143, 144,
                           145, 146, 147, 148, 149, 150, 152, 153, 154, 155, 156, 157, 158,
                           159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171,
                           173, 175, 176, 177, 178, 179, 181, 182, 183, 184, 185]),
                 np.array([24, 25, 37, 42, 48, 76, 85, 96, 100, 105, 109, 110, 121,
                           137, 141, 151, 172, 174, 180])),
                (np.array([0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14,
                           15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                           28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
                           42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                           55, 56, 59, 60, 63, 64, 65, 66, 67, 68, 69, 70, 71,
                           72, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86,
                           87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                           100, 101, 102, 103, 105, 106, 107, 108, 109, 110, 111, 112, 113,
                           114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 125, 127, 128,
                           129, 130, 131, 132, 133, 135, 136, 137, 138, 139, 140, 141, 142,
                           144, 145, 146, 147, 148, 150, 151, 152, 153, 154, 155, 156, 158,
                           159, 161, 162, 163, 165, 166, 168, 169, 170, 171, 172, 173, 174,
                           175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185]),
                 np.array([4, 11, 29, 57, 58, 61, 62, 73, 74, 104, 120, 126, 134,
                           143, 149, 157, 160, 164, 167])),
                (np.array([0, 1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14,
                           15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                           28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42,
                           44, 45, 46, 48, 49, 50, 52, 53, 54, 55, 57, 58, 59,
                           60, 61, 62, 63, 64, 67, 68, 69, 70, 72, 73, 74, 76,
                           77, 78, 79, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                           91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,
                           104, 105, 106, 107, 108, 109, 110, 112, 113, 114, 115, 116, 117,
                           118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 130, 131,
                           132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144,
                           146, 147, 148, 149, 150, 151, 153, 154, 155, 156, 157, 159, 160,
                           161, 162, 163, 164, 165, 167, 168, 169, 170, 171, 172, 173, 174,
                           175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185]),
                 np.array([3, 10, 31, 39, 43, 47, 51, 56, 65, 66, 71, 75, 80,
                           111, 129, 145, 152, 158, 166])),
                (np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                           13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26,
                           27, 29, 30, 31, 33, 34, 35, 36, 37, 39, 40, 41, 42,
                           43, 44, 45, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
                           57, 58, 59, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                           71, 72, 73, 74, 75, 76, 78, 80, 81, 82, 83, 84, 85,
                           86, 87, 88, 89, 90, 92, 93, 94, 95, 96, 97, 98, 99,
                           100, 101, 102, 103, 104, 105, 106, 108, 109, 110, 111, 112, 113,
                           114, 115, 117, 118, 120, 121, 122, 123, 124, 126, 127, 128, 129,
                           130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 143,
                           144, 145, 146, 147, 148, 149, 150, 151, 152, 154, 155, 156, 157,
                           158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 171,
                           172, 173, 174, 175, 178, 179, 180, 182, 183, 184, 185]),
                 np.array([19, 28, 32, 38, 46, 60, 77, 79, 91, 107, 116, 119, 125,
                           142, 153, 170, 176, 177, 181])),
                (np.array([0, 1, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14, 16,
                           17, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31,
                           32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                           45, 46, 47, 48, 49, 50, 51, 53, 56, 57, 58, 59, 60,
                           61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73,
                           74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86,
                           88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 99, 100, 101,
                           102, 103, 104, 105, 107, 108, 109, 110, 111, 112, 113, 114, 115,
                           116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 128, 129,
                           131, 132, 134, 135, 136, 137, 138, 140, 141, 142, 143, 144, 145,
                           146, 147, 148, 149, 151, 152, 153, 154, 155, 156, 157, 158, 159,
                           160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172,
                           173, 174, 175, 176, 177, 178, 179, 180, 181, 183, 184, 185]),
                 np.array([2, 5, 12, 15, 18, 27, 52, 54, 55, 87, 98, 106, 127,
                           130, 133, 139, 150, 182])),
                (np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                           13, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27,
                           28, 29, 30, 31, 32, 33, 35, 36, 37, 38, 39, 40, 41,
                           42, 43, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
                           57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 70,
                           71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
                           85, 86, 87, 88, 90, 91, 92, 93, 94, 95, 96, 97, 98,
                           99, 100, 101, 102, 103, 104, 105, 106, 107, 109, 110, 111, 112,
                           114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126,
                           127, 128, 129, 130, 131, 132, 133, 134, 137, 138, 139, 140, 141,
                           142, 143, 145, 149, 150, 151, 152, 153, 154, 156, 157, 158, 159,
                           160, 161, 162, 163, 164, 165, 166, 167, 168, 170, 171, 172, 173,
                           174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185]),
                 np.array([14, 21, 34, 44, 45, 69, 83, 89, 108, 113, 135, 136, 144,
                           146, 147, 148, 155, 169])),
                (np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 14,
                           15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28,
                           29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43,
                           44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
                           57, 58, 60, 61, 62, 64, 65, 66, 67, 68, 69, 70, 71,
                           73, 74, 75, 76, 77, 79, 80, 81, 82, 83, 84, 85, 87,
                           88, 89, 90, 91, 94, 95, 96, 97, 98, 100, 101, 102, 104,
                           105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117,
                           118, 119, 120, 121, 122, 124, 125, 126, 127, 128, 129, 130, 132,
                           133, 134, 135, 136, 137, 138, 139, 141, 142, 143, 144, 145, 146,
                           147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
                           160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172,
                           173, 174, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185]),
                 np.array([9, 13, 20, 35, 41, 59, 63, 72, 78, 86, 92, 93, 99,
                           103, 123, 131, 140, 175])),
                (np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                           13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26,
                           27, 28, 29, 30, 31, 32, 34, 35, 37, 38, 39, 40, 41,
                           42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                           55, 56, 57, 58, 59, 60, 61, 62, 63, 65, 66, 67, 69,
                           70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,
                           83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 96, 98,
                           99, 100, 101, 103, 104, 105, 106, 107, 108, 109, 110, 111, 113,
                           116, 117, 118, 119, 120, 121, 123, 125, 126, 127, 128, 129, 130,
                           131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
                           144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156,
                           157, 158, 159, 160, 161, 162, 164, 165, 166, 167, 169, 170, 171,
                           172, 174, 175, 176, 177, 178, 180, 181, 182, 183, 184, 185]),
                 np.array([22, 33, 36, 64, 68, 94, 95, 97, 102, 112, 114, 115, 122,
                           124, 163, 168, 173, 179]))]



def caculate(pred, labels, cut_off):
    ''''all the result you should get from the data
    Pred: you got from model
    labels: the real result you want to prediacte
    cut_off: how to binary value
    -------------------------------------------------------
    accuracy:
    sensitivity:
    specificity
    '''
    pred_softmax = torch.sigmoid(pred)
    binary_predictions = torch.where(pred_softmax >= cut_off, 1, 0)
    assert pred.shape == labels.shape, 'check data shape'

    TP = torch.sum((binary_predictions == 1) & (labels == 1)).item()
    TN = torch.sum((binary_predictions == 0) & (labels == 0)).item()
    FP = torch.sum((binary_predictions == 1) & (labels == 0)).item()
    FN = torch.sum((binary_predictions == 0) & (labels == 1)).item()

    accuracy = (TP + TN) / (TP + FP + TN + FN)
    if (TP==0) & (FN==0):
        sensitivity = None
    else:
        sensitivity = TP / (TP + FN)

    # Calculate Specificity
    if (TN == 0) & (FP == 0):
        specificity = None
    else:
        specificity = TN / (TN + FP)

    return accuracy, sensitivity, specificity


def calculate_youden_index(confusion_matrix):
    ''''Youden index can help to find the best threshold for cut-off value'''
    # Extract values from confusion matrix
    TP, FP, FN, TN = confusion_matrix.ravel()

    # Calculate Sensitivity and Specificity
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    # Calculate Youden Index
    youden_index = sensitivity + specificity - 1

    return youden_index

def load_auc(loader, model, model_Save_address, mode):
    ''''this work for how to draw the auc curve
    loader: our dataloader;
    model: which model we want to draw;
    model_Save_address:where you save the model checkpoin;
    mode: is training or validation'''
    g = torch.Generator()
    g.manual_seed(0)
    torch.manual_seed(1234)

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = "5, 6, 7"

    device_ids = [0, 1, 2]


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    chekpoint_path = os.path.join(model_Save_address, 'checkpoints', 'checkpoint.pth')
    checkpoint = torch.load(chekpoint_path)
    model.load_state_dict(checkpoint)

    model.to(device)
    model = nn.DataParallel(model, device_ids=device_ids)


    model.eval()

    label_All = []
    pred_All =[]
    prob_all = []
    with torch.no_grad():
        for val, val_data in enumerate(loader, 1):

            Img = val_data['Image'].type('torch.cuda.FloatTensor').to(device)
            clin_feature = val_data['clinc_feature'].type('torch.cuda.FloatTensor').to(device)
            label = val_data['label'].to(device)

            output = model(clin_feature)

            binary = nn.functional.sigmoid(output)
            binary_predictions = torch.where(binary >= cut_off, 1, 0)

            prob_all.append(binary.detach().cpu().numpy())
            pred_All.append(binary_predictions.detach().cpu().numpy())
            label_All.append(label.detach().cpu().numpy())

    label_All = np.array(label_All)
    prob_all = np.array(prob_all)

    fpr, tpr, thresholds = roc_curve(label_All, prob_all)

    youden_index = tpr - fpr
    best_threshold = thresholds[np.argmax(youden_index)]

    y_pred_binary = (prob_all > best_threshold).astype(int)
    fpr, tpr, _ = roc_curve(label_All, y_pred_binary)
    roc_auc = auc(fpr, tpr)

    ''''plot AUC curve'''
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()
    plt.savefig(os.path.join(model_Save_address, mode+ '.png'))

    return roc_auc


def main(root, batch_size, model_name, Img_path, cut_off, save_path, resize_Shape):


    g = torch.Generator()
    g.manual_seed(0)
    torch.manual_seed(1234)

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    device_ids = [0, 1]

    pin_memory = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(10):
        os.makedirs(
            os.path.join(save_path, model_name, 'Fold_' + str(i)),
            exist_ok=True)
        model_Save_address = os.path.join(save_path, model_name, 'Fold_' + str(i))
        Writer_save = os.path.join(save_path, model_name, 'Writer', 'Fold_' + str(i))
        writer = SummaryWriter(Writer_save)

        train_patient_list = Patient_list[i][0]
        val_patient_list = Patient_list[i][1]
        Load_feature_t, all_labels_t, patient_file_t = load_all_clinical(root, train_patient_list)
        Load_feature_v, all_labels_v, patient_file_v = load_all_clinical(root, val_patient_list)

        # ScaleIntensity(), EnsureChannelFirst(),
        # train_trans = Compose([Resize((100, 128, 128)), RandRotate90()])
        # train_transforms = Compose([
        #     OneOf([RandRotate(range_x=[-30, 30], range_y=[-30, 30], range_z=0.0, prob=0.5),
        #            RandFlip(spatial_axis=(1, 2), prob=0.5),
        #            ]),
        #     RandGaussianNoise(0.3),
        # ])

        train_ds = All_info_loader(Load_feature_t, all_labels_t)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=pin_memory)

        val_ds = All_info_loader(Load_feature_v, all_labels_v)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin_memory)

        # model = monai.networks.nets.resnet50()
        # model.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(1, 1, 1), padding=(3, 3, 3), bias=False)
        # model.fc = nn.Linear(in_features=2048, out_features=1, bias=True)


        # model = monai.networks.nets.EfficientNetBN('efficientnet-b0',spatial_dims =3, in_channels=1, num_classes=1)
        model = nn.Sequential(nn.Linear(23, 1))

        # for m in model.modules():
        #     if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)):
        #         torch.nn.init.xavier_uniform_(m.weight)
        #
        # for m in model.modules():
        #     if isinstance(m, torch.nn.Conv2d):
        #         torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('relu'))

        model.to(device)
        model = nn.DataParallel(model, device_ids=device_ids)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.01)  # 0.000928061399619526
        criterion = nn.BCEWithLogitsLoss()
        early_stopping = EarlyStopping(patience=20, verbose=True, delta=0, path=model_Save_address)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                                                  verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0,
                                                  min_lr=0, eps=1e-08)

        epoch = 0
        start = True
        try:
            while start:

                train_loss = 0
                train_acc = 0

                model.train()
                pred_prob = []
                train_labels = []
                for i, data in enumerate(Bar(train_loader), 1):

                    clin_feature = data['clinc_feature'].type('torch.cuda.FloatTensor').to(device)
                    label = data['label'].type('torch.cuda.FloatTensor').to(device)
                    patient_ID = data['Patient_ID']

                    optimizer.zero_grad()

                    output = model(clin_feature)
                    loss = criterion(output, label.unsqueeze(1))
                    acc, sen, spec = caculate(output.squeeze(1), label, cut_off=cut_off)
                    for batch in range(output.shape[0]):
                        pred_prob.append(output[batch, 0, 0].detach().cpu().numpy())
                        train_labels.append(label[batch].detach().cpu().numpy())
                    train_acc += acc
                    train_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                fpr, tpr, _ = roc_curve(np.array(train_labels), np.array(pred_prob))
                roc_auc = auc(fpr, tpr)
                train_loss /= len(train_loader)
                train_acc /= len(train_loader)
                print(
                    f'epoch:{epoch} train_loss:{train_loss:.4f}, train_acc:{train_acc:.4f}, train_auc:{roc_auc:.4f}')

                writer.add_scalar('train_accuary', train_acc, epoch)
                writer.add_scalar('train_loss', train_loss, epoch)


                val_prob = []
                val_labels = []
                val_loss = 0
                val_acc = 0
                model.eval()

                for j, data_v in enumerate(Bar(val_loader), 1):
                    with torch.no_grad():

                        clin_feature_v = data_v['clinc_feature'].type('torch.cuda.FloatTensor').to(device)
                        label_v = data_v['label'].type('torch.cuda.FloatTensor').to(device)
                        patient_ID = data_v['Patient_ID']

                        output_v = model(clin_feature_v)
                        loss_v = criterion(output_v, label_v.unsqueeze(1))
                        acc_v, sen_v, spec_v = caculate(output_v.squeeze(1), label_v, cut_off=cut_off)
                        val_loss += loss_v.item()
                        val_acc += acc_v
                        for batch in range(output_v.shape[0]):
                            val_prob.append(output_v[batch, 0, 0].detach().cpu().numpy())
                            val_labels.append(label_v[batch].detach().cpu().numpy())

                val_loss /= len(val_loader)
                val_acc /= len(val_loader)
                fpr_val, tpr_val, _ = roc_curve(np.array(val_labels), np.array(val_prob))
                roc_auc_val = auc(fpr_val, tpr_val)
                print(f'epoch:{epoch} val_loss:{val_loss:.4f}, val_acc:{val_acc:.4f}, val_auc:{roc_auc_val:.4f}')
                writer.add_scalar('val_accuary', val_acc, epoch)
                writer.add_scalar('val_loss', val_loss, epoch)
                scheduler.step(val_loss)


                early_stopping(val_loss, model)

                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                if epoch>=1000:
                    start =False
                    print('out of epoch')
                else:
                    epoch += 1

        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    root = 'D:/Yi/David/document/Finally_document.xlsx'
    batch_size = 4
    model_name = 'Linear_early20_March25'
    Img_path = 'D:/Yi/HCC_v2/Cropped_Image'
    cut_off = 0.5
    model_save_path = 'D:\Yi\HCC_v2\Train_all_in_April\Train_clinical/model_save_Address'
    resize_Shape = [100, 256, 256]
    main(root, batch_size, model_name, Img_path, cut_off, model_save_path, resize_Shape)