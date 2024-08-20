# -*- coding: utf-8 -*-
# @Time    : 4-4-2024  15:34
# @Author  : Yi Chen
# @File    : Image_clinical_radiomics_dataloader.py
# @Software: PyCharm
# @Describe: 
# -*- encoding:utf-8 -*-
# -*- coding: utf-8 -*-
# @Time    : 10-2-2024  23:10
# @Author  : Yi Chen
# @File    : Imag_radiomics_dataloader_Feb10.py
# @Software: PyCharm
# @Describe:
# -*- encoding:utf-8 -*-

import os
import numpy as np
from scipy.stats import zscore
import SimpleITK as sitk
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from skimage import transform

def process_data(data, normalized, onehot_list):
    from sklearn.preprocessing import StandardScaler

    one_hot_data_all = pd.get_dummies(data, columns=onehot_list, prefix=onehot_list)
    scaler = StandardScaler()
    one_hot_data_all[normalized] = scaler.fit_transform(one_hot_data_all[normalized])
    return one_hot_data_all



def load_all_information(file, sub_patient_index):

    data = pd.read_excel(file)
    sub_patient_list = data['Lille'][sub_patient_index]

    extract_list = ['Lille', 'Time', 'Survival', 'cirrhosis_cause', 'aFP_before_trt_200', 'ascitis_before_trt', 'child_stage_split_before_trt','BCLC', 'GTV_Shape', 'Healthy_Shape']
    normalized = ['GTV_Shape', 'Healthy_Shape']
    One_hot_feature= ['cirrhosis_cause', 'aFP_before_trt_200', 'ascitis_before_trt', 'child_stage_split_before_trt', 'BCLC']
    Candidate_data = data[extract_list]
    data_preprocess = process_data(Candidate_data, normalized, One_hot_feature)
    Load_feature = data_preprocess.loc[data_preprocess['Lille'].isin(sub_patient_list), :]
    all_labels = data.loc[data['Lille'].isin(sub_patient_list), ['Lille', '2yearsurvival']]
    patient_file = ['Lille_' + str(i) for i in data['Lille'][sub_patient_index].values]

    return Load_feature, all_labels, patient_file


def load_all_information_noBCLC(file, sub_patient_index):

    data = pd.read_excel(file)
    sub_patient_list = data['Lille'][sub_patient_index]

    extract_list = ['Lille', 'Time', 'Survival', 'cirrhosis_cause', 'aFP_before_trt_200', 'ascitis_before_trt', 'child_stage_split_before_trt', 'GTV_Shape', 'Healthy_Shape']
    normalized = ['GTV_Shape', 'Healthy_Shape']
    One_hot_feature= ['cirrhosis_cause', 'aFP_before_trt_200', 'ascitis_before_trt', 'child_stage_split_before_trt'] #from blood, patient chart,

    Candidate_data = data[extract_list]
    one_hot_data_all = pd.get_dummies(Candidate_data, columns=One_hot_feature, prefix=One_hot_feature)

    '''select patient'''
    Load_feature = one_hot_data_all.loc[one_hot_data_all['Lille'].isin(sub_patient_list), :]
    '''Z-Score for number data'''
    Load_feature[normalized] = Load_feature[normalized].apply(zscore)

    all_labels = data.loc[data['Lille'].isin(sub_patient_list), ['Lille', '2yearsurvival']]

    patient_file = ['Lille_' + str(i) for i in data['Lille'][sub_patient_index].values]

    return Load_feature, all_labels, patient_file

class All_info_loader(Dataset):
    def __init__(self,
                 clinc_feature, labels, image_file, Img_root, resize_Shape,
                 transform=None):

        self.clinc_feature = clinc_feature
        self.Img_root = Img_root
        self.Mask = labels
        self.img_file = image_file
        self.transform = transform
        self.resize_Shape = resize_Shape

    def __len__(self):
       return len(self.img_file)

    def __getitem__(self, index):

        data = np.load(os.path.join(self.Img_root, self.img_file[index] + '.npy')).astype(np.float32)
        # data = transform.resize(data, self.resize_Shape)
        # data = np.expand_dims(data, 0)
        if self.transform is not None:
            data = self.transform(data)

        clinc_feature = np.array([self.clinc_feature.iloc[index].values[3::]])
        patient_ID = self.img_file[index]
        Mask = np.expand_dims(self.Mask['2yearsurvival'].values[index], 0)
        data = {'Patient_ID': patient_ID, 'Image': data, 'clinc_feature': clinc_feature, 'label': Mask}
        return data


class All_info_loader_noBCLC(Dataset):
    def __init__(self,
                 clinc_feature, labels, image_file, Img_root, resize_Shape,
                 transform=None):

        self.clinc_feature = clinc_feature
        self.Img_root = Img_root
        self.Mask = labels
        self.img_file = image_file
        self.transform = transform
        self.resize_Shape = resize_Shape

    def __len__(self):
       return len(self.img_file)

    def __getitem__(self, index):

        data = np.load(os.path.join(self.Img_root, self.img_file[index] + '.npy')).astype(np.float32)
        data = transform.resize(data, self.resize_Shape)

        if self.transform is not None:
            data = self.transform(data)
        data = np.expand_dims(data, 0)
        clinc_feature = np.array([self.clinc_feature.iloc[index].values[3::]])
        patient_ID = self.img_file[index]
        Mask = np.expand_dims(self.Mask['2yearsurvival'].values[index], 0)
        data = {'Patient_ID': patient_ID, 'Image': data, 'clinc_feature': clinc_feature, 'label': Mask}
        return data

