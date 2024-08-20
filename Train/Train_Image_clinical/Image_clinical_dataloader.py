# -*- coding: utf-8 -*-
# @Time    : 4-4-2024  15:22
# @Author  : Yi Chen
# @File    : Image_clinical_dataloader.py
# @Software: PyCharm
# @Describe: 
# -*- encoding:utf-8 -*-
# -*- coding: utf-8 -*-
# @Time    : 4-4-2024  15:14
# @Author  : Yi Chen
# @File    : Clinical_dataloader.py
# @Software: PyCharm
# @Describe:
# -*- encoding:utf-8 -*-
# -*- coding: utf-8 -*-
# @Time    : 19-10-2023  12:51
# @Author  : Yi Chen
# @File    : clincal_dataloader.py
# @Software: PyCharm
# @Describe:
# -*- encoding:utf-8 -*-
import os
import numpy as np
from skimage import transform
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

def load_all_clinical(file, sub_patient_index):

    data = pd.read_excel(file)
    sub_patient_list = data['Lille'][sub_patient_index]

    extract_list = ['Lille', 'Time', 'Survival', 'cirrhosis_cause', 'aFP_before_trt_200', 'ascitis_before_trt', 'child_stage_split_before_trt', 'BCLC']

    One_hot_feature= ['cirrhosis_cause', 'aFP_before_trt_200', 'ascitis_before_trt', 'child_stage_split_before_trt', 'BCLC']

    Candidate_data = data[extract_list]
    one_hot_data_all = pd.get_dummies(Candidate_data, columns=One_hot_feature, prefix=One_hot_feature)

    '''select patient'''
    Load_feature = one_hot_data_all.loc[one_hot_data_all['Lille'].isin(sub_patient_list), :]
    '''Z-Score for number data'''



    # all_labels = Load_feature.loc[['Lille', 'Survival']]
    all_labels = data.loc[data['Lille'].isin(sub_patient_list), ['Lille', '2yearsurvival']]

    patient_file = ['Lille_' + str(i) for i in data['Lille'][sub_patient_index].values]

    return Load_feature, all_labels, patient_file


def load_all_clinical_noBCLC(file, sub_patient_index):

    data = pd.read_excel(file)
    sub_patient_list = data['Lille'][sub_patient_index]

    extract_list = ['Lille', 'Time', 'Survival', 'cirrhosis_cause', 'aFP_before_trt_200', 'ascitis_before_trt', 'child_stage_split_before_trt']

    One_hot_feature= ['cirrhosis_cause', 'aFP_before_trt_200', 'ascitis_before_trt', 'child_stage_split_before_trt']

    Candidate_data = data[extract_list]
    one_hot_data_all = pd.get_dummies(Candidate_data, columns=One_hot_feature, prefix=One_hot_feature)

    '''select patient'''
    Load_feature = one_hot_data_all.loc[one_hot_data_all['Lille'].isin(sub_patient_list), :]
    '''Z-Score for number data'''
    all_labels = data.loc[data['Lille'].isin(sub_patient_list), ['Lille', '2yearsurvival']]

    patient_file = ['Lille_' + str(i) for i in data['Lille'][sub_patient_index].values]

    return Load_feature, all_labels, patient_file

class All_info_loader(Dataset):
    def __init__(self,  clinc_feature, labels, image_file, Img_root, resize_Shape,
                 transform=None):
        self.image_file = image_file
        self.clinc_feature = clinc_feature
        self.Mask = labels
        self.Img_root = Img_root
        self.resize_Shape = resize_Shape
        self.transform = transform


    def __len__(self):
       return self.Mask.shape[0]

    def __getitem__(self, index):

        data = np.load(os.path.join(self.Img_root, self.image_file[index] + '.npy')).astype(np.float32)
        data = transform.resize(data, self.resize_Shape)

        if self.transform is not None:
            data = self.transform(data)
        data = np.expand_dims(data, 0)
        clinc_feature = np.array([self.clinc_feature.iloc[index].values[3::]])
        patient_ID = self.image_file[index]
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