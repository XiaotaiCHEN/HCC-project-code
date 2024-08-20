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
from scipy.stats import zscore
import SimpleITK as sitk
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
    # scaler = StandardScaler()
    # zscore_columns = list(set(extract_list).difference(set(One_hot_feature)))
    # Load_feature[zscore_columns] = scaler.fit_transform(Load_feature[zscore_columns])


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


    # all_labels = Load_feature.loc[['Lille', 'Survival']]
    all_labels = data.loc[data['Lille'].isin(sub_patient_list), ['Lille', '2yearsurvival' ]]

    patient_file = ['Lille_' + str(i) for i in data['Lille'][sub_patient_index].values]

    return Load_feature, all_labels, patient_file

class All_info_loader(Dataset):
    def __init__(self, clinc_feature, labels):

        self.clinc_feature = clinc_feature
        self.Mask = labels
        # self.img_file = image_file


    def __len__(self):
       return self.Mask.shape[0]

    def __getitem__(self, index):

        clinc_feature = np.array([self.clinc_feature.iloc[index].values[3::]])
        patient_ID = 'Lille_'+ str(self.clinc_feature.iloc[index].values[0])
        Mask = np.expand_dims(self.Mask['2yearsurvival'].values[index], 0)
        data = {'Patient_ID': patient_ID, 'clinc_feature': clinc_feature, 'label': Mask}
        return data