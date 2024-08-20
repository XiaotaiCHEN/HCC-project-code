#!/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2/26/2024 12:13 AM
# @Author : Yi Chen
# @File : HCC_Emsemble.py
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import scipy.stats as st
import matplotlib.pyplot as plt


def draw(tprs, aucs):
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    # 绘制平均AUC曲线
    ax.plot(mean_fpr, mean_tpr, color='blue',
            label=f'Mean ROC (AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})',
            lw=2, alpha=0.8)

    # 绘制AUC曲线的变化范围（标准差）
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2,
                    label=r'$\pm$ 1 std. dev.')

    # 绘制对角线（表示随机猜测的性能）
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='red', label='Chance', alpha=0.8)

    ax.set(xlim=[0, 1], ylim=[0, 1.05], xlabel='False Positive Rate', ylabel='True Positive Rate',
           title='Receiver Operating Characteristic')
    ax.legend(loc='lower right')
    plt.show()

radiomics_file = 'C:/Users/p70072776/Desktop/excel document/HCC/clinical_LargeGTV.xlsx'
# deep_learning_file = 'C:/Users/p70072776/Desktop/excel document/HCC/image_clinical_net_v2_ReduceLROnPlateau_Image256_try2_noBCLC_Feb12.xlsx'
deep_learning_file = 'C:\Users\p70072776\Desktop\excel document\HCC\All_result\Image_clinical_radiomics_result/noBCLC_April4th_early10_noscheduler_validation.xlsx'
AUC_all = []
dataframes = {}

fig, ax = plt.subplots()
for i in range(10):
    emsemble_prob_all = []
    label_all = []
    Patient_ID = []
    sheetname = 'fold_' + str(i)
    df_radiomics = pd.read_excel(radiomics_file, sheet_name=sheetname)
    df_deeplearning = pd.read_excel(deep_learning_file, sheet_name=sheetname)
    for j in range(1, df_radiomics.shape[1]):
        assert df_radiomics.iloc[2, j] == np.uint8(df_deeplearning.iloc[2, j]), 'label is not same'
        assert 'Lille_'+ str(np.int16(df_radiomics.iloc[0, j].split('_')[-1])) == df_deeplearning.iloc[0, j], 'patient is not same'
        Patient_ID.append(df_radiomics.iloc[0, j])
        emsemble_prob = np.mean([df_radiomics.iloc[1, j], np.float32(df_deeplearning.iloc[1, j])])
        # emsemble_prob = df_radiomics.iloc[1, j] #only radiomics
        emsemble_prob_all.append(emsemble_prob)
        label_all.append(df_radiomics.iloc[2, j])
    AUC = roc_auc_score(np.array(label_all), np.array(emsemble_prob_all))
    AUC_all.append(AUC)
    #
    df = pd.DataFrame({'Patient_ID': np.array(Patient_ID), 'emseble_prob': np.array(emsemble_prob_all), 'label': np.array(label_all)})
    dataframes['fold_' + str(i)] = df

print(AUC_all)
print(np.mean(AUC_all))
a = np.quantile(AUC_all, 0.75)  # 上四分之一数
b = np.quantile(AUC_all, 0.25)
data = np.array(AUC_all)
print(st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data)))


from sklearn.metrics import roc_curve, auc
tprs_rad = []
aucs_rad = []
tprs_dl = []
aucs_dl = []
tprs_en = []
aucs_en = []
mean_fpr = np.linspace(0, 1, 100)
for i in range(10):
    rad_prob_all =[]
    dl_prob_all = []
    emsemble_prob_all = []
    label_all = []
    Patient_ID = []
    sheetname = 'fold_' + str(i)
    df_radiomics = pd.read_excel(radiomics_file, sheet_name=sheetname)
    df_deeplearning = pd.read_excel(deep_learning_file, sheet_name=sheetname)
    for j in range(1, df_radiomics.shape[1]):
        assert df_radiomics.iloc[2, j] == np.uint8(df_deeplearning.iloc[2, j]), 'label is not same'
        assert 'Lille_'+ str(np.int16(df_radiomics.iloc[0, j].split('_')[-1])) == df_deeplearning.iloc[0, j], 'patient is not same'
        Patient_ID.append(df_radiomics.iloc[0, j])
        emsemble_prob = np.mean([df_radiomics.iloc[1, j], np.float32(df_deeplearning.iloc[1, j])])

        rad_prob_all.append(df_radiomics.iloc[1, j])
        dl_prob_all.append(np.float32(df_deeplearning.iloc[1, j]))
        emsemble_prob_all.append(emsemble_prob)

        label_all.append(df_radiomics.iloc[2, j])

    fpr_rad_fold, tpr_rad_fold, _ = roc_curve(label_all, np.array(rad_prob_all))
    roc_auc_rad_fold = auc(fpr_rad_fold, tpr_rad_fold,)
    tprs_rad.append(np.interp(mean_fpr, fpr_rad_fold, tpr_rad_fold))
    tprs_rad[-1][0] = 0.0
    aucs_rad.append(roc_auc_rad_fold)


    fpr_dl_fold, tpr_dl_fold, _ = roc_curve(label_all, np.array(dl_prob_all))
    roc_auc_dl_fold = auc(fpr_dl_fold, tpr_dl_fold, )
    tprs_dl.append(np.interp(mean_fpr, fpr_dl_fold, tpr_dl_fold))
    tprs_dl[-1][0] = 0.0
    aucs_dl.append(roc_auc_dl_fold)

    fpr_en_fold, tpr_en_fold, _ = roc_curve(label_all, np.array(emsemble_prob_all))
    roc_auc_en_fold = auc(fpr_en_fold, tpr_en_fold, )
    tprs_en.append(np.interp(mean_fpr, fpr_en_fold, tpr_en_fold))
    tprs_en[-1][0] = 0.0
    aucs_en.append(roc_auc_en_fold)

mean_tpr_rad = np.mean(tprs_rad, axis=0)
mean_tpr_rad[-1] = 1.0
std_tpr_rad = np.std(tprs_rad, axis=0)
mean_auc_rad = auc(mean_fpr, mean_tpr_rad)
std_auc_rad = np.std(aucs_rad)

mean_tpr_dl = np.mean(tprs_dl, axis=0)
mean_tpr_dl[-1] = 1.0
std_tpr_dl = np.std(tprs_dl, axis=0)
mean_auc_dl = auc(mean_fpr, mean_tpr_dl)
std_auc_dl = np.std(aucs_dl)


mean_tpr_en = np.mean(tprs_en, axis=0)
mean_tpr_en[-1] = 1.0
std_tpr_en = np.std(tprs_en, axis=0)
mean_auc_en = auc(mean_fpr, mean_tpr_en)
std_auc_en = np.std(aucs_en)


#plot radiomics
ax.plot(mean_fpr, mean_tpr_rad, color='green',
        label=f'radiomics model (AUC = {mean_auc_rad:.2f} $\pm$ {std_auc_rad:.2f})',
        lw=2, alpha=0.8)

# 绘制AUC曲线的变化范围（标准差）
tprs_upper_rad = np.minimum(mean_tpr_rad + std_tpr_rad, 1)
tprs_lower_rad = np.maximum(mean_tpr_rad - std_tpr_rad, 0)
ax.fill_between(mean_fpr, tprs_lower_rad, tprs_upper_rad, color='green', alpha=0.2,)

#plot deep learning
ax.plot(mean_fpr, mean_tpr_dl, color='blue',
        label=f'deep learning model (AUC = {mean_auc_dl:.2f} $\pm$ {std_auc_dl:.2f})',
        lw=2, alpha=0.8)

# 绘制AUC曲线的变化范围（标准差）
tprs_upper_dl = np.minimum(mean_tpr_dl + std_tpr_dl, 1)
tprs_lower_dl = np.maximum(mean_tpr_dl - std_tpr_dl, 0)
ax.fill_between(mean_fpr, tprs_lower_dl, tprs_upper_dl, color='blue', alpha=0.1,)

#ensembel
ax.plot(mean_fpr, mean_tpr_en, color='red',
        label=f'Ensemble model = {mean_auc_en:.2f}  $\pm$ {std_auc_rad:.2f})',
        lw=2, alpha=0.8)

# 绘制AUC曲线的变化范围（标准差）
tprs_upper_en = np.minimum(mean_tpr_en + std_tpr_en, 1)
tprs_lower_en = np.maximum(mean_tpr_en - std_tpr_en, 0)
ax.fill_between(mean_fpr, tprs_lower_en, tprs_upper_en, color='red', alpha=0.2)

ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', label='Chance', alpha=0.8)

ax.set(xlim=[0, 1], ylim=[0, 1.05], xlabel='False Positive Rate', ylabel='True Positive Rate',
       title='Receiver Operating Characteristic')
ax.legend(loc='lower right')
plt.show()