# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 17:26:49 2020

@author: NCAA's submission authors
"""

import os
path = r'..\neural-computing-and-applications-call\results\output'
os.chdir(path)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dict_alg = {
    "knn": "KNNModel",
    "rf": "RfModel",
    "nb": "NBModel",
    "ext": "ExtModel",
    "mlp": "MLPModel",
    "vfi": "VFIModel",
    "mlp2l": "MLPModel_2layers",
    "mlp3l": "MLPModel_3layers",
}

problems = 12
choices = ['MrgS', 'StdS', 'EntS', 'RndS']
mode = ['SelfTrain' , 'SelfTrainModified']

for alg in dict_alg.keys():
    
    alg_val = dict_alg[alg]
    print(alg_val)
    x_acc       = pd.read_excel('Rankings.xlsx', sheet_name = 'acc_' + alg_val)
    x_prec      = pd.read_excel('Rankings.xlsx', sheet_name = 'prec_' + alg_val)
    x_recall    = pd.read_excel('Rankings.xlsx', sheet_name = 'recall_' + alg_val)
    x_f1score   = pd.read_excel('Rankings.xlsx', sheet_name = 'f1score_' + alg_val)



    df_sns = pd.DataFrame(columns = ['alg','rank','metric','ssl_kind'])
    
    for k in ['acc','prec','recall','f1score']:
        if k == 'acc':
            xx = x_acc
        elif k == 'prec':
            xx = x_prec
        elif k =='recall':
            xx = x_recall
        else:
            xx = x_f1score
        
        for i in choices:
            for j in mode:
                
                scenario = alg_val + '_' +  i + '_' + j   
                print(scenario)
                pos = np.where(xx.model == scenario)[0][0]
        
                df = pd.DataFrame()
                df['alg'] = [xx.model[pos]] * problems
                df['rank'] = xx.iloc[pos,2:].values
                df['metric'] = [k] * problems
                df['ssl_kind'] = [j] * problems
                df['QS'] = [i] * problems
        
                df_sns = pd.concat([df_sns,df])


    for i in ['acc','prec','recall','f1score']:
        sns_plot = sns.catplot(
            data=df_sns.iloc[np.where(df_sns.metric == i)[0],:],
            x='ssl_kind',
            y='rank',
            hue='QS',
            kind='box',
            width = 0.5,
            saturation = 0.9,
            whis = 1,
            height=3,
            aspect=2.5) 
        
        plt.gca().invert_yaxis()
        sns_plot.savefig(alg_val + "_" + i + ".png", dpi = 300)
