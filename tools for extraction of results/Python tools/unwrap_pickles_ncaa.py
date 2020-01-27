# -*- coding: utf-8 -*-

import os
import pickle
import pandas as pd
import numpy as np
path = '../neural-computing-and-applications-call/initial pickles for check'
os.chdir(path)

for folder in ['chains', 'voice', 'anad']:
    print('folder: ', folder)
    os.chdir(path + '/' + folder)
    files = os.listdir()
    print()
    print(files)
    print('########################')
    for file in files:

        with open(file, 'rb') as f:
            d = pickle.load(f)

        Ks_str = [ '2', '5', '10', '25' ]
        selection_functions_str = ['MarginSamplingSelection','EntropySelection', 'MinStdSelection', 'RandomSelection', 'NoActiveLearningSelection']
        selection_functions_ssl_str = ['SSLselection','SSLselectionModified', 'NOSSLselection']
        models_str = [  'KNNModel' , 'RfModel' , 'NBModel' ,  'ExtModel' , 'VFIModel', 'MLPModel' , 'MLPModel_2layers' , 'MLPModel_3layers']
        repeats = 3

        l = []
        counter = 0
        for m in models_str:
                for al in selection_functions_str:
                    for ssl in selection_functions_ssl_str: 
                        for k in Ks_str:
                            for fold in range(0, repeats):
                                counter += 1

                                x = pd.DataFrame()
                                amount = int(np.array(d[m][al][ssl][k][0][0]).reshape(1,-1).shape[1])
                                x['step']    = [k] * amount    
                                x['al']      = [al] * amount
                                x['learner'] = [m] * amount
                                x['fold']    = [fold] * amount
                                x['ssl']    = [ssl] * amount
                                x['acc']     = np.array(d[m][al][ssl][k][fold][0]).tolist() # edo bainoun ana fold ola ta metrics
                                x['prec'] = np.array(d[m][al][ssl][k][fold][1]).tolist()
                                x['recall'] = np.array(d[m][al][ssl][k][fold][2]).tolist()
                                x['f1score'] = np.array(d[m][al][ssl][k][fold][3]).tolist()

                                l.append(x)

        y = pd.DataFrame()
        for i in range(0, len(selection_functions_ssl_str) * len(selection_functions_str) * len(models_str) * len(Ks_str) * repeats):
            y = pd.concat([y,l[i]])

        print(y.shape)
        y.to_csv(file + '.csv')
    os.chdir(path)