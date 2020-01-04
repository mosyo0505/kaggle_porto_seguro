#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
=================================================================================================

              3. Feature engineering : Feature extraction usnig normal distribution
                                                               
=================================================================================================
'''

'''
=================================================================================================

-----------------------------------------------
 0. Set Environment
-----------------------------------------------
'''

'''
 << Assign default namescope >>
 
'''

Q = set(globals())

'''
-----------------------------------------------------------------------------------------
'''

'''
<< Import global modules >>

'''
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from scipy import stats
import seaborn as sns
from sklearn import linear_model
from sklearn import tree
from sklearn import ensemble

'''
----------------------------------------------------------------------------------------

<< Set working directory >>

'''

os.chdir('/Users/young-oh/Documents/research/machine_learning_practice/kaggle/porto_seguro/version_0.1')

'''
----------------------------------------------------------------------------------------

<< Import custom functions >>

'''

exec(open('code/0.custom_functions.py').read())

'''
----------------------------------------------------------------------------------------

<< Import required objects >>

'''

total_df = pickle.load(open('out/1.preprocessing/total_df.pickle', 'rb'))
selected_feature_conti = pickle.load(open('out/2.continous_feature_selection/selected_feature_conti.pickle', 'rb'))
catego_list = pickle.load(open('out/1.preprocessing/catego_list.pickle', 'rb'))

'''
=================================================================================================

-----------------------------------------------
 1. 
-----------------------------------------------
'''

conti_df = total_df.loc[:, ['target'] + selected_feature_conti]

X_total = np.array(conti_df.loc[:, selected_feature_conti])
y_total = np.array(conti_df.target)

y_total = np.where(y_total == 0, -1, y_total)

n_train = round(X_total.shape[0] * 0.7)
n_test = X_total.shape[0] - n_train

np.random.seed(123)

train_idx = np.random.choice(np.arange(X_total.shape[0]), n_train, replace = False).tolist()
test_idx = list(set(range(X_total.shape[0])) - set(train_idx))

X_train = X_total[train_idx]
X_test = X_total[test_idx]
y_train = y_total[train_idx]
y_test = y_total[test_idx]

'''
------------------------------------------------------------------
'''

mean_pos = 

















































