#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
=================================================================================================

                                       1. Preprocessing
        
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
from matplotlib import colors
import pickle
#plt.style.use('ggplot')
from scipy import stats
import seaborn as sns
from sklearn import linear_model
from sklearn import tree
from sklearn import ensemble
from sklearn import decomposition
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics
import math
from sklearn import preprocessing

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

raw_df = pd.read_csv('data/train.csv')


'''
======================================================================================

-----------------------------------------------------------------------------
 1. 데이터 전처리
-----------------------------------------------------------------------------
(1) Categorical feature를 one hot encoding을 실시하여 새로운 binary feature 생성
    생성된 data frame : total_df
-----------------------------------------------------------------------------
'''

column_list = raw_df.columns.tolist()
feature_list = column_list[2:]
catego_list = []

for feature in feature_list:
    
    if 'cat' in feature:
        
        catego_list.append(feature)

one_hot_df = pd.get_dummies(raw_df, columns = catego_list)
one_hot_df = one_hot_df.iloc[:, 1:]

total_df = one_hot_df

pickle.dump(total_df, open('out/1.preprocessing/total_df.pickle', 'wb'))
pickle.dump(catego_list, open('out/1.preprocessing/catego_list.pickle','wb'))

# git

