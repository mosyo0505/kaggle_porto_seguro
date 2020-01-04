#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
=================================================================================================

                       2. Feature engineering : Continuous feature selection
                                                               
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

total_df = pickle.load(open('out/1.preprocessing/total_df.pickle', 'rb'))


'''
===================================================================================================

----------------------------------------------------------------
 1. Continous feature selection using t-test
----------------------------------------------------------------
(1) feature들을 크게 continuous feature와 binary feature로 나눔
(2) continuous feature에 대하여 zero와 one인 경우에 대하여 t-test를 실시
    유의한 continuous feature를 찾고자 함
----------------------------------------------------------------
'''

'''
 << Continuous feature, binary feature 나누기 >>
'''

ratio_one = np.array(total_df.target).tolist().count(1) / total_df.shape[0]

one_df = total_df[total_df.target == 1]
zero_df = total_df[total_df.target == 0]

bin_feature, catego_feature, conti_feature = [], [], []
feature_list = np.array(total_df.columns).tolist()
feature_list.remove('target')

for feature in feature_list:
    
    if 'bin' in feature:
        
        bin_feature.append(feature)
        
    if 'cat' in feature:
        
        catego_feature.append(feature)
        
    if ('bin' not in feature) and ('cat' not in feature):
        
        conti_feature.append(feature)

'''
  << T test 실시 >>
'''


selected_feature_conti = []

for feat in conti_feature:
    
    one_ar = np.array(one_df[feat])
    zero_ar = np.array(zero_df[feat])
    
    mean_one = np.mean(one_ar)
    mean_zero = np.mean(zero_ar)
    
    p_value = t_test_fn(one_ar, zero_ar)
    
    #print(feat, ':', p_value,',', mean_zero,',',mean_one)

    if p_value < 0.05:
        
        print(feat, ':', p_value,',', mean_zero,',',mean_one)
        selected_feature_conti.append(feat)

pickle.dump(selected_feature_conti, open('out/2.continous_feature_selection/selected_feature_conti.pickle',
                                         'wb'))

































