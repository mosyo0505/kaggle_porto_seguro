#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
=================================================================================================

                                          Prototype
        
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

os.chdir('/Users/young-oh/Documents/research/machine_learning_practice/kaggle/porto_seguro_2')

'''
----------------------------------------------------------------------------------------

<< Import custom functions >>

'''

exec(open('PSC/custom_functions.py').read())

'''
======================================================================================

-----------------------------------------------------------------------------
 1. 데이터 전처리
-----------------------------------------------------------------------------
(1) Categorical feature를 one hot encoding을 실시하여 새로운 binary feature 생성
    생성된 data frame : total_df
-----------------------------------------------------------------------------
'''

raw_df = pd.read_csv('data/train.csv')


column_list = raw_df.columns.tolist()
feature_list = column_list[2:]
catego_list = []

for feature in feature_list:
    
    if 'cat' in feature:
        
        catego_list.append(feature)

one_hot_df = pd.get_dummies(raw_df, columns = catego_list)
one_hot_df = one_hot_df.iloc[:, 1:]

total_df = one_hot_df


'''
===================================================================================================

----------------------------------------------------------------
 2. Exploratoty Data Analysis - part 1
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

'''
===================================================================================================

--------------------------------------------------------------------------
 3. Exploratory Data Analysis -part 2
--------------------------------------------------------------------------
(1) Continuous feature 에 대하여 단변량 분석
(2) selected_feature_conti에 대하여 zero, one 별로 box plot 및 density plot
--------------------------------------------------------------------------
'''

'''
 << Selected_feature_conti 들 사이에 correlation 확인해보기 >>
'''

pd.set_option('display.max_columns', None)

total_df[selected_feature_conti].corr(method = 'pearson')

pd.set_option('display.max_columns', 4)

# 선택된 feature들에 대하여 scatter plot 그려보기

cor_names1 = ['ps_reg_01', 'ps_reg_01', 'ps_reg_02', 'ps_car_12', 'ps_car_13']
cor_names2 = ['ps_reg_02', 'ps_reg_03', 'ps_reg_03', 'ps_car_13', 'ps_car_15']

cor_names = list(zip(cor_names1, cor_names2))

for i in range(len(cor_names)):
    
    x_axis = np.array(total_df[cor_names[i][0]])
    y_axis = np.array(total_df[cor_names[i][1]])
    
    fig = plt.figure(figsize = (10, 7))
    
    plt.scatter(x_axis, y_axis)
    
    plt.xlabel(cor_names[i][0])
    plt.ylabel(cor_names[i][1])
    
#    y = x line 그리기
    
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = x_vals
    plt.plot(x_vals, y_vals, color = sns.color_palette()[1])
    
    plt.savefig('OUT/PLOT/cor_scatter/cor_plot_{}'.format(i+1))

'''
 << Selected_feature_conti에 대하여 box plot 및 density plot 그리기 >>
'''    

for i in range(len(selected_feature_conti)):
    
    feat = selected_feature_conti[i]
    one_feat = np.array(one_df[feat])
    zero_feat = np.array(zero_df[feat])
    total_feat = np.array(total_df[feat])
    
    fig = plt.figure(figsize = (10, 7))
    fig.suptitle(feat)
    
    ax1 = plt.subplot(2, 1, 1)
    concat_data = [zero_feat, one_feat]
    
    ax1.boxplot(concat_data, labels = ['0', '1'])
    
    ax2 = plt.subplot(2, 1, 2)
    sns.distplot(zero_feat, color = sns.color_palette()[0], bins = 100)
    sns.distplot(one_feat, color = sns.color_palette()[1], bins = 100)
    
    plt.savefig('OUT/PLOT/t_test_result/{}.png'.format(feat))
    
'''
===================================================================================================

----------------------------------------------
 4. Feature engineering -part 1
----------------------------------------------
(1) Data preperation
(2) Gradient Boosting fitting
----------------------------------------------


 << Data preperation >>

'''

# Trian set, Test set 나누기

X_total = total_df.loc[:, selected_feature_conti]
y_total = total_df.target
y_total = y_total.replace(0, -1)

np.random.seed(123)

train_idx = np.random.choice(range(X_total.shape[0]), round(X_total.shape[0] * 0.7), replace = False)
test_idx = np.array(list(set(range(X_total.shape[0])) - set(train_idx.tolist())))


X_total = X_total.reset_index(drop = True)
y_total = y_total.reset_index(drop = True)


X_train = X_total.iloc[train_idx, :]
y_train = y_total.iloc[train_idx]

X_test = X_total.iloc[test_idx, ]
y_test = y_total.iloc[test_idx]

X_train = X_train.reset_index(drop = True)
y_train = y_train.reset_index(drop = True)

X_test = X_test.reset_index(drop = True)
y_test = y_test.reset_index(drop = True)

print('Positive ratio in train set : ', y_train.value_counts()[1] / y_train.value_counts().sum())
print('Negative ratio in test set : ', y_test.value_counts()[1] / y_test.value_counts().sum())


'''
--------------------------------------------------------------------------------------

 << Gradient boosting fitting >>
 
 
'''

# Model 선언

clf_gdb_1 = ensemble.GradientBoostingClassifier(n_estimators = 20000,
                                                max_depth = 1,
                                                random_state = 0,
                                                verbose = 3)

clf_gdb_2 = ensemble.GradientBoostingClassifier(n_estimators = 20000,
                                                max_depth = 2,
                                                random_state = 0,
                                                verbose = 3)

clf_gdb_3 = ensemble.GradientBoostingClassifier(n_estimators = 20000,
                                                max_depth = 3,
                                                random_state = 0,
                                                verbose = 3)

clf_gdb_4 = ensemble.GradientBoostingClassifier(n_estimators = 20000,
                                                max_depth = 4,
                                                random_state = 0,
                                                verbose = 3)

clf_gdb_5 = ensemble.GradientBoostingClassifier(n_estimators = 20000,
                                                max_depth = 5,
                                                random_state = 0,
                                                verbose = 3)

clf_gdb_6 = ensemble.GradientBoostingClassifier(n_estimators = 20000,
                                                max_depth = 6,
                                                random_state = 0,
                                                verbose = 3)


clf_gdb_7 = ensemble.GradientBoostingClassifier(n_estimators = 20000,
                                                max_depth = 7,
                                                random_state = 0,
                                                verbose = 3)







# Model training

clf_gdb_1.fit(X_train, y_train)
clf_gdb_2.fit(X_train, y_train)
clf_gdb_3.fit(X_train, y_train)
clf_gdb_4.fit(X_train, y_train)
clf_gdb_5.fit(X_train, y_train)
pickle.dump(clf_gdb_5, open('OUT/Pfile/clf_gdb_5.sav', 'wb'))
clf_gdb_6.fit(X_train, y_train)
pickle.dump(clf_gdb_6, open('OUT/Pfile/clf_gdb_6.sav', 'wb'))
clf_gdb_7.fit(X_train, y_train)



pickle.dump(clf_gdb_1, open('OUT/Pfile/clf_gdb_1.sav', 'wb'))
pickle.dump(clf_gdb_2, open('OUT/Pfile/clf_gdb_2.sav', 'wb'))
pickle.dump(clf_gdb_3, open('OUT/Pfile/clf_gdb_3.sav', 'wb'))
pickle.dump(clf_gdb_4, open('OUT/Pfile/clf_gdb_4.sav', 'wb'))
pickle.dump(clf_gdb_7, open('OUT/Pfile/clf_gdb_7.sav', 'wb'))


clf_gdb_1 = pickle.load(open('OUT/Pfile/clf_gdb_1.sav', 'rb'))
clf_gdb_2 = pickle.load(open('OUT/Pfile/clf_gdb_2.sav', 'rb'))
clf_gdb_3 = pickle.load(open('OUT/Pfile/clf_gdb_3.sav', 'rb'))
clf_gdb_4 = pickle.load(open('OUT/Pfile/clf_gdb_4.sav', 'rb'))
clf_gdb_7 = pickle.load(open('OUT/Pfile/clf_gdb_7.sav', 'rb'))

# 추정확률 조정하기


ratio_pos = y_train.value_counts()[1] / y_train.value_counts().sum()
ratio_neg = 1 - ratio_pos


dec_dep_1_train = clf_gdb_1.decision_function(X_train)
dec_dep_2_train = clf_gdb_2.decision_function(X_train)
dec_dep_3_train = clf_gdb_3.decision_function(X_train)
dec_dep_4_train = clf_gdb_4.decision_function(X_train)
dec_dep_5_train = clf_gdb_5.decision_function(X_train)
dec_dep_6_train = clf_gdb_6.decision_function(X_train)
dec_dep_7_train = clf_gdb_7.decision_function(X_train)

dec_dep_1_test = clf_gdb_1.decision_function(X_test)
dec_dep_2_test = clf_gdb_2.decision_function(X_test)
dec_dep_3_test = clf_gdb_3.decision_function(X_test)
dec_dep_4_test = clf_gdb_4.decision_function(X_test)
dec_dep_5_test = clf_gdb_5.decision_function(X_test)
dec_dep_6_test = clf_gdb_6.decision_function(X_test)
dec_dep_7_test = clf_gdb_7.decision_function(X_test)


est_prob_1_train = est_prob_gdb_fn(ratio_pos, ratio_neg, dec_dep_1_train)
est_prob_2_train = est_prob_gdb_fn(ratio_pos, ratio_neg, dec_dep_2_train)
est_prob_3_train = est_prob_gdb_fn(ratio_pos, ratio_neg, dec_dep_3_train)
est_prob_4_train = est_prob_gdb_fn(ratio_pos, ratio_neg, dec_dep_4_train)
est_prob_5_train = est_prob_gdb_fn(ratio_pos, ratio_neg, dec_dep_5_train)
est_prob_6_train = est_prob_gdb_fn(ratio_pos, ratio_neg, dec_dep_6_train)
est_prob_7_train = est_prob_gdb_fn(ratio_pos, ratio_neg, dec_dep_7_train)


est_prob_1_test = est_prob_gdb_fn(ratio_pos, ratio_neg, dec_dep_1_test)
est_prob_2_test = est_prob_gdb_fn(ratio_pos, ratio_neg, dec_dep_2_test)
est_prob_3_test = est_prob_gdb_fn(ratio_pos, ratio_neg, dec_dep_3_test)
est_prob_4_test = est_prob_gdb_fn(ratio_pos, ratio_neg, dec_dep_4_test)
est_prob_5_test = est_prob_gdb_fn(ratio_pos, ratio_neg, dec_dep_5_test)
est_prob_6_test = est_prob_gdb_fn(ratio_pos, ratio_neg, dec_dep_6_test)
est_prob_7_test = est_prob_gdb_fn(ratio_pos, ratio_neg, dec_dep_7_test)


# 조정된 추정 확률로 예측 및 결과 도출

pred_2_train = np.where(est_prob_2_train >= 0.5, 1, -1)

pred_2_test = np.where(est_prob_2_test >= 0.5, 1, -1)

y_train_np = np.array(y_train)
y_test_np = np.array(y_test)

np.sum(y_train_np == pred_train) / len(y_train_np)

np.sum(y_test_np == pred_test) / len(y_test_np)

np.sum(pred_train[y_train_np == 1] == 1) / np.sum(y_train_np == 1)

np.sum(pred_test[y_test_np == 1] == 1) / np.sum(y_test_np == 1)


est_prob_test = est_prob_1_test
est_prob_test = est_prob_2_test
est_prob_test = est_prob_3_test
est_prob_test = est_prob_4_test
est_prob_test = est_prob_5_test
est_prob_test = est_prob_6_test
est_prob_test = est_prob_7_test



fig_test, axes = plt.subplots(2, 1, figsize = (20, 10))

x_axis_pos = np.array(list(range(len(y_test[y_test == 1]))))
x_axis_neg = np.array(list(range(len(y_test[y_test == -1]))))

y_axis_pos = est_prob_test[y_test == 1]
y_axis_neg = est_prob_test[y_test == -1]

axes[0].scatter(x_axis_pos, y_axis_pos, color = sns.color_palette()[1])
axes[1].scatter(x_axis_neg, y_axis_neg, color = sns.color_palette()[0])

fig_test.savefig('OUT/PLOT/gdb_result/max_depth_5_test.png')


est_prob_train = est_prob_1_train
est_prob_train = est_prob_2_train
est_prob_train = est_prob_3_train
est_prob_train = est_prob_4_train
est_prob_train = est_prob_5_train
est_prob_train = est_prob_6_train
est_prob_train = est_prob_7_train


fig_train, axes = plt.subplots(2, 1, figsize = (20, 10))

x_axis_pos = np.array(list(range(len(y_train[y_train == 1]))))
x_axis_neg = np.array(list(range(len(y_train[y_train == -1]))))

y_axis_pos = est_prob_train[y_train == 1]
y_axis_neg = est_prob_train[y_train == -1]

axes[0].scatter(x_axis_pos, y_axis_pos, color = sns.color_palette()[1])
axes[1].scatter(x_axis_neg, y_axis_neg, color = sns.color_palette()[0])

fig_train.savefig('OUT/PLOT/gdb_result/max_depth_6_train.png')



'''
===================================================================================================

----------------------------------------------
 4. Feature engineering -part 2
----------------------------------------------
(1) Data preperation
(2) binary feature pattern recognition
----------------------------------------------


 << Data preperation >>

'''


feature_bin = bin_feature + catego_feature

X_total = total_df.loc[:, feature_bin]
y_total = total_df.target
y_total = y_total.replace(0, -1)



'''
--------------------------------------------------------------------------------------

 << Positive와 negative의 pattern 확인해보기 >>
 
 
'''

X_total_pos = X_total.loc[np.array(y_total) == 1, :]
X_total_neg = X_total.loc[np.array(y_total) == -1, :]

X_total_pos = X_total_pos.reset_index(drop = True)
X_total_neg = X_total_neg.reset_index(drop = True)

X_total_pos = np.array(X_total_pos)
X_total_neg = np.array(X_total_neg)


newcmap = colors.ListedColormap(colors = [sns.color_palette()[0],
                                         sns.color_palette()[1]])

    

fig_2, axes = plt.subplots(2, 1, figsize = (40, 20))

axes[0].pcolormesh(np.flip(X_total_pos[:, 25:60], 0), cmap = newcmap)

axes[1].pcolormesh(np.flip(X_total_neg[:, 25:60], 0), cmap = newcmap)    
    
fig_2.savefig('OUT/PLOT/heatmap/binary_feature_sub_1.png')    
    

'''
--------------------------------------------------------------------------------------

 << catego_list의 조합 확인해보기 >>
  
'''

catego_df = raw_df.loc[:, catego_list]

catego_df_pos = catego_df.loc[np.array(y_total) == 1, :]
catego_df_neg = catego_df.loc[np.array(y_total) == -1, :]

catego_df_pos = catego_df_pos.reset_index(drop = True)
catego_df_neg = catego_df_neg.reset_index(drop = True)

catego_pos = np.array(catego_df_pos).tolist()
catego_neg = np.array(catego_df_neg).tolist()

tmp_fn = lambda x : '++'.join(list(map(str, x)))

pattern_pos = list(map(tmp_fn, catego_pos))
pattern_neg = list(map(tmp_fn, catego_neg))

pattern_pos_uq = list(set(pattern_pos))
pattern_neg_uq = list(set(pattern_neg))

pattern_pos_dict = {pattern : pattern_pos.count(pattern) for pattern in pattern_pos_uq}
pattern_neg_dict = {pattern : pattern_neg.count(pattern) for pattern in pattern_neg_uq}

# maximum number of pattern in pattern_pos = '1++0++0++7++1++-1++0++-1++1++1++1++2++1++64'

pattern_idx_pos = list(pattern_pos_dict.values()).index(20)

max_pattern_pos = list(pattern_pos_dict.keys())[pattern_idx_pos]
pattern_neg.count(max_pattern_pos)

np.where(np.array(list(pattern_pos_dict.values())) == 22)[0]


'''
--------------------------------------------------------------------------------------

 << catego_list binary feature로 확장 >>
  
'''

catego_onehot_df = pd.get_dummies(catego_df, columns = catego_df.columns.tolist())

catego_onehot_df_pos = catego_onehot_df.loc[np.array(y_total) == 1, :]
catego_onehot_df_neg = catego_onehot_df.loc[np.array(y_total) == -1, :]

catego_onehot_df_pos = catego_onehot_df_pos.reset_index(drop = True)
catego_onehot_df_neg = catego_onehot_df_neg.reset_index(drop = True)

catego_onehot_pos = np.array(catego_onehot_df_pos)
catego_onehot_neg = np.array(catego_onehot_df_neg)

tmp_fn = lambda x : '++'.join(list(map(str, x.tolist())))

pattern_onehot_pos = np.vectorize(tmp_fn, signature = '(m)->()')(catego_onehot_pos)
pattern_onehot_neg = np.vectorize(tmp_fn, signature = '(m)->()')(catego_onehot_neg)

pattern_onehot_pos = pattern_onehot_pos.tolist()
pattern_onehot_neg = pattern_onehot_neg.tolist()

pattern_onehot_pos_uq = list(set(pattern_onehot_pos))
pattern_onehot_neg_uq = list(set(pattern_onehot_neg))

pattern_onehot_pos_dict = {pattern : pattern_onehot_pos.count(pattern) for pattern in pattern_onehot_pos_uq}


'''
--------------------------------------------------------------------------------------

 << catego_list의 최적 조합 찾기 >>
  
'''

feature_bin = bin_feature + catego_feature

X_total = total_df.loc[:, feature_bin]
y_total = total_df.target
y_total = y_total.replace(0, -1)

catego_df = raw_df.loc[:, catego_list]

pattern_result_2 = pattern_combination_fn(catego_df, 2, np.array(y_total))
pattern_result_3 = pattern_combination_fn(catego_df, 3, np.array(y_total))



pickle.dump(pattern_result_2, open('OUT/Pfile/pattern_result_2.pickle', 'wb'))
pickle.dump(pattern_result_3, open('OUT/Pfile/pattern_result_3.pickle', 'wb'))




pattern_result_2[np.argmax(pattern_result_2[:, 5])]
pattern_result_3[np.argmax(pattern_result_3[:, 5])]

# 최적 feature 조합에 대한 탐색 : ['ps_ind_02_cat', 'ps_car_01_cat', 'ps_car_08_cat']

opt_combi_catego_df = catego_df.loc[:, ['ps_ind_02_cat', 'ps_car_01_cat', 'ps_car_08_cat']]

opt_combi_catego_df.iloc[:, 0].value_counts()
opt_combi_catego_df.iloc[:, 1].value_counts()
opt_combi_catego_df.iloc[:, 2].value_counts()

opt_combi_catego_np = np.array(opt_combi_catego_df)
opt_loc = (opt_combi_catego_np[:, 0] == -1) & (opt_combi_catego_np[:, 1] == -1) & (opt_combi_catego_np[:, 2] == 1)

one_hot_catego_df = pd.get_dummies(catego_df, columns = catego_df.columns.tolist())
one_hot_catego_np = np.array(one_hot_catego_df)

opt_vector = np.array(one_hot_catego_df)[opt_loc]
opt_centroid = opt_vector.sum(axis = 0) / opt_vector.shape[0]

tmp_fn = lambda vec : np.sqrt(np.sum(opt_centroid - vec) ** 2)

euclid_dist = np.vectorize(tmp_fn, signature = '(m)->()')(one_hot_catego_np)

scaler = preprocessing.MinMaxScaler()
scaler.fit(euclid_dist.reshape(-1, 1))

euclid_dist_minmax = scaler.transform(euclid_dist.reshape(-1, 1)).reshape(1, -1)[0]

fig_dist, axes = plt.subplots(2, 1, figsize = (20, 10))

axes[0].scatter(np.arange(1, len(euclid_dist_minmax)+1)[np.array(y_total) == 1],
                euclid_dist_minmax[np.array(y_total) == 1],
                color = sns.color_palette()[1],
                alpha = 0.5,
                s = 0.7)

axes[1].scatter(np.arange(1, len(euclid_dist_minmax)+1)[np.array(y_total) == -1],
                euclid_dist_minmax[np.array(y_total) == -1],
                color = sns.color_palette()[0],
                alpha = 0.5,
                s = 0.7)

plt.close(fig_dist)

#-------------------------------------------------------------------------------

one_hot_opt_catego_df = pd.get_dummies(opt_combi_catego_df, columns = opt_combi_catego_df.columns.tolist())
opt_vector = np.array(one_hot_opt_catego_df)[opt_loc][0]

tmp_fn = lambda vec : np.sqrt(np.sum(opt_vector - vec) ** 2)

euclid_dist = np.vectorize(tmp_fn, signature = '(m)->()')(one_hot_opt_catego_df)

scaler = preprocessing.MinMaxScaler()
scaler.fit(euclid_dist.reshape(-1, 1))

euclid_dist_minmax = scaler.transform(euclid_dist.reshape(-1, 1)).reshape(1, -1)[0]

fig_dist, axes = plt.subplots(2, 1, figsize = (20, 10))

axes[0].scatter(np.arange(1, len(euclid_dist_minmax)+1)[np.array(y_total) == 1],
                euclid_dist_minmax[np.array(y_total) == 1],
                color = sns.color_palette()[1],
                alpha = 0.5,
                s = 0.7)

axes[1].scatter(np.arange(1, len(euclid_dist_minmax)+1)[np.array(y_total) == -1],
                euclid_dist_minmax[np.array(y_total) == -1],
                color = sns.color_palette()[0],
                alpha = 0.5,
                s = 0.7)

plt.close(fig_dist)

'''
--------------------------------------------------------------------------------------

 << PCA를 통한 feature extraction >>
 
 
'''

feature_bin = bin_feature + catego_feature

X_total = total_df.loc[:, feature_bin]
y_total = total_df.target
y_total = y_total.replace(0, -1)
y_total_np = np.array(y_total)



pca = decomposition.PCA(n_components = X_total.shape[1])
pca.fit(X_total)


fig, ax = plt.subplots(figsize = (10, 7))

ax.scatter(np.arange(X_total.shape[1]), pca.explained_variance_ratio_, s = 5)

plt.close(fig)

PC_total = pca.transform(X_total)
PC_total_df = pd.DataFrame(PC_total, columns = list(map(lambda x : 'PC'+x, list(map(str, list(range(1, X_total.shape[1]+1)))))))

for i in range(PC_total_df.shape[1]):
    
    pc = np.array(PC_total_df.iloc[:, i])
    
    fig_loop, ax = plt.subplots(2, 1, figsize = (20, 10))
    
    x_axis_pos = np.arange(1, X_total.shape[0]+1)[y_total_np == 1]
    x_axis_neg = np.arange(1, X_total.shape[0]+1)[y_total_np == -1]
    
    
    ax[0].scatter(x_axis_pos, pc[y_total_np == 1], color = sns.color_palette()[1], s = 5, alpha = 0.3)
    ax[1].scatter(x_axis_neg, pc[y_total_np == -1], color = sns.color_palette()[0], s = 5, alpha = 0.3)
    
    
    fig_loop.savefig('OUT/PLOT/PCA_plot/PC{}.png'.format(i+1))
    
    plt.close(fig_loop)

    










