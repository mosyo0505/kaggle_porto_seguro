#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
==============================================================================================

                                        Custom functions

==============================================================================================

---------------------------------
 0. Set environment
---------------------------------


'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import ensemble
from itertools import combinations




'''
===================================================================================

-------------------------------------------------------
 1. Scope funtion
-------------------------------------------------------
 현재 사용하고 있는 개체들을 list 형태로 return 해주는 함수.
 Input : default_namespace.
-------------------------------------------------------
'''

def scope(x):

    cur = (set(globals().keys()) - set(['x'])) | set(['scope'])
    
    return([y for y in list(cur - x) if y[0] != '_'])


'''
===================================================================================

-----------------------------
 2. T-test function
-----------------------------
 Input : 1차원 array
 return : T-test 결과 p value
-----------------------------
'''

def t_test_fn(ar1, ar2):
    
    if np.std(ar2) > 0:
            
        f_stat = np.sqrt(np.std(ar1)) / np.sqrt(np.std(ar2))
        
        df_1 = len(ar1) - 1
        df_2 = len(ar2) - 1
        p_value_f = stats.f.cdf(f_stat, df_1, df_2)
        
        if p_value_f >= 0.05:
            
            p_value = stats.ttest_ind(ar1, ar2, equal_var = True)[1]
            
            
        else:
            
            p_value = stats.ttest_ind(ar1, ar2, equal_var = False)[1]
            
        return p_value
    
    else:
        
        print('Warning : standard deviation is 0')
        

'''
==================================================================================

------------------------------------------------------------------------------
 3. clf_gdb_bag_train_fn
------------------------------------------------------------------------------
 각 bootstrapping sample에 대하여 gradient boosting model을 training 시키는 함수
------------------------------------------------------------------------------
'''


def clf_gdb_bag_train_fn(X, y, n_bag):

#    Positive class, Negative class index 구하기
    
    pos_idx = np.where(np.array(y) == 1)[0]
    neg_idx = np.where(np.array(y) == -1)[0]

#    n_boots_pos : n_boots_neg = 40 : 60
    
    n_boots_pos = int(round(sum(y == 1) * 0.6))
    n_boots_neg = int(round(n_boots_pos * 1.5))
    
#    Bootstrapping에 사용할 index set 10000개 만들기
    
    n_boots = np.c_[np.array([n_boots_pos]*n_bag), [n_boots_neg]*n_bag]
    
    def tmp_fn(x):
        
        chosen_pos_idx = np.random.choice(pos_idx, x[0])
        chosen_neg_idx = np.random.choice(neg_idx, x[1])
        
        return np.r_[chosen_pos_idx, chosen_neg_idx]
        
    boots_idx = np.vectorize(tmp_fn, signature = '(m)->(k)')(n_boots)
    
    del tmp_fn
    
#    각 bootstrapping에 사용할 X와 y를 list에 저장하기
    
    tmp_fn = lambda x : np.array(X)[x, :]
    
    boots_X = np.vectorize(tmp_fn, signature = '(m)->(m,k)')(boots_idx)
    
    del tmp_fn
    
    tmp_fn = lambda x : np.array(y)[x]
    
    boots_y = np.vectorize(tmp_fn, signature = '(m)->(m)')(boots_idx)
    
#    Bootstrapping하여 gradient boosting model training 시키기
    
    boots_model_idx = np.array(range(boots_X.shape[0]))
    
    def tmp_fn(x):
        
        print('=============== ', x+1,'====================')
        
        clf_gdb = ensemble.GradientBoostingClassifier(n_estimators = 10000,
                                              max_depth = 3,
                                              random_state = 0,
                                              verbose = 2,
                                              )
        
        return clf_gdb.fit(boots_X[x], boots_y[x])
        
        
    boots_model = np.vectorize(tmp_fn)(boots_model_idx)
    
    return boots_model


'''
================================================================================

---------------------------------------------------
 4. clf_gdb_bag_test_fn
---------------------------------------------------
 Bagging gradient boosting model을 test 하는 함수
 Input으로 clf_gdb_bag_train_fn의 output이 필요함
---------------------------------------------------
'''


def clf_gdb_bag_test_fn(X, y, boots):
    
    tmp_fn = lambda x : x.decision_function(X)
    
    decision_functions = np.vectorize(tmp_fn, signature = '()->(m)')(boots)
    
    bagging_result = decision_functions.mean(axis = 0)
    
    return bagging_result


    
'''
================================================================================

---------------------------------------------------
 5. est_prob_gdb_fn
---------------------------------------------------
 Gradient boosting model의 추정확률을 보정해주는 함수.
---------------------------------------------------
'''

def est_prob_gdb_fn(pos, neg, decision_function):
    
    tmp_fn = lambda x : (neg * np.exp(2*x)) / (pos + neg*np.exp(2*x))
    
    est_prob = np.vectorize(tmp_fn)(decision_function)
    
    return est_prob



'''
================================================================================

---------------------------------------------------
 6. pattern_combination_fn
---------------------------------------------------
 
---------------------------------------------------
'''
    
def pattern_combination_fn(df, n_combi, y):
    
    feature_names = df.columns.tolist()
    
    combi = list(combinations(feature_names, n_combi))
    combi = list(map(list, combi))
    
    combi = np.array(combi)
    
    
#----------------------------------------------------------------------------
#                      Sub custom functon 1.
#----------------------------------------------------------------------------
    
    def sub_custom_fn_1(x):
        
        x = x.tolist()
        
        sub_df = df.loc[:, x]
        sub_df = np.array(sub_df)
        
        tmp_fn = lambda k : '++'.join(list(map(str, k)))
        
        pattern = list(map(tmp_fn, sub_df))
        
        pattern_uq = np.array(list(set(pattern)))
        
        pattern_np = np.array(pattern)
        
        del tmp_fn
        
#       ----------------------------------------------------------------------
#                           Sub custom function 2
#       ----------------------------------------------------------------------
        
        def sub_custom_fn_2(w):
            
            pattern_np_pos = pattern_np[(pattern_np == w) & (y == 1)]
            pattern_np_total = pattern_np[pattern_np == w]
            ratio = (len(pattern_np_pos) / len(pattern_np_total)) * 100
            
            sub_result = [w, len(pattern_np_total), len(pattern_np_pos), ratio]
            
            return np.array(sub_result)
            
#       ----------------------------------------------------------------------
        
        sub_custom_fn_2_result = np.vectorize(sub_custom_fn_2, signature = '()->(m)')(pattern_uq)
        sub_custom_fn_2_result = sub_custom_fn_2_result[sub_custom_fn_2_result[:, 1].astype(int) > 50]
        
        max_idx = np.argmax(sub_custom_fn_2_result[:, 3].astype(float))
        min_idx = np.argmin(sub_custom_fn_2_result[:, 3].astype(float))
        
        result = np.array([x,
                           sub_custom_fn_2_result[max_idx][0],
                           sub_custom_fn_2_result[min_idx][0],
                           sub_custom_fn_2_result[max_idx][3].astype(float),
                           sub_custom_fn_2_result[min_idx][3].astype(float),
                           sub_custom_fn_2_result[max_idx][3].astype(float)-sub_custom_fn_2_result[min_idx][3].astype(float)])
        
        return result
#----------------------------------------------------------------------------
    
    pattern_combination_result = np.vectorize(sub_custom_fn_1, signature = '(m)->(k)')(combi)
    
    return pattern_combination_result
    
    

    
    
    

    
    
    

    
    
    
    
    


        
    
    
    
















































        

