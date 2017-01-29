#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 23:35:35 2016

@author: jeremy
"""
import math
import datetime
import numpy as np
import pandas as pd
#from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from operator import itemgetter
import xgboost as xgb
import random
import time
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from numpy import genfromtxt
import seaborn as sns

target_cols = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']
target_cols = target_cols[2:]

def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.

    This function computes the average prescision at k between two lists of
    items.

    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The average precision at k over the input lists

    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p == actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 1.0

    return score / min(1, k)

def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.

    This function computes the mean average prescision at k between two lists
    of lists of items.

    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The mean average precision at k over the input lists

    """
    return np.mean([apk(actual,p,k) for p in zip(predicted)])

    
    
random.seed(2016)
       
    
    
def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()


def get_importance(gbm, features):
    create_feature_map(features)
    importance = gbm.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=itemgetter(1), reverse=True)
    return importance

def get_features(train, test):
    trainval = list(train.columns.values)
#    testval = list(test.columns.values)
    output = trainval
#    output.remove('people_id')
#    output.remove('activity_id')
    return sorted(output)


def run_single(train, test, features, target, random_state=0, check_test_score=True):
    eta = 0.05
    max_depth= 6 #6
    subsample = .9
    colsample_bytree = .9
    min_chil_weight=1
    start_time = time.time()

    print('XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample, colsample_bytree))
    params = {
        "objective": "multi:softprob",
        "booster" : "gbtree",
        "eval_metric": "mlogloss",
        "eta": eta,
        "tree_method": 'exact',
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "silent": 1,
        "min_chil_weight":min_chil_weight,
        "seed": random_state,
        "num_class" : 22,
    }
    num_boost_round = 1000
    early_stopping_rounds = 50
    test_size = 0.05

   
    
    X_train, X_valid = train_test_split(train, test_size=test_size, random_state=random_state)
    print('Length train:', len(X_train.index))
    print('Length valid:', len(X_valid.index))
    y_train = X_train[target]
    y_valid = X_valid[target]
    dtrain = xgb.DMatrix(X_train[features], y_train, missing=-99)
    dvalid = xgb.DMatrix(X_valid[features], y_valid, missing =-99)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=True)

    print("Validating...")
    check = gbm.predict(xgb.DMatrix(X_valid[features]), ntree_limit=gbm.best_iteration+1)
    check = np.argsort(check, axis=1)
    a = np.fliplr(check)[:,:1] 
    check = np.fliplr(check)[:,:7] 
    score = accuracy_score(X_valid[target].values, a)
    score2 = accuracy_score(X_valid[target].values, check[:,1])
    score3 = accuracy_score(X_valid[target].values, check[:,2])
    score4 = accuracy_score(X_valid[target].values, check[:,3])
    print('Accuracy score first col: {:.6f}'.format(score))
    print('Accuracy score second col: {:.6f}'.format(score2))
    print('Accuracy score third col: {:.6f}'.format(score3))
    print('Accuracy score fourth col: {:.6f}'.format(score4))
    map7_sum=0
    i=0
    for index,row in X_valid.iterrows():
        map7=mapk(row[target],check[i,:],7)
        i+=1
        map7_sum += map7
    print('MAP@7 Score: {:.6f}'.format(map7_sum/len(a)))

    imp = get_importance(gbm, features)
    print('Importance array: ', imp)

    print("Predict test set... ")
    test_prediction = gbm.predict(xgb.DMatrix(test[features],missing = -99), ntree_limit=gbm.best_iteration+1)


    #xgb.plot_importance(gbm)
    #plt.show()

    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
    return test_prediction, imp, gbm.best_iteration+1
    
    
    
    
    
    
    
    
    
    
    
if __name__ == "__main__":
    start_time = datetime.datetime.now()
    print("Start time: ",start_time)
    data_path = "../input/"



    
    
    X_train = pd.read_csv("../input/train_X.csv",names=range(236))
    
    

    y_train = pd.read_csv("../input/train_y.csv",names=['target'])
 
  
    test_X = pd.read_csv("../input/test_X.csv",names=range(236))
    
    
    test_X[19] = test_X[19]/X_train[19].max()
    X_train[19]=X_train[19]/X_train[19].max()
    

 
  
    train = X_train
    features = list(X_train.columns.values)
    train['target']=y_train
   

    print(features)

    print("Building model.. ",datetime.datetime.now()-start_time)
    preds, imp, num_boost_rounds = run_single(train, test_X, features, 'target',42,False)
 
    print(datetime.datetime.now()-start_time)


    
    print("Getting the top products..")
    target_cols = np.array(target_cols)
    preds = np.argsort(preds, axis=1)
    preds = np.fliplr(preds)[:,:7] 
    

    test_id = np.array(pd.read_csv(data_path + "test_ver2.csv", usecols=['ncodpers'])['ncodpers'])
    final_preds = [" ".join(list(target_cols[pred])) for pred in preds]
    out_df = pd.DataFrame({'ncodpers':test_id, 'added_products':final_preds})
    out_df.to_csv('submission_file.csv', index=False)  
    print("Submission file created: ",datetime.datetime.now()-start_time) 