```python

####################
####################
### Full dataset ###
####################
####################

###################################
## I- XGBoost for classification ##
###################################

import keras
from keras import layers
from keras.layers import Dense
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Prepare learning rate shrinkage
def learning_rate_010_decay_power_099(current_iter):
    base_learning_rate = 0.1
    lr = base_learning_rate  * np.power(.99, current_iter)
    return lr if lr > 1e-3 else 1e-3

def learning_rate_010_decay_power_0995(current_iter):
    base_learning_rate = 0.1
    lr = base_learning_rate  * np.power(.995, current_iter)
    return lr if lr > 1e-3 else 1e-3

def learning_rate_005_decay_power_099(current_iter):
    base_learning_rate = 0.05
    lr = base_learning_rate  * np.power(.99, current_iter)
    return lr if lr > 1e-3 else 1e-3
    
# Use test subset for early stopping criterion

def fit_para(X_test,y_test):
  fit_params={"early_stopping_rounds":30, 
          "eval_metric" : 'auc', 
          "eval_set" : [(X_test,y_test)],
          'verbose': 100} 

  return fit_params
  
# Set up HyperParameter search

from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

param_test ={'num_leaves': [9, 12], 
             'learning_rate': [0.001, 0.01, 0.1],
            'min_child_samples': [100, 250], 
            'min_child_weight': [1e-1, 1, 1e1],
            'subsample': [0.6, 0.7, 0.8], 
            'colsample_bytree': [0.6, 0.7, 0.8],
            'reg_alpha': [0, 1e-1, 1],
            'reg_lambda': [0, 1e-1, 1],
            'n_estimators': [100, 250]}
            
#This parameter defines the number of HP points to be tested
# n_HP_points_to_test = 100

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

#n_estimators is set to a "large value". The actual number of trees build will depend on early stopping and 5000 define only the absolute maximum

def model_search():
  #This parameter defines the number of HP points to be tested
  n_HP_points_to_test = 100

  clf_XGB = XGBClassifier(max_depth=1, random_state=314, silent=True,
                         metric='None', n_jobs=4, n_estimators=5000)

  gs_XGB = RandomizedSearchCV(
      estimator=clf_XGB,
      param_distributions=param_test, 
      n_iter=n_HP_points_to_test,
      scoring='roc_auc',
      cv=5,
      refit=True,
      random_state=314,
      verbose=True)
  
  return clf_XGB, gs_XGB
  
def gs_samp(clf_sw):
  gs_sample_weight = GridSearchCV(estimator=clf_sw, 
                                param_grid={'scale_pos_weight':[1,2,6,12]},
                                scoring='roc_auc',
                                cv=5,
                                refit=True,
                                verbose=True,
                                return_train_score=True)
  return gs_sample_weight
  
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def mod_perf_on_fold(i):
  col = ["TEI", "IPP", "CPP", "ER", "RP", "aveFA", "aveAT", "Accuracy", "Time"]
  tmp = pd.DataFrame(columns = col)
  for k in range(1):
    X_train = Train_data[i].drop("target_MAP", axis = 1)
    y_train = Train_data[i]["target_MAP"].astype(int)
    X_test = Test_data[i].drop("target_MAP", axis = 1)
    y_test = Test_data[i]["target_MAP"].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X_train, y_train, test_size=0.3)

    fit_params = fit_para(X_te,y_te)
    clf_XGB, gs_XGB = model_search()
    # HP parameters turnning
    gs_XGB.fit(X_tr, y_tr, **fit_params)
    print('Best score reached: {} with params: {} '.format(gs_XGB.best_score_, gs_XGB.best_params_))
    opt_parameters = gs_XGB.best_params_

    # Tune the weights of unbalanced classes
    clf_sw = XGBClassifier(**clf_XGB.get_params())
    #set optimal parameters
    clf_sw.set_params(**opt_parameters)
    gs_sample_weight = gs_samp(clf_sw)
    gs_sample_weight.fit(X_tr, y_tr, **fit_params)
    print('Best score reached: {} with params: {} '.format(gs_sample_weight.best_score_, gs_sample_weight.best_params_))

    # Build the final model
    #Configure locally from hardcoded values
    clf_final1 = XGBClassifier(**clf_XGB.get_params())
    #set optimal parameters
    clf_final1.set_params(**opt_parameters)
    #Train the final model with learning rate decay
    from datetime import datetime
    start=datetime.now()
    clf_final1.fit(X_tr, y_tr, **fit_params)
    stop=datetime.now()
    Time = stop-start
    run_time = Time.total_seconds()

    # Predict on the submission test sample
    probabilities_full = clf_final1.predict_proba(X_test)
    y_pred_full = clf_final1.predict(X_test)

    #Converting probabilities into 1 or 0  
    for i in range(len(y_pred_full)): 
        if y_pred_full[i]>=.5:       # setting threshold to .5 
          y_pred_full[i]=1 
        else: 
          y_pred_full[i]=0  

    # Accuracy
    acc_full = accuracy_score(y_test, y_pred_full)

    EM_LGBM_full = EvaMec(y_test.astype(int), y_pred_full)
    EM_LGBM_full["Accuracy"] = [acc_full] # Try to add another mesurement like "Number of Unpredicted events"
    EM_LGBM_full["Time"] = [run_time]

    model_probs = probabilities_full[:, 1]
    try:
        model_auc = roc_auc_score(y_test, model_probs)
        EM_LGBM_full["AUC"] = [model_auc]
    except ValueError:
        pass

    tmp = pd.concat([tmp, EM_LGBM_full], ignore_index = True)

  return tmp

#####################################
## II- LightGBM for classification ##
#####################################

# Prepare learning rate shrinkage
def learning_rate_010_decay_power_099(current_iter):
    base_learning_rate = 0.1
    lr = base_learning_rate  * np.power(.99, current_iter)
    return lr if lr > 1e-3 else 1e-3

def learning_rate_010_decay_power_0995(current_iter):
    base_learning_rate = 0.1
    lr = base_learning_rate  * np.power(.995, current_iter)
    return lr if lr > 1e-3 else 1e-3

def learning_rate_005_decay_power_099(current_iter):
    base_learning_rate = 0.05
    lr = base_learning_rate  * np.power(.99, current_iter)
    return lr if lr > 1e-3 else 1e-3
    
# Use test subset for early stopping criterion

def fit_para_lgbm(X_test,y_test):
  fit_params={"early_stopping_rounds":30, 
            "eval_metric" : 'auc', 
            "eval_set" : [(X_test,y_test)],
            'eval_names': ['valid'],
            'verbose': 100,
            'categorical_feature': 'auto'} 

  return fit_params
  
# Set up HyperParameter search
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
param_test_lgbn ={'num_leaves': [9, 12], 
              'learning_rate': [0.001, 0.01, 0.1],
             'min_child_samples': [100, 250], 
             'min_child_weight': [1e-1, 1, 1e1],
             'subsample': [0.6, 0.7, 0.8], 
             'colsample_bytree': [0.6, 0.7, 0.8],
             'reg_alpha': [0, 1e-1, 1],
            'reg_lambda': [0, 1e-1, 1]}
            
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

#n_estimators is set to a "large value". The actual number of trees build will depend on early stopping and 5000 define only the absolute maximum

def lgbm_search():
  #This parameter defines the number of HP points to be tested
  n_HP_points_to_test = 100

  clf_LGBM = lgb.LGBMClassifier(max_depth=1, random_state=314, silent=True,
                         metric='None', n_jobs=4, n_estimators=5000)

  gs_LGBM = RandomizedSearchCV(
    estimator=clf_LGBM,
    param_distributions=param_test_lgbn, 
    n_iter=n_HP_points_to_test,
    scoring='roc_auc',
    cv=5,
    refit=True,
    random_state=314,
    verbose=True)
  
  return clf_LGBM, gs_LGBM
  
def gs_samp(clf_sw):
  gs_sample_weight = GridSearchCV(estimator=clf_sw, 
                                param_grid={'scale_pos_weight':[1,2,6,12]},
                                scoring='roc_auc',
                                cv=5,
                                refit=True,
                                verbose=True,
                                return_train_score=True)
  return gs_sample_weight
  
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def lgbm_perf_on_fold(i):
  col = ["TEI", "IPP", "CPP", "ER", "RP", "aveFA", "aveAT", "Accuracy", "Time"]
  tmp = pd.DataFrame(columns = col)
  for k in range(1):
    X_train = Train_data[i].drop("target_MAP", axis = 1)
    y_train = Train_data[i]["target_MAP"].astype(int)
    X_test = Test_data[i].drop("target_MAP", axis = 1)
    y_test = Test_data[i]["target_MAP"].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X_train, y_train, test_size=0.3)

    fit_params = fit_para_lgbm(X_te,y_te)

    clf_LGBM, gs_LGBM = lgbm_search()
    # HP parameters turnning
    gs_LGBM.fit(X_tr, y_tr, **fit_params)
    print('Best score reached: {} with params: {} '.format(gs_LGBM.best_score_, gs_LGBM.best_params_))
    opt_parameters = gs_LGBM.best_params_
    
    # Tune the weights of unbalanced classes
    clf_sw = lgb.LGBMClassifier(**clf_LGBM.get_params())
    #set optimal parameters
    clf_sw.set_params(**opt_parameters)
    gs_sample_weight = gs_samp(clf_sw)
    gs_sample_weight.fit(X_tr, y_tr, **fit_params)
    print('Best score reached: {} with params: {} '.format(gs_sample_weight.best_score_, gs_sample_weight.best_params_))

    # Build the final model
    #Configure locally from hardcoded values
    clf_final1 = lgb.LGBMClassifier(**clf_LGBM.get_params())
    #set optimal parameters
    clf_final1.set_params(**opt_parameters)
    #Train the final model with learning rate decay
    from datetime import datetime
    start=datetime.now()
    #clf_final1.fit(X_tr, y_tr, **fit_params)
    clf_final1.fit(X_tr, y_tr, **fit_params, callbacks=[lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_0995)])
    stop=datetime.now()
    Time = stop-start
    run_time = Time.total_seconds()

    # Predict on the submission test sample
    probabilities_full = clf_final1.predict_proba(X_test)
    y_pred_full = clf_final1.predict(X_test)

    #Converting probabilities into 1 or 0  
    for i in range(len(y_pred_full)): 
        if y_pred_full[i]>=.5:       # setting threshold to .5 
          y_pred_full[i]=1 
        else: 
          y_pred_full[i]=0  

    # Accuracy
    acc_full = accuracy_score(y_test, y_pred_full)

    EM_LGBM_full = EvaMec(y_test.astype(int), y_pred_full)
    EM_LGBM_full["Accuracy"] = [acc_full] # Try to add another mesurement like "Number of Unpredicted events"
    EM_LGBM_full["Time"] = [run_time]

    model_probs = probabilities_full[:, 1]
    try:
        model_auc = roc_auc_score(y_test, model_probs)
        EM_LGBM_full["AUC"] = [model_auc]
    except ValueError:
        pass

    tmp = pd.concat([tmp, EM_LGBM_full], ignore_index = True)

  return tmp, opt_parameters

############################
## III- SVM for classification ##
############################


import pandas as pd  
import numpy as np  
import seaborn as sns
from sklearn.svm import SVC  
from sklearn.metrics import classification_report, confusion_matrix  
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV 
from sklearn.svm import SVC as svc 
from sklearn.metrics import make_scorer, roc_auc_score
from scipy import stats

from sklearn.model_selection import GridSearchCV

param_grid_SVC = {'C': [0.1,1, 10, 100],
              'gamma': [1,0.1,0.01,0.001],
              'kernel': ['rbf']}
              
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

#n_estimators is set to a "large value". The actual number of trees build will depend on early stopping and 5000 define only the absolute maximum

def SVC_search():
  #This parameter defines the number of HP points to be tested
  n_HP_points_to_test = 100

  clf_SVC = SVC(probability = True, random_state = 1)
  auc = make_scorer(roc_auc_score)

  gs_SVC = RandomizedSearchCV(clf_SVC,
                              param_distributions = param_grid_SVC,
                              n_iter = 10,
                              n_jobs = 4,
                              cv = 5,
                              random_state = 314,
                              scoring = auc,
                              verbose=1)
  
  return clf_SVC, gs_SVC
  
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

def SVC_perf_on_fold(i):
  col = ["TEI", "IPP", "CPP", "ER", "RP", "aveFA", "aveAT", "Accuracy", "Time"]
  tmp = pd.DataFrame(columns = col)
  for k in range(1):
    X_train = final_d[i].drop("target_MAP", axis = 1)
    y_train = final_d[i]["target_MAP"].astype(int)
    X_test = Test_data[i].drop("target_MAP", axis = 1)
    y_test = Test_data[i]["target_MAP"].astype(int)

    clf_SVC, gs_SVC = SVC_search()
    # # # HP parameters turnning
    gs_SVC.fit(X_train, y_train)
    print('Best score reached: {} with params: {} '.format(gs_SVC.best_score_, gs_SVC.best_params_))
    opt_parameters = gs_SVC.best_params_

    clf_SVC = SVC(probability = True)
    # # Build the final model
    # #Configure locally from hardcoded values
    clf_final1 = SVC(**clf_SVC.get_params())
    #set optimal parameters
    clf_final1.set_params(**opt_parameters)
    #Train the final model with learning rate decay
    from datetime import datetime
    start=datetime.now()
    clf_final1.fit(X_train, y_train)
    stop=datetime.now()
    Time = stop-start
    run_time = Time.total_seconds()

    # Predict on the submission test sample
    probabilities_full = clf_final1.predict_proba(X_test)
    y_pred_full = clf_final1.predict(X_test)
    print(confusion_matrix(y_test,y_pred_full))
    print(classification_report(y_test,y_pred_full))#Output

    #Converting probabilities into 1 or 0  
    for i in range(len(y_pred_full)): 
        if y_pred_full[i]>=.5:       # setting threshold to .5 
          y_pred_full[i]=1 
        else: 
          y_pred_full[i]=0

    # Accuracy
    acc_full = accuracy_score(y_test, y_pred_full)

    EM_LGBM_full = EvaMec(y_test.astype(int), y_pred_full)
    EM_LGBM_full["Accuracy"] = [acc_full] # Try to add another mesurement like "Number of Unpredicted events"
    EM_LGBM_full["Time"] = [run_time]

    model_probs = probabilities_full[:, 1]
    try:
        model_auc = roc_auc_score(y_test, model_probs)
        EM_LGBM_full["AUC"] = [model_auc]
    except ValueError:
        pass

    tmp = pd.concat([tmp, EM_LGBM_full], ignore_index = True)

  return tmp, opt_parameters

#####################################
## IV - Naive Bafes Classification ##
#####################################
  
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd  
from sklearn.svm import SVC  
from sklearn.metrics import classification_report, confusion_matrix  
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import RepeatedStratifiedKFold

cv_method = RepeatedStratifiedKFold(n_splits=5, 
                                    n_repeats=1, 
                                    random_state=999)
                                    
from sklearn.preprocessing import PowerTransformer
from sklearn.naive_bayes import GaussianNB

def search_NB():
  model = GaussianNB()
  params_NB = {'var_smoothing': [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]}

  gs_NB = GridSearchCV(estimator=model, 
                      param_grid=params_NB, 
                      cv=cv_method,
                      verbose=1, 
                      scoring='roc_auc')
  
  return model, gs_NB
  
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

def NB_perf_on_fold(i):
  col = ["TEI", "IPP", "CPP", "ER", "RP", "aveFA", "aveAT", "Accuracy", "Time"]
  tmp = pd.DataFrame(columns = col)
  for k in range(1):
    X_train = Train_data[i].drop("target_MAP", axis = 1)
    y_train = Train_data[i]["target_MAP"].astype(int)
    X_test = Test_data[i].drop("target_MAP", axis = 1)
    y_test = Test_data[i]["target_MAP"].astype(int)

    clf_SVC, gs_SVC = search_NB()
    # # # HP parameters turnning
    gs_SVC.fit(X_train, y_train)
    print('Best score reached: {} with params: {} '.format(gs_SVC.best_score_, gs_SVC.best_params_))
    opt_parameters = gs_SVC.best_params_
    
    # # Build the final model
    # #Configure locally from hardcoded values
    clf_final1 = GaussianNB(**clf_SVC.get_params())
    #set optimal parameters
    clf_final1.set_params(**opt_parameters)
    #Train the final model with learning rate decay
    from datetime import datetime
    start=datetime.now()
    clf_final1.fit(X_train, y_train)
    stop=datetime.now()
    Time = stop-start
    run_time = Time.total_seconds()

    # Predict on the submission test sample
    probabilities_full = clf_final1.predict_proba(X_test)
    y_pred_full = clf_final1.predict(X_test)
    print(confusion_matrix(y_test,y_pred_full))
    print(classification_report(y_test,y_pred_full))#Output

    #Converting probabilities into 1 or 0  
    for i in range(len(y_pred_full)): 
        if y_pred_full[i]>=.5:       # setting threshold to .5 
          y_pred_full[i]=1 
        else: 
          y_pred_full[i]=0

    # Accuracy
    acc_full = accuracy_score(y_test, y_pred_full)

    EM_LGBM_full = EvaMec(y_test.astype(int), y_pred_full)
    EM_LGBM_full["Accuracy"] = [acc_full] # Try to add another mesurement like "Number of Unpredicted events"
    EM_LGBM_full["Time"] = [run_time]

    model_probs = probabilities_full[:, 1]
    try:
        model_auc = roc_auc_score(y_test, model_probs)
        EM_LGBM_full["AUC"] = [model_auc]
    except ValueError:
        pass

    tmp = pd.concat([tmp, EM_LGBM_full], ignore_index = True)

  return tmp, opt_parameters



```
