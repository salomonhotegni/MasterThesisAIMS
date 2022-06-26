```python

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
```
