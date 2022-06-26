```python

#################################
## III- SVM for classification ##
#################################

##########
### AHE ###
##########



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

##########
### TE ###
##########

Change the target "target_MAP" to "target_HR"

```
