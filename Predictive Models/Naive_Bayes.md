```python

#####################################
## IV - Naive Bafes Classification ##
#####################################

##########
### AHE ###
##########


  
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

##########
### TE ###
##########

Change the target "target_MAP" to "target_HR"

```
