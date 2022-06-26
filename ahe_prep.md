```python

############################
############################
### AHE early prediction ###
############################
############################

##########################################################################
## Cross validation + RandomUnderSampling: Train and test data building ##
##########################################################################

from sklearn.model_selection import cross_val_score,KFold
from imblearn.under_sampling import RandomUnderSampler

def cv_rus_train_test(y_pref = "target_MAP", y_alter = "target_HR"):
  patients = np.linspace(0,Nt-1,Nt)
  kf = KFold(n_splits=5, shuffle=True, random_state=10)

  Train_folds = []
  Test_folds = []
  for train, test in kf.split(patients): # Cross validation
    te_f = []
    for j in test:
      te_f.append(final_data[j].drop("target_HR", axis = 1))
    Test_folds.append(te_f)

    tr_f = []
    for i in train:
      X = final_data[i].drop([y_pref, y_alter], axis = 1)
      y = final_data[i][y_pref].astype(int)
      if(len(np.unique(y))==2):
        rus = RandomUnderSampler(sampling_strategy='majority', random_state=10) # RandomUnderSampling
        X, y = rus.fit_resample(X, y)
      train_rus_rec = pd.concat([X, y], axis=1)
      tr_f.append(train_rus_rec)
    Train_folds.append(tr_f)

  return Train_folds, Test_folds

Train_folds, Test_folds = cv_rus_train_test(y_pref = "target_MAP", y_alter = "target_HR")

# Combine training data by fold
Train_data = []
dat = data2[3].drop("target_HR", axis = 1)
for fold in Train_folds:
  tmp = pd.DataFrame(columns = dat.columns)
  for tr_d in fold:
    tmp = pd.concat([tmp, tr_d], ignore_index = True)
  Train_data.append(tmp)

# Combine testing data by fold
Test_data = []
for fold in Test_folds:
  tmp = pd.DataFrame(columns = dat.columns)
  for te_d in fold:
    tmp = pd.concat([tmp, te_d], ignore_index = True)
  Test_data.append(tmp)
  
############################################
## Number of observation windows per fold ##
############################################

def count_OW():
  OW_train = []
  OW_test = []
  for i in range(5):
    n = Train_data[i].shape[0]
    m = Test_data[i].shape[0]
    OW_train.append(n/60)
    OW_test.append(m/60)
  
  return OW_train, OW_test

OW_train, OW_test = count_OW()

```

