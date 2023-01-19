```python

##########
### AHE ###
##########

def Light_Import(i):
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
    clf_final1.fit(X_tr, y_tr, **fit_params, callbacks=[lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_0995)])

    feat_imp = pd.Series(clf_final1.feature_importances_, index=X_train.columns)
    feat_imp = feat_imp.sort_values(ascending=False)
    feat_imp_norm = feat_imp/feat_imp.sum()

  return feat_imp_norm
  
def select_features_lgbm(IMPORT_DCS):
  feat_dsc = IMPORT_DCS.index.tolist()
  feat_names = []
  selected_import = []
  for i in range(len(IMPORT_DCS)):
    if(IMPORT_DCS[i]>0):
      selected_import.append(IMPORT_DCS[i])
      feat_names.append(feat_dsc[i])

  return feat_names, selected_import
  
import pandas as pd
from matplotlib import pyplot as plt
from google.colab import files 

def plot_import_lgbm(Sel_names, Sel_imp, num_fold, color_pref):
  name = Sel_names
  price = Sel_imp

  # Figure Size
  fig, ax = plt.subplots(figsize =(20, 10))

  # Horizontal Bar Plot
  ax.barh(name, price, color = color_pref)

  # Add padding between axes and labels
  ax.xaxis.set_tick_params(pad = 5)
  ax.yaxis.set_tick_params(pad = 10)

  # Add x, y gridlines
  ax.grid(b = True, color ='black',
      linestyle ='-.', linewidth = 0.5,
      alpha = 0.2)

  # Show top values
  ax.invert_yaxis()

  # Add annotation to bars
  for i in ax.patches:
    plt.text(i.get_width()+0.001, i.get_y()+0.5,
        str(round((i.get_width()), 4)),
        fontsize = 10, fontweight ='bold',
        color ='black')

  # Add Plot Title
  ax.set_title('Most Important Features in Fold'+num_fold,
              fontsize = 20, fontweight ='bold',
              loc ='center', )

  plt.savefig("AHE_LGBM_fold"+num_fold+".jpg")
  files.download("AHE_LGBM_fold"+num_fold+".jpg")

  # Show Plot
  plt.show()
  
  
# Select features on each fold
def fold_select_LGBM(Train_data, Test_data):
  Train_MIG_map = [] #Train_data[Select_names]
  Test_MIG_map = [] #Test_data[Select_names]
  NAMES_imp = [] 
  SELECT_imp = []
  for i in range(len(Train_data)):
    d_fold = Train_data[i]
    X_select = d_fold.drop(["target_MAP"], axis = 1)
    y_select = d_fold["target_MAP"].astype(int)

    feat_import = Light_Import(i)
    plot_import_lgbm(feat_import.index.tolist(), feat_import, "original "+str(i), "red")
    Select_names = select_features_lgbm(feat_import)[0]
    Select_imp = select_features_lgbm(feat_import)[1]
    NAMES_imp.append(Select_names)
    SELECT_imp.append(Select_imp)
    col = [c for c in Select_names]
    col.append("target_MAP")

    Train_MIG_map.append(d_fold[col])
    Test_MIG_map.append(Test_data[i][col])

  return Train_MIG_map, Test_MIG_map, NAMES_imp, SELECT_imp
  
Train_LGBM_map, Test_LGBM_map, NAMES_imp_lgbm, SELECT_imp_lgbm = fold_select_LGBM(Train_data, Test_data)
 
##########
### TE ###
##########

Change the target "target_MAP" to "target_HR"

```
