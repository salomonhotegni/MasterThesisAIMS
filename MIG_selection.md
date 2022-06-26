```python

# Feature Selection Based on Mutual Information Gain for Classification
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest

# Normalized feature importances
def MIGC(X_tr_k,y_tr_k):
    mutual_inf = mutual_info_classif(X_tr_k, y_tr_k)
    mutual_inf = pd.Series(mutual_inf)
    mutual_inf.index = X_tr_k.columns
    mutual_inf = mutual_inf.sort_values(ascending=False)
    norm_feat_import = mutual_inf/mutual_inf.sum()
    
    return norm_feat_import
    
# Select features with at least 1% importance

def select_features(IMPORT_DCS):
  feat_dsc = IMPORT_DCS.index.tolist()
  feat_names = []
  selected_import = []
  for i in range(len(IMPORT_DCS)):
    if(IMPORT_DCS[i]>=0.01):
      selected_import.append(IMPORT_DCS[i])
      feat_names.append(feat_dsc[i])

  return feat_names, selected_import
  
# Select features on each fold
def fold_select_MIG(Train_data, Test_data):
  Train_MIG_map = [] #Train_data[Select_names]
  Test_MIG_map = [] #Test_data[Select_names]
  NAMES_imp = [] 
  SELECT_imp = []
  for i in range(len(Train_data)):
    d_fold = Train_data[i]
    X_select = d_fold.drop(["target_MAP"], axis = 1)
    y_select = d_fold["target_MAP"].astype(int)

    feat_import = MIGC(X_select,y_select)
    Select_names = select_features(feat_import)[0]
    Select_imp = select_features(feat_import)[1]
    NAMES_imp.append(Select_names)
    SELECT_imp.append(Select_imp)
    col = [c for c in Select_names]
    col.append("target_MAP")

    Train_MIG_map.append(d_fold[col])
    Test_MIG_map.append(Test_data[i][col])

  return Train_MIG_map, Test_MIG_map, NAMES_imp, SELECT_imp
  
Train_MIG_map, Test_MIG_map, NAMES_imp, SELECT_imp = fold_select_MIG(Train_data, Test_data)


import pandas as pd
from matplotlib import pyplot as plt
from google.colab import files 

def plot_import(Sel_names, Sel_imp, num_fold, color_pref):
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
        str(round((i.get_width()), 3)),
        fontsize = 10, fontweight ='bold',
        color ='black')

  # Add Plot Title
  ax.set_title('Most Important Features in Fold'+num_fold,
              fontsize = 20, fontweight ='bold',
              loc ='center', )
              
  plt.savefig("AHE_MIG_fold"+num_fold+".jpg")
  files.download("AHE_MIG_fold"+num_fold+".jpg")

  # Show Plot
  plt.show()



```
