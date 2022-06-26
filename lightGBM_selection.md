```python

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
    # clf_LGBM, gs_LGBM = lgbm_search()
    clf_LGBM = lgb.LGBMClassifier(max_depth=1, random_state=314, silent=True,
                          metric='None', n_jobs=4, n_estimators=5000)
    
    opt_parameters = paralgbm[i]

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

```
