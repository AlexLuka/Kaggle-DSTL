Random Forest-1:
        {n_estimators=40,
        criterion='gini',
        max_depth=None,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        n_jobs=4,
        random_state=2,
        verbose=1}

        IMAGE_SET_1         F1 = (0.9199, 0.8799)                   SCORE = 0.08340
        IMAGE_SET_2         F1 = (0.9074, 0.8497)
        IMAGE_SET_3         F1 = (???)

XGB-1:
        params = {
            # 'seed': np.random.randint(np.iinfo(np.int32).max),
            'colsample_bytree': 0.7,
            'silent': 1,
            'subsample': 0.7,
            'learning_rate': 0.3,
            'objective': 'binary:logistic',
            'max_depth': 10,
            'min_child_weight': 50,
            'booster': 'gbtree',
            'eval_metric': 'logloss'
        }

        IMAGE_SET_1         F1 = (0.8666, 0.8578)                   SCORE = 0.08054
        IMAGE_SET_2         F1 = (0.8238, 0.8162)


XGB-1,2 + RF-1,2,3: area > 5.                                       SCORE = 0.08760
XGB-1,2 + RF-1,2,3, area > 100., fill_binary_holes                  SCORE = 0.08840         <---- BEST

