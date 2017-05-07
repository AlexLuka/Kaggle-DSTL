RF model:
        n_estimators=40,
        criterion='gini',
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=20,
        max_features='sqrt',
        n_jobs=2,
        random_state=2,
        verbose=1

        model-rf-1: F1 = (0.7704, 0.7376)
        model-rf-2: F1 = (0.7779, 0.7476)
        model-rf-3: F1 = ??
        model-rf-4: F1 = (0.7698, 0.7422)



XGB model:
        params = {
            # 'seed': np.random.randint(np.iinfo(np.int32).max),
            'colsample_bytree': 0.7,
            'silent': 1,
            'subsample': 0.7,
            'learning_rate': 0.3,
            'objective': 'binary:logistic',
            'max_depth': 10,
            'min_child_weight': 20,
            'booster': 'gbtree',
            'eval_metric': 'logloss'
        }

        model-xgb-1:    F1 = (0.7610, 0.7423)
        model-xgb-2:    F1 = (0.7687, 0.7518)
        model-xgb-3:    F1 = ()
        model-xgb-4:    F1 = (0.6707, 0.6573)   Here I used threshold=0.8


ANN model:
        (512, 256, 128, 0.5, 0.5, 0.5)
        patience=4

        model-ann-1:    F1 = (0.7432, 0.7433)
