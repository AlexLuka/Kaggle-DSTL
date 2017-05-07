Ann-1:
    F1 = ???



Class 5 generates too many polygons. Therefore, Kaggle's submission system cannot process
all of them within specified Time Out frame.



RF 5-fold:
    F1 = (0.6338, 0.6064)
    F1 = (0.6325, 0.6071)
    F1 = (0.6332, 0.6063)
    F1 = (0.6325, 0.6070)
    F1 = (0.6332, 0.6047)

XGB-1:          image-set-1
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
        threshold = 0.7

    F1 = (0.5545, 0.5401)

XGB-2:          image-set-2
        threshold = 0.5
    F1 = (0.6614, 0.6482)

XGB-3:          image-set-3     threshold=0.5
    F1 = (0.6574, 0.6438)

XGB-4:          image-set-4     threshold=0.5
    params = {
        # 'seed': np.random.randint(np.iinfo(np.int32).max),
        'colsample_bytree': 0.7,
        'silent': 1,
        'subsample': 0.7,
        'learning_rate': 0.3,
        'objective': 'binary:logistic',
        'max_depth': 12,
        'min_child_weight': 30,
        'booster': 'gbtree',
        'eval_metric': 'logloss'
    }
    F1 = (0.6432, 0.6254)

XGB-5:          image-set-5     threshold=0.5    min_child_weight=100
    F1 = (0.6612, 0.6483)

XGB-6:          image-set-6     threshold=0.5
    F1 = (0.6721, 0.6575)
