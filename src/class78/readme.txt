RANDOM FOREST:
    model-rf-1:
            n_estimators=40,
            criterion='gini',
            max_depth=None,
            min_samples_split=20,
            min_samples_leaf=20,
            max_features='auto',
            n_jobs=2,
            random_state=1567,
            verbose=1

            F1 = (0.8807, 0.8728)
    model-rf-2:
            F1 = (0.8830, 0.8741)

XGB:
    model-xgb-1:
            F1 = (0.8901, 0.8782)
    model-xgb-3:
            F1 = (0.9006, 0.8751)   there was no 6070_2_3 image
    model-xgb-4:
            F1 = (0.8917, 0.8823)   still worse than models 1 and 2


COMPLEX-1:
    model-1 (OLD) + RF-1 + RF-2 + XGB-1 + XGB-2             SCORE = 0.06305              BEST SO FAR