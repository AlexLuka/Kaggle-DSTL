model-rf-2:
        n_estimators=20,
        criterion='gini',
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=20,
        max_features='auto',
        n_jobs=2,
        random_state=2,
        verbose=1

        F1 = (0.2277, 0.1389)           SCORE = 0.07741     BEST

    1) class_weight = 'balanced':
        F1 = (0.5188, 0.3493)
    2) min_sample_leaf = 50:
        F1 = (0.3704, 0.3014)
    3) min_sample_split = 20:
        F1 = (0.3710, 0.3025)
    4) min_sample_leaf = 80:
        F1 = (0.3246, 0.2811)
    5) min_sample_leaf = 10:
        F1  =(0.6714, 0.3744)
    6) max_features = 'sqrt':
        F1 = (0.6711, 0.3766)


Average of multiple models:
    model4: 16/25 images for training
        F1 = (0.6711, 0.3748)
    model5: 16/25 images, but contains those images that were not used in training of model4
        F1 = (0.6717, 0.3644)
    model6: 16/25 mixed images again
        F1 = (0.6776, 0.3756)

    model4 + model5 give SCORE = 0.07961


ann-2 (200,200,0.5,0.5):
    F1 = (0.1172, 0.1171)        quite biased. need to expand the model

xgb-1:
    'colsample_bytree': 0.7,
    'silent': 1,
    'subsample': 0.7,
    'learning_rate': 0.3,
    'objective': 'binary:logistic',
    'max_depth': 10,
    'min_child_weight': 10,
    'booster': 'gbtree',
    'eval_metric': 'logloss'
        F1 = (0.1851, 0.1616)       SCORE = 0.07851

xgb-2:
    min_child_weight=100
        F1 = (0.2072, 0.1831)       SCORE = 0.07731

ann-2 (400,200,100, 0.4, 0.4, 0.2) ImageSet2
    F1 = (0.2081, 0.2065)           SCORE = 0.07430