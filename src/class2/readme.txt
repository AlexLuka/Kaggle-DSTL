Random Forest:
        n_estimators=40,
        criterion='gini',
        max_depth=None,
        min_samples_split=20,
        min_samples_leaf=20,
        max_features='auto',
        n_jobs=1,
        random_state=2,
        verbose=1,
        class_weight='balanced'

                                F1 = (0.4378, 0.1617)               SCORE = Non noded intrsection

    min_sample_split = 20:      F1 = (0.4393, 0.1638)

    n_estimators = 40:          F1 = (0.5665, 0.1529)       (0.43, 0.44)        rf-3 image-set-1

    min_samples_leaf = 50:      F1 = (0.2165, 0.1291)
