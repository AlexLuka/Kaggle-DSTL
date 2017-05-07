Model1:
ANN (200,200,0.1,0.1), 10 Fold, logical OR mask prediction, threshold=0.5, area>25
    SCORE:          0.15137
    SCORE + trick:  0.15160

    Submission 67:  logical_OR, threshold=0.7, 10 Fold, no area restrictions => SCORE = 0.14258
    Submission 68:  logical_OR, threshold=0.7, 10 Fold, area > 50            => SCORE = 0.14571


Model2:
ANN(300,200,0.2,0.1), 10 Fold, logical AND, threshold=0.5
    F1-scores (train, cv, test)
    Fold1:      0.564, 0.529, 0.567
    Fold2:      0.585, 0.579, 0.588
    Fold3:      0.596, 0.587, 0.596
    Fold4:      0.550, 0.523, 0.549
    Fold5:      0.562, 0.501, 0.560
    Fold6:      0.596, 0.553, 0.593
    Fold7:      0.580, 0.573, 0.580
    Fold8:      0.559, 0.564, 0.566
    AND doesn't work, OR works, but produce the same result


Model3:
ANN(100,100,0.1,0.1), 5 Fold, logical OR, threshold=0.5
    F1-scores (train, cv, test)
    Fold1:      0.569, 0.591
    Fold2:      0.573, 0.539
    Fold3:      0.572, 0.574
    Fold4:      0.585, 0.582
    Fold5:      0.592, 0.566
    Submission 63: area restriction > 25 pixels   SCORE:    0.14854
    Submission 65: area restriction > 25 pixels,
                   moment > 5,
                   threshold = 0.6                SCORE:    0.14541


Model rf-1:
    rfm = RandomForestClassifier(n_estimators=40,
                                 criterion='gini',
                                 max_depth=None,
                                 min_samples_split=20,
                                 min_samples_leaf=12,
                                 max_features='auto',
                                 n_jobs=1,
                                 random_state=2,
                                 verbose=1,
                                 class_weight='balanced')
        SCORE = 0.15624

