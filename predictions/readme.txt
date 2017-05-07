SAMPLE SUBMISSION = 0.07524

submission number, score
Submission 60, class-3: SCORE = 0.15137. This is without trick!
Submission 79-class1,	0.12128
Submission 68-class3,	0.14571
Submission 81	Evaluation Exception: For image 6150_3_4, class 5, there's an exception in geometry 
		NetTopologySuite.Geometries.TopologyException: side location conflict 
		[ (3.2509062009291477E-05, -0.00086449093799070853, NaN) ] at 
		Kaggle.Metrics.Custom.JaccardDSTL.compareMultipolygon(IGeometry predMultipoly, IGeometry truthMultipoly) at 
		Kaggle.Metrics.Custom.JaccardDSTL.calculateIoUVector(Dictionary`2 submission, Dictionary`2 solution).

		There was a reduced presicion: 6 digits
Submission 82- class 5, Random Forest: SCORE = TOO LARGE FILE: 863.2 Mb
					Reduced presicion to 6 digits:
					Send zip on Chrome: time out error
Submission 83- class 4, Random Forest: SCORE = 0.07520
Submission 84- class 6, Random Forest, area threshold=100: SCORE
Submission 85-class9; RF => score = 0.07856
Submission 86-class9: RF balanced => SCORE = 0.10275
Submission 87-classes9,10 (identical predictions, RF balanced, model-2-rf): SCORE = 0.10275
Submission 88-classes9,10 (differentiated large and small cars by area: small<3.0 and >8.0(parkings)): SCORE=0.07693 TOO BAD
Submission 89-classes9,10 (model-3-rf):	SCORE = 0.10175
			model:	n_estimators=40,
                                 criterion='gini',
                                 max_depth=None,
                                 min_samples_split=20,
                                 min_samples_leaf=12,
                                 max_features='auto',
                                 n_jobs=1,
                                 random_state=2,
                                 verbose=1,
                                 class_weight='balanced'

Submission 90-classes9,10 (model-1-xgboost): SCORE = 0.09483
		'colsample_bytree': 0.7,
           	'silent': 1,
            	'subsample': 0.7,
            	'learning_rate': 0.03,
            	'objective': 'binary:logistic',
            	'max_depth': 10,
            	'min_child_weight': 10,
            	'booster': 'gbtree',
            	'eval_metric': 'logloss'
Submission 91-class 3: SCORE=0.15624

Submission 94-class 1: SCORE= 0.11590







