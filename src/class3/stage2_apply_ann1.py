# import os
# import numpy as np
# import pandas as pd
# import tifffile as tif
# from scipy import misc
# from keras.models import Sequential, load_model
# from keras.layers import Dense, Dropout
# from keras.callbacks import EarlyStopping
# from keras.layers.normalization import BatchNormalization
# from keras.layers.advanced_activations import PReLU
#
# from sklearn.utils import shuffle
# from sklearn.metrics import f1_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import KFold
#
# import matplotlib.pyplot as plt
#
#
# np.set_printoptions(threshold=np.nan)
# plt.style.use('ggplot')
#
#
# TRAIN_DIR = 'train-data'
# TEST_DIR = 'test-data'
# # TRAIN_IMAGE_NAMES = ['6010_1_2', '6010_4_2', '6010_4_4', '6040_1_0', '6040_1_3', '6040_2_2',
# #                      '6040_4_4', '6060_2_3', '6070_2_3', '6100_1_3', '6100_2_2', '6100_2_3',
# #                      '6110_4_0', '6120_2_0', '6120_2_2', '6140_1_2', '6140_3_1', '6150_2_3',
# #                      '6160_2_1', '6170_0_4', '6170_2_4']
# N_FOLDS = 10
#
# MODEL_DIR = 'ann-models'
# PREDICTED_MASK_DIR = 'predicted-masks-2'
# MODEL_ID = 'model-1'
#
#
# # ====================== ANN model ===============================
# def nn_model(input_size,
#              n1, n2,
#              dropout1=0.4,
#              dropout2=0.4):
#     model_ = Sequential()
#
#     model_.add(Dense(n1, input_dim=input_size, init='he_normal'))
#     model_.add(PReLU())
#     model_.add(BatchNormalization())
#     model_.add(Dropout(dropout1))
#
#     model_.add(Dense(n2, init='he_normal'))
#     model_.add(PReLU())
#     model_.add(BatchNormalization())
#     model_.add(Dropout(dropout2))
#
#     model_.add(Dense(1, init='normal', activation='sigmoid'))
#
#     model_.compile(loss='binary_crossentropy', optimizer='adadelta')
#     return model_
#
#
# # get the test data generated from 16P, 16M and 3RGB images
# def get_test_data(image_name_):
#     return pd.read_csv('{}/{}.csv'.format(TEST_DIR, image_name_)).values
#
#
# def get_train_data(image_name_):
#     dt = pd.read_csv('{}/{}.csv'.format(TRAIN_DIR, image_name_))
#     target_ = dt['target'].values
#     data_ = dt.drop(['target'], axis=1).values
#     return data_, target_
#
#
# def get_image_shape(image_name_):
#     image_ = tif.imread('../../data/sixteen_band/{}_{}.tif'.format(image_name_, 'M'))[0, :, :]
#     h_, w_ = image_.shape
#     del image_
#     return h_, w_
#
#
# def train_ann(x_, y_):
#
#     # K-folds
#     kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=np.random.randint(np.iinfo(np.int32).max))
#
#     n = 0
#     models_ = []
#
#     for train_index, test_index in kf.split(x):
#         model_name = os.path.join(MODEL_DIR, MODEL_ID, 'model-{}.h5'.format(n))
#         # if os.path.exists(model_name):
#         #
#         #     # check the model
#         #     p1 = m.predict_classes(x_trn, batch_size=1000, verbose=0)
#         #     p2 = m.predict_classes(x_tst, batch_size=1000, verbose=0)
#         #     p3 = m.predict_classes(x_cv_, batch_size=1000, verbose=0)
#         #
#         #     print '======= Model {} local score ======='.format(n)
#         #     print 'Train score: {}'.format(f1_score(y_trn, p1))
#         #     print 'Test score: {}'.format(f1_score(y_tst, p2))
#         #     print 'CV score: {}'.format(f1_score(y_cv_, p3))
#         #
#         #     continue
#
#         x_trn = x_[train_index]
#         x_cv = x_[test_index]
#
#         y_trn = y_[train_index]
#         y_cv = y_[test_index]
#
#         if not os.path.exists(model_name):
#             m = nn_model(x_trn.shape[1],
#                          200, 200,
#                          dropout1=0.1,
#                          dropout2=0.1)
#
#             early_stopping = EarlyStopping(monitor='val_loss', patience=15)
#             m.fit(x_trn, y_trn,
#                   batch_size=256,
#                   nb_epoch=1000000,
#                   verbose=2,
#                   shuffle=True,
#                   validation_data=(x_cv, y_cv),
#                   callbacks=[early_stopping])
#
#             # save the model
#             m.save(model_name)
#         else:
#             m = load_model(model_name)
#
#         models_.append(m)
#         n += 1
#
#         # check the model
#         p1 = m.predict_classes(x_trn, batch_size=1000, verbose=0)
#         p2 = m.predict_classes(x_cv, batch_size=1000, verbose=0)
#         # p3 = m.predict_classes(x_tst, batch_size=1000, verbose=0)
#
#         print '======= Model {} local score ======='.format(n)
#         print 'Train score: {}'.format(f1_score(y_trn, p1))
#         print 'Test score: {}'.format(f1_score(y_cv, p2))
#         # print 'CV score: {}'.format(f1_score(y_tst, p3))
#
#     return models_
#
#
# if __name__ == '__main__':
#     submission_data = pd.read_csv('../../data/sample_submission.csv')
#
#     if not os.path.exists(MODEL_DIR):
#         os.mkdir(MODEL_DIR)
#
#     if not os.path.exists(PREDICTED_MASK_DIR):
#         os.mkdir(PREDICTED_MASK_DIR)
#
#     if not os.path.exists(os.path.join(MODEL_DIR, MODEL_ID)):
#         os.mkdir(os.path.join(MODEL_DIR, MODEL_ID))
#
#     if not os.path.exists(os.path.join(PREDICTED_MASK_DIR, MODEL_ID)):
#         os.mkdir(os.path.join(PREDICTED_MASK_DIR, MODEL_ID))
#
#     # ================== Collect the data ==================
#     dft = pd.read_csv('../../data/train_wkt_v4.csv')
#     data = []
#     target = []
#
#     for train_image in dft.ImageId.unique():
#         x, y = get_train_data(train_image)
#
#         data.append(x)
#         target.append(y)
#     data = np.vstack(data)
#     target = np.hstack(target).astype(np.uint8)
#
#     ss = StandardScaler()
#     data = ss.fit_transform(data)
#     data, target = shuffle(data, target, random_state=np.random.randint(np.iinfo(np.int32).max))
#
#     print 'Total size of training data: {} | {}'.format(data.shape, target.shape)
#     print '\twith {} positive examples'.format(np.sum(target))
#
#     # ================= Split data to train, test and cv sets ========================
#     # x_train, x_test, y_train, y_test = train_test_split(data, target,
#     #                                                    test_size=0.3,
#     #                                                    random_state=65296402)
#     #
#     #
#     # del (data, target, dft)
#
#     print 'Train set contains {} / {}  positive examples. '.format(np.sum(target), data.shape[0])
#     # print 'Test set contains {} / {}  positive examples. '.format(np.sum(y_test), x_test.shape[0])
#
#     # ================= Train the model ==============================================
#
#     models = train_ann(data, target)
#
#     for test_image in submission_data.ImageId.unique():
#         test_mask = os.path.join(PREDICTED_MASK_DIR,
#                                  MODEL_ID,
#                                  '{}-{}.png'.format(test_image, MODEL_ID))
#         if os.path.exists(test_mask):
#             continue
#
#         h, w = get_image_shape(test_image)
#
#         prediction_data = get_test_data(test_image)
#         prediction_data = ss.transform(prediction_data)
#
#         prediction_res = np.zeros(shape=(h * w, 1))
#         prediction_prob = np.zeros(shape=(h * w, 1))
#
#         for model in models:
#             # prediction_res += model.predict_classes(prediction_data, batch_size=1000, verbose=0)
#             prediction_prob = model.predict_proba(prediction_data, batch_size=1000, verbose=0)
#
#             # reduce probability cutoff
#             p = np.zeros(shape=(h * w, 1))
#             p[np.where(prediction_prob > 0.7)[0]] = 1
#
#             # test logical OR and logical AND
#             prediction_res = np.logical_or(prediction_res, p.astype(np.bool))
#
#         print 'Summation: ', np.sum(prediction_res)
#
#         prediction_image = np.reshape(prediction_res, newshape=(h, w)) * 255
#
#         print prediction_image.shape
#
#         # misc.toimage(img_16m).save('predicted-ann/{}-actual.png'.format(prediction_image_name))
#         misc.toimage(prediction_image).save(test_mask)
