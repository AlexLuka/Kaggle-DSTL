import os
import pickle
import numpy as np
import pandas as pd
import tifffile as tif
import matplotlib.pyplot as plt

from scipy import misc, ndimage
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


np.set_printoptions(threshold=np.nan)
plt.style.use('ggplot')


TRAIN_IMAGE_NAMES1 = ['6010_1_2', '6010_4_2', '6010_4_4', '6040_1_0', '6040_2_2', '6040_4_4',
                      '6060_2_3', '6070_2_3', '6100_1_3', '6100_2_2', '6100_2_3', '6110_1_2',
                      '6110_4_0', '6120_2_0', '6120_2_2', '6140_1_2', '6140_3_1', '6160_2_1']

TRAIN_IMAGE_NAMES2 = ['6010_4_2', '6040_1_0', '6040_2_2', '6040_4_4', '6060_2_3', '6070_2_3',
                      '6100_2_2', '6100_2_3', '6110_4_0', '6120_2_0', '6120_2_2', '6140_1_2',
                      '6040_1_3', '6090_2_0', '6110_3_1', '6150_2_3', '6170_2_4', '6170_0_4']

TRAIN_IMAGE_NAMES3 = ['6010_1_2', '6010_4_4', '6040_1_0', '6040_1_3', '6040_4_4', '6060_2_3',
                      '6090_2_0', '6100_1_3', '6110_1_2', '6110_3_1', '6140_3_1', '6150_2_3',
                      '6160_2_1', '6170_0_4', '6170_2_4', '6170_4_1', '6120_2_0', '6120_2_2']

MODEL_DIR = 'models'
PREDICTED_MASK_DIR = 'predicted-masks'
MODEL_ID = 'model-rf-3'
MAX_11BIT = 2047.0
MAX_8BIT = 255.0
MASK_DIR = 'masks'
IS_TUNING = False


def get_image_shape(image_name_):
    image_ = tif.imread('../../data/sixteen_band/{}_{}.tif'.format(image_name_, 'M'))[0, :, :]
    h_, w_ = image_.shape
    del image_
    return h_, w_


def collect_data(image_name_, is_train=True):
    # load M-images (MULTISPECTRAL, 11BIT). Size: 8 x H x W
    img_16m = tif.imread('../../data/sixteen_band/{}_M.tif'.format(image_name_))

    h_ = img_16m.shape[1]
    w_ = img_16m.shape[2]

    print 'Height = {},   Width = {} '.format(h_, w_)

    #
    #
    # scale each pixel value to (0, 1) range
    a16m_0 = np.reshape(img_16m[0, :, :], newshape=(w_ * h_,)) / MAX_11BIT      # coastal
    a16m_1 = np.reshape(img_16m[1, :, :], newshape=(w_ * h_,)) / MAX_11BIT      # blue
    a16m_2 = np.reshape(img_16m[2, :, :], newshape=(w_ * h_,)) / MAX_11BIT      # green
    a16m_3 = np.reshape(img_16m[3, :, :], newshape=(w_ * h_,)) / MAX_11BIT      # yellow
    a16m_4 = np.reshape(img_16m[4, :, :], newshape=(w_ * h_,)) / MAX_11BIT      # red
    a16m_5 = np.reshape(img_16m[5, :, :], newshape=(w_ * h_,)) / MAX_11BIT      # red edge
    a16m_6 = np.reshape(img_16m[6, :, :], newshape=(w_ * h_,)) / MAX_11BIT      # NIR-1
    a16m_7 = np.reshape(img_16m[7, :, :], newshape=(w_ * h_,)) / MAX_11BIT      # NIR-2

    # new features
    evi1 = 2.5*np.divide(a16m_7 - a16m_5, a16m_7 + 7.5*a16m_5 - 4.0*a16m_1 + 1)
    evi2 = 2.5*np.divide(a16m_7 - a16m_4, a16m_7 + 7.0*a16m_4 - 4.0*a16m_1 + 0.8)
    f8 = np.divide(1.0*a16m_6 - 1.0*a16m_4 - 0.9, (0.16 - a16m_4)**2.0 + (1.2 - a16m_6)**2.0 + .5)
    f9 = 2.5*np.divide(a16m_6 - a16m_4, a16m_6 + 2.4*a16m_4 + 1)
    f10 = np.divide(a16m_6 ** 2.0 - a16m_4, a16m_6 ** 2.0 + a16m_4)
    f11 = np.divide(a16m_7 ** 2.0 - a16m_4, a16m_7 ** 2.0 + a16m_4)
    f12 = np.divide(a16m_7 ** 3.0 - a16m_4 ** 3., a16m_7 ** 3.0 + a16m_4 ** 3.)

    c1 = np.divide(a16m_1 - a16m_6, a16m_1 + a16m_6)        # BLUE and NIR-1
    c2 = np.divide(a16m_2 - a16m_7, a16m_2 + a16m_7)        # GREEN and NIR-2
    c3 = np.divide(a16m_5 - a16m_6, a16m_5 + a16m_6)        # RED EDGE and NIR-1
    c4 = np.divide(a16m_1 - a16m_5, a16m_1 + a16m_5)        # BLUE and RED EDGE
    c5 = np.divide(a16m_4 - a16m_6, a16m_4 + a16m_6)        # RED and NIR-1
    c6 = np.divide(a16m_1 - a16m_3, a16m_1 + a16m_3)        # BLUE and YELLOW
    c7 = np.divide(a16m_1 - a16m_4, a16m_1 + a16m_4)        # BLUE and RED
    c8 = np.divide(a16m_5 - a16m_7, a16m_5 + a16m_7)        # RED EDGE and NIR-2
    c9 = np.divide(a16m_4 - a16m_7, a16m_4 + a16m_7)        # RED and NIR-2
    c10 = np.divide(a16m_4 - a16m_5, a16m_4 + a16m_5)       # RED and RED EDGE
    c11 = np.divide(a16m_3 - a16m_6, a16m_3 + a16m_6)       # YELLOW and NIR-1

    k1 = np.divide(a16m_1 - a16m_6, a16m_6 + np.multiply(a16m_1, a16m_6))
    k2 = np.divide(a16m_1 - a16m_5, a16m_5 + np.multiply(a16m_1, a16m_5))

    td_ = np.vstack((a16m_0, a16m_1, a16m_2, a16m_3, a16m_4, a16m_5, a16m_6, a16m_7,
                     evi1, evi2,
                     c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11,
                     f8, f9, f10, f11, f12,
                     k1, k2)).T

    if is_train:
        baw_image = ndimage.imread('{}/{}-mask.png'.format(MASK_DIR, image_name_), flatten=True)
        t_ = np.around(np.reshape(baw_image, newshape=(w_ * h_,)) / MAX_8BIT)
        return td_, t_

    return td_


def train_model(x_trn, y_trn, x_cv, y_cv):

    rfm = RandomForestClassifier(n_estimators=40,
                                 criterion='gini',
                                 max_depth=None,
                                 min_samples_split=20,
                                 min_samples_leaf=10,
                                 max_features='sqrt',
                                 n_jobs=4,
                                 random_state=2,
                                 verbose=1)

    rfm.fit(x_trn, y_trn)

    # check the model
    p1 = rfm.predict(x_trn)
    p2 = rfm.predict(x_cv)

    print '======= Single model local score ======='
    print 'Train score: {}'.format(f1_score(y_trn, p1))
    print 'Test score: {}'.format(f1_score(y_cv, p2))

    return rfm


def check_on_train(model_, ss_, train_images_):
    for train_image_ in train_images_:
        train_mask_ = os.path.join(PREDICTED_MASK_DIR,
                                   MODEL_ID,
                                   '{}-train.png'.format(train_image_))

        h_, w_ = get_image_shape(train_image_)

        p_data_ = collect_data(train_image_, is_train=False)
        p_data_ = ss_.transform(p_data_)

        p_res_ = model_.predict(p_data_)

        print 'Summation: ', np.sum(p_res_)

        p_img_ = np.reshape(p_res_, newshape=(h_, w_)) * MAX_8BIT

        misc.toimage(p_img_).save(train_mask_)

    if IS_TUNING:
        exit(0)


if __name__ == '__main__':
    submission_data = pd.read_csv(os.path.join('..', '..', 'data', 'sample_submission.csv'))

    model_name = os.path.join(MODEL_DIR, '{0}'.format(MODEL_ID))

    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    if not os.path.exists(PREDICTED_MASK_DIR):
        os.mkdir(PREDICTED_MASK_DIR)

    if not os.path.exists(os.path.join(PREDICTED_MASK_DIR, MODEL_ID)):
        os.mkdir(os.path.join(PREDICTED_MASK_DIR, MODEL_ID))

    # ================== Collect data ========================================
    dft = pd.read_csv(os.path.join('..', '..', 'data', 'train_wkt_v4.csv'))
    data = []
    target = []

    #
    for train_image in TRAIN_IMAGE_NAMES3:
        x, y = collect_data(train_image)

        data.append(x)
        target.append(y)

    data = np.vstack(data)
    target = np.hstack(target)

    ss = StandardScaler()
    data = ss.fit_transform(data)

    data, target = shuffle(data, target, random_state=np.random.randint(np.iinfo(np.int32).max))

    print 'Total size of training data: {0} | {1}'.format(data.shape, target.shape)
    print '\twith {} positive examples'.format(np.sum(target))

    # ================= Split data to train and cv sets =============================
    x_train, x_test, y_train, y_test = train_test_split(data, target,
                                                        test_size=0.3,
                                                        random_state=65296402)
    #
    #
    del (data, target)

    print 'Train set contains {0} / {1}  positive examples. '.format(np.sum(y_train), x_train.shape[0])

    # ================= Train the model and dump it to disk =========================
    if os.path.exists('{}.model'.format(model_name)) and not IS_TUNING:
        model = pickle.load(open('{}.model'.format(model_name), 'rb'))
        ss = pickle.load(open('{}.scale'.format(model_name), 'rb'))
    else:
        model = train_model(x_train, y_train, x_test, y_test)

        pickle.dump(model, open('{}.model'.format(model_name), 'wb'))
        pickle.dump(ss, open('{}.scale'.format(model_name), 'wb'))

    del (x_train, y_train, x_test, y_test)

    # ================= Make prediction on train images ==============================
    check_on_train(model, ss, dft.ImageId.unique())

    # ================= Predict masks for the test data ==============================
    for test_image in submission_data.ImageId.unique():
        test_mask = os.path.join(PREDICTED_MASK_DIR,
                                 MODEL_ID,
                                 '{}.png'.format(test_image))
        if os.path.exists(test_mask):
            continue

        h, w = get_image_shape(test_image)

        prediction_data = collect_data(test_image, is_train=False)
        prediction_data = ss.transform(prediction_data)

        prediction_res = model.predict(prediction_data)

        print 'Summation: ', np.sum(prediction_res)

        prediction_image = np.reshape(prediction_res, newshape=(h, w)) * MAX_8BIT

        print prediction_image.shape

        misc.toimage(prediction_image).save(test_mask)
