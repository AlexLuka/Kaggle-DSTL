import os
import numpy as np
import pandas as pd
import tifffile as tif
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, KFold
from scipy import misc, ndimage


np.set_printoptions(threshold=np.nan)


N_FOLDS = 10
MODEL_DIR = os.path.join('models', 'ann-models')
PREDICTED_MASK_DIR = 'predicted-masks'
MODEL_ID = 'model-2'
grid_sizes = pd.read_csv('../../data/grid_sizes.csv')
MAX_8BIT = 255.0
MAX_11BIT = 2047.0
MASK_DIR = 'masks'
THRESHOLD = 0.5


# ====================== ANN model ===============================
def nn_model(input_size,
             n1, n2,
             dropout1=0.4,
             dropout2=0.4):
    model_ = Sequential()

    model_.add(Dense(n1, input_dim=input_size, init='he_normal'))
    model_.add(PReLU())
    model_.add(BatchNormalization())
    model_.add(Dropout(dropout1))

    model_.add(Dense(n2, init='he_normal'))
    model_.add(PReLU())
    model_.add(BatchNormalization())
    model_.add(Dropout(dropout2))

    model_.add(Dense(1, init='normal', activation='sigmoid'))

    model_.compile(loss='binary_crossentropy', optimizer='adadelta')
    return model_


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

    #
    #
    # Actually can use flatten() method here
    a16m_0 = np.reshape(img_16m[0, :, :], newshape=(w_ * h_,)) / MAX_11BIT
    a16m_1 = np.reshape(img_16m[1, :, :], newshape=(w_ * h_,)) / MAX_11BIT
    a16m_2 = np.reshape(img_16m[2, :, :], newshape=(w_ * h_,)) / MAX_11BIT
    a16m_3 = np.reshape(img_16m[3, :, :], newshape=(w_ * h_,)) / MAX_11BIT
    a16m_4 = np.reshape(img_16m[4, :, :], newshape=(w_ * h_,)) / MAX_11BIT
    a16m_5 = np.reshape(img_16m[5, :, :], newshape=(w_ * h_,)) / MAX_11BIT
    a16m_6 = np.reshape(img_16m[6, :, :], newshape=(w_ * h_,)) / MAX_11BIT
    a16m_7 = np.reshape(img_16m[7, :, :], newshape=(w_ * h_,)) / MAX_11BIT

    evi = 2.5*np.divide(a16m_7 - a16m_5, a16m_7 + 7.5*a16m_5 - 4.0*a16m_1 + 1)

    ndwi = np.divide(a16m_2 - a16m_7, a16m_2 + a16m_7)

    # in early versions I used only original features from the M-images and two
    # additional coefficients: EVI and NDWI. Later, I applied the same principle
    # of coefficient building in order to derrive new features.
    td_ = np.vstack((a16m_0,
                     a16m_1,
                     a16m_2,
                     a16m_3,
                     a16m_4,
                     a16m_5,
                     a16m_6,
                     a16m_7,
                     evi,
                     ndwi)).T

    if is_train:
        baw_image = ndimage.imread('{}/{}-mask.png'.format(MASK_DIR, image_name_), flatten=True)
        t_ = np.around(np.reshape(baw_image, newshape=(w_ * h_,)) / MAX_8BIT)
        return td_, t_

    return td_


def train_ann(x_, y_, x_tst, y_tst):

    # K-folds
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=np.random.randint(np.iinfo(np.int32).max))

    n = 0
    models_ = []

    for train_index, test_index in kf.split(x):
        model_name = '{}/model-{}.h5'.format(MODEL_DIR, n)

        x_trn = x_[train_index]
        x_cv = x_[test_index]

        y_trn = y_[train_index]
        y_cv = y_[test_index]

        if not os.path.exists(model_name):
            m = nn_model(x_train.shape[1],
                         200, 200,
                         dropout1=0.1,
                         dropout2=0.1)

            early_stopping = EarlyStopping(monitor='val_loss', patience=15)
            m.fit(x_trn, y_trn,
                  batch_size=256,
                  nb_epoch=1000000,
                  verbose=2,
                  shuffle=True,
                  validation_data=(x_cv, y_cv),
                  callbacks=[early_stopping])

            # save the model
            m.save(model_name)
        else:
            m = load_model(model_name)

        models_.append(m)
        n += 1

        # check the model
        p1 = m.predict_classes(x_trn, batch_size=1000, verbose=0)
        p2 = m.predict_classes(x_cv, batch_size=1000, verbose=0)
        p3 = m.predict_classes(x_tst, batch_size=1000, verbose=0)

        print '======= Model {} local score ======='.format(n)
        print 'Train score: {}'.format(f1_score(y_trn, p1))
        print 'Test score: {}'.format(f1_score(y_cv, p2))
        print 'CV score: {}'.format(f1_score(y_tst, p3))

    return models_


if __name__ == '__main__':
    submission_data = pd.read_csv('../../data/sample_submission.csv')

    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    if not os.path.exists(PREDICTED_MASK_DIR):
        os.mkdir(PREDICTED_MASK_DIR)

    if not os.path.exists(os.path.join(MODEL_DIR, MODEL_ID)):
        os.mkdir(os.path.join(MODEL_DIR, MODEL_ID))

    if not os.path.exists(os.path.join(PREDICTED_MASK_DIR, MODEL_ID)):
        os.mkdir(os.path.join(PREDICTED_MASK_DIR, MODEL_ID))

    # ================== Collect the data ==================
    dft = pd.read_csv('../../data/train_wkt_v4.csv')
    data = []
    target = []

    for train_image in dft.ImageId.unique():
        x, y = collect_data(train_image, is_train=True)

        data.append(x)
        target.append(y)
    data = np.vstack(data)
    target = np.hstack(target).astype(np.uint8)

    ss = StandardScaler()
    data = ss.fit_transform(data)

    # probably it will be impossible to reproduce the result with 100% equality, because
    # I used random seed to shuffle the data. However, I wasn't able to overfit the network
    # and I expect that the result obtained with that architecture will be very close to mine.
    data, target = shuffle(data, target, random_state=np.random.randint(np.iinfo(np.int32).max))

    print 'Total size of training data: {} | {}'.format(data.shape, target.shape)
    print '\twith {} positive examples'.format(np.sum(target))

    # ================= Split data to train, test and cv sets ========================
    x_train, x_test, y_train, y_test = train_test_split(data, target,
                                                        test_size=0.3,
                                                        random_state=65296402)

    del (data, target, dft)

    print 'Train set contains {} / {}  positive examples. '.format(np.sum(y_train), x_train.shape[0])
    print 'Test set contains {} / {}  positive examples. '.format(np.sum(y_test), x_test.shape[0])

    # ================= Train the model ==============================================
    models = train_ann(x_train, y_train, x_test, y_test)

    # ================= Generate black-and-white prediction masks ====================
    for test_image in submission_data.ImageId.unique():
        test_mask = '{}/{}.png'.format(PREDICTED_MASK_DIR, MODEL_ID)
        if os.path.exists(test_mask):
            continue

        h, w = get_image_shape(test_image)

        prediction_data = collect_data(test_image, is_train=False)
        prediction_data = ss.transform(prediction_data)

        prediction_res = np.zeros(shape=(h * w, 1))
        prediction_prob = np.zeros(shape=(h * w, 1))

        for model in models:
            # here one can use predict_classes() method. I used predict_proba() in order to
            # manipulate with the sigmoid threshold value
            prediction_prob = model.predict_proba(prediction_data, batch_size=1000, verbose=0)

            #
            p = np.zeros(shape=(h * w, 1))
            p[np.where(prediction_prob > THRESHOLD)[0]] = 1

            # Here we don't average over the k-folds, but use logical_or operator:
            #   if the pixel was predicted as positive at least once, consider it as a positive example
            # Actually, later I figured out that logical_and works much better!!!
            # Basically because ANN can easily identify the water due to its distinguishable spectral
            # signal, but most of the errors comes from the incorrectly identified objects (noise).
            # Logical_and method fights with a noise quite successfully.
            prediction_res = np.logical_or(prediction_res, p.astype(np.bool))

        #
        prediction_image = np.reshape(prediction_res, newshape=(h, w)) * 255

        #
        misc.toimage(prediction_image).save(test_mask)
