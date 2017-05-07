import os
import pandas as pd
import numpy as np
from numpy import matlib as mtl
import tifffile as tif
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef
from scipy import misc
np.set_printoptions(threshold=np.nan)


# Constants
N_CLUSTER = 6
CLUSTER_DISTANCE_CUT = 100.0
TRAIN_IMAGE_NAMES = ['6010_1_2', '6010_4_4',
                     '6040_4_4', '6060_2_3', '6070_2_3', '6100_2_2', '6100_2_3',
                     '6110_1_2', '6110_3_1', '6140_1_2',
                     '6140_3_1', '6150_2_3', '6160_2_1', '6170_0_4', '6170_2_4']
# THE_CHOSEN_ONES = ['6110_4_0', '6140_1_2', '6140_3_1']

TRAIN_DIR = 'train-data'
TEST_DIR = 'test-data'
MASK_DIR = 'masks'
TEMP_DIR = 'temp-files'
PREDICTED_DIR = 'predicted-knn-1'
IMG_DIR = 'images'


# get the test data generated from 16P, 16M and 3RGB images
def get_test_data(image_name_):
    return pd.read_csv('{}/{}.csv'.format(TEST_DIR, image_name_)).values


def get_train_data(image_name_):
    dt = pd.read_csv('{}/{}.csv'.format(TRAIN_DIR, image_name_))
    target_ = dt['target'].values
    data_ = dt.drop(['target'], axis=1).values
    return data_, target_


def get_image_shape(image_name_):
    image_ = tif.imread('../../data/sixteen_band/{}_{}.tif'.format(image_name_, 'M'))[0, :, :]
    h, w = image_.shape
    del image_
    return h, w


def train_knn(image_name_):
    # get the data
    train_data_, target_ = get_train_data(image_name_)

    ss = StandardScaler()
    train_data_ = ss.fit_transform(train_data_)

    # train K-Means algorithm
    km = KMeans(n_clusters=N_CLUSTER, max_iter=500, n_init=50)
    km.fit(train_data_)

    labs_ = km.labels_

    #
    # img_mask = ndimage.imread('{}/{}-16m-mask.png'.format(MASK_DIR, image_name_), flatten=True)
    # h, w = img_mask.shape
    # del img_mask
    h, w = get_image_shape(image_name_)

    # target_true = np.reshape(img_black, newshape=(img_black.shape[0] * img_black.shape[1], 1)) / 256.0
    # target_true = np.around(target_true)
    # print np.sum(target_true)

    mcc = 0
    p_res = []
    cln = -1

    for i_ in range(N_CLUSTER):
        ind = np.where(labs_ == i_)[0]       # return indices
        p_tot = np.zeros(shape=(h * w, 1))
        p_tot[ind] = 1

        mcc_ = matthews_corrcoef(target_, p_tot)

        print 'F1 train {}: {},    ACC: {},   MCC: {}'.format(i_,
                                                              f1_score(target_, p_tot, pos_label=1, average='binary'),
                                                              accuracy_score(target_, p_tot),
                                                              mcc_)

        if mcc_ > mcc:
            mcc = mcc_
            cln = i_
            p_res = p_tot

    predicted_image = np.reshape(p_res, newshape=(h, w)) * 256
    misc.toimage(predicted_image).save('{}/{}-predicted-train.png'.format(TEMP_DIR, image_name_))

    return km, cln, ss, mcc


def apply_knn(model_=None, image_name_=None, num=0):
    # kNN model and cluster's number
    km_ = model_[0]
    cln_ = model_[1]
    ss_ = model_[2]

    # get test data
    test_data = get_test_data(image_name_)
    test_data = ss_.transform(test_data)

    # predict label for each data entry (each pixel)
    labels_ = km_.predict(test_data)

    # get original image size (M-Image)
    h, w = get_image_shape(image_name_)

    # predicted values
    predicted = np.zeros(shape=(h * w, 1))
    ind = np.where(labels_ == cln_)
    predicted[ind] = 1

    # calculate average distance to the cluster's center
    cm = km_.cluster_centers_[cln_]
    test_data = test_data[ind, :]
    cmm = mtl.repmat(cm, test_data.shape[0], 1)
    test_data = np.subtract(test_data, cmm)         # centering
    test_data = np.power(test_data, 2)              # power
    test_data = np.sum(test_data, axis=1)           # sum squares
    test_data = np.sqrt(test_data)                  # element wise square root
    mean_distance = np.mean(test_data)

    del test_data, cmm

    print 'Model {}:  frac {}    score = {}'.format(num,
                                                    float(np.sum(predicted)) / float(len(predicted)),
                                                    mean_distance)

    predicted_image = np.reshape(predicted, newshape=(h, w)) * 256
    misc.toimage(predicted_image).save('{}/{}-{}.png'.format(PREDICTED_DIR, image_name_, num))

    # if not os.path.exists('{}/{}.png'.format(IMG_DIR, image_name_)):
    #     misc.toimage(img_16m[:, :, 4]).save('{}/{}.png'.format(IMG_DIR, image_name_))

    return predicted, mean_distance                  # float(np.sum(pred)) / float(len(pred))


if __name__ == '__main__':

    if not os.path.exists(TEMP_DIR):
        os.mkdir(TEMP_DIR)

    if not os.path.exists(PREDICTED_DIR):
        os.mkdir(PREDICTED_DIR)

    # Collect KNN models
    models = []

    # for image_name in THE_CHOSEN_ONES:
    for image_name in TRAIN_IMAGE_NAMES:
        # for image_name in ['6070_2_3']:
        print '=' * 50
        print 'Train model: {}'.format(image_name)
        models.append(train_knn(image_name))

    # print models

    submission_data = pd.read_csv('../../data/sample_submission.csv')

    # load the list of test images
    for test_image_name in submission_data.ImageId.unique():
        print '=' * 50
        print 'Image {}'.format(test_image_name)

        thresh = 0
        predictions = []

        for k, knn_model in enumerate(models):
            if knn_model[3] > 0.2:
                p, f = apply_knn(model_=knn_model, image_name_=test_image_name, num=k)

                if f < CLUSTER_DISTANCE_CUT:
                    print 'Threshold {}'.format(f)
                    predictions.append(p)

        height, width = get_image_shape(test_image_name)

        # img_16m = tif.imread('../../data/sixteen_band/{}_M.tif'.format(test_image_name))
        # img_16m = np.swapaxes(img_16m, 0, 2)

        prediction_final = np.zeros(shape=(width * height, 1))
        for p in predictions:
            prediction_final = np.logical_or(prediction_final, p.astype(np.bool))

        prediction_image = np.reshape(prediction_final, newshape=(height, width)) * 256
        misc.toimage(prediction_image).save('{}/{}-final.png'.format(PREDICTED_DIR, test_image_name))
