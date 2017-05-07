import numpy as np
import pandas as pd
import tifffile as tif
from scipy import ndimage
import os


"""
INFO:
"""

grid_sizes = pd.read_csv('../../data/grid_sizes.csv')
MAX_8BIT = 255.0
MAX_11BIT = 2047.0
# MAX_14BIT = 16383.0
TEST_DIR = 'test-data'
TRAIN_DIR = 'train-data'
MASK_DIR = 'masks'
# IMG_DIR = 'images'


def collect_data(image_name_, is_train=True):
    if is_train:
        if os.path.exists('{}/{}.csv'.format(TRAIN_DIR, image_name_)):
            return
    else:
        if os.path.exists('{}/{}.csv'.format(TEST_DIR, image_name_)):
            return

    # load M-images (MULTISPECTRAL, 11BIT). Size: 8 x H x W
    img_16m = tif.imread('../../data/sixteen_band/{}_M.tif'.format(image_name_))

    h = img_16m.shape[1]
    w = img_16m.shape[2]

    print 'Height = {},   Width = {} '.format(h, w)

    #
    #
    # replace with flatten()
    a16m_0 = np.reshape(img_16m[0, :, :], newshape=(w * h,)) / MAX_11BIT
    a16m_1 = np.reshape(img_16m[1, :, :], newshape=(w * h,)) / MAX_11BIT
    a16m_2 = np.reshape(img_16m[2, :, :], newshape=(w * h,)) / MAX_11BIT
    a16m_3 = np.reshape(img_16m[3, :, :], newshape=(w * h,)) / MAX_11BIT
    a16m_4 = np.reshape(img_16m[4, :, :], newshape=(w * h,)) / MAX_11BIT
    a16m_5 = np.reshape(img_16m[5, :, :], newshape=(w * h,)) / MAX_11BIT
    a16m_6 = np.reshape(img_16m[6, :, :], newshape=(w * h,)) / MAX_11BIT
    a16m_7 = np.reshape(img_16m[7, :, :], newshape=(w * h,)) / MAX_11BIT

    evi1 = 2.5*np.divide(a16m_7 - a16m_5, a16m_7 + 7.5*a16m_5 - 4.0*a16m_1 + 1)
    ndwi = np.divide(a16m_2 - a16m_7, a16m_2 + a16m_7)
    evi2 = 2.5*np.divide(a16m_7 - a16m_4, a16m_7 + 7.0*a16m_4 - 4.0*a16m_1 + 0.8)
    ndvi = np.divide(a16m_6 - a16m_4, a16m_6 + a16m_4)
    c1 = np.divide(1.0*a16m_6 - 1.0*a16m_4 - 0.9, (0.16 - a16m_4)**2.0 + (1.2 - a16m_6)**2.0 + .5)
    c2 = 2.5*np.divide(a16m_6 - a16m_4, a16m_6 + 2.4*a16m_4 + 1)
    c3 = np.divide(a16m_6**2.0 - a16m_4, a16m_6**2.0 + a16m_4)
    c4 = np.divide(a16m_7 ** 2.0 - a16m_4, a16m_7 ** 2.0 + a16m_4)
    c5 = np.divide(a16m_7 ** 3.0 - a16m_4**3., a16m_7 ** 3.0 + a16m_4**3.)
    rei = np.divide(a16m_6 - a16m_1, a16m_6 + np.multiply(a16m_6, a16m_1))
    bai = np.divide(a16m_1 - a16m_6, a16m_1 + a16m_6)

    train_data = np.vstack((evi1,
                            evi2,
                            ndvi,
                            ndwi,
                            c1,
                            c2,
                            c3,
                            c4,
                            c5,
                            rei,
                            bai)).T

    df = pd.DataFrame(train_data, columns=['evi1',
                                           'evi2',
                                           'ndvi',
                                           'ndwi',
                                           'c1',
                                           'c2',
                                           'c3',
                                           'c4',
                                           'c5',
                                           'rei',
                                           'bai'])
    if is_train:
        baw_image = ndimage.imread('{}/{}-mask.png'.format(MASK_DIR, image_name_), flatten=True)
        target = np.around(np.reshape(baw_image, newshape=(w * h,)) / MAX_8BIT)
        df['target'] = target
        df.to_csv('{}/{}.csv'.format(TRAIN_DIR, image_name_), index=False)
        return

    df.to_csv('{}/{}.csv'.format(TEST_DIR, image_name_), index=False)


if __name__ == '__main__':

    if not os.path.exists(TEST_DIR):
        os.mkdir(TEST_DIR)

    if not os.path.exists(TRAIN_DIR):
        os.mkdir(TRAIN_DIR)

    # Collect train data
    dft = pd.read_csv('../../data/train_wkt_v4.csv')

    for train_image in dft.ImageId.unique():
        collect_data(train_image)

    # Collect test data
    dft = pd.read_csv('../../data/sample_submission.csv')

    for test_image in dft.ImageId.unique():
        print 'Image: ', test_image
        collect_data(test_image, is_train=False)
