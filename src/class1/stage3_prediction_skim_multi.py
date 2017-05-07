import os
import numpy as np
import pandas as pd
import tifffile as tif

from tqdm import tqdm
from scipy import ndimage
from skimage import measure
from shapely.ops import transform, unary_union
from shapely.geometry import Polygon, MultiPolygon, Point
from matplotlib.patches import Polygon as Pol

import matplotlib.pyplot as plt


SUBMISSION = 125
WANNA_PLOT = True
MODELS = ['model-rf-1', 'model-rf-2', 'model-rf-3', 'model-xgb-1', 'model-xgb-2', 'model-xgb-3', 'model-ann-1']
MODEL_ID = 'complex-3'


def get_image_shape(image_name_):
    image_ = tif.imread(os.path.join('..', '..', 'data', 'sixteen_band', '{}_M.tif'.format(image_name_)))[0, :, :]
    h_, w_ = image_.shape
    del image_
    return h_, w_


IMAGES_TO_PLOT = ['6020_0_4', '6080_0_2', '6070_2_3', '6100_0_3', '6100_1_1', '6130_0_4', '6120_3_2']


if __name__ == '__main__':
    submission_data = pd.read_csv(os.path.join('..', '..', 'predictions', 'empty_submission.csv'))
    grid_sizes = pd.read_csv(os.path.join('..', '..', 'data', 'grid_sizes.csv'))

    if WANNA_PLOT and not os.path.exists('submission-{}'.format(SUBMISSION)):
        os.mkdir('submission-{}'.format(SUBMISSION))

    if WANNA_PLOT:
        plt.figure(figsize=(10, 10))

    if not os.path.exists(os.path.join('predicted-masks', MODEL_ID)):
        os.mkdir(os.path.join('predicted-masks', MODEL_ID))

    for test_image_name in tqdm(submission_data.ImageId.unique()):

        # since we are interested only in pixels, which are predicted to be building by all models
        # we can just multiply element wise all predicted masks
        img_ = np.ones(shape=(get_image_shape(test_image_name)))

        for model in MODELS:
            img_mask = ndimage.imread('predicted-masks/{1}/{0}.png'.format(test_image_name, model),
                                      flatten=True)
            img_ = np.multiply(img_, img_mask)

        # Not sure if this step gives any advantages. I tried models with and without it. But
        # forgot to record which model produce better result.
        img_ = ndimage.binary_fill_holes(img_)

        # frame of zeros around a mask
        img_[0, :] = 0
        img_[-1, :] = 0
        img_[:, 0] = 0
        img_[:, -1] = 0

        # scale coefficients
        x_max = grid_sizes[grid_sizes.IMG == test_image_name].Xmax.values[0]
        y_min = grid_sizes[grid_sizes.IMG == test_image_name].Ymin.values[0]

        x_scale = np.float(img_.shape[1] ** 2) / np.float((img_.shape[1] + 1) * x_max)
        y_scale = np.float(img_.shape[0] ** 2) / np.float((img_.shape[0] + 1) * y_min)

        # find contours
        contours = measure.find_contours(img_.T, 0.2)

        polygons = []

        for contour in contours:

            if contour.shape[0] > 2:        # contour must have at least 3 points

                poly = Polygon(contour)

                # exclude small buildings (they are just a noise) and too large buildings,
                # which are most probably incorrectly classified regions of pixels
                if poly.is_valid and 5.0 < poly.area < 2500.0:
                    polygons.append(poly)

        m_pol_vec = MultiPolygon(polygons)

        if WANNA_PLOT and test_image_name in IMAGES_TO_PLOT:
            # I used M-images for object recognition, but in order to confirm the result
            # I plot the identified objects on top of RGB images. The have different scale, and
            # if everything works, interested us objects can be visually identified without any problems.
            img_orig = tif.imread(os.path.join('..', '..', 'data', 'three_band', '{}.tif'.
                                               format(test_image_name)))[0, :, :]
            x_scale2 = np.float(img_orig.shape[1] ** 2) / np.float((img_orig.shape[1] + 1) * x_max)
            y_scale2 = np.float(img_orig.shape[0] ** 2) / np.float((img_orig.shape[0] + 1) * y_min)

            plt.imshow(img_orig)

            ax = plt.gca()

            for p in m_pol_vec.geoms:
                p = transform(lambda x, y: (x*x_scale2/x_scale, y*y_scale2/y_scale), p)
                pol = Pol(np.array(p.exterior), color='r', alpha=0.35)
                ax.add_patch(pol)

                for inter in p.interiors:
                    pol = Pol(np.array(inter), color='y', alpha=0.3)
                    ax.add_patch(pol)

            plt.title(test_image_name)

            plt.tight_layout()
            plt.savefig(os.path.join('submission-{}', '{}.png'.format(SUBMISSION, test_image_name)),
                        format='png',
                        dpi=500)

            # plt.show()

            plt.clf()
            plt.cla()

        m_pol_uu = unary_union(m_pol_vec)

        m_pol_uu = transform(lambda x, y: (np.round(float(x) / x_scale, 8), np.round(float(y) / y_scale, 8)),
                             m_pol_uu)

        # Bug fix, remove one bad polygon!!! For some reason, Kaggle's website
        # doesn't accept absolutely correct polygon
        if test_image_name == '6050_4_4':
            polygons2 = []
            for pp in m_pol_uu.geoms:
                # There is some problem with only one polygon in the image 6050_4_4.
                # Not sure whether this problem have been caused by my model or
                # it is Kaggle's internal error. But! How both RF and XGB may independently predict
                # something that cause that problem????? IDK.... This trick fixes that!
                # It removes only one polygon, that corresponds to some building. Is it
                # possible that this is a building of someone who don't want to be found by us?
                # Is he James Bond or Mr. Evil?
                # I definitely need a new tin foil hat.
                if pp.contains(Point(0.00495888, -0.0034872)):
                    continue

                if pp.is_valid:
                    polygons2.append(pp)
                else:
                    polygons2.append(pp.buffer(0))

            m_pol_uu = unary_union(MultiPolygon(polygons2))

        ind_val_buildings = submission_data[(submission_data.ImageId == test_image_name) &
                                            (submission_data.ClassType == 1)].index

        submission_data.set_value(ind_val_buildings, 'MultipolygonWKT', m_pol_uu.wkt)

    submission_data.to_csv(os.path.join('..', '..', 'predictions', 'subm{}_cls1.csv'.format(SUBMISSION), index=False))
