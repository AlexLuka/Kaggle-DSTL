import pandas as pd
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import transform, unary_union
from scipy import ndimage, misc
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as Pol
from tqdm import tqdm
from skimage import measure
import os
import tifffile as tif


SUBMISSION = 150
WANNA_PLOT = False
# MODELS = ['model-rf-1', 'model-rf-2', 'model-1']
MODELS = ['model-xgb-2', 'model-xgb-3', 'model-rf-1f', 'model-ann-1']
MODEL_ID = 'complex-2'
SUBMISSION_IMG_DIR = 'submission-{}'.format(SUBMISSION)
LIST_OF_IMPORTANT_IMAGES = ['6020_0_4', '6020_4_3', '6030_2_4', '6030_3_3', '6030_3_4', '6030_4_2',
                            '6030_4_3', '6030_4_4', '6050_0_0', '6050_0_1', '6050_0_4', '6050_1_0',
                            '6050_1_4', '6050_2_0', '6050_2_1', '6050_2_2', '6050_2_3', '6050_2_4',
                            '6050_3_2', '6050_3_3', '6050_3_4', '6050_4_1', '6050_4_2', '6050_4_3',
                            '6050_4_4', '6070_0_0', '6070_0_1', '6070_0_2', '6070_0_3', '6070_1_0',
                            '6070_1_1', '6070_1_2', '6070_1_3', '6070_1_4', '6070_2_1', '6070_2_2',
                            '6070_2_3', '6070_2_4', '6070_3_1', '6070_3_2', '6070_3_3', '6070_3_4',
                            '6070_4_0', '6070_4_1', '6070_4_2', '6070_4_3', '6070_4_4', '6080_0_0',
                            '6080_0_1', '6080_0_2', '6080_0_3', '6080_0_4', '6080_1_0', '6080_1_1',
                            '6080_1_2', '6080_1_3', '6080_1_4', '6080_2_0', '6080_2_1', '6080_2_2',
                            '6080_2_3', '6080_2_4', '6080_3_0', '6080_3_1', '6080_3_2', '6080_3_3',
                            '6080_3_4', '6080_4_0', '6080_4_1', '6080_4_2', '6080_4_3', '6080_4_4',
                            '6100_0_0', '6100_0_2', '6100_2_2', '6100_1_4', '6100_1_2', '6100_3_1',
                            '6100_3_3', '6110_0_2', '6110_1_1', '6110_2_2', '6110_2_3', '6110_2_4',
                            '6110_3_0', '6130_0_4', '6140_4_0', '6150_2_4', '6150_3_4', '6150_4_3',
                            '6150_4_4', '6150_4_2']
IMAGES_TO_PLOT = ['6080_4_0', '6080_0_2', '6070_2_3', '6070_0_0', '6150_3_3', '6150_3_4', '6150_4_2']


def second_moment(poly_):
    x, y = poly_.exterior.coords.xy

    x -= np.mean(x)
    y -= np.mean(y)

    inertia = np.zeros(shape=(2, 2))

    for k in range(len(x)-1):
        inertia[0, 0] += (y[k]**2 + y[k]*y[k+1] + y[k+1]**2)*(x[k]*y[k+1] - x[k+1]*y[k])
        inertia[1, 1] += (x[k]**2 + x[k]*x[k+1] + x[k+1]**2)*(x[k]*y[k+1] - x[k+1]*y[k])
        inertia[1, 0] += (x[k]*y[k+1] + 2*x[k]*y[k] + 2*x[k+1]*y[k+1] + x[k+1]*y[k])*(x[k]*y[k+1] - x[k+1]*y[k])
    inertia[0, 1] = inertia[1, 0]

    inertia[0, 0] /= 12.
    inertia[1, 1] /= 12.
    inertia[1, 0] /= 24.
    inertia[0, 1] /= 24.

    l, _ = np.linalg.eig(np.absolute(inertia))
    l = -np.sort(-l)
    return l[0] / l[1]


def get_image_shape(image_name_):
    image_ = tif.imread('../../data/sixteen_band/{}_{}.tif'.format(image_name_, 'M'))[0, :, :]
    h_, w_ = image_.shape
    del image_
    return h_, w_


if __name__ == '__main__':
    submission_data = pd.read_csv(os.path.join('..', '..', 'predictions', 'empty_submission.csv'))
    grid_sizes = pd.read_csv(os.path.join('..', '..', 'data', 'grid_sizes.csv'))

    if not os.path.exists(os.path.join('predicted-masks', MODEL_ID)):
        os.mkdir(os.path.join('predicted-masks', MODEL_ID))

    if WANNA_PLOT and not os.path.exists(SUBMISSION_IMG_DIR):
        os.mkdir(SUBMISSION_IMG_DIR)

    if WANNA_PLOT:
        plt.figure(figsize=(10, 10))

    for test_image_name in tqdm(submission_data.ImageId.unique()):
        # for test_image_name in tqdm(LIST_OF_IMPORTANT_IMAGES):
        # if test_image_name != '6050_2_2':
        #     continue
        img_ = np.ones(shape=(get_image_shape(test_image_name)))

        for model in MODELS:
            img_mask = ndimage.imread('predicted-masks/{1}/{0}.png'.format(test_image_name, model),
                                      flatten=True)
            img_ = np.multiply(img_, img_mask)

        img_ = (img_ * 255. / np.max(img_)).astype(np.uint8)        # convert to integer

        misc.toimage(img_).save(os.path.join('predicted-masks', MODEL_ID, '{}.png'.format(test_image_name)))

        # zero padding
        img_[0, :] = 0
        img_[-1, :] = 0
        img_[:, 0] = 0
        img_[:, -1] = 0

        # scale coefficients
        x_max = grid_sizes[grid_sizes.IMG == test_image_name].Xmax.values[0]
        y_min = grid_sizes[grid_sizes.IMG == test_image_name].Ymin.values[0]

        x_scale = np.float(img_.shape[1] ** 2) / np.float((img_.shape[1] + 1) * x_max)
        y_scale = np.float(img_.shape[0] ** 2) / np.float((img_.shape[0] + 1) * y_min)

        # recognize contours
        contours = measure.find_contours(img_.T, 0.2)

        polygons = []

        for contour in contours:

            if contour.shape[0] > 2:        # contour must have at least 3 points

                poly = Polygon(contour)

                if poly.is_valid and poly.area > 2.:
                    polygons.append(poly)

        polygons_trees = []     #

        n = 0
        while len(polygons) > 0:
            p1 = polygons[n]
            is_hole = False

            holes = []
            for i in range(1, len(polygons)):
                if p1.contains(polygons[i]):
                    holes.append(polygons[i])

                elif polygons[i].contains(p1):
                    is_hole = True
                    break

            if not is_hole:

                polygons.remove(p1)
                n = 0

                for hole in holes:
                    polygons.remove(hole)

                polygons_trees.append(p1.difference(MultiPolygon(holes)))

            else:
                n += 1

        multipolygon_trees = MultiPolygon(polygons_trees)

        if WANNA_PLOT and test_image_name in LIST_OF_IMPORTANT_IMAGES:
            img_orig = tif.imread('../../data/three_band/{}.tif'.format(test_image_name))[0, :, :]
            x_scale2 = np.float(img_orig.shape[1] ** 2) / np.float((img_orig.shape[1] + 1) * x_max)
            y_scale2 = np.float(img_orig.shape[0] ** 2) / np.float((img_orig.shape[0] + 1) * y_min)

            plt.imshow(img_orig)

            ax = plt.gca()

            for p in multipolygon_trees.geoms:
                p = transform(lambda x, y: (x*x_scale2/x_scale, y*y_scale2/y_scale), p)
                pol = Pol(np.array(p.exterior), color='r', alpha=0.35)
                ax.add_patch(pol)

                for inter in p.interiors:
                    pol = Pol(np.array(inter), color='y', alpha=0.3)
                    ax.add_patch(pol)

            plt.title(test_image_name)

            plt.tight_layout()
            plt.savefig(os.path.join(SUBMISSION_IMG_DIR, '{}.png'.format(test_image_name)),
                        format='png',
                        dpi=500)

            plt.clf()
            plt.cla()
        #

        cu_ = unary_union(multipolygon_trees)
        cu_ = transform(lambda x, y: (np.round(float(x) / x_scale, 8), np.round(float(y) / y_scale, 8)), cu_)

        # fix broken polygons
        polygons2 = []
        for pp in cu_.geoms:

            if pp.is_valid:
                polygons2.append(pp)
            else:
                polygons2.append(pp.buffer(0))
        cu_ = unary_union(MultiPolygon(polygons2))

        ind_val = submission_data[(submission_data.ImageId == test_image_name) &
                                  (submission_data.ClassType == 5)].index
        submission_data.set_value(ind_val, 'MultipolygonWKT', cu_.wkt)

    submission_data.to_csv(os.path.join('..', '..', 'predictions', 'subm{}_cls5.csv'.format(SUBMISSION)), index=False)
