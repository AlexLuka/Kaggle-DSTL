import os
import numpy as np
import pandas as pd
import tifffile as tif
from tqdm import tqdm
from scipy import ndimage
from skimage import measure
from shapely.ops import transform, cascaded_union
from shapely.geometry import Polygon, MultiPolygon

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as Pol


SUBMISSION = 91
WANNA_PLOT = True
MODEL_ID = 'model-rf-1'
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


# def second_moment(poly_):
#     x, y = poly_.exterior.coords.xy
#
#     x -= np.mean(x)
#     y -= np.mean(y)
#
#     inertia = np.zeros(shape=(2, 2))
#
#     for k in range(len(x)-1):
#         inertia[0, 0] += (y[k]**2 + y[k]*y[k+1] + y[k+1]**2)*(x[k]*y[k+1] - x[k+1]*y[k])
#         inertia[1, 1] += (x[k]**2 + x[k]*x[k+1] + x[k+1]**2)*(x[k]*y[k+1] - x[k+1]*y[k])
#         inertia[1, 0] += (x[k]*y[k+1] + 2*x[k]*y[k] + 2*x[k+1]*y[k+1] + x[k+1]*y[k])*(x[k]*y[k+1] - x[k+1]*y[k])
#     inertia[0, 1] = inertia[1, 0]
#
#     inertia[0, 0] /= 12.
#     inertia[1, 1] /= 12.
#     inertia[1, 0] /= 24.
#     inertia[0, 1] /= 24.
#
#     l, _ = np.linalg.eig(np.absolute(inertia))
#     l = -np.sort(-l)
#     return l[0] / l[1]


if __name__ == '__main__':
    submission_data = pd.read_csv(os.path.join('..', '..', 'data', 'sample_submission.csv'))
    grid_sizes = pd.read_csv(os.path.join('..', '..', 'data', 'grid_sizes.csv'))

    if WANNA_PLOT and not os.path.exists('submission-{}'.format(SUBMISSION)):
        os.mkdir('submission-{}'.format(SUBMISSION))

    if WANNA_PLOT:
        plt.figure(figsize=(10, 10))

    for test_image_name in tqdm(submission_data.ImageId.unique()):

        img_mask = ndimage.imread(os.path.join('predicted-masks-2', MODEL_ID, '{}.png'.format(test_image_name)),
                                  flatten=True)
        # img_mask = ndimage.binary_fill_holes(img_mask)
        img_ = img_mask.astype(np.uint8)        # convert to integer

        # zero padding
        img_[0, :] = 0
        img_[-1, :] = 0
        img_[:, 0] = 0
        img_[:, -1] = 0

        # scale coefficients
        x_max = grid_sizes[grid_sizes.IMG == test_image_name].Xmax.values[0]
        y_min = grid_sizes[grid_sizes.IMG == test_image_name].Ymin.values[0]

        x_scale = np.float(img_mask.shape[1] ** 2) / np.float((img_mask.shape[1] + 1) * x_max)
        y_scale = np.float(img_mask.shape[0] ** 2) / np.float((img_mask.shape[0] + 1) * y_min)

        # recognize contours
        contours = measure.find_contours(img_.T, 0.2)

        polygons = []

        for contour in contours:

            if contour.shape[0] > 2:        # contour must have at least 3 points

                poly = Polygon(contour)

                #
                if poly.is_valid and poly.area > 50:
                    polygons.append(poly)

        multipolygon_roads = MultiPolygon(polygons)

        if WANNA_PLOT:
            img_orig = tif.imread(os.path.join('..', '..', 'data', 'three_band', '{}.tif'.
                                               format(test_image_name)))[0, :, :]

            x_scale2 = np.float(img_orig.shape[1] ** 2) / np.float((img_orig.shape[1] + 1) * x_max)
            y_scale2 = np.float(img_orig.shape[0] ** 2) / np.float((img_orig.shape[0] + 1) * y_min)

            plt.imshow(img_orig)

            ax = plt.gca()

            for p in multipolygon_roads.geoms:
                p = transform(lambda x, y: (x*x_scale2/x_scale, y*y_scale2/y_scale), p)
                pol = Pol(np.array(p.exterior), color='r', alpha=0.35)
                ax.add_patch(pol)

                for inter in p.interiors:
                    pol = Pol(np.array(inter), color='y', alpha=0.3)
                    ax.add_patch(pol)

            plt.title(test_image_name)

            plt.tight_layout()
            plt.savefig('submission-{}/{}.png'.format(SUBMISSION, test_image_name), format='png', dpi=500)

            # plt.show()

            plt.clf()
            plt.cla()

        cu_l = cascaded_union(multipolygon_roads)

        cu_l = transform(lambda x, y: (np.round(float(x) / x_scale, 8), np.round(float(y) / y_scale, 8)), cu_l)

        ind_val = submission_data[(submission_data.ImageId == test_image_name) &
                                  (submission_data.ClassType == 3)].index
        submission_data.set_value(ind_val, 'MultipolygonWKT', cu_l.wkt)

    submission_data.to_csv(os.path.join('..', '..', 'predictions', 'subm{}_cls3.csv'.format(SUBMISSION)), index=False)
