import os
import numpy as np
import pandas as pd
import tifffile as tif
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import ndimage
from skimage import measure
from shapely.ops import transform, unary_union
from shapely.geometry import Polygon, MultiPolygon, LineString
from matplotlib.patches import Polygon as Pol


#
#       Important constants that must be defined before execution
#
SUBMISSION = 155
WANNA_PLOT = False
MODELS = ['model-xgb-1', 'model-xgb-2', 'model-rf-2', 'model-rf-1', 'model-1']
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


def second_moment(poly_):
    """

    """
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
    submission_data = pd.read_csv(os.path.join('..', '..', 'predictions', 'final-11.csv'))
    grid_sizes = pd.read_csv(os.path.join('..', '..', 'data', 'grid_sizes.csv'))

    if WANNA_PLOT and not os.path.exists(SUBMISSION_IMG_DIR):
        os.mkdir(SUBMISSION_IMG_DIR)

    if WANNA_PLOT:
        plt.figure(figsize=(10, 10))

    for test_image_name in tqdm(submission_data.ImageId.unique()):
        #
        img_ = np.zeros(shape=(get_image_shape(test_image_name)))

        for model in MODELS:
            img_mask = ndimage.imread('predicted-masks/{1}/{0}.png'.format(test_image_name, model),
                                      flatten=True) / 255.

            img_ = np.add(img_, img_mask)

        img_ /= float(len(MODELS))
        img_ = np.rint(img_)

        img_ = img_.astype(np.uint8)        # convert to integer

        # frame of zeros. need to avoid appearance of polygons that cut corners
        img_[0, :] = 0
        img_[-1, :] = 0
        img_[:, 0] = 0
        img_[:, -1] = 0

        # scale coefficients
        x_max = grid_sizes[grid_sizes.IMG == test_image_name].Xmax.values[0]
        y_min = grid_sizes[grid_sizes.IMG == test_image_name].Ymin.values[0]

        x_scale = np.float(img_.shape[1] ** 2) / np.float((img_.shape[1] + 1) * x_max)
        y_scale = np.float(img_.shape[0] ** 2) / np.float((img_.shape[0] + 1) * y_min)

        # identify contours
        contours = measure.find_contours(img_.T, 0.2)

        polygons = []

        for contour in contours:

            if contour.shape[0] > 2:        # contour must have at least 3 points

                poly = Polygon(contour)

                if poly.is_valid and poly.area > 10.:
                    polygons.append(poly)

        polygons_lakes = []     # for class 8 - standing water
        polygons_rivers = []    # for class 7 - rivers, waterways

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

                #
                m = second_moment(p1)

                #
                polygons.remove(p1)
                n = 0

                for hole in holes:
                    polygons.remove(hole)

                border_intersections = 0
                if p1.intersects(LineString([(2, 2), (img_.shape[1]-3, 2)])):
                    border_intersections += 1
                if p1.intersects(LineString([(img_.shape[1]-3, 2), (img_.shape[1]-3, img_.shape[0]-3)])):
                    border_intersections += 1
                if p1.intersects(LineString([(img_.shape[1]-3, img_.shape[0]-3), (2, img_.shape[0]-3)])):
                    border_intersections += 1
                if p1.intersects(LineString([(2, img_.shape[0]-3), (2, 2)])):
                    border_intersections += 1

                if border_intersections > 1:
                    polygons_rivers.append(p1.difference(MultiPolygon(holes)))
                elif border_intersections == 1:
                    if p1.area > 5000.:
                        polygons_rivers.append(p1.difference(MultiPolygon(holes)))
                    else:
                        polygons_lakes.append(p1)
                else:
                    polygons_lakes.append(p1)

            else:
                n += 1

        if len(polygons_rivers) > 0:
            polygons_rivers.extend(polygons_lakes)
            polygons_lakes = []

        multipolygon_rivers = MultiPolygon(polygons_rivers)
        multipolygon_lakes = MultiPolygon(polygons_lakes)

        if WANNA_PLOT and test_image_name in LIST_OF_IMPORTANT_IMAGES:
            img_orig = tif.imread('../../data/three_band/{}.tif'.format(test_image_name))[0, :, :]
            x_scale2 = np.float(img_orig.shape[1] ** 2) / np.float((img_orig.shape[1] + 1) * x_max)
            y_scale2 = np.float(img_orig.shape[0] ** 2) / np.float((img_orig.shape[0] + 1) * y_min)

            plt.imshow(img_orig)

            ax = plt.gca()

            for p in multipolygon_rivers.geoms:
                p = transform(lambda x, y: (x*x_scale2/x_scale, y*y_scale2/y_scale), p)
                pol = Pol(np.array(p.exterior), color='r', alpha=0.35)
                ax.add_patch(pol)

                for inter in p.interiors:
                    pol = Pol(np.array(inter), color='y', alpha=0.3)
                    ax.add_patch(pol)

            for p in multipolygon_lakes.geoms:
                p = transform(lambda x, y: (x*x_scale2/x_scale, y*y_scale2/y_scale), p)
                pol = Pol(np.array(p.exterior), color='k', alpha=0.5)
                ax.add_patch(pol)

            plt.title(test_image_name)

            plt.tight_layout()
            plt.savefig(os.path.join(SUBMISSION_IMG_DIR, '{}.png'.format(test_image_name)),
                        format='png',
                        dpi=500)

            plt.clf()
            plt.cla()
        #
        #
        cu_l = unary_union(multipolygon_lakes)
        cu_r = unary_union(multipolygon_rivers)

        #
        cu_l = transform(lambda x, y: (np.round(float(x) / x_scale, 8), np.round(float(y) / y_scale, 8)), cu_l)
        cu_r = transform(lambda x, y: (np.round(float(x) / x_scale, 8), np.round(float(y) / y_scale, 8)), cu_r)

        #
        ind_val = submission_data[(submission_data.ImageId == test_image_name) &
                                  (submission_data.ClassType == 7)].index
        submission_data.set_value(ind_val, 'MultipolygonWKT', cu_r.wkt)

        ind_val = submission_data[(submission_data.ImageId == test_image_name) &
                                  (submission_data.ClassType == 8)].index
        submission_data.set_value(ind_val, 'MultipolygonWKT', cu_l.wkt)

    submission_data.to_csv(os.path.join('..', '..', 'predictions', 'sub-{}-classes-7,8.csv'.format(SUBMISSION)),
                           index=False)
