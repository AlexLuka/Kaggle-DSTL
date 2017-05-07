import os
import numpy as np
import pandas as pd
import tifffile as tif
from tqdm import tqdm
from scipy import ndimage, misc
from skimage import measure
from shapely.ops import transform, unary_union
from shapely.geometry import Polygon, MultiPolygon
from matplotlib.patches import Polygon as Pol
import matplotlib.pyplot as plt


SUBMISSION = 154
WANNA_PLOT = False
MODELS = ['model-rf-3', 'model-rf-4', 'model-rf-5']
MODEL_ID = 'complex-1'
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


def get_image_shape(image_name_):
    image_ = tif.imread(os.path.join('..', '..', 'data', 'sixteen_band', '{}_M.tif'.format(image_name_)))[0, :, :]
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
        #
        img_ = np.ones(shape=(get_image_shape(test_image_name)))

        for model in MODELS:
            img_mask = ndimage.imread('predicted-masks/{1}/{0}.png'.format(test_image_name, model),
                                      flatten=True)
            img_ = np.multiply(img_, img_mask)

        img_ = img_.astype(np.uint8)        # convert to integer

        misc.toimage(img_).save(os.path.join('predicted-masks', MODEL_ID, '{}.png'.format(test_image_name)))

        # zero border
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

                if poly.is_valid:
                    polygons.append(poly)

        mp = MultiPolygon(polygons)

        ################################################################################################################
        if WANNA_PLOT and test_image_name in LIST_OF_IMPORTANT_IMAGES:
            img_orig = tif.imread(os.path.join('..', '..', 'data', 'three_band', '{}.tif'.
                                               format(test_image_name)))[0, :, :]
            x_scale2 = np.float(img_orig.shape[1] ** 2) / np.float((img_orig.shape[1] + 1) * x_max)
            y_scale2 = np.float(img_orig.shape[0] ** 2) / np.float((img_orig.shape[0] + 1) * y_min)

            plt.imshow(img_orig)

            ax = plt.gca()

            for p in mp.geoms:
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
        ################################################################################################################

        cu_ = unary_union(mp)
        cu_ = transform(lambda x, y: (np.round(float(x) / x_scale, 8), np.round(float(y) / y_scale, 8)), cu_)

        # fix broken polygons
        if isinstance(cu_, MultiPolygon):
            polygons2 = []
            for pp in cu_.geoms:

                if pp.is_valid:
                    polygons2.append(pp)
                else:
                    polygons2.append(pp.buffer(0))
            cu_ = unary_union(MultiPolygon(polygons2))
        elif isinstance(cu_, Polygon):
            if not cu_.is_valid:
                cu_ = unary_union(cu_.buffer(0))

        ind_val = submission_data[(submission_data.ImageId == test_image_name) &
                                  (submission_data.ClassType == 2)].index
        submission_data.set_value(ind_val, 'MultipolygonWKT', cu_.wkt)

    submission_data.to_csv(os.path.join('..', '..', 'predictions', 'subm{}_cls2.csv'.format(SUBMISSION)), index=False)
