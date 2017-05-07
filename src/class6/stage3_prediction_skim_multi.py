import pandas as pd
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import transform, unary_union
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as Pol
from tqdm import tqdm
from skimage import measure
import os
import tifffile as tif


SUBMISSION = 119
WANNA_PLOT = True
WITH_HOLES = True
MODELS_TO_USE = [
    'model-rf-1',
    'model-rf-2',
    'model-rf-3',
    'model-xgb-1',
    'model-xgb-2'
]


if __name__ == '__main__':
    submission_data = pd.read_csv('../../data/sample_submission.csv')
    grid_sizes = pd.read_csv('../../data/grid_sizes.csv')

    if WANNA_PLOT and not os.path.exists('submission-{}'.format(SUBMISSION)):
        os.mkdir('submission-{}'.format(SUBMISSION))

    if WANNA_PLOT:
        plt.figure(figsize=(7, 7))

    for test_image_name in tqdm(submission_data.ImageId.unique()):
        # for test_image_name in tqdm(LIST_OF_IMPORTANT_IMAGES):
        # if test_image_name != '6050_4_4':
        #     continue
        models_img = []

        # img_ = ndimage.imread('predicted-masks/{}/{}.png'.format(MODELS_TO_USE[0], test_image_name),
        #                       flatten=True)

        for model_ in MODELS_TO_USE:
            img_mask = ndimage.imread('predicted-masks/{}/{}.png'.format(model_, test_image_name),
                                      flatten=True)
            models_img.append(img_mask)

            # img_ = np.logical_and(img_, img_mask)

        img_ = np.median(np.asarray(models_img), axis=0)
        img_ = ndimage.binary_fill_holes(img_)

        # print np.asarray(models_img).shape, img_.shape
        #
        # plt.subplot(121)
        # plt.imshow(models_img[0])
        #
        # plt.subplot(122)
        # plt.imshow(img_)
        #
        # plt.show()
        # exit(0)
        # img_mask = ndimage.binary_fill_holes(img_mask)
        img_ = img_.astype(np.uint8)        # convert to integer

        # zero padding
        img_[0, :] = 0
        img_[-1, :] = 0
        img_[:, 0] = 0
        img_[:, -1] = 0

        # bounding_polygon = Polygon([(3, 3),
        #                             (img_.shape[0]-4, 3),
        #                             (img_.shape[0]-4, img_.shape[1]-4),
        #                             (3, img_.shape[1]-4)])

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

                if poly.is_valid and poly.area > 100.:
                    polygons.append(poly)

        polygons_crop = []

        if WITH_HOLES:
            n = 0
            while len(polygons) > 0:
                p1 = polygons.pop(n)
                is_hole = False

                holes = []
                for i, p2 in enumerate(polygons):
                    if p1.contains(p2):
                        holes.append(polygons.pop(i))
                        # print holes
                    elif p2.contains(p1):
                        is_hole = True
                        polygons.append(p1)     # return it back
                        break

                if not is_hole:

                    n = 0

                    # may cause a "no outgoing dirEdge found" problem
                    # polygons_trees.append(p1.difference(MultiPolygon(holes)))

                    # possible workaround
                    polygons_crop.append(p1.difference(unary_union(MultiPolygon(holes))))
                    # precision reduction may work as well

                else:
                    n += 1
        else:
            polygons_crop = polygons

        mp1 = MultiPolygon(polygons_crop)

        if WANNA_PLOT:
            img_orig = tif.imread('../../data/three_band/{}.tif'.format(test_image_name))[0, :, :]
            x_scale2 = np.float(img_orig.shape[1] ** 2) / np.float((img_orig.shape[1] + 1) * x_max)
            y_scale2 = np.float(img_orig.shape[0] ** 2) / np.float((img_orig.shape[0] + 1) * y_min)

            plt.imshow(img_orig)
            ax = plt.gca()

            for p in mp1.geoms:
                p = transform(lambda x, y: (float(x) * x_scale2 / x_scale, float(y) * y_scale2 / y_scale), p)
                # p = transform(lambda x, y: (x / x_scale, y / y_scale), p)
                pol = Pol(np.array(p.exterior), color='k', alpha=0.35)
                ax.add_patch(pol)

                for inter in p.interiors:
                    pol = Pol(np.array(inter), color='w', alpha=0.3)
                    ax.add_patch(pol)

            plt.title(test_image_name)

            plt.tight_layout()
            plt.savefig('submission-{}/{}.png'.format(SUBMISSION, test_image_name), format='png', dpi=500)

            plt.clf()
            plt.cla()

        mp_uu = unary_union(mp1)

        mp_uu = transform(lambda x, y: (np.round(float(x) / x_scale, 8), np.round(float(y) / y_scale, 8)), mp_uu)

        # Bug fix
        polygons2 = []
        if isinstance(mp_uu, MultiPolygon):
            polygons2 = []
            for pp in mp_uu.geoms:
                if pp.is_valid:
                    polygons2.append(pp)
                else:
                    polygons2.append(pp.buffer(0))

            mp_uu = unary_union(MultiPolygon(polygons2))

        ind_val = submission_data[(submission_data.ImageId == test_image_name) &
                                  (submission_data.ClassType == 6)].index
        submission_data.set_value(ind_val, 'MultipolygonWKT', mp_uu.wkt)

    submission_data.to_csv('../../predictions/subm{}_cls6.csv'.format(SUBMISSION), index=False)
