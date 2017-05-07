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


SUBMISSION = 117
WANNA_PLOT = True
WITH_HOLES = False


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

        img_mask = ndimage.imread('predicted-masks/model-xgb-1/{}.png'.format(test_image_name),
                                  flatten=True)

        # img_mask = ndimage.binary_fill_holes(img_mask)
        img_ = img_mask.astype(np.uint8)        # convert to integer

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

                if poly.is_valid and poly.area > 5.:
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
            # img_orig = ndimage.imread('images/{}.png'.format(test_image_name), flatten=True).T

            plt.imshow(img_orig)

            # plt.subplot(121)
            # plt.plot([0],[0],'ok')
            ax = plt.gca()

            # print multipolygon_trees
            # pp = transform(lambda x, y: (int(x), int(y)), multipolygon_trees)
            # mp = unary_union(transform(lambda x, y: (int(x), int(y)), multipolygon_trees))
            # print mm

            for p in mp1.geoms:
                p = transform(lambda x, y: (float(x) * x_scale2 / x_scale, float(y) * y_scale2 / y_scale), p)
                # p = transform(lambda x, y: (x / x_scale, y / y_scale), p)
                pol = Pol(np.array(p.exterior), color='k', alpha=0.35)
                ax.add_patch(pol)

                for inter in p.interiors:
                    pol = Pol(np.array(inter), color='w', alpha=0.3)
                    ax.add_patch(pol)

            # plt.subplot(122)
            # plt.plot([0],[0],'ok')
            # ax = plt.gca()
            #
            # for p in unary_union(multipolygon_trees).geoms:
            #     # p = transform(lambda x, y: (x * x_scale2 / x_scale, y * y_scale2 / y_scale), p)
            #     p = transform(lambda x, y: (x / x_scale, y / y_scale), p)
            #     pol = Pol(np.array(p.exterior), color='r', alpha=0.35)
            #     ax.add_patch(pol)
            #
            #     for inter in p.interiors:
            #         pol = Pol(np.array(inter), color='y', alpha=0.3)
            #         ax.add_patch(pol)

            # for p in multipolygon_lakes.geoms:
            #     p = transform(lambda x, y: (x*x_scale2/x_scale, y*y_scale2/y_scale), p)
            #     pol = Pol(np.array(p.exterior), color='k', alpha=0.5)
            #     ax.add_patch(pol)

            # ax.add_patch(Pol(np.array(transform(lambda x, y:
            #                                     (x*x_scale2/x_scale, y*y_scale2/y_scale),
            #                                     bounding_polygon).exterior), color='w', alpha=0.1))

            plt.title(test_image_name)

            plt.tight_layout()
            plt.savefig('submission-{}/{}.png'.format(SUBMISSION, test_image_name), format='png', dpi=500)

            # plt.show()

            plt.clf()
            plt.cla()

        # round coordinates (in pixels)
        # multipolygon_trees = transform(lambda x, y: (int(x), int(y)), multipolygon_trees)
        # pols = list(multipolygon_trees.geoms)
        #
        # n = 0
        # while len(pols) > 0:
        #     p1 = pols.pop(n)
        #
        # print 'VALIDATION 1: ', cu_trees.is_valid

        mp_uu = unary_union(mp1)

        mp_uu = transform(lambda x, y: (np.round(float(x) / x_scale, 8), np.round(float(y) / y_scale, 8)), mp_uu)
        # mp = transform(lambda x, y: (x / x_scale, y / y_scale), mp1)

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

        # print submission_data[(submission_data.ImageId == test_image_name) & (submission_data.ClassType == 5)]
        ind_val = submission_data[(submission_data.ImageId == test_image_name) &
                                  (submission_data.ClassType == 6)].index
        submission_data.set_value(ind_val, 'MultipolygonWKT', mp_uu.wkt)

    submission_data.to_csv('../../predictions/subm{}_cls6.csv'.format(SUBMISSION), index=False)
