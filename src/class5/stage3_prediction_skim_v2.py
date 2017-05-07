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


SUBMISSION = 9
WANNA_PLOT = True


if __name__ == '__main__':
    submission_data = pd.read_csv('../../predictions/final_submission8.csv')
    grid_sizes = pd.read_csv('../../data/grid_sizes.csv')

    if WANNA_PLOT and not os.path.exists('submission-{}'.format(SUBMISSION)):
        os.mkdir('submission-{}'.format(SUBMISSION))

    if WANNA_PLOT:
        plt.figure(figsize=(7, 7))

    # areas = []
    # moments = []

    for test_image_name in tqdm(submission_data.ImageId.unique()):
        # for test_image_name in tqdm(LIST_OF_IMPORTANT_IMAGES):
        # if test_image_name != '6050_4_4':
        #     continue

        img_mask = ndimage.imread('predicted-masks/model-rf-1f/{}.png'.format(test_image_name),
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

                if poly.is_valid and poly.area > 10.:
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

                if p1.area < 10000.:
                    polygons_trees.append(p1.difference(MultiPolygon(holes)))

            else:
                n += 1

        multipolygon_trees = MultiPolygon(polygons_trees)

        if WANNA_PLOT:
            img_orig = tif.imread('../../data/three_band/{}.tif'.format(test_image_name))[0, :, :]
            x_scale2 = np.float(img_orig.shape[1] ** 2) / np.float((img_orig.shape[1] + 1) * x_max)
            y_scale2 = np.float(img_orig.shape[0] ** 2) / np.float((img_orig.shape[0] + 1) * y_min)
            # img_orig = ndimage.imread('images/{}.png'.format(test_image_name), flatten=True).T

            plt.imshow(img_orig)
            ax = plt.gca()

            for p in multipolygon_trees.geoms:
                p = transform(lambda x, y: (float(x) * x_scale2 / x_scale, float(y) * y_scale2 / y_scale), p)
                pol = Pol(np.array(p.exterior), color='k', alpha=0.35)
                ax.add_patch(pol)

                for inter in p.interiors:
                    pol = Pol(np.array(inter), color='y', alpha=0.3)
                    ax.add_patch(pol)

            plt.title(test_image_name)

            plt.tight_layout()
            plt.savefig('submission-{}/{}.png'.format(SUBMISSION, test_image_name), format='png', dpi=500)

            plt.clf()
            plt.cla()

        #
        #
        #
        #
        mp_uu = unary_union(multipolygon_trees)
        mp_uu = transform(lambda x, y: (np.round(float(x) / x_scale, 6), np.round(float(y) / y_scale, 6)), mp_uu)

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
                                  (submission_data.ClassType == 5)].index
        submission_data.set_value(ind_val, 'MultipolygonWKT', mp_uu.wkt)

    submission_data.to_csv('../../predictions/final_submission{}.csv'.format(SUBMISSION), index=False)
