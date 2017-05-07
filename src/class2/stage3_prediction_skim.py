import os
import numpy as np
import pandas as pd
import tifffile as tif
from tqdm import tqdm
from scipy import ndimage
from skimage import measure
from shapely.ops import transform, unary_union
from shapely.geometry import Polygon, MultiPolygon
from matplotlib.patches import Polygon as Pol
import matplotlib.pyplot as plt


SUBMISSION = 153
WANNA_PLOT = False
WITH_HOLES = False


if __name__ == '__main__':
    submission_data = pd.read_csv(os.path.join('..', '..', 'predictions', 'empty_submission.csv'))
    grid_sizes = pd.read_csv(os.path.join('..', '..', 'data', 'grid_sizes.csv'))

    if WANNA_PLOT and not os.path.exists('submission-{}'.format(SUBMISSION)):
        os.mkdir('submission-{}'.format(SUBMISSION))

    if WANNA_PLOT:
        plt.figure(figsize=(7, 7))

    for test_image_name in tqdm(submission_data.ImageId.unique()):

        img_mask = ndimage.imread(os.path.join('predicted-masks', 'model-rf-3', '{}.png'.format(test_image_name)),
                                  flatten=True)

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

                if poly.is_valid:
                    polygons.append(poly)

        #
        mp1 = MultiPolygon(polygons)

        ################################################################################################################
        if WANNA_PLOT:
            img_orig = tif.imread(os.path.join('..', '..', 'data', 'three_band', '{}.tif'.
                                               format(test_image_name)))[0, :, :]
            x_scale2 = np.float(img_orig.shape[1] ** 2) / np.float((img_orig.shape[1] + 1) * x_max)
            y_scale2 = np.float(img_orig.shape[0] ** 2) / np.float((img_orig.shape[0] + 1) * y_min)

            plt.imshow(img_orig)

            ax = plt.gca()

            for p in mp1.geoms:
                p = transform(lambda x, y: (float(x) * x_scale2 / x_scale, float(y) * y_scale2 / y_scale), p)
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

        ################################################################################################################
        mp = transform(lambda x, y: (np.round(float(x) / x_scale, 8), np.round(float(y) / y_scale, 8)),
                       MultiPolygon(polygons))

        mp_uu = unary_union(mp)

        # Bug fix
        if isinstance(mp_uu, MultiPolygon):
            polygons2 = []
            for pp in mp_uu.geoms:
                if pp.is_valid:
                    polygons2.append(pp)
                else:
                    polygons2.append(pp.buffer(0))

            mp_uu = unary_union(MultiPolygon(polygons2))
        elif isinstance(mp_uu, Polygon):
            if not mp_uu.is_valid:
                mp_uu = mp_uu.buffer(0)

        #
        ind_val = submission_data[(submission_data.ImageId == test_image_name) &
                                  (submission_data.ClassType == 2)].index

        submission_data.set_value(ind_val, 'MultipolygonWKT', mp_uu.wkt)

    submission_data.to_csv(os.path.join('..', '..', 'predictions', 'subm{}-cls2.csv'.format(SUBMISSION)), index=False)
