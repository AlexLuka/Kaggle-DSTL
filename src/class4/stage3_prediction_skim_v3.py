import pandas as pd
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import transform, unary_union
from scipy import ndimage, misc
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as Pol
from tqdm import tqdm
from skimage import measure
import os
import tifffile as tif


SUBMISSION = 111
WANNA_PLOT = False
MODEL_ID1 = 'model-rf-4'
MODEL_ID2 = 'model-rf-5'
MODEL_ID3 = 'model-xgb-1'


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


if __name__ == '__main__':
    submission_data = pd.read_csv('../../data/sample_submission.csv')
    grid_sizes = pd.read_csv('../../data/grid_sizes.csv')

    if WANNA_PLOT and not os.path.exists('submission-{}'.format(SUBMISSION)):
        os.mkdir('submission-{}'.format(SUBMISSION))

    if WANNA_PLOT:
        plt.figure(figsize=(10, 10))

    for test_image_name in tqdm(submission_data.ImageId.unique()):
        # for test_image_name in tqdm(LIST_OF_IMPORTANT_IMAGES):
        # if test_image_name != '6050_2_2':
        #     continue

        img_mask1 = ndimage.imread('predicted-masks/{1}/{0}.png'.format(test_image_name, MODEL_ID1),
                                   flatten=True)

        img_mask2 = ndimage.imread('predicted-masks/{1}/{0}.png'.format(test_image_name, MODEL_ID2),
                                   flatten=True)

        img_mask3 = ndimage.imread('predicted-masks/{1}/{0}.png'.format(test_image_name, MODEL_ID3),
                                   flatten=True)

        # img_mask = ndimage.binary_fill_holes(img_mask)
        img_1 = img_mask1.astype(np.uint8).astype(np.bool)
        img_2 = img_mask2.astype(np.uint8).astype(np.bool)
        img_3 = img_mask3.astype(np.uint8)

        # take a median of these predictions
        img_ = np.median(np.asarray((np.logical_and(img_1, img_2), img_3)), axis=0)

        misc.toimage(img_).save(os.path.join('predicted-masks', 'RF-45+XGB-1', '{}.png'.format(test_image_name)))

        # zero padding
        img_[0, :] = 0
        img_[-1, :] = 0
        img_[:, 0] = 0
        img_[:, -1] = 0

        # scale coefficients
        x_max = grid_sizes[grid_sizes.IMG == test_image_name].Xmax.values[0]
        y_min = grid_sizes[grid_sizes.IMG == test_image_name].Ymin.values[0]

        x_scale = np.float(img_mask1.shape[1] ** 2) / np.float((img_mask1.shape[1] + 1) * x_max)
        y_scale = np.float(img_mask1.shape[0] ** 2) / np.float((img_mask1.shape[0] + 1) * y_min)

        # recognize contours
        contours = measure.find_contours(img_.T, 0.2)

        polygons = []

        for contour in contours:

            if contour.shape[0] > 2:        # contour must have at least 3 points

                poly = Polygon(contour)

                if poly.is_valid and poly.area > 2.:
                    polygons.append(poly)

        # n = 0
        # polygons_tracks = []
        # while len(polygons) > 0:
        #     p1 = polygons.pop(n)
        #     is_hole = False
        #
        #     holes = []
        #     for i, p2 in enumerate(polygons):
        #         if p1.contains(p2):
        #             holes.append(polygons.pop(i))
        #         elif p2.contains(p1):
        #             is_hole = True
        #             polygons.append(p1)  # return it back
        #             break
        #
        #     if not is_hole:
        #
        #         n = 0
        #
        #         polygons_tracks.append(p1.difference(unary_union(MultiPolygon(holes))))
        #     else:
        #         n += 1

        m_pol_vec = MultiPolygon(polygons)

        if WANNA_PLOT:
            img_orig = tif.imread('../../data/three_band/{}.tif'.format(test_image_name))[0, :, :]
            x_scale2 = np.float(img_orig.shape[1] ** 2) / np.float((img_orig.shape[1] + 1) * x_max)
            y_scale2 = np.float(img_orig.shape[0] ** 2) / np.float((img_orig.shape[0] + 1) * y_min)

            plt.imshow(img_orig)

            ax = plt.gca()

            for p in m_pol_vec.geoms:
                p = transform(lambda x, y: (x*x_scale2/x_scale, y*y_scale2/y_scale), p)
                pol = Pol(np.array(p.exterior), color='k', alpha=0.35)
                ax.add_patch(pol)

                for inter in p.interiors:
                    pol = Pol(np.array(inter), color='r', alpha=0.3)
                    ax.add_patch(pol)

            plt.title(test_image_name)

            plt.tight_layout()
            plt.savefig('submission-{}/{}.png'.format(SUBMISSION, test_image_name), format='png', dpi=500)

            # plt.show()

            plt.clf()
            plt.cla()

        m_pol_uu = unary_union(m_pol_vec)

        m_pol_uu = transform(lambda x, y: (np.round(float(x) / x_scale, 8), np.round(float(y) / y_scale, 8)),
                             m_pol_uu)

        # Bug fix
        if isinstance(m_pol_uu, MultiPolygon):
            polygons2 = []
            for pp in m_pol_uu.geoms:
                if pp.is_valid:
                    polygons2.append(pp)
                else:
                    polygons2.append(pp.buffer(0))

            m_pol_uu = unary_union(MultiPolygon(polygons2))

        ind_val_class4 = submission_data[(submission_data.ImageId == test_image_name) &
                                         (submission_data.ClassType == 4)].index

        submission_data.set_value(ind_val_class4, 'MultipolygonWKT', m_pol_uu.wkt)

    submission_data.to_csv('../../predictions/subm{}_cls4.csv'.format(SUBMISSION), index=False)
