import os
import numpy as np
import pandas as pd
import tifffile as tif
from tqdm import tqdm
from scipy import misc
from shapely.wkt import loads
from shapely.ops import transform
from shapely.geometry import shape, Point
from multiprocessing import Pool

from matplotlib.patches import Polygon as Pol
import matplotlib.pyplot as plt


"""
Each image is a matrix of size [H x W], first axis is height, second one is width.
X is associated with W and this is the second axis in matrix,
Y is associated with H and this is the first axis in matrix.
Therefore, image.shape = [H, W] and scale_x -> image.shape[1]
                                    scale_y -> image.shape[0]
"""

train_wkt = pd.read_csv(os.path.join('..', '..', 'data', 'train_wkt_v4.csv'))
grid_sizes = pd.read_csv(os.path.join('..', '..', 'data', 'grid_sizes.csv'))

SLICE_NUMBER = 4
BIT_MAX = 2048
PROCESSES = 2
MASK_DIR = 'masks'
IMG_DIR = 'images'
TRAIN_IMG_DIR = 'images-train'
PLOT_POLYGONS = True
CLASS_TYPE = 3


def decolorization((baw_image_, polygons_, margin_)):
    # print 'Process {}, {}'.format(margin_, baw_image_.shape)

    for i_ in tqdm(range(baw_image_.shape[0])):     # y axis
        for j_ in range(baw_image_.shape[1]):       # x axis
            for polygon_ in polygons_:
                if polygon_.contains(Point(j_, i_ + margin_)):
                    baw_image_[i_, j_] = BIT_MAX
                    break
    return baw_image_


if __name__ == '__main__':

    if PLOT_POLYGONS:
        plt.figure(figsize=(10, 5))

    pool = Pool(processes=PROCESSES)

    # directory with black and white masks
    if not os.path.exists(MASK_DIR):
        os.mkdir(MASK_DIR)

    # directory with actual images
    if not os.path.exists(IMG_DIR):
        os.mkdir(IMG_DIR)

    if not os.path.exists(TRAIN_IMG_DIR):
        os.mkdir(TRAIN_IMG_DIR)

    for image_name in reversed(train_wkt.ImageId.unique()):
        print 'Image {}'.format(image_name)

        if os.path.exists(os.path.join(MASK_DIR, '{}-mask.png'.format(image_name))):
            print 'Calculated'
            continue

        # read image in. The shape of the image is 8 x W x H. Read only one band
        img_16m = tif.imread(os.path.join('..', '..', 'data', 'sixteen_band', '{}_M.tif'.
                                          format(image_name)))[SLICE_NUMBER, :, :]
        img_16rgb = tif.imread(os.path.join('..', '..', 'data', 'three_band', '{}.tif'.format(image_name)))[0, :, :]

    # EXTRACT SHAPES
        # get all shapes identified for that image
        shapes = train_wkt[train_wkt.ImageId == image_name]

        # pick two classes (both involve water as object of detection)
        roads = shapes[shapes.ClassType == CLASS_TYPE]

        # obtain scaling constants for that image
        x_max = grid_sizes[grid_sizes.IMG == image_name].Xmax.values[0]
        y_min = grid_sizes[grid_sizes.IMG == image_name].Ymin.values[0]

        # NOW
        x_scale = float(img_16m.shape[1] ** 2) / float((img_16m.shape[1] + 1) * x_max)
        y_scale = float(img_16m.shape[0] ** 2) / float((img_16m.shape[0] + 1) * y_min)

        x_scale2 = float(img_16rgb.shape[1] ** 2) / float((img_16rgb.shape[1] + 1) * x_max)
        y_scale2 = float(img_16rgb.shape[0] ** 2) / float((img_16rgb.shape[0] + 1) * y_min)

        # collect tree-objects in a list
        objects_on_that_image = []

        # loop over polygon trees, scale them to pixel coordinates and store them
        # print 'Collecting polygons'
        # for multipolygon in lakes.MultipolygonWKT.values:
        #
        #     polygons = shape(loads(multipolygon))
        #
        #     # iterate over polygons in multipolygon
        #     for polygon in polygons.geoms:
        #
        #         # scale polygon to pixels coordinates
        #         # polygon_scaled = transform(lambda x, y: (y * y_scale, x * x_scale), polygon)
        #         polygon_scaled = transform(lambda x, y: (x * x_scale, y * y_scale), polygon)
        #
        #         objects_on_that_image.append(polygon_scaled)

        for multipolygon in roads.MultipolygonWKT.values:

            polygons = shape(loads(multipolygon))

            # iterate over polygons in multipolygon
            for polygon in polygons.geoms:

                # scale polygon to pixels coordinates
                # polygon_scaled = transform(lambda x, y: (y * y_scale, x * x_scale), polygon)
                polygon_scaled = transform(lambda x, y: (x * x_scale, y * y_scale), polygon)

                objects_on_that_image.append(polygon_scaled)

        if PLOT_POLYGONS:
            plt.subplot(121)
            plt.imshow(img_16rgb)
            plt.title('Image size: {}'.format(img_16rgb.shape))
            ax = plt.gca()

            for object_ in objects_on_that_image:
                # rescale to the biggest image
                object_ = transform(lambda x,y: (x * x_scale2 / x_scale, y * y_scale2 / y_scale), object_)
                pol = Pol(np.array(object_.exterior), color='g', alpha=0.35)
                ax.add_patch(pol)

            plt.subplot(122)
            plt.imshow(img_16m)
            plt.title('Image size: {}'.format(img_16m.shape))
            ax = plt.gca()

            for object_ in objects_on_that_image:
                # rescale to the biggest image
                pol = Pol(np.array(object_.exterior), color='g', alpha=0.35)
                ax.add_patch(pol)

            plt.tight_layout()
            # plt.show()

            plt.savefig(os.path.join(TRAIN_IMG_DIR, '{}.png'.format(image_name)), format='png', dpi=500)

            plt.clf()
            plt.cla()

        # print 'Generating black and white masks'
        # black and white image. Generate black image (all zeros) and
        # paint pixels inside polygons into white color (BIT_MAX=256 for 8bit color)
        baw_image = np.zeros(shape=img_16m.shape)

        pixels_per_pool = baw_image.shape[0] / PROCESSES

        # print 'Pixels per process: ', pixels_per_pool

        input_data = []
        print img_16m.shape

        # prepare the data
        for i in range(PROCESSES-1):
            input_data.append((baw_image[i*pixels_per_pool:(i+1)*pixels_per_pool, :],
                               objects_on_that_image,
                               i*pixels_per_pool))
        input_data.append((baw_image[(PROCESSES-1)*pixels_per_pool:, :],
                           objects_on_that_image,
                           (PROCESSES-1)*pixels_per_pool))

        res = pool.map(decolorization, input_data)

        baw_image = np.vstack(res)

        misc.toimage(baw_image).save(os.path.join(MASK_DIR, '{}-mask.png'.format(image_name)))
