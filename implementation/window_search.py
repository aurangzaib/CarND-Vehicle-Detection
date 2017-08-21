import time

import cv2 as cv
import numpy as np

from configuration import Configuration
from feature_extraction import FeatureExtraction
from helper import Helper

config = Configuration().__dict__


class WindowSearch:
    @staticmethod
    def get_box(x_left, y_start, y_top, window):
        x_box_left = np.int(x_left * config["scale"])
        y_top_draw = np.int(y_top * config["scale"])
        win_draw = np.int(window * config["scale"])

        box = [
            (x_box_left, y_top_draw + y_start),
            (x_box_left + win_draw, y_top_draw + win_draw + y_start)
        ]
        return box

    @staticmethod
    def get_frame_hog(img, y_start_top):

        y_start, y_stop = y_start_top

        img_tosearch = img[y_start:y_stop, :, :]
        ctrans_tosearch = Helper.convert_color(img_tosearch, conv='RGB2YCrCb')
        if config["scale"] != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv.resize(ctrans_tosearch,
                                        (np.int(imshape[1] / config["scale"]),
                                         np.int(imshape[0] / config["scale"])))

        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]

        # Define blocks and steps as above
        n_xblocks = (ch1.shape[1] // config["pix_per_cell"]) - config["cell_per_block"] + 1
        n_yblocks = (ch1.shape[0] // config["pix_per_cell"]) - config["cell_per_block"] + 1

        # 64 was the original sampling rate, with 8 cells and 8 pix per cell
        window = 64
        n_blocks_per_window = (window // config["pix_per_cell"]) - config["cell_per_block"] + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        n_xsteps = (n_xblocks - n_blocks_per_window) // cells_per_step
        n_ysteps = (n_yblocks - n_blocks_per_window) // cells_per_step

        # Compute individual channel HOG features for the entire image
        # Y channel
        hog1 = FeatureExtraction.get_hog_features(ch1, folder="../buffer/hog-features/")
        # Cr  channel
        hog2 = FeatureExtraction.get_hog_features(ch2)
        # Cb channel
        hog3 = FeatureExtraction.get_hog_features(ch3)
        return (hog1, hog2, hog3), (n_xsteps, n_ysteps), cells_per_step, (window, n_blocks_per_window, ctrans_tosearch)

    @staticmethod
    # Define a single function that can extract features using hog sub-sampling and
    # make predictions
    def get_bounding_boxes(img, classifier, y_start_top):

        # get hog features for full image
        hog, n_steps, cells_per_step, window = WindowSearch.get_frame_hog(img, y_start_top)

        svc, scaler = classifier
        hog1, hog2, hog3 = hog
        n_xsteps, n_ysteps = n_steps
        window, n_blocks_per_window, ctrans_tosearch = window
        y_start, y_stop = y_start_top
        bounding_boxes = []

        t_start = int(time.time())
        for xb in range(n_xsteps):
            for yb in range(n_ysteps):
                y_pos = yb * cells_per_step
                x_pos = xb * cells_per_step

                # Extract HOG for this patch
                hog_feat1 = hog1[y_pos:y_pos + n_blocks_per_window, x_pos:x_pos + n_blocks_per_window].ravel()
                hog_feat2 = hog2[y_pos:y_pos + n_blocks_per_window, x_pos:x_pos + n_blocks_per_window].ravel()
                hog_feat3 = hog3[y_pos:y_pos + n_blocks_per_window, x_pos:x_pos + n_blocks_per_window].ravel()

                x_left = x_pos * config["pix_per_cell"]
                y_top = y_pos * config["pix_per_cell"]

                # Extract the image patch
                sub_sample_img = cv.resize(ctrans_tosearch[y_top:y_top + window, x_left:x_left + window], (64, 64))

                # Get color and gradient features
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                spatial_features = FeatureExtraction.bin_spatial(sub_sample_img, size=config["spatial_size"])
                hist_features = FeatureExtraction.color_hist(sub_sample_img, nbins=config["hist_bins"])

                # append merge features
                feature = np.hstack((spatial_features, hist_features, hog_features))

                # normalize the features
                features = scaler.transform(np.array(feature).reshape(1, -1))

                # predict the label for the features: 1 = car, 0 = not car
                predicted_labels = svc.predict(features)

                # get the bounding box for detected cars
                if predicted_labels == 1:
                    bounding_boxes.append(WindowSearch.get_box(x_left, y_start, y_top, window))
        t_end = int(time.time())
        print("prediction time: {}".format(t_end - t_start))

        return bounding_boxes
