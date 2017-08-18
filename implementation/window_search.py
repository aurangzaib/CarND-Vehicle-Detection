import cv2 as cv
import numpy as np

from configuration import Configuration
from helper import Helper

hyper_params = Configuration().__dict__


class WindowSearch:
    @staticmethod
    # Define a single function that can extract features using hog sub-sampling and
    # make predictions
    def get_bounding_boxes(img, classifier, y_start_top):
        # region of interest (ROI)
        y_start, y_stop = y_start_top

        # classifier and scaler
        svc, scaler = classifier

        bounding_boxes = []

        img_tosearch = img[y_start:y_stop, :, :]
        ctrans_tosearch = Helper.convert_color(img_tosearch, conv='RGB2YCrCb')
        if hyper_params["scale"] != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv.resize(ctrans_tosearch,
                                        (np.int(imshape[1] / hyper_params["scale"]),
                                         np.int(imshape[0] / hyper_params["scale"])))

        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]

        # Define blocks and steps as above
        n_xblocks = (ch1.shape[1] // hyper_params["pix_per_cell"]) \
                    - hyper_params["cell_per_block"] + 1
        n_yblocks = (ch1.shape[0] // hyper_params["pix_per_cell"]) \
                    - hyper_params["cell_per_block"] + 1

        # 64 was the original sampling rate, with 8 cells and 8 pix per cell
        window = 64
        n_blocks_per_window = (window // hyper_params["pix_per_cell"]) - hyper_params["cell_per_block"] + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        n_xsteps = (n_xblocks - n_blocks_per_window) // cells_per_step
        n_ysteps = (n_yblocks - n_blocks_per_window) // cells_per_step

        # Compute individual channel HOG features for the entire image
        # Y channel
        hog1 = Helper.get_hog_features(ch1, save_hog_features=True)
        # Cr  channel
        hog2 = Helper.get_hog_features(ch2)
        # Cb channel
        hog3 = Helper.get_hog_features(ch3)

        for xb in range(n_xsteps):
            for yb in range(n_ysteps):
                y_pos = yb * cells_per_step
                x_pos = xb * cells_per_step

                # Extract HOG for this patch
                hog_feat1 = hog1[y_pos:y_pos + n_blocks_per_window, x_pos:x_pos + n_blocks_per_window].ravel()
                hog_feat2 = hog2[y_pos:y_pos + n_blocks_per_window, x_pos:x_pos + n_blocks_per_window].ravel()
                hog_feat3 = hog3[y_pos:y_pos + n_blocks_per_window, x_pos:x_pos + n_blocks_per_window].ravel()

                x_left = x_pos * hyper_params["pix_per_cell"]
                y_top = y_pos * hyper_params["pix_per_cell"]

                # Extract the image patch
                sub_sample_img = cv.resize(ctrans_tosearch[y_top:y_top + window,
                                           x_left:x_left + window],
                                           (64, 64))

                # Get color and gradient features
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                spatial_features = Helper.bin_spatial(sub_sample_img, size=hyper_params["spatial_size"])
                hist_features = Helper.color_hist(sub_sample_img, nbins=hyper_params["hist_bins"])

                feature = np.hstack((spatial_features, hist_features, hog_features))

                # normalize the features
                features = scaler.transform(np.array(feature).reshape(1, -1))

                # predict the label for the features: 1 = car, 0 = not car
                predicted_labels = svc.predict(features)

                if predicted_labels == 1:
                    x_box_left = np.int(x_left * hyper_params["scale"])
                    y_top_draw = np.int(y_top * hyper_params["scale"])
                    win_draw = np.int(window * hyper_params["scale"])

                    bounding_boxes.append([
                        (x_box_left, y_top_draw + y_start),
                        (x_box_left + win_draw, y_top_draw + win_draw + y_start)
                    ])

        return bounding_boxes
