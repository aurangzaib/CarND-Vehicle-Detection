import time

import cv2 as cv
import matplotlib.image as mpimg
import numpy as np
from skimage.feature import hog

from configuration import Configuration

config = Configuration().__dict__


class FeatureExtraction:
    @staticmethod
    def bin_spatial(img, size=(32, 32)):
        color1 = cv.resize(img[:, :, 0], size).ravel()
        color2 = cv.resize(img[:, :, 1], size).ravel()
        color3 = cv.resize(img[:, :, 2], size).ravel()
        return np.hstack((color1, color2, color3))

    @staticmethod
    def color_hist(img, nbins=32):  # bins_range=(0, 256)
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:, :, 0], bins=nbins)[0]
        channel2_hist = np.histogram(img[:, :, 1], bins=nbins)[0]
        channel3_hist = np.histogram(img[:, :, 2], bins=nbins)[0]
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist, channel2_hist, channel3_hist))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features

    @staticmethod
    def get_hog_features(img, feature_vec=False, folder="", filename=None):
        # Call with two outputs if vis==True
        features = hog(img,
                       orientations=config["orient"],
                       pixels_per_cell=(config["pix_per_cell"],
                                        config["pix_per_cell"]),
                       cells_per_block=(config["cell_per_block"],
                                        config["cell_per_block"]),
                       transform_sqrt=True,
                       visualise=False,
                       feature_vector=feature_vec)
        return features

    @staticmethod
    def extract_features(img_files):
        from helper import Helper
        """
        combine spatial bin, color histogram and gradient histogram features
        """
        # Create a list to append feature vectors to
        features = []
        # Iterate through the list of images
        for img_file in img_files:

            # Read in each one by one
            img = mpimg.imread(img_file)

            # apply color conversion if other than 'RGB'
            feature_image = Helper.change_cspace(img)

            # get hog features for either specific channel or for all channels
            if config["hog_channel"] == 'ALL':
                hog_features = []
                # get features for all 3 channels
                seconds = int(time.time() % 60)
                for channel in range(feature_image.shape[2]):
                    single_channel_img = feature_image[:, :, channel]
                    filename = "{}-channel-{}".format(seconds, str(channel))
                    hog_features.append(FeatureExtraction.get_hog_features(single_channel_img,
                                                                           folder="../buffer/hog-train-features/",
                                                                           filename=filename,
                                                                           feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                # get features for specific channel
                single_channel_img = feature_image[:, :, config["hog_channel"]]
                hog_features = FeatureExtraction.get_hog_features(single_channel_img, feature_vec=True)

            # Apply bin_spatial() to get spatial color features
            bin_features = FeatureExtraction.bin_spatial(feature_image, config["spatial_size"])

            # Apply color_hist() to get color histogram features
            color_hist_features = FeatureExtraction.color_hist(feature_image, config["hist_bins"])

            # concatenate all 3 types of features
            feature = np.concatenate((bin_features, color_hist_features, hog_features), axis=0)

            # Append the new feature vector to the features list
            features.append(feature)

        # Return list of feature vectors
        return features
