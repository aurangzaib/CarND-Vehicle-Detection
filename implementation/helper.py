import cv2 as cv
import matplotlib.image as mpimg
import numpy as np
from scipy.ndimage.measurements import label
from skimage.feature import hog

from configuration import Configuration

hyper_params = Configuration().__dict__


class Helper:
    @staticmethod
    def scale_to_png(img):
        return img.astype(np.float32) / 255

    @staticmethod
    def convert_color(img, conv='RGB2YCrCb'):
        if conv == 'RGB2YCrCb':
            return cv.cvtColor(img, cv.COLOR_RGB2YCrCb)
        if conv == 'BGR2YCrCb':
            return cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
        if conv == 'RGB2LUV':
            return cv.cvtColor(img, cv.COLOR_RGB2LUV)

    @staticmethod
    def change_cspace(img, cspace):
        feature_image = []
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv.cvtColor(img, cv.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv.cvtColor(img, cv.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv.cvtColor(img, cv.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv.cvtColor(img, cv.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv.cvtColor(img, cv.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(img)

        return feature_image

    @staticmethod
    def draw_boxes(img, boxes, color=(0, 0, 0), thick=3):
        img_pts_drawn = np.copy(img)
        for box in boxes:
            cv.rectangle(img_pts_drawn, box[0], box[1], color, thick)
        return img_pts_drawn

    @staticmethod
    def get_hog_features(img, feature_vec=False, save_hog_features=False):
        from contrib import Contrib
        # Call with two outputs if vis==True
        features, hog_image = hog(img,
                                  orientations=hyper_params["orient"],
                                  pixels_per_cell=(hyper_params["pix_per_cell"], hyper_params["pix_per_cell"]),
                                  cells_per_block=(hyper_params["cell_per_block"], hyper_params["cell_per_block"]),
                                  transform_sqrt=True,
                                  visualise=True,
                                  feature_vector=feature_vec)
        if save_hog_features is True:
            Contrib.save_hog_features(img, hog_image)
        return features

    @staticmethod
    def bin_spatial(img, size=(32, 32)):
        color1 = cv.resize(img[:, :, 0], size).ravel()
        color2 = cv.resize(img[:, :, 1], size).ravel()
        color3 = cv.resize(img[:, :, 2], size).ravel()
        return np.hstack((color1, color2, color3))

    @staticmethod
    def extract_features(img_files):
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
            feature_image = Helper.change_cspace(img, hyper_params["cspace"])

            # get hog features for either specific channel or for all channels
            if hyper_params["hog_channel"] == 'ALL':
                hog_features = []
                # get features for all 3 channels
                for channel in range(feature_image.shape[2]):
                    hog_features.append(Helper.get_hog_features(feature_image[:, :, channel],
                                                                feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                # get features for specific channel
                hog_features = Helper.get_hog_features(feature_image[:, :, hyper_params["hog_channel"]],
                                                       feature_vec=True)

            # Apply bin_spatial() to get spatial color features
            bin_features = Helper.bin_spatial(feature_image, hyper_params["spatial_size"])

            # Apply color_hist() to get color histogram features
            color_hist_features = Helper.color_hist(feature_image, hyper_params["hist_bins"])

            # concatenate all 3 types of features
            feature = np.concatenate((bin_features, color_hist_features, hog_features), axis=0)

            # Append the new feature vector to the features list
            features.append(feature)

        # Return list of feature vectors
        return features

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
    def add_heat(heatmap, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        # Return updated heatmap
        return heatmap  # Iterate through list of bboxes

    @staticmethod
    def apply_threshold(heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        return heatmap

    @staticmethod
    def draw_labeled_bboxes(img, labels):
        to_png = 255
        # Iterate through all detected cars
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv.rectangle(img, bbox[0], bbox[1], (0 / to_png, 0 / to_png, 255 / to_png), 6)
        # Return the image
        return img

    @staticmethod
    def remove_false_positives(img, bounding_boxes):
        import matplotlib.pyplot as plt
        heat = np.zeros_like(img[:, :, 0]).astype(np.float)

        # Add heat to each box in box list
        heat = Helper.add_heat(heat, bounding_boxes)

        # Apply threshold to help remove false positives
        heat = Helper.apply_threshold(heat, 1)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 1)
        plt.imshow(heatmap, cmap="gray")

        # Find final boxes from heatmap using label function
        labels = label(heatmap)

        detected_cars = Helper.draw_labeled_bboxes(np.copy(img), labels)

        plt.show()

        return detected_cars
