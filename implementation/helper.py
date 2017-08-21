import cv2 as cv
import numpy as np
from scipy.ndimage.measurements import label

from configuration import Configuration

config = Configuration().__dict__


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
    def change_cspace(img):
        feature_image = []
        cspace = config["cspace"]
        if config["cspace"] != 'RGB':
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
        img_with_boxes = np.copy(img)
        for box in boxes:
            cv.rectangle(img_with_boxes,
                         box[0], box[1],
                         color, thick)
        return img_with_boxes

    @staticmethod
    def add_heat(heatmap, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        # Return updated heatmap
        return heatmap

    @staticmethod
    def get_heatmap(heat):
        # Apply threshold to help remove false positives
        heat_binary = Helper.apply_threshold(heat, 1)

        # Visualize the heatmap when displaying
        heatmap_binary = np.clip(heat_binary, 0, 1)

        return heatmap_binary

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
        from visualization import Visualization
        import matplotlib.pyplot as plt
        heat = np.zeros_like(img[:, :, 0]).astype(np.float)

        # Add heat to each box in box list
        heat = Helper.add_heat(heat, bounding_boxes)

        # Get binary heat map
        heatmap_binary = Helper.get_heatmap(heat)

        # Find final boxes from heatmap using label function
        labels = label(heatmap_binary)

        # show box where label is 1
        detected_cars = Helper.draw_labeled_bboxes(np.copy(img), labels)

        # save heatmaps
        if config["save_debug_samples"] is True:
            plt.imshow(heat, cmap="gist_heat")
            Visualization.save_heat_map(heat)

        return detected_cars
