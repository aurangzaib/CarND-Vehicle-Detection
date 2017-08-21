import pickle

import cv2 as cv
import numpy as np

from configuration import Configuration

config = Configuration().__dict__


class PreProcessing:
    @staticmethod
    def save_calibration_params(camera_matrix, dist_coef, filename=config["calibration_parameters"]):
        """
        save the matrix and coef in a pickle file
        :param camera_matrix: camera matrix found using cv.calibrateCamera
        :param dist_coef: distortion coef found using cv.calibrateCamera
        :param filename: to store the pickle file
        :return: None
        """
        parameters = {
            'camera_matrix': camera_matrix,
            'dist_coef': dist_coef
        }
        pickle.dump(parameters, open(filename, "wb"))
        print("parameters saved to disk")

    @staticmethod
    def load_calibration_params(filename=config["calibration_parameters"]):
        """
        load pickle files for train, validation and test
        :param filename: to read the pickle file
        :return: calibration params
        """
        with open(filename, mode='rb') as f:
            parameters = pickle.load(f)
        return parameters['camera_matrix'], parameters['dist_coef']

    @staticmethod
    def get_undistorted_image(nx, ny, img, camera_matrix, dist_coef, load_params=True):
        """
        Using cv.undistort with calibration params as arguments, get the undistorted image
        :param nx: number of corners in x direction
        :param ny: number of corners in y direction
        :param camera_matrix: camera matrix found using cv.calibrateCamera
        :param dist_coef: distortion coef found using cv.calibrateCamera
        :param img:source image
        :param load_params: flag to load or find calibration params
        :return: undistorted image
        """

        # undistorted image
        undistorted = cv.undistort(src=img,
                                   cameraMatrix=camera_matrix,
                                   distCoeffs=dist_coef,
                                   dst=None,
                                   newCameraMatrix=camera_matrix)

        return undistorted

    @staticmethod
    def get_binary_image(img, sx_thresh=(20, 200), rgb_thresh=(170, 255), hls_thresh=(120, 255)):
        """
        apply color and gradient threshold
        color threshold --> R channel and S channel binarized
        gradient threshold --> sobel x binarized
        :param img: undistorted image
        :param sx_thresh: threshold range for sobel x
        :param rgb_thresh: threshold range for r channel in RGB
        :param hls_thresh: threshold range for s channel in HLS
        :return: binary image
        """
        is_binary_debug_enabled = False
        # sx_thresh=(40, 180), rgb_thresh=(190, 255), hls_thresh=(100, 255)
        # grayscale
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        gray_binary = np.zeros_like(gray)
        gray_binary[(gray >= 20) & (gray <= 80)] = 1

        # sobelx gradient threshold
        dx, dy = (1, 0)
        sx = cv.Sobel(gray, cv.CV_64F, dx, dy, ksize=9)
        sx_abs = np.absolute(sx)
        sx_8bit = np.uint8(255 * sx_abs / np.max(sx_abs))
        sx_binary = np.zeros_like(sx_8bit)
        sx_binary[(sx_8bit > sx_thresh[0]) & (sx_8bit <= sx_thresh[1])] = 1

        # RGB color space
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        r_binary = np.zeros_like(r)
        r_binary[(r >= rgb_thresh[0]) & (r <= rgb_thresh[1])] = 1

        # HLS color space
        hls = cv.cvtColor(img, cv.COLOR_RGB2HLS)
        h, l, s = hls[:, :, 0], hls[:, :, 1], hls[:, :, 2]
        s_binary = np.zeros_like(s)
        s_binary[(s >= hls_thresh[0]) & (s <= hls_thresh[1])] = 1

        # resultant of r, s and sx
        binary_image = np.zeros_like(sx_binary)
        binary_image[((sx_binary == 1) | (s_binary == 1)) & (r_binary == 1)] = 1

        return binary_image
