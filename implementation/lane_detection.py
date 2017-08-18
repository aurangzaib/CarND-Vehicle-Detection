from lanes_fitting import LanesFitting
from metrics import Metrics
from perspective_transform import PerspectiveTransform
from pre_processing import PreProcessing
from visualization import Visualization


class LaneDetection:
    @staticmethod
    def pipeline(img, lanes_fit, camera_matrix, dist_coef):
        # debug flag
        is_debug_enabled = True

        # checkbox dimensions for calibration
        nx, ny, channels = 9, 6, 3

        # calibrate camera and undistort the image
        undistorted_image = PreProcessing.get_undistorted_image(nx, ny, img, camera_matrix, dist_coef)

        # get the color and gradient threshold image
        binary_image = PreProcessing.get_binary_image(undistorted_image)

        # get source and destination points
        src, dst = PerspectiveTransform.get_perspective_points(img)

        # get image with source and destination points drawn
        img_src, img_dst = PerspectiveTransform.get_sample_wrapped_images(img, src, dst)

        # perspective transform to bird eye view
        warped_image = PerspectiveTransform.get_wrapped_image(binary_image, src, dst)

        # find the lanes lines and polynomial fit
        if len(lanes_fit) == 0:
            lane_lines, lanes_fit, left_xy, right_xy = LanesFitting.get_lanes_fit(warped_image)
        else:
            lane_lines, lanes_fit, left_xy, right_xy = LanesFitting.update_lanes_fit(warped_image, lanes_fit)

        # find the radius of curvature
        radius = Metrics.get_curvature_radius(lane_lines, left_xy, right_xy)

        # find the car distance from center lane
        center_distance, lane_width = Metrics.get_distance_from_center(lane_lines, lanes_fit)

        # unwrap the image
        resultant = PerspectiveTransform.get_unwrapped_image(undistorted_image, warped_image, src, dst, lanes_fit)

        # visualize the pipeline
        if is_debug_enabled is True:
            resultant = Visualization.visualize_pipeline(resultant, img_dst,
                                                         binary_image, lane_lines,
                                                         radius, center_distance,
                                                         lane_width)

        return lanes_fit, resultant
