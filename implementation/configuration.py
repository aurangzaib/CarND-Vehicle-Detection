class Configuration:
    # hyper-parameters for vehicle detection
    def __init__(self):
        # flags
        self.is_training_png = True
        self.save_debug_samples = False

        # which features to include
        self.with_spatial_feature = True
        self.with_color_feature = True
        self.with_gradient_feature = True

        # HOG params
        self.orient = 10
        self.pix_per_cell = 8
        self.cell_per_block = 2

        # spatial and color params
        self.spatial_size = (32, 32)
        self.hist_bins = 32
        self.hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
        self.hist_range = (0, 256)
        self.cspace = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb

        # sliding window search params
        self.scale = 1.7
        self.window_color = (0, 0, 255)
        self.window_thickness = 3

        # camera calibration and classifier pickles
        self.classifier = "../trained_classifier_new.p"
        self.calibration_parameters = "../calibration_parameters.p"

        # training datasets
        self.training_not_cars = "../training_datasets/non-vehicles/*/*.png"
        self.training_cars = "../training_datasets/vehicles/*/*.png"

        # testing datasets
        self.testing_video = "../test_video.mp4"
        self.testing_video_2 = "../test_video_2.avi"
        self.project_video = "../project_video.mp4"
