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
        self.hog_channel = 'ALL'  # 'ALL'  # 0, 1, 2, or "ALL"
        self.hist_range = (0, 256)
        self.cspace = 'YCrCb'  # RGB, HSV, LUV, HLS, YUV, YCrCb
        self.channels=3

        # sliding window search params
        self.window_size = 64  # as size of training images is 64x64
        self.cells_per_step = 2
        self.window_color = (0, 0, 255)
        self.window_thickness = 3
        self.scale = 1.7
        self.skip_frames=2

        # camera calibration and classifier pickles
        self.classifier = "../classifier.p"
        self.calibration_parameters = "../calibration_parameters.p"

        # ROI
        self.xy_start_stop_left = (0, 400), (370, 600)
        self.xy_start_stop_top = (400, 800), (380, 560)
        self.xy_start_stop_right = (800, 1270), (370, 600)

        # heatmap
        self.history_limit=8
        self.threshold = 2
        # training datasets
        self.training_not_cars = "../training_datasets/non-vehicles/*/*.png"
        self.training_cars = "../training_datasets/vehicles/*/*.png"
        self.training_not_cars_small = "../training_datasets_small/non-vehicles_smallset/*/*.jpeg"
        self.training_cars_small = "../training_datasets_small/vehicles_smallset/*/*.jpeg"

        # testing datasets
        self.testing_video = "../test_video.mp4"
        self.testing_video_2 = "../test_video_2.avi"
        self.project_video = "../project_video.mp4"
