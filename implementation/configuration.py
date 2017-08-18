class Configuration:
    # hyper-parameters for vehicle detection
    def __init__(self):
        self.orient = 10
        self.pix_per_cell = 8
        self.cell_per_block = 2
        self.spatial_size = (32, 32)
        self.hist_bins = 32
        self.hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
        self.scale = 1.5
        self.hist_range = (0, 256)
        self.cspace = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.xy_window = (96, 96)
        self.xy_overlap = (0.5, 0.5)
        self.with_spatial_feature = True
        self.with_color_feature = True
        self.with_gradient_feature = True
        self.window_color = (0, 0, 255)
        self.window_thickness = 3
        self.training_not_cars = "../training_datasets/non-vehicles/*/*.png"
        self.training_cars = "../training_datasets/vehicles/*/*.png"
        self.testing = "../test_images/*.jpg"

    def get_config_object(self):
        """
        return hyper-parameters for the vehicle detection
        """
        return self
