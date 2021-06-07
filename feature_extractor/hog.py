from feature_extractor.feature_extractor_base import Feature_Extractor_Base
from skimage.feature import hog


class HOG_feature_extractor(Feature_Extractor_Base):
    def __init__(self, config: dict):
        super().__init__(config)
        self.min_window_size = config['min_window_size']
        self.step_size = config['step_size']
        self.orientations = config['orientations']
        self.pixel_per_cell = config['pixel_per_cell']
        self.cell_per_block = config['cell_per_block']
        self.visualize = config['visualize']
        self.transform_sqrt = config['transform_sqrt']

    def get_feature(self, image):
        """
        :param image: numpy array
        :return: fd: feature of image
        """
        return hog(image, self.orientations, self.pixel_per_cell, self.cell_per_block, visualize=self.visualize,
                   transform_sqrt=self.transform_sqrt)
