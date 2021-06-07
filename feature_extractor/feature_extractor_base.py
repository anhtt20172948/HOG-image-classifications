from abc import ABC, abstractmethod


class Feature_Extractor_Base(ABC):
    def __init__(self, config):
        self.config = config
        pass

    @abstractmethod
    def get_feature(self, image):
        pass
