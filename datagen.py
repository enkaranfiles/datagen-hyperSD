from abc import ABC, abstractmethod
import numpy as np
import cv2

class Context:
    """The Context defines the interface of interest to clients."""
    def __init__(self, strategy):
        self._strategy = strategy

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, strategy):
        self._strategy = strategy

    def process_image(self, image_path):
        return self._strategy.do_process(image_path)


class Strategy(ABC):
    """The Strategy interface declares operations common to all supported versions
    of some algorithm."""

    @abstractmethod
    def do_process(self, image_path):
        pass


class CannyStrategy(Strategy):
    def do_process(self, image_path, low_threshold=100, high_threshold=200):
        # read the image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # check if image reading is successful
        if image is None:
            print(f"Failed to load image at {image_path}")
            return None

        # convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # perform the canny edge detection
        edges = cv2.Canny(blurred, low_threshold, high_threshold)

        return edges


class HEDStrategy(Strategy):
    def do_process(self, image_path):
        # implement the HED here
        pass


class DepthStrategy(Strategy):
    def do_process(self, image_path):
        # implement the Midas for monocular depth estimation here
        pass
