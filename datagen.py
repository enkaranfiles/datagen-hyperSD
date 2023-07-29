from abc import ABC, abstractmethod
import numpy as np
import cv2
import random
from torchvision.transforms import Compose
import cv2
import torch

class Context:
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
    """The Strategy interface declares operations common to all supported versions of some algorithm."""

    @abstractmethod
    def do_process(self, image_path):
        pass


class CannyStrategy(Strategy):
    def do_process(self, image):
        low_threshold = random.randint(0, 255)
        high_threshold = random.randint(low_threshold, 255)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        return edges

class HEDStrategy(Strategy):
    def do_process(self, image_path):
        #leaved blanked for later implementation
        pass


class DepthStrategy(Strategy):
        def do_process(self, image):
            model_type = "DPT_Large"  # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
            # model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
            # model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
            midas = torch.hub.load("intel-isl/MiDaS", model_type)
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            midas.to(device)
            midas.eval()

            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

            if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
                transform = midas_transforms.dpt_transform
            else:
                transform = midas_transforms.small_transform

            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            input_batch = transform(img).to(device)

            with torch.no_grad():
                prediction = midas(input_batch)

                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            output = prediction.cpu().numpy()
            # return the depth map
            return output
