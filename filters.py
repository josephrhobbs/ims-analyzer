# IMS Analyzer
# Image filters

import cv2
import numpy as np

class Filter(object):
    """
    A generic image filter.  All filters inherit behavior from `Filter`.
    """
    def __init__(self, args=None):
        self.args = args

    def __call__(self, image):
        """
        Evaluate this filter on a given image, optionally including
            arguments such as thresholds
        """
        # Resultant image channels
        result = {}

        # Optionally include arguments
        if self.args is not None:
            for name, channel in image.items():
                arg = self.args[name]
                if arg is None:
                    continue
                result[name] = self.filter(channel, arg)
        else:
            for name, channel in image.items():
                result[name] = self.filter(channel)

        return result

class Normalize(Filter):
    """
    Apply min-max normalization to 16-bit images.
    """
    def filter(self, image):
        # Normalize each channel
        image_min = np.min(image)
        image_max = np.max(image)
        normed = (image - image_min) / (image_max - image_min)

        # Discretize to 16 bits
        return (65535 * normed).astype(np.uint16)

class Percentiles(Filter):
    """
    Apply percentile scaling to 16-bit images.
    """
    def filter(self, image, pctl):
        min_pctl, max_pctl = pctl
        min_value = np.percentile(image, min_pctl)
        max_value = np.percentile(image, max_pctl)
        return np.clip(
            65535 * (image - min_value) / (max_value - min_value),
            0,
            65535,
        ).astype(np.uint16)

class InRange(Filter):
    """
    Apply a binary mask corresponding to pixels of a 16-bit image in a given range.
    """
    def filter(self, image, thresholds):
        return 257 * cv2.inRange(image, *thresholds).astype(np.uint16)

class Area(Filter):
    """
    Determine the area of non-zero pixels in a given 16-bit image.
    """
    def filter(self, image, res):
        x_res, y_res = res
        return cv2.countNonZero(image) * x_res * y_res

class Watershed(Filter):
    """
    Apply watershedding to an image.
    """
    def filter(self, image, pctl=20):
        # Open image
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(
            image,
            cv2.MORPH_OPEN,
            kernel,
            iterations=2,
        )

        # Apply distance transform
        distance_transform = cv2.distanceTransform(
            opening,
            cv2.DIST_L2,
            5,
        )
        nonzero_distances = distance_transform.flatten()[distance_transform.flatten() != 0]

        # Apply cutoff distance
        cutoff_distance = np.percentile(
            nonzero_distances if nonzero_distances.size else distance_transform,
            pctl,
        )
        _, distance_transform = cv2.threshold(
            distance_transform,
            cutoff_distance,
            255,
            cv2.THRESH_BINARY,
        )
        distance_transform = (255 * distance_transform / np.max(distance_transform)).astype(np.uint8)

        # Count particles
        contours, _ = cv2.findContours(
            distance_transform.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )

        return len(contours)