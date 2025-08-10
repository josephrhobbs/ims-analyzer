# IMS Analyzer

from pathlib import Path

import cv2
from imaris_ims_file_reader.ims import ims
from matplotlib import pyplot as plt
import numpy as np

# Channel mapping
CHANNEL_MAPPING = {
    "nuclei":   0,
    "vimentin": 1,
    "f-actin":  2,
    "edu":      3,
}

# Color mapping (BGR)
COLOR_MAPPING = {
    "nuclei":   (1, 0, 0),
    "vimentin": (0, 0, 1),
    "f-actin":  (0, 1, 0),
    "edu":      (1, 1, 1),
}

# Contrast percentiles
CONTRAST_PERCENTILES = {
    "nuclei":   (  5,  95),
    "vimentin": (  0, 100),
    "f-actin":  (  5,  80),
    "edu":      (  0, 100),
}

# Channel-specific thresholds (post-contrast)
CHANNEL_THRESHOLDS = {
    "nuclei":   (30_000, 65_535),
    "vimentin": (60_000, 65_535),
    "f-actin":  (60_000, 65_535),
    "edu":      (60_000, 65_535),
}

# Resolution level (lower is better)
RESOLUTION = 0

# Timepoint
TIMEPOINT = 0

# Z layer
Z_LAYER = 6

# Display size
DISPLAY_SIZE = 512, 512

# Histogram bins
HISTOGRAM_BINS = 100

def analyze_image(filepath):
    """
    Analyze an IMS microscopy image, given its file path
    """
    # Open IMS
    im = ims(filepath)

    # Microns per pixel (X and Y directions)
    x_res, y_res = im.metaData[RESOLUTION, TIMEPOINT, 0, "resolution"][1:3]

    # Unnormalized 16-bit grayscale channels
    raw_channels = {
        channel_name: im[TIMEPOINT, idx, Z_LAYER, :, :]
        for channel_name, idx in CHANNEL_MAPPING.items()
    }

    # Normalized 8-bit grayscale channels
    channels = {}

    ############################
    # NORMALIZE AND DISCRETIZE #
    ############################

    for name, image in raw_channels.items():
        # Normalize image
        image_min = np.min(image)
        image_max = np.max(image)
        delta     = image_max - image_min
        normed = (image - image_min) / delta

        # Discretize image
        discretized = (65535 * normed).astype(np.uint16)

        # Add channel to dictionary
        channels[name] = discretized

    #################################
    # APPLY CONTRAST BY PERCENTILES #
    #################################

    for name, image in channels.items():
        min_pctl, max_pctl = CONTRAST_PERCENTILES[name]
        min_value          = np.percentile(image, min_pctl)
        max_value          = np.percentile(image, max_pctl)
        channels[name] = np.clip(
            65535 * (image - min_value) / (max_value - min_value),
            0,
            65535,
        ).astype(np.uint16)

    ####################
    # APPLY THRESHOLDS #
    ####################

    for name, image in channels.items():
        thresholds = CHANNEL_THRESHOLDS[name]
        if thresholds is not None:
            channels[name] = 257 * cv2.inRange(image, *thresholds).astype(np.uint16)

    ############
    # COLORIZE #
    ############

    # # False color channels
    # false_color_channels = {}

    # for name, image in channels.items():
    #     # Get BGR color for this channel
    #     color = COLOR_MAPPING[name]

    #     # Create false color 16-bit image
    #     colored = np.zeros_like(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
    #     colored[:] = color
    #     colored = colored.astype(np.float64)
    #     print(image.astype(np.float64))
    #     for i in range(3):
    #         colored[:, :, i] *= image.astype(np.float64)
    #     colored = colored.astype(np.uint16)

    #     # Add false color channel to dictionary
    #     false_color_channels[name] = colored

    # TODO
    #  1. Find area for vimentin (um2)
    #  2. Find area for f-actin (um2)
    #  3. Find particle count for nuclei (watershed algorithm)
    #  4. Find particle count for edu (watershed algorithm)
    #  5. Find % edu in f-actin
    #  6. Find % edu in vimentin
    #  7. Find % nuclei in f-actin
    #  8. Find % nuclei in vimentin
    #  9. Find total area of bundle (um2; use perimeter of composite)
    # 10. Save JPG of false-color composite (with 100 um scale bar!)
    # 11. Put 1 through 9 in a spreadsheet :)

    return channels

if __name__ == "__main__":
    # Analyze all images
    images = analyze_image(Path("./test.ims"))

    # TODO formalize
    # fig, axs = plt.subplots(4, sharex=True)
    # for i, (name, image) in enumerate(images.items()):
    #     axs[i].hist(image.flatten(), bins=HISTOGRAM_BINS)
    #     axs[i].set_title(name)
    # axs[-1].set_xlabel("Value")
    # fig.suptitle("Pixel Histogram")
    # plt.show()

    # TODO remove?
    # Show all images
    for name, image in images.items():
        cv2.imshow(
            name,
            cv2.resize(image, DISPLAY_SIZE),
        )
    cv2.waitKey()