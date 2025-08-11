# IMS Analyzer

from pathlib import Path

import cv2
from imaris_ims_file_reader.ims import ims
from matplotlib import pyplot as plt
import numpy as np

# Channel mapping
CHANNEL_MAPPING = {
    "f-actin":  2,
    "vimentin": 1,
    "nuclei":   0,
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

# Composite channel-specific minimum thresholds (post-contrast)
COMPOSITE_THRESHOLDS = {
    "nuclei":   0.3,
    "vimentin": 0.3,
    "f-actin":  0.3,
    "edu":      0.3,
}

# Channel-specific thresholds (post-contrast)
CHANNEL_THRESHOLDS = {
    "nuclei":   (30_000, 65_535),
    "vimentin": (60_000, 65_535),
    "f-actin":  (60_000, 65_535),
    "edu":      (60_000, 65_535),
}

# Find area for this layer?
FIND_AREA = {
    "nuclei":   False,
    "vimentin": True,
    "f-actin":  True,
    "edu":      False,
}

# Overlaps
OVERLAPS = [
    ("nuclei", "f-actin"),
    ("nuclei", None),
    ("nuclei", "vimentin"),
    ("edu", None),
    ("edu", "vimentin"),
    ("edu", "f-actin"),
]

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

# Opening kernel size
OPEN_SIZE = 7

# Distance cutoff percentile (watershedding)
# NOTE: in this calculation, zero values are ignored
DISTANCE_CUTOFF_PERCENTILE = 20

# Threshold ratio for determining overlap
OVERLAP_THRESHOLD = 0.5

# Scale bar dimensions
SCALE_BAR_LENGTH = 100 # um
SCALE_BAR_HEIGHT_PX = 10 # px
SCALE_BAR_FONT_SIZE = 2

# Scale bar offsets
SCALE_BAR_X_OFFSET = 50
SCALE_BAR_Y_OFFSET = 100

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

    # Save channels at this point for false color
    pretty_channels = channels.copy()
    for name, image in pretty_channels.items():
        _, image = cv2.threshold(
            image,
            65535*COMPOSITE_THRESHOLDS[name],
            255,
            cv2.THRESH_TOZERO,
        )
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = np.uint16(65535 * image)
        pretty_channels[name] = image

    ####################
    # APPLY THRESHOLDS #
    ####################

    for name, image in channels.items():
        thresholds = CHANNEL_THRESHOLDS[name]
        if thresholds is not None:
            channels[name] = 257 * cv2.inRange(image, *thresholds).astype(np.uint16)

    ##################
    # ANALYZE IMAGES #
    ##################

    # Determine area (um2) of each layer
    areas = {}
    for name, image in channels.items():
        # Get area of each layer, if necessary
        if FIND_AREA[name]:
            areas[name] = cv2.countNonZero(image) * x_res * y_res

    # Count particles in each layer, if necessary
    particle_counts = {}
    particle_fractions = {}
    for particles, outer in OVERLAPS:
        # Get pertinent image
        image = channels[particles]

        # Mask with other image, if necessary
        # cv2.imshow(f"{particles} / before watershedding", cv2.resize(image, DISPLAY_SIZE))
        if outer is not None:
            outer_image = (channels[outer] / 257).astype(np.uint8)
            _, outer_image = cv2.threshold(
                outer_image,
                OVERLAP_THRESHOLD*np.max(outer_image),
                255,
                cv2.THRESH_BINARY,
            )
            image = cv2.bitwise_and(
                image,
                image,
                mask=outer_image,
            )
            # cv2.imshow(f"{outer} / mask", cv2.resize(outer_image, DISPLAY_SIZE))

        # Erode image
        kernel = np.ones((OPEN_SIZE, OPEN_SIZE), np.uint8)
        opening = cv2.morphologyEx(
            image,
            cv2.MORPH_OPEN,
            kernel,
            iterations=2,
        )

        # Apply distance transform
        distance_transform = cv2.distanceTransform(
            (opening / 257).astype(np.uint8),
            cv2.DIST_L2,
            5,
        )
        cutoff_distance = np.percentile(
            distance_transform.flatten()[distance_transform.flatten() != 0],
            DISTANCE_CUTOFF_PERCENTILE,
        )
        _, distance_transform = cv2.threshold(
            distance_transform,
            cutoff_distance,
            255,
            cv2.THRESH_BINARY
        )
        distance_transform = (255 * distance_transform / np.max(distance_transform)).astype(np.uint8)

        # Count particles
        contours, _ = cv2.findContours(
            distance_transform.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        particle_counts[particles, outer] = len(contours)

        # cv2.imshow(f"{particles} / {outer}", cv2.resize(distance_transform, DISPLAY_SIZE))
        # cv2.waitKey()

    # Calculate fractions
    for (particles, outer), count in particle_counts.items():
        if outer is not None:
            particle_fractions[particles, outer] = count / particle_counts[particles, None]

    ############
    # COLORIZE #
    ############

    # False color image
    false_color_image = np.zeros_like(
        cv2.cvtColor(
            list(pretty_channels.values())[0],
            cv2.COLOR_GRAY2BGRA,
        )
    )

    for name, image in pretty_channels.items():
        # Get BGR color for this channel
        color = COLOR_MAPPING[name]

        # Create false color 16-bit image
        colored = np.full(image.shape + (3,), color)
        colored = colored.astype(np.float64)
        for i in range(3):
            colored[:, :, i] *= image.astype(np.float64)
        colored = colored.astype(np.uint16)

        # Add alpha channel and make black pixels transparent
        alpha = np.sum(colored, axis=-1) > 0
        colored = np.dstack((colored, np.uint16(alpha * 65535)))

        # Add false color channel to image
        false_color_image = np.where(
            np.dstack((colored[:, :, 3] != 0,)*4),
            colored,
            false_color_image,
        )

    # Draw contour around false color
    grayscale_false_color = np.uint8((np.sum(false_color_image, axis=-1) != 0) * 255)
    contours, _ = cv2.findContours(
        grayscale_false_color.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    c = max(contours, key=cv2.contourArea)
    areas["section"] = cv2.contourArea(c) * x_res * y_res

    # Add scale bar
    scale_bar_px = int(SCALE_BAR_LENGTH / x_res)
    x_end = false_color_image.shape[0] - SCALE_BAR_X_OFFSET
    x_start = x_end - scale_bar_px
    y_bar = false_color_image.shape[1] - SCALE_BAR_Y_OFFSET
    cv2.line(
        false_color_image,
        (x_start, y_bar),
        (x_end, y_bar),
        (65535,)*4,
        SCALE_BAR_HEIGHT_PX,
    )
    (text_width, text_height), _ = cv2.getTextSize(
        f"{SCALE_BAR_LENGTH} um",
        cv2.FONT_HERSHEY_SIMPLEX, 
        SCALE_BAR_FONT_SIZE,
        SCALE_BAR_HEIGHT_PX,
    )
    x_text, y_text = x_start + (x_end - x_start) // 2 - text_width // 2, y_bar + 3 * text_height // 2
    cv2.putText(
        false_color_image,
        f"{SCALE_BAR_LENGTH} um",
        (x_text, y_text),
        cv2.FONT_HERSHEY_SIMPLEX, 
        SCALE_BAR_FONT_SIZE,
        (65535,)*4,
        SCALE_BAR_HEIGHT_PX,
    )

    # Convert false color to 8-bit unsigned
    false_color_image = np.uint8(false_color_image / 257)[:, :, :3]

    return false_color_image, areas, particle_counts, particle_fractions
