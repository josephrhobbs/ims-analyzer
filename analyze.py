# IMS Analyzer

from pathlib import Path
from sys import argv

import cv2
from filters import *
from user_menu import *
from imaris_ims_file_reader.ims import ims
from matplotlib import pyplot as plt
import numpy as np
import tifffile
from tkinter import messagebox

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
    "vimentin": (  5,  95),
    "f-actin":  ( 40,  80),
    "edu":      (  0, 100),
}

# Distance cutoff percentile (watershedding)
# NOTE: in this calculation, zero values are ignored if they exist
DISTANCE_CUTOFF_PERCENTILE = {
    "nuclei":   20,
    "vimentin": 20,
    "f-actin":  20,
    "edu":      20,
}

# Resolution level (lower is better)
RESOLUTION = 0

# Timepoint
TIMEPOINT = 0

# Display size
DISPLAY_SIZE = 512, 512

# # Find area for this layer?
# FIND_AREA = {
#     "nuclei":   False,
#     "vimentin": True,
#     "f-actin":  True,
#     "edu":      False,
# }

# # Overlaps
# OVERLAPS = [
#     ("nuclei", "f-actin"),
#     ("nuclei", None),
#     ("nuclei", "vimentin"),
#     ("edu", None),
#     ("edu", "vimentin"),
#     ("edu", "f-actin"),
# ]

def analyze_image(filepath):
    """
    Analyze an IMS microscopy image, given its file path
    """
    # Open IMS
    im = ims(filepath)

    # Show Z-stack to user
    layers = []
    normalize = Normalize()
    for i in range(im.shape[2]):
        channels = {
            f"Channel {idx}": im[TIMEPOINT, idx, i, :, :]
            for idx in range(im.shape[1])
        }
        layers.append(normalize(channels))

    # Prompt user to select Z layer
    channels, z = select_z(layers)

    # Save TIFF image
    tif_filename = filepath.parent / (filepath.stem + ".tif")
    channels_list = list(channels.items())
    channels_list.sort(key=lambda item: item[0])
    channel_names = [name for name, channel in channels_list]
    tif = np.stack([cv2.cvtColor(channel, cv2.COLOR_GRAY2BGR) for name, channel in channels_list], axis=0)
    tifffile.imwrite(
        tif_filename,
        tif,
        metadata={
            "Channel": {"Name": channel_names},
        },
    )

    # Prompt user to assign channels
    channel_assignments = assign_channels(channels)
    channels = list(channels.values())
    channels = {name: channels[idx] for name, idx in channel_assignments.items()}

    # Microns per pixel (X and Y directions)
    resolution = {name: (0.153, 0.153) for c, (name, _) in enumerate(channels.items())}

    ##################
    # APPLY CONTRAST #
    ##################

    contrast_channels = set_contrast(channels)

    ##################
    # ANALYZE IMAGES #
    ##################

    # Calculate areas
    calculate_areas = Area(resolution)
    areas = calculate_areas(contrast_channels)

    # Calculate cross-sectional area for whole bundle
    contrasted_layers = list(contrast_channels.values())
    all_layers = contrasted_layers[0]
    for layer in contrasted_layers[1:]:
        all_layers = cv2.add(all_layers, layer)
    contours, _ = cv2.findContours(
        all_layers.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    hull = cv2.convexHull(np.vstack([c for c in contours]))
    whole_bundle_mask = np.zeros_like(all_layers)
    cv2.drawContours(
        whole_bundle_mask,
        [hull],
        -1,
        (255,),
        -1,
    )
    whole_bundle_area = cv2.countNonZero(all_layers) * 0.153**2
    areas["whole"] = whole_bundle_area

    # Count particles in each layer
    watershed_count = Watershed()
    particle_counts = watershed_count(contrast_channels)

    # Calculate overlap
    overlap = [
        ("edu", "vimentin"),
        ("edu", "f-actin"),
        ("nuclei", "vimentin"),
        ("nuclei", "f-actin"),
    ]
    overlap_image = {}
    
    # Create f-actin segmentation mask
    factin_mask = np.zeros_like(contrast_channels["f-actin"])
    contours, _ = cv2.findContours(
        contrast_channels["f-actin"].copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    hull = cv2.convexHull(np.vstack([c for c in contours]))
    cv2.drawContours(
        factin_mask,
        [hull],
        -1,
        (255,),
        -1,
    )

    for one, two in overlap:
        mask = factin_mask
        if two != "f-actin":
            # Invert segmentation mask if not f-actin
            mask = cv2.bitwise_not(mask)
        overlapping = cv2.bitwise_and(
            contrast_channels[one],
            contrast_channels[one],
            mask=mask,
        )
        overlap_image[one, two] = overlapping

    # Count particles in each overlap layer
    overlap_counts = watershed_count(overlap_image)

    ############
    # COLORIZE #
    ############

    create_composite = messagebox.askyesno(
        "Create Composite",
        "Create composite image?",
    )

    composite = None
    if create_composite:
        channels = set_contrast(channels, composite=True)
        composite = make_false_color(channels, resolution)

    return contrast_channels, composite, areas, particle_counts, overlap_counts

    # # False color image
    # false_color_image = np.zeros_like(
    #     cv2.cvtColor(
    #         list(pretty_channels.values())[0],
    #         cv2.COLOR_GRAY2BGRA,
    #     )
    # )

    # for name, image in pretty_channels.items():
    #     # Get BGR color for this channel
    #     color = COLOR_MAPPING[name]

    #     # Create false color 16-bit image
    #     colored = np.full(image.shape + (3,), color)
    #     colored = colored.astype(np.float64)
    #     for i in range(3):
    #         colored[:, :, i] *= image.astype(np.float64)
    #     colored = colored.astype(np.uint16)

    #     # Add alpha channel and make black pixels transparent
    #     alpha = np.sum(colored, axis=-1) > 0
    #     colored = np.dstack((colored, np.uint16(alpha * 65535)))

    #     # Add false color channel to image
    #     false_color_image = np.where(
    #         np.dstack((colored[:, :, 3] != 0,)*4),
    #         colored,
    #         false_color_image,
    #     )

    # # Draw contour around false color
    # grayscale_false_color = np.uint8((np.sum(false_color_image, axis=-1) != 0) * 255)
    # contours, _ = cv2.findContours(
    #     grayscale_false_color.copy(),
    #     cv2.RETR_EXTERNAL,
    #     cv2.CHAIN_APPROX_NONE,
    # )
    # c = max(contours, key=cv2.contourArea)
    # areas["section"] = cv2.contourArea(c) * x_res * y_res

    # # Add scale bar
    # scale_bar_px = int(SCALE_BAR_LENGTH / x_res)
    # x_end = false_color_image.shape[0] - SCALE_BAR_X_OFFSET
    # x_start = x_end - scale_bar_px
    # y_bar = false_color_image.shape[1] - SCALE_BAR_Y_OFFSET
    # cv2.line(
    #     false_color_image,
    #     (x_start, y_bar),
    #     (x_end, y_bar),
    #     (65535,)*4,
    #     SCALE_BAR_HEIGHT_PX,
    # )
    # (text_width, text_height), _ = cv2.getTextSize(
    #     f"{SCALE_BAR_LENGTH} um",
    #     cv2.FONT_HERSHEY_SIMPLEX, 
    #     SCALE_BAR_FONT_SIZE,
    #     SCALE_BAR_HEIGHT_PX,
    # )
    # x_text, y_text = x_start + (x_end - x_start) // 2 - text_width // 2, y_bar + 3 * text_height // 2
    # cv2.putText(
    #     false_color_image,
    #     f"{SCALE_BAR_LENGTH} um",
    #     (x_text, y_text),
    #     cv2.FONT_HERSHEY_SIMPLEX, 
    #     SCALE_BAR_FONT_SIZE,
    #     (65535,)*4,
    #     SCALE_BAR_HEIGHT_PX,
    # )

    # # Convert false color to 8-bit unsigned
    # false_color_image = np.uint8(false_color_image / 257)[:, :, :3]

    # return false_color_image, areas, particle_counts, particle_fractions

if __name__ == "__main__":
    analyze_image(Path(argv[1]))
