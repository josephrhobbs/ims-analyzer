# IMS Analyzer
# User Interaction

import cv2
import tkinter as tk
import numpy as np

PREVIEW_SIZE = 512, 512
CONCAT_PREVIEW_SIZE = 256, 256

CHANNELS = [
    "nuclei",
    "f-actin",
    "vimentin",
    "edu",
]

# Color mapping (BGRA)
COLOR_MAPPING = {
    "nuclei":   (1, 0, 0, 0),
    "vimentin": (0, 0, 1, 0),
    "f-actin":  (0, 1, 0, 0),
    "edu":      (1, 1, 1, 0),
}

# Scale bar dimensions
SCALE_BAR_LENGTH = 100 # um
SCALE_BAR_HEIGHT_PX = 10 # px
SCALE_BAR_FONT_SIZE = 2

# Scale bar offsets
SCALE_BAR_X_OFFSET = 50
SCALE_BAR_Y_OFFSET = 100

def show_image(channels, z, brightness=0.5, contrast=1.0):
    """
    Preview the image selected by the user.
    """
    resized = []
    for _, channel in channels.items():
        im = cv2.resize(channel, CONCAT_PREVIEW_SIZE)
        im = np.clip(
            0.5 + contrast*((im/65535) - (1 - brightness)),
            0.0,
            1.0,
        )
        resized.append((65535*im).astype(np.uint16))
    cv2.imshow(f"Z = {z}", cv2.hconcat(resized))

def show_channel(channel, name):
    """
    Preview a channel.
    """
    # Get BGR color for this channel
    color = COLOR_MAPPING[name]

    # Create false color 16-bit image
    colored = np.full(channel.shape + (4,), color)
    colored = colored.astype(np.float64)
    for i in range(3):
        colored[:, :, i] *= channel.astype(np.float64)
    colored = colored.astype(np.uint16)
    im = cv2.resize(colored, PREVIEW_SIZE)

    cv2.imshow(f"{name}", im)

def show_channel_contours(channel, name):
    """
    Preview a channel with contours.
    """
    # Resize channel
    channel = cv2.resize(channel, PREVIEW_SIZE)

    # Get BGR color for this channel
    color = COLOR_MAPPING[name]

    # Create false color 16-bit image
    colored = np.full(channel.shape + (4,), color)
    colored = colored.astype(np.float64)
    for i in range(3):
        colored[:, :, i] *= channel.astype(np.float64)
    colored = colored.astype(np.uint16)

    # Find contours
    contours, _ = cv2.findContours(
        (channel / 257).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    # Draw contours
    colored = (colored / 257).astype(np.uint8)
    cv2.drawContours(
        colored,
        contours,
        -1,
        (65535,)*3,
        2,
    )

    cv2.imshow(f"{name}", colored)

def show_composite(channels, title, resolution):
    """
    Given a collection of channels, show a false color composite.
    """
    # Resolution
    x_res, y_res = resolution

    # False color image
    false_color_image = np.zeros_like(
        cv2.cvtColor(
            list(channels.values())[0],
            cv2.COLOR_GRAY2BGRA,
        )
    )

    for name, image in channels.items():
        # Get BGR color for this channel
        color = COLOR_MAPPING[name]

        # Create false color 8-bit image
        colored = np.full(image.shape + (4,), color)
        colored = colored.astype(np.float64)
        for i in range(3):
            colored[:, :, i] *= image.astype(np.float64)
        colored = colored.astype(np.uint8)

        # Create mask
        _, mask = cv2.threshold(image, 255//20, 255, cv2.THRESH_BINARY)
        colored = cv2.bitwise_and(colored, colored, mask=mask)

        # Add false color channel to image
        false_color_image = cv2.add(false_color_image, colored)

    # # Draw contour around false color
    # grayscale_false_color = np.uint8((np.sum(false_color_image, axis=-1) != 0) * 255)
    # contours, _ = cv2.findContours(
    #     grayscale_false_color.copy(),
    #     cv2.RETR_EXTERNAL,
    #     cv2.CHAIN_APPROX_NONE,
    # )
    # c = max(contours, key=cv2.contourArea)
    # areas["section"] = cv2.contourArea(c) * x_res * y_res

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

    cv2.imshow(title, cv2.resize(false_color_image, PREVIEW_SIZE))

    return false_color_image

def select_z(layers):
    """
    Given a Z-stack of images, ask the user to select one slice.
    """
    idx = 0
    count = len(layers)

    print("SELECT Z SLICE")
    print("\tPress [space] to view next layer")
    print("\tPress [x] to increase brightness")
    print("\tPress [z] to decrease brightness")
    print("\tPress [m] to increase contrast")
    print("\tPress [n] to decrease contrast")
    print("\tPress [r] to [r]eset brightness and contrast")
    print("\tPress [s] to [s]elect current layer")
    print()

    brightness = 0.5
    contrast = 1
  
    while True:
        # Preview image
        show_image(layers[idx], idx, brightness, contrast)

        # Await key
        key = cv2.waitKey(0)

        if key == ord(" "):
            cv2.destroyAllWindows()
            idx = (idx + 1) % count
            brightness = 0.5
            contrast = 1
            continue
        elif key == ord("m"):
            contrast *= 1.05
            continue
        elif key == ord("n"):
            contrast /= 1.05
            continue
        elif key == ord("x"):
            brightness += 0.02
            continue
        elif key == ord("z"):
            brightness -= 0.02
            continue
        elif key == ord("r"):
            brightness = 0.5
            contrast = 1
            continue
        elif key == ord("s"):
            cv2.destroyAllWindows()
            return layers[idx], idx

def assign_channels(image):
    """
    Given a collection of channels, ask the user to assign names to each channel.
    """
    assignments = {channel: None for channel in CHANNELS}

    # Discard keys
    image = list(image.values())
    count = len(image)

    print("ASSIGN CHANNELS")
    print("\tPress [m] to view next channel")
    print("\tPress [n] to view previous channel")
    print("\tPress [s] to [s]elect current channel")
    print()

    for channel in CHANNELS:
        idx = 0
        while True:
            # Preview channel
            show_channel(image[idx], channel)

            # Await key
            key = cv2.waitKey(0)

            if key == ord("m"):
                idx = (idx + 1) % count
                continue
            elif key == ord("n"):
                idx = (idx - 1) % count
                continue
            elif key == ord("s"):
                break

        # Save channel assignment
        assignments[channel] = idx

    cv2.destroyAllWindows()
    return assignments

def set_contrast(image, composite=False):
    """
    Given a collection of channels, ask the user to set the contrast in each.
    """
    if composite:
        print("SET BRIGHTNESS AND CONTRAST (FOR COMPOSITE)")
    else:
        print("SET BRIGHTNESS AND CONTRAST (FOR QUANTIFICATION)")
    print("\tPress [x] to increase brightness")
    print("\tPress [z] to decrease brightness")
    print("\tPress [m] to increase contrast")
    print("\tPress [n] to decrease contrast")
    print("\tPress [r] to [r]eset brightness and contrast")
    print("\tPress [s] to [s]ave current contrast setting")
    print()

    result = {}

    for name, channel in image.items():
        # Initial brightness and contrast settings
        brightness = 0.5
        contrast = 1

        # Convert channel to floating-point
        original_channel = channel / 65535

        while True:
            current_channel = np.clip(
                0.5 + contrast*(original_channel - (1 - brightness)),
                0.0,
                1.0,
            )

            # Preview channel
            show_channel_contours((65535 * current_channel).astype(np.uint16), name)

            # Await key
            key = cv2.waitKey(0)

            if key == ord("m"):
                contrast *= 1.05
                continue
            elif key == ord("n"):
                contrast /= 1.05
                continue
            elif key == ord("x"):
                brightness += 0.02
                continue
            elif key == ord("z"):
                brightness -= 0.02
                continue
            elif key == ord("r"):
                brightness = 0.5
                contrast = 1
                continue
            elif key == ord("s"):
                break

        current_channel = np.clip(
            0.5 + contrast*(original_channel - (1 - brightness)),
            0.0,
            1.0,
        )

        # Save new channel
        result[name] = (current_channel * 255).astype(np.uint8)

    cv2.destroyAllWindows()
    return result

def make_false_color(image, resolution):
    """
    Given a collection of channels, ask the user to adjust a false color image.
    """
    print("ADJUST COMPOSITE IMAGE")
    print("\tPress [x] to increase brightness")
    print("\tPress [z] to decrease brightness")
    print("\tPress [m] to increase contrast")
    print("\tPress [n] to decrease contrast")
    print("\tPress [space] to change layer")
    print("\tPress [r] to [r]eset brightness and contrast")
    print("\tPress [s] to [s]ave current contrast setting")
    print()

    idx = 0
    names = list(image.keys())
    count = len(names)

    # Initial brightness and contrast settings
    brightness = {name: 0.5 for name in image.keys()}
    contrast = {name: 1 for name in image.keys()}

    # Convert channel to floating-point
    original = {name: channel / 255 for name, channel in image.items()}

    while True:
        this_name = names[idx]

        current = {name: np.clip(
            0.5 + contrast[name]*(channel - (1 - brightness[name])),
            0.0,
            1.0,
        ) for name, channel in original.items()}

        # Preview composite
        show_composite(
            {name: (255*channel).astype(np.uint8) for name, channel in current.items()},
            f"Active: {this_name}",
            resolution[this_name],
        )

        # Await key
        key = cv2.waitKey(0)

        if key == ord("m"):
            contrast[this_name] *= 1.05
            continue
        elif key == ord("n"):
            contrast[this_name] /= 1.05
            continue
        elif key == ord("x"):
            brightness[this_name] += 0.02
            continue
        elif key == ord("z"):
            brightness[this_name] -= 0.02
            continue
        elif key == ord("r"):
            brightness[this_name] = 0.5
            contrast[this_name] = 1
            continue
        elif key == ord(" "):
            idx = (idx + 1) % count
            cv2.destroyAllWindows()
            continue
        elif key == ord("s"):
            break

    result = show_composite(
        {name: (255*channel).astype(np.uint8) for name, channel in current.items()},
        f"Final Composite",
        resolution[this_name],
    )
    cv2.destroyAllWindows()

    return result
