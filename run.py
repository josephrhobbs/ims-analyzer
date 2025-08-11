#!/usr/bin/env python3
# Analyze Directory

from pathlib import Path

import cv2
from os import walk
import tkinter as tk
from tkinter import filedialog, messagebox

from analyze import analyze_image

IMAGE_EXT = ".ims"

def analyze_directory(dirname):
    """
    Analyze a directory of images recursively
    """
    # Walk directory
    dirname = Path(dirname)
    dirpath, subdirs, files = next(walk(dirname))
    dirpath = Path(dirpath)

    # Sort subdirectories
    subdirs.sort()

    # Analyze all subdirectories
    _ = [
        analyze_directory(dirname / s)
        for s in subdirs
    ]

    # If no IMS files in this directory, stop
    if not any([f.endswith(IMAGE_EXT) for f in files]):
        return

    # Sort filenames alphabetically
    files.sort()

    # Initialize data
    data = {}

    for filename in files:
        # Skip non-image files
        if not filename.endswith(IMAGE_EXT):
            continue

        # Analyze this image
        false_color, areas, counts, fractions = analyze_image(dirpath / filename)

        # Data row
        row = {}

        # Store data
        row |= {f"area_{name}": area for name, area in areas.items()}
        row |= {f"count_{name}" if other is None else f"count_{name}_in_{other}": count for (name, other), count in counts.items()}
        row |= {f"fraction_{name}" if other is None else f"fraction_{name}_in_{other}": fraction for (name, other), fraction in fractions.items()}

        # Filename for false color image
        fc_filename = filename.removesuffix(IMAGE_EXT) + ".jpg"

        # Save false color image
        cv2.imwrite(dirpath / fc_filename, false_color)

        # Add row to data
        data[filename] = row

    # Construct CSV
    headers = set()
    for values in data.values():
        # Create headers list
        headers |= set(values.keys())
    headers = ["Filename"] + sorted(list(headers))
    output = []
    for filename, values in data.items():
        row = [filename] + [None,] * len(headers)
        for header, value in values.items():
            i = headers.index(header)
            row[i] = value
        output.append(row)
    output = [headers] + output
    print(headers)
    print(output)
    output_str = "\n".join([",".join([str(v) for v in row]) for row in output])

    # Save CSV
    with open("output.csv", "w") as f:
        f.write(output_str)

if __name__ == "__main__":
    # Start Tkinter
    root = tk.Tk()
    root.withdraw()

    # Get directory
    dirname = filedialog.askdirectory(title="Select Input Directory")

    # Analyze given directory
    analyze_directory(dirname)

    # Inform user
    messagebox.showinfo(
        "IMS Analyzer Complete",
        "Analysis is complete! :)",
    )

    # End Tkinter
    root.destroy()
