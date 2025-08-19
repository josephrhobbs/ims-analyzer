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
        composite, areas, particle_counts, overlap_counts = analyze_image(dirpath / filename)

        # Data row
        row = {
            "F-actin Area (um2)":    areas["f-actin"],
            "Vimentin Area (um2)":   areas["vimentin"],
            "Nuclei":                particle_counts["nuclei"],
            "EdU-Positive Nuclei":   particle_counts["edu"],
            "EdU/Nuclei":            particle_counts["edu"] / particle_counts["nuclei"],
            "EdU-Positive F-actin":  overlap_counts["edu", "f-actin"] / overlap_counts["nuclei", "f-actin"],
            "EdU-Positive Vimentin": overlap_counts["edu", "vimentin"] / overlap_counts["nuclei", "vimentin"],
            "Nuclei in Vimentin":    1 - overlap_counts["nuclei", "f-actin"] / particle_counts["nuclei"],
            "Nuclei in F-actin":     overlap_counts["nuclei", "f-actin"] / particle_counts["nuclei"],
        }

        # Filename for false color image
        fc_filename = filename.removesuffix(IMAGE_EXT) + ".jpg"

        # Save false color image
        cv2.imwrite(dirpath / fc_filename, composite)

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
        row = [filename] + [None,] * (len(headers) - 1)
        for header, value in values.items():
            i = headers.index(header)
            row[i] = value
        output.append(row)
    output = [headers] + output
    output_str = "\n".join([",".join([str(v) for v in row]) for row in output])

    # Save CSV
    with open(dirpath / "output.csv", "w") as f:
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
