import cv2
import numpy as np


def bitmap_to_contour_svg(input_bitmap_path: str, output_svg_path: str):
    """Convert a bitmap image to an SVG file containing its contour paths
    (including interior holes)."""
    image = cv2.imread(input_bitmap_path)
    if image is None:
        print(f"Error: image not found — {input_bitmap_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binarize the image so contours are crisp.
    # THRESH_BINARY_INV inverts dark-on-light images (e.g. black text on white).
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # RETR_LIST extracts all contours without building a hierarchy,
    # which preserves interior holes as separate paths.
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    h, w = gray.shape
    with open(output_svg_path, 'w', encoding='utf-8') as f:
        f.write(f'<?xml version="1.0" encoding="utf-8"?>\n')
        f.write(f'<svg version="1.1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">\n')

        for cnt in contours:
            # Skip noise contours that are too small (threshold adjustable).
            if len(cnt) < 10:
                continue

            # Build SVG path: M = move to start, L = line to, Z = close path.
            d_attr = "M " + " L ".join([f"{p[0][0]},{p[0][1]}" for p in cnt]) + " Z"
            f.write(f'  <path d="{d_attr}" fill="none" stroke="black" stroke-width="1" />\n')

        f.write('</svg>')
