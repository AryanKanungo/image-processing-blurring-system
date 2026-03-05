# ==========================================
# INTERACTIVE PIXELIZATION TOOL (Python Script)
# ==========================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector


# ----------------------------
# Pixelization Function
# ----------------------------
def pixelate_region(image, roi, pixel_size=15):

    x1, y1, x2, y2 = roi

    region = image[y1:y2, x1:x2]
    h, w = region.shape[:2]

    if h == 0 or w == 0:
        return image

    # shrink image
    temp = cv2.resize(region, (max(1, w // pixel_size), max(1, h // pixel_size)),
                      interpolation=cv2.INTER_LINEAR)

    # enlarge again (blocky look)
    pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

    output = image.copy()
    output[y1:y2, x1:x2] = pixelated

    return output


# ----------------------------
# Main Program
# ----------------------------
def main():

    image_path = "image.jpg"   # CHANGE THIS

    image = cv2.imread(image_path)

    if image is None:
        print("Image not found!")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(8,6))
    ax.imshow(image_rgb)
    ax.set_title("Drag to select region, then press ENTER")
    ax.axis("off")

    roi = {}

    # ----------------------------
    # Mouse Selection
    # ----------------------------
    def onselect(eclick, erelease):

        roi["x1"], roi["y1"] = int(eclick.xdata), int(eclick.ydata)
        roi["x2"], roi["y2"] = int(erelease.xdata), int(erelease.ydata)

    rect_selector = RectangleSelector(
        ax,
        onselect,
        useblit=True,
        button=[1],
        interactive=True
    )

    # ----------------------------
    # Key Press Event
    # ----------------------------
    def on_key(event):

        if event.key == "enter" and roi:

            x1, y1 = roi["x1"], roi["y1"]
            x2, y2 = roi["x2"], roi["y2"]

            # Normalize coordinates
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])

            result = pixelate_region(image, (x1, y1, x2, y2), pixel_size=20)

            plt.figure(figsize=(8,6))
            plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            plt.title("Pixelized Image")
            plt.axis("off")
            plt.show()

            cv2.imwrite("pixelized_output.jpg", result)
            print("Saved as pixelized_output.jpg")

    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()


# ----------------------------
# Run Script
# ----------------------------
if __name__ == "__main__":
    main()