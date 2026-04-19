import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, CheckButtons, Button
import os

# Configuration
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
REGIONS = {
    "Eyes":      list(range(36, 48)),
    "Nose":      list(range(27, 36)),
    "Mouth":     list(range(48, 68)),
    "Full Face": list(range(0,  68)),
}

# Initialize dlib
if not os.path.exists(PREDICTOR_PATH):
    raise FileNotFoundError(f"Please download '{PREDICTOR_PATH}' and place it in the script folder.")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def pixelate(img, roi, blocks=15):
    """Applies a pixelation effect to a specific Region of Interest."""
    x1, y1, x2, y2 = roi
    
    # Ensure coordinates are within image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    
    region = img[y1:y2, x1:x2]
    h, w = region.shape[:2]
    if h <= 0 or w <= 0: return img

    # Resize down and back up using NEAREST interpolation for the pixel look
    temp = cv2.resize(region, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
    pixelated_roi = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    
    out = img.copy()
    out[y1:y2, x1:x2] = pixelated_roi
    return out

def get_face_roi(img, region_name):
    """Detects landmarks and returns a bounding box for the specific region."""
    
    # Standard OpenCV to dlib conversion
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Force memory alignment just to be safe
    dlib_img = np.ascontiguousarray(rgb, dtype=np.uint8)
    
    faces = detector(dlib_img, 1)
    if not faces:
        print(f"No face detected when searching for {region_name}.")
        return None
    
    shape = predictor(dlib_img, faces[0])
    idxs = REGIONS[region_name]
    
    xs = [shape.part(i).x for i in idxs]
    ys = [shape.part(i).y for i in idxs]
    
    pad = 10
    return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)

def main():
    original_img = cv2.imread("image.jpg")
    if original_img is None:
        print("Error: Could not find 'image.jpg' in the current directory.")
        return

    state = {"img": original_img.copy(), "roi": None}
    checked_status = {label: False for label in REGIONS}

    # Setup UI Layout
    fig, ax = plt.subplots(figsize=(12, 7))
    plt.subplots_adjust(right=0.75)
    ax.axis("off")

    def refresh():
        ax.imshow(cv2.cvtColor(state["img"], cv2.COLOR_BGR2RGB))
        fig.canvas.draw_idle()

    # --- UI Elements ---
    ax_check = plt.axes([0.78, 0.6, 0.15, 0.25])
    check = CheckButtons(ax_check, list(REGIONS.keys()), [False]*4)
    
    def toggle_label(label):
        checked_status[label] = not checked_status[label]
    check.on_clicked(toggle_label)

    # Button: Apply Auto-Pixelate
    ax_btn_auto = plt.axes([0.78, 0.5, 0.15, 0.06])
    btn_auto = Button(ax_btn_auto, 'Apply Selected', color='lightblue')

    def apply_auto(event):
        for region, is_checked in checked_status.items():
            if is_checked:
                box = get_face_roi(state["img"], region)
                if box:
                    state["img"] = pixelate(state["img"], box)
                    print(f"Applied pixelation to: {region}")
        refresh()
    btn_auto.on_clicked(apply_auto)

    # Button: Apply Manual Selection
    ax_btn_manual = plt.axes([0.78, 0.4, 0.15, 0.06])
    btn_manual = Button(ax_btn_manual, 'Pixelate Manual', color='lightgreen')

    def apply_manual(event):
        if state["roi"]:
            state["img"] = pixelate(state["img"], state["roi"])
            print("Applied manual pixelation.")
            refresh()
        else:
            print("Please draw a rectangle on the image first.")
    btn_manual.on_clicked(apply_manual)

    # Button: Reset
    ax_btn_reset = plt.axes([0.78, 0.3, 0.15, 0.06])
    btn_reset = Button(ax_btn_reset, 'Reset Image', color='tomato')

    def reset(event):
        state["img"] = original_img.copy()
        print("Image reset.")
        refresh()
    btn_reset.on_clicked(reset)

    # Button: Save
    ax_btn_save = plt.axes([0.78, 0.2, 0.15, 0.06])
    btn_save = Button(ax_btn_save, 'Save Image', color='gold')

    def save_img(event):
        cv2.imwrite("pixelized_output.jpg", state["img"])
        print("Saved as 'pixelized_output.jpg'")
    btn_save.on_clicked(save_img)

    # --- Interaction ---
    def on_select(eclick, erelease):
        x1, x2 = sorted([int(eclick.xdata), int(erelease.xdata)])
        y1, y2 = sorted([int(eclick.ydata), int(erelease.ydata)])
        state["roi"] = (x1, y1, x2, y2)

    selector = RectangleSelector(
        ax, 
        on_select, 
        useblit=True, 
        button=[1], 
        minspanx=5, 
        minspany=5, 
        interactive=True
    )

    refresh()
    plt.show()

if __name__ == "__main__":
    main()