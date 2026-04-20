import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, CheckButtons, Button
import os

# Define facial landmark indices for dlib
LANDMARKS = {
    "Left Eye": list(range(42, 48)),
    "Right Eye": list(range(36, 42)),
    "Nose": list(range(27, 36)),
    "Mouth": list(range(48, 68))
}

def pixelate(img, roi, size=15):
    x1, y1, x2, y2 = roi
    
    # Ensure coordinates are within image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    
    region = img[y1:y2, x1:x2]
    h, w = region.shape[:2]

    if h <= 0 or w <= 0:
        return img

    temp = cv2.resize(region, (max(1, w // size), max(1, h // size)))
    pixel = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

    out = img.copy()
    out[y1:y2, x1:x2] = pixel
    return out

def get_landmark_bbox(shape, indices, img_shape, padding=10):
    """Calculates a padded bounding box for specific facial landmarks."""
    pts = np.array([[shape.part(i).x, shape.part(i).y] for i in indices])
    x1, y1 = np.min(pts, axis=0)
    x2, y2 = np.max(pts, axis=0)
    
    return (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(img_shape[1], x2 + padding),
        min(img_shape[0], y2 + padding)
    )

def select_region(ax, state):
    def onselect(e1, e2):
        state["roi"]["x1"], state["roi"]["y1"] = int(e1.xdata), int(e1.ydata)
        state["roi"]["x2"], state["roi"]["y2"] = int(e2.xdata), int(e2.ydata)

    selector = RectangleSelector(ax, onselect, useblit=True, button=[1], interactive=True)
    return selector

def apply_pixelation(state):
    """Applies pixelation based on selected checkboxes AND manual ROI."""
    img_copy = state["orig_img"].copy()
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    
    # 1. Apply automated facial landmark pixelation
    selected_features = [label for label, state_val in zip(state["labels"], state["check"].get_status()) if state_val]
    
    if selected_features and state["detector"] and state["predictor"]:
        faces = state["detector"](gray)
        for face in faces:
            # Pixelate entire face bounding box
            if "Face" in selected_features:
                bbox = (face.left(), face.top(), face.right(), face.bottom())
                img_copy = pixelate(img_copy, bbox, size=20)
            
            # Pixelate specific features
            shape = state["predictor"](gray, face)
            for feature in selected_features:
                if feature in LANDMARKS:
                    bbox = get_landmark_bbox(shape, LANDMARKS[feature], img_copy.shape)
                    img_copy = pixelate(img_copy, bbox, size=15)

    # 2. Apply manual selection pixelation
    if "x1" in state["roi"]:
        x1, x2 = sorted([state["roi"]["x1"], state["roi"]["x2"]])
        y1, y2 = sorted([state["roi"]["y1"], state["roi"]["y2"]])
        img_copy = pixelate(img_copy, (x1, y1, x2, y2), size=20)

    # Update display
    state["curr_img"] = img_copy
    state["ax_img"].set_data(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
    state["fig"].canvas.draw_idle()

def btn_pixelate_clicked(event, state):
    apply_pixelation(state)
    cv2.imwrite("pixelized_output.jpg", state["curr_img"])
    print("Saved as pixelized_output.jpg")

def btn_reset_clicked(event, state):
    state["roi"] = {}  # Clear manual selection
    state["curr_img"] = state["orig_img"].copy()
    
    # Uncheck all boxes
    for i, is_checked in enumerate(state["check"].get_status()):
        if is_checked:
            state["check"].set_active(i)
            
    state["ax_img"].set_data(cv2.cvtColor(state["curr_img"], cv2.COLOR_BGR2RGB))
    state["fig"].canvas.draw_idle()
    print("Reset applied.")

def key_handler(event, state):
    if event.key == "enter":
        btn_pixelate_clicked(event, state)

def main():
    image_path = "image.jpg"
    model_path = "shape_predictor_68_face_landmarks.dat"

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: '{image_path}' not found.")
        return

    # Initialize dlib
    detector = dlib.get_frontal_face_detector()
    predictor = None
    if os.path.exists(model_path):
        predictor = dlib.shape_predictor(model_path)
    else:
        print(f"Warning: '{model_path}' not found. Facial mapping disabled.")

    # Application state to share data across callbacks easily
    state = {
        "orig_img": img,
        "curr_img": img.copy(),
        "roi": {},
        "detector": detector,
        "predictor": predictor,
        "labels": ["Face", "Left Eye", "Right Eye", "Nose", "Mouth"]
    }

    # Setup Main Figure and Subplots
    state["fig"] = plt.figure(figsize=(10, 7))
    
    # Image axes (leaving room on the left for UI)
    ax_main = state["fig"].add_axes([0.3, 0.05, 0.65, 0.9])
    state["ax_img"] = ax_main.imshow(cv2.cvtColor(state["curr_img"], cv2.COLOR_BGR2RGB))
    ax_main.axis("off")

    # UI Controls Axes
    ax_chk = state["fig"].add_axes([0.05, 0.5, 0.2, 0.35])
    ax_btn_pix = state["fig"].add_axes([0.05, 0.35, 0.2, 0.08])
    ax_btn_res = state["fig"].add_axes([0.05, 0.25, 0.2, 0.08])

    # Add Checkboxes
    state["check"] = CheckButtons(ax_chk, state["labels"], [False] * len(state["labels"]))
    
    # Add Buttons
    btn_pix = Button(ax_btn_pix, 'Pixelate / Save')
    btn_res = Button(ax_btn_res, 'Reset')

    # Connect Callbacks
    selector = select_region(ax_main, state)
    btn_pix.on_clicked(lambda event: btn_pixelate_clicked(event, state))
    btn_res.on_clicked(lambda event: btn_reset_clicked(event, state))
    state["fig"].canvas.mpl_connect("key_press_event", lambda event: key_handler(event, state))

    plt.show()

if __name__ == "__main__":
    main()