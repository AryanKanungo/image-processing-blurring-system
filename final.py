import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, CheckButtons, Button


def pixelate(img, roi, size=15):
    x1, y1, x2, y2 = [int(v) for v in roi]
    
    # Keep coordinates inside the image
    x1, x2 = max(0, min(x1, x2)), min(img.shape[1], max(x1, x2))
    y1, y2 = max(0, min(y1, y2)), min(img.shape[0], max(y1, y2))
    
    region = img[y1:y2, x1:x2]
    h, w = region.shape[:2]

    if h == 0 or w == 0:
        return img

    temp = cv2.resize(region, (max(1, w // size), max(1, h // size)))
    pixel = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

    out = img.copy()
    out[y1:y2, x1:x2] = pixel
    return out


def get_landmark_roi(shape, start, end):
    """Gets padded bounding box for specific facial landmarks."""
    pts = np.array([[shape.part(i).x, shape.part(i).y] for i in range(start, end)])
    x1, y1 = np.min(pts, axis=0) - 10
    x2, y2 = np.max(pts, axis=0) + 10
    return (x1, y1, x2, y2)


def select_region(ax, roi):
    def onselect(e1, e2):
        roi["x1"], roi["y1"] = int(e1.xdata), int(e1.ydata)
        roi["x2"], roi["y2"] = int(e2.xdata), int(e2.ydata)

    selector = RectangleSelector(ax, onselect, useblit=True, button=[1], interactive=True)
    return selector


def apply_pixelation(event, img, im_display, roi, checks, detector, predictor, fig):
    """Triggered by the Pixelate button or Enter key."""
    result = img.copy()
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    
    # 1. Apply AI Pixelation based on checkboxes
    selected = [label for label, state in zip(["Face", "Eyes", "Nose", "Mouth"], checks.get_status()) if state]
    
    if selected and predictor:
        for face in detector(gray):
            if "Face" in selected:
                result = pixelate(result, (face.left(), face.top(), face.right(), face.bottom()), 20)
            
            shape = predictor(gray, face)
            if "Eyes" in selected:
                result = pixelate(result, get_landmark_roi(shape, 36, 42), 15) # Left
                result = pixelate(result, get_landmark_roi(shape, 42, 48), 15) # Right
            if "Nose" in selected:
                result = pixelate(result, get_landmark_roi(shape, 27, 36), 15)
            if "Mouth" in selected:
                result = pixelate(result, get_landmark_roi(shape, 48, 68), 15)

    # 2. Apply Manual Pixelation
    if "x1" in roi:
        x1, x2 = sorted([roi["x1"], roi["x2"]])
        y1, y2 = sorted([roi["y1"], roi["y2"]])
        result = pixelate(result, (x1, y1, x2, y2), 20)

    # Update image on screen
    im_display.set_data(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    fig.canvas.draw_idle()

    cv2.imwrite("pixelized_output.jpg", result)
    print("Saved as pixelized_output.jpg")


def reset_image(event, img, im_display, roi, checks, selector, fig):
    """Triggered by the Reset button."""
    roi.clear()
    selector.extents = (0, 0, 0, 0) # Clear manual box
    
    # Uncheck all boxes
    for i, state in enumerate(checks.get_status()):
        if state:
            checks.set_active(i)

    # Restore original image
    im_display.set_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    fig.canvas.draw_idle()
    print("Image Reset")


def main():
    img = cv2.imread("image.jpg")
    if img is None:
        print("Image not found")
        return

    # Load AI
    detector = dlib.get_frontal_face_detector()
    try:
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    except:
        print("Warning: shape_predictor_68_face_landmarks.dat not found. AI features disabled.")
        predictor = None

    # Setup UI Elements
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(left=0.3) # Make room on the left for buttons
    
    im_display = ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.axis("off")
    roi = {}
    selector = select_region(ax, roi)

    # Checkboxes
    ax_chk = plt.axes([0.05, 0.5, 0.2, 0.25])
    checks = CheckButtons(ax_chk, ["Face", "Eyes", "Nose", "Mouth"], [False] * 4)

    # Pixelate Button
    ax_pix = plt.axes([0.05, 0.35, 0.2, 0.08])
    btn_pix = Button(ax_pix, 'Pixelate / Save')
    
    # Reset Button
    ax_res = plt.axes([0.05, 0.25, 0.2, 0.08])
    btn_res = Button(ax_res, 'Reset')

    # Bind Events using Lambda (to avoid global variables)
    btn_pix.on_clicked(lambda e: apply_pixelation(e, img, im_display, roi, checks, detector, predictor, fig))
    btn_res.on_clicked(lambda e: reset_image(e, img, im_display, roi, checks, selector, fig))
    
    # Keep the Enter key functionality too
    fig.canvas.mpl_connect(
        "key_press_event",
        lambda e: apply_pixelation(e, img, im_display, roi, checks, detector, predictor, fig) if e.key == "enter" else None
    )

    plt.show()


if __name__ == "__main__":
    main()