import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, CheckButtons, Button

# Load
img = cv2.imread("o.jpg")
detector = dlib.get_frontal_face_detector()

try:
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
except:
    predictor = None
    print("Predictor not found")

roi = {}

# pixelate
def pixelate(image, x1, y1, x2, y2, size=10):
    region = image[y1:y2, x1:x2]
    if region.size == 0:
        return image

    small = cv2.resize(region, (size, size))
    pixel = cv2.resize(small, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)

    image[y1:y2, x1:x2] = pixel
    return image


# landmark box
def get_box(shape, start, end):
    pts = [(shape.part(i).x, shape.part(i).y) for i in range(start, end)]
    pts = np.array(pts)

    x1, y1 = np.min(pts, axis=0)
    x2, y2 = np.max(pts, axis=0)

    return x1-10, y1-10, x2+10, y2+10


# mapping
LANDMARK_MAP = {
    "Eyes": [(36,42), (42,48)],
    "Nose": [(27,36)],
    "Mouth": [(48,68)]
}


# apply
def apply(event=None):
    result = img.copy()
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    labels = ["Face", "Eyes", "Nose", "Mouth"]
    selected = checks.get_status()

    for face in detector(gray):

        # FACE (no predictor needed)
        if selected[0]:
            result = pixelate(result,
                              face.left(), face.top(),
                              face.right(), face.bottom())

        #other features
        if predictor:
            shape = predictor(gray, face)

            for i in range(1, len(labels)):  # skip face
                if selected[i]:
                    part = labels[i]

                    for (start, end) in LANDMARK_MAP[part]:
                        x1, y1, x2, y2 = get_box(shape, start, end)
                        result = pixelate(result, x1, y1, x2, y2)

    # Manual ROI
    if "x1" in roi:
        result = pixelate(result,
                          roi["x1"], roi["y1"],
                          roi["x2"], roi["y2"])

    display.set_data(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    fig.canvas.draw_idle()

    cv2.imwrite("output.jpg", result)
    print("Saved output.jpg")


# reset
def reset(event=None):
    roi.clear()
    selector.extents = (0,0,0,0)

    for i, val in enumerate(checks.get_status()):
        if val:
            checks.set_active(i)

    display.set_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    fig.canvas.draw_idle()


# ui
fig, ax = plt.subplots(figsize=(10,6))
plt.subplots_adjust(left=0.3)

display = ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax.axis("off")

def onselect(e1, e2):
    roi["x1"], roi["y1"] = int(e1.xdata), int(e1.ydata)
    roi["x2"], roi["y2"] = int(e2.xdata), int(e2.ydata)

selector = RectangleSelector(ax, onselect, interactive=True)

checks = CheckButtons(
    plt.axes([0.05,0.5,0.2,0.25]),
    ["Face","Eyes","Nose","Mouth"],
    [False]*4
)

btn1 = Button(plt.axes([0.05,0.35,0.2,0.08]), "Pixelate")
btn2 = Button(plt.axes([0.05,0.25,0.2,0.08]), "Reset")

btn1.on_clicked(apply)
btn2.on_clicked(reset)

fig.canvas.mpl_connect("key_press_event",
    lambda e: apply() if e.key=="enter" else None)

plt.show()