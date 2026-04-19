import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector


def pixelate(img, roi, size=15):
    x1, y1, x2, y2 = roi
    region = img[y1:y2, x1:x2]
    h, w = region.shape[:2]

    if h == 0 or w == 0:
        return img

    temp = cv2.resize(region, (max(1, w // size), max(1, h // size)))
    pixel = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

    out = img.copy()
    out[y1:y2, x1:x2] = pixel
    return out


def select_region(ax, roi):

    def onselect(e1, e2):
        roi["x1"], roi["y1"] = int(e1.xdata), int(e1.ydata)
        roi["x2"], roi["y2"] = int(e2.xdata), int(e2.ydata)

    selector = RectangleSelector(ax, onselect, useblit=True, button=[1], interactive=True)
    return selector


def key_handler(event, img, roi):

    if event.key != "enter" or not roi:
        return

    x1, y1 = roi["x1"], roi["y1"]
    x2, y2 = roi["x2"], roi["y2"]

    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])

    result = pixelate(img, (x1, y1, x2, y2), 20)

    plt.figure(figsize=(8,6))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

    cv2.imwrite("pixelized_output.jpg", result)
    print("Saved as pixelized_output.jpg")


def main():


    img = cv2.imread("image.jpg")

    if img is None:
        print("Image not found")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    fig, ax = plt.subplots(figsize=(8,6))
    ax.imshow(img_rgb)
    ax.axis("off")
    roi = {}

    selector = select_region(ax, roi)

    fig.canvas.mpl_connect(
        "key_press_event",
        lambda event: key_handler(event, img, roi)
    )

    plt.show()


if __name__ == "__main__":
    main()
