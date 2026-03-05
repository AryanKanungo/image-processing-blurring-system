# ==========================================
# REGION BLUR USING MANUAL PIXEL AVERAGING
# ==========================================

import cv2
import numpy as np

# ----------------------------
# Manual Blur Function
# ----------------------------
def blur_region_manual(image, roi):

    x1, y1, x2, y2 = roi

    output = image.copy()

    height, width = image.shape

    for y in range(y1, y2):
        for x in range(x1, x2):

            pixel_sum = 0
            count = 0

            # iterate through 3x3 neighbourhood
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:

                    nx = x + dx
                    ny = y + dy

                    if 0 <= nx < width and 0 <= ny < height:

                        pixel_sum += image[ny][nx]
                        count += 1

            avg_pixel = int(pixel_sum / count)

            output[ny][nx] = avg_pixel

    return output


# ----------------------------
# Load Image
# ----------------------------
image = cv2.imread("image.jpg")

if image is None:
    print("Image not found")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Example ROI
x1, y1 = 100, 100
x2, y2 = 300, 300

# Apply blur
result = blur_region_manual(gray, (x1, y1, x2, y2))

# ----------------------------
# Display Result
# ----------------------------
cv2.imshow("Original", gray)
cv2.imshow("Blurred Region", result)

cv2.imwrite("blurred_output.jpg", result)

cv2.waitKey(0)
cv2.destroyAllWindows()