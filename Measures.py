
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2


# Function to calculate midpoint
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


# Initialize camera for real-time video display
cap = cv2.VideoCapture(0)


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Resize frame
    frame = cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)

    # Extract ROI (Region of Interest)
    orig = frame[:1080, 0:1920]

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (15, 15), 0)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Morphological closing
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Copy the closing image for further processing
    result_img = closing.copy()

    # Find contours in the image
    contours, hierarchy = cv2.findContours(result_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize object count
    object_count = 0

    # Loop over the contours
    for cnt in contours:
        # Calculate contour area
        area = cv2.contourArea(cnt)

        # Filter contours based on area
        if area < 1000 or area > 120000:
            continue

        # Compute the bounding box for the contour
        orig = frame.copy()
        box = cv2.minAreaRect(cnt)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 64), 2)

        # Calculate midpoints and draw them
        for (x, y) in box:
            cv2.circle(orig, (int(x), int(y)), 5, (0, 255, 64), -1)

        # Calculate distances between midpoints
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        # Draw midpoints
        cv2.circle(orig, (int(tltrX), int(tltrY)), 0, (0, 255, 64), 5)
        cv2.circle(orig, (int(blbrX), int(blbrY)), 0, (0, 255, 64), 5)
        cv2.circle(orig, (int(tlblX), int(tlblY)), 0, (0, 255, 64), 5)
        cv2.circle(orig, (int(trbrX), int(trbrY)), 0, (0, 255, 64), 5)

        # Draw lines between midpoints
        cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
        cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

        # Calculate Euclidean distances between midpoints
        width_pixel = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        length_pixel = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        # Draw object dimensions on the frame
        cv2.putText(orig, "W: {:.1f}CM".format(width_pixel / 25.5), (int(trbrX + 10), int(trbrY)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(orig, "L: {:.1f}CM".format(length_pixel / 25.5), (int(tltrX - 15), int(tltrY - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        object_count += 1

    # Display the number of detected objects
    cv2.putText(orig, "Count: {}".format(object_count), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                cv2.LINE_AA)
    cv2.imshow('Camera', orig)

    # Break the loop when 'Esc' is pressed
    key = cv2.waitKey(1)
    if key == 27:
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
