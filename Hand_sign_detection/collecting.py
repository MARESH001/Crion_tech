import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

# Initialize Camera and Detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Parameters
offset = 20
imgSize = 300
folder = 'data\C'
counter = 0

# Create folder if it doesn't exist
if not os.path.exists(folder):
    os.makedirs(folder)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from camera")
        break

    hands, img = detector.findHands(img)  # Detect hands in the frame
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']  # Get bounding box of the hand

        # Create a white canvas and crop the image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]
        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:  # Height > Width
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal+wGap] = imgResize

        else:  # Width > Height
            j = imgSize / w
            hCal = math.ceil(j * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal+hGap, :] = imgResize

        # Show the processed images
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

        # Save image on pressing 's'
        key = cv2.waitKey(1)
        if key == ord('s'):
            counter += 1
            cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
            print(f"Image saved: {counter}")

    # Exit loop on pressing 'q'
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
