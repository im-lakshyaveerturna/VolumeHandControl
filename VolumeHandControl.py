import cv2
import numpy as np
import time
import HandTrackingModule as htm
import math
import os

#####################################
wCam, hCam = 640, 480
#####################################

# Function to set volume using osascript
def set_volume(level):
    # Clamp the volume level between 0 and 100
    level = max(0, min(100, int(level)))
    os.system(f"osascript -e 'set volume output volume {level}'")


# Initialize hand detector
detector = htm.HandDetector()

# Setup webcam
cap = cv2.VideoCapture(1)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # Get positions of the thumb and index finger
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Draw visual indicators on the hand
        cv2.circle(img, (int(x1), int(y1)), 5, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (int(x2), int(y2)), 5, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.circle(img, (int(cx), int(cy)), 5, (255, 0, 0), cv2.FILLED)

        # Calculate the length between the thumb and index finger
        length = math.hypot(x2 - x1, y2 - y1)

        # Map the length to volume levels (0 to 100)
        vol = np.interp(length, [50, 300], [0, 100])
        set_volume(vol)

    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (30, 40), cv2.FONT_HERSHEY_PLAIN, 1, 255)

    # Display the webcam feed
    cv2.imshow('Video', img)
    cv2.waitKey(1)
