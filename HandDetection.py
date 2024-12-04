import cv2
import mediapipe as mp



cap = cv2.VideoCapture(1)

mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
hands = mpHands.Hands()


while True:
    success, image = cap.read()
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:

        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(image, handLms)

    cv2.imshow('Video',image)
    cv2.waitKey(1)
