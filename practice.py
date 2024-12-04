import cv2
import mediapipe as mp

cap = cv2.VideoCapture(1)

mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
Hands = mpHands.Hands()

while True:
    success, image = cap.read()
    RBGimg = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    results = Hands.process(RBGimg)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Video", image)
    cv2.waitKey(1)