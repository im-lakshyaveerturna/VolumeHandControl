import cv2
import mediapipe as mp
import time

class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )


    def findHands(self, image, draw=True):
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)

        return image

    def findPosition(self, image, handNo=0, draw=True):
        """Finds hand landmarks and returns their pixel positions."""
        lmList = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[handNo]
            h, w, c = image.shape
            for id, lm in enumerate(hand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))
                if draw:
                    cv2.circle(image, (int(cx), int(cy)), 10, (255, 0, 0), cv2.FILLED)
        return lmList


def main():
    cap = cv2.VideoCapture(0)  # Use camera index 0 for the default camera
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    detector = HandDetector()
    pTime = 0  # Initialize pTime before the loop

    while True:
        success, image = cap.read()
        if not success:
            print("Failed to grab frame.")
            break

        image = detector.findHands(image)
        lmList = detector.findPosition(image)
        if lmList:
            print(lmList[0])  # Print the first landmark (ID 0)

        # Calculate and display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime) if pTime != 0 else 0
        pTime = cTime

        cv2.putText(image, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Video', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
