import cv2
import mediapipe as mp
import time


class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        """
        Parameters
        ----------
        mode:           是否输入静态图像
        maxHands:       检测到手的最大数量
        detectionCon:   检测手的置信度
        trackCon:       追踪手的置信度
        """
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(min_detection_confidence=0.7,
                                min_tracking_confidence=0.5,
                                max_num_hands=2)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        """
        Parameters
        ----------
        img:        要识别的一帧图像
        draw:       是否对手的标志点进行绘图

        Returns:    返回完成的一帧图像
        -------
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        with self.mpHands.Hands() as hands:
            self.result = self.hands.process(imgRGB)
        if self.result.multi_hand_landmarks:
            for handLms in self.result.multi_hand_landmarks:
                if draw:

                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        img = cv2.flip(img, 1)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 255, 255), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
