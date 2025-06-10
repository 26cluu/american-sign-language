import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands = 1)
folder = "myData/D"
counter = 0
imgSize = 300

offset = 20


while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255

        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

        imgCropShape = imgCrop.shape


        aspectRatio = h/w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(w * k)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((300-wCal)/2)
            imgWhite[:, wGap:wCal+wGap] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(h * k)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((300-hCal)/2)
            imgWhite[hGap:hCal+hGap, :] = imgResize


        cv2.imshow("imageCrop", imgCrop)
        cv2.imshow("imgwhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)

