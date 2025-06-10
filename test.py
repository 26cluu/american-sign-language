import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands = 1)
classifier = Classifier('oldModel/keras_model.h5', 'oldModel/labels.txt')
folder = "Data/C"
counter = 0
imgSize = 300

offset = 20

# labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
labels = ['A', 'B', 'C', 'D']
current = ''

time_dif = 5
current_time = time.time()

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        print(x, y, w, h)

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
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            if time.time() - current_time >= 5:
                current += labels[index]
                current_time = time.time()
            print(prediction, index)

        else:
            k = imgSize / w
            hCal = math.ceil(h * k)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((300-hCal)/2)
            imgWhite[hGap:hCal+hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            if time.time() - current_time >= 5:
                current += labels[index]
                current_time = time.time()
            print(prediction, index)
        cv2.rectangle(imgOutput, (x - offset, y-offset-50), (x - offset+90, y-offset-50 + 50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y-26), cv2.FONT_HERSHEY_DUPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x-offset,y-offset), (x+w+offset, y+h+offset), (255, 0, 255), 4)
        cv2.putText(imgOutput, current, (100, 100), cv2.FONT_HERSHEY_DUPLEX, 1.7, (255, 255, 255), 2)
        if time.time() - current_time < 1:
            cv2.putText(imgOutput, '5', (30, 250), cv2.FONT_HERSHEY_DUPLEX, 1.7, (255, 255, 255), 2)
        elif time.time() - current_time <= 2:
            cv2.putText(imgOutput, '4', (30, 250), cv2.FONT_HERSHEY_DUPLEX, 1.7, (255, 255, 255), 2)
        elif time.time() - current_time <= 3:
            cv2.putText(imgOutput, '3', (30, 250), cv2.FONT_HERSHEY_DUPLEX, 1.7, (255, 255, 255), 2)
        elif time.time() - current_time <= 4:
            cv2.putText(imgOutput, '2', (30, 250), cv2.FONT_HERSHEY_DUPLEX, 1.7, (255, 255, 255), 2)
        else:
            cv2.putText(imgOutput, '1', (30, 250), cv2.FONT_HERSHEY_DUPLEX, 1.7, (255, 255, 255), 2)
        cv2.imshow("imageCrop", imgOutput)
        cv2.imshow("imgwhite", imgWhite)

    cv2.imshow("Image", img)
    cv2.waitKey(1)


