import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands = 1)


alphabet = ['y']

def img_convert(folder):
    print(folder)
    counter = 0
    imgSize = 300
    offset = 20
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)


    for img in images:
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            print(x, y, w, h)

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255

            imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

            imgCropShape = imgCrop.shape


            aspectRatio = h/w

            try:
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(w * k)
                    print(imgCrop)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((300-wCal)/2)
                    try:
                        imgWhite[:, wGap:wCal+wGap] = imgResize
                    except ValueError:
                        continue

                else:
                    k = imgSize / w
                    hCal = math.ceil(h * k)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((300-hCal)/2)
                    try:
                        imgWhite[hGap:hCal+hGap, :] = imgResize
                    except ValueError:
                        continue
            except cv2.error:
                continue
            cv2.imwrite(f'new_data/{folder[-1]}/Image_{time.time()}.jpg', imgWhite)
        counter += 1
        print(counter)

for i in range(1):
    img_convert("old_data/" + alphabet[i])