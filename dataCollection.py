import cv2
from cvzone2.HandTrackingModule import HandDetector
import numpy as np
import math


# variables
offset = 20  # To ensure Cropped image contains entire hand
imgSize = 400

folder = 'Data/Right'
counter = 0

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]                                 # Since only one hand
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y-offset: y+h+offset, x-offset: x+w+offset]

        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize/h
            wCalculated = math.ceil(w * k)
            imgResized = cv2.resize(imgCrop, (wCalculated, imgSize))
            wGap = math.ceil((imgSize-wCalculated)/2)
            imgWhite[:, wGap: wGap+wCalculated] = imgResized
        else:
            k = imgSize/w
            hCalculated = math.ceil(h * k)
            imgResized = cv2.resize(imgCrop, (imgSize, hCalculated))
            hGap = math.ceil((imgSize-hCalculated)/2)
            imgWhite[hGap: hGap+hCalculated, :] = imgResized

        cv2.imshow('Hand', imgCrop)
        cv2.moveWindow('Hand', 400, 0)
        cv2.imshow('imgWhite', imgWhite)

    cv2.imshow("Sign Language Detector", img)
    cv2.moveWindow('Sign Language Detector', 200, 200)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    if key == ord('s'):
        counter += 1
        print(counter)
        cv2.imwrite(f'{folder}/image_{counter}.jpg', imgWhite)
