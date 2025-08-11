import cv2
from cvzone2.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math


# variables
offset = 20  # To ensure Cropped image contains entire hand
imgSize = 400

folder = 'Data/F'
counter = 0
labels = ['A', 'B', 'C', 'D', 'E', 'F']

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier('Model/keras_model.h5', 'Model/labels.txt')
while True:
    success, img = cap.read()
    imgOutput = img.copy()

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
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
        else:
            k = imgSize/w
            hCalculated = math.ceil(h * k)
            imgResized = cv2.resize(imgCrop, (imgSize, hCalculated))
            hGap = math.ceil((imgSize-hCalculated)/2)
            imgWhite[hGap: hGap+hCalculated, :] = imgResized
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        cv2.putText(imgOutput, labels[index], (x, y-20), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 69, 100), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset), (x+w+offset, y+h+offset), (0, 69, 100), 4)

    cv2.imshow("Sign Language Detector", imgOutput)
    cv2.moveWindow('Sign Language Detector', 200, 200)
    if cv2.waitKey(1) == ord('q'):
        break
