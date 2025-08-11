from tkinter import *
import tkinter as tk, threading
import cv2
from cvzone2.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math


# variables
offset = 20  # To ensure Cropped image contains entire hand
imgSize = 400

labels = ['A', 'Are', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Hi', 'How', 'I', 'K', 'L', 'O', 'Y', 'You', 'Z']

cap1 = cv2.VideoCapture('tut.mp4')


def tutshow():
    while True:
        succ, tutFrame = cap1.read()
        cv2.imshow('Tutorial', tutFrame)
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break


def detectionfunction():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=2)
    classifier = Classifier('Model/keras_model.h5', 'Model/labels.txt')
    while True:
        success, img = cap.read()
        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]  # Since only one hand
            x, y, w, h = hand['bbox']
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset: y + h + offset, x - offset: x + w + offset]

            aspectRatio = h / w
            if aspectRatio > 1:
                k = imgSize / h
                wCalculated = math.ceil(w * k)
                imgResized = cv2.resize(imgCrop, (wCalculated, imgSize))
                wGap = math.ceil((imgSize - wCalculated) / 2)
                imgWhite[:, wGap: wGap + wCalculated] = imgResized
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
            else:
                k = imgSize / w
                hCalculated = math.ceil(h * k)
                imgResized = cv2.resize(imgCrop, (imgSize, hCalculated))
                hGap = math.ceil((imgSize - hCalculated) / 2)
                imgWhite[hGap: hGap + hCalculated, :] = imgResized
                prediction, index = classifier.getPrediction(imgWhite, draw=False)

            cv2.putText(imgOutput, labels[index], (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 69, 100), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 69, 100), 4)

        cv2.imshow("Sign Language Detector", imgOutput)
        cv2.moveWindow('Sign Language Detector', 200, 200)
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break


window = Tk()
window.geometry("1920x1080")
window.title("Real Time Sign Language Detector")
window.config(background='black')


# Define a function to switch between frames
def switch_frame(frame_to_show, frame_to_hide):
    frame_to_show.pack(fill=BOTH, expand=True)  # Pack the frame to show
    frame_to_hide.pack_forget()  # Hide the other frame


Menu_frame = Frame(window, bg="black")
Tutorial_frame = Frame(window, bg="black")

# Pack both frames initially (only one will be visible)
Menu_frame.pack(fill=BOTH, expand=True)

img=PhotoImage(file="bg.png")
Imglabel=Label(master=Menu_frame,
              image=img)
Imglabel.pack()
Tlabel = Label(master=Menu_frame,
               text="REAL TIME \nSIGN LANGUAGE DETECTOR",
               font=("Arial", 60, "bold", "italic"),
               bg="white",
               fg="cyan",
               highlightbackground="white",
               highlightthickness=2)
Tlabel.place(relx=.5, rely=.3, anchor="center")

canvas = Canvas(master=Menu_frame, width=1920, height=1080)  # Adjust dimensions
canvas.pack()

# Background image (assuming "bg.png" exists)
canvas.create_image(0, 0, anchor="nw", image=img)  # Adjust image placement

# Text with desired styling
canvas.create_text(650, 200, text="REAL TIME",
                   font=("Arial", 60, "bold", "italic"), fill="cyan")
canvas.create_text(650, 300, text="SIGN LANGUAGE DETECTOR",
                   font=("Arial", 60, "bold", "italic"), fill="cyan")



Sbtn = Button(master=Menu_frame,
              text="Start",
              font=("Arial", 32, "italic"),
              fg="cyan",
              bg="black",
              activebackground="black",
              activeforeground="cyan", bd=10, padx=20,
              command=detectionfunction)
Sbtn.place(relx=.3, rely=.8, anchor="center")

Tbtn = Button(master=Menu_frame,
              text="Tutorial",
              font=("Arial", 32, "italic"),
              fg="cyan",
              bg="black",
              activebackground="black",
              activeforeground="cyan", bd=10, padx=20,
              command=tutshow)
Tbtn.place(relx=.7, rely=.8, anchor="center")

Tutlabel = Label(master=Tutorial_frame,
                text="TUTORIAL",
                font=("Arial", 60, "bold", "italic"),
                bg="black",
                fg="cyan",
                highlightbackground="white",
                highlightthickness=2)
Tutlabel.place(relx=.5, rely=.3, anchor="center")

video_name = "tut.mp4" 

window.mainloop()
