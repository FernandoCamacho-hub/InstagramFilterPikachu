# Program to create Instagram Face Filters using Opencv with Python.
# Date: 25 de Marzo de 2022
# Authors:
# Jorge Carrillo Castro A01634630 
# Paulina Lizet Gutiérrez Amezcua A01639948
# Salvador Fernando Camacho Hernández A01634777
# Source code reference:
# Canu, S (2019) Pig's nose (Face instagram filters) - Opencv with Python [Source code]. 
# https://pysource.com/2019/03/25/pigs-nose-instagram-face-filter-opencv-with-python/

import cv2
import numpy as np
import dlib
from math import hypot

# Loading Camera and Nose image and Creating mask
cap = cv2.VideoCapture(0)
duck_img = cv2.imread("pikachu.png") # 276 x 276
_, frame = cap.read()
rows, cols, _ = frame.shape
duck_mask = np.zeros((rows, cols), np.uint8)

# Loading Face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#comienza la parte 2 del codigo 
while True:
    _, frame = cap.read()
    duck_mask.fill(0)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(frame)
    for face in faces:
        landmarks = predictor(gray_frame, face)

        #face coordinates
        top_left = (landmarks.part(17).x, landmarks.part(17).y)
        top_right = (landmarks.part(26).x, landmarks.part(26).y)
        bottom_left = (landmarks.part(5).x, landmarks.part(5).y)
        bottom_right = (landmarks.part(11).x, landmarks.part(11).y)
        
        duck_face_width = int(hypot(top_left[0] - top_right[0], 
                                    top_left[1] - top_right[1]))
        duck_face_height = int(duck_face_width)