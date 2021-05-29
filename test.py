import cv2
import numpy as np
import imutils
import time

lowerRed = np.array([0, 10, 0], dtype='uint8')
upperRed = np.array([0,59, 204], dtype='uint8')

camera = cv2.VideoCapture(0)

while True:

    timer = cv2.getTickCount()

    (grabed, frame) = camera.read()
    fps = cv2.getTickFrequency() / (cv2.getTickCount()-timer)


    red = cv2.inRange(frame, lowerRed, upperRed)
    red = cv2.GaussianBlur(red, (5, 5), 0)


    cv2.imshow('frame', red)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
