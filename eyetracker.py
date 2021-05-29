import cv2
from eyedetection import EyeTracker
import imutils


et = EyeTracker('cascades/haarcascade_frontalface_default.xml', 'cascades/haarcascade_eye.xml')

camera = cv2.VideoCapture(0)

while True:
    (grabbed, frame) = camera.read()

    if not grabbed:
        break

    frame = imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = et.track(gray)

    for rect in rects:
        cv2.rectangle(frame, (rect[0], rect[1]),(rect[2], rect[3]), (0,0,255),2)

        cv2.imshow('tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


camera.release()
cv2.destroyAllWindows()