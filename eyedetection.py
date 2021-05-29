import cv2


class EyeTracker:
    def __init__(self, faceCascadePath, eyesCascadePath):
        self.faceCascade = cv2.CascadeClassifier(faceCascadePath)
        self.eyesCascade = cv2.CascadeClassifier(eyesCascadePath)

    def track(self,image, scaleFactor =1.1, minNeighbors= 5, minSize = (30,30)):
        faceRects = self.faceCascade.detectMultiScale(image,
                                                  scaleFactor= scaleFactor,
                                                  minNeighbors=minNeighbors,
                                                  minSize = minSize,
                                                  flags= cv2.CASCADE_SCALE_IMAGE)
        rects = []

        for (fx , fy , fw, fh) in faceRects:
            RoI = image[fy:fy+fh, fx:fx+fw]
            rects.append((fx, fy, fx+fw, fy+fh))

            eyesRect = self.eyesCascade.detectMultiScale(RoI,
                                                         scaleFactor=1.1,
                                                         minNeighbors=10,
                                                         minSize= (20,20),
                                                         flags = cv2.CASCADE_SCALE_IMAGE)

            for (ex, ey, ew, eh) in eyesRect:
                rects.append((fx+ex, fy+ey, fx+ex+ew, fy+ey+eh))

        return rects




