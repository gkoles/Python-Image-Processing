import numpy as np
import cv2
import matplotlib.pyplot as plt
import imutils

img = cv2.imread('excerpt.png',0)
#img = cv2.imread('wood2.jpg',0)
resized = imutils.resize(img, width=600)
ratio = img.shape[0] / float(resized.shape[0])

blurred = cv2.GaussianBlur(resized,(5,5),0)
kernel = np.ones((5,5),np.uint8)
gradient = cv2.morphologyEx(blurred, cv2.MORPH_GRADIENT, kernel)

#ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#
# thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
#            cv2.THRESH_BINARY,11,2)

#ret,thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret,thresh = cv2.threshold(resized,127, 255, cv2.THRESH_BINARY)
image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

color = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
contours_img = cv2.drawContours(color, contours, -1, (128,255,0), 2)

cnt = contours[0]
M = cv2.moments(cnt)
print (M)

area = cv2.contourArea(cnt)
print ("Surface area: %d" %area)


epsilon = 0.2*cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt,epsilon,True)
area2 = cv2.contourArea(approx)
print ("Surface area: %d" %area2)

cv2.imshow('image',contours_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
