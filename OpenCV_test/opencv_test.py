import numpy as np
import cv2
import matplotlib.pyplot as plt

# img = cv2.imread('shapes_and_colors.png',0)
img = cv2.imread('shapes_and_colors.png',0)

blur = cv2.GaussianBlur(img,(5,5),0)
kernel = np.ones((5,5),np.uint8)
gradient = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, kernel)

#ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#
# thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
#            cv2.THRESH_BINARY,11,2)

ret,thresh = cv2.threshold(gradient,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

color = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
contours_img = cv2.drawContours(color, contours, -1, (128,255,0), 2)

cnt = contours[0]
M = cv2.moments(cnt)
print (M)

area = cv2.contourArea(cnt)
print ("Surface area: %d" %area)

cv2.imshow('image',contours_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
