import cv2
import numpy as np
import math

scale = 1
delta = 0
img = cv2.imread('picture/lena.png')
#img = cv2.imread('picture/123.png')

print(img.shape)

img_gr = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

img_B = img[:,:, 0]
img_G = img[:,:, 1]
img_R = img[:,:, 2]


img_Gray = np.zeros((img.shape[0],img.shape[1]),np.uint8)
for a in range(img.shape[0]):
    for b in range(img.shape[1]):
        img_Gray[a,b]= (img[a,b, 0]*0.114) + (img[a,b, 1]*0.587) + (img[a,b, 2] * 0.299)

cv2.imshow('origin',img)
cv2.imshow('img_myselfGray',img_Gray)
cv2.imshow('img_gray',img_gr)


cv2.waitKey(0)
cv2.destroyAllWindows()
