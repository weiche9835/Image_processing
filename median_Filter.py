import random
import cv2
import numpy as np
import math


list2 = []
img = cv2.imread('picture/sails.png')

mask = np.array([[1,1,1],[1,1,1],[1,1,1]],dtype = 'uint8')


#RGB to Gray
gray = np.zeros((img.shape[0],img.shape[1]),np.uint8)

for a in range(img.shape[0]):
    for b in range(img.shape[1]):
        gray[a,b]= (img[a,b, 0]*0.114) + (img[a,b, 1]*0.587) + (img[a,b, 2] * 0.299)

#blur
blur = np.zeros(gray.shape,np.uint8)
grayBorder = np.zeros((gray.shape[0]+2,gray.shape[1]+2),np.uint8)
grayBorder[1:-1,1:-1] = gray
grayBorder[0,0]=gray[1,1]
grayBorder[0,-1]=gray[1,-2]
grayBorder[-1,0]=gray[-2,1]
grayBorder[-1,-1]=gray[-2,-2]

grayBorder[0,1:-1]=gray[1,:]
grayBorder[1:-1,0]=gray[:,1]
grayBorder[-1,1:-1]=gray[-2,:]
grayBorder[1:-1,-1]=gray[:,-2]

for i in range(gray.shape[0]):
    for j in range(gray.shape[1]):
        list2 = []
        for a in range(0,3):
            for b in range(0,3):
                list2.append(grayBorder[i+a,j+b]*mask[a,b])
        for x in range(0,len(list2)-1): 
            for y in range(0,len(list2)-1-x): 
                if list2[y] > list2[y+1]: 
                    tmp = list2[y]
                    list2[y] = list2[y+1]
                    list2[y+1] = tmp
        blur[i,j] = list2[4]   
dst = cv2.medianBlur(gray, 3)        

cv2.imshow('blurAPI',dst)
cv2.imshow('blur',blur)

print(dst[120:150,50])
print('-------------------------------------')
print(blur[120:150,50])

cv2.waitKey(0)
cv2.destroyAllWindows()