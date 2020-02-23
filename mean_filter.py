import cv2
import numpy as np
import math

img = cv2.imread('picture/lena.png')
#img = cv2.imread('Butterfly.jpg')

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
        temp = 0
        for a in range(0,3):
            for b in range(0,3):
                temp = temp + (grayBorder[i+a,j+b]*mask[a,b])
        temp = temp / 9
        blur[i,j] = round(temp)   
dst = cv2.blur(gray,(3,3))        

cv2.imshow('blurAPI',dst)
cv2.imshow('blur',blur)

print(dst)
print('-------------------------------------')
print(blur)

cv2.waitKey(0)
cv2.destroyAllWindows()