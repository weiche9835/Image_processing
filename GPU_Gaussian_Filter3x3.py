import cv2
import numpy as np
import math
import time
from numba import jit


@jit
def func(gray,grayBorder,mask_gaussian,gaussian):
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            temp = 0
            for a in range(0,3):
                for b in range(0,3):
                    temp = temp + (grayBorder[i+a,j+b]*mask_gaussian[a,b])
            temp = temp/16
            gaussian[i,j] = round(temp)
        


img = cv2.imread('picture/lena.png')
img = cv2.imread('picture/airplane.png')


#avg mask
mask_gaussian = np.array([[1,2,1],[2,4,2],[1,2,1]],dtype = 'int16')


#BGR to Gray
gray = np.zeros((img.shape[0],img.shape[1]),np.uint8)

for a in range(img.shape[0]):
    for b in range(img.shape[1]):
        gray[a,b]= (img[a,b, 0]*0.114) + (img[a,b, 1]*0.587) + (img[a,b, 2] * 0.299)


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
cv2.imshow('grayAddBorder',grayBorder)


grayBorder = grayBorder.astype('int16')

gaussian = np.zeros(gray.shape,np.int16)

stime = time.time()
func(gray,grayBorder,mask_gaussian,gaussian)
etime = time.time() - stime
print('用時:{}秒'.format(etime))

grayBorder = grayBorder.astype('uint8')
gaussian = gaussian.astype('uint8')

      
cv2.imshow('gaussian',gaussian)

gau = cv2.GaussianBlur(gray,(3,3),0)
cv2.imshow('gaussianAPI',gau)

print(gaussian[100:120,200])
print('-----------------------------')
print(gau[100:120,200])


cv2.waitKey(0)
cv2.destroyAllWindows()