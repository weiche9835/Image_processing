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
            for a in range(0,5):
                for b in range(0,5):
                    temp = temp + (grayBorder[i+a,j+b]*mask_gaussian[a,b])
            temp = temp/273
            gaussian[i,j] = round(temp)

img = cv2.imread('picture/lena.png')

#avg mask
mask_gaussian = np.array([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]],dtype = 'int16')


#BGR to Gray
gray = np.zeros((img.shape[0],img.shape[1]),np.uint8)

for a in range(img.shape[0]):
    for b in range(img.shape[1]):
        gray[a,b]= (img[a,b, 0]*0.114) + (img[a,b, 1]*0.587) + (img[a,b, 2] * 0.299)


grayBorder = np.zeros((gray.shape[0]+4,gray.shape[1]+4),np.uint8)
grayBorder[2:-2,2:-2] = gray

grayBorder[0,0]=gray[2,2]
grayBorder[0,1]=gray[1,2]
grayBorder[1,0]=gray[2,1]
grayBorder[1,1]=gray[1,1]

grayBorder[0,-1]=gray[2,-3]
grayBorder[0,-2]=gray[1,-3]
grayBorder[1,-1]=gray[2,-2]
grayBorder[1,-2]=gray[1,-2]

grayBorder[-1,-1]=gray[-3,-3]
grayBorder[-2,-1]=gray[-3,-2]
grayBorder[-1,-2]=gray[-2,-3]
grayBorder[-2,-2]=gray[-2,-2]

grayBorder[-1,0]=gray[-3,2]
grayBorder[-2,0]=gray[-3,1]
grayBorder[-1,1]=gray[-2,2]
grayBorder[-2,1]=gray[-2,1]

grayBorder[0,2:-2]=gray[2,:]
grayBorder[2:-2,0]=gray[:,2]
grayBorder[-2,2:-2]=gray[-3,:]
grayBorder[2:-2,-2]=gray[:,-3]
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

gau = cv2.GaussianBlur(gray,(5,5),0)
cv2.imshow('gaussianAPI',gau)

print(gray[100:120,200])
print('-----------------------------')
print(gaussian[100:120,200])
print('-----------------------------')
print(gau[100:120,200])


cv2.waitKey(0)
cv2.destroyAllWindows()