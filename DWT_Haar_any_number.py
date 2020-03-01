import cv2
import numpy as np
import math
import time
from numba import jit

 
@jit
def dwt(gray,dwtim,deep,temp):
    deepcount = 0
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            dwtim[i][j] = gray[i][j]
    while(deepcount<deep):
        deepcount = deepcount + 1
        x = int(gray.shape[0]/deepcount)
        y = int(gray.shape[1]/deepcount)
        for u in range(0,x):
            for v in range(0,y/2):
                temp[u][v] = (dwtim[u][2*v] + dwtim[u][2*v+1])/2
                temp[u][v+int(y/2)] = abs((dwtim[u][2*v] - dwtim[u][2*v+1]))
        for u in range(0,x/2):
            for v in range(0,y):
                dwtim[u][v] = (temp[2*u][v] + temp[(2*u)+1][v])/2
                dwtim[u+int(x/2)][v] = abs((temp[2*u][v] - temp[(2*u)+1][v]))
     
img = cv2.imread('picture/lena.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
dwtim = np.zeros(gray.shape,np.int16)
temp = np.zeros(gray.shape,np.int16)
gray = gray.astype(np.int16)
deep = 2



dwt(gray,dwtim,deep,temp)

cv2.imshow('DWT',dwtim.astype(np.uint8))






cv2.waitKey(0)
cv2.destroyAllWindows()