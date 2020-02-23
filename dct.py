import cv2
import numpy as np
import math
import time
from numba import jit
 
@jit
def func(gray,dcttra):
    for u in range(0,gray.shape[0],8):
        for v in range(0,gray.shape[1],8):
            for i in range(0,8):
                for j in range(0,8):
                    temp = 0
                    for k in range(0,8):
                        for l in range(0,8):
                            k_cos = math.cos((2*k+1)*i*math.pi*0.0625)
                            l_cos = math.cos((2*l+1)*j*math.pi*0.0625)
                            temp = temp + gray[u+k,v+l]*k_cos*l_cos
                    if i == 0:
                        temp = temp/(2**0.5)
                    if j == 0:
                        temp = temp/(2**0.5)
                    dcttra[u+i,v+j] = temp*0.25
 
@jit 
def funcc(dcttra,idct):
    for u in range(0,dcttra.shape[0],8):
        for v in range(0,dcttra.shape[1],8):
            for i in range(0,8):
                for j in range(0,8):
                    temp = 0
                    for k in range(0,8):
                        for l in range(0,8):
                            k_cos = math.cos((2*i+1)*k*math.pi*0.0625)
                            l_cos = math.cos((2*j+1)*l*math.pi*0.0625)
                            temp2 = dcttra[u+k,v+l]*k_cos*l_cos
                            if k == 0:
                                temp2 = temp2 /(2**0.5)
                            if l == 0:
                                temp2 = temp2/(2**0.5)
                            temp = temp + temp2
                    temp = temp*0.25
                    idct[u+i,v+j] = temp
 

@jit 
def funnn(dcttra,lowfilter):
    temp = 0
    for i in range(0,dcttra.shape[0]):
        for j in range(0,dcttra.shape[1]):
            temp = temp + dcttra[i][j]
    for i in range(0,dcttra.shape[0],8):
        for j in range(0,dcttra.shape[1],8):        
            for k in range(0,8):
                for l in range(0,8):
                    lowfilter[i+k][j+l] = dcttra[i+k][j+l]*100 / temp




img = cv2.imread('picture/lena.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
dcttra = np.zeros(gray.shape,np.float32)
idct = np.zeros(gray.shape,np.float32)
lowfilter = np.zeros(gray.shape,np.float32)
ilowfilter = np.zeros(gray.shape,np.float32)


mask = np.array([[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5],
                 [0.5,0.7,0.7,0.7,0.7,0.7,0.7,0.5],
                 [0.5,0.7,0.9,0.9,0.9,0.9,0.7,0.5],
                 [0.5,0.7,0.9,1.0,1.0,0.9,0.7,0.5],
                 [0.5,0.7,0.9,1.0,1.0,0.9,0.7,0.5],
                 [0.5,0.7,0.9,0.9,0.9,0.9,0.7,0.5],
                 [0.5,0.7,0.7,0.7,0.7,0.7,0.7,0.5],
                 [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]],dtype = 'float32')
mask2 = np.array([[1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0],
                  [1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0],
                  [1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0],
                  [1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                  [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                  [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                  [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                  [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]],dtype = 'float32')


#cv2.imshow('ori',img)
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img1 = np.float32(img)


func(gray,dcttra)
funnn(dcttra,lowfilter)
funcc(lowfilter,ilowfilter)
funcc(dcttra,idct)



cv2.imshow('frequency',ilowfilter.astype(np.uint8))
cv2.imshow('Transform',lowfilter.astype(np.uint8))
#cv2.imshow('DCT',dcttra.astype(np.uint8))
cv2.imshow('GRAY',idct.astype(np.uint8))
#img2 = np.uint8(img_dict)




cv2.waitKey(0)
cv2.destroyAllWindows()