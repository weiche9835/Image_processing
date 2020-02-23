import cv2
import numpy as np
import math
import time
from numba import jit
 
@jit
def func(gray,dcttra,m,n):
    for u in range(0,gray.shape[0],m):
        for v in range(0,gray.shape[1],n):
            for i in range(0,m):
                for j in range(0,n):
                    temp = 0
                    for k in range(0,m):
                        for l in range(0,n):
                            k_cos = math.cos((2*k+1)*i*math.pi/(2*m))
                            l_cos = math.cos((2*l+1)*j*math.pi/(2*n))
                            temp = temp + gray[u+k,v+l]*k_cos*l_cos
                    temp = temp/((n*m)**0.5)
                    if i != 0:
                        temp = temp*(2**0.5)
                    if j != 0:
                        temp = temp*(2**0.5)
                    dcttra[u+i,v+j] = temp
 
@jit 
def funcc(dcttra,idct,m,n):
    for u in range(0,dcttra.shape[0],m):
        for v in range(0,dcttra.shape[1],n):
            for i in range(0,m):
                for j in range(0,n):
                    temp = 0
                    for k in range(0,m):
                        for l in range(0,n):
                            k_cos = math.cos((2*i+1)*k*math.pi/(2*m))
                            l_cos = math.cos((2*j+1)*l*math.pi/(2*n))
                            temp2 = dcttra[u+k,v+l]*k_cos*l_cos
                            temp2 = temp2 /(m**0.5)
                            temp2 = temp2 /(n**0.5)
                            if k != 0:
                                temp2 = temp2*(2**0.5)
                            if l != 0:
                                temp2 = temp2*(2**0.5)
                            temp = temp + temp2
                    temp = temp
                    idct[u+i,v+j] = temp
 



m = 16
n = 16

img = cv2.imread('picture/lena.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
dcttra = np.zeros(gray.shape,np.float32)
idct = np.zeros(gray.shape,np.float32)


cv2.imshow('ori',img)
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img1 = np.float32(img)


func(gray,dcttra,m,n)
funcc(dcttra,idct,m,n)



cv2.imshow('DCT',dcttra.astype(np.uint8))
cv2.imshow('IDCT',idct.astype(np.uint8))





cv2.waitKey(0)
cv2.destroyAllWindows()