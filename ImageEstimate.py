import cv2
import numpy as np
import math
import time
from numba import jit



@jit
def MEAN(gray,u,v):
    mean = 0
    for i in range(8):
        for j in range(8):
            mean = mean + gray[u+i][v+j]
    mean = mean/(8*8)
    return mean
@jit
def MSE(gray,mean,u,v):
    mse = 0
    for i in range(8):
        for j in range(8):
            mse = mse + ((gray[u+i][v+j] - mean)**2)
    mse = mse/((8*8)-1)
    mse = mse**0.5
    return mse
@jit    
def CC(gray,gray2,meanx,meany,u,v):
    cc = 0
    for i in range(8):
        for j in range(8):
            cc = cc + ((gray[u+i][v+j] - meanx)*(gray2[u+i][v+j] - meany))
    cc = cc/((8*8)-1)
    return cc
def Calssim(gray,gray2):
    gray = cv2.cvtColor(gray,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(gray2,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    gray2 = np.float32(gray2)
    ans = 0
    if(gray.shape[0]==gray2.shape[0]) and (gray.shape[1]==gray2.shape[1]):
        for u in range(0,gray.shape[0],8):
            for v in range(0,gray.shape[1],8):
                C1 = (0.01*255)**2 #C1 = (K1*L)**2
                C2 = (0.03*255)**2 #C2 = (K2*L)**2
                C3 = C2/2 #C3 = C2/2
                meanx = MEAN(gray,u,v)
                meany = MEAN(gray2,u,v)
                
                msex = MSE(gray,meanx,u,v)
                msey = MSE(gray2,meany,u,v)
                
                cc = CC(gray,gray2,meanx,meany,u,v)
                
                #l(x,y)
                lxy = (2*meanx*meany + C1)/(meanx**2 + meany**2 + C1)
                #c(x,y)
                cxy = (2*msex*msey + C2)/(msex**2 + msey**2 + C2)
                #s(x,y)
                sxy = (cc + C3)/(msex*msey + C3)
                
                ssim = lxy*cxy*sxy
                ''' the same funciton to calculate ssim
                ssim = (2*meanx*meany + C1)*(2*cc + C2)
                ssim = ssim/(((meanx**2) + (meany**2) + C1)*((msex**2) + (msey**2) + C2))
                '''
                ans = ans + ssim
        ans = ans/((gray.shape[0]/8)*(gray.shape[1]/8))
        return ans
    else:
        raise BaseException("Images are different size")
@jit
def Calmse(gray,gray2):
    if(gray.shape[0]==gray2.shape[0]) and (gray.shape[1]==gray2.shape[1]):
        ans = 0
        for i in range(gray.shape[0]):
            for j in range(gray.shape[1]):
                ans = ans + (gray[i][j] - gray2[i][j])**2
        ans = ans/(gray.shape[0]*gray.shape[1])
        return ans
    else:
        raise BaseException("Images are different size")

def Calpsnr(gray,gray2):
    if(gray.shape[0]==gray2.shape[0]) and (gray.shape[1]==gray2.shape[1]):
        gray = cv2.cvtColor(gray,cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(gray2,cv2.COLOR_BGR2GRAY)
        mse = Calmse(gray,gray2)
        if mse ==0:
            return float("inf")
        else:
            ans = (255**2)/mse
            return 10*math.log(ans,10)
    else:
        raise BaseException("Images are different size")



if __name__ == '__main__':
    print("This is for images' psnr and ssim")

