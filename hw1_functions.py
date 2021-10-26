import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

def print_IDs():
	#print("123456789")
    print("322773946+987654321\n")


def contrastEnhance(im,range1):
    # TODO: implement fucntion
    a = (range1[1] - range1[0]) / 255
    b = range1[0]
    width = len(im[0])
    height = len(im)
    nim = np.zeros((width, height))
    for i in range(width):
        for j in range(height):
            nim[i][j] = (im[i][j] * a) + b
    return nim, a, b


def showMapping(old_range, a, b):
    imMin = np.min(old_range)
    imMax = np.max(old_range)
    x = np.arange(imMin, imMax+1, dtype=np.float)
    y = a * x + b
    plt.figure()
    plt.plot(x, y)
    plt.xlim([0, 255])
    plt.ylim([0, 255])
    plt.title('contrast enhance mapping')

#TODO:Redo it
def minkowski2Dist(im1,im2):
    # TODO: implement fucntion
    hist_im1 = cv2.calcHist([im1], channels=[0], mask=None, histSize=[256], ranges=[0, 256])#first  image histogram
    hist_im2 = cv2.calcHist([im2], channels=[0], mask=None, histSize=[256], ranges=[0, 256])#second  image histogram
    #calculating the num of pixels in each image
    pixels_im1= len(im1[0])*len(im1)
    pixels_im2 = len(im2[0])*len(im2)
    d=0
    # lib_d=distance.minkowski(hist_im1,hist_im2,2)
    # print(lib_d)
    for i in range(256):
       p_A=(hist_im1[i]/pixels_im1)#Pa(k)
       p_B=(hist_im2[i]/pixels_im2)#Pb(k)
       d=d+abs((p_A-p_B)**2)
    d=d**(1/2)

    return np.asscalar(d)


def meanSqrDist(im1, im2):
	# TODO: implement fucntion - one line
    d=np.sum((im1.astype("float") - im2.astype("float")) ** 2)/float(im1.shape[0] * im2.shape[1])
    return d


def sliceMat(im):
    # TODO: implement fucntion

    return Slices


# def SLTmap(im1, im2):
#     # TODO: implement fucntion
#     return mapImage(im1, TM), TM
#
#
# def mapImage (im,tm):
#     # TODO: implement fucntion
#     return TMim
#
#
# def sltNegative(im):
# 	# TODO: implement fucntion - one line
#     return nim
#
#
# def sltThreshold(im, thresh):
#     # TODO: implement fucntion
#     return nim
if __name__ == '__main__':
    path_image = r'D:\ImageProcessing\HW1\Images\fruit.tif'
    darkimg = cv2.imread(path_image)
    darkimg_gray = cv2.cvtColor(darkimg, cv2.COLOR_BGR2GRAY)
    # #_______________________________________________________
    path_image = r'D:\ImageProcessing\HW1\Images\lena.tif'
    darkimg2 = cv2.imread(path_image)
    darkimg_gray2 = cv2.cvtColor(darkimg2, cv2.COLOR_BGR2GRAY)
    # #____________________________________________________
    # my_d = minkowski2Dist(darkimg_gray, darkimg_gray2);
    # print("my function=", my_d)
    meanSquaredError=meanSqrDist(darkimg_gray,darkimg_gray2)
    print("mean squaredError=",meanSquaredError)