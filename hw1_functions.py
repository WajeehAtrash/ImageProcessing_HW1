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
    hist_im1,_ = np.histogram(im1,bins=256,range=(0,255))#first  image histogram
    hist_im2,_ = np.histogram(im2,bins=256,range=(0,255))#second  image histogram
    #calculating the num of pixels in each image
    N=hist_im1.sum()
    d=np.sum(abs(hist_im1-hist_im2)**2)**(0.5)

    return d


def meanSqrDist(im1, im2):
	# TODO: implement fucntion - one line
    d=np.sum((im1.astype("float") - im2.astype("float")) ** 2)/float(im1.shape[0] * im2.shape[1])
    return d


def sliceMat(im):
    # TODO: implement fucntion
    Slices=np.zeros((len(im[0])*len(im),256))
    for i in range(256):
        curr_slice=((im==i )*1).flatten()
        Slices[:,i]=curr_slice
    return Slices


# def SLTmap(im1, im2):
#     # TODO: implement fucntion
#     return mapImage(im1, TM), TM
#
#
def mapImage (im,tm):
    # TODO: implement fucntion
    Slices=sliceMat(im)
    TMim=np.matmul(Slices,tm)
    TMim=TMim.reshape(len(im[0]),len(im))
    return TMim


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
    # path_image = r'D:\ImageProcessing\HW1\Images\lena.tif'
    # darkimg2 = cv2.imread(path_image)
    # darkimg_gray2 = cv2.cvtColor(darkimg2, cv2.COLOR_BGR2GRAY)
    # d= meanSqrDist(darkimg_gray,darkimg_gray2)
    # print(d)
    vec=np.mat = np.arange(0,256).reshape(256,1)
    im= mapImage(darkimg_gray,vec)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(darkimg)
    plt.title('original')

    plt.subplot(1, 2, 2)
    plt.imshow(im, cmap='gray', vmin=0, vmax=255)
    plt.title('Tone Maping')
    plt.show()
    input()