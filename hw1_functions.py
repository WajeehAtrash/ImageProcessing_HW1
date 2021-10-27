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
    row,col=im.shape
    nim = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
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
    row, col = im.shape
    Slices=np.zeros((row*col,256))
    for i in range(256):
        curr_slice=((im==i )*1).flatten()
        Slices[:,i]=curr_slice
    return Slices


def SLTmap(im1, im2):
    # TODO: implement fucntion
    row, col = im1.shape
    im1_slices=sliceMat(im1)
    TM = np.zeros((256,1))
    for i in range(256):
        grayScale_col=im1_slices[:,i]
        pixels_num =grayScale_col.sum()# pixel num
        grayScale_col=grayScale_col.reshape(row,col)
        if pixels_num==0:
            continue
        inew=(im2*grayScale_col).sum()/pixels_num
        TM[i]=inew
    return mapImage(im1, TM), TM


def mapImage (im,tm):
    # TODO: implement fucntion
    row, col = im.shape
    Slices=sliceMat(im)
    TMim=np.matmul(Slices,tm)
    TMim=TMim.reshape(row,col)
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
    im, TM = SLTmap(darkimg_gray, darkimg_gray)
    print(TM)
    # path_image = r'D:\ImageProcessing\HW1\Images\lena.tif'
    # darkimg2 = cv2.imread(path_image)
    # darkimg_gray2 = cv2.cvtColor(darkimg2, cv2.COLOR_BGR2GRAY)
    # d= meanSqrDist(darkimg_gray,darkimg_gray2)
    # print(d)
    # vec=np.mat = np.arange(0,256).reshape(256,1)
    # im= mapImage(darkimg_gray,vec)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(darkimg)
    plt.title('original')

    plt.subplot(1, 2, 2)
    plt.imshow(im, cmap='gray', vmin=0, vmax=255)
    plt.title('Tone Maping')
    plt.show()
    # im,TM=SLTmap(darkimg_gray,darkimg_gray)
    input()