from hw1_functions import *

if __name__ == "__main__":
    # feel free to add/remove/edit lines

    path_image = r'Images\darkimage.tif'
    darkimg = cv2.imread(path_image)
    darkimg_gray = cv2.cvtColor(darkimg, cv2.COLOR_BGR2GRAY)

    print("Start running script  ------------------------------------\n")
    print_IDs()

    print("a ------------------------------------\n")
    enhanced_img, a, b = contrastEnhance(darkimg_gray,[0,255])  # add parameters

    # display images
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(darkimg)
    plt.title('original')

    plt.subplot(1, 2, 2)
    plt.imshow(enhanced_img, cmap='gray', vmin=0, vmax=255)
    plt.title('enhanced contrast')

    # print a,b
    print("a = {}, b = {}\n".format(a, b))

    # display mapping
    showMapping(darkimg_gray,a,b)  # add parameters

    print("b ------------------------------------\n")
    enhanced2_img, a, b = contrastEnhance(enhanced_img,[0,255])  # add parameters
    # print a,b
    print("enhancing an already enhanced image\n")
    print("a = {}, b = {}\n".format(a, b))

    # TODO: display the difference between the two image (Do not simply display both images)
    print("Minkowski distance between 1Xenhanced and 2Xenhanced image ")
    enhanced_d=minkowski2Dist(enhanced_img,enhanced2_img)
    print("d = {}\n".format(enhanced_d))
    print("c ------------------------------------\n")
    mdist = minkowski2Dist(darkimg_gray,darkimg_gray)  # add parameters
    print("Minkowski dist between image and itself\n")
    print("d = {}\n".format(mdist))

    # TODO:
    # implement the loop that calculates minkowski distance as function of increasing contrast
    #using the first image
    img_max=np.max(darkimg_gray)
    img_min=np.min(darkimg_gray)
    step=(img_max-img_min)/20
    contrast = [img_min]
    dists = [0]
    for k in range(1,21):
        if(img_min+step*k<=img_max):
            contrast.append(img_min+step*k)
            enh_img,_,_=contrastEnhance(darkimg_gray,[img_min,img_min+step*k])
            dists.append(minkowski2Dist(darkimg_gray,enh_img))
        else:
            break

    plt.figure()
    plt.plot(contrast, dists)
    plt.xlabel("contrast")
    plt.ylabel("distance")
    plt.title("Minkowski distance as function of contrast")

    print("d ------------------------------------\n")
    path_image = r'Images\lena.tif'
    lena = cv2.imread(path_image)
    lena_gray = cv2.cvtColor(darkimg, cv2.COLOR_BGR2GRAY)
    lena_slices=sliceMat(lena_gray)
    color_vec=np.arange(0,256).reshape(256,1)
    sliced_img=np.matmul(lena_slices,color_vec)
    row,col=lena_gray.shape
    sliced_img=sliced_img.reshape(row,col)
    d =  meanSqrDist(lena_gray,sliced_img)# computationally prove that sliceMat(im) * [0:255] == im
    print("mean square distance between an image and slicedMat*[0:255] \n d={} ".format(d))

    print("e ------------------------------------\n")
    TMim,TM=SLTmap(darkimg_gray,enhanced_img)
    TMim=mapImage(darkimg_gray,TM)
    d =  meanSqrDist(darkimg_gray,TMim)
    print("sum of diff between image and slices*[0..255] = {}".format(d))
    #
    # then display
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(darkimg)
    plt.title("original image")
    plt.subplot(1, 2, 2)
    plt.imshow(TMim, cmap='gray', vmin=0, vmax=255)
    plt.title("tone mapped")

    print("f ------------------------------------\n")
    negative_im = sltNegative(darkimg_gray)
    plt.figure()
    plt.imshow(negative_im, cmap='gray', vmin=0, vmax=255)
    plt.title("negative image using SLT")

    print("g ------------------------------------\n")
    thresh = 120  # play with it to see changes
    lena = cv2.imread(r"Images\lena.tif")
    lena_gray = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)
    thresh_im = sltThreshold(lena_gray,thresh)  # add parameters

    plt.figure()
    plt.imshow(thresh_im, cmap='gray', vmin=0, vmax=255)
    plt.title("thresh image using SLT")

    print("h ------------------------------------\n")
    im1 = lena_gray
    path_image = r'Images\cups.tif'
    SLT_im = cv2.imread(path_image)
    im2 =cv2.cvtColor(SLT_im, cv2.COLOR_BGR2GRAY)
    SLTim,_ =SLTmap(im1,im2)
    # then print
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(im1)
    plt.title("original image")
    plt.subplot(1, 3, 2)
    plt.imshow(SLTim, cmap='gray', vmin=0, vmax=255)
    plt.title("tone mapped")
    plt.subplot(1, 3, 3)
    plt.imshow(im2, cmap='gray', vmin=0, vmax=255)
    plt.title("tone mapped")

    d1 = meanSqrDist(im1,im2)
    d2 = meanSqrDist(im1,SLTim)
    print("mean sqr dist between im1 and im2 = {}\n".format(d1))
    print("mean sqr dist between mapped image and im2 = {}\n".format(d2))

    print("i ------------------------------------\n")
    # prove comutationally
    SLTim2,_=SLTmap(im2,im1)
    d =  meanSqrDist(im2,SLTim)
    print(" {}".format(d))
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(im2)
    plt.title("original image")
    plt.subplot(1, 3, 2)
    plt.imshow(SLTim2, cmap='gray', vmin=0, vmax=255)
    plt.title("tone mapped")
    plt.subplot(1, 3, 3)
    plt.imshow(im1, cmap='gray', vmin=0, vmax=255)
    plt.title("tone mapped")
    plt.show()