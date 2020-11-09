from skimage import io, filters, color, measure

import os

from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np

from scipy import ndimage

from math import cos, sin, radians

from scipy.signal import savgol_filter

import math

import pandas as pd

import DataMethods as dm


def getLowerThreshold(jellyimage):
    """
    Returns the expected lower threshold of an image based on mean pixel intensity.
    Through method testing the mean threshold was found to be best at segmenting the jellyfish
    :param jellyimage: rgb np image array (shape: y, x, 3)
    :return: float of intensity to threshold around
    """
    grayJelly = getGrayJelly(jellyimage)
    lower_bound = filters.threshold_mean(grayJelly)
    return lower_bound


def getGrayJelly(jellyimage):
    """
    Returns grayscale np array from an rgb image.
    :param jellyimage: rgb np image array (shape: y, x, 3)
    :return: grayscale np image array (shape: y, x)
    """
    return color.rgb2gray(jellyimage)


def getBinaryJelly(jellyimage, lower_bound=None, upper_bound=1, DEBUG=False):
    """
    Thresholds jellfish into binary True/False image around upper and lower bounds
    :param jellyimage:
    :param lower_bound:
    :param upper_bound:
    :param DEBUG:
    :return: binary np boolean array (shape: y, x)
    """

    jellyimagegray = getGrayJelly(jellyimage)

    if lower_bound is None: lower_bound = getLowerThreshold(jellyimagegray)

    jellyBinary = jellyimagegray > lower_bound
    return jellyBinary


def getJellyImageFromFile(imgfile):
    """
    returns np image array for further processing

    :param imgfile: path object for frame of interest
    :return: rgb np image array (shape: y, x, 3)
    """
    try:
        return io.imread(str(imgfile))
    except Exception as error:
        print('\"{}\" error occured.'.format(error))
        print('frame that error occured on: {}'.format(imgfile))
        print('shape of image using PIL: {}'.format(io.imread(str(imgfile)).shape))
        print('shape of image using Matplotlib: {}'.format(io.imread(str(imgfile), plugin='matplotlib').shape))
    else:
        return io.imread(str(imgfile), plugin='matplotlib')


def getJellyGrayImageFromFile(imgfile):
    image = getJellyImageFromFile(imgfile)
    return getGrayJelly(image)


def getJellyBinaryImageFromFile(imgfile):
    image = getJellyImageFromFile(imgfile)

    binaryJellyImage = np.zeros((image.shape[0], image.shape[1]), dtype=bool)

    for y in range(binaryJellyImage.shape[0]):
        for x in range(binaryJellyImage.shape[1]):
            if np.all(image[y][x]): binaryJellyImage[y][x] = True

    return binaryJellyImage



def findJellyRegion(binaryJellyImage, DEBUG=False):

    labeledmask, numlabels = ndimage.label(binaryJellyImage, structure=[[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    dimensions = list(labeledmask.shape)

    height = dimensions[0]
    width = dimensions[1]

    clusters = measure.regionprops(labeledmask)  # create regions
    jelly = None  # the region that represents the jelly so far
    jellyarea = 0  # largest jelly area so far
    for i in range(0, numlabels):  # finds the jelly by finding which region is max area
        jellydimensions = list(clusters[i].bbox)
        minrow = jellydimensions[0]
        mincol = jellydimensions[1]
        maxrow = jellydimensions[2]
        maxcol = jellydimensions[3]
        if minrow != 0 and mincol != 0 and maxrow != height and maxcol != width:  # checks if area is touching the edges
            if clusters[i].area > jellyarea:
                jelly = clusters[i]
                jellyarea = jelly.area
    if jelly == None:
        return None
    else:
        return jelly


def findJellyRegionWithGray(binaryJellyImage, grayJellyImage, DEBUG=False):

    labeledmask, numlabels = ndimage.label(binaryJellyImage, structure=[[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    dimensions = list(labeledmask.shape)

    height = dimensions[0]
    width = dimensions[1]

    clusters = measure.regionprops(labeledmask, grayJellyImage)  # create regions
    jelly = None  # the region that represents the jelly so far
    jellyarea = 0  # largest jelly area so far
    for i in range(0, numlabels):  # finds the jelly by finding which region is max area
        jellydimensions = list(clusters[i].bbox)
        minrow = jellydimensions[0]
        mincol = jellydimensions[1]
        maxrow = jellydimensions[2]
        maxcol = jellydimensions[3]
        if minrow != 0 and mincol != 0 and maxrow != height and maxcol != width:  # checks if area is touching the edges
            if clusters[i].area > jellyarea:
                jelly = clusters[i]
                jellyarea = jelly.area
    if jelly is None:
        return None
    else:
        return jelly


def findArea(jellyRegion):
    return jellyRegion.area


def findCentroid_regionProp(jellyRegion):
    """
    param:
     jellyRegion:
    return: (x,y) tubple of coordinates of jelly region
    """
    centroid = list(jellyRegion.centroid)

    # skimage returns centroid as a y, x list, needs reversal
    return (int(centroid[1]), int(centroid[0]))

def findCentroid_boundingBox(jellyRegion):
    """
    param:
     jellyRegion:
    return: (x,y) tubple of coordinates of jelly region
    """
    bbox = list(jellyRegion.bbox)

    y_coord = int((bbox[0] + bbox[2]) / 2)        # based on row location
    x_coord = int((bbox[1] + bbox[3]) / 2)      # based on col location

    # skimage returns centroid as a y, x list, needs reversal
    return (x_coord, y_coord)


def findBinaryArea(binaryJellyImage):
    area = 0
    for y in range(binaryJellyImage.shape[0]):
        for x in range(binaryJellyImage.shape[1]):
            if binaryJellyImage[y][x]:
                area += 1
    return area


def getBinaryImageDiff(img1, img2):
    """
    :param img1: np binary image array to be compared
    :param img2: np binary image array to be compared

    :return: np image array with differences
    """
    assert img1.shape == img2.shape

    difference = np.zeros((img1.shape), dtype=bool)

    for y in range(difference.shape[0]):
        for x in range(difference.shape[1]):
            if img1[y][x] != img2[y][x]:
                difference[y][x] = True

    return difference

def getGrayscaleImageDiff(img1, img2):
    """
    :param img1: np grayscale image array to be compared
    :param img2: np grayscale image array to be compared

    :return: np image array with intensity differences (can be positive or negative)
    """
    assert img1.shape == img2.shape

    difference = np.zeros((img1.shape))

    for y in range(difference.shape[0]):
        for x in range(difference.shape[1]):
            difference[y][x] = img1[y][x] - img2[y][x]

    return difference

def applyMask2Img(binaryImgMask, grayscaleImg):
    """
    :param img1: np binary image array to be compared
    :param img2: np grayscale image array to be compared

    :return: np image array with 0's in False positions and native intensity in True positions
    """

    difference = np.zeros((grayscaleImg.shape))

    for y in range(difference.shape[0]):
        for x in range(difference.shape[1]):
            if binaryImgMask[y][x]:
                difference[y][x] = grayscaleImg[y][x]

    return difference


def getGrayscaleImageDiff_absolute(img1, img2):
    """
    :param img1: np grayscale image array to be compared
    :param img2: np grayscale image array to be compared

    :return: np image array with absolute intensity differences
    """
    assert img1.shape == img2.shape

    difference = np.zeros((img1.shape))

    for y in range(difference.shape[0]):
        for x in range(difference.shape[1]):
            difference[y][x] = abs(img1[y][x] - img2[y][x])

    return difference


def dynamicRangeImg_AreaBased(grayImg, binaryPeakDiff, boxRadius2test = 5):
    row_limit = grayImg.shape[0]
    col_limit = grayImg.shape[1]

    dynRngImg = np.ones((grayImg.shape))
    for y in range(dynRngImg.shape[0]):
        for x in range(dynRngImg.shape[1]):
            if binaryPeakDiff[y][x]:
                min = grayImg[y][x]
                max = grayImg[y][x]

                rows2test = range(y - boxRadius2test, y + boxRadius2test + 1)
                cols2test = range(x - boxRadius2test, x + boxRadius2test + 1)

                for y1 in rows2test:
                    for x1 in cols2test:
                        if 0 <= y1 < row_limit and 0 <= x1 < col_limit:
                            t = grayImg[y1][x1]
                            if t < min: min = t
                            if t > max: max = t

                if min != max:
                    dynRngImg[y][x] = max - min

    return dynRngImg


def applyMask2DynamicRangeImg(binaryImgMask, dynamicRangeImg):
    """
    :param img1: np binary image array to be compared
    :param img2: np grayscale image array to be compared

    :return: np image array with 0's in False positions and native intensity in True positions
    """

    difference = np.ones((dynamicRangeImg.shape))

    for y in range(difference.shape[0]):
        for x in range(difference.shape[1]):
            if binaryImgMask[y][x]:
                difference[y][x] = dynamicRangeImg[y][x]

    return difference


def quantifyDifference(diff_image):
    # if being used to compare images they should all have the same thresholding if binary
    metric = 0

    for y in range(diff_image.shape[0]):
        for x in range(diff_image.shape[1]):
            metric += diff_image[y][x]

    return metric

def getMaxIntensity(grayscaleImg):
    return np.amax(grayscaleImg)

def getMinIntensity(grayscaleImg):
    """
    :param grayscaleImg:
    :return: non zero minimum intensity
    """
    min = 1

    for y in range(grayscaleImg.shape[0]):
        for x in range(grayscaleImg.shape[1]):
            if grayscaleImg[y][x] != 0 and grayscaleImg[y][x] < min:
                min = grayscaleImg[y][x]

    return min

def saveHistogram(grayscaleImg, outfile):
    plt.hist(grayscaleImg.ravel(), bins = 255)
    plt.savefig(outfile)
    plt.close()

def aggregateIntensityImage(imgList):
    aggregate = np.zeros((imgList[0].shape))

    for img in imgList:
        for y in range(aggregate.shape[0]):
            for x in range(aggregate.shape[1]):
                aggregate[y][x] += img[y][x]

    for y in range(aggregate.shape[0]):
        for x in range(aggregate.shape[1]):
            if aggregate[y][x] > 1:
                aggregate[y][x] = 1

    return aggregate


def juxtaposeImages(imgArr):
    # all images must be same size
    baseImg = imgArr[0][0]
    heightImg = baseImg.shape[0]
    widthImg = baseImg.shape[1]

    rows = imgArr.shape[0]
    cols = imgArr.shape[1]

    outShape = (heightImg * rows, widthImg * cols)

    outImg = np.zeros(outShape)

    for i, row in enumerate(imgArr):
        for j, image in enumerate(row):
            startY = i * heightImg
            startX = j * widthImg

            inImg = imgArr[i][j]

            for y in range(heightImg):
                for x in range(widthImg):
                    outImg[startY + y][startX + x] = inImg[y][x]

    # adding boarders
    for i in range(1, rows):
        for z in range(7):
            outImg[i * heightImg + z] = 1
    for j in range(1, cols):
        for i in range(outShape[0]):
            for z in range(7):
                outImg[i][j * widthImg + z] = 1

    return outImg

def distance(p1, p2):
    dist = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
    return dist


def annotateRGBImageAtPoint(image, coord, color_rgb_arr):
    row_limit = image.shape[0]
    col_limit = image.shape[1]

    radius = 10

    for y in range(-radius, radius + 1):
        for x in range(-radius, radius + 1):
            if distance((0, 0), (y, x)) < radius:
                y_coord = coord[1] + y
                x_coord = coord[0] + x
                if 0 <= y_coord < row_limit and 0 <= x_coord < col_limit:
                    image[y_coord][x_coord] = color_rgb_arr


def convertGrayscale2RGB(grayscaleImage):
    rgbImg = np.zeros((grayscaleImage.shape[0], grayscaleImage.shape[1], 3), dtype = 'uint8')

    for y in range(grayscaleImage.shape[0]):
        for x in range(grayscaleImage.shape[1]):
            rgbImg[y][x][:] = int(grayscaleImage[y][x]*255)

    return rgbImg


def saveDifferenceTestingAggregationImage(relaxedImg, diffImgList, thresholdList, outfile, discludeVerificationArrayImg = False, centroid = None):

    diffImgList.insert(0, relaxedImg)

    binaryThresholdDiffs = [diffImgList]
    for t in thresholdList:
        binaryThresholdDiff = [getBinaryJelly(img, lower_bound=t) for img in diffImgList]
        binaryThresholdDiffs.append(binaryThresholdDiff)

    imgRows = np.array(binaryThresholdDiffs)

    if discludeVerificationArrayImg is False:
        compositeImage = juxtaposeImages(imgRows)

        rgbCompositeImage = convertGrayscale2RGB(compositeImage)

        del compositeImage

    baseImg = imgRows[0][0]
    heightImg = baseImg.shape[0]
    widthImg = baseImg.shape[1]

    rows = imgRows.shape[0]
    cols = imgRows.shape[1]

    if centroid is not None:
        zeroDegreePoint = (centroid[0], 15)
        angleData = np.zeros((rows-1,cols-1))
        for i in range(rows):
            for j in range(cols):
                y_coord = i * heightImg + centroid[1]          # centroid is (x, y)
                x_coord = j * widthImg + centroid[0]

                if discludeVerificationArrayImg is False:
                    annotateRGBImageAtPoint(rgbCompositeImage, (x_coord, y_coord), [0,0,255])

                if i != 0 and j != 0:
                    biggestRegion = findJellyRegion(imgRows[i][j])
                    local_com = None
                    if biggestRegion is not None:
                        local_com = findCentroid_regionProp(biggestRegion)
                        global_com = (j * widthImg + local_com[0], i * heightImg + local_com[1])

                        if discludeVerificationArrayImg is False:
                            annotateRGBImageAtPoint(rgbCompositeImage, global_com, [255, 0, 0])

                        y_coord = i * heightImg + zeroDegreePoint[1]  # centroid is (x, y)
                        x_coord = j * widthImg + zeroDegreePoint[0]

                        if discludeVerificationArrayImg is False:
                            annotateRGBImageAtPoint(rgbCompositeImage, (x_coord, y_coord), [0, 255, 0])

                    if local_com is not None:
                        angle = dm.getAngle(zeroDegreePoint, centroid, local_com)
                    else:
                        angle = np.nan
                    angleData[i-1][j-1] = angle
    else:
        angleData = None

    if discludeVerificationArrayImg is False:
        plt.imsave(str(outfile), rgbCompositeImage, cmap=plt.cm.gray)
        del rgbCompositeImage

    return angleData



def getCentroidVerificationImg(differenceImg, binaryDifferenceImg, centroid):
    compositeImage = juxtaposeImages(np.array([[differenceImg, binaryDifferenceImg]]))
    compositeImageRGB = convertGrayscale2RGB(compositeImage)
    annotateRGBImageAtPoint(compositeImageRGB, centroid, [0,0,255])
    annotateRGBImageAtPoint(compositeImageRGB, (centroid[0]+differenceImg.shape[1], centroid[1]), [0, 0, 255])
    return compositeImageRGB


def getAnnotatedVerificationImage(relaxedBinaryImage, diffImage, DEBUG = False):
    shape = relaxedBinaryImage.shape
    outimage = np.zeros([shape[0], shape[1], 3], dtype=np.uint8)

    if DEBUG: print(shape)
    if DEBUG: print(outimage.shape)

    for y in range(shape[0]):
        for x in range(shape[1]):
            if relaxedBinaryImage[y][x] == True:
                outimage[y][x] = [255, 255, 255]

    for y in range(shape[0]):
        for x in range(shape[1]):
            if diffImage[y][x] == True:
                outimage[y][x] = [255, 0, 0]

    return outimage

def saveJellyPlot(jelly_array, outfilePath, centroids=None, centroid_of_mass = None, justImage = True):
    plt.imshow(jelly_array, cmap=plt.cm.gray)
    if centroids is not None:
        for centroid in centroids:
            plt.plot(centroid[0], centroid[1], marker='o')
    if justImage is True:
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(str(outfilePath), bbox_inches='tight', pad_inches=0)
    else:
        plt.savefig(str(outfilePath))
    plt.close()

