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

import ImageMethods as im



def getBinaryAreas(filesSubset, lowerThreshold, DEBUG = False):
    binaryImageAreas = []
    for i in range(len(filesSubset)):
        if i % 100 == 0 and DEBUG: print("recieved areas up to pulse: {}".format(i))
        jellyimage = im.getJellyImageFromFile(str(filesSubset[i]))
        jellyimagebinary = im.getBinaryJelly(jellyimage, lowerThreshold)
        jellyBinaryArea = im.findBinaryArea(jellyimagebinary)
        binaryImageAreas.append(jellyBinaryArea)

    return binaryImageAreas


def saveAreasPlot(areas, peaks, outpath, diffsList, refractionaryPeriod = None):

    diffFrameLists = []
    for diff in diffsList:
        diffFramesBasedOnPeak = [x + diff for x in peaks]
        diffFrameLists.append(diffFramesBasedOnPeak)

    fig, ax1 = plt.subplots(figsize=(len(areas) / 10, 10))

    ax1.margins(x=0)
    ax1.set_xticks(np.arange(0, len(areas), 50))

    # looking at first area method based on binary image areas and properties
    ax1.plot(range(len(areas)), areas, color='k')
    for x in peaks:
        ax1.axvline(x, color='m')
    for diffFramesBasedOnPeak in diffFrameLists:
        for x in diffFramesBasedOnPeak:
            ax1.axvline(x, color='c')

    if refractionaryPeriod is not None:
        for x in peaks:
            ax1.axvspan(x, x + refractionaryPeriod, alpha=0.2, color='red')

    plt.savefig(str(outpath))
    plt.close()


def dynamicRangeImg(peaksOnBinaryImage, fileSubset, peak2InflectionDiff, peak2TroughDiff):
    dynamicRangeImagesMasked = []

    # getting dynamic range images for testing
    for i, peak in enumerate(peaksOnBinaryImage):
        troughInfile = fileSubset[peak + peak2TroughDiff]
        relaxedInfile = fileSubset[peak + peak2InflectionDiff]
        peakInfile = fileSubset[peak]

        troughImg = im.getJellyGrayImageFromFile(troughInfile)
        relaxedImg = im.getJellyGrayImageFromFile(relaxedInfile)
        peakImg = im.getJellyGrayImageFromFile(peakInfile)


        # misnomer
        peakDiff = im.getGrayscaleImageDiff_absolute(troughImg, peakImg)
        binaryPeakDiff = im.getBinaryJelly(peakDiff, lower_bound=0.05)

        dynamicRangeImg = im.dynamicRangeImg_AreaBased(relaxedImg, binaryPeakDiff, 5)

        dynamicRangeImagesMasked.append(dynamicRangeImg)

    return np.average(dynamicRangeImagesMasked, axis=0)



