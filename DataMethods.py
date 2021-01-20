from skimage import io, filters, color, measure

import os

import shutil

from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np

from scipy import ndimage

from math import cos, sin, radians

from scipy.signal import savgol_filter

import math

import pandas as pd

import ImageMethods as im

import subprocess as sp

import re

def makeOutDir(outputDir, folderName, DEBUG=False):
    outdir = outputDir / folderName
    if not (outdir.exists()):
        outdir.mkdir()
        if DEBUG: print('dir %s made' % outdir)
    else:
        if DEBUG: print('dir %s already exists' % outdir)
    return outdir

def getCSVFilePaths(videoImageDir):
    """
    method returns list of sorted, jpg files from inputted stack directory

    :param stackDirectory: directory of FFMPEG image stack
    :return: sorted list of frame file paths
    """

    files = [file for file in sorted(videoImageDir.iterdir()) if file.suffix == '.csv']

    return files



def getFrameFilePaths(videoImageDir):
    """
    method returns list of sorted, jpg files from inputted stack directory

    :param stackDirectory: directory of FFMPEG image stack
    :return: sorted list of frame file paths
    """

    imgPaths = [imgPath for imgPath in sorted(videoImageDir.iterdir()) if imgPath.suffix == '.jpg']

    return imgPaths

def getSubDirectoryFilePaths(imageStackDir):
    stackPaths = [stackDir for stackDir in sorted(imageStackDir.iterdir()) if stackDir.name != '.DS_Store']

    return stackPaths

def getFrameCountFromDir(videoImageDir):
    return sum(1 for entry in os.listdir(videoImageDir) if os.path.isfile(os.path.join(videoImageDir,entry)))

def getFrameCountFromDir_grep(videoImageDir):
    output = sp.check_output('ls -1 {} | wc -l'.format(videoImageDir), shell=True)
    stringOutput = output.decode("utf-8")
    numStacks = int(re.findall(r'\d+', stringOutput)[0])
    return numStacks

def readCSV2pandasDF(CSVpath):
    return pd.read_csv(str(CSVpath), index_col=0)

def replaceDir(outputDir):
    folder = str(outputDir)

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    return outputDir


def getFrameNumFromFile(file):
    """

    :param file: Pathlib object
    :return:
    """
    return int(str(file.name)[:str(file.name).find('.')])


def getFileFromFrameNum(frameNum, directory):
    file_name = f'{frameNum}'.zfill(14) + '.jpg'
    path = directory / file_name
    return path


def is_peak_side2side(index, areas):
    """
    Boolean returns True if index represents a peak in area, false otherwise
    :param index: index to test if peak
    :param areas: 1 dimensional list of areas
    :return: True if index is peak, False if not.
    """
    pot_peak = areas[index]  # potential peak

    for i in [-2, -1, 1, 2]:
        if pot_peak < areas[index + i]:
            return False

    return True

def is_peak_side2side_robust(index, areas):
    """
    Boolean returns True if index represents a peak in area, false otherwise
    :param index: index to test if peak
    :param areas: 1 dimensional list of areas
    :return: True if index is peak, False if not.
    """
    pot_peak = areas[index]  # potential peak

    for i in [-4, -3, -2, -1, 1, 2, 3, 4]:
        if pot_peak < areas[index + i]:
            return False

    return True

def is_downturn(index, areas, numConsecutiveDrops):
    """
    Boolean returns True if index represents a downturn in area, false otherwise
    Looks to see that the next areas are decreasing for 'numConsecutiveDrops' frames in a row
    :param index: index to test if peak
    :param areas: 1 dimensional list of areas
    :return: True if index is peak, False if not.
    """
    first = areas[index]
    for j in range(1, numConsecutiveDrops):
        if areas[index+j] < first:
            first = areas[index+j]
        else:
            return False
    return True


def is_trough(index, areas):
    """
    Boolean returns True if index represents a trough in area, false otherwise
    :param index: index to test if trough
    :param areas: 1 dimensional list of areas
    :return: True if index is trough, False if not.
    """
    pot_trough = areas[index]  # potential trough

    for i in [-2, -1, 1, 2]:
        if pot_trough > areas[index + i]:
            return False

    return True

def getPeaks_side2side(areas, frameNums=None):
    """
    Gets all peak frame nums
    :param areas: list of jelly region areas (can try thresholded True areas at some point)
    :param frameNums: list of corresponding frame nums to the areas.
    :return: list of peak frame nums
    """
    if frameNums is None: frameNums = range(len(areas))

    peakFrameNums = []

    for i in range(2, len(areas) - 2):
        if is_peak_side2side(i, areas):
            peakFrameNums.append(frameNums[i])

    return peakFrameNums

def getPeaks_side2side_robust(areas, frameNums=None):
    """
    Gets all peak frame nums
    :param areas: list of jelly region areas (can try thresholded True areas at some point)
    :param frameNums: list of corresponding frame nums to the areas.
    :return: list of peak frame nums
    """
    if frameNums is None: frameNums = range(len(areas))

    peakFrameNums = []

    for i in range(4, len(areas) - 4):
        if is_peak_side2side_robust(i, areas):
            peakFrameNums.append(frameNums[i])

    return peakFrameNums

def getDownturns(areas, numConsecutiveDownturns, frameNums=None):
    """
    Gets all downturn frame nums
    :param areas: list of jelly region areas (can try thresholded True areas at some point)
    :param frameNums: list of corresponding frame nums to the areas.
    :return: list of downturn frame nums
    """
    if frameNums is None: frameNums = range(len(areas))


    downturnFrameNums = []
    for i in range(len(areas) - numConsecutiveDownturns):
        if is_downturn(i, areas, numConsecutiveDownturns):
            downturnFrameNums.append(frameNums[i])


def getTroughs(areas, frameNums=None):
    """
    Gets all trough frame nums
    :param areas: list of jelly region areas (can try thresholded True areas at some point)
    :param frameNums: list of corresponding frame nums to the areas.
    :return: list of trough frame nums
    """
    if frameNums is None: frameNums = range(len(areas))

    troughFrameNums = []

    for i in range(2, len(areas) - 2):
        if is_trough(i, areas):
            troughFrameNums.append(frameNums[i])

    return troughFrameNums


def finitediffSecondOrder(index, nums):
    """
    :param index: index to get finite second derivative
    :param nums: list of numbers on which to find second derivative
    :return: finite second derivative at given index
    """
    a = nums[index-1]
    b = nums[index]
    c = nums[index+1]
    return a - 2*b + c


def getLikelyInflectionDiff(areas, peaks):

    secderivIndicies = np.arange(1, len(areas) - 1)
    secderiv = []
    for x in secderivIndicies:
        secderiv.append(finitediffSecondOrder(x, areas))
    smoothedSecondMoment = savgol_filter(secderiv, 35, 3)

    # finds potential inflection points
    pot_inflection = getPeaks_side2side(smoothedSecondMoment, secderivIndicies)

    # sorts through potential inflection points for likely inflection points
    inflectionDiff2Peak = []
    for peak in peaks:
        closest = pot_inflection[0]
        for i in pot_inflection:
            if 0 < (peak - i) < (peak - closest):
                closest = i
        inflectionDiff2Peak.append(closest - peak)

    # takes average of likely inflection points and peaks
    prePeakInflectionDiff = int(np.mean(inflectionDiff2Peak))

    return prePeakInflectionDiff


def likelyPostPeakTroughDiff(troughs, peaks):

    trough2peakDiffs = []

    for peak in peaks:
        closest = troughs[-1]
        for trough in troughs:
            if trough > peak and abs(trough - peak) < abs(closest - peak):
                closest = trough
        trough2peakDiffs.append(closest - peak)

    postPeakTroughDiff = int(np.mean(trough2peakDiffs))

    return int(postPeakTroughDiff)


def averageInterpulseInterval(peakFrameNums):
    intervals = []

    for i in range(len(peakFrameNums)-1):
        intervals.append(peakFrameNums[i+1]-peakFrameNums[i])

    return np.mean(intervals)


def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    return ang + 360 if ang < 0 else ang


def angleDifferenceCalc(a, b):
    if math.isnan(a) or math.isnan(b):
        return 360
    phi = abs(b - a) % 360;      # This is either the distance or 360 - distance
    if phi > 180:
        phi = 360 - phi
    return phi


def reassignIntVariable(currentVar, varTitle):
    n = None
    while type(n) is not int:
        try:
            n = input("Replace {} with (current value is {}): ".format(varTitle, currentVar))
            n = int(n)
            return n
        except ValueError:
            print("%s is not an integer.\n" % n)

def reassignFloatVariable(currentVar, varTitle):
    n = None
    while type(n) is not float:
        try:
            n = input("Replace {} with (current value is {}): ".format(varTitle, currentVar))
            n = float(n)
            return n
        except ValueError:
            print("%s is not an float.\n" % n)


def getSelection(selectionChoices):
    while True:
        n = input("Please enter one of the following selections: {} - ".format(selectionChoices))
        if n not in [str(x) for x in selectionChoices]:
            print("{} is not in {}.".format(n, selectionChoices))
            n = getSelection(selectionChoices)
        return n

def inputAngle(pulse):
    n = None
    while type(n) is not int:
        try:
            n = input('angle for peak {}: '.format(pulse))
            if n == 'nan':
                return np.nan
            n = int(n)
            if n < 0 or n >= 360:
                print('enter within bounds [0, 360)')
                n = inputAngle(pulse)
            return n
        except ValueError:
            print("%s is not an integer.\n" % n)

def chime(MAC, text = None):
    if text is None: text = 'task complete'

    if MAC:
        os.system('say "{}"'.format(text))
    else:
        os.system('mshta vbscript:Execute("CreateObject(""SAPI.SpVoice"").Speak(""{}"")(window.close)")'.format(text))
