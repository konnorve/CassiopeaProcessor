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

import VerificationMethods as vm

import datetime

########################################################################################################################
# **** GLOBAL VARIABLES ****

DEBUG = True
CHIME = True

discludeVerificationArrayImg = False

CONFIRMATIONIMAGES = False
confirmationImagesPath = Path('/Users/kve/Desktop/Clubs/Harland_Lab/Round_8/thresholdingAndAutomationTesting_July30/ScrupulousData')


########################################################################################################################

def runFFMPEG(videoPath, stackDir=None, framerate=120, starttime='00:00:00', duration=None, quality=0):
    """
    This method runs FFMPEG on the video at the path's end, and creates a folder with the frames nested within in the
    proper format for processing. The stackDir is where the user wishes to export the stack.

    :param videoPath: input path where video is located
    :param stackDir: OPTIONAL, default is the source directory. Change if desired stack placement is somewhere else.
    :param framerate: OPTIONAL, default is 120
    :param starttime: OPTIONAL, default is 0 hr, 0 min, 0 sec. Input in HH:MM:SS format for later times.
    :param duration: OPTIONAL, default is full video. Input in seconds.
    :param quality: OPTIONAL, dault is 0 AKA loss-less
    :return: path object where stack is located
    """
    pass


def getFrameFilePaths(videoImageDir=None):
    """
    method returns list of sorted, jpg files from inputted stack directory

    :param stackDirectory: directory of FFMPEG image stack
    :return: sorted list of frame file paths
    """

    if videoImageDir is None: videoImageDir = videoImageStackDir

    files = [file for file in sorted(videoImageDir.iterdir()) if file.suffix == '.jpg']
    return files


def saveVariableParams():
    imp_parameters = pd.DataFrame(np.array([
        [chunkName, 'the name of this jelly and its video chunk id'],
        [recordingName, 'the name of the recording being processed'],
        [videoImageStackDir, 'where the image stack for this specific chunk is located'],
        [verificationOutputDir, 'place where all the segment verification directeries are stored'],
        [angleOutputDir, 'place where all the data is stored'],
        [pathOfCurrentParamDF, 'path of the latest param dataframe for the latest segment'],
        [framesInChunk, 'absolute number of frames in specific video chunk'],
        [lastFrameOfPreviousChunk, 'the last frame the occurred before this recording chunk'],
        [lastFrameOfThisChunk, 'the last frame that occurs in this recording chunk'],
        [isMoving, 'if the Jellyfish was moving at the end of the last video'],
        [lastStationaryCentroid, 'used in analysis to detect movement. Passed in return statement and as a parameter.'],

        [globalmovementSegment, 'the latest segment number across all video chunks (segment = non-moving period)'],
        [averageTroughBinaryArea, 'lower threshold to create binary image of jelly to assess area (for downturns)'],
        [lowerThreshold, 'lower threshold to create binary image of jelly to assess area (for downturns)'],
        [peak2InflectionDiff,
         'the number of frames past the peak where the inflection point occurs (this should always be negative)'],
        [peak2TroughDiff, 'the number of frames past the peak where the lowest area is found on average'],
        [postPeakRefractoryPeriod, 'the number of frames to preclude from analysis'],
        [inflectionTestDiff, 'the number of frames after inflection point where the difference in calculated'],
        [inflectionTestBinaryThreshold, 'the ideal threshold to locate the area of difference'],
        [chosenSD, 'the sd of the chosen test diff and threshold when they were initialized'],
        [numConsecutiveDrops, 'the number of consecutive drops needed to count something as a downturn'],

        [initialRefracPeriod, 'initial refractionary period used in'],
        [movementThreshold4reinitialization, 'number of pixels from one centroid to another to consider a jelly as moving.'],
        [movementThreshold2KeepMoving, 'number of pixels from one centroid to the next to continue to be considered moving'],
        [movementThreshold4newNormalizationImg, 'number of pixels from one centroid to another to reinitialize the average normalization img'],
        [pct2skip4RefractionaryPeriod,
         'percent on average Interpulse Interval to skip when initializing postPeakRefractoryPeriod'],
        [numFramesForOrientation, 'number of frames to save in the Orientation folder at the start of a new segment'],
        [numFramesForParamInitialization, 'number of frames to use when initializing params for a new segment'],
        [numFrames2ConfirmStationary,
         'number of frames after first stationary frame after movement to confirm jellyfish is stationary'],
    ]),

        index=['chunkName',
               'recordingName',
               'videoImageStackDir',
               'verificationOutputDir',
               'angleOutputDir',
               'pathOfCurrentParamDF',
               'framesInChunk',
               'lastFrameOfPreviousChunk',
               'lastFrameOfThisChunk',
               'isMoving',
               'lastStationaryCentroid',

               'globalmovementSegment',
               'averageTroughBinaryArea',
               'lowerThreshold',
               'peak2InflectionDiff',
               'peak2TroughDiff',
               'postPeakRefractoryPeriod',
               'inflectionTestDiff',
               'inflectionTestBinaryThreshold',
               'chosenSD',
               'numConsecutiveDrops',

               'initialRefracPeriod',
               'movementThreshold4reinitialization',
               'movementThreshold2KeepMoving',
               'movementThreshold4newNormalizationImg',
               'pct2skip4RefractionaryPeriod',
               'numFramesForOrientation',
               'numFramesForParamInitialization',
               'numFrames2ConfirmStationary'],

        columns=['data', 'notes'])

    imp_parameters.to_csv(str(pathOfCurrentParamDF))


def initialize_params(files, startingFrameNum):
    fileSubset = files[startingFrameNum: startingFrameNum + numFramesForParamInitialization]

    # all of these must be read in from the previous param segment
    global globalmovementSegment  # the latest segment number across all video chunks (segment = non-moving period)
    global averageTroughBinaryArea  # lower threshold to create binary image of jelly to assess area (for downturns)
    global lowerThreshold  # upper threshold to create binary image of jelly to assess area (for downturns)
    global peak2InflectionDiff  # the number of frames past the peak where the inflection point occurs (this should always be negative)
    global peak2TroughDiff  # the number of frames past the peak where the lowest area is found on average
    global postPeakRefractoryPeriod  # the number of frames to preclude from analysis
    global inflectionTestDiff  # the number of frames after inflection point where the difference in calculated
    global inflectionTestBinaryThreshold  # the ideal threshold to locate the area of difference
    global chosenSD  # the sd of the chosen test diff and threshold when they were initialized
    global numConsecutiveDrops  # the number of consecutive drops needed to count something as a downturn

    global pathOfCurrentParamDF

    # create segment directory named with global movement segment
    segmentName = '{}_{}'.format(recordingName, globalmovementSegment)
    segmentVerificationDir = dm.makeOutDir(verificationOutputDir, segmentName)

    param_dir = dm.makeOutDir(segmentVerificationDir, '{}_ParamFiles'.format(segmentName))
    pathOfCurrentParamDF = param_dir / '{}_{}.csv'.format(chunkName, globalmovementSegment)

    saveVariableParams()

    thresholdingDir = dm.makeOutDir(segmentVerificationDir, '{}_ThresholdingPlots'.format(segmentName))
    plotDir = dm.makeOutDir(segmentVerificationDir, '{}_AngleVerificationPlots'.format(segmentName))
    centroidDir = dm.makeOutDir(segmentVerificationDir, '{}_CentroidVerificationPlots'.format(segmentName))
    orientationDir = dm.makeOutDir(segmentVerificationDir, '{}_RelaxedFramesForOrientation'.format(segmentName))
    dynamicRangeDir = dm.makeOutDir(segmentVerificationDir, '{}_dynamicRangeNormalizationImages'.format(segmentName))

    # TODO: input by eye measurements into a mutable csv to load in

    # TODO: move more of these into verification methods


    if averageTroughBinaryArea is None:
        binaryImageAreas4thresholding = vm.getBinaryAreas(fileSubset, 0.1, DEBUG)
        peaksOnBinaryImage4thresholding = downturnFinder(fileSubset, lowerThresh=0.1)
        troughsOnBinaryImage4thresholding = dm.getTroughs(binaryImageAreas4thresholding)
        peak2TroughDiff4thresholding = dm.likelyPostPeakTroughDiff(troughsOnBinaryImage4thresholding,
                                                                   peaksOnBinaryImage4thresholding)
        peak2InflectionDiff4thresholding = dm.getLikelyInflectionDiff(binaryImageAreas4thresholding,
                                                                      peaksOnBinaryImage4thresholding)

        maxIntensities = []
        minIntensities = []
        for i, peak in enumerate(peaksOnBinaryImage4thresholding):
            if peak + peak2InflectionDiff4thresholding >= 0 and peak + peak2TroughDiff4thresholding < numFramesForParamInitialization:
                troughInfile = fileSubset[peak + peak2TroughDiff4thresholding]
                relaxedInfile = fileSubset[peak + peak2InflectionDiff4thresholding]
                troughImg = im.getJellyGrayImageFromFile(troughInfile)
                relaxedImg = im.getJellyGrayImageFromFile(relaxedInfile)

                centroidDiff = im.getGrayscaleImageDiff_absolute(troughImg, relaxedImg)
                binaryCentroidDiff = im.getBinaryJelly(centroidDiff, lower_bound=0.05)
                maskedImg = im.applyMask2Img(binaryCentroidDiff, relaxedImg)
                jellyRegion = im.findJellyRegionWithGray(binaryCentroidDiff, maskedImg)
                maxIntensity = jellyRegion.max_intensity
                minIntensity = jellyRegion.min_intensity
                maxIntensities.append(maxIntensity)
                minIntensities.append(minIntensity)

        indensityDifference = np.mean(maxIntensities) - np.mean(minIntensities)

        lowerThreshold = np.mean(minIntensities) + (0.1 * indensityDifference)

        if DEBUG: print(
            'finished initializing lower threshold as {}'.format(lowerThreshold))

        if CHIME: dm.chime(MAC, 'input time')
        while True:

            troughAreas = []

            for peak in peaksOnBinaryImage4thresholding:
                peakInfile = fileSubset[peak]
                troughInfile = fileSubset[peak+peak2TroughDiff4thresholding]

                peakImg = im.getJellyGrayImageFromFile(peakInfile)
                troughImg = im.getJellyGrayImageFromFile(troughInfile)

                binaryPeakImg = im.getBinaryJelly(peakImg, lowerThreshold)
                binaryTroughImg = im.getBinaryJelly(troughImg, lowerThreshold)

                im.saveJellyPlot(im.juxtaposeImages(np.array([[binaryPeakImg, binaryTroughImg]])),
                                 (thresholdingDir / '{}_thresholdVerification_{}.png'.format(chunkName, peak)))

                jellyTroughBinaryArea = im.findBinaryArea(binaryTroughImg)

                troughAreas.append(jellyTroughBinaryArea)

            print('average trough area: {}, sd of trough areas: {}'.format(np.mean(troughAreas), np.std(troughAreas)))

            print('Change thresholds: ')
            print('select \'1\' to change {} which is {}'.format('lowerThreshold', lowerThreshold))
            print('select \'2\' to remove a peak from peaksOnBinaryImage4thresholding')
            print('or \'3\' to continue.')

            selectionVar = dm.getSelection([1, 2, 3])
            if selectionVar == '1':
                lowerThreshold = dm.reassignFloatVariable(lowerThreshold, 'lowerThreshold')
                dm.replaceDir(thresholdingDir)
            elif selectionVar == '2':
                print('peaksOnBinaryImage: {}'.format(peaksOnBinaryImage4thresholding))
                index2Pop = int(dm.getSelection(list(range(len(peaksOnBinaryImage4thresholding)))))
                peaksOnBinaryImage4thresholding.pop(index2Pop)
            else:
                averageTroughBinaryArea = np.mean(troughAreas)
                break

    # completed automated based on averageTroughBinaryArea
    if lowerThreshold is None:
        thresholdStep = 0.005
        chosenThreshold = 0.05

        peaksOnBinaryImage4thresholding = downturnFinder(fileSubset, lowerThresh=0.1)

        testTroughAverage = averageTroughBinaryArea + 1

        while testTroughAverage > averageTroughBinaryArea:
            chosenThreshold += thresholdStep

            troughAreas = []
            for i, peak in enumerate(peaksOnBinaryImage4thresholding):
                if peak + peak2TroughDiff < numFramesForParamInitialization:

                    peakInfile = fileSubset[peak]
                    troughInfile = fileSubset[peak + peak2TroughDiff]

                    peakImg = im.getJellyGrayImageFromFile(peakInfile)
                    troughImg = im.getJellyGrayImageFromFile(troughInfile)

                    binaryPeakImg = im.getBinaryJelly(peakImg, chosenThreshold)
                    binaryTroughImg = im.getBinaryJelly(troughImg, chosenThreshold)

                    im.saveJellyPlot(im.juxtaposeImages(np.array([[binaryPeakImg, binaryTroughImg]])),
                                     (thresholdingDir / '{}_thresholdVerification_{}.png'.format(chunkName, peak)))

                    jellyTroughBinaryArea = im.findBinaryArea(binaryTroughImg)

                    troughAreas.append(jellyTroughBinaryArea)

            testTroughAverage = np.mean(troughAreas)

            print('chosenThreshold: {} (test area, {}; target area, {})'.format(chosenThreshold, testTroughAverage,
                                                                                averageTroughBinaryArea))

        lowerThreshold = chosenThreshold

    saveVariableParams()

    if peak2InflectionDiff == 0 or peak2TroughDiff == 0 or postPeakRefractoryPeriod is None:
        # get areas of jellies both the region and the whole value true in binary image.
        binaryImageAreas = vm.getBinaryAreas(fileSubset, lowerThreshold, DEBUG)
        if DEBUG: print('finished getting test areas')

        # gets peak frame nums from binaryImageAreas
        peaksOnBinaryImage = downturnFinder(fileSubset)
        if DEBUG: print('finished getting downturns')

        # gets peak2TroughDiff from peaksOnBinaryImage and binaryImageAreas
        troughsOnBinaryImage = dm.getTroughs(binaryImageAreas)
        peak2TroughDiff = dm.likelyPostPeakTroughDiff(troughsOnBinaryImage, peaksOnBinaryImage)
        if DEBUG: print('finished initializing peak2TroughDiff as: {}'.format(peak2TroughDiff))

        # initializes inflection diff from jellyRegionAreas
        peak2InflectionDiff = dm.getLikelyInflectionDiff(binaryImageAreas, peaksOnBinaryImage)
        if DEBUG: print('finished initializing peak2InflectionDiff as: {}'.format(peak2InflectionDiff))

        # initializes frames to skip for analysis after each peak
        averageIPI = dm.averageInterpulseInterval(peaksOnBinaryImage)
        postPeakRefractoryPeriod = int(pct2skip4RefractionaryPeriod * averageIPI)
        if DEBUG: print('finished initializing postPeakRefractoryPeriod as: {}'.format(postPeakRefractoryPeriod))

        if CHIME: dm.chime(MAC, 'input time')
        # verification of areas plot to confirm chosen values
        while True:

            plotTitle = '{0}_areaVerificationPlot.png'.format(segmentName)
            plotOutpath = segmentVerificationDir / plotTitle
            vm.saveAreasPlot(binaryImageAreas, peaksOnBinaryImage, plotOutpath,
                             [peak2InflectionDiff, peak2InflectionDiff + 5, peak2TroughDiff],
                             postPeakRefractoryPeriod)
            if DEBUG: print('finished saving figure at: {}'.format(plotOutpath))

            print('Params to change: ')
            print('select \'1\' to change {} which is {}'.format('postPeakRefractoryPeriod', postPeakRefractoryPeriod))
            print('select \'2\' to change {} which is {}'.format('numConsecutiveDrops', numConsecutiveDrops))
            print('select \'3\' to change {} which is {}'.format('peak2InflectionDiff', peak2InflectionDiff))
            print('select \'4\' to change {} which is {}'.format('peak2TroughDiff', peak2TroughDiff))
            print('or \'5\' to continue.')

            selectionVar = dm.getSelection([1, 2, 3, 4, 5])

            if selectionVar == '1':
                postPeakRefractoryPeriod = dm.reassignIntVariable(postPeakRefractoryPeriod, 'postPeakRefractoryPeriod')
            elif selectionVar == '2':
                numConsecutiveDrops = dm.reassignIntVariable(numConsecutiveDrops, 'numConsecutiveDrops')

                peaksOnBinaryImage = downturnFinder(fileSubset)
                troughsOnBinaryImage = dm.getTroughs(binaryImageAreas)
                peak2TroughDiff = dm.likelyPostPeakTroughDiff(troughsOnBinaryImage, peaksOnBinaryImage)
                peak2InflectionDiff = dm.getLikelyInflectionDiff(binaryImageAreas, peaksOnBinaryImage)
                postPeakRefractoryPeriod = int(
                    pct2skip4RefractionaryPeriod * dm.averageInterpulseInterval(peaksOnBinaryImage))
            elif selectionVar == '3':
                peak2InflectionDiff = dm.reassignIntVariable(peak2InflectionDiff, 'peak2InflectionDiff')
            elif selectionVar == '4':
                peak2TroughDiff = dm.reassignIntVariable(peak2TroughDiff, 'peak2TroughDiff')
            else:
                break
    else:
        # get areas of jellies both the region and the whole value true in binary image.
        binaryImageAreas = vm.getBinaryAreas(fileSubset, lowerThreshold, DEBUG)
        if DEBUG: print('finished getting test areas')

        # gets peak frame nums from binaryImageAreas
        peaksOnBinaryImage = downturnFinder(fileSubset)
        if DEBUG: print('finished getting downturns')

        plotTitle = '{0}_areaVerificationPlot.png'.format(segmentName)
        plotOutpath = segmentVerificationDir / plotTitle
        vm.saveAreasPlot(binaryImageAreas, peaksOnBinaryImage, plotOutpath,
                         [peak2InflectionDiff, peak2InflectionDiff + inflectionTestDiff, peak2TroughDiff],
                         postPeakRefractoryPeriod)
        if DEBUG: print('finished saving figure at: {}'.format(plotOutpath))

    saveVariableParams()

    i = 0
    while i < len(peaksOnBinaryImage):
        if peaksOnBinaryImage[i] + peak2InflectionDiff < 0 or peaksOnBinaryImage[i] + peak2TroughDiff >= numFramesForParamInitialization:
            peaksOnBinaryImage.pop(i)
        else:
            i += 1

    if DEBUG: print('peaksOnBinaryImg: {}'.format(peaksOnBinaryImage))


    if inflectionTestBinaryThreshold is None or inflectionTestDiff is None or chosenSD is None:

        # make directory to store verification jelly plots
        postInflectionDiffCases = list(range(2, 9))
        thresholdCases = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

        # out data: pulse angles x testDiffs
        angleData = np.empty((len(peaksOnBinaryImage), len(postInflectionDiffCases), len(thresholdCases)))  # 3D array
        angleData[:] = np.nan

        otherDataCols = np.array(['relaxed', 'peak', 'trough', 'by eye', ''])
        otherData = np.empty((len(peaksOnBinaryImage), len(otherDataCols)))
        otherData[:] = np.nan

        for i, peak in enumerate(peaksOnBinaryImage):
            troughInfile = fileSubset[peak + peak2TroughDiff]
            relaxedInfile = fileSubset[peak + peak2InflectionDiff]
            troughImg = im.getJellyGrayImageFromFile(troughInfile)
            relaxedImg = im.getJellyGrayImageFromFile(relaxedInfile)

            centroidDiff = im.getGrayscaleImageDiff_absolute(troughImg, relaxedImg)
            binaryCentroidDiff = im.getBinaryJelly(centroidDiff, lower_bound=0.05)
            centroidRegion = im.findJellyRegion(binaryCentroidDiff)
            centroid = im.findCentroid_boundingBox(centroidRegion)

            centroidVerOutFile = centroidDir / 'centroid for {} - {}.png'.format(segmentName, peak + peak2InflectionDiff)
            im.saveJellyPlot(im.getCentroidVerificationImg(centroidDiff, binaryCentroidDiff, centroid), centroidVerOutFile)

            orientationOutFile = orientationDir / 'relaxedFrame_{}.png'.format(peak + peak2InflectionDiff)
            im.saveJellyPlot(relaxedImg, orientationOutFile, [centroid, (centroid[0], 15)])

            peakInfile = fileSubset[peak]
            peakImg = im.getJellyGrayImageFromFile(peakInfile)
            peakDiff = im.getGrayscaleImageDiff_absolute(troughImg, peakImg)
            binaryPeakDiff = im.getBinaryJelly(peakDiff, lower_bound=0.05)
            averagedDynamicRangeMaskedImg = im.dynamicRangeImg_AreaBased(relaxedImg, binaryPeakDiff, 5)

            dynamicRangeImgOutfile = dynamicRangeDir / 'dynamicRangeImg_{}.png'.format(peak + peak2InflectionDiff)

            im.saveJellyPlot(averagedDynamicRangeMaskedImg, dynamicRangeImgOutfile)

            # dealing with inflection thresholding
            testDiffImages = []
            for j in postInflectionDiffCases:
                testInfile = fileSubset[peak + peak2InflectionDiff + j]
                testImg = im.getJellyGrayImageFromFile(testInfile)
                testDiff = im.getGrayscaleImageDiff_absolute(testImg, relaxedImg)
                normalizedTestDiff = testDiff / averagedDynamicRangeMaskedImg
                testDiffImages.append(normalizedTestDiff)

            testingOutfile = plotDir / 'testPlot for {} - {}.png'.format(segmentName, peak + peak2InflectionDiff)
            pulseAngleData = im.saveDifferenceTestingAggregationImage(relaxedImg, testDiffImages, thresholdCases,
                                                                      testingOutfile, discludeVerificationArrayImg, centroid)

            if DEBUG: print("{}: {}".format(peak, pulseAngleData.ravel()))

            for n, row in enumerate(pulseAngleData):
                for m, angle in enumerate(row):
                    angleData[i][m][n] = angle

            otherData[i][0] = peak + peak2InflectionDiff + startingFrameNum
            otherData[i][1] = peak + startingFrameNum
            otherData[i][2] = peak + peak2TroughDiff + startingFrameNum

        angleDataAsRows = [x.ravel() for x in angleData]
        pulseAngleOutput = np.concatenate((np.tile([postInflectionDiffCases], len(thresholdCases)), angleDataAsRows))
        otherDataOut = np.concatenate(([otherDataCols], otherData))

        # warning: this results in mixed data. This cannot be saved by numpy csv methods. Pandas is easiest way to save.
        outframe = np.concatenate((otherDataOut, pulseAngleOutput), axis=1)

        # saves data into verification frame
        dfOut = pd.DataFrame(outframe)
        dataTitle = '{}_testDifferenceVerification.csv'.format(segmentName)
        verificationCSVOutFile = segmentVerificationDir / dataTitle
        dfOut.to_csv(str(verificationCSVOutFile), header=False, index=False)

        # setting test difference and threshold

        # read in by eye angle measurements
        byEyeAngles = []

        if CHIME: dm.chime(MAC, 'input time')

        print('time to enter by eye angles for each pulse')
        print('entries must be from 0 to 360')
        for pulse in peaksOnBinaryImage:
            byEyeAngle = dm.inputAngle(pulse + startingFrameNum)
            byEyeAngles.append(byEyeAngle)

        i = 0
        while i < len(byEyeAngles):
            if byEyeAngles[i] == np.nan:
                np.delete(angleData, i, 0)
                byEyeAngles.pop(i)
            else:
                i += 1

        angleDataShape = angleData.shape
        if DEBUG: print(angleData.shape)
        if DEBUG: print(len(byEyeAngles))

        diff2byeye = np.empty((angleDataShape[2], angleDataShape[1], angleDataShape[0]))
        diff2byeye[:] = np.nan
        for i in range(angleDataShape[0]):
            for j in range(angleDataShape[1]):
                for k in range(angleDataShape[2]):
                    diff2byeye[k][j][i] = dm.angleDifferenceCalc(angleData[i][j][k], byEyeAngles[i])

        squaredDiffs = np.square(diff2byeye)
        summedDiffs = np.sum(squaredDiffs, axis=2)
        varianceTable = summedDiffs / diff2byeye.shape[2]
        sdTable = np.sqrt(varianceTable)

        sdTableMinIndex = list(
            [np.where(sdTable == np.nanmin(sdTable))[0][0], np.where(sdTable == np.nanmin(sdTable))[1][0]])

        lowSDthresholds = []
        lowSDtestFrames = []

        for x in np.sort(sdTable.ravel())[0:5]:
            loc = np.where(sdTable == x)
            lowSDthresholds.append(loc[0][0])
            lowSDtestFrames.append(loc[1][0])

        inflectionTestBinaryThreshold = thresholdCases[int(np.median(lowSDthresholds))]
        inflectionTestDiff = postInflectionDiffCases[int(np.median(lowSDtestFrames))]

        if CHIME: dm.chime(MAC, 'input time')
        while True:
            print('thresholding options: {}'.format(thresholdCases))
            print('test frame options: {}'.format(postInflectionDiffCases))
            print(sdTable)
            print('index of min sd: {}'.format(sdTableMinIndex))
            print('selected sd: {}'.format(sdTable[thresholdCases.index(inflectionTestBinaryThreshold)][
                                               postInflectionDiffCases.index(inflectionTestDiff)]))

            print('Params to change: ')
            print('select \'1\' to change {} which is {}'.format('inflectionTestBinaryThreshold',
                                                                 inflectionTestBinaryThreshold))
            print('select \'2\' to change {} which is {}'.format('inflectionTestDiff', inflectionTestDiff))
            print('or \'3\' to continue.')

            selectionVar = dm.getSelection([1, 2, 3])
            if selectionVar == '1':
                inflectionTestBinaryThreshold = float(dm.getSelection(thresholdCases))
            elif selectionVar == '2':
                inflectionTestDiff = int(dm.getSelection(postInflectionDiffCases))
            else:
                break

        chosenSD = sdTable[thresholdCases.index(inflectionTestBinaryThreshold)][
            postInflectionDiffCases.index(inflectionTestDiff)]
    else:
        for i, peak in enumerate(peaksOnBinaryImage):

            relaxedInfile = fileSubset[peak + peak2InflectionDiff]
            testInfile = fileSubset[peak + peak2InflectionDiff + inflectionTestDiff]
            peakInfile = fileSubset[peak]
            troughInfile = fileSubset[peak + peak2TroughDiff]

            relaxedImg = im.getJellyGrayImageFromFile(relaxedInfile)
            testImg = im.getJellyGrayImageFromFile(testInfile)
            peakImg = im.getJellyGrayImageFromFile(peakInfile)
            troughImg = im.getJellyGrayImageFromFile(troughInfile)

            centroidDiff = im.getGrayscaleImageDiff_absolute(troughImg, relaxedImg)
            binaryCentroidDiff = im.getBinaryJelly(centroidDiff, lower_bound=0.05)
            centroidRegion = im.findJellyRegion(binaryCentroidDiff)
            centroid = im.findCentroid_boundingBox(centroidRegion)

            centroidVerOutFile = centroidDir / 'centroid for {} - {}.png'.format(segmentName, peak + peak2InflectionDiff)
            im.saveJellyPlot(im.getCentroidVerificationImg(centroidDiff, binaryCentroidDiff, centroid), centroidVerOutFile)

            orientationOutFile = orientationDir / 'relaxedFrame_{}.png'.format(peak + peak2InflectionDiff)
            im.saveJellyPlot(relaxedImg, orientationOutFile, [centroid, (centroid[0], 15)])

            peakDiff = im.getGrayscaleImageDiff_absolute(troughImg, peakImg)
            binaryPeakDiff = im.getBinaryJelly(peakDiff, lower_bound=0.05, upper_bound=1)
            averagedDynamicRangeMaskedImg = im.dynamicRangeImg_AreaBased(relaxedImg, binaryPeakDiff, 5)

            dynamicRangeImgOutfile = dynamicRangeDir / 'dynamicRangeImg_{}.png'.format(peak + peak2InflectionDiff)
            im.saveJellyPlot(averagedDynamicRangeMaskedImg, dynamicRangeImgOutfile)

            testDiff = im.getGrayscaleImageDiff_absolute(testImg, relaxedImg)
            normalizedTestDiff = testDiff / averagedDynamicRangeMaskedImg

            binaryDiffImg = im.getBinaryJelly(normalizedTestDiff, lower_bound=inflectionTestBinaryThreshold)

            biggestRegion = im.findJellyRegion(binaryDiffImg)

            if biggestRegion is not None:
                local_com = im.findCentroid_regionProp(biggestRegion)
                zeroDegreePoint = (centroid[0], 0)

            testingOutfile = plotDir / 'testPlot for {} - {}.png'.format(segmentName, peak + peak2InflectionDiff)
            im.saveJellyPlot(binaryDiffImg, testingOutfile, [centroid, zeroDegreePoint, local_com])

    # saves important parameters used in analysis to csv
    saveVariableParams()

    if DEBUG: print('saved parameter data')

    if DEBUG: print('finished saving outplots and angle verification at: {}'.format(segmentVerificationDir))

    return True


def downturnFinder(files, refactoryPeriod=None, lowerThresh=None, DEBUG = False):
    if refactoryPeriod is None:
        if postPeakRefractoryPeriod is None:
            refactoryPeriod = initialRefracPeriod
        else:
            refactoryPeriod = postPeakRefractoryPeriod

    if lowerThresh is None:
        lowerThresh = lowerThreshold

    i = 0
    numFiles = len(files)

    if DEBUG:
        print('lt: {}, pprp: {}, numConsecutiveDrops: {}, numfiles: {}'.format(
            lowerThresh, refactoryPeriod, numConsecutiveDrops, numFiles
        ))

    peakIndicies = []

    # initializes lists with 'numConsecutiveDrops' of files
    def reinitializeTestFramesAndAreas(j):
        testFrames = []  # this list should never be more than 5 entries long, ex. [51, 52, 53, 54, 55]
        testAreas = []  # this list should never be more than 5 entries long, ex. [253, 255, 256, 255, 255]

        while len(testFrames) < numConsecutiveDrops and j < numFiles:
            image = im.getJellyImageFromFile(files[j])
            binary_image = im.getBinaryJelly(image, lowerThresh)
            area = im.findBinaryArea(binary_image)

            testFrames.append(j)
            testAreas.append(area)
            j += 1

        return testFrames, testAreas, j

    testFrames, testAreas, i = reinitializeTestFramesAndAreas(i)

    while i < numFiles:
        isDownturn = dm.is_downturn(0, testAreas, numConsecutiveDrops)

        if DEBUG: print('i: {}, isDownturn: {}, testAreas: {}, testFrames: {}'.format(i, isDownturn, testAreas, testFrames))

        if isDownturn:
            peak = i - numConsecutiveDrops
            if peak + peak2InflectionDiff >= 0 and peak + peak2TroughDiff < numFiles:
                peakIndicies.append(peak)

            i += refactoryPeriod

            testFrames, testAreas, i = reinitializeTestFramesAndAreas(i)

        else:
            testFrames.pop(0)
            testAreas.pop(0)

            image = im.getJellyImageFromFile(files[i])
            binary_image = im.getBinaryJelly(image, lowerThresh)
            area = im.findBinaryArea(binary_image)

            testFrames.append(i)
            testAreas.append(area)
            i += 1

    return peakIndicies

def differenceAngleFinder(files):

    global globalmovementSegment  # the latest segment number across all video chunks (segment = non-moving period)
    global lowerThreshold  # lower threshold to create binary image of jelly to assess area (for downturns)
    global peak2InflectionDiff  # the number of frames past the peak where the inflection point occurs (this should always be negative)
    global peak2TroughDiff  # the number of frames past the peak where the lowest area is found on average
    global postPeakRefractoryPeriod  # the number of frames to preclude from analysis
    global inflectionTestDiff  # the number of frames after inflection point where the difference in calculated
    global inflectionTestBinaryThreshold  # the ideal threshold to locate the area of difference
    global chosenSD  # the sd of the chosen test diff and threshold when they were initialized
    global numConsecutiveDrops  # the number of consecutive drops needed to count something as a downturn

    global isMoving
    global isQuestionablyStationary
    global lastStationaryCentroid

    startTime = datetime.datetime.now()

    i = 0

    # movement parameters
    firstStationaryAfterMovement = 0 # first stationary frame (i) after movement
    isQuestionablyStationary = False
    centroidBefore = None
    counter = 0
    peak = 0
    pulseCountInQuestionablyStationary = 0

    data = []
    movingPeaks = []

    # initializes lists with 'numConsecutiveDrops' of files
    def reinitializeTestFramesAndAreas(j):
        testFrames = []  # this list should never be more than 5 entries long, ex. [51, 52, 53, 54, 55]
        testAreas = []  # this list should never be more than 5 entries long, ex. [253, 255, 256, 255, 255]

        while len(testFrames) < numConsecutiveDrops and j < framesInChunk:
            image = im.getJellyImageFromFile(files[j])
            binary_image = im.getBinaryJelly(image, lowerThreshold)
            area = im.findBinaryArea(binary_image)

            testFrames.append(j)
            testAreas.append(area)
            j += 1

        return testFrames, testAreas, j

    # function to save out data
    def saveOutData():
        df = pd.DataFrame(data, columns=['global frame', 'chunk frame', 'angle', 'centroid x', 'centroid y'])
        if DEBUG: print(df.head())
        dataTitle = '{}_{:03}.csv'.format(chunkName, globalmovementSegment)
        df.to_csv(str(angleOutputDir / dataTitle), index=False)


    testFrames, testAreas, i = reinitializeTestFramesAndAreas(i)

    try:
        while i < framesInChunk:
            if counter%100 == 0: print('i: '+str(i)+ ', peak: ' + str(peak))

            isDownturn = dm.is_downturn(0, testAreas, numConsecutiveDrops)

            if isDownturn:
                peak = i - numConsecutiveDrops

                # checks that peaks are within testing bounds
                if peak + peak2InflectionDiff >= 0 and peak + peak2TroughDiff < framesInChunk:

                    troughInfile = files[peak + peak2TroughDiff]
                    relaxedInfile = files[peak + peak2InflectionDiff]

                    troughImg = im.getJellyGrayImageFromFile(troughInfile)
                    relaxedImg = im.getJellyGrayImageFromFile(relaxedInfile)

                    centroidDiff = im.getGrayscaleImageDiff_absolute(troughImg, relaxedImg)
                    binaryCentroidDiff = im.getBinaryJelly(centroidDiff, lower_bound=0.05, upper_bound=1)
                    centroidRegion = im.findJellyRegion(binaryCentroidDiff)
                    centroid = im.findCentroid_boundingBox(centroidRegion)

                    if lastStationaryCentroid is None:
                        lastStationaryCentroid = centroid

                    if CONFIRMATIONIMAGES: im.saveJellyPlot(
                        im.getCentroidVerificationImg(centroidDiff, binaryCentroidDiff, centroid),
                        str(confirmationImagesPath / '{}_{}_centroid.png'.format(peak, chunkName)))

                    if isMoving:

                        data.append([peak + lastFrameOfPreviousChunk, peak, np.nan, centroid[0], centroid[1]])

                        movedBefore = isMoving
                        isMoving = im.distance(centroid, lastStationaryCentroid) > movementThreshold2KeepMoving

                        lastStationaryCentroid = centroid

                        if movedBefore and not isMoving:
                            firstStationaryAfterMovement = i
                            pulseCountInQuestionablyStationary = 0
                            isQuestionablyStationary = True

                    elif isQuestionablyStationary:

                        data.append([peak + lastFrameOfPreviousChunk, peak, np.nan, centroid[0], centroid[1]])

                        isMoving = im.distance(centroid, lastStationaryCentroid) > movementThreshold4reinitialization

                        if isMoving:
                            movingPeaks.append(peak)
                            isQuestionablyStationary = False

                        pulseCountInQuestionablyStationary += 1

                        if i - firstStationaryAfterMovement > numFrames2ConfirmStationary:

                            if firstStationaryAfterMovement == 0:
                                data = []
                            else:
                                # must mutate data to take out
                                data = data[:-pulseCountInQuestionablyStationary]
                                saveOutData()
                                data = []

                                globalmovementSegment += 1

                            i = firstStationaryAfterMovement

                            # now there is confirmed 5 minutes after initial stationary point
                            lowerThreshold = None

                            # peak2InflectionDiff, peak2TroughDiff, postPeakRefractoryPeriod, infflectionTestDiff,
                            # inflectionTestBinaryThreshold, and chosen SD are all static.



                            initialize_params(files, i)

                            isQuestionablyStationary = False

                            pulseCountInQuestionablyStationary = 0

                        # until count from current i to last stationary i reaches this point,
                        # the program is in a holding pattern of sorts.

                    else:
                        testInfile = files[peak + peak2InflectionDiff + inflectionTestDiff]
                        testImg = im.getJellyGrayImageFromFile(testInfile)

                        if CONFIRMATIONIMAGES: plt.imsave(
                            str(confirmationImagesPath / '{}_{}_interestFrames.png'.format(peak, chunkName)),
                            im.juxtaposeImages(np.array([[relaxedImg, testImg, peakImg, troughImg]])))


                        if centroidBefore is not None:
                            reinitializeAreaPlot = im.distance(centroid, centroidBefore) > movementThreshold4newNormalizationImg
                            if reinitializeAreaPlot:
                                peakInfile = files[peak]
                                peakImg = im.getJellyGrayImageFromFile(peakInfile)
                                peakDiff = im.getGrayscaleImageDiff_absolute(troughImg, peakImg)
                                binaryPeakDiff = im.getBinaryJelly(peakDiff, lower_bound=0.05, upper_bound=1)
                                averagedDynamicRangeMaskedImg = im.dynamicRangeImg_AreaBased(relaxedImg, binaryPeakDiff, 5)

                        else:
                            peakInfile = files[peak]
                            peakImg = im.getJellyGrayImageFromFile(peakInfile)
                            peakDiff = im.getGrayscaleImageDiff_absolute(troughImg, peakImg)
                            binaryPeakDiff = im.getBinaryJelly(peakDiff, lower_bound=0.05, upper_bound=1)
                            averagedDynamicRangeMaskedImg = im.dynamicRangeImg_AreaBased(relaxedImg, binaryPeakDiff, 5)

                        centroidBefore = centroid

                        if CONFIRMATIONIMAGES: im.saveJellyPlot(
                            averagedDynamicRangeMaskedImg, str(confirmationImagesPath / '{}_{}_dynRng.png'.format(peak, chunkName)))

                        testDiff = im.getGrayscaleImageDiff_absolute(testImg, relaxedImg)
                        normalizedTestDiff = testDiff / averagedDynamicRangeMaskedImg

                        binaryDiffImg = im.getBinaryJelly(normalizedTestDiff, lower_bound=inflectionTestBinaryThreshold)

                        biggestRegion = im.findJellyRegion(binaryDiffImg)

                        if biggestRegion is not None:
                            local_com = im.findCentroid_regionProp(biggestRegion)
                            zeroDegreePoint = (centroid[0], 0)

                            angle = dm.getAngle(zeroDegreePoint, centroid, local_com)

                            if CONFIRMATIONIMAGES: im.saveJellyPlot(
                                binaryDiffImg, str(confirmationImagesPath / '{}_{}_angle.png'.format(peak, chunkName)),
                                [centroid, local_com, zeroDegreePoint])
                        else:
                            angle = np.nan

                            if CONFIRMATIONIMAGES: im.saveJellyPlot(
                                binaryDiffImg, str(confirmationImagesPath / '{}_{}_angle.png'.format(peak, chunkName)),
                                [centroid])


                        data.append([peak + lastFrameOfPreviousChunk, peak, angle, centroid[0], centroid[1]])

                        movedBefore = isMoving
                        isMoving = im.distance(centroid, lastStationaryCentroid) > movementThreshold4reinitialization

                        if isMoving and not movedBefore:
                            isQuestionablyStationary = False

                            lastStationaryCentroid = centroid

                    """
                    if DEBUG: print("index: {}, isMoving: {}, isQStat: {}, centroid: {}".format(i,
                                                                                               isMoving,
                                                                                               isQuestionablyStationary,
                                                                                               str(centroid)))
                    """

                i += postPeakRefractoryPeriod
                counter += 1

                testFrames, testAreas, i = reinitializeTestFramesAndAreas(i)

            else:
                testFrames.pop(0)
                testAreas.pop(0)

                image = im.getJellyImageFromFile(files[i])
                binary_image = im.getBinaryJelly(image, lowerThreshold)
                area = im.findBinaryArea(binary_image)

                testFrames.append(i)
                testAreas.append(area)
                i += 1
                counter += 1

    except Exception as error:
        print('{} error occurred.'.format(error))
        print("index: {}, isMoving: {}, isQStat: {}, centroid: {}".format(i,
                                                                       isMoving,
                                                                       isQuestionablyStationary,
                                                                       str(centroid)))
        raise

    finally:
        saveOutData()

        td = datetime.datetime.now() - startTime
        if DEBUG: print('time since start: {}'.format(td))

        if DEBUG: print(movingPeaks)

    return True

# TODO: if in holding pattern and change to new jelly it needs None currentParamDF
    # if the analysis is in the questionably stationary period at the end of the chunk, all these variables must be initiated upon the next chunk


# TODO: Test 60 fps system to see how accurate it is compared to 120 fps



def runFullVideoAnalysis(
        vidImgStkDir,  # changes chunk to chunk
        verOutDir,  # only changes for each recording
        angOutDir,  # only changes for each recording
        recName,  # only changes for each recording
        pathOfLastParamDF,  # changes chunk segment to chunk segment (each video chunk has 1 or more segments)
        isPotentiallyStationary,
        lstStatCentroid,
        macintosh # True is computer is a Mac, false if it is a windows
):
    # any parameters that are not imputed into the function are set automatically

    # creates global references to mutate below if different input parameters are specified
    # static params for each chunk (fraction over overall video recording ex. xaa, xab, xac, etc.)
    global chunkName  # the name of this jelly and its video chunk id
    global recordingName  # the name of the recording being processed
    global videoImageStackDir  # where the image stack for this specific chunk is located
    global verificationOutputDir  # place where all the segment verification directeries are stored
    global angleOutputDir  # place where all the data is stored
    global pathOfCurrentParamDF  # path of the latest param dataframe for the latest segment
    global framesInChunk  # absolute number of frames in specific video chunk
    global lastFrameOfPreviousChunk  # the last frame the occured before this recording chunk
    global lastFrameOfThisChunk  # the last frame that occurs in this recording chunk
    global isMoving  # if the Jellyfish was moving at the end of the last video
    global isQuestionablyStationary # if the jellyfish was questionably stationary at the end of the last video

    # static params for each non-moving segment
    # all of these must be read in from the previous param segment
    global globalmovementSegment  # the latest segment number across all video chunks (segment = non-moving period)
    global averageTroughBinaryArea  # trough area to use in setting lower threshold automatically
    global lowerThreshold  # lower threshold to create binary image of jelly to assess area (for downturns)
    global peak2InflectionDiff  # the number of frames past the peak where the inflection point occurs (this should always be negative)
    global peak2TroughDiff  # the number of frames past the peak where the lowest area is found on average
    global postPeakRefractoryPeriod  # the number of frames to preclude from analysis
    global inflectionTestDiff  # the number of frames after inflection point where the difference in calculated
    global inflectionTestBinaryThreshold  # the ideal threshold to locate the area of difference
    global chosenSD  # the sd of the chosen test diff and threshold when they were initialized
    global numConsecutiveDrops  # the number of consecutive drops needed to count something as a downturn

    # static params for each recording
    # intialized as static variables, do not need to be read in.
    global initialRefracPeriod  # initial refractionary period used in
    global movementThreshold4reinitialization  # number of pixels from one centroid to another to consider a jelly as moving.
    global movementThreshold2KeepMoving # number of pixels from one centroid to the next to continue to be considered moving
    global movementThreshold4newNormalizationImg    # number of pixels from one centroid to another to reinitialize the average normalization img
    global pct2skip4RefractionaryPeriod  # percent on average Interpulse Interval to skip when initializing postPeakRefractoryPeriod
    global numFramesForOrientation  # number of frames to save in the Orientation folder at the start of a new segment
    global numFramesForParamInitialization  # number of frames to use when initializing params for a new segment
    global numFrames2ConfirmStationary  # number of frames after first stationary frame after movement to confirm jellyfish is stationary

    global lastStationaryCentroid  # used in analysis to detect movement. Passed in return statement and as a parameter.

    global MAC
    MAC = macintosh

    lastStationaryCentroid = lstStatCentroid

    # initializing chunk level params
    videoImageStackDir = vidImgStkDir
    chunkName = vidImgStkDir.stem
    recordingName = recName
    pathOfCurrentParamDF = None
    verificationOutputDir = verOutDir
    angleOutputDir = angOutDir
    isQuestionablyStationary = isPotentiallyStationary

    # intiailizing recording static variables (do not need to be read in)
    initialRefracPeriod = 40
    movementThreshold4reinitialization = 50
    movementThreshold2KeepMoving = 5
    movementThreshold4newNormalizationImg = 5
    pct2skip4RefractionaryPeriod = 3 / 4
    numFramesForOrientation = 500  # cannot be more than numFramesForParamInitialization
    numFramesForParamInitialization = 2500
    numFrames2ConfirmStationary = 7200  # 120 * 60, 60 seconds of recording to stay stationary (just for testing)

    # loads in files from FFMPEG Stack located at 'videoImageStackDir'
    files = getFrameFilePaths()

    if pathOfLastParamDF is not None and not isQuestionablyStationary:
        params_df = pd.read_csv(str(pathOfLastParamDF), index_col=0)
        print(params_df)

        globalmovementSegment = int(params_df.loc['globalmovementSegment']['data'])

        averageTroughBinaryArea = float(params_df.loc['averageTroughBinaryArea']['data'])
        lowerThreshold = float(params_df.loc['lowerThreshold']['data'])

        peak2InflectionDiff = int(params_df.loc['peak2InflectionDiff']['data'])
        peak2TroughDiff = int(params_df.loc['peak2TroughDiff']['data'])
        postPeakRefractoryPeriod = int(params_df.loc['postPeakRefractoryPeriod']['data'])
        inflectionTestDiff = int(params_df.loc['inflectionTestDiff']['data'])
        inflectionTestBinaryThreshold = float(params_df.loc['inflectionTestBinaryThreshold']['data'])
        chosenSD = float(params_df.loc['chosenSD']['data'])
        numConsecutiveDrops = int(params_df.loc['numConsecutiveDrops']['data'])

        isMoving = params_df.loc['isMoving']['data'] == 'True'

        lastFrameOfPreviousChunk = int(params_df.loc['lastFrameOfThisChunk']['data'])

        pathOfCurrentParamDF = pathOfLastParamDF.parent / '{}_{}.csv'.format(chunkName, globalmovementSegment)

        framesInChunk = len(files)

        lastFrameOfThisChunk = lastFrameOfPreviousChunk + framesInChunk

        if DEBUG: print('parameters initialized')

        initialize_params(files, 0)

    elif isQuestionablyStationary:
        params_df = pd.read_csv(str(pathOfLastParamDF), index_col=0)
        globalmovementSegment = int(params_df.loc['globalmovementSegment']['data'])
        averageTroughBinaryArea = float(params_df.loc['averageTroughBinaryArea']['data'])
        lowerThreshold = float(params_df.loc['lowerThreshold']['data'])

        peak2InflectionDiff = int(params_df.loc['peak2InflectionDiff']['data'])
        peak2TroughDiff = int(params_df.loc['peak2TroughDiff']['data'])
        postPeakRefractoryPeriod = int(params_df.loc['postPeakRefractoryPeriod']['data'])
        inflectionTestDiff = int(params_df.loc['inflectionTestDiff']['data'])
        inflectionTestBinaryThreshold = float(params_df.loc['inflectionTestBinaryThreshold']['data'])
        chosenSD = float(params_df.loc['chosenSD']['data'])

        numConsecutiveDrops = int(params_df.loc['numConsecutiveDrops']['data'])
        lastFrameOfPreviousChunk = int(params_df.loc['lastFrameOfThisChunk']['data'])
        isMoving = False

        pathOfCurrentParamDF = pathOfLastParamDF.parent / '{}_{}.csv'.format(chunkName, globalmovementSegment)

        framesInChunk = len(files)

        lastFrameOfThisChunk = lastFrameOfPreviousChunk + framesInChunk

    else:
        globalmovementSegment = 0
        averageTroughBinaryArea = None
        lowerThreshold = None
        peak2InflectionDiff = 0
        peak2TroughDiff = 0
        postPeakRefractoryPeriod = None
        inflectionTestDiff = None
        inflectionTestBinaryThreshold = None
        chosenSD = None
        numConsecutiveDrops = 5
        lastFrameOfPreviousChunk = 0
        isMoving = False

        framesInChunk = len(files)

        lastFrameOfThisChunk = lastFrameOfPreviousChunk + framesInChunk

        if DEBUG: print('parameters initialized')

        # initialize parameters (on first video run)
        initialize_params(files, 0)

    # all global variables should be initialized, saving params
    saveVariableParams()

    # run analysis

    if CHIME: dm.chime(MAC, 'starting analysis')
    differenceAngleFinder(files)

    saveVariableParams()

    if CHIME: dm.chime(MAC)

    return pathOfCurrentParamDF, isQuestionablyStationary, lastStationaryCentroid
