
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import DataMethods as dm

import ImageMethods as im

from datetime import timedelta

########################################################################################################################
DEBUG = True
CHIME = True
########################################################################################################################

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

def downturnFinder(files, refactoryPeriod, lowerThresh, numberOfConsecutiveDrops, peak2InflectionDiff, peak2TroughDiff, DEBUG = False):

    print('searching for peaks (downturnfinder)')

    i = 0
    numFiles = len(files)

    peakIndicies = []

    # initializes lists with 'numberOfConsecutiveDrops' of files
    def reinitializeTestFramesAndAreas(j):
        testFrames = []  # this list should never be more than 5 entries long, ex. [51, 52, 53, 54, 55]
        testAreas = []  # this list should never be more than 5 entries long, ex. [253, 255, 256, 255, 255]

        while len(testFrames) < numberOfConsecutiveDrops and j < numFiles:
            image = im.getJellyImageFromFile(files[j])
            binary_image = im.getBinaryJelly(image, lowerThresh)
            area = im.findBinaryArea(binary_image)

            testFrames.append(j)
            testAreas.append(area)
            j += 1

        return testFrames, testAreas, j

    testFrames, testAreas, i = reinitializeTestFramesAndAreas(i)

    while i < numFiles:
        isDownturn = dm.is_downturn(0, testAreas, numberOfConsecutiveDrops)

        if DEBUG: print('i: {}, isDownturn: {}, testAreas: {}, testFrames: {}'.format(i, isDownturn, testAreas, testFrames))

        if isDownturn:
            peak = i - numberOfConsecutiveDrops
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

def initialParameters4thresholding(fileSubset, lowerThreshold, initialRefracPeriod):

    if DEBUG: print('calculating binaryImageAreas\n')
    binaryImageAreas4thresholding = getBinaryAreas(fileSubset, lowerThreshold)

    if DEBUG: print('calculating peaksOnBinaryImage\n')
    peaksOnBinaryImage4thresholding = downturnFinder(fileSubset, initialRefracPeriod, lowerThreshold, 10, 15, 25)
    if peaksOnBinaryImage4thresholding[0] < initialRefracPeriod: peaksOnBinaryImage4thresholding.pop(0)

    if DEBUG: print('calculating differences\n')
    troughsOnBinaryImage4thresholding = dm.getTroughs(binaryImageAreas4thresholding)
    peak2TroughDiff4thresholding = dm.likelyPostPeakTroughDiff(troughsOnBinaryImage4thresholding,
                                                               peaksOnBinaryImage4thresholding)
    peak2InflectionDiff4thresholding = dm.getLikelyInflectionDiff(binaryImageAreas4thresholding,
                                                                  peaksOnBinaryImage4thresholding)

    return [peaksOnBinaryImage4thresholding, peak2TroughDiff4thresholding, peak2InflectionDiff4thresholding]

def autoLowerThreshold(averageTroughBinaryArea, peak2TroughDiff, peaksOnBinaryImage, fileSubset, thresholdingDir, recordingName):
    # completed automated based on averageTroughBinaryArea
    thresholdStep = 0.005
    chosenThreshold = 0.05

    testTroughAverage = averageTroughBinaryArea + 1

    while testTroughAverage > averageTroughBinaryArea:
        chosenThreshold += thresholdStep

        troughAreas = []
        for i, peak in enumerate(peaksOnBinaryImage):
            if peak + peak2TroughDiff < len(fileSubset):

                troughInfile = fileSubset[peak + peak2TroughDiff]

                troughImg = im.getJellyGrayImageFromFile(troughInfile)

                binaryTroughImg = im.getBinaryJelly(troughImg, chosenThreshold)

                jellyTroughBinaryArea = im.findBinaryArea(binaryTroughImg)

                troughAreas.append(jellyTroughBinaryArea)

        testTroughAverage = np.mean(troughAreas)

        print('chosenThreshold: {} (test area, {}; target area, {})'.format(chosenThreshold, testTroughAverage,
                                                                            averageTroughBinaryArea))

    for i, peak in enumerate(peaksOnBinaryImage):
        if peak + peak2TroughDiff < len(fileSubset):
            peakInfile = fileSubset[peak]
            troughInfile = fileSubset[peak + peak2TroughDiff]

            peakImg = im.getJellyGrayImageFromFile(peakInfile)
            troughImg = im.getJellyGrayImageFromFile(troughInfile)

            binaryPeakImg = im.getBinaryJelly(peakImg, chosenThreshold)
            binaryTroughImg = im.getBinaryJelly(troughImg, chosenThreshold)

            im.saveJellyPlot(im.juxtaposeImages(np.array([[binaryPeakImg, binaryTroughImg]])),
                             (thresholdingDir / '{}_thresholdVerification_{}.png'.format(recordingName, peak)))

    return chosenThreshold


def selectAverageTroughBinaryArea(fileSubset, thresholdingDir, recordingName, peaksOnBinaryImage, peak2InflectionDiff, peak2TroughDiff):
    maxIntensities = []
    minIntensities = []
    for i, peak in enumerate(peaksOnBinaryImage):
        if peak + peak2InflectionDiff >= 0 and peak + peak2TroughDiff < len(fileSubset):
            troughInfile = fileSubset[peak + peak2TroughDiff]
            relaxedInfile = fileSubset[peak + peak2InflectionDiff]
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

    print('peak2TroughDiff: {}'.format(peak2TroughDiff))

    while True:

        troughAreas = []

        for peak in peaksOnBinaryImage:
            peakInfile = fileSubset[peak]
            troughInfile = fileSubset[peak+peak2TroughDiff]

            peakImg = im.getJellyGrayImageFromFile(peakInfile)
            troughImg = im.getJellyGrayImageFromFile(troughInfile)

            binaryPeakImg = im.getBinaryJelly(peakImg, lowerThreshold)
            binaryTroughImg = im.getBinaryJelly(troughImg, lowerThreshold)

            im.saveJellyPlot(im.juxtaposeImages(np.array([[binaryPeakImg, binaryTroughImg]])),
                             (thresholdingDir / '{}_thresholdVerification_{}.png'.format(recordingName, peak)))

            jellyTroughBinaryArea = im.findBinaryArea(binaryTroughImg)

            troughAreas.append(jellyTroughBinaryArea)

        if CHIME: dm.chime(MAC, 'input time')
        print('average trough area: {}, sd of trough areas: {}'.format(np.mean(troughAreas), np.std(troughAreas)))

        print('Change thresholds: ')
        print('select \'1\' to change {} which is {}'.format('lowerThreshold', lowerThreshold))
        print('select \'2\' to remove a peak from peaksOnBinaryImage')
        print('or \'3\' to continue.')

        selectionVar = dm.getSelection([1, 2, 3])
        if selectionVar == '1':
            lowerThreshold = dm.reassignFloatVariable(lowerThreshold, 'lowerThreshold')
            dm.replaceDir(thresholdingDir)
        elif selectionVar == '2':
            print('peaksOnBinaryImage: {}'.format(peaksOnBinaryImage))
            index2Pop = int(dm.getSelection(list(range(len(peaksOnBinaryImage)))))
            peaksOnBinaryImage.pop(index2Pop)
        else:
            return np.mean(troughAreas)


def selectInflectionThresholdandDiff(peaksOnBinaryImage, fileSubset, recordingName, peak2InflectionDiff, peak2TroughDiff, initializationOutputDir, angleArrImageDir, centroidDir, dynamicRangeDir):

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

        centroidVerOutFile = centroidDir / 'centroid for {} - {}.png'.format(recordingName, peak + peak2InflectionDiff)
        im.saveJellyPlot(im.getCentroidVerificationImg(centroidDiff, binaryCentroidDiff, centroid), centroidVerOutFile)

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

        testingOutfile = angleArrImageDir / 'testPlot for {} - {}.png'.format(recordingName, peak + peak2InflectionDiff)
        pulseAngleData = im.saveDifferenceTestingAggregationImage(relaxedImg, testDiffImages, thresholdCases,
                                                                  testingOutfile, False, centroid)

        for n, row in enumerate(pulseAngleData):
            for m, angle in enumerate(row):
                angleData[i][m][n] = angle

        otherData[i][0] = peak + peak2InflectionDiff
        otherData[i][1] = peak
        otherData[i][2] = peak + peak2TroughDiff

    angleDataAsRows = [x.ravel() for x in angleData]
    pulseAngleOutput = np.concatenate((np.tile([postInflectionDiffCases], len(thresholdCases)), angleDataAsRows))
    otherDataOut = np.concatenate(([otherDataCols], otherData))

    # warning: this results in mixed data. This cannot be saved by numpy csv methods. Pandas is easiest way to save.
    outframe = np.concatenate((otherDataOut, pulseAngleOutput), axis=1)

    # saves data into verification frame
    dfOut = pd.DataFrame(outframe)
    dataTitle = '{}_testDifferenceVerification.csv'.format(recordingName)
    verificationCSVOutFile = initializationOutputDir / dataTitle
    dfOut.to_csv(str(verificationCSVOutFile), header=False, index=False)

    # setting test difference and threshold

    # read in by eye angle measurements
    byEyeAngleDF = pd.DataFrame(peaksOnBinaryImage, columns=['peaks'])
    byEyeAngleDF['by eye measurement (0 to 360)'] = np.nan
    byEyeAngleDFioPath = initializationOutputDir / '{}_byEyeAngles.csv'.format(recordingName)
    byEyeAngleDF.to_csv(str(byEyeAngleDFioPath))

    if CHIME: dm.chime(MAC, 'input time')

    print('time to enter by eye angles for each pulse')
    print('entries must be from 0 to 360')
    print('Enter \'1\' to continue.')
    dm.getSelection([1])

    byEyeAngleDF = pd.read_csv(str(byEyeAngleDFioPath))

    byEyeAngles = byEyeAngleDF['by eye measurement (0 to 360)'].tolist()

    i = 0
    while i < len(byEyeAngles):
        if byEyeAngles[i] == np.nan:
            np.delete(angleData, i, 0)
            byEyeAngles.pop(i)
        else:
            i += 1

    angleDataShape = angleData.shape

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

    chosenSD = sdTable[thresholdCases.index(inflectionTestBinaryThreshold)][postInflectionDiffCases.index(inflectionTestDiff)]

    return inflectionTestDiff, inflectionTestBinaryThreshold, chosenSD


def initialization_Main(pathOfPreInitializationDF, pathOfInitializationStack, recordingOutputDir, macintosh):
    global MAC
    MAC = macintosh

    preInitializationDf = pd.read_csv(str(pathOfPreInitializationDF))
    initializationStack = dm.getFrameFilePaths(pathOfInitializationStack)

    # static variables across single recording that must be read in from df
    recordingName = preInitializationDf.iloc[0]['RecordingName']
    initializationOutputDir = dm.makeOutDir(recordingOutputDir, 'InitializationVerification')
    framerate = preInitializationDf.iloc[0]['FrameRate']
    framesInRecording = sum(preInitializationDf['NumFramesInChunk'].tolist())  # not saved in final DF
    lengthOfRecording = timedelta(0, framesInRecording/framerate)

    # static variables across single recording that must be initialized
    averageTroughBinaryArea = None
    lowerThreshold = 0.1   # not saved in final DF
    peak2InflectionDiff = None
    peak2TroughDiff = None
    postPeakRefractoryPeriod = 40
    inflectionTestDiff = None
    inflectionTestBinaryThreshold = None
    chosenSD = None   # not saved in final DF
    numConsecutiveDrops = 5

    # static variables across all recordings
    movementThreshold4reinitialization = 50
    movementThreshold2KeepMoving = 5
    movementThreshold4newNormalizationImg = 5
    pct2skip4RefractionaryPeriod = 3 / 4
    numFramesForParamInitialization = 3200 # 120 * 30
    numFrames2ConfirmStationary = 7200  # 120 * 60, 60 seconds of recording to stay stationary (just for testing)

    thresholdingDir = dm.makeOutDir(initializationOutputDir, '{}_ThresholdingPlots'.format(recordingName))
    angleArrImageDir = dm.makeOutDir(initializationOutputDir, '{}_AngleArrImageDir'.format(recordingName))
    centroidDir = dm.makeOutDir(initializationOutputDir, '{}_CentroidVerificationDir'.format(recordingName))
    dynamicRangeDir = dm.makeOutDir(initializationOutputDir, '{}_dynamicRangeVerificationDir'.format(recordingName))

    def saveVariableParams():
        imp_parameters = pd.DataFrame(np.array([
            [recordingName, 'the name of the recording being processed'],
            [initializationOutputDir,
             'place where all the initialization verification images and directories are stored'],
            [framerate, 'framerate of the specified recording'],
            [framesInRecording, 'number of total frames in recording'],
            [lengthOfRecording, 'length of recording in timedelta format'],

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

            [movementThreshold4reinitialization,
             'number of pixels from one centroid to another to consider a jelly as moving.'],
            [movementThreshold2KeepMoving,
             'number of pixels from one centroid to the next to continue to be considered moving'],
            [movementThreshold4newNormalizationImg,
             'number of pixels from one centroid to another to reinitialize the average normalization img'],
            [pct2skip4RefractionaryPeriod,
             'percent on average Interpulse Interval to skip when initializing postPeakRefractoryPeriod'],
            [numFramesForParamInitialization, 'number of frames to use when initializing params for a new segment'],
            [numFrames2ConfirmStationary,
             'number of frames after first stationary frame after movement to confirm jellyfish is stationary'],
        ]),

            index=['recordingName',
                   'initializationOutputDir',
                   'framerate',
                   'framesInRecording',
                   'lengthOfRecording',

                   'averageTroughBinaryArea',
                   'lowerThreshold',
                   'peak2InflectionDiff',
                   'peak2TroughDiff',
                   'postPeakRefractoryPeriod',
                   'inflectionTestDiff',
                   'inflectionTestBinaryThreshold',
                   'chosenSD',
                   'numConsecutiveDrops',

                   'movementThreshold4reinitialization',
                   'movementThreshold2KeepMoving',
                   'movementThreshold4newNormalizationImg',
                   'pct2skip4RefractionaryPeriod',
                   'numFramesForParamInitialization',
                   'numFrames2ConfirmStationary'],

            columns=['data', 'notes'])

        imp_parameters.to_csv(str(initializationOutputDir / '{}_initializationParameters.csv'.format(recordingName)))

    saveVariableParams()

    if DEBUG: print('intial parameters set\n')

    peaksOnBinaryImage, peak2TroughDiff, peak2InflectionDiff = initialParameters4thresholding(initializationStack,
                                                                                             lowerThreshold,
                                                                                             postPeakRefractoryPeriod)

    print('{}:\n {}, {}'.format(peaksOnBinaryImage, peak2TroughDiff, peak2InflectionDiff))

    if DEBUG: print('calculating averageTroughBinaryArea\n')
    averageTroughBinaryArea = selectAverageTroughBinaryArea(initializationStack, thresholdingDir, recordingName, peaksOnBinaryImage, peak2InflectionDiff, peak2TroughDiff)

    saveVariableParams()

    if DEBUG: print('calculating lowerThreshold\n')
    lowerThreshold = autoLowerThreshold(averageTroughBinaryArea, peak2TroughDiff, peaksOnBinaryImage,initializationStack, thresholdingDir, recordingName)

    saveVariableParams()

    if DEBUG: print('getting BinaryAreas\n')
    binaryImageAreas = getBinaryAreas(initializationStack, lowerThreshold)

    if DEBUG: print('getting peaksOnBinaryImage\n')
    # gets peak frame nums from binaryImageAreas
    peaksOnBinaryImage = downturnFinder(initializationStack, postPeakRefractoryPeriod, lowerThreshold, numConsecutiveDrops, peak2InflectionDiff, peak2TroughDiff)

    # gets peak2TroughDiff from peaksOnBinaryImage and binaryImageAreas
    troughsOnBinaryImage = dm.getTroughs(binaryImageAreas)
    peak2TroughDiff = dm.likelyPostPeakTroughDiff(troughsOnBinaryImage, peaksOnBinaryImage)

    # initializes inflection diff from jellyRegionAreas
    peak2InflectionDiff = dm.getLikelyInflectionDiff(binaryImageAreas, peaksOnBinaryImage)

    # initializes frames to skip for analysis after each peak
    averageIPI = dm.averageInterpulseInterval(peaksOnBinaryImage)
    postPeakRefractoryPeriod = int(pct2skip4RefractionaryPeriod * averageIPI)

    if CHIME: dm.chime(MAC, 'input time')
    while True:
        plotOutpath = initializationOutputDir / 'areaVerificationPlot.png'
        saveAreasPlot(binaryImageAreas, peaksOnBinaryImage, plotOutpath,
                      [peak2InflectionDiff, peak2InflectionDiff + 5, peak2TroughDiff],
                      postPeakRefractoryPeriod)

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

            peaksOnBinaryImage = downturnFinder(initializationStack, postPeakRefractoryPeriod, lowerThreshold, numConsecutiveDrops, peak2InflectionDiff, peak2TroughDiff)
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

    if DEBUG: print('finished setting postPeakRefractoryPeriod, numConsecutiveDrops, peak2InflectionDiff, peak2TroughDiff')

    saveVariableParams()

    peak2InflectionDiff = -16
    peak2TroughDiff = 33

    i = 0
    while i < len(peaksOnBinaryImage):
        if peaksOnBinaryImage[i] + peak2InflectionDiff < 0 or peaksOnBinaryImage[i] + peak2TroughDiff >= numFramesForParamInitialization:
            peaksOnBinaryImage.pop(i)
        else:
            i += 1



    if DEBUG: print('Running \'selectInflectionThresholdandDiff\'\n')

    inflectionTestDiff, inflectionTestBinaryThreshold, chosenSD = selectInflectionThresholdandDiff(peaksOnBinaryImage, initializationStack, recordingName, peak2InflectionDiff, peak2TroughDiff, initializationOutputDir, angleArrImageDir, centroidDir, dynamicRangeDir)

    saveVariableParams()

    # static params for each chunk (fraction over overall video recording ex. xaa, xab, xac, etc.)
    lastFrameOfPreviousChunk = 0  # calculated from framesInChunk

    postInitiationDF = preInitializationDf.copy()

    # static params for each recording
    postInitiationDF['averageTroughBinaryArea'] =  averageTroughBinaryArea  # trough area to use in setting lower threshold automatically
    postInitiationDF['peak2InflectionDiff'] =  peak2InflectionDiff  # the number of frames past the peak where the inflection point occurs (this should always be negative)
    postInitiationDF['peak2TroughDiff'] =  peak2TroughDiff  # the number of frames past the peak where the lowest area is found on average
    postInitiationDF['postPeakRefractoryPeriod'] =  postPeakRefractoryPeriod  # the number of frames to preclude from analysis
    postInitiationDF['inflectionTestDiff'] =  inflectionTestDiff  # the number of frames after inflection point where the difference in calculated
    postInitiationDF['inflectionTestBinaryThreshold'] =  inflectionTestBinaryThreshold  # the ideal threshold to locate the area of difference
    postInitiationDF['numConsecutiveDrops'] =  numConsecutiveDrops  # the number of consecutive drops needed to count something as a downturn

    # static params for all recording
    postInitiationDF['movementThreshold4reinitialization'] =  movementThreshold4reinitialization  # number of pixels from one centroid to another to consider a jelly as moving.
    postInitiationDF['movementThreshold2KeepMoving'] =  movementThreshold2KeepMoving  # number of pixels from one centroid to the next to continue to be considered moving
    postInitiationDF['movementThreshold4newNormalizationImg'] =  movementThreshold4newNormalizationImg  # number of pixels from one centroid to another to reinitialize the average normalization img
    postInitiationDF['numFramesForParamInitialization'] =  numFramesForParamInitialization  # number of frames to use when initializing params for a new segment
    postInitiationDF['numFrames2ConfirmStationary'] =  numFrames2ConfirmStationary  # number of frames after first stationary frame after movement to confirm jellyfish is stationary

    chunkFrameCounts = preInitializationDf['NumFramesInChunk'].tolist()
    previousChunkTotals = []

    for i in range(len(postInitiationDF)):
        previousChunkTotals.append(sum(chunkFrameCounts[:i]))

    postInitiationDF['lastFrameOfPreviousChunk'] = previousChunkTotals

    postInitiationDFOutName = pathOfPreInitializationDF.stem[:pathOfPreInitializationDF.stem.rindex('_')] + '_PostInitializationDF.csv'

    postInitiationDFOutpath = pathOfPreInitializationDF.parent / postInitiationDFOutName

    postInitiationDF.to_csv(str(postInitiationDFOutpath))