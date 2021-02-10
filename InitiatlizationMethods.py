
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


# Bottleneck of init_movie usage
def getBinaryAreas(init_movie_np, lowerThreshold):

    temp_thresholded_movie_arr = init_movie_np > lowerThreshold
    temp_area_list = np.sum(temp_thresholded_movie_arr, axis=(1, 2))

    return temp_area_list


def saveAreasPlot(areas, peaks, outpath, diffsList, refractionaryPeriod = None):

    diffFrameLists = []
    for diff in diffsList:
        diffFramesBasedOnPeak = [x + diff for x in peaks]
        diffFrameLists.append(diffFramesBasedOnPeak)

    fig, ax1 = plt.subplots(figsize=(len(areas) / 20, 5))

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

    plt.savefig(str(outpath), bbox_inches='tight')
    plt.close()

def saveRoughnessPlot(roughness_values, threshold_ops, outdir):

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(threshold_ops,roughness_values)

    ax1.axvline(get_min_roughness_threshold(roughness_values, threshold_ops))

    ax1.set_title(str(outdir.parent.name))

    plt.savefig(str(outdir / '{}_roughnessPlot.png'.format(outdir.parent.name)), bbox_inches='tight')
    plt.close()


def downturnFinder(init_movie, refactoryPeriod, lowerThresh, numberOfConsecutiveDrops, peak2InflectionDiff, peak2TroughDiff, DEBUG = False):

    i = 0
    numFiles = len(init_movie)

    print('searching for peaks (downturnfinder) on {} number of files'.format(numFiles))

    binary_areas = getBinaryAreas(init_movie, lowerThresh)

    peakIndicies = []

    # initializes lists with 'numberOfConsecutiveDrops' of files
    def reinitializeTestFramesAndAreas(j):
        testFrames = []  # this list should never be more than 5 entries long, ex. [51, 52, 53, 54, 55]
        testAreas = []  # this list should never be more than 5 entries long, ex. [253, 255, 256, 255, 255]

        while len(testFrames) < numberOfConsecutiveDrops and j < numFiles:
            area = binary_areas[j]

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

            area = binary_areas[i]

            testFrames.append(i)
            testAreas.append(area)
            i += 1

    return peakIndicies


def get_roughness_value(np_area_list):
    area_list = np_area_list.tolist()
    roughness_value = 0
    range_of_list = max(area_list) - min(area_list)
    scalar = np.power(range_of_list, 2)

    while len(area_list) >= 4:
        a = area_list[0]
        b = area_list[1]
        c = area_list[2]
        d = area_list[3]
        jerk = (a - (3 * b) + (3 * c) - d)
        normalized_jerk = np.abs(jerk) / scalar
        roughness_value += np.abs(normalized_jerk)
        area_list.pop(0)
    return roughness_value

def polynomialAdjustment(thresh, power=4, center=0.2, amplitude=1500):
    return amplitude*pow(thresh-center, power) + 1

def get_area_array(init_movie_np, threshold_ops):
    area_array = []

    # rows are lists of image areas
    # each row is tied to a specific threshold

    for i in range(len(threshold_ops)):
        area_array.append(getBinaryAreas(init_movie_np, threshold_ops[i]))

    return area_array


def get_roughness_list(area_array):
    roughness_values = []

    for i in range(len(area_array)):
        roughness_value = get_roughness_value(area_array[i])

        roughness_values.append(roughness_value)

    return roughness_values

def get_roughness_list_adj(area_array, threshold_options):
    roughness_values = []

    for i in range(len(threshold_options)):
        roughness_value = get_roughness_value(area_array[i])
        roughness_value_adj = roughness_value*polynomialAdjustment(threshold_options[i])
        # debugging line below
        # print('thresh: {:04}, rv: {:04}, adj rv: {:04}, ratio: {:04}'.format(threshold_options[i], roughness_value, roughness_value_adj, roughness_value_adj/roughness_value))
        roughness_values.append(roughness_value_adj)

    return roughness_values


def get_init_movie(frames):

    movie_frame_list = []

    for frame in frames:
        movie_frame_list.append(im.getJellyGrayImageFromFile(frame))

    init_movie_np = np.array(movie_frame_list)

    return init_movie_np


def get_min_roughness_threshold(roughness_list, threshold_ops):
    return threshold_ops[roughness_list.index(min(roughness_list))]


def autoLowerThreshold(init_movie, threshold_ops = [x / 1000 for x in range(60, 350, 5)], roughness_saveOut_dir = None):

    area_array = get_area_array(init_movie, threshold_ops)

    roughness_list = get_roughness_list_adj(area_array, threshold_ops)

    if roughness_saveOut_dir:
        saveRoughnessPlot(roughness_list, threshold_ops, roughness_saveOut_dir)

    return get_min_roughness_threshold(roughness_list, threshold_ops)


def selectInflectionThresholdandDiff(peaksOnBinaryImage, init_movie, recordingName, peak2InflectionDiff, peak2TroughDiff, initializationOutputDir, angleArrImageDir, centroidDir, dynamicRangeDir):

    # make directory to store verification jelly plots
    postInflectionDiffCases = list(range(4, 14))
    thresholdCases = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

    # out data: pulse angles x testDiffs
    angleData = np.empty((len(peaksOnBinaryImage), len(postInflectionDiffCases), len(thresholdCases)))  # 3D array
    angleData[:] = np.nan

    otherDataCols = np.array(['relaxed', 'peak', 'trough', 'by eye', ''])
    otherData = np.empty((len(peaksOnBinaryImage), len(otherDataCols)))
    otherData[:] = np.nan

    # read in by eye angle measurements
    byEyeAngleDF = pd.DataFrame(peaksOnBinaryImage, columns=['peaks'])
    byEyeAngleDF['by eye measurement (0 to 360)'] = np.nan
    byEyeAngleDFioPath = initializationOutputDir / '{}_byEyeAngles.csv'.format(recordingName)
    byEyeAngleDF.to_csv(str(byEyeAngleDFioPath), index=False)

    for i, peak in enumerate(peaksOnBinaryImage):
        troughImg = init_movie[peak + peak2TroughDiff]
        relaxedImg = init_movie[peak + peak2InflectionDiff]

        centroidDiff = im.getGrayscaleImageDiff_absolute(troughImg, relaxedImg)
        binaryCentroidDiff = im.getBinaryJelly(centroidDiff, lower_bound=0.05)
        centroidRegion = im.findJellyRegion(binaryCentroidDiff)
        centroid = im.findCentroid_boundingBox(centroidRegion)

        centroidVerOutFile = centroidDir / 'centroid for {} - {:03}.png'.format(recordingName, peak + peak2InflectionDiff)
        im.saveJellyPlot(im.getCentroidVerificationImg(centroidDiff, binaryCentroidDiff, centroid), centroidVerOutFile)

        peakImg = init_movie[peak]
        peakDiff = im.getGrayscaleImageDiff_absolute(troughImg, peakImg)
        binaryPeakDiff = im.getBinaryJelly(peakDiff, lower_bound=0.05)
        averagedDynamicRangeMaskedImg = im.dynamicRangeImg_AreaBased(relaxedImg, binaryPeakDiff, 5)

        dynamicRangeImgOutfile = dynamicRangeDir / 'dynamicRangeImg_{:03}.png'.format(peak + peak2InflectionDiff)

        im.saveJellyPlot(averagedDynamicRangeMaskedImg, dynamicRangeImgOutfile)

        # dealing with inflection thresholding
        testDiffImages = []
        for j in postInflectionDiffCases:
            testImg = init_movie[peak + peak2InflectionDiff + j]
            testDiff = im.getGrayscaleImageDiff_absolute(testImg, relaxedImg)
            normalizedTestDiff = testDiff / averagedDynamicRangeMaskedImg
            testDiffImages.append(normalizedTestDiff)

        testingOutfile = angleArrImageDir / 'testPlot for {} - {:03}.png'.format(recordingName, peak + peak2InflectionDiff)
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
    def runSDanalysis():
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

        return inflectionTestBinaryThreshold, inflectionTestDiff, sdTable, sdTableMinIndex

    inflectionTestBinaryThreshold, inflectionTestDiff, sdTable, sdTableMinIndex = runSDanalysis()

    if CHIME: dm.chime(MAC, 'input time')
    while True:
        print('thresholding options: {}'.format(thresholdCases))
        print('test frame options: {}'.format(postInflectionDiffCases))
        np.set_printoptions(threshold=np.inf)
        print(np.asarray(sdTable))
        print('index of min sd: {}'.format(sdTableMinIndex))
        print('selected sd: {}'.format(sdTable[thresholdCases.index(inflectionTestBinaryThreshold)][
                                           postInflectionDiffCases.index(inflectionTestDiff)]))

        print('Params to change: ')
        print('select \'1\' to change {} which is {}'.format('inflectionTestBinaryThreshold',
                                                             inflectionTestBinaryThreshold))
        print('select \'2\' to change {} which is {}'.format('inflectionTestDiff', inflectionTestDiff))
        print('select \'3\' to update by eye measurements')
        print('or \'4\' to continue.')

        selectionVar = dm.getSelection([1, 2, 3, 4])
        if selectionVar == '1':
            inflectionTestBinaryThreshold = float(dm.getSelection(thresholdCases))
        elif selectionVar == '2':
            inflectionTestDiff = int(dm.getSelection(postInflectionDiffCases))
        elif selectionVar == '3':
            inflectionTestBinaryThreshold, inflectionTestDiff, sdTable, sdTableMinIndex = runSDanalysis()
        else:
            break

    chosenSD = sdTable[thresholdCases.index(inflectionTestBinaryThreshold)][postInflectionDiffCases.index(inflectionTestDiff)]

    return inflectionTestDiff, inflectionTestBinaryThreshold, chosenSD


def initialization_Main(pathOfPreInitializationDF, pathOfInitializationStack, recordingHomeDir, macintosh):
    global MAC
    MAC = macintosh


    preInitializationDf = pd.read_csv(str(pathOfPreInitializationDF))
    initializationStack = dm.getFrameFilePaths(pathOfInitializationStack)

    print(len(initializationStack))

    # static variables across single recording that must be read in from df
    recordingName = preInitializationDf.iloc[0]['RecordingName']
    initializationOutputDir = dm.makeOutDir(recordingHomeDir, 'InitializationVerification')
    framerate = preInitializationDf.iloc[0]['FrameRate']
    framesInRecording = sum(preInitializationDf['NumFramesInChunk'].tolist())  # not saved in final DF
    lengthOfRecording = timedelta(0, framesInRecording/framerate)

    # static variables across single recording that must be initialized
    lowerThreshold = None
    peak2InflectionDiff = -15
    peak2TroughDiff = 30
    postPeakRefractoryPeriod = 40
    inflectionTestDiff = None
    inflectionTestBinaryThreshold = None
    chosenSD = None   # not saved in final DF
    numConsecutiveDrops = 10

    # static variables across all recordings
    movementThreshold4reinitialization = 20
    movementThreshold2KeepMoving = 5
    movementThreshold4newNormalizationImg = 5
    pct2skip4RefractionaryPeriod = 2 / 5
    numFramesForParamInitialization = 3200 # 120 * 30
    numFrames2ConfirmStationary = 7200  # 120 * 60, 60 seconds of recording to stay stationary (just for testing)

    thresholdingDir = dm.makeOutDir(initializationOutputDir, '{}_ThresholdingPlots'.format(recordingName))
    angleArrImageDir = dm.makeOutDir(initializationOutputDir, '{}_AngleArrImageDir'.format(recordingName))
    centroidDir = dm.makeOutDir(initializationOutputDir, '{}_CentroidVerificationDir'.format(recordingName))
    dynamicRangeDir = dm.makeOutDir(initializationOutputDir, '{}_dynamicRangeVerificationDir'.format(recordingName))
    areaPlotOutpath = initializationOutputDir / 'areaVerificationPlot.jpg'

    def saveVariableParams():
        imp_parameters = pd.DataFrame(np.array([
            [recordingName, 'the name of the recording being processed'],
            [initializationOutputDir,
             'place where all the initialization verification images and directories are stored'],
            [framerate, 'framerate of the specified recording'],
            [framesInRecording, 'number of total frames in recording'],
            [lengthOfRecording, 'length of recording in timedelta format'],

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

    if DEBUG: print('loading initialization stack\n')
    init_movie = get_init_movie(initializationStack)

    if DEBUG: print('calculating lowerThreshold\n')
    lowerThreshold = autoLowerThreshold(init_movie, roughness_saveOut_dir = initializationOutputDir)

    saveVariableParams()

    if DEBUG: print('getting BinaryAreas\n')
    binaryImageAreas = getBinaryAreas(init_movie, lowerThreshold)

    if DEBUG: print('getting peaksOnBinaryImage\n')
    # gets peak frame nums from binaryImageAreas
    peaksOnBinaryImage = downturnFinder(init_movie, postPeakRefractoryPeriod, lowerThreshold, numConsecutiveDrops, peak2InflectionDiff, peak2TroughDiff)

    saveAreasPlot(binaryImageAreas, peaksOnBinaryImage, areaPlotOutpath,
                  [peak2InflectionDiff, peak2InflectionDiff + 5, peak2TroughDiff],
                  postPeakRefractoryPeriod)

    if DEBUG: print('peaks: {}\n'.format(peaksOnBinaryImage))

    for peak in peaksOnBinaryImage:
        init_movie_binary = init_movie > lowerThreshold

        thresholdingImgOutfile = thresholdingDir / 'thresholdingImg_{:03}.png'.format(peak)

        im.saveJellyPlot(init_movie_binary[peak], thresholdingImgOutfile)

    # gets peak2TroughDiff from peaksOnBinaryImage and binaryImageAreas
    troughsOnBinaryImage = dm.getTroughs(binaryImageAreas)


    if DEBUG: print('troughs: {}'.format(troughsOnBinaryImage))

    peak2TroughDiff = dm.likelyPostPeakTroughDiff(troughsOnBinaryImage, peaksOnBinaryImage)

    # initializes inflection diff from jellyRegionAreas
    peak2InflectionDiff = dm.getLikelyInflectionDiff(binaryImageAreas, peaksOnBinaryImage)

    # initializes frames to skip for analysis after each peak
    averageIPI = dm.averageInterpulseInterval(peaksOnBinaryImage)
    postPeakRefractoryPeriod = int(pct2skip4RefractionaryPeriod * averageIPI)

    if CHIME: dm.chime(MAC, 'input time')
    while True:

        saveAreasPlot(binaryImageAreas, peaksOnBinaryImage, areaPlotOutpath,
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

            peaksOnBinaryImage = downturnFinder(init_movie, postPeakRefractoryPeriod, lowerThreshold, numConsecutiveDrops, peak2InflectionDiff, peak2TroughDiff)
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

    inflectionTestDiff, inflectionTestBinaryThreshold, chosenSD = selectInflectionThresholdandDiff(peaksOnBinaryImage, init_movie, recordingName, peak2InflectionDiff, peak2TroughDiff, initializationOutputDir, angleArrImageDir, centroidDir, dynamicRangeDir)

    saveVariableParams()

    # static params for each chunk (fraction over overall video recording ex. xaa, xab, xac, etc.)
    lastFrameOfPreviousChunk = 0  # calculated from framesInChunk

    postInitiationDF = preInitializationDf.copy()

    # static params for each recording
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