from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import DataMethods as dm

import ImageMethods as im

import datetime

import InitiatlizationMethods as init

import shutil

########################################################################################################################
# **** GLOBAL VARIABLES ****

DEBUG = True

CONFIRMATIONIMAGES = False
confirmationImagesPath = Path('')

########################################################################################################################


def saveSegmentVariableParams():
    imp_parameters = pd.DataFrame(np.array([
        [recordingName, 'the name of the recording being processed'],
        [recordingHomeDirPath, 'home path on savio for all recording out information'],
        [framerate,  'framerate that the video was run at'],

        [videoImageStackDir, 'where the image stack for this specific chunk is located'],
        [chunkName, 'the name of this jelly and its video chunk id'],

        [averageTroughBinaryArea, 'trough area to use in setting lower threshold automatically'],
        [peak2InflectionDiff, 'the number of frames past the peak where the inflection point occurs (this should always be negative)'],
        [peak2TroughDiff, 'the number of frames past the peak where the lowest area is found on average'],
        [postPeakRefractoryPeriod, 'the number of frames to preclude from analysis'],
        [inflectionTestDiff, 'the number of frames after inflection point where the difference in calculated'],
        [inflectionTestBinaryThreshold, 'the ideal threshold to locate the area of difference'],
        [numConsecutiveDrops, 'the number of consecutive drops needed to count something as a downturn'],

        [movementThreshold4reinitialization, 'number of pixels from one centroid to another to consider a jelly as moving.'],
        [movementThreshold2KeepMoving, 'number of pixels from one centroid to the next to continue to be considered moving'],
        [movementThreshold4newNormalizationImg, 'number of pixels from one centroid to another to reinitialize the average normalization img'],
        [numFramesForParamInitialization, 'number of frames to use when initializing params for a new segment'],
        [numFrames2ConfirmStationary, 'number of frames after first stationary frame after movement to confirm jellyfish is stationary'],

        [lastFrameOfPreviousChunk, 'the last frame the occured before this recording chunk'],
        [framesInChunk, 'absolute number of frames in specific video chunk'],

        [movementSegment, 'the latest segment number across all video chunks (segment = non-moving period)'],
        [lowerThreshold, 'lower threshold to create binary image of jelly to assess area (for downturns)'],

        [angleOutputDir, 'place where all the angle data (IMPORTANT DATA) is stored'],
        [segmentDir, 'place where all the chunk.movement segment data is stored'],
        [orientationDir, 'place where orientation frames for each chunk.movement segment is stored'],
        [verificationDir, 'place where image stacks are stored for future verification of initiation site'],
        [pathOfCurrentParamDF, 'path of the latest param dataframe for the latest segment'],
        [segmentVerificationDir, 'path of the latest chunk.movement segment initiation/verification info is stored'],

        [startTimeOfAnalysis, 'start time of data analysis on Savio'],
        [elapsedTime, 'elapsed time since start of chunk analysis to when that movement segment is completed'],
    ]),

        index=[
            'recordingName',
            'recordingHomeDirPath',
            'framerate',

            'videoImageStackDir',
            'chunkName',

            'averageTroughBinaryArea',
            'peak2InflectionDiff',
            'peak2TroughDiff',
            'postPeakRefractoryPeriod',
            'inflectionTestDiff',
            'inflectionTestBinaryThreshold',
            'numConsecutiveDrops',

            'movementThreshold4reinitialization',
            'movementThreshold2KeepMoving',
            'movementThreshold4newNormalizationImg',
            'numFramesForParamInitialization',
            'numFrames2ConfirmStationary',

            'lastFrameOfPreviousChunk',
            'framesInChunk',

            'movementSegment',
            'lowerThreshold',

            'angleOutputDir',
            'segmentDir',
            'orientationDir',
            'verificationDir',
            'pathOfCurrentParamDF',
            'segmentVerificationDir',

            'startTimeOfAnalysis',
            'elapsedTime'],

        columns=['data', 'notes'])

    imp_parameters.to_csv(str(pathOfCurrentParamDF))


def reinitializeElapsedTime():
    global elapsedTime
    elapsedTime = datetime.datetime.now() - startTimeOfAnalysis
    pass

def initialize_params(files, startingFrameNum):

    global lowerThreshold
    global pathOfCurrentParamDF
    global segmentVerificationDir
    global movementSegment

    # TODO: save out starting frame number of each movement segment!!!! Easiser to track how far something is in analysis
    # TODO: save orientation frame with leading zeros up to 1000 (001,002,003..)

    reinitializeElapsedTime()
    saveSegmentVariableParams()

    if startingFrameNum != 0:
        movementSegment += 1

    if startingFrameNum + numFramesForParamInitialization < framesInChunk:
        fileSubset = files[startingFrameNum: startingFrameNum + numFramesForParamInitialization]
    else:
        fileSubset = files[startingFrameNum:]

    # create segment directory named with global movement segment
    segmentName = '{}_{}_{:03}'.format(recordingName, chunkName, movementSegment)
    segmentVerificationDir = dm.makeOutDir(segmentDir, segmentName)

    pathOfCurrentParamDF = segmentVerificationDir / '{}_params.csv'.format(segmentName)

    saveSegmentVariableParams()

    thresholdingDir = dm.makeOutDir(segmentVerificationDir, '{}_ThresholdingPlots'.format(segmentName))
    plotDir = dm.makeOutDir(segmentVerificationDir, '{}_AngleVerificationPlots'.format(segmentName))
    centroidDir = dm.makeOutDir(segmentVerificationDir, '{}_CentroidVerificationPlots'.format(segmentName))
    segmentOrientationDir = dm.makeOutDir(segmentVerificationDir, '{}_RelaxedFramesForOrientation'.format(segmentName))
    dynamicRangeDir = dm.makeOutDir(segmentVerificationDir, '{}_dynamicRangeNormalizationImages'.format(segmentName))

    peaksOnBinaryImage = init.downturnFinder(fileSubset, postPeakRefractoryPeriod, lowerThreshold, numConsecutiveDrops, peak2InflectionDiff, peak2TroughDiff)

    # complete automated area thresholding based on averageTroughBinaryArea
    lowerThreshold = init.autoLowerThreshold(averageTroughBinaryArea, peak2TroughDiff, peaksOnBinaryImage, fileSubset, thresholdingDir, recordingName)

    saveSegmentVariableParams()

    # get areas of jellies both the region and the whole value true in binary image.
    binaryImageAreas = init.getBinaryAreas(fileSubset, lowerThreshold)
    peaksOnBinaryImage = init.downturnFinder(fileSubset, postPeakRefractoryPeriod, lowerThreshold, numConsecutiveDrops, peak2InflectionDiff, peak2TroughDiff)

    plotOutpath = segmentVerificationDir / '{}_areaPlot.png'.format(segmentName)
    init.saveAreasPlot(binaryImageAreas, peaksOnBinaryImage, plotOutpath,
                     [peak2InflectionDiff, peak2InflectionDiff + inflectionTestDiff, peak2TroughDiff],
                     postPeakRefractoryPeriod)

    saveSegmentVariableParams()

    i = 0
    while i < len(peaksOnBinaryImage):
        if peaksOnBinaryImage[i] + peak2InflectionDiff < 0 or peaksOnBinaryImage[i] + peak2TroughDiff >= numFramesForParamInitialization:
            peaksOnBinaryImage.pop(i)
        else:
            i += 1

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

        if i == 0:
            orientationOutFile = orientationDir / '{}_{}.png'.format(chunkName, movementSegment)
            im.saveJellyPlot(relaxedImg, orientationOutFile, [centroid, (centroid[0], 15)])

        orientationOutFile = segmentOrientationDir / 'relaxedFrame_{}.png'.format(peak + peak2InflectionDiff)
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
    saveSegmentVariableParams()

    if DEBUG: print('saved parameter data')

    if DEBUG: print('finished saving outplots and angle verification at: {}'.format(segmentVerificationDir))

    return True


def differenceAngleFinder(files):

    i = 0

    # movement parameters
    firstStationaryAfterMovement = 0 # first stationary frame (i) after movement
    isMoving = False
    isQuestionablyStationary = False
    centroidBefore = None
    lastStationaryCentroid = None
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
        dataTitle = '{}_{:03}.csv'.format(chunkName, movementSegment)
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
                            # now there is confirmed time after initial stationary point

                            if firstStationaryAfterMovement == 0:
                                data = []
                            else:
                                # must mutate data to take out
                                data = data[:-pulseCountInQuestionablyStationary]
                                saveOutData()
                                data = []

                            i = firstStationaryAfterMovement

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
        print('{} error occured.'.format(error))
        print("index: {}, isMoving: {}, isQStat: {}, centroid: {}".format(i,
                                                                       isMoving,
                                                                       isQuestionablyStationary,
                                                                       str(centroid)))
        raise

    finally:
        saveOutData()


def runFullVideoAnalysis(chunkRow, postInitiationDFPath):
    # any parameters that are not imputed into the function are set automatically

    params_df = dm.readCSV2pandasDF(postInitiationDFPath)
    params_chunkRow = params_df.iloc[chunkRow]

    # creates global references to mutate below if different input parameters are specified
    # variable that are constant for entire recording
    global recordingName  # the name of the recording being processed
    global recordingHomeDirPath # home path on savio for all recording out information
    global framerate # framerate that the video was run at

    # variables that characterize chunk
    global videoImageStackDir  # where the image stack for this specific chunk is located
    global chunkName  # the name of this jelly and its video chunk id

    # initalized during local manual initiation steps
    global averageTroughBinaryArea  # trough area to use in setting lower threshold automatically
    global peak2InflectionDiff  # the number of frames past the peak where the inflection point occurs (this should always be negative)
    global peak2TroughDiff  # the number of frames past the peak where the lowest area is found on average
    global postPeakRefractoryPeriod  # the number of frames to preclude from analysis
    global inflectionTestDiff  # the number of frames after inflection point where the difference in calculated
    global inflectionTestBinaryThreshold  # the ideal threshold to locate the area of difference
    global numConsecutiveDrops  # the number of consecutive drops needed to count something as a downturn

    # static variables that rarely change across recordings
    global movementThreshold4reinitialization  # number of pixels from one centroid to another to consider a jelly as moving.
    global movementThreshold2KeepMoving  # number of pixels from one centroid to the next to continue to be considered moving
    global movementThreshold4newNormalizationImg  # number of pixels from one centroid to another to reinitialize the average normalization img
    global numFramesForParamInitialization  # number of frames to use when initializing params for a new segment
    global numFrames2ConfirmStationary  # number of frames after first stationary frame after movement to confirm jellyfish is stationary

    # automatically initialized parameters in initialization method
    global lastFrameOfPreviousChunk  # the last frame the occured before this recording chunk
    global framesInChunk  # absolute number of frames in specific video chunk
    
    # changes for each movement segment. reinitialized in each segment folder
    global movementSegment  # the latest segment number across all video chunks (segment = non-moving period)
    global lowerThreshold  # lower threshold to create binary image of jelly to assess area (for downturns)

    # directories and path objects for saving data
    global angleOutputDir  # place where all the angle data (IMPORTANT DATA) is stored
    global segmentDir  # place where all the chunk.movement segment data is stored
    global orientationDir  # place where orientation frames for each chunk.movement segment is stored
    global verificationDir  # place where image stacks are stored for future verification of initiation site
    global pathOfCurrentParamDF  # path of the latest param dataframe for the latest segment
    global segmentVerificationDir  # path of the latest chunk.movement segment initiation/verification info is stored

    # Savio specific information on running these chunks
    global startTimeOfAnalysis #start time of data analysis on Savio
    global elapsedTime #elapsed time since start of chunk analysis to when that data is completed

    # Initializing Savio specific information on running these chunks
    startTimeOfAnalysis = datetime.datetime.now()
    elapsedTime = None
    
    # initializing variables from param df
    # intitalizing variable that are constant for entire recording
    recordingName = params_chunkRow['RecordingName']
    recordingHomeDirPath = Path(params_chunkRow['RecordingDirPath'])
    framerate = params_chunkRow['FrameRate']

    # intitalizing variables that characterize chunk
    videoImageStackDir = Path(params_chunkRow['SavioChunkPath'])
    chunkName = params_chunkRow['ChunkName']
    
    # parameters that were manually initialized
    averageTroughBinaryArea = params_chunkRow['averageTroughBinaryArea']
    peak2InflectionDiff = params_chunkRow['peak2InflectionDiff']
    peak2TroughDiff = params_chunkRow['peak2TroughDiff']
    postPeakRefractoryPeriod = params_chunkRow['postPeakRefractoryPeriod']
    inflectionTestDiff = params_chunkRow['inflectionTestDiff']
    inflectionTestBinaryThreshold = params_chunkRow['inflectionTestBinaryThreshold']
    numConsecutiveDrops = params_chunkRow['numConsecutiveDrops']

    # intiailizing static variables that rarely change across recordings
    movementThreshold4reinitialization = params_chunkRow['movementThreshold4reinitialization']
    movementThreshold2KeepMoving = params_chunkRow['movementThreshold2KeepMoving']
    movementThreshold4newNormalizationImg = params_chunkRow['movementThreshold4newNormalizationImg']
    numFramesForParamInitialization = params_chunkRow['numFramesForParamInitialization']
    numFrames2ConfirmStationary = params_chunkRow['numFrames2ConfirmStationary']
    
    # parameters that were automatically initialized
    lastFrameOfPreviousChunk = params_chunkRow['lastFrameOfPreviousChunk']
    framesInChunk = params_chunkRow['NumFramesInChunk']

    # Parameters not from param_df
    # Parameters reinitialized at the start of each data segment
    lowerThreshold = None  # un-initialized
    movementSegment = 0

    # output paths and directory initialization
    angleOutputDir = dm.makeOutDir(recordingHomeDirPath, '{}_AngleData'.format(recordingName))
    segmentDir = dm.makeOutDir(recordingHomeDirPath, '{}_SegmentData'.format(recordingName))
    orientationDir = dm.makeOutDir(recordingHomeDirPath, '{}_OrientationDir'.format(recordingName))
    verificationDir = dm.makeOutDir(recordingHomeDirPath, '{}_VerificationStacks'.format(recordingName))
    segmentVerificationDir = None
    pathOfCurrentParamDF = None

    # loads in files from FFMPEG Stack located at 'videoImageStackDir'
    files = dm.getFrameFilePaths(videoImageStackDir)

    if chunkRow%4 == 0:
        chunkVerDir = dm.makeOutDir(verificationDir, chunkName)
        for file in files[:numFramesForParamInitialization]:
            shutil.copy(str(file), str(chunkVerDir))

    initialize_params(files, 0)

    # run analysis
    differenceAngleFinder(files)

    reinitializeElapsedTime()
    saveSegmentVariableParams()