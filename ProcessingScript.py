import AnalysisMethods as am

from pathlib import Path

import os

import DataMethods as dm

videoImageStackDir = Path(
    '/Users/kve/Desktop/Clubs/Harland_Lab/Round_8/Test_Videos_and_Stacks/Videos_3_stacks/Beyonce_test_stacks')

angleOutputDir = Path('/Users/kve/Desktop/Clubs/Harland_Lab/Round_8/thresholdingAndAutomationTesting_July30/AngleData')

segmentVerOutputDir = Path('/Users/kve/Desktop/Clubs/Harland_Lab/Round_8/thresholdingAndAutomationTesting_July30/SegmentData')

recordingName = 'beyonceTesting4thresholding'

chunks = [file for file in sorted(videoImageStackDir.iterdir()) if file.name != '.DS_Store']

print(angleOutputDir)
print(segmentVerOutputDir)
[print(chunk) for chunk in chunks]


pathOfPreviousParamDF = None
questionablyStationary = False
lastStationaryCentroid = None

for chunk in chunks:
    pathOfPreviousParamDF, questionablyStationary, lastStationaryCentroid = am.runFullVideoAnalysis(  vidImgStkDir=chunk,
                                                                            verOutDir=segmentVerOutputDir,
                                                                            angOutDir=angleOutputDir,
                                                                            recName=recordingName,
                                                                            pathOfLastParamDF=pathOfPreviousParamDF,
                                                                            isPotentiallyStationary=questionablyStationary,
                                                                            lstStatCentroid=lastStationaryCentroid,
                                                                            macintosh=True
                                                                            )
