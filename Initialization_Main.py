import InitiatlizationMethods as init
from pathlib import Path
import DataMethods as dm

# should all be in the same recording directory from Savio (recordingOutputDir)
recordingHomeDir = Path('/home/kve/Desktop/Labora/Harland_Lab/2021-04/TroughBasedVP/Sample-Initalizations/20200901_Shakira_432pm_cam1_1_Peak_Based')

#automatic procurement from home directories if labeled right
pathOfPreInitializationDFDir = recordingHomeDir / 'Initialization_DF'
pathOfPreInitializationDF = dm.getCSVFilePaths(pathOfPreInitializationDFDir)[0]
pathOfInitializationStack = recordingHomeDir / 'Initialization_Stack'

print(recordingHomeDir.name)

init.initialization_Main(pathOfPreInitializationDF, pathOfInitializationStack, recordingHomeDir, True)


# TODO: add script that creates an orientation CSV after all the video processing chunks run. Should be after all the processes are joined and on Savio-HTC