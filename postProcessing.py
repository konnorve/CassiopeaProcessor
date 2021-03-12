from pathlib import Path

import pandas as pd

import numpy as np

DEBUG = True

def saveOutOrientationCSV(orientationDirPath):

    imgNames = [imgPath.stem for imgPath in sorted(orientationDirPath.iterdir()) if imgPath.suffix == '.png']

    dataRows = [(name[:name.rfind("_")], name[name.rfind("_")+1:])for name in imgNames]

    outFrame = pd.DataFrame(dataRows, columns=['chunk name', 'movement segment'])

    outFrame['orientation factor'] = np.nan

    outPath = orientationDirPath.parent / '{}_orientations_blank.csv'.format(orientationDirPath.name)

    outFrame.to_csv(outPath, index=False)

def saveOutVerificationCSV(angleDataPath):

    # initiate angle data dataframe from directory
    dfPaths = [dir for dir in sorted(angleDataPath.iterdir()) if dir.name != '.DS_Store']

    # simple DFs aka raw angle data are put together into a list and concatenated

    simpleDFs = []
    # segment angleData in to chunks based on naming conventions? ***clarify AJ***
    for i, dfPath in enumerate(dfPaths):

        # use pandas to read csv
        tempSimpleData = pd.read_csv(str(dfPath), header=0)

        # reads in the name of the angle data
        pathStem = dfPath.stem

        if DEBUG: print('pathStem: {}'.format(pathStem))

        # determines the movement segment of the data
        movementSegment = int(pathStem[pathStem.rindex('_') + 1:])

        # determines the name of the Chunk
        chunkName = pathStem[:pathStem.rindex('_')]

        # assigns columns equal to the chunk name and movement segment to the dataframe
        tempSimpleData['chunk name'] = chunkName
        tempSimpleData['movement segment'] = movementSegment

        # adds the angle data + its chunk name and movement segment to a list of similar dataframes to be concatenated
        simpleDFs.append(tempSimpleData)

    # concats all the dataframes into one pandas df
    simpleConcatDF = pd.concat(simpleDFs)

    # TODO: add verification column

    outPath = angleDataPath.parent / '{}_orientations_blank.csv'.format(orientationDirPath.name)


    

    simpleConcatDF.to_csv(outPath, index=False)


orientationDirPath = Path('/home/kve/Desktop/Labora/Harland_Lab/2021-1/Shakira_home/Shakira Orientation Dirs')

[saveOutOrientationCSV(dir) for dir in orientationDirPath.iterdir() if dir.is_dir()]

angleDirPath = Path('/home/kve/Desktop/Labora/Harland_Lab/2021-1/Shakira_home/Shakira Orientation Dirs')

# TODO: Check function

# [saveOutVerificationCSV(dir) for dir in angleDirPath.iterdir() if dir.is_dir()]

