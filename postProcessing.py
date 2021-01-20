from pathlib import Path

import pandas as pd

import numpy as np


def saveOutOrientationCSV(orientationDirPath):

    imgNames = [imgPath.stem for imgPath in sorted(orientationDirPath.iterdir()) if imgPath.suffix == '.png']

    dataRows = [(name[:name.rfind("_")], name[name.rfind("_")+1:])for name in imgNames]

    outFrame = pd.DataFrame(dataRows, columns=['chunk name', 'movement segment'])

    outFrame['orientation factor'] = np.nan

    outPath = orientationDirPath.parent / '{}_orientations_blank.csv'.format(orientationDirPath.name)

    outFrame.to_csv(outPath, index=False)



orientationDirPath = Path('/home/kve/Desktop/Labora/Harland_Lab/2021-1/Shakira_home/Shakira Orientation Dirs')

[saveOutOrientationCSV(dir) for dir in orientationDirPath.iterdir() if dir.is_dir()]

