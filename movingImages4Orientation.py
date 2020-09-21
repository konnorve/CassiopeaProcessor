

from pathlib import Path
import shutil
import os



def makeOutDir(outputDir, folderName, DEBUG=False):
    outdir = outputDir / folderName
    if not (outdir.exists()):
        outdir.mkdir()
        if DEBUG: print('dir %s made' % outdir)
    else:
        if DEBUG: print('dir %s already exists' % outdir)
    return outdir

segmentDir = Path('/Users/kve/Desktop/Clubs/Harland_Lab/Round_9/testdata4dataprocessingscript/20200706_Beyonce_755pm_cam1_1/SegmentData')

globalOrientationDir = Path('/Users/kve/Desktop/Clubs/Harland_Lab/Round_9/testdata4dataprocessingscript/20200706_Beyonce_755pm_cam1_1/OrientationDirGlobal')

recordingName = '20200706_Beyonce_755pm_cam1_1'

for segment in [dir for dir in sorted(segmentDir.iterdir()) if dir.name != '.DS_Store']:

    globalmovementSegment = int(str(segment.stem)[len(recordingName)+1:])

    print(globalmovementSegment)

    segmentName = '{}_{}'.format(recordingName, globalmovementSegment)
    orientaitonLocalDir = makeOutDir(segment, '{}_RelaxedFramesForOrientation'.format(segmentName))

    imgPath4Orientation = [file for file in sorted(orientaitonLocalDir.iterdir()) if file.suffix == '.png'][0]

    shutil.copy(str(imgPath4Orientation), str(globalOrientationDir))

    copiedFile = globalOrientationDir / imgPath4Orientation.name

    newName = globalOrientationDir / "{}.png".format(globalmovementSegment)

    os.rename(str(copiedFile), str(newName))