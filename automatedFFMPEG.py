"""
Created to mass run FFMPEG on our recording folders which contian multiple file segments.
Author: Konnor von Emster

"""

import os

from pathlib import Path

DEBUG = True

def makeOutDir(outputDir, folderName):
    outdir = outputDir / folderName
    if not outdir.exists():
        outdir.mkdir()
    return outdir


def runFFMPEG(videoInPath, stackOutDir):
    try:
        # designed to work on windows, not mac
        os.system('ffmpeg -i {} -r 120 -q 0 {}/%14d.jpg'.format(str(videoInPath), str(stackOutDir)))
        return True
    except:
        return False

parentDir = Path('/Users/kve/Desktop/Clubs/Harland_Lab/Round_8/Test_Videos_and_Stacks')

videoDir = parentDir / 'Videos_3'

stackDir = makeOutDir(parentDir, '{}_stacks'.format(videoDir.name))

if DEBUG: print("parent dir: {}".format(parentDir))
if DEBUG: print("video dir: {}".format(videoDir))
if DEBUG: print("stack dir: {}".format(stackDir))

errors = []

for recording in sorted(videoDir.iterdir()):
    if recording.name != '.DS_Store':
        recording_stack_dir = makeOutDir(stackDir, '{}_stacks'.format(recording.name))

        if DEBUG: print("recording stack dir: {}".format(recording_stack_dir))

        for videoFile in sorted(recording.iterdir()):
            if videoFile.suffix == '.mp4' and videoFile.name != '.DS_Store':

                chunk_stack_dir = makeOutDir(recording_stack_dir, '{}_{}'.format(recording_stack_dir.name, videoFile.stem))

                worked = runFFMPEG(videoFile, chunk_stack_dir)

                if worked:
                    print('image stack successfully saved to: {}'.format(chunk_stack_dir))
                else:
                    print('ffmpeg failed on {}'.format(videoFile))
                    errors.append(videoFile)

print('FFMPEG errors on these videos: ')
[print(file) for file in errors]

