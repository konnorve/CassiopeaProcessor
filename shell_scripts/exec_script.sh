#!/bin/bash

IMAGESTACKPATH=/tmp/Image_Stacks
FRAMERATE=120

mkdir -p $IMAGESTACKPATH/$2
ffmpeg -i $1 -r $FRAMERATE -q 0 $IMAGESTACKPATH/$2/%14d.jpg
