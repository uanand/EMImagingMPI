import cv2
import numpy
import os
import imageio
import h5py
import imageProcess

##############################################################
# GET SOME KEY INFROMATION FROM HDF5 FILE POINTER
##############################################################
def getVitals(fp):
    [row,col,numFrames] = fp.attrs['row'],fp.attrs['col'],fp.attrs['numFrames']
    frameList = fp.attrs['frameList']
    return row,col,numFrames,frameList
##############################################################
