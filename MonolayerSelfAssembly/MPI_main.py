import os, sys
import h5py
import cv2
import numpy
import gc
import mpi4py
import matplotlib.pyplot as plt
from scipy import ndimage
from mpi4py import MPI
from time import time

sys.path.append(os.path.abspath('../myFunctions'))
import fileIO
import imageProcess
import myCythonFunc
import dataViewer
import misc

#######################################################################
# USER INPUTS
#######################################################################
inputFile = '/home/uanand/images/utkarsh/Guanhua/MonolayerSelfAssembly/38/38-crop_gray.avi'
outputFile = '/home/uanand/images/utkarsh/Guanhua/MonolayerSelfAssembly/38/38-crop.h5'
inputDir = '/home/uanand/images/utkarsh/Guanhua/MonolayerSelfAssembly/38'
outputDir = '/home/uanand/images/utkarsh/Guanhua/MonolayerSelfAssembly/38/output'
pixInNM = 1.45
fps = 100
microscope = 'JOEL2010' #'JOEL2010','T12'
camera = 'One-view' #'Orius', 'One-view'
owner = 'Guanhua'

zfillVal = 6
fontScale = 0.5
structure = [[1,1,1],[1,1,1],[1,1,1]]
#######################################################################


#######################################################################
# INITIALIZATION FOR THE MPI ENVIRONMENT
#######################################################################
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#######################################################################


#######################################################################
# DATA PROCESSSING
# 1. READ THE INPUT FILES AND STORE THEM FRAME-WISE IN H5 FILE
# 2. PERFORM BACKGROUND SUBTRACTION (IF REQUIRED)
#######################################################################
#########
# PART 1
#########
if (rank==0):
    fp = fileIO.createH5(outputFile)
    
    [gImgRawStack,row,col,numFrames] = fileIO.readAVI(inputFile)
    #[gImgRawStack,row,col,numFrames] = fileIO.readImageSequence(folder,frameList=range(1,134))
    #[gImgRawStack,row,col,numFrames] = fileIO.readDM4Sequence(folder)
    
    frameList = range(1,numFrames+1)
    for frame in frameList:
        fp.create_dataset('/dataProcessing/gImgRawStack/'+str(frame).zfill(zfillVal), data=gImgRawStack[:,:,frame-1], compression='gzip', compression_opts=9)
        
    fp.attrs['inputFile'] = inputFile
    fp.attrs['outputFile'] = outputFile
    fp.attrs['inputDir'] = inputDir
    fp.attrs['outputDir'] = outputDir
    fp.attrs['pixInNM'] = pixInNM
    fp.attrs['pixInAngstrom'] = pixInNM*10
    fp.attrs['fps'] = fps
    fp.attrs['microscope'] = microscope
    fp.attrs['camera'] = camera
    fp.attrs['owner'] = owner
    fp.attrs['row'] = row
    fp.attrs['col'] = col
    fp.attrs['numFrames'] = numFrames
    fp.attrs['frameList'] = range(1,numFrames+1)
    fp.attrs['zfillVal'] = zfillVal
    
    fileIO.mkdirs(outputDir)
    fileIO.saveImageSequence(gImgRawStack,outputDir+'/dataProcessing/gImgRawStack')
    
    fp.flush(), fp.close()
    del gImgRawStack
    gc.collect()
comm.Barrier()

#########
# PART 2
#########
if (rank==0):
    print "Inverting the image and performing background subtraction"
invertFlag=True
bgSubFlag=True; bgSubSigmaTHT=2; radiusTHT=15

if (rank==0):
    fp = h5py.File(outputFile, 'r+')
else:
    fp = h5py.File(outputFile, 'r')
[row,col,numFrames,frameList] = misc.getVitals(fp)
procFrameList = numpy.array_split(frameList,size)

for frame in procFrameList[rank]:
    gImgProc = fp['/dataProcessing/gImgRawStack/'+str(frame).zfill(zfillVal)].value
    if (invertFlag==True):
        gImgProc = imageProcess.invertImage(gImgProc)
    if (bgSubFlag==True):
        gImgProc = imageProcess.subtractBackground(gImgProc, sigma=bgSubSigmaTHT, radius=radiusTHT)
    cv2.imwrite(outputDir+'/dataProcessing/processedStack/'+str(frame).zfill(zfillVal)+'.png',gImgProc)

comm.Barrier()
    
if (rank==0):
    for frame in frameList:
        gImgProc = cv2.imread(outputDir+'/dataProcessing/processedStack/'+str(frame).zfill(zfillVal)+'.png',0)
        fp.create_dataset('/dataProcessing/processedStack/'+str(frame).zfill(zfillVal), data=gImgProc, compression='gzip', compression_opts=9)
fp.flush(), fp.close()
comm.Barrier()
#######################################################################



#######################################################################
# IMAGE SEGMENTATION
#######################################################################
#if (rank==0):
    #print "Performing segmentation for all the frames"
    
#fp = h5py.File(outputFile, 'r')
#[row,col,numFrames,frameList] = misc.getVitals(fp)
#procFrameList = numpy.array_split(frameList,size)

#areaRange = numpy.array([60,500], dtype='float64')
#circularityRange = numpy.array([0.85,1], dtype='float64')
#sigma = 1

#for frame in procFrameList[rank]:
    #gImgRaw = fp['/dataProcessing/gImgRawStack/'+str(frame).zfill(zfillVal)].value
    #gImgNorm = imageProcess.normalize(gImgRaw,min=0,max=230)
    #gImgProc = fp['/dataProcessing/processedStack/'+str(frame).zfill(zfillVal)].value
    #bImgKapur = gImgProc>=myCythonFunc.threshold_kapur(gImgProc.flatten())
    
    #gImgInv = 255-gImgRaw
    #gImgBlur = ndimage.gaussian_filter(gImgInv, sigma=sigma)
    #bImgAdaptive = cv2.adaptiveThreshold(gImgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 0).astype('bool')
    
    #bImg = numpy.logical_and(bImgKapur,bImgAdaptive)
    #bImg = imageProcess.fillHoles(bImg)
    #bImg = myCythonFunc.removeBoundaryParticles(bImg.astype('uint8'))
    #bImg = myCythonFunc.areaThreshold(bImg.astype('uint8'), areaRange=areaRange)
    #bImg = myCythonFunc.circularThreshold(bImg.astype('uint8'), circularityRange=circularityRange)
    #bImg = imageProcess.convexHull(bImg)
    
    #bImgBdry = imageProcess.normalize(imageProcess.boundary(bImg))
    #finalImage = numpy.column_stack((numpy.maximum(gImgNorm,bImgBdry), gImgNorm))
    #cv2.imwrite(outputDir+'/segmentation/result/'+str(frame).zfill(zfillVal)+'.png', finalImage)
#fp.flush(), fp.close()
#comm.Barrier()
#######################################################################


#######################################################################
# CREATE BINARY IMAGES INTO HDF5 FILE
#######################################################################
#if (rank==0):
    #print "Creating binary images and writing into h5 file"
    
#if (rank==0):
    #fp = h5py.File(outputFile, 'r+')
#else:
    #fp = h5py.File(outputFile, 'r')
#[row,col,numFrames,frameList] = misc.getVitals(fp)
#procFrameList = numpy.array_split(frameList,size)

#for frame in procFrameList[rank]:
    #bImg = cv2.imread(outputDir+'/dataProcessing/processedStack/'+str(frame).zfill(zfillVal)+'.png',0)[0:row,0:col]
    #bImg = bImg==255
    #bImg = bImg = imageProcess.fillHoles(bImg)
    #numpy.save(outputDir+'/segmentation/result/'+str(frame).zfill(zfillVal)+'.npy', bImg)
   
#comm.barrier()
#if (rank==0):
    #for frame in frameList:
        #bImg = numpy.load(outputDir+'/segmentation/result/'+str(frame).zfill(zfillVal)+'.npy')
        #fp.create_dataset('/segmentation/bImgStack/'+str(frame).zfill(zfillVal), data=bImg, compression='gzip', compression_opts=9)
        #fileIO.delete(outputDir+'/segmentation/result/'+str(frame).zfill(zfillVal)+'.npy')
        
#fp.flush(), fp.close()
#comm.Barrier()
#######################################################################


#######################################################################
# LABELLING PARTICLES
#######################################################################
#if (rank==0):
	#print "LABELLING PARTICLES"
    #fp = h5py.File(outputFile, 'r+')
    #[row,col,numFrames,frameList] = misc.getVitals(fp)
	
	#centerDispRange = [5,5]
	#perAreaChangeRange = [1,2]
	#missFramesTh = 10
    
    #maxID, occurenceFrameList = myPythonFunc.labelParticles(bImgDir, gImgDir, labelDataDir, labelImgDir, row, col, numFrames, centerDispRange, perAreaChangeRange, missFramesTh, frameList, structure, fontScale=fontScale)
    
	#labelImgDir = inputDir+'/output/images/segmentation/tracking'
	
	#maxID, occurenceFrameList = myPythonFunc.labelParticles(bImgDir, gImgDir, labelDataDir, labelImgDir, row, col, numFrames, centerDispRange, perAreaChangeRange, missFramesTh, frameList, structure, fontScale=fontScale)
	#metaData['particleList'] = range(1,maxID+1)
	#print zip(metaData['particleList'],occurenceFrameList)
	#pickle.dump(metaData, open(inputDir+'/metaData', 'wb'))
##############################################################
