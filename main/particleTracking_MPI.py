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
import tracking

#######################################################################
# USER INPUTS
#######################################################################
inputFile = '/mnt/NAS-Drive/Utkarsh-Share/Guanhua/MonolayerSelfAssembly/38/38-crop-1-100.avi'
outputFile = '/mnt/NAS-Drive/Utkarsh-Share/Guanhua/MonolayerSelfAssembly/38/38-crop.h5'
inputDir = '/mnt/NAS-Drive/Utkarsh-Share/Guanhua/MonolayerSelfAssembly/38/'
outputDir = '/mnt/NAS-Drive/Utkarsh-Share/Guanhua/MonolayerSelfAssembly/38/output'
pixInNM = 1.45
fps = 100
microscope = 'JOEL2010' #'JOEL2010','T12'
camera = 'One-view' #'Orius', 'One-view'
owner = 'Guanhua'

zfillVal = 6
fontScale = 0.3
structure = [[1,1,1],[1,1,1],[1,1,1]]
#######################################################################


#######################################################################
# INITIALIZATION FOR THE MPI ENVIRONMENT
#######################################################################
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
#######################################################################

if (rank==0):
    tic = time()
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
        fileIO.writeH5Dataset(fp,'/dataProcessing/gImgRawStack/'+str(frame).zfill(zfillVal),gImgRawStack[:,:,frame-1])
        
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
    
    del gImgRawStack
    fp.flush(), fp.close()
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
        fileIO.writeH5Dataset(fp,'/dataProcessing/processedStack/'+str(frame).zfill(zfillVal),gImgProc)
        
fp.flush(), fp.close()
comm.Barrier()
#######################################################################


#######################################################################
# IMAGE SEGMENTATION
#######################################################################
if (rank==0):
    print "Performing segmentation for all the frames"
    
fp = h5py.File(outputFile, 'r')
[row,col,numFrames,frameList] = misc.getVitals(fp)
procFrameList = numpy.array_split(frameList,size)

areaRange = numpy.array([60,500], dtype='float64')
circularityRange = numpy.array([0.85,1], dtype='float64')
sigma = 1

for frame in procFrameList[rank]:
    gImgRaw = fp['/dataProcessing/gImgRawStack/'+str(frame).zfill(zfillVal)].value
    gImgNorm = imageProcess.normalize(gImgRaw,min=0,max=230)
    gImgProc = fp['/dataProcessing/processedStack/'+str(frame).zfill(zfillVal)].value
    bImgKapur = gImgProc>=myCythonFunc.threshold_kapur(gImgProc.flatten())
    
    gImgInv = 255-gImgRaw
    gImgBlur = ndimage.gaussian_filter(gImgInv, sigma=sigma)
    bImgAdaptive = cv2.adaptiveThreshold(gImgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 0).astype('bool')
    
    bImg = numpy.logical_and(bImgKapur,bImgAdaptive)
    bImg = imageProcess.fillHoles(bImg)
    bImg = myCythonFunc.removeBoundaryParticles(bImg.astype('uint8'))
    bImg = myCythonFunc.areaThreshold(bImg.astype('uint8'), areaRange=areaRange)
    bImg = myCythonFunc.circularThreshold(bImg.astype('uint8'), circularityRange=circularityRange)
    bImg = imageProcess.convexHull(bImg)
    
    bImgBdry = imageProcess.normalize(imageProcess.boundary(bImg))
    finalImage = numpy.column_stack((numpy.maximum(gImgNorm,bImgBdry), gImgNorm))
    cv2.imwrite(outputDir+'/segmentation/result/'+str(frame).zfill(zfillVal)+'.png', finalImage)
fp.flush(), fp.close()
comm.Barrier()
#######################################################################


#######################################################################
# CREATE BINARY IMAGES INTO HDF5 FILE
#######################################################################
if (rank==0):
    print "Creating binary images and writing into h5 file"
    
if (rank==0):
    fp = h5py.File(outputFile, 'r+')
else:
    fp = h5py.File(outputFile, 'r')
[row,col,numFrames,frameList] = misc.getVitals(fp)
procFrameList = numpy.array_split(frameList,size)

for frame in procFrameList[rank]:
    bImg = cv2.imread(outputDir+'/segmentation/result/'+str(frame).zfill(zfillVal)+'.png',0)[0:row,0:col]
    bImg = bImg==255
    bImg = imageProcess.fillHoles(bImg)
    numpy.save(outputDir+'/segmentation/result/'+str(frame).zfill(zfillVal)+'.npy', bImg)
   
comm.barrier()
if (rank==0):
    for frame in frameList:
        bImg = numpy.load(outputDir+'/segmentation/result/'+str(frame).zfill(zfillVal)+'.npy')
        fileIO.writeH5Dataset(fp,'/segmentation/labelStack/'+str(frame).zfill(zfillVal),bImg)
        fileIO.delete(outputDir+'/segmentation/result/'+str(frame).zfill(zfillVal)+'.npy')
        
fp.flush(), fp.close()
comm.Barrier()
#######################################################################


#######################################################################
# LABELLING PARTICLES
#######################################################################
centerDispRange = [5,5]
perAreaChangeRange = [1000,1000]
missFramesTh = 10
    
if (rank==0):
    print "Labelling segmented particles"
    fp = h5py.File(outputFile, 'r+')
    [row,col,numFrames,frameList] = misc.getVitals(fp)
    maxID, occurenceFrameList = tracking.labelParticles(fp, centerDispRange=centerDispRange, perAreaChangeRange=perAreaChangeRange, missFramesTh=missFramesTh, structure=structure)
    fp.attrs['particleList'] = range(1,maxID+1)
    numpy.savetxt(outputDir+'/frameOccurenceList.dat',numpy.column_stack((fp.attrs['particleList'],occurenceFrameList)),fmt='%d')
    fp.flush(), fp.close()
comm.Barrier()

if (rank==0):
    print "Generating images with labelled particles"
fp = h5py.File(outputFile, 'r')
tracking.generateImages(fp,outputDir+'/segmentation/tracking',fontScale,size,rank)
fp.flush(), fp.close()
comm.Barrier()
#######################################################################


#######################################################################
# REMOVING UNWANTED PARTICLES
#######################################################################
keepList = range(1,5)+range(6,57)
removeList = []

if (rank==0):
	print "Removing unwanted particles"

if (rank==0):
    fp = h5py.File(outputFile, 'r+')
else:
    fp = h5py.File(outputFile, 'r')
particleList = fp.attrs['particleList']
if not removeList:
	removeList = [s for s in particleList if s not in keepList]
tracking.removeParticles(fp,removeList,comm,size,rank)
fp.flush(), fp.close()    
comm.Barrier()
#######################################################################


#######################################################################
# GLOBAL RELABELING OF PARTICLES
#######################################################################
#correctionList = [[7,8,71]]

#if (rank==0):
	#print "Global relabeling of  particles"
    
#if (rank==0):
    #fp = h5py.File(outputFile, 'r+')
#else:
    #fp = h5py.File(outputFile, 'r')
#tracking.globalRelabelParticles(fp,correctionList,comm,size,rank)
#fp.flush(), fp.close()    
#comm.Barrier()
#######################################################################


#######################################################################
# FRAME-WISE CORRECTION OF PARTICLE LABELS
#######################################################################
#frameWiseCorrectionList = [\
#[range(2,5),[18,77]],\
#[range(1,10),[1,0]]\
#]

#if (rank==0):
    #print "Frame-wise relabeling of  particles"
    
#if (rank==0):
    #fp = h5py.File(outputFile, 'r+')
#else:
    #fp = h5py.File(outputFile, 'r')
#tracking.framewiseRelabelParticles(fp,frameWiseCorrectionList,comm,size,rank)
#fp.flush(), fp.close()    
#comm.Barrier()
#######################################################################


#######################################################################
# RELABEL PARTICLES IN THE ORDER OF OCCURENCE
#######################################################################
if (rank==0):
    fp = h5py.File(outputFile, 'r+')
else:
    fp = h5py.File(outputFile, 'r')
tracking.relabelParticles(fp,comm,size,rank)
fp.flush(), fp.close()    
comm.Barrier()
#######################################################################


#######################################################################
# GENERATING IMAGES WITH LABELLED PARTICLES
#######################################################################
if (rank==0):
    print "Generating images with labelled particles"
fp = h5py.File(outputFile, 'r')
tracking.generateImages(fp,outputDir+'/segmentation/tracking',fontScale,size,rank)
fp.flush(), fp.close()
comm.Barrier()
#######################################################################


#######################################################################
# FINDING OUT THE MEASURES FOR TRACKED PARTICLES
#######################################################################
if (rank==0):
	print "Finding measures for tracked particles"

fp = h5py.File(outputFile, 'r')
[row,col,numFrames,frameList] = misc.getVitals(fp)
particleList = fp.attrs['particleList']
zfillVal = fp.attrs['zfillVal']
procFrameList = numpy.array_split(frameList,size)
fps = fp.attrs['fps']
pixInNM = fp.attrs['pixInNM']

outFile = open(str(rank)+'.dat','wb')

area=True
perimeter=True
circularity=True
pixelList=False
bdryPixelList=False
centroid=True
intensityList=False
sumIntensity=False
effRadius=True
radius=False
circumRadius=False
inRadius=False
radiusOFgyration=False

for frame in procFrameList[rank]:
    labelImg = fp['/segmentation/labelStack/'+str(frame).zfill(zfillVal)].value
    gImgRaw = fp['/dataProcessing/gImgRawStack/'+str(frame).zfill(zfillVal)].value
    outFile.write("%f " %(1.0*frame/fps))
    for particle in particleList:
        bImg = labelImg==particle
        if (bImg.max() == True):
            label, numLabel, dictionary = imageProcess.regionProps(bImg, gImgRaw, structure=structure, centroid=centroid, area=area, perimeter=perimeter, circularity=circularity, pixelList=pixelList, bdryPixelList=bdryPixelList, effRadius=effRadius, radius=radius, circumRadius=circumRadius, inRadius=inRadius, radiusOFgyration=radiusOFgyration)
            outFile.write("%f %f %f %f %f %f " %(dictionary['centroid'][0][1]*pixInNM, (row-dictionary['centroid'][0][0])*pixInNM, dictionary['area'][0]*pixInNM*pixInNM, dictionary['perimeter'][0]*pixInNM, dictionary['circularity'][0], dictionary['effRadius'][0]*pixInNM))
        else:
            outFile.write("nan nan nan nan nan nan ")
    outFile.write("\n")
outFile.close()
fp.flush(), fp.close()
comm.Barrier()

if (rank==0):
    for r in range(size):
        if (r==0):
            measures = numpy.loadtxt(str(r)+'.dat')
        else:
            measures = numpy.row_stack((measures,numpy.loadtxt(str(r)+'.dat')))
        fileIO.delete(str(r)+'.dat')
    measures = measures[numpy.argsort(measures[:,0])]
    numpy.savetxt(outputDir+'/segmentation/measures/imgDataNM.dat', measures, fmt='%.6f')
#######################################################################

if (rank==0):
    toc = time()
    print toc-tic, (toc-tic)/numFrames
