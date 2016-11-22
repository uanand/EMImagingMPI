import cv2
import numpy
import h5py

#######################################################################
# LABELING PARTICLES
#######################################################################
def labelParticles(fp, centerDispRange=[5,5], perAreaChangeRange=[0.1,0.2], missFramesTh=10, structure=[[0,1,0],[1,1,1],[0,1,0]]):
    
    [row,col,numFrames] = fp.attrs['row'],fp.attrs['col'],fp.attrs['numFrames']
    frameList = fp.attrs['frameList']
    zfillVal = fp.attrs['zfillVal']
    
    labelStack = numpy.zeros([row,col,numFrames], dtype='uint32')
    for frame in frameList:
        str1 = str(frame)+'/'+str(frameList[-1]); str2 = '\r'+' '*len(str1)+'\r'
        sys.stdout.write(str1)
        bImg = fp['/segmentation/bImgStack/'+str(frame).zfill(zfillVal)].value
        gImg = fp['/dataProcessing/gImgRawStack/'+str(frame).zfill(zfillVal)].value

        if (frame==frameList[0]):
            labelImg_0, numLabel_0, dictionary_0 = imageProcess.regionProps(bImg, gImg, structure=structure, centroid=True, area=True)
            maxID = numLabel_0
            occurenceFrameList = [frame]*maxID
            dictionary_0['frame'] = []
            for i in range(len(dictionary_0['id'])):
                dictionary_0['frame'].append(frame)
            labelStack[:,:,frame-1] = labelImg_0
        else:
            labelImg_1, numLabel_1, dictionary_1 = regionProps(bImg, gImg, structure=structure, centroid=True, area=True)
            if (numLabel_1>0):
                areaMin = min(dictionary_1['area']); areaMax = max(dictionary_1['area'])
            for i in range(len(dictionary_1['id'])):
                flag = 0
                bImg_1_LabelN = labelImg_1==dictionary_1['id'][i]
                center_1 = dictionary_1['centroid'][i]
                area_1 = dictionary_1['area'][i]
                frame_1 = frame
                if (areaMax-areaMin>0):
                    factor = 1.0*(area_1-areaMin)/(areaMax-areaMin)
                    perAreaChangeTh = perAreaChangeRange[1] - factor*(perAreaChangeRange[1]-perAreaChangeRange[0])
                    centerDispTh = centerDispRange[1] - factor*(centerDispRange[1]-centerDispRange[0])
                else:
                    perAreaChangeTh = perAreaChangeRange[1]
                    centerDispTh = centerDispRange[1]
                closeness = 1e10
                for j in range(len(dictionary_0['id'])-1,-1,-1):
                    center_0 = dictionary_0['centroid'][j]
                    area_0 = dictionary_0['area'][j]
                    frame_0 = dictionary_0['frame'][j]
                    centerDisp = numpy.sqrt((center_1[0]-center_0[0])**2. + (center_1[1]-center_0[1])**2.)
                    perAreaChange = 1.0*numpy.abs(area1-area0)/numpy.maximum(area1,area0)
                    missFrames = frame_1-frame_0
                    if (centerDisp <= centerDispTh):
                        if (perAreaChange <= perAreaChangeTh):
                            if (missFrames <= missFramesTh):
                                if (centerDisp < closeness):
                                    J = j
                                    flag = 1
                if (flag == 1):
                    labelStack[:,:,frame-1] += (bImg_1_LabelN*dictionary_0['id'][j]).astype('uint16')
                    dictionary_0['centroid'][j] = center_1
                    dictionary_0['area'][j] = area_1
                    dictionary_0['frame'][j] = frame
                if (flag == 0):
                    maxID += 1
                    occurenceFrameList.append(frame)
                    labelN_1 = bImg_1_LabelN*maxID
                    labelStack[:,:,frame-1] += labelN_1.astype('uint16')
                    dictionary_0['id'].append(maxID)
                    dictionary_0['centroid'].append(center_1)
                    dictionary_0['area'].append(area_1)
                    dictionary_0['frame'].append(frame)
        sys.stdout.flush(); sys.stdout.write(str2)
    sys.stdout.flush()

    if (labelStack.max() < 256):
        labelStack = labelStack.astype('uint8')
    elif (labelStack.max()<65536):
        labelStack = labelStack.astype('uint16')
        
    for frame in frameList:
        labelImg = labelStack[:,:,frame-1]
        for N in numpy.unique(labelImg)[1:]:
            labelImgN = labelImg==N
            numLabel = regionProps(bImg, gImg, structure=structure)[1]
            if (numLabel>1):
                labelImg[labelImg==N] = 0
                labelStack[:,:,frame-1] = labelImg
                
    for frame in frameList:
        create_dataset('/segmentation/labelStack/'+str(frame).zfill(zfillVal), data=labelStack[:,:,frame-1], compression='gzip', compression_opts=9)
    fp.attrs['maxID'] = maxID
    fp.attrs['occurenceFrameList'] = occurenceFrameList
    del labelStack
    return maxID, occurenceFrameList
#######################################################################


#######################################################################
# 
#######################################################################


#######################################################################
