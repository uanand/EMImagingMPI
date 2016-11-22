import numpy
from scipy import ndimage
from skimage.morphology import disk, white_tophat
from mahotas.polygon import fill_convexhull


#######################################################################
# 
#######################################################################
def normalize(gImg, min=0, max=255):
    if (gImg.max() > gImg.min()):
        gImg = 1.0*(max-min)*(gImg - gImg.min())/(gImg.max() - gImg.min())
        gImg=gImg+min
    elif (gImg.max() > 0):
        gImg[:] = max
    gImg=gImg.astype('uint8')
    return gImg
#######################################################################


#######################################################################
#
#######################################################################
def invertImage(gImg):
    return 255-gImg
#######################################################################


#######################################################################
#
#######################################################################
def subtractBackground(gImg,sigma,radius):
    gImgBlur = ndimage.gaussian_filter(gImg, sigma=sigma)
    gImgTHT=white_tophat(gImgBlur, selem=disk(radius))
    gImg = normalize(gImgTHT)
    return gImg
#######################################################################


#######################################################################
#
#######################################################################
def fillHoles(bImg):
    return ndimage.binary_fill_holes(bImg)
#######################################################################


#######################################################################
# 
#######################################################################
def convexHull(bImg):
    label,numLabel=ndimage.label(bImg)
    bImg[:]=False
    for i in range(1,numLabel+1):
        bImgN=label==i
        bImgN=fill_convexhull(bImgN)
        bImg=numpy.logical_or(bImg,bImgN)
    return bImg
#######################################################################


#######################################################################
# 
#######################################################################
def boundary(bImg):
    bImgErode = ndimage.binary_erosion(bImg)
    bImgBdry = (bImg - bImgErode).astype('bool')
    return bImgBdry
#######################################################################



    
    #print "GENERATING IMAGES"
    #for frame in frameList:
        #bImg = numpy.load(bImgDir+'/'+str(frame)+'.npy')
        #gImg = numpy.load(gImgDir+'/'+str(frame)+'.npy')
        #bImgBdry = normalize(boundary(bImg))
        #labelImg, numLabel, dictionary = regionProps(bImg, gImg, structure=structure, centroid=True)
        #bImg = normalize(bImg)
        #for j in range(len(dictionary['id'])):
            #bImgLabelN = labelImg == dictionary['id'][j]
            #ID = numpy.max(bImgLabelN*labelStack[:,:,frame-1])
            #cv2.putText(bImg, str(ID), (int(dictionary['centroid'][j][1]),int(dictionary['centroid'][j][0])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale, color=127, thickness=2, bottomLeftOrigin=False)
        #finalImage = numpy.column_stack((bImg, numpy.maximum(bImgBdry,gImg)))
        #numpy.save(labelDataDir+'/'+str(frame)+'.npy', labelStack[:,:,frame-1])
        #cv2.imwrite(labelImgDir+'/'+str(frame)+'.png', finalImage)
