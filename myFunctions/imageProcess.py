import numpy
from scipy import ndimage
from skimage.morphology import disk, white_tophat
from mahotas.polygon import fill_convexhull
from skimage import measure


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


#######################################################################
# 
#######################################################################
def regionProps(bImg, gImg=0, structure=[[1,1,1],[1,1,1],[1,1,1]], area=False, perimeter=False, circularity=False, orientation=False, pixelList=False, bdryPixelList = False, centroid=False, intensityList=False, sumIntensity=False, avgIntensity=False, maxIntensity=False, effRadius=False, radius=False, theta=False, rTick=False, qTick=False, circumRadius=False, inRadius=False, radiusOFgyration=False, rTickMMM=False, thetaMMM=False):
    [labelImg, numLabel] = ndimage.label(bImg, structure=structure)
    [row, col] = bImg.shape
    dictionary = {}
    dictionary['id'] = []
    if (area == True):
        dictionary['area'] = []
    if (perimeter == True):
        dictionary['perimeter'] = []
    if (circularity == True):
        dictionary['circularity'] = []
    if (orientation == True):
        dictionary['orientation'] = []
    if (pixelList == True):
        dictionary['pixelList'] = []
    if (bdryPixelList == True):
        dictionary['bdryPixelList'] = []
    if (centroid == True):
        dictionary['centroid'] = []
    if (intensityList == True):
        dictionary['intensityList'] = []
    if (sumIntensity == True):
        dictionary['sumIntensity'] = []
    if (avgIntensity == True):
        dictionary['avgIntensity'] = []
    if (maxIntensity == True):
        dictionary['maxIntensity'] = []
    if (effRadius == True):
        dictionary['effRadius'] = []
    if (radius == True):
        dictionary['radius'] = []
    if (circumRadius == True):
        dictionary['circumRadius'] = []
    if (inRadius == True):
        dictionary['inRadius'] = []
    if (radiusOFgyration == True):
        dictionary['radiusOFgyration'] = []
    if (rTick == True):
        dictionary['rTick'] = []
    if (qTick == True):
        dictionary['qTick'] = []
    if (rTickMMM == True):
        dictionary['rTickMean'] = []
        dictionary['rTickMin'] = []
        dictionary['rTickMax'] = []
    if (theta == True):
        dictionary['theta'] = []
    if (thetaMMM == True):
        dictionary['thetaMean'] = []
        dictionary['dThetaP'] = []
        dictionary['dThetaM'] = []
        
    for i in range(1, numLabel+1):
        bImgLabelN = labelImg == i
        dictionary['id'].append(i)
        if (area == True):
            Area = bImgLabelN.sum()
            dictionary['area'].append(Area)
        if (perimeter == True):
            pmeter = measure.perimeter(bImgLabelN)
            dictionary['perimeter'].append(pmeter)
        if (circularity == True):
            Area = bImgLabelN.sum()
            pmeter = measure.perimeter(bImgLabelN)
            circlarity = (4*numpy.pi*Area)/(pmeter**2)
            if (circlarity>1):
                circlarity=1-(circularity-1)
            dictionary['circularity'].append(circlarity)
        if (orientation == True):
            regions = regionprops(bImgLabelN.astype('uint8'))
            for props in regions:
                dictionary['orientation'].append(numpy.rad2deg(props.orientation))
        if (pixelList == True):
            pixelsRC = numpy.nonzero(bImgLabelN)
            dictionary['pixelList'].append([pixelsRC[0].tolist(),pixelsRC[1].tolist()])
        if (bdryPixelList == True):
            bdry = boundary(bImgLabelN)
            pixelsRC = numpy.nonzero(bdry)
            dictionary['bdryPixelList'].append([pixelsRC[0].tolist(),pixelsRC[1].tolist()])
        if (centroid == True):
            pixelsRC = numpy.nonzero(bImgLabelN)
            centerRC = [numpy.average(pixelsRC[0]), numpy.average(pixelsRC[1])]
            dictionary['centroid'].append(centerRC)
        if (intensityList == True):
            pixelsRC = numpy.nonzero(bImgLabelN)
            intensities = gImg[pixelsRC]
            dictionary['intensityList'].append(intensities)
        if (sumIntensity == True):
            pixelsRC = numpy.nonzero(bImgLabelN)
            sumInt = numpy.sum(gImg[pixelsRC])
            dictionary['sumIntensity'].append(sumInt)
        if (avgIntensity == True):
            pixelsRC = numpy.nonzero(bImgLabelN)
            avgInt = numpy.mean(gImg[pixelsRC])
            dictionary['avgIntensity'].append(avgInt)
        if (maxIntensity == True):
            pixelsRC = numpy.nonzero(bImgLabelN)
            maxInt = numpy.max(gImg[pixelsRC])
            dictionary['maxIntensity'].append(maxInt)
        if (radius == True):
            pixelsRC = numpy.nonzero(bImgLabelN)
            centerRC = (numpy.average(pixelsRC[0]), numpy.average(pixelsRC[1]))
            radii = numpy.max(numpy.sqrt((pixelsRC[0]-centerRC[0])**2 + (pixelsRC[1]-centerRC[1])**2))
            dictionary['radius'].append(radii)
        if (effRadius == True):
            Area = bImgLabelN.sum()
            effRadii = numpy.sqrt(Area/numpy.pi)
            dictionary['effRadius'].append(effRadii)
        if (circumRadius == True):
            pixelsRC = numpy.nonzero(bImgLabelN)
            centerRC = (numpy.average(pixelsRC[0]), numpy.average(pixelsRC[1]))
            bdryPixelsRC = numpy.nonzero(boundary(bImgLabelN))
            radii = numpy.max(numpy.sqrt((bdryPixelsRC[0]-centerRC[0])**2 + (bdryPixelsRC[1]-centerRC[1])**2))
            dictionary['circumRadius'].append(radii)
        if (inRadius == True):
            pixelsRC = numpy.nonzero(bImgLabelN)
            centerRC = (numpy.average(pixelsRC[0]), numpy.average(pixelsRC[1]))
            bdryPixelsRC = numpy.nonzero(boundary(bImgLabelN))
            radii = numpy.min(numpy.sqrt((bdryPixelsRC[0]-centerRC[0])**2 + (bdryPixelsRC[1]-centerRC[1])**2))
            dictionary['inRadius'].append(radii)
        if (radiusOFgyration == True):
            pixelsRC = numpy.nonzero(bImgLabelN)
            centerRC = (numpy.average(pixelsRC[0]), numpy.average(pixelsRC[1]))
            gyration = numpy.sqrt(numpy.average((pixelsRC[0]-centerRC[0])**2 + (pixelsRC[1]-centerRC[1])**2))
            dictionary['radiusOFgyration'].append(gyration)
        if (rTick == True):
            pixelsRC = numpy.nonzero(bImgLabelN)
            centerRC = [numpy.average(pixelsRC[0]), numpy.average(pixelsRC[1])]
            center = [row/2,col/2]
            if (row<=col):
                sc = 1.0*col/row
                qArrScale = col
                dist = numpy.sqrt((sc*(centerRC[0]-center[0]))**2 + (centerRC[1]-center[1])**2)
            else:
                sc = 1.0*row/col
                qArrScale = row
                dist = numpy.sqrt((centerRC[0]-center[0])**2 + (sc*(centerRC[1]-center[1]))**2)
            if (dist==0):
                rTck = 0
            else:
                rTck = qArrScale/dist
            dictionary['rTick'].append(rTck)
        if (qTick == True):
            pixelsRC = numpy.nonzero(bImgLabelN)
            centerRC = [numpy.average(pixelsRC[0]), numpy.average(pixelsRC[1])]
            center = [row/2,col/2]
            if (row<=col):
                sc = 1.0*col/row
                qArrScale = col
                qTck = numpy.sqrt((sc*(centerRC[0]-center[0]))**2 + (centerRC[1]-center[1])**2)
            else:
                sc = 1.0*row/col
                qArrScale = row
                qTck = numpy.sqrt((centerRC[0]-center[0])**2 + (sc*(centerRC[1]-center[1]))**2)
            dictionary['qTick'].append(qTck)
        if (rTickMMM == True):
            pixelsRC = numpy.nonzero(bImgLabelN)
            centerRC = [numpy.average(pixelsRC[0]), numpy.average(pixelsRC[1])]
            center = [row/2,col/2]
            if (row<=col):
                sc = 1.0*col/row
                qArrScale = col
                dist = numpy.sqrt((sc*(centerRC[0]-center[0]))**2 + (centerRC[1]-center[1])**2)
                distAll = numpy.sqrt((sc*(pixelsRC[0]-center[0]))**2 + (pixelsRC[1]-center[1])**2)
            else:
                sc = 1.0*row/col
                qArrScale = row
                dist = numpy.sqrt((centerRC[0]-center[0])**2 + (sc*(centerRC[1]-center[1]))**2)
                distAll = numpy.sqrt((pixelsRC[0]-center[0])**2 + (sc*(pixelsRC[1]-center[1]))**2)
            if (dist==0):
                rTck = 0
            else:
                rTck = qArrScale/dist
            rTckAll = qArrScale/distAll
            dictionary['rTickMean'].append(rTck)
            dictionary['rTickMin'].append(rTckAll.min())
            dictionary['rTickMax'].append(rTckAll.max())
        if (theta == True):
            pixelsRC = numpy.nonzero(bImgLabelN)
            centerRC = (numpy.average(pixelsRC[0]), numpy.average(pixelsRC[1]))
            center = [row/2,col/2]
            angle = numpy.arctan2(center[0]-centerRC[0], centerRC[1]-col/2.0)*180/numpy.pi
            if (angle<0):
                angle = 360+angle
            dictionary['theta'].append(angle)
        if (thetaMMM == True):
            pixelsRC = numpy.nonzero(bImgLabelN)
            centerRC = [numpy.average(pixelsRC[0]), numpy.average(pixelsRC[1])]
            center = [row/2,col/2]
            angle = numpy.arctan2(center[0]-centerRC[0], centerRC[1]-center[1])*180/numpy.pi
            angleAll = numpy.arctan2(center[0]-pixelsRC[0], pixelsRC[1]-center[1])*180/numpy.pi
            if (angle<0):
                angle = 360+angle
            for i in range(len(angleAll)):
                if (angleAll[i]<0):
                    angleAll[i] = 360+angleAll[i]
            if (numpy.max([angleAll<10])==True and numpy.max(angleAll>350)==True):
                if (angle < 180):
                    dThetaP = numpy.max(angleAll[angleAll<180])-angle
                    dThetaM = angle-(numpy.min(angleAll[angleAll>180])-360)
                else:
                    dThetaP = numpy.max(angleAll[angleAll<180])-(angle-360)
                    dThetaM = angle-numpy.min(angleAll[angleAll>180])
            else:
                dThetaP = numpy.max(angleAll)-angle
                dThetaM = angle-numpy.min(angleAll)
            dictionary['thetaMean'].append(angle)
            dictionary['dThetaP'].append(dThetaP)
            dictionary['dThetaM'].append(dThetaM)
    return labelImg, numLabel, dictionary
#######################################################################
