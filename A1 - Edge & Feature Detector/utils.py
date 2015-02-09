import pylab
import pdb
import numpy
import os, os.path
import skimage
import time
import math
from math import sqrt, pi, e
from skimage import io, color
from scipy import ndimage, signal, mgrid, misc, linalg

sigma = 0.5

########## Other ##########
###########################
def to_degrees( radian ):
    return radian * 180. / pi

##### Guassian Kernels ####
###########################
def gauss_partials( sigma ):
    ## x and y kernels of the partial derivative of a bivariate gaussian ##

    ## 2D version of: [-2 -1 0 1 2]  ##
    ## TODO: Would it be better to be 1sigma, 2sigma, etc?
    x = mgrid[-2:3].astype(float)
    y = numpy.swapaxes([mgrid[-2:3]], 0, 1).astype(float)

    ## Formula for partial derivative
    dgx = -x / (sigma**3. * sqrt(2.*pi)) * e**(-x**2. / (2.*sigma**2.))
    dgy = -y / (sigma**3. * sqrt(2.*pi)) * e**(-y**2. / (2.*sigma**2.))

    ## Normalize values
    dgx = dgx / sum(abs(dgx))
    dgy = dgy / sum(abs(dgy))
    
    return (dgx, dgy) 

def gauss( sigma ):
    ## x and y kernels of a bivariate gaussian ##
    x = mgrid[-2:3].astype(float)
    y = numpy.swapaxes([mgrid[-2:3]], 0, 1).astype(float)

    ## Compute gaussian values at x,y = -2, -1, 0, 1, 2
    gx = 1. / (sigma * sqrt(2. * pi)) * e**(-x**2. / (2. * sigma**2.))
    gy = 1. / (sigma * sqrt(2. * pi)) * e**(-y**2. / (2. * sigma**2.))

    ## Normalize values
    gx = gx / sum(abs(gx))
    gy = gy / sum(abs(gy))

    return (gx, gy)

#### Displaying images ####
###########################

def downsample( I, factor ):
    xDim = int( math.ceil( I.shape[0]/float(factor) ) )
    yDim = int( math.ceil( I.shape[1]/float(factor) ) )
    newI = numpy.zeros( (xDim, yDim) )

    for x in range(xDim):
        for y in range(yDim):
            newI[x][y] = I[factor*x][factor*y]

    return newI

def showim( i ):
    pylab.imshow( i, cmap="gray")
    pylab.show()

def saveim( i, fname ):
    pylab.imsave( "./writeup_html/" + fname, i, cmap="gray" )

def showkpts( allKeypoints, Is, maxS ):
    ## Spatial data-structure would speed this up (kd-tree, BSP tree) ##
    for ind, I in enumerate(Is[0:-1]):
        kpts = allKeypoints[ind]

        pylab.imshow( I, cmap="gray" )
        for s in range(len(kpts)):
            for x in range(len(kpts[s])):
                for y in range(len(kpts[s][x])):
                    if kpts[s][x][y] != 0:
                        pylab.plot( y, x, marker='o', ms=((3.*s/maxS)**2), mec='r', mfc='none' )
        pylab.show()

def showims( imgs ):
    fig = pylab.figure()
    for i in range(len(imgs)):
        a = fig.add_subplot(1, len(imgs), (i+1))
        pylab.imshow(imgs[i], cmap="gray")
    pylab.show()

#### Gradient Stuff #######
###########################
def gradient_components( I ):
    ## Smooth (reduce noise)
    img_smooth = ndimage.gaussian_filter(I, sigma=sigma)

    ## Convolve ##
    dgx, dgy = gauss_partials( sigma=sigma )
    gx, gy = gauss( sigma=sigma )

    Fx1 = signal.convolve(img_smooth, [dgx], mode='same')
    Fx = signal.convolve(Fx1, gy, mode='same')

    Fy1 = signal.convolve(img_smooth, dgy, mode='same')
    Fy = signal.convolve(Fy1, [gx], mode='same')

    return Fx, Fy

def gradient_mag_and_dir( Fx, Fy ):
    ''' Finds gradient magnitude & direction '''
    F = numpy.sqrt( Fx**2 + Fy**2 )  
    D = to_degrees( numpy.arctan2( Fy, Fx ) )
    return F, D

###### Prompt Stuff #######
###########################
def prompt_fpath():
    img_name = raw_input("Enter image name: ")
    img_path = "." + os.path.sep + "TestImages" + os.path.sep + img_name  

    if not os.path.isfile(img_path):
        print img_name + " is not a file under the TestImages directory. Try again."
        return prompt_fpath()
    else:
        print "Found file."
        return img_path

def prompt_fwhat():
    what = int(raw_input("What do you want to do?\n1 - Edge Detection\n2 - Corner Detection\n3 - Feature Detection\n"))

    if what not in [1, 2, 3]:
        print "You can only choose from these options. Type the number and press 'enter'."
        prompt_fwhat()
    else:
        if what == 1:
            print "Doing Canny edge detection..."
        elif what == 2:
            print "Doing Tomasi-Kanade corner detection..."
        elif what == 1:
            print "Doing SIFT feature detection..."
        return what

