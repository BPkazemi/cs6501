import utils
import edge
import corner
import sift
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

## GLOBAL VARS ##
sigma = utils.sigma

if __name__ == "__main__":
    print "~~~ Edge, Line, & Feature Detector ~~~"
    print "\t --> Images are loaded from the './TestImages/' directory, so place your images there!"
    # Load image
    img_path = utils.prompt_fpath()
    I = skimage.img_as_float(skimage.io.imread(img_path))
    I = color.rgb2gray(I)  ## Keep intensities

    what = utils.prompt_fwhat()

    if what == 1:
        ### Canny Edge Detection ###
        start = time.clock()
        edge.go( I )
        end = time.clock()
        print "Canny edge detection completed in " + str(end - start) + "s"
    elif what == 2:
        ### Tomasi-Kanade Corner Detector ###
        start = time.clock()
        corner.go( I )
        end = time.clock()
        print "Canny edge detection completed in " + str(end - start) + "s"

    elif what == 3:
        ### SIFT Feature Detector ###
        start = time.clock()
        sift.go( I )
        end = time.clock()
        print "SIFT feature detection completed in " + str(end - start) + "s"
