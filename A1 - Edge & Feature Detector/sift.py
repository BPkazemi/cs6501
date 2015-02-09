import utils
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

# ~~~~~~~~~~~~ Entry Point ~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
def go( I ):
    ## 1. Find scale-space extrema
    # k = 2^(1/s)
    kpts, dogs, originals = findScaleSpaceExtrema( I, nOctaves=1, nScales=3, delSigma=1.6, k=2.**(1./3.) )

    ## 2. Localize keypoints
    kpts2 = localizeKeypoints( kpts, dogs )

    maxSList = []
    for oc in range(len(kpts2)):
        sOnly = [a[0] for a in kpts2[oc]]
        if len(sOnly) > 0:
            maxSList.append( max(sOnly) )
        else:
            maxSList.append(1)

    for oc in range(len(kpts2)):
        fig = pylab.figure()
        pylab.imshow( originals[oc], cmap="gray" )
        for i in range(len(kpts2[oc])):
            (s, x, y) = kpts2[oc][i]
            pylab.plot( y, x, marker='o', ms=((3.*s/maxSList[oc])**2), mec='r', mfc='none' )
        # pylab.savefig('FILENAME.png', bbox_inches="tight")

        pylab.show()

def findScaleSpaceExtrema( I, nOctaves, nScales, delSigma, k ):
    ## TODO: Do values of k depend on nScales? Does nScales depend on the number of intervals we seek?
    print "Calculating keypoints..."

    nScales = nScales + 3
    originals = [I]
    octaves = []
    DoGs = {}
    keypoints = {}
    nKeypoints = 0
    ## 0. For each octave...
    for oc in range( nOctaves ):
        ## 1. Define octave's scale-space. sigma = 1.6, k = sqrt(2)
        scaledImages = []
        for i in range( nScales ):
            sig = k**(i+1.) * delSigma
            scaledImages.append( ndimage.gaussian_filter(I, sigma=sig) )
        octaves.append(scaledImages)

        ## 2. Take DoG. Per octave: nDoGs = (nScales - 1), nDoGs-2 Triples.
        DoGs[oc] = []
        for i in range( nScales-1 ):
            L2 = scaledImages[ i+1 ]
            L1 = scaledImages[ i ]

            DoGs[oc].append(L2-L1)

        ## 3. Find local extrema among DoGs.
        keypoints[oc] = numpy.zeros( 
            (nScales,
             scaledImages[0].shape[0], 
             scaledImages[0].shape[1]))
        nTriples = len( DoGs[0] )-2
        curDoGs = DoGs[oc]
        maxS = 0
        for s in range( nTriples ):
            d0 = curDoGs[ s ]
            d1 = curDoGs[ s+1 ]
            d2 = curDoGs[ s+2 ]

            for x in range(len(d1)):
                for y in range(len(d1[x])):
                    local_min, local_max = checkNeighbors3( [d0, d1, d2], (x, y), m=3 )
                    if local_min or local_max:
                        keypoints[oc][s+1][x][y] = d1[x][y]
                        nKeypoints += 1
                        maxS = max( maxS, s+1 )

        sampleIndex = int(round(math.log(2) / math.log(k)))  # k^x = 2 -> Index where blur = 2*sigma
        originals.append( utils.downsample( scaledImages[sampleIndex], 2 ) )
        I = originals[ len(originals)-1 ]
        print str(nKeypoints) + " keypoints found for octave " + str(oc)
        nKeypoints = 0

    # showkpts( keypoints, originals, maxS )
    I = originals[0]
    return keypoints, DoGs, originals

def checkNeighbors3( DoGs, (startx, starty), m ):
    if len(DoGs) != 3:
        raise Exception( "checkNeighbors3 must be called on a list of length 3 only." )

    d0 = DoGs[0]
    d1 = DoGs[1]
    d2 = DoGs[2]

    candidate = d1[ startx ][ starty ]

    local_min, local_max = True, True
    for x in range( -m/2 + 1, m/2 + 1 ):
        for y in range( -m/2 + 1, m/2 + 1 ):
            # Check bounds
            if 0 <= startx + x < len(d1) and 0 <= starty + y < len(d1[x]):
                if d0[startx+x][starty+y] > candidate or d2[startx+x][starty+y] > candidate:
                    local_max = False
                if d0[startx+x][starty+y] < candidate or d2[startx+x][starty+y] < candidate:
                    local_min = False

    return local_min, local_max

def taylor_series( (s, x, y), grad1, grad2 ):
    # Derivatives of Taylor Series approximated with finite differences
    # pdb.set_trace()
    dSigma = grad1[0][s][x][y]
    dX = grad1[1][s][x][y]
    dY = grad1[2][s][x][y]

    dSigmaSigma = grad2[0][0][s][x][y]
    dXX = grad2[1][1][s][x][y]
    dYY = grad2[2][2][s][x][y]

    dSigmaX = grad2[0][1][s][x][y]
    dSigmaY = grad2[0][2][s][x][y]
    dYX = grad2[2][1][s][x][y]

    d1D = numpy.zeros( ( 3, 1 ) )
    d1D[0][0] = dSigma
    d1D[1][0] = dX
    d1D[2][0] = dY
    d2D = numpy.array(( 
        [dSigmaSigma, dSigmaY, dSigmaX], 
        [dSigmaY, dYY, dYX], 
        [dSigmaX, dYX, dXX] 
    ))

    # Must be invertible 
    # -- This function looks pretty good --
    if abs(linalg.det(d2D)) > 1e-10:
        extrema_dir = -linalg.inv(d2D).dot(d1D)  # indices: sigma, y, x
        return numpy.round(extrema_dir).astype(int)
    else:
        return numpy.zeros((3, 1))

def find_extrema( (s, x, y), i, grad1, grad2 ):
    # pdb.set_trace()
    extrema_dir = taylor_series((s,x,y), grad1, grad2)  ## Returned same thing twice...Looked like a fluke

    if i >= 4:
        # Prevent thrashing
        return extrema_dir

    ## TODO: Is absolute value necessary?
    ## -- This function looks pretty good --
    if (
        len([num[0] for num in extrema_dir if abs(num[0]) > 0.5]) > 0
    ):
        # Maximum is far enough away to repeat process
        newX, newY, newS = x, y, s
        if abs(extrema_dir[0][0]) > 0.5:
            newS += extrema_dir[0][0]
        if abs(extrema_dir[1][0]) > 0.5:
            newY += extrema_dir[1][0]
        if abs(extrema_dir[2][0]) > 0.5:
            newX += extrema_dir[2][0]

        newS = min(len(grad1[0])-1, max(0, newS))
        newX = min(len(grad1[0][0])-1, max(0, newX))
        newY = min(len(grad1[0][0][0])-1, max(0, newY))

        # Repeat with better location
        return find_extrema( (newS, newX, newY), (i+1), grad1, grad2 )
    else:
        return extrema_dir

def filter_low_contrast( grad, DoGs, extrema_locs, curOctave):
    numTossed = 0
    filtered_keypoints = []
    for i in range(len(extrema_locs[curOctave])):
        # For each (accurate) keypoint, calculate D(x) - 0.5*dD(X)
        (s, x, y) = extrema_locs[curOctave][i]
        Dx = DoGs[s][x][y] - 0.5*grad[curOctave][s][x][y]
        if Dx >= 0.03:
            filtered_keypoints.append( (s,x,y) )
        else:
            numTossed += 1

    print str(numTossed) + " low-contrast keypoints removed for octave " + str(curOctave)
    return filtered_keypoints

def filter_hessian( grad2, filtered_keypoints ):
    ## Compute Hessian of D, use as threshold
    numTossed = 0
    hessian_threshold = []
    for i in range(len(filtered_keypoints)):
        (s, x, y) = filtered_keypoints[i]
        dXX = grad2[1][1][s][x][y]
        dYY = grad2[2][2][s][x][y]
        dXY = grad2[1][2][s][x][y]

        H = numpy.array(( [dXX, dXY], [dXY, dYY] ))
        trace = H.trace()
        det = linalg.det( H )

        r = 10.

        lhs = trace**2. / det
        rhs = (r+1.)**2 / r
        if (lhs < rhs):
            hessian_threshold.append( (s, x, y) )
        else:
            numTossed += 1

    print str(numTossed) + " keypoints along edges removed"
    return hessian_threshold

def localizeKeypoints( octaveKeypoints, allDoGs ):
    print "Localizing keypoints..."

    # For each octave, localize keypoints
    improved_keypoints = {}
    filtered_keypoints = {}
    for oc in range(len(allDoGs)):
        DoGs = allDoGs[oc]
        improved_keypoints[oc] = []  ## Holds the x, y, s locations of the new keypoints
        filtered_keypoints[oc] = []
        
        ## 1. Filter out low-contrast points
        grad1 = numpy.gradient( DoGs )
        grad2 = numpy.gradient( grad1 )

        keypoints = octaveKeypoints[oc]
        for s in range(len(keypoints)):
            for x in range(len(keypoints[s])):
                for y in range(len(keypoints[s][x])):
                    if keypoints[s][x][y] != 0:
                        # Approximate Taylor Series expansion with finite differences
                        extrema_dir = find_extrema((s,x,y), 0, grad1, grad2)

                        # extrema_dir indexed in order of: s, y, x
                        newS = s + extrema_dir[0][0]
                        newY = y + extrema_dir[1][0]
                        newX = x + extrema_dir[2][0]

                        # grad1 indexed in order of: s, x, y
                        newS = int(min(len(grad1[0])-1, max(0, newS)))
                        newX = int(min(len(grad1[0][0])-1, max(0, newX)))
                        newY = int(min(len(grad1[0][0][0])-1, max(0, newY)))

                        improved_keypoints[oc].append((newS, newX, newY))

        filtered_keypoints[oc] = filter_low_contrast( grad1, DoGs, improved_keypoints, curOctave=oc )

        ## 2. Discard keypoints along edges
        filtered_keypoints[oc] = filter_hessian(grad2, filtered_keypoints[oc]) 

    return filtered_keypoints 

