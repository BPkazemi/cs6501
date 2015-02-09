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
sigma = 1.0

## Utils ##
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

def showim( i ):
    pylab.imshow(i, cmap="gray")
    pylab.show()

def showkpts( kpts, I, maxS ):
    ## Spatial data-structure would speed this up (kd-tree, BSP tree) ##
    pylab.imshow( I, cmap="gray" )
    for x in range(len(keypoints)):
        for y in range(len(keypoints[x])):
            for s in range(len(keypoints[x][y])):
                if keypoints[x][y][s] != 0:
                    pylab.plot( y, x, marker='o', ms=((3.*s/maxS)**2), c='y', mfc='none' )
    pylab.show()

def showims( imgs ):
    fig = pylab.figure()
    for i in range(len(imgs)):
        a = fig.add_subplot(1, len(imgs), (i+1))
        pylab.imshow(imgs[i], cmap="gray")
    pylab.show()


def prompt_fpath():
    img_name = raw_input("Enter image name: ")
    img_path = ".." + os.path.sep + "TestImages" + os.path.sep + img_name  

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

def to_degrees( radian ):
    return radian * 180. / pi

def set_bounds_flags( A, x, y ):
    at_top, at_bottom, at_left, at_right = False, False, False, False
    if x == 0:
        at_top = True
    if (x == len(A)-1):
        at_bottom = True
    if y == 0:
        at_left = True
    if (y == len(A[0])-1):
        at_right = True

    return (at_top, at_bottom, at_left, at_right)

## Potentially util
def quantize_ang( ang ):
    if (ang < 22.5 and ang > -22.5) or (ang > 157.5) or (ang < -157.5):
        return 0  # Check East-West 
    elif (ang > 22.5 and ang < 67.5) or (ang < -112.5 and ang > -157.5):
        return 45  # Check Northeast-Southwest
    elif (ang > 67.5 and ang < 112.5) or (ang < -67.5 and ang > -112.5):
        return 90  # Check North-South
    elif (ang > 112.5 and ang < 157.5) or (ang < -22.5 and ang > -67.5 ):
        return 135 # Check Northwest-Southeast

def quantize_orientation( orientation ):
    quantized = numpy.zeros( orientation.shape )

    for row in range(len(orientation)):
        for col in range(len(orientation[row])):
            quantized[row][col] = ( quantize_ang(orientation[row][col]) )

    return quantized

## Canny
def local_maxima( F, orientation ):
    maxima = numpy.ones( F.shape )
    at_top, at_bottom, at_left, at_right = False, False, False, False
    is_maxima = True 

    for row in range(len(F)):
        for col in range(len(F[row])):
            cur_angle = orientation[row][col]
            cur_magnitude = F[row][col]

            at_top, at_bottom, at_left, at_right = set_bounds_flags( F, row, col )
            '''
            if row == 0:
                at_top = True
            if (row == len(F)-1):
                at_bottom = True
            if col == 0:
                at_left = True
            if (col == len(F[0])-1):
                at_right = True
            '''

            is_maxima = True 
            if cur_angle == 0:
                # Check East-West
                if not at_right:
                    if cur_magnitude < F[row][col+1]:
                        maxima[row][col] = 0
                        is_maxima = False
                if not at_left:
                    if cur_magnitude < F[row][col-1]:
                        maxima[row][col] = 0
                        is_maxima = False
            elif cur_angle == 45:
                # Check Northeast-Southwest
                if not at_right and not at_top:
                    if cur_magnitude < F[row-1][col+1]:
                        maxima[row][col] = 0
                        is_maxima = False
                if not at_left and not at_bottom:
                    if cur_magnitude < F[row+1][col-1]:
                        maxima[row][col] = 0
                        is_maxima = False
            elif cur_angle == 90:
                # Check North-South
                if not at_top:
                    if cur_magnitude < F[row-1][col]:
                        maxima[row][col] = 0
                        is_maxima = False
                if not at_bottom:
                    if cur_magnitude < F[row+1][col]:
                        maxima[row][col] = 0
                        is_maxima = False
            elif cur_angle == 135:
                # Check Northwest-Southeast
                if not at_left and not at_top:
                    if cur_magnitude < F[row-1][col-1]:
                        maxima[row][col] = 0
                        is_maxima = False
                if not at_right and not at_bottom:
                    if cur_magnitude < F[row+1][col+1]:
                        maxima[row][col] = 0
                        is_maxima = False
            
            if is_maxima:
                maxima[row][col] = cur_magnitude

    return maxima

def threshold_edges( I, Q, t_high, t_low ):
    ''' Thresholding, with hysteresis ''' 

    t_edges = numpy.zeros( I.shape )
    for row in range(len(I)):
        for col in range(len(I[row])):
            if not t_edges[row][col] and I[row][col] > t_high:
                find_edge_chains( I, row, col, t_edges, Q[row][col], t_low )
    
    return t_edges

def find_edge_chains( I, x, y, t_edges, direction, t_low ):
    magnitude = I[x][y]
    if magnitude > t_low and not t_edges[x][y]:
        ## Pixel passes threshold. Save and mark visited
        t_edges[x][y] = magnitude

        at_top, at_bottom, at_left, at_right = set_bounds_flags( I, x, y )
        '''
        if x == 0:
            at_top = True
        if (x == len(I)-1):
            at_bottom = True
        if y == 0:
            at_left = True
        if (y == len(I[0])-1):
            at_right = True
        '''

        if direction == 0:
            # Check East-West
            if not at_left and not t_edges[x][y-1]:
                find_edge_chains( I, x, y-1, t_edges, direction,t_low )
            if not at_right and not t_edges[x][y+1]:
                find_edge_chains( I, x, y+1, t_edges, direction, t_low )
        elif direction == 45:
            # Check Northeast-Southwest
            if not at_right and not at_top and not t_edges[x-1][y+1]:
                find_edge_chains( I, x-1, y+1, t_edges, direction, t_low )
            if not at_left and not at_bottom and not t_edges[x+1][y-1]:
                find_edge_chains( I, x+1, y-1, t_edges, direction, t_low )
        elif direction == 90:
            # Check North-South
            if not at_top and not t_edges[x-1][y]:
                find_edge_chains( I, x-1, y, t_edges, direction, t_low )
            if not at_bottom and not t_edges[x+1][y]:
                find_edge_chains( I, x+1, y, t_edges, direction, t_low )
        elif direction == 135:
            # Check Northwest-Southeast
            if not at_left and not at_top and not t_edges[x-1][y-1]:
                find_edge_chains( I, x-1, y-1, t_edges, direction, t_low )
            if not at_right and not at_bottom and not t_edges[x+1][y+1]:
                find_edge_chains( I, x+1, y+1, t_edges, direction, t_low )

## Util
def gradient_components( I ):
    ## Smooth (reduce noise) ##  TODO: Necessary?
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

####### Corner Detection #######
def eigenv_test( Fx, Fy, m, t_low ):
    # TODO: Can parallelize
    L = []  
    L_square = numpy.zeros( Fx.shape )
    for row in range(len( Fx )):
        for col in range(len( Fy )):
            C = covar_matrix( Fx, Fy, row, col, m )

            # find eigenvalues
            eigvals = linalg.eigvals( C )
            R = min( eigvals ) # R is the 'cornerness' metric

            # Test eigenvalue
            if R > t_low:
                L.append( ((row, col), R) )
                L_square[row][col] = R

    L.sort(key=lambda entry: entry[1], reverse=True)

    return L, L_square

def covar_matrix( Fx, Fy, startx, starty, m ):
    C = numpy.zeros( (2, 2) )

    sx, sy, sxy = 0., 0., 0.
    for x in range( -m/2 + 1, m/2 + 1 ):
        for y in range( -m/2 + 1, m/2 + 1 ):
            # Compute covariance matrix for m x m neighborhood
            if 0 <= startx + x < len(Fx) and 0 <= starty + y < len(Fy):
                sx = sx + Fx[ startx + x ][ starty + y ]**2
                sy = sy + Fy[ startx + x ][ starty + y ]**2
                sxy = sxy + Fx[ startx + x ][ starty + y ] * Fy[ startx + x ][ starty + y ]

    C[0][0] = sx
    C[0][1] = sxy
    C[1][0] = sxy
    C[1][1] = sy

    return C

def filter_eigvals( F, L, L_square, m ):
    corners = numpy.zeros( F.shape )
    for i in range(len( L )):
        ((x, y), r) = L[i]
        corners[x][y] = F[x][y]
     
    # Nonmaximum Suppression #
    for i in range(len(L)):
        ((x, y), r) = L[i]
        for dx in range( -m/2 + 1, m/2 + 1 ):
            for dy in range( -m/2 + 1, m/2 + 1 ):
                if 0 <= dx + x < len(L_square) and 0 <= dy + y < len(L_square[x]):
                    # Remove smaller neighbors
                    if r > L_square[x + dx][y + dy]:
                        corners[x + dx][y + dy] = 0
    return corners

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

def downsample( I, factor ):
    xDim = int( math.ceil( I.shape[0]/float(factor) ) )
    yDim = int( math.ceil( I.shape[1]/float(factor) ) )
    newI = numpy.zeros( (xDim, yDim) )

    for x in range(xDim):
        for y in range(yDim):
            newI[x][y] = I[factor*x][factor*y]

    return newI

def findScaleSpaceExtrema( I, nOctaves, nScales, delSigma, k ):
    ## TODO: Do values of k depend on nScales? Does nScales depend on the number of intervals we seek?
    ## TODO: Double size of input for first level of pyramid. Use linear extrema_dir. Pre-smooth with sigma=0.5
    print "Calculating keypoints..."

    ## TODO: Create s+3 images to cover the whole domain. Necessary?
    nScales = nScales + 3
    original = I
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
        for i in range( nTriples ):
            d0 = curDoGs[ i ]
            d1 = curDoGs[ i+1 ]
            d2 = curDoGs[ i+2 ]

            for x in range(len(d1)):
                for y in range(len(d1[x])):
                    local_min, local_max = checkNeighbors3( [d0, d1, d2], (x, y), m=3 )
                    if local_min or local_max:
                        keypoints[oc][i+1][x][y] = d1[x][y]
                        nKeypoints += 1
                        maxS = max( maxS, i+1 )

        sampleIndex = int(round(math.log(2) / math.log(k)))  # k^x = 2 -> Index where blur = 2*sigma
        I = downsample( scaledImages[sampleIndex], 2 )

        # showkpts( keypoints[oc], original, maxS )

    I = original
    print str(nKeypoints) + " keypoints found"
    return keypoints, DoGs

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
    # TODO: Normalize intensities to [0,1]
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

    print str(numTossed) + " low-contrast keypoints removed"
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

def localizeKeypoints( keypoints, allDoGs ):
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

        keypoints = keypoints[oc]
        for s in range(len(keypoints)):
            for x in range(len(keypoints[s])):
                for y in range(len(keypoints[s][x])):
                    if keypoints[s][x][y] != 0:
                        # Approximate Taylor Series expansion with finite differences
                        extrema_dir = find_extrema((s,x,y), 0, grad1, grad2)

                        # extrema_dir indices: s, y, x
                        newS = s + extrema_dir[0][0]
                        newY = y + extrema_dir[1][0]
                        newX = x + extrema_dir[2][0]

                        # grad1 indices: s, x, y
                        newS = int(min(len(grad1[0])-1, max(0, newS)))
                        newX = int(min(len(grad1[0][0])-1, max(0, newX)))
                        newY = int(min(len(grad1[0][0][0])-1, max(0, newY)))

                        improved_keypoints[oc].append((newS, newX, newY))

        filtered_keypoints[oc] = filter_low_contrast( grad1, DoGs, improved_keypoints, curOctave=oc )

        ## 2. Discard keypoints along edges
        post_hfilter = filter_hessian( grad2, filtered_keypoints[oc] )
        filtered_keypoints[oc] = post_hfilter 

    return filtered_keypoints 

if __name__ == "__main__":
    print "~~~ Edge, Line, & Feature Detector ~~~"

    ## Load Image
    print "\t --> Images are loaded from the '../TestImages/' directory, so place your images there!"
    # img_path = prompt_fpath()
    img_path = "../TestImages/building.jpg"
    I = skimage.img_as_float(skimage.io.imread(img_path))
    I = color.rgb2gray(I)  ## We want intensities

    # what = prompt_fwhat()
    what = 3

    if what == 1:
        ### Canny Edge Detection ###
        start = time.clock()
        Fx, Fy = gradient_components( I )
        F, D = gradient_mag_and_dir( Fx, Fy )

        ## Nonmaximum suppression ##
        Q = quantize_orientation( D )
        I = local_maxima( F, Q )

        ## Thresholding with hysteresis ##
        tI = threshold_edges( I, Q, t_high=0.1, t_low=0.05 )

        end = time.clock()
        print "Canny edge detection completed in " + str(end - start) + "s"

        showim( tI )
    elif what == 2:
        ### Tomasi-Kanade Corner Detector ###
        start = time.clock()
        Fx, Fy = gradient_components( I )
        L, L_square = eigenv_test( Fx, Fy, m=3, t_low=0.01 )

        F, D = gradient_mag_and_dir( Fx, Fy )
        strong_corners = filter_eigvals( F, L, L_square, m=5 )

        end = time.clock()
        print "Canny edge detection completed in " + str(end - start) + "s"

        showim(strong_corners)
    elif what == 3:
        ### SIFT Feature Detector ###
        start = time.clock()

        ## 1. Find scale-space extrema
        # k = 2^(1/s)
        # TODO: Multiple octaves
        kpts, dogs = findScaleSpaceExtrema( I, nOctaves=1, nScales=1, delSigma=1.6, k=2.**(1./3.) )

        ## 2. Localize keypoints
        kpts2 = localizeKeypoints( kpts, dogs )

        pylab.imshow( I, cmap="gray" )
        for oc in range(len(kpts2)):
            for i in range(len(kpts2[oc])):
                (s, x, y) = kpts2[0][i]
                pylab.plot( y, x, marker='o', ms=5, mec='y', mfc='none' )
            pylab.show()

        end = time.clock()
        print "SIFT feature detection completed in " + str(end - start) + "s"
