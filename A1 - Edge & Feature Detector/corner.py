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

sigma = utils.sigma

# ~~~~~~~~~~~~ Entry Point ~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
def go( I ):
    Fx, Fy = utils.gradient_components( I )
    L, L_square = eigenv_test( Fx, Fy, m=3, t_low=0.01 )

    F, D = utils.gradient_mag_and_dir( Fx, Fy )
    strong_corners = filter_eigvals( F, L, L_square, m=5 )

    # utils.saveim(strong_corners, "YOUR_FILENAME_HERE.jpg")
    utils.showim(strong_corners)

####### Corner Detection #######
def eigenv_test( Fx, Fy, m, t_low ):
    print "Checking eigenvalues..."
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
    print "Performing nonmaximum suppression..."
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

