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
    F, D = utils.gradient_mag_and_dir( Fx, Fy )

    ## Nonmaximum suppression ##
    Q = quantize_orientation( D )
    I = local_maxima( F, Q )

    ## Thresholding with hysteresis ##
    tI = threshold_edges( I, Q, t_high=0.1, t_low=0.05 )
    utils.saveim( tI, "edge_flickr.jpg" )
    # utils.showim( tI )

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

