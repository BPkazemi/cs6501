import pylab
import numpy
import os, os.path
import skimage
from math import sqrt, pi, e
from skimage import io, color
from scipy import ndimage, signal, mgrid, misc

## GLOBAL VARS ##
sigma = 0.75

def gauss_partial_kernels():
    ## 1D kernel for x and y partial derivatives of bivariate gaussian ##
    x = mgrid[-2:3]
    y = numpy.swapaxes([mgrid[-2:3]], 0, 1)
    x = x.astype(float)
    y = y.astype(float)

    ## Compute partial derivative at x,y = -2, -1, 0, 1, 2
    dgx = -x / (sigma**3. * sqrt(2.*pi)) * e**(-x**2. / (2.*sigma**2.))
    dgy = -y / (sigma**3. * sqrt(2.*pi)) * e**(-y**2. / (2.*sigma**2.))

    ## Normalize values
    dgx = dgx / sum(abs(dgx))
    dgy = dgy / sum(abs(dgy))
    
    return (dgx, dgy) 

def gauss_kernels():
    ## 1D kernel for bivariate gaussian ##
    x = mgrid[-2:3]
    y = numpy.swapaxes([mgrid[-2:3]], 0, 1)
    x = x.astype(float)
    y = y.astype(float)

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

def prompt_fpath():
    # TODO: Test
    img_name = raw_input("Enter image name: ")
    img_path = ".." + os.path.sep + "TestImages" + os.path.sep + img_name  

    if not os.path.isfile(img_path):
        print img_name + " is not a file under the TestImages directory. Try again."
        return prompt_fpath()
    else:
        return img_path

def to_degrees( radian ):
    return radian * 180. / pi

def quantize_angle( ang ):
    if (ang < 22.5 and ang > -22.5) or (ang > 157.5) or (ang < -157.5):
        return 0  # Check East-West 
    elif (ang > 22.5 and ang < 67.5) or (ang < -112.5 and ang > -157.5):
        return 45  # Check Northeast-Southwest
    elif (ang > 67.5 and ang < 112.5) or (ang < -67.5 and ang > -112.5):
        return 90  # Check North-South
    elif (ang > 112.5 and ang < 157.5) or (ang < -22.5 and ang > -67.5 ):
        return 135 # Check Northwest-Southeast

def quantize_all( orientation ):
    quantized = numpy.zeros( orientation.shape )

    for row in range(len(orientation)):
        for col in range(len(orientation[row])):
            quantized[row][col] = ( quantize_angle(orientation[row][col]) )

    return quantized

def keep_maxima( F, gradient_angles ):
    maxima = numpy.ones( F.shape )
    at_top, at_bottom, at_left, at_right = False, False, False, False
    is_maxima = True 

    for row in range(len(F)):
        for col in range(len(F[row])):
            cur_angle = gradient_angles[row][col]
            cur_magnitude = F[row][col]

            at_top, at_bottom, at_left, at_right = False, False, False, False
            if row == 0:
                at_top = True
            if (row == len(F)-1):
                at_bottom = True
            if col == 0:
                at_left = True
            if (col == len(F[0])-1):
                at_right = True

            is_maxima = True 
            if cur_angle == 0:
                # Check East-West
                if not at_left:
                    if cur_magnitude < F[row][col-1]:
                        maxima[row][col] = 0
                        is_maxima = False
                if not at_right:
                    if cur_magnitude < F[row][col+1]:
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
    t_edges = numpy.zeros( I.shape )

    for row in range(len(I)):
        for col in range(len(I[row])):
            if not t_edges[row][col] and I[row][col] > t_high:
                find_edge_chains( I, t_edges, Q[row][col], row, col, t_low )
    
    return t_edges

def find_edge_chains( I, t_edges, direction, x, y, t_low ):
    # Find chains!
    # if I[x][y] > t_low, add to t_edges
    # else stop
    # t_edges[x][y] = 1
    # For unt_edges neighbors in the both directions...
        # if neither neighbor is available, stop!
        # else find_edge_chains

    magnitude = I[x][y]
    if magnitude > t_low and not t_edges[x][y]:
        ## Pixel passes threshold. Save and mark t_edges
        t_edges[x][y] = magnitude

        at_top, at_bottom, at_left, at_right = False, False, False, False
        if x == 0:
            at_top = True
        if (x == len(I)-1):
            at_bottom = True
        if y == 0:
            at_left = True
        if (y == len(I[0])-1):
            at_right = True

        if direction == 0:
            # Check East-West
            if not at_left and not t_edges[x][y-1]:
                find_edge_chains( I, t_edges, direction, x, y-1, t_low )
            if not at_right and not t_edges[x][y+1]:
                find_edge_chains( I, t_edges, direction, x, y+1, t_low )
        elif direction == 45:
            # Check Northeast-Southwest
            if not at_right and not at_top and not t_edges[x-1][y+1]:
                find_edge_chains( I, t_edges, direction, x-1, y+1, t_low )
            if not at_left and not at_bottom and not t_edges[x+1][y-1]:
                find_edge_chains( I, t_edges, direction, x+1, y-1, t_low )
        elif direction == 90:
            # Check North-South
            if not at_top and not t_edges[x-1][y]:
                find_edge_chains( I, t_edges, direction, x-1, y, t_low )
            if not at_bottom and not t_edges[x+1][y]:
                find_edge_chains( I, t_edges, direction, x+1, y, t_low )
        elif direction == 135:
            # Check Northwest-Southeast
            if not at_left and not at_top and not t_edges[x-1][y-1]:
                find_edge_chains( I, t_edges, direction, x-1, y-1, t_low )
            if not at_right and not at_bottom and not t_edges[x+1][y+1]:
                find_edge_chains( I, t_edges, direction, x+1, y+1, t_low )

if __name__ == "__main__":
    print "~~~ Edge, Line, & Feature Detector ~~~"

    ## Load Image
    # print "~~~~ Images are loaded from the 'TestImages' directory, so place your images there! ~~~~"
    # img_path = prompt_fpath()
    # img_path = raw_input("Enter image name: ")
    img_path = "../TestImages/building.jpg"
    I = skimage.img_as_float(skimage.io.imread(img_path))
    I = color.rgb2gray(I)  ## We want intensities
    I = misc.lena()

    ## Smooth (reduce noise) ##  TODO: Necessary?
    img_smooth = ndimage.gaussian_filter(I, sigma=sigma)

    ## Convolve ##
    dgx, dgy = gauss_partial_kernels()
    gx, gy = gauss_kernels()

    Fx1 = signal.convolve(img_smooth, [dgx], mode='same')
    Fx = signal.convolve(Fx1, gy, mode='same')

    Fy1 = signal.convolve(img_smooth, dgy, mode='same')
    Fy = signal.convolve(Fy1, [gx], mode='same')

    ## Gradient magnitude & direction ##
    F = numpy.sqrt( Fx**2 + Fy**2 )  
    D = to_degrees( numpy.arctan2( Fy, Fx ) )

    ## Nonmaximum suppression ##
    Q = quantize_all( D )
    I = keep_maxima( F, Q )

    ## Thresholding with hysteresis ##
    tI = threshold_edges( I, Q, 0.1, 0.05 )
    showim( tI )

