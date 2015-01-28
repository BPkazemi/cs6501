import pylab
import numpy
import os, os.path
import skimage
from math import sqrt, pi, e
from skimage import io, color
from scipy import ndimage, signal, mgrid

## GLOBAL VARS ##
sigma = 1.0

def gauss_partial_kernels():
    ## 1D kernel for x and y partial derivatives of bivariate gaussian ##
    '''
    dgx = numpy.zeros(5)
    dgy = numpy.zeros((5, 1))
    x = [abs(x) for x in range(-2, 3)]
    y = [abs(y) for y in range(-2, 3)]
    '''
    x = mgrid[-2:3]
    y = numpy.swapaxes([mgrid[-2:3]], 0, 1)
    x = x.astype(float)
    y = y.astype(float)

    ## Compute partial derivative values at x,y = -2, -1, 0, 1, 2
    dgx = -x / (sigma**3. * sqrt(2.*pi)) * e**(-x**2. / (2.*sigma**2.))
    dgy = -y / (sigma**3. * sqrt(2.*pi)) * e**(-y**2. / (2.*sigma**2.))

    ## Normalize values
    dgx = dgx / sum(abs(dgx))
    dgy = dgy / sum(abs(dgy))
    
    return (dgx, dgy) 

def gauss_kernels():
    ## 1D kernel for bivariate gaussian ##
    '''
    x_real = numpy.zeros(5)
    y_real = numpy.zeros((5, 1))
    x = [abs(x) for x in range(-2, 3)]
    y = [abs(y) for y in range(-2, 3)]
    '''
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

if __name__ == "__main__":
    print "~~~ Edge, Line, & Feature Detector ~~~"

    ## Load Image
    # print "~~~~ Images are loaded from the 'TestImages' directory, so place your images there! ~~~~"
    # img_path = prompt_fpath()
    # img_path = raw_input("Enter image name: ")
    img_path = "../TestImages/hyde2.jpg"
    I = skimage.img_as_float(skimage.io.imread(img_path))
    I = color.rgb2gray(I)  ## We want intensities

    ## Smooth (reduce noise) ##
    img_smooth = ndimage.gaussian_filter(I, sigma=sigma)

    dgx, dgy = gauss_partial_kernels()
    gx, gy = gauss_kernels()

    imgx = signal.convolve(img_smooth, [dgx], mode='same')
    imgx2 = signal.convolve(imgx, gy, mode='same')

    imgy = signal.convolve(img_smooth, dgy, mode='same')
    imgy2 = signal.convolve(imgy, [gx], mode='same')

    F = numpy.sqrt( imgx2**2 + imgy2**2 )  ## Gradient magnitude
    showim( F )

    '''
    dgx, dgy = gauss_partial_kernels()
    gx, gy = gauss_kernels()

    imgx = signal.convolve(img_smooth[:, :, 0], dgx, mode='same')
    imgy = signal.convolve(img_smooth[:, :, 0], dgy, mode='same')
    Wx = signal.convolve(imgx, gy, mode='same')
    Wy = signal.convolve(imgy, gx, mode='same')
    print(type(Wx))

    W = abs(Wx)**2 + abs(Wy)**2
    pylab.imshow(W)
    pylab.show()

    # print img_smooth[:, :, 0]
    # print imgx
    '''
