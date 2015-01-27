import pylab
import numpy
from math import sqrt, pi, e
import skimage
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
    y, x = mgrid[-2:3, -2:3]
    x = x.astype(float)
    y = y.astype(float)

    ## Compute partial derivative values at x,y = -2, -1, 0, 1, 2
    dgx = -x / (sigma**3. * sqrt(2.*pi)) * e**(-x**2. / (2.*sigma**2.))
    dgy = -y / (sigma**3. * sqrt(2.*pi)) * e**(-y**2. / (2.*sigma**2.))

    ## Normalize values
    dgx = dgx / sum(sum(abs(dgx)))
    dgy = dgy / sum(sum(abs(dgy)))

    return (dgx, dgy) 

def gauss_kernels():
    ## 1D kernel for bivariate gaussian ##
    '''
    x_real = numpy.zeros(5)
    y_real = numpy.zeros((5, 1))
    x = [abs(x) for x in range(-2, 3)]
    y = [abs(y) for y in range(-2, 3)]
    '''
    y, x = mgrid[-2:3, -2:3]
    x = x.astype(float)
    y = y.astype(float)

    ## Compute gaussian values at x,y = -2, -1, 0, 1, 2
    gx = 1 / (sigma * sqrt(2 * pi)) * e**(-x**2 / (2 * sigma**2))
    gy = 1 / (sigma * sqrt(2 * pi)) * e**(-y**2 / (2 * sigma**2))

    ## Normalize values
    gx = gx / sum(sum(abs(gx)))
    gy = gy / sum(sum(abs(gy)))

    return (gx, gy)

if __name__ == "__main__":
    print "~~~ Edge, Line, & Feature Detector ~~~"

    ## Load Image
    # img_path = raw_input("Enter image filepath: ")
    img_path = "../TestImages/hyde2.jpg"
    I = skimage.img_as_float(skimage.io.imread(img_path))

    # I = color.rgb2gray(I) # convert to grayscale

    ## Smooth (reduce noise) ##
    sig = [sigma, sigma, 0] ## TODO: Grayscale?
    img_smooth = ndimage.gaussian_filter(I, sigma=sig)

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

    ## Convolve with partial derivative of bivariate Gaussian, for each channel
    # (dgx, dgy) = gauss_partial_kernels()
    # (x_real, y_real) = gauss_kernels()

    # inter_x_r = signal.convolve2d(img_smooth[:, :, 0], [dgx], 'same')
    # gradient_x_r = signal.convolve2d(inter_x_r, y_real)

    # inter_x_g = signal.convolve2d(img_smooth[:, :, 1], [dgx], 'same')
    # gradient_x_g = signal.convolve2d(inter_x_r, y_real)

    # inter_x_b = signal.convolve2d(img_smooth[:, :, 2], [dgx], 'same')
    # gradient_x_b = signal.convolve2d(inter_x_r, y_real)

    # gradient_x = [ list(e) for e in zip(gradient_x_r, gradient_x_g, gradient_x_b) ]
    # print gradient_x

    # inter_y_r = signal.convolve2d(img_smooth[:, :, 0], [x_real], 'same')
    # gradient_y_r = signal.convolve2d(inter_y_r, dgy)

    # inter_y_g = signal.convolve2d(img_smooth[:, :, 1], [x_real], 'same')
    # gradient_y_g = signal.convolve2d(inter_y_g, dgy)

    # inter_y_b = signal.convolve2d(img_smooth[:, :, 2], [x_real], 'same')
    # gradient_y_b = signal.convolve2d(inter_y_b, dgy)

    # pylab.imshow(img_smooth)
    # pylab.show()
