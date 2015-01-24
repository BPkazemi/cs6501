import pylab
import skimage
import skimage.io

if __name__ == "__main__":
    print "~~~ Canny Edge Detector, Line Detector, & Feature Detector ~~~"

    ## Read img ##
    img_path = raw_input("Enter image filepath: ")
    I = img_as_float(skimage.io.imread(path))
