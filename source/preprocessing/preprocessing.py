import numpy as np
from scipy import misc
from PIL import Image
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
from skimage.feature import canny
from skimage import filters
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

PROJECT_ROOT = '../../'

def generate_features(image_filename):
    # RGB
    img = mpimg.imread(image_filename)

    # # Canny edge detection
    # filtered_imgs = preprocessing_edge_canny_hough(image_filename)
    # img = np.dstack((img, filtered_imgs[0]))
    #
    # # Sobel edge detection
    # sobeled_img = preprocessing_edge_sobleFilter(Image.fromarray(img, 'RGB').convert('1'))
    # img = np.dstack((img, sobeled_img.astype(np.float32)))

    return img


def get_number_of_channels():
    img = generate_features(PROJECT_ROOT+'data_dir/images/satImage_001.png')
    return img.shape[2]


# Create the result of the picture by the sobel filter (edge detection)
def preprocessing_edge_sobleFilter(binaryPicture) :
    return filters.sobel(binaryPicture)


def average(pixel):
        return (pixel[0]* 299. / 1000 + pixel[1]* 587. / 1000 + pixel[2]* 114. / 1000)


# Create the result of the picture by the hough transform (edge detection)
# canny filter :
        # by default : parameter = 6
# hough transform :
        # by default  : lineLength = 5
        # by default  : lineGap = 3
        # by default  : Threshodl = 10
def preprocessing_edge_canny_hough(path, parameter=6, lineLength=5, lineGap=3, Threshold=10):

   image = misc.imread(path)
   grey = np.zeros((image.shape[0], image.shape[1])) # init 2D numpy array
   for rownum in range(len(image)):
      for colnum in range(len(image[rownum])):
         grey[rownum][colnum] = average(image[rownum][colnum])

   edges = canny(grey, sigma=parameter)
   hough = probabilistic_hough_line(edges, threshold=Threshold, line_length=lineLength,
                                     line_gap=lineGap)

   return edges,hough

# parameter = 2
# lineLength = 50
# lineGap = 3
# Threshodl = 10
#
# path ='../../data_dir/images/satImage_006.png' # open colour image
#
# edges,lines = preprocessing_edge_canny_hough(path, parameter, lineLength , lineGap , Threshodl)
# print(len(lines))
#
# fig, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(8,4), sharex=True, sharey=True)
#
# ax1.imshow(edges, cmap=plt.cm.gray)
# ax1.set_title('Input image')
# ax1.set_axis_off()
# ax1.set_adjustable('box-forced')
# for line in lines:
#             p0, p1 = line
#             ax2.plot((p0[0], p1[0]), (p0[1], p1[1]))
# ax2.imshow(edges)
# ax2.set_title('Canny edges')
# ax2.set_axis_off()
# ax2.set_adjustable('box-forced')
# image = misc.imread(path)
# ax3.imshow(image)
# plt.show()


def test() :

    # Load image
    image = misc.imread(PROJECT_ROOT+'data_dir/images/satImage_004.png')



    grey = np.zeros((image.shape[0], image.shape[1])) # init 2D numpy array
    for rownum in range(len(image)):
       for colnum in range(len(image[rownum])):
          grey[rownum][colnum] = average(image[rownum][colnum])


    binary = Image.open(PROJECT_ROOT+'data_dir/images/satImage_004.png') # open colour image
    binary = binary.convert('1') # convert image to black and white


    #### Hough #####


    for i in range(6,8):
        image = grey
        edges = canny(image, sigma=i)
        #edges = filters.sobel(binary)


        lines = probabilistic_hough_line(edges, threshold=10, line_length=5,
                                         line_gap=3)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8,4), sharex=True, sharey=True)

        ax1.imshow(image, cmap=plt.cm.gray)
        ax1.set_title('Input image')
        ax1.set_axis_off()
        ax1.set_adjustable('box-forced')

        ax2.imshow(edges, cmap=plt.cm.gray)
        ax2.set_title('Canny edges'+str(i))
        ax2.set_axis_off()
        ax2.set_adjustable('box-forced')

        ax3.imshow(edges * 0)

        for line in lines:
            p0, p1 = line
            ax3.plot((p0[0], p1[0]), (p0[1], p1[1]))

        ax3.set_title('Probabilistic Hough')
        ax3.set_axis_off()
        ax3.set_adjustable('box-forced')
        plt.show()


if __name__ == '__main__':
    print(get_number_of_channels())
