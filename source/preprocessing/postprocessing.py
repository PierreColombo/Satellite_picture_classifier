from skimage.morphology import erosion, dilation
from skimage.morphology import disk
import os

import matplotlib.pyplot as plt
from PIL import Image

DISK_SIZE = 10

def post_processing(binary_picture , diskSize):
    selem = disk(diskSize)
    dilated = dilation(binary_picture, selem) #dilatation of the white regions
    selem = disk(diskSize)
    post_processed_picture = erosion(dilated, selem) #dilataion of the black regions

    return post_processed_picture


def test() :
    binary = Image.open('../../predictions_test_test/submission_003.png') # open colour image
    #binary = binary.convert('1') # convert image to black and white

    selem = disk(DISK_SIZE)
    dilated = dilation(binary, selem)
    plot_comparison(binary, dilated, 'dilation')
    selem = disk(DISK_SIZE)
    eroded = erosion(dilated, selem)
    plot_comparison(binary, eroded, 'erosion')


def plot_comparison(original, filtered, filter_name):

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True, sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax1.set_adjustable('box-forced')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(filter_name)
    ax2.axis('off')
    ax2.set_adjustable('box-forced')
    plt.show()

def convert_submission():
    postprocessed_dir = '../../postprocessed_ten/'
    if not os.path.isdir(postprocessed_dir):
        os.mkdir(postprocessed_dir)
    for i in range(1, 51):
        image_filename = '../../predictions_test/submission_' + '%.3d' % i + '.png'
        print (image_filename)
        pre = Image.open(image_filename)
        post = post_processing(pre, DISK_SIZE)
        post = Image.fromarray(post)
        post.save(postprocessed_dir + "submission_" + '%.3d' % i + ".png")

if __name__ == '__main__':
    #test()
    convert_submission()