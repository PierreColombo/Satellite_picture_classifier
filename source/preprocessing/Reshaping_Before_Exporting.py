import PIL
from PIL import Image, ImageOps
import numpy as np

from pathlib import Path


## we add 32 points : new size of the image 640
# list of possible patch size : 16, 20,40

# open path : path were images are
# start indice : indice of the first image "satImage_%.3d" % i +'.png'
# end_indice : same

def reshaping_before_mask( open_path, start_indice, end_indice) :
    for i in range(start_indice,end_indice) :
        print(i)


        # fil the path
        path = open_path+  "satImage_%.3d" % i + '.png'
        p = Path(path)
        img = Image.open(path)

        img_resized = np.asarray(img)[0:608,0:608,:]

        # save that beautiful picture
        imgs_comb = PIL.Image.fromarray( img_resized)
        print('saving')
        imgs_comb.save(p)


#reshaping_before_mask('../../test_dir/test_dir_40/', 1,51)
