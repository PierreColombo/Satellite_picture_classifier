import PIL
from PIL import Image, ImageOps
import numpy as np
from pathlib import Path



## we add 32 points : new size of the image 640
# list of possible patch size : 16, 20,40


# save_path : path we the image will be saved
# open path : path were images are
# start indice : indice of the first image "satImage_%.3d" % i +'.png'
# end_indice : same

def reshaping_for_patch( save_path, open_path, start_indice, end_indice) :
    patchsize = 32
    for i in range(start_indice,end_indice) :
        path = open_path+  "satImage_%.3d" % i + '.png'
        p = Path(save_path+"satImage_%.3d" % i + '.png')
        img = Image.open(path)

        img_larger = np.vstack( (np.asarray(img),np.asarray(PIL.ImageOps.flip(img))[0:patchsize,:,:]) )
        img_resize = np.hstack( (np.asarray(img_larger),np.asarray(PIL.ImageOps.mirror(PIL.Image.fromarray( img_larger)))[:,0:patchsize,:]  ))
        # save that beautiful picture
        imgs_comb = PIL.Image.fromarray( img_resize)
        imgs_comb.save(p)

