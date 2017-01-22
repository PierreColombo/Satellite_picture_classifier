# in this script we increase our data set by doing two transformations :
            # - flip
            # - mirror
            # - rotation 90
            # - rotation 180
            # - rotation 270
# of our whole data set like that we have 3 times more pictures with the same characteristics

import PIL
from PIL import Image, ImageOps

from pathlib import Path


for i in range(1,101) :

        # image

        path_img = '../../data_dir/images/' +  "satImage_%.3d" % i + '.png'
        img = Image.open(path_img)
        path_img = Path('../../data_dir/images_3/' +  "satImage_%.3d" % i + '.png')
        img.save(path_img)

        path_img_mirror = Path('../../data_dir/images_3/' +  "satImage_%.3d" % (100+i) + '.png')
        img_mirror = PIL.ImageOps.mirror(img)
        img_mirror.save(path_img_mirror)

        path_img_flipped = Path('../../data_dir/images_3/' +  "satImage_%.3d" % (200+i) + '.png')
        img_flipped = PIL.ImageOps.flip(img)
        img_flipped.save(path_img_flipped)

        path_img_ninety= Path('../../data_dir/images_3/' +  "satImage_%.3d" % (300+i) + '.png')
        img_ninety = img.rotate(90)
        img_ninety.save(path_img_ninety)


        path_img_mirror = Path('../../data_dir/images_3/' +  "satImage_%.3d" % (400+i) + '.png')
        img_mirror = PIL.ImageOps.mirror(img_ninety)
        img_mirror.save(path_img_mirror)

        path_img_flipped = Path('../../data_dir/images_3/' +  "satImage_%.3d" % (500+i) + '.png')
        img_flipped = PIL.ImageOps.flip(img_ninety)
        img_flipped.save(path_img_flipped)



        path_img_hundredeighty= Path('../../data_dir/images_3/' +  "satImage_%.3d" % (600+i) + '.png')
        img_hundredeighty = img.rotate(180)
        img_hundredeighty.save(path_img_hundredeighty)


        path_img_mirror = Path('../../data_dir/images_3/' +  "satImage_%.3d" % (700+i) + '.png')
        img_mirror = PIL.ImageOps.mirror(img_hundredeighty)
        img_mirror.save(path_img_mirror)

        path_img_flipped = Path('../../data_dir/images_3/' +  "satImage_%.3d" % (800+i) + '.png')
        img_flipped = PIL.ImageOps.flip(img_hundredeighty)
        img_flipped.save(path_img_flipped)


        path_img_twohundredseventy= Path('../../data_dir/images_3/' +  "satImage_%.3d" % (900+i) + '.png')
        img_twohundredseventy = img.rotate(270)
        img_twohundredseventy.save(path_img_twohundredseventy)

        path_img_mirror = Path('../../data_dir/images_3/' +  "satImage_%.3d" % (1000+i) + '.png')
        img_mirror = PIL.ImageOps.mirror(img_twohundredseventy)
        img_mirror.save(path_img_mirror)

        path_img_flipped = Path('../../data_dir/images_3/' +  "satImage_%.3d" % (1100+i) + '.png')
        img_flipped = PIL.ImageOps.flip(img_twohundredseventy)
        img_flipped.save(path_img_flipped)



        # labels
        path_lab = '../../data_dir/groundtruth/' +  "satImage_%.3d" % i + '.png'
        lab = Image.open(path_lab)
        path_img = Path('../../data_dir/groundtruth_3/' +  "satImage_%.3d" % i + '.png')
        lab.save(path_img)

        path_lab_mirror = Path('../../data_dir/groundtruth_3/' +  "satImage_%.3d" % (100+i) + '.png')
        lab_mirror = PIL.ImageOps.mirror(lab)
        lab_mirror.save(path_lab_mirror)

        path_lab_flipped = Path('../../data_dir/groundtruth_3/' +  "satImage_%.3d" % (200+i) + '.png')
        lab_flipped = PIL.ImageOps.flip(lab)
        lab_flipped.save(path_lab_flipped)



        path_lab_ninety= Path('../../data_dir/groundtruth_3/' +  "satImage_%.3d" % (300+i) + '.png')
        lab_ninety = lab.rotate(90)
        lab_ninety.save(path_lab_ninety)

        path_lab_mirror = Path('../../data_dir/groundtruth_3/' +  "satImage_%.3d" % (400+i) + '.png')
        lab_mirror = PIL.ImageOps.mirror(lab_ninety)
        lab_mirror.save(path_lab_mirror)

        path_lab_flipped = Path('../../data_dir/groundtruth_3/' +  "satImage_%.3d" % (500+i) + '.png')
        lab_flipped = PIL.ImageOps.flip(lab_ninety)
        lab_flipped.save(path_lab_flipped)



        path_lab_hundredeighty= Path('../../data_dir/groundtruth_3/' +  "satImage_%.3d" % (600+i) + '.png')
        lab_hundredeighty = lab.rotate(180)
        lab_hundredeighty.save(path_lab_hundredeighty)

        path_lab_mirror = Path('../../data_dir/groundtruth_3/' +  "satImage_%.3d" % (700+i) + '.png')
        lab_mirror = PIL.ImageOps.mirror(lab_hundredeighty)
        lab_mirror.save(path_lab_mirror)

        path_lab_flipped = Path('../../data_dir/groundtruth_3/' +  "satImage_%.3d" % (800+i) + '.png')
        lab_flipped = PIL.ImageOps.flip(lab_hundredeighty)
        lab_flipped.save(path_lab_flipped)



        path_lab_twohundredseventy= Path('../../data_dir/groundtruth_3/' +  "satImage_%.3d" % (900+i) + '.png')
        lab_twohundredseventy = lab.rotate(270)
        lab_twohundredseventy.save(path_lab_twohundredseventy)

        path_lab_mirror = Path('../../data_dir/groundtruth_3/' +  "satImage_%.3d" % (1000+i) + '.png')
        lab_mirror = PIL.ImageOps.mirror(lab_twohundredseventy)
        lab_mirror.save(path_lab_mirror)

        path_lab_flipped = Path('../../data_dir/groundtruth_3/' +  "satImage_%.3d" % (1100+i) + '.png')
        lab_flipped = PIL.ImageOps.flip(lab_twohundredseventy)
        lab_flipped.save(path_lab_flipped)
