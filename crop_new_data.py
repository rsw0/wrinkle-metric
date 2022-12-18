from PIL import Image
import os

original_images_list = os.listdir('./data/rgb/')
# below assignment for testing purpose
# original_images_list = ['square_rgb_64.png', 'square_rgb_87.png']

for elem in original_images_list:
    im = Image.open('./data/rgb/'+elem)
    left = 465
    top = 80
    right = 855
    bottom = 470
    im1 = im.crop((left, top, right, bottom))
    im1.save('./data/cropped/' + elem[:-4]+'_cropped.png')
    # input below for testing
    # input("Press Enter to continue...")