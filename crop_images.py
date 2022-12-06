from PIL import Image
import os

original_images_list = os.listdir('./data/original/square')
# below assignment for testing purpose
# original_images_list = ['square_rgb_64.png', 'square_rgb_87.png']

for elem in original_images_list:
    im = Image.open('data/original/square/'+elem)
    left = 320
    top = 200
    right = 900
    bottom = 630
    im1 = im.crop((left, top, right, bottom))
    im1.save('./data/cropped/square/'+elem[:-4]+'_cropped.png')
    # input below for testing
    # input("Press Enter to continue...")