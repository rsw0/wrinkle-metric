import cv2
import numpy as np
import random as rng

# Generate the various rotations of the goal state
# 0. read images, create a copy in case you want to visualize anything 
# (so that you don't alter the original file)
# also reading in the depth image
goal_rgb = cv2.imread('./data/small_test_dataset/goal_rgb.png')
goal_rgb_copy = goal_rgb.copy()
# note that we want the color space to include alpha values so that we can paste later
goal_rgb_copy_for_rotate = goal_rgb.copy()
goal_depth = cv2.imread('./data/small_test_dataset/goal_depth.png')

# 1. Converting to HSV, and take only the hue value
# note that the second parameter defines how we encode and decode the colors
goal_rgb_hsv = cv2.cvtColor(goal_rgb, cv2.COLOR_BGR2HSV)

# 2. changing all red to the same red (we want the entire piece of cloth
# to have the same color). This step segments the object
# we set it to red in the line goal_rgb_hsv[mask>0]=(0,255,0)
mask1 = cv2.inRange(goal_rgb_hsv, (0,50,20), (15,255,255))
mask2 = cv2.inRange(goal_rgb_hsv, (160,50,20), (180,255,255))
mask = cv2.bitwise_or(mask1, mask2)
goal_rgb[mask>0]=(0,255,0)
goal_rgb_hsv = cv2.cvtColor(goal_rgb, cv2.COLOR_BGR2HSV)

# 3. Obtain only the hue parameter, and find the the area of the masked region
# finding area is not necessary for this particular task, but if we want to extend
# it to include multiple objects, it could be useful
# note that findContours returns two variables, first is contour, second is hierachy
# hierachy has information about how various contours are related
goal_hue,_,_ = cv2.split(goal_rgb_hsv)
goal_contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
goal_contours = goal_contours[0] if len(goal_contours) == 2 else goal_contours[1]
goal_area = 0
for c in goal_contours:
    goal_area += cv2.contourArea(c)
    # cv2.drawContours(goal_rgb_copy,[c], 0, (0,0,0), 2)
# cv2.imwrite('contoured_rgb.png', goal_rgb_copy)

# 4. Find the minimum enclosing circle of the goal state of the cloth
# Approximate contours to polygons + get bounding rects and circles
# https://docs.opencv.org/4.x/da/d0c/tutorial_bounding_rects_circles.html
contours_poly = [None]*len(goal_contours)
boundRect = [None]*len(goal_contours)
centers = [None]*len(goal_contours)
radius = [None]*len(goal_contours)
for i, c in enumerate(goal_contours):
    contours_poly[i] = cv2.approxPolyDP(c, 3, True)
    boundRect[i] = cv2.boundingRect(contours_poly[i])
    centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])
# obtain pixel locations of the bounding box and the bounding circle below
for i in range(len(goal_contours)):
    color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
    cv2.drawContours(goal_rgb_copy,[c], 0, (0,0,0), 2)
    cv2.rectangle(goal_rgb_copy, (int(boundRect[i][0]), int(boundRect[i][1])), \
        (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
    cv2.circle(goal_rgb_copy, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)
cv2.imwrite('./data/output/goal_with_bounding.png', goal_rgb_copy)

# 5. Crop the region defined by the minimum enclosing circle
# round the center and radius of the circle to int
center_int = (int(centers[0][0]),int(centers[0][1]))
radius_int = int(radius[0])+1
# draw_circle = cv2.circle(goal_rgb_copy, center_int, radius_int, color=(0, 0, 255), thickness=-1)
# cv2.imwrite('goal_filled_circle.png', draw_circle)
x_center = center_int[0]
y_center = center_int[1]
# draw filled circles in white on black background as masks
circle_mask = np.zeros_like(goal_rgb)
circle_mask = cv2.circle(circle_mask, (x_center,y_center), radius_int, (255,255,255), -1)
# Bitwise-and for Range of Interest
goal_roi = cv2.bitwise_and(goal_rgb, circle_mask)
# Find a square bounding box that bounds the ROI (note that this is different from the minimum bounding box
# produced earlier, we are now bounding the circular region)
circle_mask = cv2.cvtColor(circle_mask, cv2.COLOR_BGR2GRAY)
x,y,w,h = cv2.boundingRect(circle_mask)
cropped_goal_circle = goal_roi[y:y+h,x:x+w]
circle_mask = circle_mask[y:y+h,x:x+w]
# now, remove the black background by setting transparency to 0
cropped_goal_circle_BGRA = cv2.cvtColor(cropped_goal_circle, cv2.COLOR_BGR2BGRA)
# replace non-masked area with with white color and transparent
# since we defined the mask earlier, there's no need to check for colors
cropped_goal_circle_BGRA[circle_mask==0] = [255,255,255,0]
cv2.imwrite('./data/output/cropped_goal_circle.png', cropped_goal_circle_BGRA)

# 6. Generate all possible rotations of the cropped region, and paste them onto the original
# image and save all of them
def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1])/2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result
# define a function to perform alpha blending on a cropped out region of the original 
# image, because simply pasting won't work as alpha values are ignored
def alpha_blend(small_foreground, background):
    """
    Puts a small BGRA picture in front of a larger BGR background.
    :param small_foreground: The overlay image. Must have 4 channels.
    :param background: The background. Must have 3 channels.
    :param top: Y position where to put the overlay.
    :param left: X position where to put the overlay.
    # not top and left not needed in this case
    :return: a copy of the background with the overlay added.
    """
    result = background.copy()
    # From everything I read so far, it seems we need the alpha channel separately
    # so let's split the overlay image into its individual channels
    fg_b, fg_g, fg_r, fg_a = cv2.split(small_foreground)
    # Make the range 0...1 instead of 0...255
    fg_a = fg_a / 255.0
    # Multiply the RGB channels with the alpha channel
    label_rgb = cv2.merge([fg_b * fg_a, fg_g * fg_a, fg_r * fg_a])
    # Work on a part of the background only
    height, width = small_foreground.shape[0], small_foreground.shape[1]
    part_of_bg = result [y:y+h,x:x+w]
    # Same procedure as before: split the individual channels
    bg_b, bg_g, bg_r = cv2.split(part_of_bg)
    # Merge them back with opposite of the alpha channel
    part_of_bg = cv2.merge([bg_b * (1 - fg_a), bg_g * (1 - fg_a), bg_r * (1 - fg_a)])
    # Add the label and the part of the background
    cv2.add(label_rgb, part_of_bg, part_of_bg)
    # Replace a part of the background
    result[y:y+h,x:x+w] = part_of_bg
    return result
angle_list = np.arange(360)[1:]
for angle in angle_list:
    rotated_cropped_goal_image = rotate_image(cropped_goal_circle_BGRA, angle)
    rotated_final_goal_image = alpha_blend(rotated_cropped_goal_image, goal_rgb_copy_for_rotate)
    cv2.imwrite('./data/output/goal_rotated/goal_rgb_rotated_'+str(angle)+'.png', rotated_final_goal_image)





""" # code below is a more generalized version for finding bounding box of multiple objects based on color
# https://stackoverflow.com/questions/50051916/bounding-box-on-objects-based-on-color-python
# Find dominant hues by counting the occurrences of each hue using numpy.bincount
# flattened the hue channel image to make it one-dimensional
# np_bincounts returns an array of the same length as goal_hue.flatten(), where the value at
# index i is the bin number that hue i belongs to. 
# The number of bins (of size 1) is one larger than the largest value in x
# note that print(color_bins.shape) would return size (180,) because in cv2 hue values range from 0 to 180
color_bins = np.bincount(goal_hue.flatten())
# 5. Find which ones are common enough using numpy.where
# note that how you define the constant MIN_PIXEL_CNT_PCT will determine your "sensitivity" in 
# identifying an object. It coulds the percentage of pixels that have a particular hue
# and if the numbers of hues representing an object is does not occur this frequently, it won't
# be considered as an object. You will ultimately get a bounding box for each element in the variable "peaks"
# we previously changed the color of the cloth to a uniform color and computed its area. here we use the threshold/cutoff
# as 95% of the value of the area of the cloth, so that we only select the cloth and the whitespace
# the value of goal_area is set earlier, which will be adaptive to the input data given
min_pixel_count_percentage = (goal_area*0.95/len(goal_hue.flatten()))
# peaks = np.where(color_bins > (goal_hue.size * min_pixel_count_percentage))[0]
peaks = np.where(color_bins > (goal_hue.size * min_pixel_count_percentage))[0][1]
print(peaks)
# 6. Find the shape of the "object" matching each peak hue

# coverting to black and white
# goal_grey = cv2.imread('test_data/goal_rgb.png', cv2.IMREAD_GRAYSCALE)
# (thresh, goal_bw) = cv2.threshold(goal_grey, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) """