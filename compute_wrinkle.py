import pickle
import os
from PIL import Image as im
import pandas as pd
import numpy as np

base_path = os.listdir('./data/depth/')
parent_dir = './data/depth/'
# goal is placed in a separate folder
goal_path = 'goal.pkl'
goal_dir = './data/'

def compute_standardized_sum(elem, my_dir):
	f_path = my_dir+elem
	with open(f_path, "rb") as f:
		img_dict = pickle.load(f)
		color_img = img_dict["color_img"]
		depth_img = img_dict["depth_img"]
		# crop both arrays to only have the pad region
		# 390x390 region
		left = 465
		right = 855
		top = 80
		bottom = 470
		color_img = color_img[top:bottom, left:right,]
		depth_img = depth_img[top:bottom, left:right]
		# you can use the code below to check if the cropped image is of the right dimension/look
		# img_out = im.fromarray(color_img)
		# img_out.save(elem[:-4]+'_reconstructed.png')
		# depth_df = pd.DataFrame(depth_img)
		# depth_df.to_csv('depth_df.csv')

		# use rgb to select the red indices of the picture
		# used color picker on various input images to determine this threshold
		red_bound = 110
		green_bound = 100
		blue_bound = 100
		cloth_indices = []
		cloth_pixel_count = 0
		for row_index in range(len(color_img)):
			for col_index in range(len(color_img[0])):
				if (color_img[row_index][col_index][0] > red_bound) and (color_img[row_index][col_index][1] < green_bound) and (color_img[row_index][col_index][2] < blue_bound):
					cloth_pixel_count += 1
					cloth_indices.append([row_index, col_index])
					# use the code below to change all the identified pixels to a different color, so you can see if your threshold worked out well
					# color_img[row_index][col_index][0] = 50
					# color_img[row_index][col_index][1] = 172
					# color_img[row_index][col_index][2] = 122
		# if you edited the image values above using the commented-out code, you can save the images to files and see how you did
		# img_out = im.fromarray(color_img)
		# img_out.save(elem[:-4]+'_reconstructed.png')

		# note that pixels locations on the cloth with higher height correspond to a lower value in the depth_img entries
		# this is probably because it is measuring the distance from that point to the camera
		# to invert its meaning into "height", we could subtract the highest possible value (representing the 'floor' by entries in depth_img
		# we can find this cap by running the command below
		# print(np.amax(depth_img))
		# when running across our 20 samples, we found that the highest value is generall 1 unit away from the camera

		# now we add all the values, then standardize by dividing the surface area
		z_sum = 0
		for elem1 in cloth_indices:
			z_sum += 1-(depth_img[elem1[0]][elem1[1]])
		standardized_z_sum = z_sum/cloth_pixel_count
	return standardized_z_sum



goal_standardized_sum = compute_standardized_sum(goal_path, goal_dir)
for elem in base_path:
	print('For ' + elem + ', the ratio is')
	print(goal_standardized_sum/compute_standardized_sum(elem, parent_dir))