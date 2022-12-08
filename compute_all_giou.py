from giou import compute_giou
from shift_test import shift_test_image
import os
import cv2

# we only need to goal to shift the test
# note that rotate_0 file is the original file
goal_rgb_path_temp = './data/output/goal_rotated/goal_rgb_rotated_0.png'
test_rgb_path = './data/small_test_dataset/test1_rgb.png'
test_rgb = shift_test_image(goal_rgb_path_temp, test_rgb_path)
cv2.imwrite('./data/output/shifted_test/shifted_test_image.png', test_rgb)
test_rgb_path = './data/output/shifted_test/shifted_test_image.png'

path = "./data/output/goal_rotated/"
goal_list = os.listdir(path)

giou_list = []

for goal in goal_list:
    goal_rgb_path = path+goal
    giou_temp = compute_giou(goal_rgb_path, test_rgb_path)
    giou_list.append(giou_temp)

print(giou_list)
max = giou_list[0]
index = 0
for i in range(1,len(giou_list)):
    if giou_list[i] > max:
        max = giou_list[i]
        index = i
print(f'Index of the maximum value is : {index}')
print(max)
