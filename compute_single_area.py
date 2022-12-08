import cv2
import numpy as np
import random as rng

# variation of method provided by 
# https://stackoverflow.com/questions/8552364/opencv-detect-contours-intersection
def indiv_area(input_path):
    # 0. read goal image and test image
    goal_rgb = cv2.imread(input_path)
    blank_canvas = np.zeros_like(goal_rgb)

    # 1. segment the cloth
    goal_rgb_hsv = cv2.cvtColor(goal_rgb, cv2.COLOR_BGR2HSV)
    goal_mask1 = cv2.inRange(goal_rgb_hsv, (0,50,20), (15,255,255))
    goal_mask2 = cv2.inRange(goal_rgb_hsv, (160,50,20), (180,255,255))
    goal_mask = cv2.bitwise_or(goal_mask1, goal_mask2)
    goal_rgb[goal_mask>0]=(0,255,0)
    goal_rgb_hsv = cv2.cvtColor(goal_rgb, cv2.COLOR_BGR2HSV)

    # 2. find the contours of goal and test
    goal_contours = cv2.findContours(goal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    goal_contours = goal_contours[0] if len(goal_contours) == 2 else goal_contours[1]
    goal_area = 0
    for c in goal_contours:
        goal_area += cv2.contourArea(c)
        cv2.drawContours(goal_rgb,[c], 0, (0,0,0), 2)
    output_img = cv2.imwrite('testtest.png', goal_rgb)
    return goal_area

print(indiv_area('blender_test_rgb.jpg'))