import cv2
import numpy as np
import random as rng
def shift_test_image(goal_rgb_image_path, test_rgb_image_path)
    goal_rgb = cv2.imread(goal_rgb_image_path)
    test_rgb = cv2.imread(test_rgb_image_path)
    blank_canvas = np.zeros_like(goal_rgb)
    goal_rgb_hsv = cv2.cvtColor(goal_rgb, cv2.COLOR_BGR2HSV)
    test_rgb_hsv = cv2.cvtColor(test_rgb, cv2.COLOR_BGR2HSV)
    goal_mask1 = cv2.inRange(goal_rgb_hsv, (0,50,20), (15,255,255))
    goal_mask2 = cv2.inRange(goal_rgb_hsv, (160,50,20), (180,255,255))
    goal_mask = cv2.bitwise_or(goal_mask1, goal_mask2)
    test_mask1 = cv2.inRange(test_rgb_hsv, (0,50,20), (15,255,255))
    test_mask2 = cv2.inRange(test_rgb_hsv, (160,50,20), (180,255,255))
    test_mask = cv2.bitwise_or(test_mask1, test_mask2)
    goal_rgb[goal_mask>0]=(0,255,0)
    goal_rgb_hsv = cv2.cvtColor(goal_rgb, cv2.COLOR_BGR2HSV)
    test_rgb[test_mask>0]=(0,255,0)
    test_rgb_hsv = cv2.cvtColor(test_rgb, cv2.COLOR_BGR2HSV)
    goal_contours = cv2.findContours(goal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    goal_contours = goal_contours[0] if len(goal_contours) == 2 else goal_contours[1]
    test_contours = cv2.findContours(test_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    test_contours = test_contours[0] if len(test_contours) == 2 else test_contours[1]
    test_bw = cv2.drawContours(blank_canvas.copy(), test_contours, -1, (255,255,255), -1)

    # find the minimum enclosing circle for the goal
    goal_contours_poly = [None]*len(goal_contours)
    goal_centers = [None]*len(goal_contours)
    goal_radius = [None]*len(goal_contours)
    for i, c in enumerate(goal_contours):
        goal_contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        goal_centers[i], goal_radius[i] = cv2.minEnclosingCircle(goal_contours_poly[i])
    goal_center_x = int(goal_centers[0][0])
    goal_center_y = int(goal_centers[0][1])
    goal_radius_int = int(goal_radius[0])+1
    # find the bounding box of the test
    test_contours_poly = [None]*len(test_contours)
    test_boundRect = [None]*len(test_contours)
    for i, c in enumerate(test_contours):
        test_contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        test_boundRect[i] = cv2.boundingRect(test_contours_poly[i])
    test_bound_x = test_boundRect[0][0] if test_boundRect[0][0]%2 == 0 else test_boundRect[0][0]+1
    test_bound_y = test_boundRect[0][1] if test_boundRect[0][1]%2 == 0 else test_boundRect[0][1]+1
    test_bound_width = test_boundRect[0][2] if test_boundRect[0][2]%2 == 0 else test_boundRect[0][2]+1
    test_bound_height = test_boundRect[0][3] if test_boundRect[0][3]%2 == 0 else test_boundRect[0][3]+1
    # use the center of the minimum enclosing circle of the goal as the center to paste the cropped image
    test_bw_crop = test_bw[test_bound_y:test_bound_y+test_bound_height, test_bound_x:test_bound_x+test_bound_width]
    shifted_output = blank_canvas.copy()
    # shifted output is the shifted test rgb image, centered on the goal
    shifted_output[(goal_center_y-(test_bound_height//2)):(goal_center_y+(test_bound_height//2)), (goal_center_x-(test_bound_width//2)):(goal_center_x+(test_bound_width//2))] = test_bw_crop
    return shifted_output