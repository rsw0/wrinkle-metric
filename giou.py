import cv2
import numpy as np
import random as rng

# variation of method provided by 
# https://stackoverflow.com/questions/8552364/opencv-detect-contours-intersection
def compute_giou(goal_rgb_image_path, test_rgb_image_path):
    # 0. read goal image and test image
    goal_rgb = cv2.imread(goal_rgb_image_path)
    test_rgb = cv2.imread(test_rgb_image_path)
    # test_depth = cv2.imread(test_depth_image_path)
    # create a blank image filled with zeros wtih the same size as the input 
    # (two inputs must have the same size so using either one would be fine)
    blank_canvas = np.zeros_like(goal_rgb)

    # 1. segment the cloth
    goal_rgb_hsv = cv2.cvtColor(goal_rgb, cv2.COLOR_BGR2HSV)
    test_rgb_hsv = cv2.cvtColor(test_rgb, cv2.COLOR_BGR2HSV)
    goal_mask1 = cv2.inRange(goal_rgb_hsv, (0,50,20), (15,255,255))
    goal_mask2 = cv2.inRange(goal_rgb_hsv, (160,50,20), (180,255,255))
    goal_mask = cv2.bitwise_or(goal_mask1, goal_mask2)




    white_sensitivity = 15
    lower_white = (0,0,255-white_sensitivity)
    upper_white = (255,white_sensitivity,255)
    test_mask = cv2.inRange(test_rgb_hsv, lower_white, upper_white)

    #test_mask1 = cv2.inRange(test_rgb_hsv, (0,50,20), (15,255,255))
    #test_mask2 = cv2.inRange(test_rgb_hsv, (160,50,20), (180,255,255))
    #test_mask = cv2.bitwise_or(test_mask1, test_mask2)



    goal_rgb[goal_mask>0]=(0,0,255)
    goal_rgb_hsv = cv2.cvtColor(goal_rgb, cv2.COLOR_BGR2HSV)
    test_rgb[test_mask>0]=(0,0,255)
    test_rgb_hsv = cv2.cvtColor(test_rgb, cv2.COLOR_BGR2HSV)

    # 2. find the contours of goal and test
    goal_contours = cv2.findContours(goal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    goal_contours = goal_contours[0] if len(goal_contours) == 2 else goal_contours[1]
    goal_area = 0
    for c in goal_contours:
        goal_area += cv2.contourArea(c)
        cv2.drawContours(goal_rgb,[c], 0, (0,0,0), 2)
    #cv2.imwrite('./data/output/contoured_goal_rgb.png', goal_rgb)
    test_contours = cv2.findContours(test_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    test_contours = test_contours[0] if len(test_contours) == 2 else test_contours[1]
    test_area = 0
    for c in test_contours:
        test_area += cv2.contourArea(c)
        cv2.drawContours(test_rgb,[c], 0, (0,0,0), 2)
    #cv2.imwrite('./data/output/contoured_test_rgb.png', test_rgb)

    # 3. create two images with the two contours
    goal_contours_bw = cv2.drawContours(blank_canvas.copy(), goal_contours, -1, (255,255,255), -1)
    test_contours_bw = cv2.drawContours(blank_canvas.copy(), test_contours, -1, (255,255,255), -1)
    #cv2.imwrite('./data/output/contoured_goal_rgb.png', goal_contours_bw)
    #cv2.imwrite('./data/output/contoured_test_rgb.png', test_contours_bw)

    # 4. find intersection and union via bitwise_and and bitwise_or
    goal_test_intersection = cv2.bitwise_and(goal_contours_bw, test_contours_bw)
    goal_test_union = cv2.bitwise_or(goal_contours_bw, test_contours_bw)
    #cv2.imwrite('./data/output/goal_test_intersection.png', goal_test_intersection)
    #cv2.imwrite('./data/output/goal_test_union.png', goal_test_union)

    # 5. Find area of intersection and union by applying a mask to those regions
    goal_test_intersection_hsv = cv2.cvtColor(goal_test_intersection, cv2.COLOR_BGR2HSV)
    goal_test_union_hsv = cv2.cvtColor(goal_test_union, cv2.COLOR_BGR2HSV)
    white_sensitivity = 15
    lower_white = (0,0,255-white_sensitivity)
    upper_white = (255,white_sensitivity,255)
    goal_test_intersection_mask = cv2.inRange(goal_test_intersection_hsv, lower_white, upper_white)
    goal_test_union_mask = cv2.inRange(goal_test_union_hsv, lower_white, upper_white)
    goal_test_intersection_contours = cv2.findContours(goal_test_intersection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    goal_test_intersection_contours = goal_test_intersection_contours[0] if len(goal_test_intersection_contours) == 2 else goal_test_intersection_contours[1]
    goal_test_union_contours = cv2.findContours(goal_test_union_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    goal_test_union_contours = goal_test_union_contours[0] if len(goal_test_union_contours) == 2 else goal_test_union_contours[1]
    goal_test_intersection_area = 0
    for c in goal_test_intersection_contours:
        goal_test_intersection_area += cv2.contourArea(c)
    goal_test_union_area = 0
    for c in goal_test_union_contours:
        goal_test_union_area += cv2.contourArea(c)
    #print(goal_area)
    #print(test_area)
    #print(goal_test_intersection_area)
    #print(goal_test_union_area)

    # 6. find the minimum bounding box of the two contours and find its area
    contours_poly = [None]*len(goal_test_union_contours)
    union_boundRect = [None]*len(goal_test_union_contours)
    for i, c in enumerate(goal_test_union_contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        union_boundRect[i] = cv2.boundingRect(contours_poly[i])
    # print(union_boundRect)
    goal_test_union_copy = goal_test_union_hsv.copy()
    for i in range(len(goal_test_union_contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv2.drawContours(goal_test_union_copy,[c], 0, (0,0,0), 2)
        cv2.rectangle(goal_test_union_copy, (int(union_boundRect[i][0]), int(union_boundRect[i][1])), \
            (int(union_boundRect[i][0]+union_boundRect[i][2]), int(union_boundRect[i][1]+union_boundRect[i][3])), color, 2)
    #cv2.imwrite('./data/output/union_minimum_bounding_box.png', goal_test_union_copy)
    union_x = union_boundRect[0][0]
    union_y = union_boundRect[0][1]
    union_width = union_boundRect[0][2]
    union_height = union_boundRect[0][3]
    union_bounding_box_area = union_height*union_width

    # 7. obtain GIoU
    giou = (goal_test_intersection_area/goal_test_union_area)-(abs(union_bounding_box_area/goal_test_union_area)/abs(union_bounding_box_area))
    return giou




""" goal_rgb_path = './data/small_test_dataset/goal_rgb.png'
test_rgb_path = 'lolol.png'
x = compute_giou(goal_rgb_path, test_rgb_path)
print(x) """