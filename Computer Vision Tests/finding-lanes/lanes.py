# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 08:12:55 2019

@author: moeelm
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(image):
    #Edge detection
        #1. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        #2. Reduce noise: Gaussian blur aka smooth image
    blur = cv2.GaussianBlur(gray, (5,5), 0)
        #3. Canny method, computes the gradient by taking the derivative
    canny = cv2.Canny(blur, 50, 150)
    return canny
    
def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    #how far will my lines extend?
    y1 = image.shape[0]
    #It will go 3/5 up in the image
    y2 = int(y1*(0.66))
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])
    
def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])
def region_of_interest(image):
    #Region of Interest
    height = image.shape[0]
    
    #Take the triangle as the region of interest
    polygons = np.array([[(200,height), (1100,height), (550,250)]])
    
    mask = np.zeros_like(image)
    
    #Fill polygon area with white
    cv2.fillPoly(mask, polygons, 255)
        
    #Bitwise and operation 
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image
    
def display_lines(image, lines):
    line_image = np.zeros_like(image)
    #check if array is not empty
    if lines is not None:
        for x1,y1,x2,y2 in lines:
            cv2.line(line_image, (x1,y1), (x2,y2), (255,0,0), 10)
    return line_image
#UNCOMMENT TO DO INDIVIDUAL IMAGE
#image = cv2.imread('test_image.png')
#lane_image = np.copy(image)
#
#canny_image = canny(lane_image)
#cropped_image = region_of_interest(canny_image)
#
##Hough Transform Detection Algorithm
#lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
#
#averaged_lines = average_slope_intercept(lane_image, lines)
#line_image = display_lines(lane_image, averaged_lines)
#
#combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
##Testing code
#cv2.imshow('result', combo_image)
#cv2.waitKey(1000)


#Finding Lanes in Video
cap = cv2.VideoCapture('test2.mp4')
while (cap.isOpened()):
    _,frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    
    #Hough Transform Detection Algorithm
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow('result', combo_image)
    if cv2.waitKey(1) == ord('q'): #We want to wait 1ms in between frames
        break
cap.release()
cv2.destroyAllWindows()
#plt.imshow(canny)
#plt.show()
