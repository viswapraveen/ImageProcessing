# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 19:35:17 2021

@author: viswa
"""
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import _pickle as cPickle

img1 = cv.imread('input\image1.jpg',cv.IMREAD_GRAYSCALE)
plt.imshow(img1),plt.show()
plt.imsave('output\image1_op.jpg', img1)
img2 = cv.imread('input\image3.jpg',cv.IMREAD_GRAYSCALE)
plt.imshow(img2),plt.show()
plt.imsave('output\image3_op.jpg', img2)

# Initiate SIFT detector
sift = cv.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
img1=cv.drawKeypoints(img1, kp1, img1)
cv.imwrite('output\sift_keypoints_image1.jpg',img1)

# Dump the keypoints
f = open("output\keypoints.txt", "w")

for point1 in kp1:
    temp = (str(point1.pt) + "\t" + "," + str(point1.size) + "," + str(point1.angle)  + "," + str(point1.response)  + "," + str(point1.octave) + "," + 
        str(point1.class_id) + "\n" )
    f.write(temp)

f.close()

# Dump x,y coordinates
f = open("output\coordinates.txt", "w")

for point1 in kp1:
    c = (str(point1.pt) +"\n" )
    f.write(c)

f.close()

kp2, des2 = sift.detectAndCompute(img2,None)
img2=cv.drawKeypoints(img2, kp1, img2)
cv.imwrite('output\sift_keypoints_image3.jpg',img2)

# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)


# Dump x,y coordinates
f = open("output\DMatch.txt", "w")

for match in matches:
  m = (str(kp1[match[0].queryIdx].pt) + "," + str(kp2[match[0].trainIdx].pt))
  f.write(m)


f.close()

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.80*n.distance:
        good.append([m])
# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(img3),plt.show()
plt.imsave('output\image1_3_match_op.jpg', img3)