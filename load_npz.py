import os
import numpy as np
# import re
import cv2 as cv

def getR2D2(root,exts=('.rd')):
    KPF = [f for f in os.listdir(root) if f.endswith(exts)]
    return KPF

if __name__ == '__main__':
    root = "data_1/000278.jpg.rd"
    KPF = getR2D2(root,exts=('.rd'))
    print(len(KPF))
    print(KPF)
    print(KPF[0])
    print(type(KPF[0]))
    print(KPF[0][0:4])
    keypoint = np.load("imgs/"+KPF[0])
    keypointCoords = keypoint['keypoints']
    print(keypointCoords)

# npz = np.load("data_1/000280.jpg.rd")
# keypoints = npz['keypoints']
# print(keypoints)
# print(keypoints.shape)

# a = '5246.jpg'
# assert(isinstance(a,str))
# b = re.sub('\D','',a)
# print(b)
# print(type(b))
# #
# # c = '5246.jpg.rd'
# # d = re.sub('\D','',c)
# # print(d)
# # print(b==d)
# # assert (b==d)
#
#
# img = cv.imread("data_one/000036.jpg")
#
# surf = cv.xfeatures2d.SURF_create(100)
#
# keypoints,des = surf.detectAndCompute(img,None)
# img = cv.drawKeypoints(img,keypoints,outImage=img)
# cv.imwrite("surf"+"T100"+str(36) + ".jpg", img)

# img1 = cv.imread(("imgs/5247.jpg"))
# keypoints1,des1 = surf.detectAndCompute(img1,None)
#
# bf = cv.BFMatcher()
# matches = bf.knnMatch(des, des1,k=2)
# good = [[m] for m,n in matches if m.distance < 0.5 * n.distance]
# print(matches[0][0].queryIdx)
# # matches = sorted(matches,key=lambda x:x.distance)
#
# imgMatch = cv.drawMatchesKnn(img, keypoints, img1, keypoints1, good,None,
#                           flags=2)
# cv.imshow("match",imgMatch)
# # # print(des)
# # print(type(des))
# # print(type(keypoints))
# #
# #
# # # for i in range(len(keypoints)):
# # #     print(keypoints[i].pt)
# # cv.imshow("outimg",img)
# #
# cv.waitKey()
#
# string = str(5246)+"-"+str(5247)+'.jpg'
# print(string)

# points1=[]
# points2=[]
# for j in range(len(good_matches)):
#     print(good_matches[j])
#     points1.append(keypoints1[good_matches[j].queryIdx].pt)
#     points2.append(keypoints2[good_matches[j].trainIdx].pt)
# print(points1)
# print(points2)
# points1 = np.array(points1)
# points2 = np.array(points2)
# print(points1)
# print(points2)
# print(type(points1))
