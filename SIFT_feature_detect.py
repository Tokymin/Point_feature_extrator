import cv2 as cv
import numpy as np
from path import Path
import re
import os


def getDigital(string):
    assert (isinstance(string, str))
    return re.sub('\D', '', string)


def loadImage(root):
    root = Path(root)
    images_path = root / "test_imgs.txt"
    imgs = [root / folder[:-1] for folder in open(images_path)]
    return imgs


def pose_estimation(K, keypoints1, keypoints2, good_matches):
    points1 = []
    points2 = []
    for i in range(len(good_matches)):
        points1.append(keypoints1[good_matches[i].queryIdx].pt)
        points2.append(keypoints2[good_matches[i].trainIdx].pt)
    points1 = np.array(points1)
    points2 = np.array(points2)
    E, mask = cv.findEssentialMat(points1, points2, K, method=cv.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, _ = cv.recoverPose(E, points1, points2, K)
    return R, t


def siftDetect(root):
    images_path = loadImage(root)

    K = np.array([728.879, 0, 334.9931, 0, 728.9891, 280.4891, 0, 0, 1]).reshape(3, 3)
    # start_position = np.array([142.15, 58.5, 107.2]).reshape(3, 1)
    one = np.array([0, 0, 0, 1]).reshape(1, 4)
    pose = np.identity(4)
    pose[0, 3] = 141.14;
    pose[1, 3] = 72.4;
    pose[2, 3] = 85.6
    pose[0, 0] = 0.9491
    pose[0, 1] = -0.1753
    pose[0, 2] = -0.2616
    pose[1, 0] = 0.2366
    pose[1, 1] = -0.1513
    pose[1, 2] = 0.9597
    pose[2, 0] = -0.2078
    pose[2, 1] = -0.9728
    pose[2, 2] = -0.1022

    camera_poses = []
    imageIndex = []
    camera_poses.append(pose)

    flag = False

    num = 0
    for i in range(len(images_path) - 1):

        img1 = cv.imread(images_path[i])

        # img1_num = getDigital(images_path[i])
        img2 = cv.imread(images_path[i + 1])
        img2_num = getDigital(images_path[i + 1])

        sift = cv.xfeatures2d_SIFT.create()
        kpts1, des1 = sift.detectAndCompute(img1, None)
        kpts2, des2 = sift.detectAndCompute(img2, None)

        # img1 = cv.drawKeypoints(img1, kpts1, img1)
        # cv.imwrite("stomach_result/ORB_featurepoints/" + str(img1_num) + ".jpg", img1)
        #
        # img2 = cv.drawKeypoints(img2, kpts2, img2)
        # cv.imwrite("stomach_result/ORB_featurepoints/" + str(img2_num) + ".jpg", img2)

        bf = cv.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.8 * n.distance]

        if (len(good_matches) < 6):
            # flag = True
            # num += 1
            # img1_num = getDigital(images_path[i + num - 1])
            # root = Path(root)
            # cur = root/str(img1_num) + ".jpg"
            # print(cur)
            continue

        R, t = pose_estimation(K, kpts1, kpts2, good_matches)
        P = np.append(R, t, axis=1)
        P = np.append(P, one, axis=0)
        pose = np.dot(pose, np.linalg.inv(P))
        imageIndex.append(img2_num)
        camera_poses.append(pose)
        # print(img2_num)
    return camera_poses, imageIndex

    # good = [[m] for m,n in matches if m.distance < 0.7*n.distance] # 调用cv.drawMatchesKnn时用
    # imgMatch = cv.drawMatchesKnn(img1,kpts1,img2,kpts2,good,None,flags=2)
    # filename = "stomach_result/sift_feature_matches_0.7/"+img1_num + "_" + img2_num + "match" + ".jpg"
    # cv.imwrite(filename,imgMatch)


def rot2quaternion(R):
    q0 = (np.sqrt(R.trace() + 1)) / 2
    q1 = (R[1, 2] - R[2, 1]) / (4 * q0)
    q2 = (R[2, 0] - R[0, 2]) / (4 * q0)
    q3 = (R[0, 1] - R[1, 0]) / (4 * q0)
    return q0, q1, q2, q3


if __name__ == '__main__':
    root = r"E:\xyw\feature_detector\data"

    global_pose, imageIndex = siftDetect(root)

    with open("stomach_result/global_pose/sift_pose10.txt", 'a') as f:
        for i in range(len(imageIndex)):
            index = imageIndex[i]
            R = global_pose[i][0:3, 0:3]
            t = global_pose[i][0:3, 3]
            q0, q1, q2, q3 = rot2quaternion(R)
            print(index)
            index = int(index)
            index = index - 1
            line = str(index) + " " + str(q0) + " " + str(q1) + " " + str(q2) + " " + str(q3) + " " + str(
                t[0]) + " " + str(t[1]) + " " + str(t[2]) + "\n"
            f.write(line)
    f.close()
