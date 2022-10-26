import time

import numpy as np
import cv2 as cv
from path import Path
import re
from matplotlib import pyplot as plt


def getKeypoint(keypoints):
    return [cv.KeyPoint(keypoints[i][0], keypoints[i][1], 1) for i in range(keypoints.shape[0])]


def getNpz(root):
    root = Path(root)
    npz_paths = root / "test_npz.txt"
    npzs = [root / folder[:-1] for folder in open(npz_paths)]
    return npzs


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
    for n in range(len(good_matches)):
        points1.append(keypoints1[good_matches[n].queryIdx].pt)
        points2.append(keypoints2[good_matches[n].trainIdx].pt)
    points1 = np.array(points1)
    points2 = np.array(points2)

    # 绘制点

    E, mask = cv.findEssentialMat(points1, points2, K, method=cv.RANSAC, prob=0.999,
                                  threshold=1.0)

    if E.shape != (3, 3):  # print(E.shape)
        print("E.cols != 3 or E.rows != 3")
        return None, None
    _, R, t, _ = cv.recoverPose(E, points1, points2, K, mask=mask)
    return R, t


def calpose(good_match, pose):
    index = []
    if len(good_match) < 6:
        return 0, 0
    R, t = pose_estimation(K, keypoints1, keypoints2, good_match)
    P = np.append(R, t, axis=1)
    P = np.append(P, one, axis=0)
    pose = np.dot(pose.astype(float), np.linalg.inv(P))
    index.append(img2_num)
    camera_poses.append(pose)
    return camera_poses, index


def savepose(root, imageIndex_, global_pose_):
    with open(root + "/OF_based_Key_Points_Extractor.txt", 'a') as f:
        # index = imageIndex_[-1]
        # R = global_pose_[-1][0:3, 0:3]
        # t = global_pose_[-1][0:3, 3]
        # q0, q1, q2, q3 = rot2quaternion(R)
        # line = str(t[0]) + " " + str(
        #     t[1]) + " " + str(t[2]) + " " + str(q0) + " " + str(q1) + " " + str(q2) + " " + str(q3) + "\n"
        t = global_pose_[-1]
        line = str(t[0]) + " " + str(
            t[1]) + " " + str(t[2]) + " " + str(t[3]) + " " + str(t[4]) + " " + str(t[5]) + " " + str(t[6]) + " " \
               + str(t[7]) + " " + str(t[8]) + " " + str(t[9]) + " " + str(t[10]) + " " + str(t[11]) + "\n"

        f.write(line)
    f.close()


def drawFlow(goodgood_matches):
    if len(goodgood_matches) < 6:
        return None
    fusion_img = cv.addWeighted(img1, 0.5, img2, 0.5, 0.)
    line = ""
    for j in range(len(goodgood_matches)):
        # print([goodgood_matches[j].queryIdx, goodgood_matches[j].trainIdx])
        # print(keypoints1[goodgood_matches[j].queryIdx].pt[0])
        x1 = int(keypoints1[goodgood_matches[j].queryIdx].pt[0])
        y1 = int(keypoints1[goodgood_matches[j].queryIdx].pt[1])
        x2 = int(keypoints2[goodgood_matches[j].trainIdx].pt[0])
        y2 = int(keypoints2[goodgood_matches[j].trainIdx].pt[1])
        arrow_img = cv.arrowedLine(fusion_img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

        line = line + str(x1) + "," + str(y1) + "," + str(x2) + "," + str(y2) + ","

    # with open(root + "/part_coord_i1.csv", 'a') as f:
    #     f.write(line)
    #     f.write("\n")
    # f.close()

    arrowString = root + "/test_feature_arrowLine/" + "arrow" + img1_num + "-" + img2_num + ".jpg"
    cv.imwrite(arrowString, arrow_img)


def calgoodgoodmatch(des1, des2):
    flann = cv.FlannBasedMatcher()
    matches = flann.match(des1, des2)
    max_dist = 0
    min_dist = 100  # 最小距离和最大距离
    for j in range(len(des1)):
        dist = matches[j].distance
        if dist < min_dist: min_dist = dist
        if dist > max_dist: max_dist = dist
    good_matches = []
    for j in range(len(des1)):
        if matches[j].distance < 3 * min_dist:
            good_matches.append(matches[j])
    if len(good_matches) == 0 or len(good_matches) <= 4:
        return None
    obj = []
    scene = []

    for j in range(len(good_matches)):
        obj.append(np.array(keypoints1[good_matches[j].queryIdx].pt))
        scene.append(np.array(keypoints2[good_matches[j].trainIdx].pt))

    H, listpoints = cv.findHomography(np.array(obj), np.array(scene), cv.RANSAC, ransacReprojThreshold=4)
    goodgood_matches = []
    for j in range(len(listpoints)):
        a = listpoints[j][:, ][0]
        if a == 1:
            goodgood_matches.append(good_matches[j])
    return goodgood_matches


def savematch(root, total_match):
    with open(root + "/global_pose/matches_0.8.txt", 'a') as f:
        for i in range(len(total_match)):
            correct = total_match[i][0]
            total = total_match[i][1]
            line = str(correct) + " " + str(total) + "\n"
            f.write(line)
    f.close()


def rot2quaternion(R):
    q0 = (np.sqrt(R.trace() + 1)) / 2
    q1 = (R[1, 2] - R[2, 1]) / (4 * q0)
    q2 = (R[2, 0] - R[0, 2]) / (4 * q0)
    q3 = (R[0, 1] - R[1, 0]) / (4 * q0)
    return q0, q1, q2, q3


if __name__ == '__main__':
    root = r"F:\Toky\Dataset\Endo_colon_unity\test_for_point_feature\photo4"
    npz = getNpz(root)
    images_path = loadImage(root)
    # K = np.array([156.0418, 0, 178.5604, 0, 155.7529, 181.8043, 0, 0, 1]).reshape(3, 3)  # 肠镜内参数
    # K = np.array([156.0418, 0, 160, 0, 155.7529, 160, 0, 0, 1]).reshape(3, 3)
    K = np.array([156.3536121, 0, 160, 0, 157.549850, 160, 0, 0, 1]).reshape(3, 3)  # 胃镜内参数

    start_position = np.array([0, 0, 0]).reshape(3, 1)
    one = np.array([0, 0, 0, 1]).reshape(1, 4)
    pose = np.identity(4)
    pose[0, 3] = 0
    pose[1, 3] = 0
    pose[2, 3] = 0
    pose[0, 0] = 1
    pose[1, 1] = 1
    pose[2, 2] = 1
    camera_poses = []
    save_kitti_pose = pose[0:3, :].flatten()
    camera_poses.append(pose[0:3, :].flatten())
    camera_poses.append(pose[0:3, :].flatten())
    assert (len(npz) == len(images_path))
    total_match = 0
    correct_match = 0
    global_pose = []
    imageIndex = []
    goodgoodmatch = []
    times_model_output = []
    start = time.time()

    for i in range(len(npz) - 1):
        img1 = cv.imread(images_path[i])
        img1_num = getDigital(images_path[i])
        img2 = cv.imread(images_path[i + 1])
        img2_num = getDigital(images_path[i + 1])
        npz1 = np.load(npz[i])
        npz1_num = getDigital(npz[i])
        npz2 = np.load(npz[i + 1])
        npz2_num = getDigital(npz[i + 1])

        assert (img1_num == npz1_num)
        assert (npz2_num == img2_num)

        arraykeypoints1 = npz1['keypoints']
        keypoints1 = getKeypoint(arraykeypoints1)
        descriptors1 = npz1['descriptors']
        img1 = cv.drawKeypoints(img1, keypoints1, img1)
        arraykeypoints2 = npz2['keypoints']
        keypoints2 = getKeypoint(arraykeypoints2)
        descriptors2 = npz2['descriptors']
        # img2 = cv.drawKeypoints(img2, keypoints2, img2)
        goodgoodmatch_ = calgoodgoodmatch(descriptors1, descriptors2)

        # outimage = cv.drawMatches(img1, keypoints1, img2, keypoints2, goodgoodmatch_, outImg=None)
        # cv.imshow("", outimage[:, :, ::-1])
        # cv.waitKey(0)
        if goodgoodmatch_ is None or len(goodgoodmatch_) == 0:
            camera_poses.append(camera_poses[i - 1])
            savepose(root, imageIndex, camera_poses)
            imageIndex.append(img2_num)
            print("goodgoodmatch_ is None")
            continue
        R, t = pose_estimation(K, keypoints1, keypoints2, goodgoodmatch_)

        if R is None:
            camera_poses.append(camera_poses[i - 1])
            savepose(root, imageIndex, camera_poses)
            imageIndex.append(img2_num)
            print("R is None")
            continue
        P = np.append(R, t, axis=1)
        P = np.append(P, one, axis=0)
        pose = np.dot(pose, np.linalg.inv(P))  # np.linalg.inv(
        camera_poses.append(pose[0:3, :].flatten())
        # global_pose.append(camera_poses)
        # drawFlow(goodgoodmatch_)  # 画surf流
        savepose(root, imageIndex, camera_poses)

    # pos=[]
    # for i in range(len(global_pose)):
    #     pos.append(global_pose[i][0:3,3])
    # pos = np.array(pos)
    # print(pos.shape)
    # print(pos[:,:])
    #
    # x1 = pos[0:1175,0]
    # y1 = pos[0:1175,1]
    # z1 = pos[0:1175,2]
    #
    # x10=pos[0,0]
    # y10=pos[0,1]
    # z10=pos[0,2]
    #
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(x,y,z,c='b',label='magnetic')
    # ax.scatter(x0,y0,z0,c='g',s=100)
    #
    # ax.scatter(x1,y1,z1,c='r',label='deep_learning')
    # ax.scatter(x10,y10,z10,c='g',s=100)
    #
    # ax.legend(loc='best')
    # ax.set_zlabel('Z',fontdict={'size':15,'color':'red'})
    # ax.set_ylabel('Y',fontdict={'size':15,'color':'red'})
    # ax.set_xlabel('X',fontdict={'size':15,'color':'red'})
    #
    # plt.savefig("position_1175.jpg",bbox_inches='tight')
    # plt.show()

    # print(type(global_pose[0]))
    # print(global_pose[1])
    # print(global_pose[2])
    # print(global_pose[17])
    # print(global_pose[1150])
