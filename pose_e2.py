import numpy as np
import cv2 as cv
from path import Path
import re
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


def getKeypoint(keypoints):
    return [cv.KeyPoint(keypoints[i][0], keypoints[i][1], 1) for i in range(keypoints.shape[0])]


# def getNpz(root,exts=('.rd')):
#     npz = [f for f in os.listdir(root) if f.endswith(exts)]
#     return npz
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
    for i in range(len(good_matches)):
        points1.append(keypoints1[good_matches[i].queryIdx].pt)
        points2.append(keypoints2[good_matches[i].trainIdx].pt)
    points1 = np.array(points1)
    points2 = np.array(points2)
    E, mask = cv.findEssentialMat(points1, points2, K, method=cv.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, _ = cv.recoverPose(E, points1, points2, K)
    return R, t


def main(root):
    npz = getNpz(root)
    images_path = loadImage(root)

    K = np.array([728.879, 0, 534.9931, 0, 728.9891, 453.4891, 0, 0, 1]).reshape(3, 3)  # 胃镜内参数
    # K = np.array([28.879,0,529.9931,0,28.9891,450.4891,0,0,1]).reshape(3,3) # 胃镜内参数

    # K = np.array([1648.149,0,402.702,0,1514.27,245.472,0,0,1]).reshape(3,3) # 气管镜内参数

    start_position = np.array([142.15, 55, 107.2]).reshape(3, 1)
    one = np.array([0, 0, 0, 1]).reshape(1, 4)
    pose = np.identity(4)
    pose[0, 3] = 142.15
    pose[1, 3] = 74
    pose[2, 3] = 107.2
    pose[1, 1] = -0.5
    pose[1, 2] = 0.866
    pose[2, 1] = -0.866
    pose[2, 2] = -0.5

    camera_poses = []
    imageIndex = []
    camera_poses.append(pose)
    assert (len(npz) == len(images_path))

    total_match = 0
    correct_match = 0
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
        # print(type(descriptors1))
        # print(descriptors1.shape)
        # print(descriptors1)
        # img1 = cv.drawKeypoints(img1, keypoints1,img1)
        # cv.imwrite("stomach_result/model_featurepoints/"+ str(img1_num)+".jpg",img1)
        arraykeypoints2 = npz2['keypoints']
        keypoints2 = getKeypoint(arraykeypoints2)
        descriptors2 = npz2['descriptors']
        # print(descriptors2.shape)
        # print(descriptors2)
        #
        # img2 = cv.drawKeypoints(img2, keypoints2,img2)
        # cv.imwrite("stomach_result/model_featurepoints/"+str(img2_num)+".jpg",img2)
        bf = cv.BFMatcher()
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)
        goodgood_matches = [m for m, n in matches if m.distance < 0.8 * n.distance]
        # --------------------------------------------------------------------------------
        # flann = cv.FlannBasedMatcher()
        # matches = flann.match(descriptors1, descriptors2)
        # max_dist = 0
        # min_dist = 100  # 最小距离和最大距离
        # for j in range(len(descriptors1)):
        #     dist = matches[j].distance
        #     if dist < min_dist: min_dist = dist
        #     if dist > max_dist: max_dist = dist
        # good_matches = []
        # for j in range(len(descriptors1)):
        #     if matches[j].distance < 3 * min_dist:
        #         good_matches.append(matches[j])
        # if len(good_matches) == 0:
        #     # return None
        #     # continue
        #     good_matches = matches
        # obj = []
        # scene = []
        # for j in range(len(good_matches)):
        #     obj.append(np.array(keypoints1[good_matches[j].queryIdx].pt))
        #     scene.append(np.array(keypoints2[good_matches[j].trainIdx].pt))
        # H, listpoints = cv.findHomography(np.array(obj), np.array(scene), cv.RANSAC, ransacReprojThreshold=4)
        # goodgood_matches = []
        # for j in range(len(listpoints)):
        #     a = listpoints[j][:, ][0]
        #     if a == 1:
        #         goodgood_matches.append(good_matches[j])
        # --------------------------------------------------------------------------------
        if (len(goodgood_matches) < 6):
            camera_poses.append(camera_poses[i - 1])
            # global_pose.append(camera_poses)
            imageIndex.append(img2_num)
            print(True)
            continue
            # 计算时间
        # times_calcu_E = []
        # start = time.time()
        R, t = pose_estimation(K, keypoints1, keypoints2, goodgood_matches)
        end = time.time()

        # times_calcu_E.append(end - start)
        # print(np.mean(np.array(times_calcu_E)))
        P = np.append(R, t, axis=1)
        P = np.append(P, one, axis=0)
        pose = np.dot(pose, np.linalg.inv(P))
        imageIndex.append(img2_num)
        camera_poses.append(pose)
    return camera_poses, imageIndex

    #     good = [[m] for m, n in matches if m.distance < 0.8 * n.distance]  # 调用cv.drawMatchesKnn时用
    #     imgMatch = cv.drawMatchesKnn(img1, keypoints1, img2, keypoints2, good, None, flags=2)
    #     # cv.imshow("match",imgMatch)
    #     # cv.waitKey()
    #     filename = "colon_result/test_feature_matches_0.7/" + img1_num + "_" + img2_num + "match" + ".jpg"
    #     cv.imwrite(filename, imgMatch)
    #
    #     good_1 = [m1 for m1, n1 in matches if m1.distance < 0.8 * n1.distance]
    #     fusion_img = cv.addWeighted(img1, 0.5, img2, 0.5, 0.)
    #     line = ""
    #     for j in range(len(good_1)):
    #         print([good_1[j].queryIdx, good_1[j].trainIdx])
    #         print(keypoints1[good_1[j].queryIdx].pt[0])
    #         x1 = int(keypoints1[good_1[j].queryIdx].pt[0])
    #         y1 = int(keypoints1[good_1[j].queryIdx].pt[1])
    #         x2 = int(keypoints2[good_1[j].trainIdx].pt[0])
    #         y2 = int(keypoints2[good_1[j].trainIdx].pt[1])
    #         arrow_img = cv.arrowedLine(fusion_img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
    #         line = line + str(x1) + "," + str(y1) + "," + str(x2) + "," + str(y2) + ","
    #
    #     with open(root + "/part_coord_50_e2.csv", 'a') as f:
    #         f.write(line)
    #         f.write("\n")
    #     f.close()
    # arrowString = "colon_result/test_feature_arrowLine_0.7/" + "arrow" + img1_num + "-" + img2_num + ".jpg"
    # cv.imwrite(arrowString, arrow_img)


def rot2quaternion(R):
    q0 = (np.sqrt(R.trace() + 1)) / 2
    q1 = (R[1, 2] - R[2, 1]) / (4 * q0)
    q2 = (R[2, 0] - R[0, 2]) / (4 * q0)
    q3 = (R[0, 1] - R[1, 0]) / (4 * q0)
    return q0, q1, q2, q3


if __name__ == '__main__':
    root = r"H:\work\ICIRA\dataset\newframes_152"

    # correct_match, total_match = main(root)
    # print(correct_match)
    # print(total_match)
    # print(correct_match / total_match)

    # with open("stomach_result/global_pose/matches_0.8.txt", 'a') as f:
    #     for i in range(len(total_match)):
    #         correct = total_match[i][0]
    #         total = total_match[i][1]
    #         line = str(correct) + " " + str(total) + "\n"
    #         f.write(line)
    # f.close()
    times_comput_pose2 = []
    start = time.time()
    global_pose, imageIndex = main(root)
    end = time.time()
    times_comput_pose2.append(end - start)
    print(np.mean(np.array(times_comput_pose2)))
    # print(len(global_pose))
    # print(global_pose)
    # print(imageIndex)
    # print("done")

    with open(root + "/pose/DMFE+Calculating_Matrix_E", 'a') as f:
        for i in range(len(imageIndex)):
            index = imageIndex[i]
            R = global_pose[i][0:3, 0:3]
            t = global_pose[i][0:3, 3]
            q0, q1, q2, q3 = rot2quaternion(R)
            line = str(index) + " " + str(t[0]) + " " + str(t[1]) + " " + str(t[2]) + " " + str(q0) + " " + str(
                q1) + " " + str(q2) + " " + str(q3) + "\n"
            f.write(line)
    f.close()

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
