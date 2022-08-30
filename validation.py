import numpy as np
import cv2 as cv
from path import Path
import re
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def getKeypoint(keypoints):
    return [cv.KeyPoint(keypoints[i][0],keypoints[i][1],1) for i in range(keypoints.shape[0])]

# def getNpz(root,exts=('.rd')):
#     npz = [f for f in os.listdir(root) if f.endswith(exts)]
#     return npz
def getNpz(root):
    root = Path(root)
    npz_paths = root/"npz.txt"
    npzs = [root/folder[:-1] for folder in open(npz_paths)]
    return npzs

def getDigital(string):
    assert(isinstance(string,str))
    return re.sub('\D','',string)

def loadImage(root):
    root = Path(root)
    images_path = root/"imgs.txt"
    imgs = [root/folder[:-1] for folder in open(images_path)]
    return imgs

def pose_estimation(K,keypoints1,keypoints2,good_matches):
    points1=[]
    points2=[]
    for i in range(len(good_matches)):
        points1.append(keypoints1[good_matches[i].queryIdx].pt)
        points2.append(keypoints2[good_matches[i].trainIdx].pt)
    points1 = np.array(points1)
    points2 = np.array(points2)
    E,mask = cv.findEssentialMat(points1,points2,K,method=cv.RANSAC,prob=0.999,threshold=1.0)
    _,R,t,_ = cv.recoverPose(E,points1,points2,K)
    return R,t,points1,points2

def pix2cam(point,K):
    x = (point[0] - K[0,2]) / K[0,0]
    y = (point[1] - K[1,2]) / K[1,1]
    return np.array([x,y]).reshape(1,2)

def compute_accuracy(R,t,points1,points2,K):
    assert(points1.shape[0] == points2.shape[0])
    rows = points1.shape[0]
    accu = 0
    one = np.ones(1)
    for i in range(rows):
        pt1 = pix2cam(points1[i,:],K)
        cp1 = np.c_[pt1,one]
        # print(cp1.shape)
        pt2 = pix2cam(points2[i,:],K)
        cp2 = np.c_[pt2,one]

        cp1_to_cp2 = np.dot(R,cp1.T) + t
        P = cp1_to_cp2.T
        diff = P - cp2
        accu = sum(sum(diff))
    return accu / rows

def main(root):
    npz = getNpz(root)
    images_path = loadImage(root)

    K = np.array([728.879,0,534.9931,0,728.9891,453.4891,0,0,1]).reshape(3,3) # 胃镜内参数

    start_position = np.array([142.15,55,107.2]).reshape(3,1)
    one = np.array([0,0,0,1]).reshape(1,4)
    pose = np.identity(4)
    pose[0,3] = 142.15;pose[1,3] = 74;pose[2,3] = 107.2
    pose[1,1] = -0.5; pose[1,2] = 0.866
    pose[2,1] = -0.866; pose[2,2] = -0.5

    camera_poses = []
    imageIndex = []
    camera_poses.append(pose)
    assert (len(npz) == len(images_path))


    total_match = 0
    correct_match = 0
    total_accu = 0
    for i in range(len(npz)-1):
        img1 = cv.imread(images_path[i])
        img1_num = getDigital(images_path[i])
        img2 = cv.imread(images_path[i+1])
        img2_num = getDigital(images_path[i+1])

        npz1 = np.load(npz[i])
        npz1_num = getDigital(npz[i])
        npz2 = np.load(npz[i+1])
        npz2_num = getDigital(npz[i+1])

        assert (img1_num == npz1_num)
        assert (npz2_num == img2_num)
        arraykeypoints1 = npz1['keypoints']
        keypoints1 = getKeypoint(arraykeypoints1)
        descriptors1 = npz1['descriptors']

        arraykeypoints2 = npz2['keypoints']
        keypoints2 = getKeypoint(arraykeypoints2)
        descriptors2 = npz2['descriptors']

        bf = cv.BFMatcher()
        matches = bf.knnMatch(descriptors1,descriptors2,k=2)
        good_matches = [m for m,n in matches if m.distance < 0.8*n.distance]

        if(len(good_matches) < 6):
            continue

        R,t,points1,points2= pose_estimation(K,keypoints1,keypoints2,good_matches)
        accu = compute_accuracy(R,t,points1,points2,K)
        total_accu += accu
    return total_accu / len(npz)

# def pix2cam(point,K):
#     x = (point[0] - K[0,2]) / K[0,0]
#     y = (point[1] - K[1,2]) / K[1,1]
#     return np.array([x,y]).reshape(1,2)
#
# def compute_accuracy(R,t,points1,points2,K):
#     assert(points1.shape[0] == points2.shape[0])
#     rows = points1.shape[0]
#     accu = 0
#     one = np.ones(1)
#     for i in range(rows):
#         pt1 = pix2cam(points1[i,:],K)
#         cp1 = np.c_[pt1,one]
#         print(cp1.shape)
#         pt2 = pix2cam(points2[i,:],K)
#         cp2 = np.c_[pt2,one]
#
#         cp1_to_cp2 = np.dot(R,cp1.T) + t
#         P = cp1_to_cp2.T
#         diff = P - cp2
#         accu = sum(sum(diff))
#     return accu / rows


def rot2quaternion(R):
    q0 = (np.sqrt(R.trace()+1)) / 2
    q1 = (R[1,2] - R[2,1]) / (4*q0)
    q2 = (R[2,0] - R[0,2]) / (4*q0)
    q3 = (R[0,1] - R[1,0]) / (4*q0)
    return q0,q1,q2,q3

if __name__ == '__main__':
    root = "data"
    K = np.array([728.879, 0, 534.9931, 0, 728.9891, 453.4891, 0, 0, 1]).reshape(3, 3)  # 胃镜内参数
    accu = main(root)

    print(accu)
    print("done")