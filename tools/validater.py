import os

from tqdm import tqdm
import cv2 as cv
import torch
import torch.nn as nn
import numpy as np

from pose_estimation import calgoodgoodmatch

model_prefix = ""
writer = {}


def getKeypoint(keypoints):
    return [cv.KeyPoint(keypoints[i][0], keypoints[i][1], 1) for i in range(keypoints.shape[0])]


def savepose(global_pose_, root):
    with open(root + "/Key_Points_Extractor_" + str(epoch) + ".txt", 'a') as f:
        t = global_pose_[-1]
        line = str(t[0]) + " " + str(
            t[1]) + " " + str(t[2]) + " " + str(t[3]) + " " + str(t[4]) + " " + str(t[5]) + " " + str(t[6]) + " " \
               + str(t[7]) + " " + str(t[8]) + " " + str(t[9]) + " " + str(t[10]) + " " + str(t[11]) + "\n"

        f.write(line)
    f.close()


epoch = 0


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

    if E is None or E.shape != (3, 3):  # print(E.shape)
        print("E.cols != 3 or E.rows != 3")
        return None, None
    _, R, t, _ = cv.recoverPose(E, points1, points2, K, mask=mask)
    return R, t


def calgoodgoodmatch(des1, des2, keypoints1, keypoints2):
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


class Validater(nn.Module):
    """ Helper class to Validate a deep network.
        Overload this class `forward_backward` for your actual needs.
    """

    def __init__(self, net, loader):
        nn.Module.__init__(self)
        self.net = net
        self.loader = loader

    def iscuda(self):
        return next(self.net.parameters()).device != torch.device('cpu')

    def todevice(self, x):
        if isinstance(x, dict):
            return {k: self.todevice(v) for k, v in x.items()}
        if isinstance(x, (tuple, list)):
            return [self.todevice(v) for v in x]

        if self.iscuda():
            return x.contiguous().cuda(non_blocking=True)
        else:
            return x.cpu()

    def drawFlow(self, goodgood_matches, img1, img2, keypoints1, keypoints2, writer, iter):
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
            arrow_img = cv.arrowedLine(fusion_img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=1)
            line = line + str(x1) + "," + str(y1) + "," + str(x2) + "," + str(y2) + ","
        # cv.imwrite(arrowString, arrow_img)
        # img1_temp = cv.drawKeypoints(img1, keypoints1, img1)  # 这里可以用tensorBroad保存一下
        writer.add_image("arrow_img", arrow_img.transpose(2, 0, 1), iter)
        # writer.add_image("arrow_img", arrow_img.transpose(2, 1, 0), iter)

    def tensor2numpy(self, img):
        img1 = img.cpu().numpy()
        maxValue = img1.max()
        img1 = img1 * 255 / maxValue
        img2 = img1.squeeze(0)
        img3 = img2.transpose(1, 2, 0)
        img3 = cv.cvtColor(img3, cv.COLOR_RGB2BGR)
        b, g, r = cv.split(img3)
        img1_temp = cv.merge([r, g, b]).copy()
        return img1_temp

    def __call__(self):
        self.net.eval()
        K = np.array([156.3536121, 0, 160, 0, 157.549850, 160, 0, 0, 1]).reshape(3, 3)  # 胃镜内参数
        one = np.array([0, 0, 0, 1]).reshape(1, 4)
        pose = np.identity(4)
        pose[0, 3] = 0
        pose[1, 3] = 0
        pose[2, 3] = 0
        pose[0, 0] = 1
        pose[1, 1] = 1
        pose[2, 2] = 1
        camera_poses = []
        camera_poses.append(pose[0:3, :].flatten())
        camera_poses.append(pose[0:3, :].flatten())
        imageIndex = []

        root = r"F:\Toky\PythonProject\Point_feature_extrator\validate\pose"
        for iter, inputs in enumerate(tqdm(self.loader)):
            input_1 = inputs['img1']
            input_2 = inputs['img2']
            input_1 = self.todevice(input_1)
            input_2 = self.todevice(input_2)
            # compute gradient and do model update
            output_1 = self.forward_backward(
                input_1)  # 这里执行实现的forward_backward函数, 验证时返回得到的东西：aflow mask repeat reliable descriptors imgs[0,1]
            output_2 = self.forward_backward(input_2)
            arraykeypoints1 = output_1['keypoints']
            keypoints1 = getKeypoint(arraykeypoints1)
            descriptors1 = output_1['descriptors']
            # img0=cv.imread(r"F:\Toky\Dataset\Endo_colon_unity\test_dataset\photo\image_0060.png")
            img1 = self.tensor2numpy(input_1).astype(np.uint8).copy()
            img2 = self.tensor2numpy(input_2).astype(np.uint8).copy()
            img1_temp = cv.drawKeypoints(img1, keypoints1, img1)  # 这里可以用tensorBroad保存一下
            writer.add_image("img1_feature_point", img1_temp.transpose(2, 0, 1), iter)

            arraykeypoints2 = output_2['keypoints']
            keypoints2 = getKeypoint(arraykeypoints2)
            descriptors2 = output_2['descriptors']
            if len(descriptors1) == 0 or len(descriptors2) == 0:
                camera_poses.append(camera_poses[iter - 1])
                savepose(camera_poses, root)
                imageIndex.append(iter)
                print("descriptors1 or des2 is None")
                continue
            else:
                goodgoodmatch_ = calgoodgoodmatch(descriptors1, descriptors2, keypoints1, keypoints2)
                if goodgoodmatch_ is None or len(goodgoodmatch_) == 0:
                    camera_poses.append(camera_poses[iter - 1])
                    savepose(camera_poses, root)
                    imageIndex.append(iter)
                    print("goodgoodmatch_ is None")
                    continue
            R, t = pose_estimation(K, keypoints1, keypoints2, goodgoodmatch_)
            if R is None:
                camera_poses.append(camera_poses[iter - 1])
                savepose(camera_poses, root)
                imageIndex.append(iter)
                print("R is None")
                continue
            P = np.append(R, t, axis=1)
            P = np.append(P, one, axis=0)
            pose = np.dot(pose, np.linalg.inv(P))
            camera_poses.append(pose[0:3, :].flatten())
            self.drawFlow(goodgoodmatch_, img1, img2, keypoints1, keypoints2, writer, iter)  # 画特征点流
            savepose(camera_poses, root)
        # 计算指标
        # # 调用evo计算APE（默认情况下是ATE）、RPE等
        from evo import entry_points
        from evo import main_rpe
        import argcomplete
        gt_path = r"F:\Toky\PythonProject\Point_feature_extrator\validate\pose" + "/position_rotation.kitti"
        main_rpe.est_file = root + "/Key_Points_Extractor_" + str(epoch) + ".txt"
        main_rpe.ref_file = gt_path
        main_rpe.model_prefix = model_prefix
        main_rpe.rpe_metric_csv_path = r"F:\Toky\PythonProject\Point_feature_extrator\validate/metric_csv/RPE_metric" + model_prefix + ".csv"  # 保存csv的文件
        main_rpe.ape_metric_csv_path = r"F:\Toky\PythonProject\Point_feature_extrator\validate/metric_csv/APE_metric" + model_prefix + ".csv"
        main_rpe.metric_index = str(epoch)

        parser = main_rpe.parser()
        main_rpe.writer = writer
        argcomplete.autocomplete(parser)
        entry_points.launch(main_rpe, parser)

    def forward_backward(self, inputs):
        raise NotImplementedError()
