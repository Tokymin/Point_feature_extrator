import os
import time

from PIL import Image
import numpy as np
import torch
import argparse

from tensorboardX import SummaryWriter

from tools import common
from tools.dataloader import norm_RGB
from nets.patchnet import *
import cv2 as cv
from path import Path
import re


def load_network(model_fn):
    checkpoint = torch.load(model_fn)
    print("\n>> Creating net = " + checkpoint['net'])
    net = eval(checkpoint['net'])
    nb_of_weights = common.model_size(net)
    print(f" ( Model size: {nb_of_weights/1000:.0f}K parameters )")

    # initialization
    weights = checkpoint['state_dict']
    net.load_state_dict({k.replace('module.', ''): v for k, v in weights.items()})
    return net.eval()


class NonMaxSuppression(torch.nn.Module):
    def __init__(self, rel_thr=0.7, rep_thr=0.7):
        nn.Module.__init__(self)
        self.max_filter = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.rel_thr = rel_thr
        self.rep_thr = rep_thr

    def forward(self, reliability, repeatability, **kw):
        assert len(reliability) == len(repeatability) == 1
        reliability, repeatability = reliability[0], repeatability[0]
        # local maxima
        maxima = (repeatability == self.max_filter(repeatability))
        # remove low peaks
        maxima *= (repeatability >= self.rep_thr)
        maxima *= (reliability >= self.rel_thr)
        return maxima.nonzero().t()[2:4]


def extract_multiscale(net, img, detector, scale_f=2 ** 0.25,
                       min_scale=0.0, max_scale=1,
                       min_size=256, max_size=1024,
                       verbose=False):
    old_bm = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False  # speedup
    # extract keypoints at multiple scales
    B, three, H, W = img.shape
    assert B == 1 and three == 3, "should be a batch with a single RGB image"
    assert max_scale <= 1
    s = 1.0  # current scale factor
    X, Y, S, C, Q, D = [], [], [], [], [], []
    while s + 0.001 >= max(min_scale, min_size / max(H, W)):
        if s - 0.001 <= min(max_scale, max_size / max(H, W)):
            nh, nw = img.shape[2:]
            if verbose: print(f"extracting at scale x{s:.02f} = {nw:4d}x{nh:3d}")
            # extract descriptors
            with torch.no_grad():
                res = net(imgs=[img])
            # get output and reliability map
            descriptors = res['descriptors'][0]
            reliability = res['reliability'][0]
            repeatability = res['repeatability'][0]
            # normalize the reliability for nms
            # extract maxima and descs
            y, x = detector(**res)  # nms
            print(y)
            c = reliability[0, 0, y, x]
            q = repeatability[0, 0, y, x]
            d = descriptors[0, :, y, x].t()
            n = d.shape[0]

            # accumulate multiple scales
            X.append(x.float() * W / nw)
            Y.append(y.float() * H / nh)
            S.append((32 / s) * torch.ones(n, dtype=torch.float32, device=d.device))
            C.append(c)
            Q.append(q)
            D.append(d)
        s /= scale_f
        # down-scale the image for next iteration
        nh, nw = round(H * s), round(W * s)
        img = F.interpolate(img, (nh, nw), mode='bilinear', align_corners=False)

    # restore value
    torch.backends.cudnn.benchmark = old_bm

    Y = torch.cat(Y)
    X = torch.cat(X)
    S = torch.cat(S)  # scale
    scores = torch.cat(C) * torch.cat(Q)  # scores = reliability * repeatability
    XYS = torch.stack([X, Y, S], dim=-1)
    D = torch.cat(D)
    return XYS, D, scores


def extract_keypoints(img_path):
    img = Image.open(img_path).convert('RGB')
    print(img.size)
    W, H = img.size
    img = norm_RGB(img)[None]
    if iscuda: img = img.cuda()
    # extract keypoints/descriptors for a single image
    xys, desc, scores = extract_multiscale(net, img, detector,
                                           scale_f=args.scale_f,
                                           min_scale=args.min_scale,
                                           max_scale=args.max_scale,
                                           min_size=args.min_size,
                                           max_size=args.max_size,
                                           verbose=True)

    xys = xys.cpu().numpy()
    desc = desc.cpu().numpy()
    scores = scores.cpu().numpy()
    idxs = scores.argsort()[-args.top_k or None:]

    return dict(imsize=(W, H),
                keypoints=xys[idxs],
                descriptors=desc[idxs],
                scores=scores[idxs])


def getKeypoint(keypoints):
    return [cv.KeyPoint(keypoints[i][0], keypoints[i][1], 1) for i in range(keypoints.shape[0])]


def get_img_path_digital(string):
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


def savepose(root, imageIndex_, global_pose_):
    if os.path.exists(root) == False:
        os.makedirs(root)
    with open(root + "/est_pose.txt", 'a') as f:
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
        x1 = int(keypoints1[goodgood_matches[j].queryIdx].pt[0])
        y1 = int(keypoints1[goodgood_matches[j].queryIdx].pt[1])
        x2 = int(keypoints2[goodgood_matches[j].trainIdx].pt[0])
        y2 = int(keypoints2[goodgood_matches[j].trainIdx].pt[1])
        arrow_img = cv.arrowedLine(fusion_img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        line = line + str(x1) + "," + str(y1) + "," + str(x2) + "," + str(y2) + ","
    arrowString = save_root + "/feature_arrowLine/" + "arrow" + img1_num + "-" + img2_num + ".jpg"
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


def init_pose():
    pose = np.identity(4)
    pose[0, 3] = 0
    pose[1, 3] = 0
    pose[2, 3] = 0
    pose[0, 0] = 1
    pose[1, 1] = 1
    pose[2, 2] = 1
    return pose


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Extract keypoints for a given image")
    parser.add_argument('kitti',help='for evo')
    parser.add_argument("--model", type=str, required=False,
                        default=".\models\history_model.pt",
                        help='model path')
    parser.add_argument("--dataset-folder", type=str,
                        default=r"F:\Toky\Dataset\Endo_colon_unity\test_for_point_feature\photo4", required=False,
                        nargs='+', help='images / list')
    parser.add_argument("--tag", type=str, default='rd', help='output file tag')
    parser.add_argument("--top-k", type=int, default=500, help='number of keypoints')
    parser.add_argument("--scale-f", type=float, default=2 ** 0.25)
    parser.add_argument("--min-size", type=int, default=0)
    parser.add_argument("--max-size", type=int, default=9999)
    parser.add_argument("--min-scale", type=float, default=0.3)
    parser.add_argument("--max-scale", type=float, default=1)

    parser.add_argument("--reliability-thr", type=float, default=0.7)
    parser.add_argument("--repeatability-thr", type=float, default=0.7)

    parser.add_argument("--gpu", type=int, nargs='+', default=[0], help='use -1 for CPU')
    args = parser.parse_args()

    # load the network...
    net = load_network(args.model)
    iscuda = common.torch_set_gpu(args.gpu)
    if iscuda: net = net.cuda()
    # create the non-maxima detector
    detector = NonMaxSuppression(
        rel_thr=args.reliability_thr,
        rep_thr=args.repeatability_thr)

    times_model_output = []
    start = time.time()
    root = args.dataset_folder
    images_path = loadImage(root)
    K = np.array([156.3536121, 0, 160, 0, 157.549850, 160, 0, 0, 1]).reshape(3, 3)  # 胃镜内参数
    start_position = np.array([0, 0, 0]).reshape(3, 1)
    one = np.array([0, 0, 0, 1]).reshape(1, 4)
    pose = init_pose()
    camera_poses = []
    save_kitti_pose = pose[0:3, :].flatten()
    camera_poses.append(pose[0:3, :].flatten())
    camera_poses.append(pose[0:3, :].flatten())
    total_match = 0
    correct_match = 0
    global_pose = []
    imageIndex = []
    goodgoodmatch = []
    times_model_output = []
    start = time.time()
    model_prefix = "history_model"
    save_root = "./testresult/" + model_prefix
    for i in range(len(images_path) - 850):
        img1 = cv.imread(images_path[i])
        img1_num = get_img_path_digital(images_path[i])
        img2 = cv.imread(images_path[i + 1])
        img2_num = get_img_path_digital(images_path[i + 1])
        npz1 = extract_keypoints(images_path[i])
        npz2 = extract_keypoints(images_path[i + 1])
        arraykeypoints1 = npz1['keypoints']
        keypoints1 = getKeypoint(arraykeypoints1)
        descriptors1 = npz1['descriptors']
        img1 = cv.drawKeypoints(img1, keypoints1, img1)
        arraykeypoints2 = npz2['keypoints']
        keypoints2 = getKeypoint(arraykeypoints2)
        descriptors2 = npz2['descriptors']
        img2 = cv.drawKeypoints(img2, keypoints2, img2)
        goodgoodmatch_ = calgoodgoodmatch(descriptors1, descriptors2)
        match_image = cv.drawMatches(img1, keypoints1, img2, keypoints2, goodgoodmatch_, outImg=None)
        match_image_save_path = save_root + "/feature_match/" + "match" + img1_num + "-" + img2_num + ".jpg"
        if os.path.exists(save_root + "/feature_arrowLine") == False:
            os.makedirs(save_root + "/feature_arrowLine")
        cv.imwrite(match_image_save_path, match_image)
        if goodgoodmatch_ is None or len(goodgoodmatch_) == 0:
            camera_poses.append(camera_poses[i - 1])
            savepose(save_root + "/pose/", imageIndex, camera_poses)
            imageIndex.append(img2_num)
            print("goodgoodmatch_ is None")
            continue
        R, t = pose_estimation(K, keypoints1, keypoints2, goodgoodmatch_)

        if R is None:
            camera_poses.append(camera_poses[i - 1])
            savepose(save_root + "/pose/", imageIndex, camera_poses)
            imageIndex.append(img2_num)
            print("R is None")
            continue
        P = np.append(R, t, axis=1)
        P = np.append(P, one, axis=0)
        pose = np.dot(pose, np.linalg.inv(P))
        camera_poses.append(pose[0:3, :].flatten())
        drawFlow(goodgoodmatch_)  # 画surf流
        savepose(save_root + "/pose/", imageIndex, camera_poses)

    # 计算指标
    # # 调用evo计算APE（默认情况下是ATE）、RPE等
    from evo import entry_points
    from evo import main_rpe
    import argcomplete

    gt_path = save_root + "\pose\position_rotation.kitti"
    main_rpe.est_file = save_root + "\pose\est_pose.txt"
    main_rpe.ref_file = gt_path
    main_rpe.model_prefix = model_prefix
    if os.path.exists(save_root + "\metric") == False:
        os.makedirs(save_root + "\metric")
    main_rpe.rpe_metric_csv_path = save_root + "\metric\RPE_metric" + model_prefix + ".csv"  # 保存csv的文件
    main_rpe.ape_metric_csv_path = save_root + "\metric\APE_metric" + model_prefix + ".csv"
    main_rpe.metric_index = str(0)
    writers = {}
    writers["test"] = SummaryWriter(
        os.path.join("./logs/" + model_prefix + "", "test"))
    writer = writers["test"]
    parser = main_rpe.parser()
    main_rpe.writer = writer
    argcomplete.autocomplete(parser)
    entry_points.launch(main_rpe, parser)
