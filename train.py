import os, pdb
import torch
import torch.optim as optim
import cv2 as cv
from tensorboardX import SummaryWriter

from datasets import *
from tools import common, trainer, validater
from tools.dataloader import *
from nets.patchnet import *
from nets.losses import *

default_net = "Quad_L2Net_ConfCFS()"
# F:/Toky/Dataset/Endo_colon_unity/photo/
toy_db_debug = """SyntheticPairDataset(
    ImgFolder('F:/Toky/Dataset/Endo_colon_unity/photo/'), 
            'RandomScale(192,192,can_upscale=True)', 
            'RandomTilting(0.5), PixelNoise(25)')"""

db_aachen_flow = "aachen_flow_pairs"
default_dataloader = """PairLoader(CatPairDataset(`data`),
    scale   = 'RandomScale(192,192,can_upscale=True)',
    distort = 'ColorJitter(0.2,0.2,0.2,0.1)',
    crop    = 'RandomCrop(192)')"""

default_sampler = """NghSampler2(ngh=7, subq=-8, subd=1, pos_d=3, neg_d=5, border=16,
                            subd_neg=-8,maxpool_pos=True)"""
# 这种写法，用在nets.losses.py里面，前面是权值，后面是loss的名字，分别去调用
default_loss = """MultiLoss(
        1, ReliabilityLoss(`sampler`, base=0.5, nq=20),
        1, CosimLoss(N=`N`),
        1, PeakyLoss(N=`N`))"""
data_sources = dict(
    D=toy_db_debug,
)
# 验证集
toy_db_validate = """SyntheticPairDataset(
    ImgFolder('F:/Toky/Dataset/UnityCam/Recordings004/photo/'), 
            'RandomScale(192,192,can_upscale=True)', 
            'RandomTilting(0.5), PixelNoise(25)')"""
data_sources_validate = dict(
    D=toy_db_validate,
)


class NonMaxSuppression(torch.nn.Module):
    def __init__(self, rel_thr=1, rep_thr=1):
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
        # print(self.rel_thr)
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


class MyTrainer(trainer.Trainer):
    """ This class implements the network training.
        Below is the function I need to overload to explain how to do the backprop.
    """

    def forward_backward(self, inputs):
        a = [inputs.pop('img1'), inputs.pop('img2')]
        output = self.net(imgs=a)
        allvars = dict(inputs, **output)
        loss, details = self.loss_func(**allvars)
        if torch.is_grad_enabled(): loss.backward()
        return loss, details


class MyValidater(validater.Validater):
    """ This class implements the network Validating.
        Below is the function I need to overload to explain how to do the backprop.
    """

    def forward_backward(self, img):
        # create the non-maxima detector
        detector = NonMaxSuppression(
            rel_thr=args.reliability_thr,
            rep_thr=args.repeatability_thr)
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
        output = dict(keypoints=xys[idxs],
                      descriptors=desc[idxs],
                      scores=scores[idxs])  # imsize=(W, H),
        return output


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


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("Train R2D2")
    parser.add_argument('kitti',
                        help='for evo')
    parser.add_argument("--data-loader", type=str, default=default_dataloader)
    parser.add_argument("--train-data", type=str, default=list('D'), nargs='+',
                        choices=set(data_sources.keys()))
    parser.add_argument("--validate-data", type=str, default=list('D'), nargs='+',
                        choices=set(data_sources_validate.keys()))

    parser.add_argument("--net", type=str, default=default_net, help='network architecture')
    parser.add_argument("--pretrained", type=str,
                        default="F:\Toky\PythonProject\Point_feature_extrator\models\\train\colon_unity_dataset_model1025.pt",
                        help='pretrained model path')
    parser.add_argument("--save-path", type=str,
                        default="F:/Toky/PythonProject/Point_feature_extrator/models/train/",
                        help='model save_path path')
    parser.add_argument("--loss", type=str, default=default_loss, help="loss function")
    parser.add_argument("--sampler", type=str, default=default_sampler, help="AP sampler")
    parser.add_argument("--N", type=int, default=16, help="patch size for repeatability")

    parser.add_argument("--epochs", type=int, default=50, help='number of training epochs')
    parser.add_argument("--batch-size", "--bs", type=int, default=10, help="batch size")
    parser.add_argument("--learning-rate", "--lr", type=str, default=1e-4)
    parser.add_argument("--weight-decay", "--wd", type=float, default=5e-4)

    parser.add_argument("--threads", type=int, default=8, help='number of worker threads')
    parser.add_argument("--gpu", type=int, nargs='+', default=[0], help='-1 for CPU')
    parser.add_argument("--tag", type=str, default='rd', help='output file tag')

    parser.add_argument("--top-k", type=int, default=200, help='number of keypoints')

    parser.add_argument("--scale-f", type=float, default=2 ** 0.25)
    parser.add_argument("--min-size", type=int, default=0)
    parser.add_argument("--max-size", type=int, default=9999)
    parser.add_argument("--min-scale", type=float, default=0.3)
    parser.add_argument("--max-scale", type=float, default=1)

    parser.add_argument("--reliability-thr", type=float, default=0.7)
    parser.add_argument("--repeatability-thr", type=float, default=0.7)
    args = parser.parse_args()

    iscuda = common.torch_set_gpu(args.gpu)
    common.mkdir_for(args.save_path)
    # 训练数据加载
    db = [data_sources[key] for key in args.train_data]
    db = eval(args.data_loader.replace('`data`', ','.join(db)).replace('\n', ''))

    print("Training image database =", db)
    loader = threaded_loader(db, iscuda, args.threads, args.batch_size, shuffle=True)

    # 验证数据加载
    db_val = [data_sources_validate[key] for key in args.validate_data]
    db_val = eval(args.data_loader.replace('`data`', ','.join(db_val)).replace('\n', ''))
    val_loader = threaded_loader(db_val, iscuda, args.threads, batch_size=1, shuffle=False)
    # create network
    print("\n>> Creating net = " + args.net)
    net = eval(args.net)
    print(f" ( Model size: {common.model_size(net)/1000:.0f}K parameters )")

    # initialization
    if True:  # False
        net = load_network(args.pretrained)

    # create losses
    loss = args.loss.replace('`sampler`', args.sampler).replace('`N`', str(args.N))
    print("\n>> Creating loss = " + loss)
    loss = eval(loss.replace('\n', ''))

    # create optimizer
    optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad],
                           lr=args.learning_rate, weight_decay=args.weight_decay)

    train = MyTrainer(net, loader, loss, optimizer).cuda()
    validate = MyValidater(net, val_loader).cuda()
    # Training loop #

    model_prefix = "compleate_KPE_v1"
    validater.model_prefix = "compleate_KPE_v1"
    trainer.model_prefix = "compleate_KPE_v1"
    writers = {}
    writers["validate"] = SummaryWriter(
        os.path.join("F:/Toky/PythonProject/Point_feature_extrator/logs/" + model_prefix + "", "validate"))
    writers["train"] = SummaryWriter(
        os.path.join("F:/Toky/PythonProject/Point_feature_extrator/logs/" + model_prefix + "", "train"))
    validater.writer = writers["validate"]
    trainer.writer = writers["train"]
    for epoch in range(args.epochs):
        print(f"\n>> Starting epoch {epoch}...")  # 目前30个epoch
        trainer.epoch = epoch
        train()
        if epoch % 1 == 0 and epoch != 0:
            validater.epoch = epoch
            validate()
            path = args.save_path + str(epoch) + '.pt'
            torch.save({'net': args.net, 'state_dict': net.state_dict()}, path)
    print(f"\n>> Saving model to {args.save_path+'colon_unity_dataset_model1026.pt'}")
    torch.save({'net': args.net, 'state_dict': net.state_dict()}, args.save_path)
