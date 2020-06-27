from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import time

from torch.autograd import Variable
from torch.autograd import Function
import torch.backends.cudnn as cudnn
import os
import numpy as np
from tqdm import tqdm
from model import FaceModel
from eval_metrics import evaluate
from logger import Logger
from TripletFaceDataset import TripletFaceDataset
from LFWDataset import LFWDataset
from PIL import Image
from utils import PairwiseDistance,display_triplet_distance,display_triplet_distance_test
import collections

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Face Recognition')
# Model options
parser.add_argument('--dataroot', type=str, default='/media/lior/LinuxHDD/datasets/MSCeleb-cleaned',
                    help='path to dataset')
parser.add_argument('--lfw-dir', type=str, default='/media/lior/LinuxHDD/datasets/lfw-aligned-mtcnn',
                    help='path to dataset')
parser.add_argument('--lfw-pairs-path', type=str, default='lfw_pairs.txt',
                    help='path to pairs file')

parser.add_argument('--log-dir', default='/media/lior/LinuxHDD/pytorch_face_logs',
                    help='folder to output model checkpoints')

parser.add_argument('--resume',
                    default='/media/lior/LinuxHDD/pytorch_face_logs/run-optim_adagrad-n1000000-lr0.1-wd0.0-m0.5-embeddings256-msceleb-alpha10/checkpoint_1.pth',
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=10, metavar='E',
                    help='number of epochs to train (default: 10)')
# Training options
parser.add_argument('--embedding-size', type=int, default=256, metavar='ES',
                    help='Dimensionality of the embedding')

parser.add_argument('--batch-size', type=int, default=64, metavar='BS',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='BST',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--n-triplets', type=int, default=1000000, metavar='N',
                    help='how many triplets will generate from the dataset')

parser.add_argument('--margin', type=float, default=0.5, metavar='MARGIN',
                    help='the margin value for the triplet loss function (default: 1.0')

parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.125)')
parser.add_argument('--lr-decay', default=1e-4, type=float, metavar='LRD',
                    help='learning rate decay ratio (default: 1e-4')
parser.add_argument('--wd', default=0.0, type=float,
                    metavar='W', help='weight decay (default: 0.0)')
parser.add_argument('--optimizer', default='adagrad', type=str,
                    metavar='OPT', help='The optimizer to use (default: Adagrad)')
# Device options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=10, metavar='LI',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()

# set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
# order to prevent any memory allocation on unused GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

# args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

if args.cuda:
    cudnn.benchmark = True

LOG_DIR = args.log_dir + '/run-optim_{}-n{}-lr{}-wd{}-m{}-embeddings{}-msceleb-alpha10'\
    .format(args.optimizer, args.n_triplets, args.lr, args.wd,
            args.margin,args.embedding_size)

# create logger
logger = Logger(LOG_DIR)


class TripletMarginLoss(Function):
    """Triplet loss function.
    """
    def __init__(self, margin):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin
        self.pdist = PairwiseDistance(2)  # norm 2

    def forward(self, anchor, positive, negative):
        d_p = self.pdist.forward(anchor, positive)
        d_n = self.pdist.forward(anchor, negative)

        dist_hinge = torch.clamp(self.margin + d_p - d_n, min=0.0)
        loss = torch.mean(dist_hinge)
        return loss

class Scale(object):
    """Rescales the input PIL.Image to the given 'size'.
    If 'size' is a 2-element tuple or list in the order of (width, height), it will be the exactly size to scale.
    If 'size' is a number, it will indicate the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the exactly size or the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)


kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
l2_dist = PairwiseDistance(2)

transform = transforms.Compose([
                         Scale(96),
                         transforms.ToTensor(),
                         transforms.Normalize(mean = [ 0.5, 0.5, 0.5 ],
                                               std = [ 0.5, 0.5, 0.5 ])
                     ])

train_dir = TripletFaceDataset(dir=args.dataroot,n_triplets=args.n_triplets,transform=transform)
train_loader = torch.utils.data.DataLoader(train_dir,
    batch_size=args.batch_size, shuffle=False, **kwargs)

test_dir = LFWDataset(dir=args.lfw_dir,pairs_path=args.lfw_pairs_path,transform=transform)
test_loader = torch.utils.data.DataLoader(test_dir,
    batch_size=args.batch_size, shuffle=False, **kwargs)
data_size = dict()
data_size['train'] = len(train_dir)
data_size['test'] = len(test_dir)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    # Views the training images and displays the distance on anchor-negative and anchor-positive
    test_display_triplet_distance = True

    # print the experiment configuration
    print('\nparsed options:\n{}\n'.format(vars(args)))
    print('\nNumber of Classes:\n{}\n'.format(len(train_dir.classes)))

    # instantiate model and initialize weights
    model = FaceModel(embedding_size=args.embedding_size,
                      num_classes=len(train_dir.classes),
                      pretrained=False)

    model.to(device)
    triplet_loss = TripletMarginLoss(args.margin).to(device)
    optimizer = create_optimizer(model, args.lr)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print('=> no checkpoint found at {}'.format(args.resume))

    start = args.start_epoch
    end = start + args.epochs

    for epoch in range(start, end):
        print(80 * '=')
        print('Epoch [{}/{}]', epoch, end - 1)
        time0 = time.time()
        own_train(train_loader, model, triplet_loss, optimizer, epoch, data_size)
        own_test(test_loader, model, epoch)
        print(f' Execution time    = {time.time() - time0}')
        print(80 * '=')

        if test_display_triplet_distance:
            display_triplet_distance(model,train_loader,LOG_DIR+"/train_{}".format(epoch))
            display_triplet_distance_test(model,test_loader,LOG_DIR+"/test_{}".format(epoch))

def own_train(train_loader, model, triploss, optimizer, epoch, data_size):
    model.train()
    labels, distances = [], []
    triplet_loss_sum = 0.0

    for batch_idx, (data_a, data_p, data_n, label_p, label_n) in enumerate(train_loader):
        anc_img, pos_img, neg_img = data_a.to(device), data_p.to(device), data_n.to(device)
        with torch.set_grad_enabled(True):
            anc_embed, pos_embed, neg_embed = model(anc_img), model(pos_img), model(neg_img)
            pos_dist = l2_dist.forward(anc_embed, pos_embed)
            neg_dist = l2_dist.forward(anc_embed, neg_embed)
            all = (neg_dist - pos_dist < args.margin).cpu().numpy().flatten()
            hard_triplets = np.where(all == 1)
            if len(hard_triplets) == 0:
                continue
            anc_hard_embed = anc_embed[hard_triplets]
            pos_hard_embed = pos_embed[hard_triplets]
            neg_hard_embed = neg_embed[hard_triplets]

            anc_hard_img = anc_img[hard_triplets]
            pos_hard_img = pos_img[hard_triplets]
            neg_hard_img = neg_img[hard_triplets]

            model.module.forward_classfier(anc_hard_img)
            model.module.forward_classfier(pos_hard_img)
            model.module.forward_classfier(neg_hard_img)

            triplet_loss = triploss.forward(anc_hard_embed, pos_hard_embed, neg_hard_embed)
            logger.log_value('triplet_loss', triplet_loss)
            optimizer.zero_grad()
            triplet_loss.backward()
            optimizer.step()

            adjust_learning_rate(optimizer)

            distances.append(pos_dist.data.cpu().numpy())
            labels.append(np.ones(pos_dist.size(0)))

            distances.append(neg_dist.data.cpu().numpy())
            labels.append(np.zeros(neg_dist.size(0)))

            triplet_loss_sum += triplet_loss.item()

    avg_triplet_loss = triplet_loss_sum / data_size['train']
    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for dist in distances for subdist in dist])

    tpr, fpr, accuracy, val, val_std, far = evaluate(distances, labels)

    print(' {} set - Triplet Loss   = {:.8f}'.format('train', avg_triplet_loss))
    print(' {} set - Accuracy       = {:.8f}'.format('train', np.mean(accuracy)))
    logger.log_value('Train Accuracy', np.mean(accuracy))
    plot_roc(fpr, tpr, figure_name="roc_train_epoch_{}.png".format(epoch))
    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()},
               '{}/checkpoint_{}.pth'.format(LOG_DIR, epoch))

def own_test(test_loader, model, epoch):
    # switch to evaluate mode
    model.eval()

    labels, distances = [], []
    for batch_idx, (data_a, data_p, label) in enumerate(test_loader):
        data_a, data_p = data_a.to(device), data_p.to(device)
        # compute output
        with torch.no_grad(True):
            out_a, out_p = model(data_a), model(data_p)
            dists = l2_dist.forward(out_a,out_p)#torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
            distances.append(dists.data.cpu().numpy())
            labels.append(label.data.cpu().numpy())

    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for dist in distances for subdist in dist])

    tpr, fpr, accuracy, val, val_std, far = evaluate(distances,labels, nrof_folds=3)

    print(' {} set - Accuracy       = {:.8f}'.format('Test', np.mean(accuracy)))
    logger.log_value('Test Accuracy', np.mean(accuracy))
    plot_roc(fpr,tpr,figure_name="roc_test_epoch_{}.png".format(epoch))

# def test(test_loader, model, epoch):
#     # switch to evaluate mode
#     model.eval()
#
#     labels, distances = [], []
#
#     pbar = tqdm(enumerate(test_loader))
#     for batch_idx, (data_a, data_p, label) in pbar:
#         if args.cuda:
#             data_a, data_p = data_a.to(device), data_p.to(device)
#         # compute output
#         out_a, out_p = model(data_a), model(data_p)
#         dists = l2_dist.forward(out_a,out_p)#torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
#         distances.append(dists.data.cpu().numpy())
#         labels.append(label.data.cpu().numpy())
#
#         if batch_idx % args.log_interval == 0:
#             pbar.set_description('Test Epoch: {} [{}/{} ({:.0f}%)]'.format(
#                 epoch, batch_idx * len(data_a), len(test_loader.dataset),
#                 100. * batch_idx / len(test_loader)))
#
#     labels = np.array([sublabel for label in labels for sublabel in label])
#     distances = np.array([subdist for dist in distances for subdist in dist])
#
#     tpr, fpr, accuracy, val, val_std, far = evaluate(distances,labels, nrof_folds=3)
#     print('\33[91mTest set: Accuracy: {:.8f}\n\33[0m'.format(np.mean(accuracy)))
#     logger.log_value('Test Accuracy', np.mean(accuracy))
#
#     plot_roc(fpr,tpr,figure_name="roc_test_epoch_{}.png".format(epoch))


# def train(train_loader, model, optimizer, epoch):
#     # switch to train mode
#     model.train()
#
#     pbar = tqdm(enumerate(train_loader))
#     labels, distances = [], []
#
#
#     for batch_idx, (data_a, data_p, data_n,label_p,label_n) in pbar:
#
#         data_a, data_p, data_n = data_a.to(device), data_p.to(device), data_n.to(device)
#         # compute output
#         out_a, out_p, out_n = model(data_a), model(data_p), model(data_n)
#
#         # Choose the hard negatives
#         d_p = l2_dist.forward(out_a, out_p)
#         d_n = l2_dist.forward(out_a, out_n)
#         all = (d_n - d_p < args.margin).cpu().data.numpy().flatten()
#         hard_triplets = np.where(all == 1)
#         if len(hard_triplets[0]) == 0:
#             continue
#         out_selected_a = torch.from_numpy(out_a.cpu().data.numpy()[hard_triplets]).to(device)
#         out_selected_p = torch.from_numpy(out_p.cpu().data.numpy()[hard_triplets]).to(device)
#         out_selected_n = torch.from_numpy(out_n.cpu().data.numpy()[hard_triplets]).to(device)
#
#         selected_data_a = torch.from_numpy(data_a.cpu().data.numpy()[hard_triplets]).to(device)
#         selected_data_p = torch.from_numpy(data_p.cpu().data.numpy()[hard_triplets]).to(device)
#         selected_data_n = torch.from_numpy(data_n.cpu().data.numpy()[hard_triplets]).to(device)
#
#         selected_label_p = torch.from_numpy(label_p.cpu().numpy()[hard_triplets])
#         selected_label_n= torch.from_numpy(label_n.cpu().numpy()[hard_triplets])
#         triplet_loss = TripletMarginLoss(args.margin).forward(out_selected_a, out_selected_p, out_selected_n)
#
#         cls_a = model.forward_classifier(selected_data_a)
#         cls_p = model.forward_classifier(selected_data_p)
#         cls_n = model.forward_classifier(selected_data_n)
#
#         criterion = nn.CrossEntropyLoss()
#         predicted_labels = torch.cat([cls_a,cls_p,cls_n])
#         true_labels = torch.cat([selected_label_p.to(device),selected_label_p.to(device),selected_label_n.to(device)])
#
#         cross_entropy_loss = criterion(predicted_labels.to(device),true_labels.to(device))
#
#         loss = cross_entropy_loss + triplet_loss
#         # compute gradient and update weights
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         # update the optimizer learning rate
#         adjust_learning_rate(optimizer)
#
#         # log loss value
#         logger.log_value('triplet_loss', triplet_loss.data).step()
#         logger.log_value('cross_entropy_loss', cross_entropy_loss.data).step()
#         logger.log_value('total_loss', loss.data).step()
#         if batch_idx % args.log_interval == 0:
#             pbar.set_description(
#                 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \t # of Selected Triplets: {}'.format(
#                     epoch + 1, batch_idx * len(data_a), len(train_loader.dataset),
#                     100. * batch_idx / len(train_loader),
#                     loss.data,len(hard_triplets[0])))
#
#
#         dists = l2_dist.forward(out_selected_a,out_selected_n) #torch.sqrt(torch.sum((out_a - out_n) ** 2, 1))  # euclidean distance
#         distances.append(dists.data.cpu().numpy())
#         labels.append(np.zeros(dists.size(0)))
#
#
#         dists = l2_dist.forward(out_selected_a,out_selected_p)#torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
#         distances.append(dists.data.cpu().numpy())
#         labels.append(np.ones(dists.size(0)))
#
#     labels = np.array([sublabel for label in labels for sublabel in label])
#     distances = np.array([subdist for dist in distances for subdist in dist])
#
#     tpr, fpr, accuracy, val, val_std, far = evaluate(distances,labels)
#     print('\33[91mTrain set: Accuracy: {:.8f}\n\33[0m'.format(np.mean(accuracy)))
#     logger.log_value('Train Accuracy', np.mean(accuracy))
#
#     plot_roc(fpr,tpr,figure_name="roc_train_epoch_{}.png".format(epoch))
#
#     # do checkpointing
#     torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()},
#                '{}/checkpoint_{}.pth'.format(LOG_DIR, epoch))

def plot_roc(fpr,tpr,figure_name="roc.png"):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    roc_auc = auc(fpr, tpr)
    fig = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    fig.savefig(os.path.join(LOG_DIR,figure_name), dpi=fig.dpi)


def adjust_learning_rate(optimizer):
    """Updates the learning rate given the learning rate decay.
    The routine has been implemented according to the original Lua SGD optimizer
    """
    for group in optimizer.param_groups:
        if 'step' not in group:
            group['step'] = 0
        group['step'] += 1

        group['lr'] = args.lr / (1 + group['step'] * args.lr_decay)


def create_optimizer(model, new_lr):
    # setup optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=new_lr,
                              momentum=0.9, dampening=0.9,
                              weight_decay=args.wd)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=new_lr,
                               weight_decay=args.wd)
    elif args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(),
                                  lr=new_lr,
                                  lr_decay=args.lr_decay,
                                  weight_decay=args.wd)
    return optimizer

if __name__ == '__main__':
    main()
