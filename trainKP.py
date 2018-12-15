from data import *
from utils.augmentations import SSDAugmentation
from layers.modules.multibox_loss import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse

import utilsKP

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=64, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=2, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.device("cuda:0")
else:
    torch.set_default_tensor_type('torch.FloatTensor')
    device  = torch.device("cpu")

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

def train():
    NUMB_EPOCHS = 50
    EPOCH_WHERE_WHOLE_NETWORK_TRAINED = 15
    BEST_WEIGHTS_FILE = 'model_best_weights_10.pth'
    MAX_LEARNING_RATE = 1e-3
    MIN_LEARNING_RATE = 1e-4
    # STEP_SIZE = [5, 5, 5, 5, 10, 10, 10]
    STEP_SIZE=[1]
    numb_epochs = sum(STEP_SIZE)*2

    cfg = voc
    dataset = VOCDetection(root=args.dataset_root,image_sets=[('2007', 'trainval')],
                               transform=SSDAugmentation(cfg['min_dim'],MEANS))

    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    # ssd_net = torch.nn.DataParallel(ssd_net)

    print('Initializing weights...')
    ssd_net.extras.apply(weights_init)
    ssd_net.loc.apply(weights_init)
    ssd_net.conf.apply(weights_init)

    # show the layer sizes, given an input of size 3 channels by 300x300
    # summary(ssd_net, (3, 300, 300))
    # print(ssd_net.vgg)
    print(ssd_net)
    vgg_weights = torch.load(args.save_folder + args.basenet)

    print('Loading base network...')
    ssd_net.vgg.load_state_dict(vgg_weights)

    if args.cuda:
        ssd_net = ssd_net.cuda()

    #freeze the base network for now, unfreeze after a couple of epochs of training
    utilsKP.do_requires_grad(ssd_net.vgg, requires_grad=False)

    # pytorch optimizer ONLY accepts parameter that requires grad
    # the first param chooses just the layers that require gradient
    # see https://github.com/pytorch/pytorch/issues/679
    # optimizer = optim.SGD(filter(lambda p: p.requires_grad, ssd_net.parameters()), lr=args.lr, momentum=args.momentum,
    #                       weight_decay=args.weight_decay)
    optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, ssd_net.parameters()), lr=args.lr,weight_decay=args.weight_decay)

    # # this raises ValueError: optimizing a parameter that doesn't require gradients
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
    #                       weight_decay=args.weight_decay)

    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)

    # learning rate = lr for epoch 1,2,3,4 - then lr/10 for 5-then lr/100 for 4 and 5
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 6], gamma=0.1)

    #lets try this new fancy cyclic learning rate
    # scheduler = utilsKP.CyclicLR(optimizer, base_lr=1e-4, max_lr=5e-3,step_size = 150)

    #or how about triangular or cosign annealing with warm restarts
    # lr = utilsKP.TriangularLR()
    lr = utilsKP.TriangularLR_LRFinder()
    # lra = utilsKP.LR_anneal_linear()
    lra = None
    scheduler = utilsKP.CyclicLR_Scheduler(optimizer, min_lr=MIN_LEARNING_RATE, max_lr=MAX_LEARNING_RATE,LR=lr,LR_anneal=lra,batch_size = 64,numb_images = utilsKP.NUMB_IMAGES, step_size = STEP_SIZE)

    # we are going to train so set up gradients
    ssd_net.train()

    # loss counters
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print(f"len(dataset)={len(dataset)}, batch_size={args.batch_size}")
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)

    # logging for tensorflow
    writer = utilsKP.Writer('./runs')

    # #use same writer for LR
    # scheduler.setWriter(writer)

    all_batch_cntr =0
    loss_lowest = 10000

    for epoch in range(numb_epochs):
        print(f"Starting epoch {epoch}")

        #we are going to use a lot of memory for the model and for the images, see whats being used on the GPU below
        print(f"torch.cuda.memory_allocated()= {torch.cuda.memory_allocated()/1000000} megabytes")
        print(f"torch.cuda.memory_cached()= {torch.cuda.memory_cached()/1000000} megabytes")
        print(f"total cuda memory= {torch.cuda.memory_allocated()/1000000 + torch.cuda.memory_cached()/1000000} megabytes")

        #create a new iterator over training data
        batch_iterator = iter(data_loader)

        # reset epoch loss counters
        loc_loss = 0
        conf_loss = 0

        #step the learning rates (for non cyclic)
        # scheduler.step()

        #for first few epochs do not backprop through vgg
        #train custom head first
        if (epoch == EPOCH_WHERE_WHOLE_NETWORK_TRAINED):
            utilsKP.do_requires_grad(ssd_net.vgg, requires_grad=True, apply_to_this_layer_on=24)
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, ssd_net.parameters()), lr=args.lr, momentum=args.momentum,
                                  weight_decay=args.weight_decay)
        # #########################
        # #dump this
        # if iteration in cfg['lr_steps']:
        #     step_index += 1
        #     adjust_learning_rate(optimizer, args.gamma, step_index)
        # #########################


        #iterate until finish epoch
        for batch_cntr, (images, targets) in enumerate(batch_iterator):

            #always want em on cuda
            if args.cuda:
                images = Variable(images.cuda())
                with torch.no_grad():
                    targets = [Variable(ann.cuda()) for ann in targets]
            else:
                images = Variable(images)
                with torch.no_grad():
                    targets = [Variable(ann) for ann in targets]

            # forward
            out = ssd_net(images)

            # backprop
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c

            # save the best model
            if  loss < loss_lowest:
                # print(f"New lowest loss! Was {loss_lowest} is now {loss}")
                loss_lowest = loss
                torch.save(ssd_net.state_dict(),
                           args.save_folder + '' + BEST_WEIGHTS_FILE)

            loss.backward()
            optimizer.step()
            scheduler.batch_step()    #for cyclic learning rate ONLY

            #keep track of these
            loc_loss += loss_l.item()
            conf_loss += loss_c.item()

            if batch_cntr % 1 == 0:
                print(f'batch_cntr ={batch_cntr} || Loss: {loss.item()}')

                all_batch_cntr += batch_cntr
                writer('loss_L', loss_l.item(), all_batch_cntr)
                writer('loss_C', loss_c.item(), all_batch_cntr)
                writer('loss_Total', loss, all_batch_cntr)
                writer('learning_rate', scheduler.cur_lr, loss)

# test for same behaviour
def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

if __name__ == '__main__':
    train()
