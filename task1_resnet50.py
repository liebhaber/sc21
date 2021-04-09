import sys
import time
import os
#import psutil

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import time
import PIL

from ImageFolder_wCache import ImageFolder_wCache

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

#DATA_DIR = '/home/ncl/twkim/datasets/ilsvrc2012/ILSVRC2012_img_val_subfolders'
DATA_DIR = '/home/ncl/twkim/project/add/cxp-nfs/ilsvrc2012/ILSVRC2012_img_val_subfolders'

BATCH_SIZE = 64
NUM_WORKERS = 0

ARCH = 'resnet50'
EPOCHS = 40

LR = 0.1
MOM = 0.9
WD = 1e-4

CACHE = ('localhost', 11211)

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    a = time.time()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    imagenet_dataset = datasets.ImageFolder(DATA_DIR, transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTens
        or(),
        normalize]))

    # imagenet_dataset = ImageFolder_wCache(DATA_DIR, CACHE, transforms.Compose([
    #     transforms.Resize(224),
    #     transforms.CenterCrop(224),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     normalize]))
    imagenet_loader = torch.utils.data.DataLoader(imagenet_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                               num_workers=NUM_WORKERS, pin_memory=True)

    # create model
    print("=> creating model '{}'".format(ARCH))
    model = models.__dict__[ARCH]()
    model = model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), LR,
                                momentum=MOM,
                                weight_decay=WD)

    completed = 0
    bTime = 0.0
    mTime = 0.0
    cTime = 0.0
    st = time.time()
    start = st

    for epoch in range(EPOCHS):
        completed = 0
        for i, (input, target) in enumerate(imagenet_loader):
            bTime += time.time() - st
            st = time.time()
            input = input.cuda(non_blocking=False)
            target = target.cuda(non_blocking=False)
            mTime += time.time() - st

            # compute output
            st = time.time()
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cTime += time.time() - st

            completed += 1
            print('{} / {} completed'.format(len(imagenet_loader), completed))
            st = time.time()

            if completed == 391:
                print('[{} batches] batching time: {:.6f}, memcpy time: {:.6f}, computation time: {:.6f}, total time: {:.6f}'.format(BATCH_SIZE,
                                                                                                                               bTime, mTime, cTime,
                                                                                                                               time.time()-start))

