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

DATA_DIR = '/home/ncl/twkim/datasets/satellite_merged_hj/'

BATCH_SIZE = 32
NUM_WORKERS = 0

ARCH = 'shufflenet_v2_x1_0'
EPOCHS = 40

LR = 0.01
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
    cls_to_idx = {'airfield': 0, 'anchorage': 1, 'beach': 2, 'dense_residential': 3,
                  'farm': 4, 'flyover': 5, 'forest': 6, 'game_space': 7,
                  'parking_space': 8, 'river': 9, 'sparse_residential': 10, 'storage_cisterns': 11}

    input_size = 224

    a = time.time()

    train_dit = os.path.join(DATA_DIR, 'train')
    test_dir = os.path.join(DATA_DIR, 'test')

    normalize = transforms.Normalize(mean=[0.3675, 0.3803, 0.3394],
                                     std=[0.1489, 0.1392, 0.1346])

    # rotate training transform
    # WARNING: no augmentation
    cls_training_transform = transforms.Compose([transforms.Resize((input_size, input_size)),
                                                 transforms.ToTensor(),
                                                 normalize])


    # # train dataset loader
    # train_dataset = datasets.ImageFolder(train_dit, transform=cls_training_transform)

    # train dataset loader
    train_dataset = ImageFolder_wCache(train_dit, CACHE, transform=cls_training_transform)


    # rot_train_dataset.cls_to_idx = cls_to_idx
    train_dataset.cls_to_idx = cls_to_idx

    print(len(train_dataset))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                                   num_workers=NUM_WORKERS, pin_memory=True)

    print('cls_train {}'.format(len(train_loader)))


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
    cTime = 0.0
    mTime = 0.0
    st = time.time()
    start = st

    for _ in range(EPOCHS):
        completed = 0
        for i, (input, target) in enumerate(train_loader):
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
            print('{} / {} completed'.format(len(train_loader), completed))
            st = time.time()
            if completed == 376:
                print('[{} batches] batching time: {:.6f}, memcpy time: {:.6f}, computation time: {:.6f}, total time: {:.6f}'.format(BATCH_SIZE,
                                                                                                                               bTime, mTime, cTime,
                                                                                                                               time.time()-start))


