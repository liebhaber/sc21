import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.datasets import DatasetFolder
import torchvision.transforms as transforms

import time
import pickle
from PIL import Image
from pymemcache.client import base


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

class PickleSerde(object):
    def serialize(self, key, value):
        if isinstance(value, str):
            return value, 1
        return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL), 2

    def deserialize(self, key, value, flags):
        if flags == 1:
            return value
        if flags == 2:
            return pickle.loads(value)
        raise Exception("Unknown flags for value: {}".format(flags))


def pil_loader_wCache(path, cache):
    #open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    cached_data = cache.get(path)
    if cached_data is None:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    else:
        return cached_data


class ImageFolder_wCache(DatasetFolder):
    def __init__(self, root, cache, transform=None, target_transform=None,
                 loader=pil_loader_wCache, is_valid_file=None):
        super(ImageFolder_wCache, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples
        self.cache = base.Client(cache, serde=PickleSerde())

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path, self.cache)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target



if __name__ == '__main__':

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
    CACHE = ('localhost', 11211)
    DATA_DIR = '/home/ncl/twkim/datasets/ilsvrc2012/ILSVRC2012_img_val_subfolders'  # ssd - jpeg

    BATCH_SIZE = 64
    NUM_WORKERS = 0

    ARCH = 'resnet50'

    LR = 0.1
    MOM = 0.9
    WD = 1e-4

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    imagenet_dataset = ImageFolder_wCache(DATA_DIR, CACHE, transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize]))

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
    cTime = 0.0
    st = time.time()
    start = st

    for i, (input, target) in enumerate(imagenet_loader):
        input = input.cuda(non_blocking=False)
        target = target.cuda(non_blocking=False)
        bTime += time.time() - st

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

    print('[{} batches] batching time: {:.6f}, computation time: {:.6f}, total time: {:.6f}'.format(BATCH_SIZE,
                                                                                                    bTime, cTime,
                                                                                                    time.time()-start))

