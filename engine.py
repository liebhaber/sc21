import math
import sys
import time
import torch

import utils
def train_one_epoch(model, optimizer, data_loader, device):
    model.train()

    completed = 0
    bTime = 0.0
    mTime = 0.0
    cTime = 0.0
    st = time.time()
    start = st

    for i, (images, targets) in enumerate(data_loader):
        bTime += time.time() - st
        st = time.time()
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]


        mTime += time.time() - st

        st = time.time()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        print(loss_value)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        cTime += time.time() - st
        completed += 1
        print('{} / {} completed'.format(len(data_loader), completed))
        st = time.time()
    print(
        'batching time: {:.6f}, memcpy time: {:.6f}, computation time: {:.6f}, total time: {:.6f}'.format(
            bTime, mTime, cTime,
            time.time() - start))

