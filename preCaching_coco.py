from pymemcache.client import base
from PIL import Image
import numpy as np
import pickle
import sys

import skimage.io
import skimage.transform
import skimage.color
import skimage

# NUM_CACHED = 5000
NUM_CACHED = 1050


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


IMAGE_LIST = './files_coco2017_val.txt' # JPEG
iList = []
with open(IMAGE_LIST, 'r') as f:
    lines = f.readlines()
    for l in lines:
        iList.append(l[:-1])

iList = iList[:NUM_CACHED]

# set client
client = base.Client(('localhost', 11211), serde=PickleSerde())
completed = 0
for ipath in iList:

    img = skimage.io.imread(ipath)
    if len(img.shape) == 2:
        img = skimage.color.gray2rgb(img)
    img = img.astype(np.float32) / 255.0
    client.set(ipath, img)
    completed += 1
    if completed % 1000 == 0:
        print('total {} / completed {}'.format(len(iList), completed))
