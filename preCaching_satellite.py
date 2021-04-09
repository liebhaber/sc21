from pymemcache.client import base
from PIL import Image
import numpy as np
import pickle
import sys

# NUM_CACHED = 24012
NUM_CACHED = 14408


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


IMAGE_LIST = './files_satellite_merged_hj_train.txt' # JPEG
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
    with open(ipath, 'rb') as f:
        img = Image.open(f)
        load = img.convert('RGB')
        client.set(ipath, load)
        completed += 1
    if completed % 10000 == 0:
        print('total {} / completed {}'.format(len(iList), completed))
