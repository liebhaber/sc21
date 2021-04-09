from torchvision.datasets.voc import VOCDetection
from PIL import Image
import xml.etree.ElementTree as ET
import torch

class_to_num ={
'background':0,
'aeroplane':1,
'bicycle':2,
'bird':3,
'boat':4,
'bottle':5,
'bus':6,
'car':7,
'cat':8,
'chair':9,
'cow':10,
'diningtable':11,
'dog':12,
'horse':13,
'motorbike':14,
'person':15,
'pottedplant':16,
'sheep':17,
'sofa':18,
'train':19,
'tvmonitor':20
}

class VOCDetection_tw(VOCDetection):
    def __init__(self,
                 root,
                 year='2012',
                 image_set='train',
                 download=False,
                 transform=None,
                 target_transform=None,
                 transforms=None):
        super(VOCDetection_tw, self).__init__(root, year, image_set, download, transform, target_transform,transforms)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = self.parse_voc_xml(
            ET.parse(self.annotations[index]).getroot())

        # target = dict(image_id=index, annotations=target['annotation'])
        # print(target)
        obj_list = target['annotation']['object']
        boxes = []
        labels = []
        for i, obj_dict in enumerate(obj_list):
            boxes.append([int(pos) for pos in obj_dict['bndbox'].values()])
            labels.append(class_to_num[obj_dict['name']])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([index])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(obj_list),), dtype=torch.int64)

        target = {}

        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
