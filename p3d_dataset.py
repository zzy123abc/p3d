import torch
import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np

class ClipRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def start_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])

class P3DDataSet(data.Dataset):
    def __init__(self, list_file,length=16, modality='RGB',image_tmpl='frame{:06d}.jpg', transform=None):

        self.list_file = list_file
        self.length = length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform

        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB':
            return Image.open(os.path.join(directory, self.image_tmpl.format(idx)))

    def _parse_list(self):
        self.clip_list = [ClipRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def __getitem__(self, index):
        record = self.clip_list[index]
        return self.get(record)

    def get(self, record):
        clips = list()
        for i in xrange(self.length):
            img = self._load_image(record.path,i+record.start_frames)
            img = self.transform(img)
            img = img.numpy()
            clips.append(img)
        clips = np.array(clips)
        clips = clips.transpose(1,0,2,3)
        clips = torch.from_numpy(clips)
        return clips, record.label

    def __len__(self):
        return len(self.clip_list)
