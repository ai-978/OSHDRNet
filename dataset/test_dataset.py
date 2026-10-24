#-*- coding:utf-8 -*-
"""Test set of OSHDRNet: full-resolution scenes in the same layout as the training set (cached in memory).
Tel's and Hu's datasets are read with their own exposure / label file names."""
import os

from torch.utils.data import Dataset

from utils.utils import make_sample, read_scene


class Test_Dataset(Dataset):
    def __init__(self, root_dir, sub_set, exposure_file_name, label_file_name):
        scenes_dir = os.path.join(root_dir, sub_set)
        self.names = sorted(os.listdir(scenes_dir))
        self.scenes = [read_scene(os.path.join(scenes_dir, name), label_file_name, exposure_file_name)
                       for name in self.names]

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index):
        sample = make_sample(*self.scenes[index])
        sample["name"] = self.names[index]
        return sample
