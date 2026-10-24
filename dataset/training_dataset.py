#-*- coding:utf-8 -*-
"""Training set of OSHDRNet: Kalantari-style scene folders (three exposures *.tif sorted short -> long, an
exposure file with log2 exposure times and the HDR label); all scenes are cached in memory, samples are
random crops with random horizontal or vertical flipping."""
import os
import random

import cv2
from torch.utils.data import Dataset

from utils.utils import make_sample, read_scene


class Training_Dataset(Dataset):
    def __init__(self, root_dir, sub_set, patch_size, repeat, exposure_file_name, label_file_name):
        scenes_dir = os.path.join(root_dir, sub_set)
        self.scenes = [read_scene(os.path.join(scenes_dir, name), label_file_name, exposure_file_name)
                       for name in sorted(os.listdir(scenes_dir))]
        self.patch_size, self.repeat = patch_size, repeat

    def __len__(self):
        return len(self.scenes) * self.repeat

    def __getitem__(self, index):
        expo_times, ldr_images, label = self.scenes[index % len(self.scenes)]
        _, h, w, _ = ldr_images.shape
        y = random.randint(0, max(0, h - self.patch_size))
        x = random.randint(0, max(0, w - self.patch_size))
        ldr_images = ldr_images[:, y:y + self.patch_size, x:x + self.patch_size]
        label = label[y:y + self.patch_size, x:x + self.patch_size]
        if random.random() > 0.5:
            axis = 0 if random.random() > 0.5 else 1                         # vertical / horizontal flip
            ldr_images = [cv2.flip(ldr, axis) for ldr in ldr_images]
            label = cv2.flip(label, axis)
        return make_sample(expo_times, ldr_images, label)
