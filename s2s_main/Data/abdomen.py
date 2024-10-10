import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset

class Abdomen1(Dataset):

    def __init__(self, root, seed, type):
        self.seed = seed
        self.type = type
        self.root = root

        if self.type=='train':
            self.file_list = os.path.join(root, 'train', 'img')
            self.label_list = os.path.join(root, 'train', 'label')
        else:
            self.file_list = os.path.join(root, 'test', 'img')
            self.label_list = os.path.join(root, 'test', 'label')

        self.patients_list = os.listdir(self.file_list)
        self.patients_label = os.listdir(self.label_list)
        self.patients_list.sort()
        self.patients_label.sort()
    

    def __getitem__(self, index):
        patient_dir = self.patients_list[index]
        patient_label = self.patients_label[index]
        
        img_path = os.path.join(self.file_list, patient_dir)
        label_path = os.path.join(self.label_list, patient_label)

        img = np.load(img_path)
        label = np.load(label_path)
        img, label = self.aug_sample(img, label)

        cls_list = [1, 2, 3, 4, 6, 7, 8, 11]
        seg_volume = []
        for cls in cls_list:
            seg_volume.append((label==cls))
        seg_volume = np.concatenate(seg_volume, axis=0).astype("float32")
        label = seg_volume

        return (torch.tensor(img.copy(), dtype=torch.float),
                    torch.tensor(label.copy(), dtype=torch.float))


    def __len__(self):
        return len(self.patients_list)


    def aug_sample(self, x, y):
        if random.random() < 0.5:
            x = np.flip(x, axis=0)
            y = np.flip(y, axis=0)
        if random.random() < 0.5:
            x = np.flip(x, axis=1)
            y = np.flip(y, axis=1)

        x = np.expand_dims(x, axis=0)
        y = np.expand_dims(y, axis=0)
        return x, y
