import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset

class uniDataset_2D(Dataset):
    def __init__(self, args, name, type, seed):
        self.args = args
        self.seed = seed
        self.type = type
        self.file_list = os.path.join(args.root, self.type, 'img')
        self.label_list = os.path.join(args.root, self.type, 'label')
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
        img = self.normlize(img)
        label = np.load(label_path)
        ar, area = np.unique(label,return_counts=True)
        area = area.tolist()
        area = np.array(area)
        img, label = self.aug_sample(img, label)
 
        _, h, w, d = label.shape
        seg_volume = np.zeros([self.args.num_classes, h, w, d])
        for i in range(len(ar)-1):
            cls = int(ar[i+1])
            temp = (label==cls)
            seg_volume[cls-1,:,:,:] = temp

        seg_volume = seg_volume.astype("float32")
        label = seg_volume
        return (torch.tensor(img.copy(), dtype=torch.float),
                    torch.tensor(label.copy(), dtype=torch.float))


    def __len__(self):
        return len(self.patients_list)
       

    def normlize(self, x):
        assert x.size != 0
        return (x - x.min()) / (x.max() - x.min())


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


    def center_crop(self, x, y):
        crop_size = np.array(self.args.img_size)
        height, width = x.shape[-2:]
        sx = (height - crop_size[0]) // 2
        sy = (width - crop_size[1]) // 2

        crop_volume = x[sx:sx + crop_size[0], sy:sy + crop_size[1]]
        crop_seg = y[sx:sx + crop_size[0], sy:sy + crop_size[1]]
        crop_volume = np.expand_dims(crop_volume, axis=0)
        crop_seg = np.expand_dims(crop_seg, axis=0)

        return crop_volume, crop_seg


    def get_pad_3d_image(self, pad_ref: tuple = (64, 64), zero_pad: bool = True):
        def pad_3d_image(image):
            if zero_pad:
                value_to_pad = 0
            else:
                value_to_pad = image.min()
            # print("image.shape = {}".format(image.shape))
            if value_to_pad == 0:
                image_padded = np.zeros(pad_ref)
            else:
                image_padded = value_to_pad * np.ones(pad_ref)
            pad = (np.array(pad_ref) - np.array(image.shape))//2
            sx = (pad_ref[0] - image.shape[0] - 1) // 2
            sy = (pad_ref[1] - image.shape[1] - 1) // 2
            image_padded[sx:sx + image.shape[0], sy:sy + image.shape[1]] = image
            image_padded = np.expand_dims(image_padded, axis=0)
            return image_padded
        return pad_3d_image


