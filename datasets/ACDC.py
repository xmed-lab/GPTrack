#jyangcu@connect.ust.hk
import os
import random
import numpy as np
import pandas as pd

from tqdm import tqdm

import torch
from torch.utils.data import Dataset

import pydicom
import SimpleITK as sitk
from PIL import Image
import torchvision.transforms as transforms
import datasets.dataset_utils.transform as transform


np.set_printoptions(threshold=np.inf)
random.seed(7777)
np.random.seed(7777)


class ACDC_Dataset(Dataset):
    def __init__(self, args, infos, crop_length=16, min_max=[-1, 1], is_train=True):
        self.args = args
        self.crop_length = crop_length
        self.min_max = min_max
        self.train_dict = infos['train']
        self.test_dict = infos['test']
        self.all_dict = self.preprocess(is_train)
        self.file_list = list(self.all_dict.keys())
        self.fineSize = [crop_length, 16, 128, 128]

        self.transform = transforms.Compose([
                                transform.ToTensorVideo(),
                                transform.CenterCropVideo((128, 128)),
                                transform.RandomHorizontalFlipVideo(p=0.5),
                                transform.Rotate(p=0.5),
                                # transform.Color_jitter(p=0.3),
                                # transform.Equalize(p=0.3),
                                ])

    def __getitem__(self, index):
        index = index // 10
        case = self.all_dict[self.file_list[index]]
        data=sitk.ReadImage(case['nill_path'])
        data=sitk.GetArrayFromImage(data).astype(np.float32) 
        length, depth, height, width = data.shape

        current_video = data
        if length < self.crop_length:
            comp_length = self.crop_length - length
            comp_frames = np.flip(current_video[-1-comp_length:-1, ...], axis=0)
            current_video = np.concatenate((current_video, comp_frames), axis=0)
        
        elif length > self.crop_length:
            start_idx = random.randint(0, length-self.crop_length-1)
            current_video = current_video[start_idx:start_idx+self.crop_length, ...]

        current_video = current_video - current_video.min()
        current_video = current_video / current_video.std()
        current_video = current_video - current_video.min()
        current_video = current_video / current_video.max()
        # Normalize & Transformation

        if depth >= self.fineSize[1]:
            sd = int((depth - self.fineSize[1]) / 2)
            current_video = current_video[:, sd:sd + self.fineSize[1], ...]
        else:
            sd = int((self.fineSize[1] - depth) / 2)
            current_video_ = np.zeros([self.crop_length, self.fineSize[1], height, width])
            current_video_[:, sd:sd + depth] = current_video
            current_video = current_video_

        current_video = self.transform(np.transpose(current_video, axes=(0,2,3,1)))
        current_video = current_video*(self.min_max[1] - self.min_max[0]) + self.min_max[0]
        
        return current_video.permute(0, 2, 3, 1).unsqueeze(1)

    def __len__(self):
        return len(self.file_list) * 10
    
    def preprocess(self, is_train):
        all_dict = dict()
        count = 0
        if is_train:
            for key in list(self.train_dict.keys()):
                all_dict[count] = (self.train_dict[key])
                count += 1
        else:
            for key in list(self.test_dict.keys()):
                all_dict[count] = (self.test_dict[key])
                count += 1
        return all_dict
    
    def augment(self, img_list, hflip=True, rot=True, split='val'):
    # horizontal flip OR rotate
        hflip = hflip and (split == 'train' and random.random() < 0.5)
        vflip = rot and (split == 'train' and random.random() < 0.5)
        rot90 = rot and (split == 'train' and random.random() < 0.5)
        def _augment(img):
            if hflip:
                img = img[:, ::-1, :]
            if vflip:
                img = img[::-1, :, :]
            if rot90:
                img = img.transpose(1, 0, 2)
            return img

        return [_augment(img) for img in img_list]


    def transform2numpy(self, img):
        img = np.array(img)
        img = img.astype(np.float32)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        return img


    def transform2tensor(self, img, min_max=(0, 1)):
        # HWD to 1DHW
        img = torch.from_numpy(np.ascontiguousarray(
            np.transpose(img, (2, 0, 1)))).float().unsqueeze(0)
        # to range min_max
        img = img*(min_max[1] - min_max[0]) + min_max[0]
        return img


    def transform_augment(self, img_list, split='val', min_max=(0, 1)):
        ret_img = []
        img_list = self.augment(img_list, split=split)
        for img in img_list:
            img = self.transform2numpy(img)
            img = self.transform2tensor(img, min_max)
            ret_img.append(img)
        return ret_img


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--image-size', type=int, default=(112,112,16), help='Image height and width (default: 256)')
    parser.add_argument('--mask-size', type=int, default=8, help='The size of mask patch (default: 16)')
    parser.add_argument('--mask-ratio', type=float, default=0.6, help='The ratio of masking area in an image (default: 0.75)')
    args = parser.parse_args()
    args.dataset_path = ['/home/jyangcu/Dataset/ACDC/train', '/home/jyangcu/Dataset/ACDC/test']
    args.new_dataset_path = ['/home/jyangcu/Dataset/ACDC_image/train', '/home/jyangcu/Dataset/ACDC_image/test']
    infos = np.load('/home/jyangcu/Pulmonary_Arterial_Hypertension/datasets/dataset_utils/ACDC_info.npy', allow_pickle=True).item()
    
    from monai.data import DataLoader
    from torchvision import utils as vutils

    train_ds = ACDC_Dataset(args, infos)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=1)

    from einops import rearrange
    count = 0
    for _ in tqdm(train_loader):
        count += 1
        # if count == 1:
        #     break
        # else:
        #     pass