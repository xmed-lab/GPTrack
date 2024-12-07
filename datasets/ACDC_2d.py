#jyangcu@connect.ust.hk
import os
import random
import numpy as np
import pandas as pd

from tqdm import tqdm
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
    def __init__(self, args, infos, crop_length=16, min_max=[-1, 1]):
        self.args = args
        self.crop_length = crop_length
        self.min_max = min_max
        self.train_dict = infos['train']
        self.test_dict = infos['test']
        self.all_dict = self.preprocess()
        self.file_list = list(self.all_dict.keys())
        # self.fine_size = [16, 192, 192]

        self.transform = transforms.Compose([
                                transform.ToTensorVideo(),
                                transform.CenterCropVideo((128, 128)),
                                transform.RandomHorizontalFlipVideo(p=0.5),
                                transform.Rotate(p=0.5),
                                transform.Color_jitter(p=0.5),
                                # transform.Equalize(p=0.3),
                                ])


    def __getitem__(self, index):
        index = index // 15
        case = self.all_dict[self.file_list[index]]
        data=sitk.ReadImage(case['nill_path'])
        data=sitk.GetArrayFromImage(data).astype(np.float32) 
        length, depth, width, height = data.shape

        depth_idx = random.randint(0, depth-1)
        current_video = data[:, depth_idx, ...]

        if length < self.crop_length:
            comp_length = self.crop_length - length
            comp_frames = np.flip(current_video[-1-comp_length:-1, ...], axis=0)
            current_video = np.concatenate((current_video, comp_frames), axis=0)
        
        elif length > self.crop_length:
            start_idx = random.randint(0, length-self.crop_length-1)
            current_video = current_video[start_idx:start_idx+self.crop_length, ...]

        # if width < self.fine_size[1] and height < self.fine_size[2]:
        #     current_video_ = np.zeros(self.fine_size)
        #     width_sf = (self.fine_size[1]-width) // 2
        #     height_sf = (self.fine_size[2]-height) // 2
        #     current_video_[:, width_f:width+width_sf, height_sf:height+height_sf] = current_video
        #     current_video = current_video_

        current_video = current_video / current_video.std()
        current_video = current_video - current_video.min()
        current_video = current_video / current_video.max()

        # Normalize & Transformation
        current_video = self.transform(np.expand_dims(current_video, axis=-1))

        return current_video.transpose(0,1)

    def __len__(self):
        return len(self.file_list) * 15
    
    def preprocess(self):
        all_dict = dict()
        count = 0
        for key in list(self.train_dict.keys()):
            all_dict[count] = (self.train_dict[key])
            count += 1
        for key in list(self.test_dict.keys()):
            all_dict[count] = (self.test_dict[key])
            count += 1
        return all_dict


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
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=1)

    from einops import rearrange
    count = 0
    for _ in tqdm(train_loader):
        count += 1
        # if count == 1:
        #     break
        # else:
        #     pass