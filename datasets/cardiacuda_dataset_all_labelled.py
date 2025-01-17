#jyangcu@connect.ust.hk
import os
import random
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms

import nibabel as nib
import cv2

from tqdm import tqdm
from shutil import copyfile

from PIL import Image, ImageFilter
from torch.utils.data import Dataset

from monai.transforms import (
    AddChanneld,
    Compose,
    CenterSpatialCropd,
    Resized,
    EnsureTyped
)

cases = ['normal-23', 'normal-27', 'patient-5', 'patient-17', 'patient-35', 
         'patient-39', 'patient-44', 'patient-60', 'patient-64', 'patient-67']

class CardiacUDA_ALL_Label(Dataset):
    def __init__(self, root, view_num=['4'], length=32, blurring=True):
        self.root = root
        self.view_num = view_num
        self.length = length
        self.id_list = cases
        self.blurring = blurring

    def __getitem__(self, index):

        name = self.id_list[index]
        image_path = self.root + '/' + name + '-' + self.view_num[0] + '_image' + '/image'
        masks_path = self.root + '/' + name + '-' + self.view_num[0] + '_label.nii.gz'

        if os.path.exists(image_path):
            for _, _, fs in os.walk(image_path):
                image_list = fs

            image_list.sort()
            frames = []
            frames_org = []
            # self.length = len(image_list) - 1
            self.length = 33
            self.transform = self.get_transform()

            for i in range(len(image_list)):
                if i == self.length:
                    break
                else:
                    frame = Image.open(os.path.join(image_path + '/' + image_list[i])).convert('L')
                    frames_org.append(np.array(frame))
                    if self.blurring:
                        frame = frame.filter(ImageFilter.GaussianBlur(radius = 2))
                    # frame = frame.filter(ImageFilter.SMOOTH_MORE)
                    # frame = np.array(frame)
                    # frame = Image.fromarray(np.where(frame>frame.mean() * 3.0, frame, 0.0)).convert('L')
                    frames.append(np.array(frame))
            frames = np.stack(frames, axis=-1)
            frames_org = np.stack(frames_org, axis=-1)
            masks = np.array(nib.load(masks_path).dataobj)[:, :, :self.length]

            colors = [(1,1,1),(2,2,2),(3,3,3),(4,4,4)]
            # colors = [(1,1,1),(1,1,1),(1,1,1),(1,1,1)]
            c_masks = []
            
            h, w, _ = masks.shape
            all_cls = list(set(list(masks.reshape(-1))))
            all_cls.remove(0)
            for i in range(self.length):
                mask = masks[:, :, i]
                e_mask = np.zeros((h,w,3), np.uint8)
                for sid in [0,1,2,3]:
                    if sid > len(all_cls):
                        break
                    img_mask_c = np.where(mask == all_cls[sid], 255, 0)
                    contours, _  = cv2.findContours(img_mask_c.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    if len(contours) > 0:
                        cv2.fillPoly(e_mask, [contours[-1]], colors[sid])
                c_masks.append(e_mask[:,:,-1])

            out = self.transform({'images':frames, 'images_org':frames_org, 'masks':np.stack(c_masks, axis=-1), 'contours':masks})
            trans_image = out['images'].permute(0, 3, 1, 2)
            trans_masks = out['masks'].permute(0, 3, 1, 2)
            trans_contours = out['contours'].permute(0, 3, 1, 2)
            trans_image_org = out['images_org'].permute(0, 3, 1, 2)

            # if self.view_num   == ['1']:
            #     BG = torch.where(trans_masks == 0, 1, 0)
            #     LV = torch.where(trans_masks == 1, 1, 0)
            #     RV = torch.where(trans_masks == 2, 1, 0)
            #     trans_masks = torch.stack([BG, LV, RV], dim=0)
            # elif self.view_num == ['2']:
            #     BG = torch.where(trans_masks == 0, 1, 0)
            #     PA = torch.where(trans_masks == 1, 1, 0)
            #     trans_masks = torch.stack([BG, PA], dim=0)
            # elif self.view_num == ['3']:
            #     BG = torch.where(trans_masks == 0, 1, 0)
            #     LV = torch.where(trans_masks == 1, 1, 0)
            #     RV = torch.where(trans_masks == 2, 1, 0)
            #     trans_masks = torch.stack([BG, LV, RV], dim=0)
            # elif self.view_num == ['4']:
            #     BG = torch.where(trans_masks == 0, 1, 0)
            #     LV = torch.where(trans_masks == 1, 1, 0)
            #     LA = torch.where(trans_masks == 2, 1, 0)
            #     RA = torch.where(trans_masks == 3, 1, 0)
            #     RV = torch.where(trans_masks == 4, 1, 0)
            #     trans_masks = torch.cat([BG, LV, LA, RA, RV], dim=0)

        return trans_image / 255, trans_image_org / 255, trans_masks, trans_contours

    def __len__(self):
        return len(self.id_list)

    def get_transform(self):
        all_keys = ['images', 'masks', 'images_org', 'contours']
        first_resize_size = (620, 460, self.length)
        first_crop_size = (460, 460, self.length)
        crop_size = (256, 256, self.length)
        resize_size = (376, 376, self.length)

        transform = Compose([
                AddChanneld(keys=all_keys, allow_missing_keys=True),
                Resized(keys=all_keys, spatial_size=first_resize_size, allow_missing_keys=True, mode='nearest'),
                CenterSpatialCropd(keys=all_keys, roi_size=first_crop_size, allow_missing_keys=True),
                Resized(keys=all_keys, spatial_size=resize_size, allow_missing_keys=True, mode='nearest'),
                CenterSpatialCropd(keys=all_keys, roi_size=crop_size, allow_missing_keys=True),
                EnsureTyped(keys=all_keys, allow_missing_keys=True),
            ])

        return transform

if __name__ == '__main__':
    data_dict = dict()
    
    from monai.data import DataLoader
    train_ds = CardiacUDA_ALL_Label('/home/jyangcu/Dataset/dataset_pa_iltrasound_nill_files_clean_image/label_all_frame', view_num=['4'])
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=1)

    for _ in tqdm(train_loader):
        pass
