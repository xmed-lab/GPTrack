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
import dataset_utils.transform as transform

from monai.transforms import (
    AddChanneld,
    Compose,
    CenterSpatialCropd,
    Resized,
    EnsureTyped
)

cases = ['normal-23', 'normal-27', 'patient-5', 'patient-17', 'patient-35', 
         'patient-39', 'patient-44', 'patient-60', 'patient-64', 'patient-67']

class Seg_PAHDataset(Dataset):
    def __init__(self, root, view_num=['4'], length=32, blurring=True):
        self.root = root
        self.view_num = view_num
        self.length = length
        self.id_list = cases
        self.blurring = blurring

        # self.transform = self.get_transform()
        self.transform = transforms.Compose([
                                transform.ToTensorVideo(),
                                #transform.NormalizeVideo(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]),
                                transform.ResizedVideo((620, 460)),
                                transform.ResizedVideo((168, 168,), p=0.3),
                                transform.RandomCropVideo((128, 128)),
                                transform.RandomHorizontalFlipVideo(),
                                transform.Rotate(p=0.4),
                                transform.Color_jitter(p=0.3),
                                transform.Equalize(p=0.3),
                                ])

    def __getitem__(self, index):

        name = self.id_list[index]
        image_path = self.root + '/' + name + '-' + self.view_num[0] + '_image' + '/image'
        masks_path = self.root + '/' + name + '-' + self.view_num[0] + '_label.nii.gz'

        if os.path.exists(image_path):
            for _, _, fs in os.walk(image_path):
                image_list = fs

            image_list.sort()
            frames = []
            for i in range(len(image_list)):
                if i == self.length:
                    break
                else:
                    frame = Image.open(os.path.join(image_path + '/' + image_list[i])).convert('L')
                    if self.blurring:
                        frame = frame.filter(ImageFilter.MedianFilter(7))
                    frames.append(np.array(frame))
            
            frames = np.stack(frames, axis=-1)
            masks = np.array(nib.load(masks_path).dataobj)[:, :, :32]
            frames = np.expand_dims(frames, axis=-1)
            frames = np.transpose(frames, (2, 0, 1, 3))
            print(np.max(frames))
            out = self.transform(frames)
            print(torch.max(out))

            out = self.transform({'images':frames, 'masks':masks})
            trans_image = out['images'].permute(0, 3, 1, 2)
            trans_masks = out['masks'].permute(0, 3, 1, 2)

            if self.view_num   == ['1']:
                BG = torch.where(trans_masks == 0, 1, 0)
                LV = torch.where(trans_masks == 1, 1, 0)
                RV = torch.where(trans_masks == 2, 1, 0)
                trans_masks = torch.stack([BG, LV, RV], dim=0)
            elif self.view_num == ['2']:
                BG = torch.where(trans_masks == 0, 1, 0)
                PA = torch.where(trans_masks == 1, 1, 0)
                trans_masks = torch.stack([BG, PA], dim=0)
            elif self.view_num == ['3']:
                BG = torch.where(trans_masks == 0, 1, 0)
                LV = torch.where(trans_masks == 1, 1, 0)
                RV = torch.where(trans_masks == 2, 1, 0)
                trans_masks = torch.stack([BG, LV, RV], dim=0)
            elif self.view_num == ['4']:
                BG = torch.where(trans_masks == 0, 1, 0)
                LV = torch.where(trans_masks == 1, 1, 0)
                LA = torch.where(trans_masks == 2, 1, 0)
                RA = torch.where(trans_masks == 3, 1, 0)
                RV = torch.where(trans_masks == 4, 1, 0)
                trans_masks = torch.cat([BG, LV, LA, RA, RV], dim=0)
        return trans_image / 127.5 - 1, trans_masks, 0

    def __len__(self):
        return len(self.id_list)

    def compute_pixel_discrepancy(self, image_list):
        if len(image_list.shape) == 2:
            image_list = np.expand_dims(image_list, axis=2)
        w, h, t = image_list.shape
        discrepancy_image = np.zeros((w, h))
        for i in range(w):
            for j in range(h):
                pixel_array = image_list[i, j, :]
                pixel_mean = np.full_like(pixel_array.shape, np.mean(pixel_array))
                pixel_discrepancy = ((pixel_array - pixel_mean)**2).mean()
                discrepancy_image[i, j] = pixel_discrepancy
        discrepancy_image = np.where(discrepancy_image > 0.01, 1, 0)
        discrepancy_image = np.repeat(np.expand_dims(discrepancy_image, axis=2), t, axis=2)

        if len(image_list.shape) == 2:
            discrepancy_image = np.squeeze(discrepancy_image)
            image_list = np.squeeze(image_list)

        return image_list * discrepancy_image

    def get_transform(self):
        all_keys = ['images', 'masks']
        first_resize_size = (620, 460, self.length)
        first_crop_size = (460, 460, self.length)
        crop_size = (192, 192, self.length)
        resize_size = (272, 272, self.length)

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
    train_ds = Seg_PAHDataset('/home/jyangcu/Dataset/dataset_pa_iltrasound_nill_files_clean_image/label_all_frame', view_num=['4'])
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=1)

    for _ in tqdm(train_loader):
        pass


import copy
import os.path

import cv2
import matplotlib.pyplot as plt
from monai.data import DataLoader
import monai.transforms
import numpy as np
import torch
import torch.nn as nn
import nibabel as nib
import random
"""
将轮廓转换为分割用的mask

target shape:5*112*112*40
"""
from monai.transforms import (
    AddChanneld,
    Compose,
    CenterSpatialCropd,
    Resized,
    EnsureTyped
)
np.set_printoptions(threshold=np.inf)
random.seed(6666)
np.random.seed(6666)

def get_transform(clip_length = 40):
    all_keys = ['images', 'masks']
    crop_size = (112, 112, clip_length)
    spatial_size = (144, 144, clip_length)# 144 144
    transform = Compose([
        AddChanneld(keys=all_keys, allow_missing_keys=True),
        Resized(keys=all_keys, spatial_size=spatial_size, allow_missing_keys=True, mode='nearest'),
        CenterSpatialCropd(all_keys, crop_size, allow_missing_keys=True),
        EnsureTyped(keys=all_keys, allow_missing_keys=True),
    ])
    return transform

def mask_to_allclass(masks,view):

    if len(masks.shape)==3:
        _, h, w = masks.shape
        if view == '1':
            # 胸骨旁左室长轴切面
            tmp_mask = torch.zeros(5,h,w)
            tmp_mask[1] = masks[1]
            tmp_mask[3] = masks[0]

        if view == '2':
            # 肺动脉长轴切面
            tmp_mask = torch.zeros(5, h, w)
            tmp_mask[4] = masks[0]

        if view == '3':
            # 左室短轴切面
            tmp_mask = torch.zeros(5, h, w)
            tmp_mask[1] = masks[1]
            tmp_mask[3] = masks[0]

        if view == '4':
            # 心尖四腔心切面
            tmp_mask = torch.zeros(5, h, w)
            tmp_mask[0] = masks[2]
            tmp_mask[1] = masks[3]
            tmp_mask[2] = masks[1]
            tmp_mask[3] = masks[0]
    else:
        _, h, w, l = masks.shape
        if view == '1':
            # 胸骨旁左室长轴切面

            tmp_mask = torch.zeros(5, h, w,l)
            tmp_mask[1] = masks[1]
            tmp_mask[3] = masks[0]

        if view == '2':
            # 肺动脉长轴切面
            tmp_mask = torch.zeros(5, h, w,l)
            tmp_mask[4] = masks[0]

        if view == '3':
            # 左室短轴切面
            tmp_mask = torch.zeros(5, h, w,l)
            tmp_mask[1] = masks[1]
            tmp_mask[3] = masks[0]

        if view == '4':
            # 心尖四腔心切面
            tmp_mask = torch.zeros(5, h, w,l)
            tmp_mask[0] = masks[2]
            tmp_mask[1] = masks[3]
            tmp_mask[2] = masks[1]
            tmp_mask[3] = masks[0]

    return tmp_mask

infos = np.load(f'/home/listu/zyzheng/PAH/test_data/save_infos_seg_contour.npy',allow_pickle=True).item()
infos['0_0']['views_images']['1'] = '/home/listu/zyzheng/PAH/test_data/test_data/normal-23-1_image.nii.gz'
infos['0_0']['views_labels']['1'] = '/home/listu/zyzheng/PAH/test_data/test_data/normal-23-1_label.nii.gz'
new_infos = copy.deepcopy(infos)
clip_length = 40
root = '/home/listu/zyzheng/PAH/test_data/rmyy_144-112'
transform = get_transform(clip_length=clip_length)
organ_num = {
    '1':2,
    '2':1,
    '3':2,
    '4':4
}
for id in list(infos.keys()):
    for view in ['1','2','3','4']:#'1','2','3',
        if infos[id]['views_images'][view] is not None and infos[id]['views_labels'][view] is not None:
            images = np.array(nib.load(infos[id]['views_images'][view]).dataobj)  # 800*600*172
            contours = np.array(nib.load(infos[id]['views_labels'][view]).dataobj)

            if images.shape[-1] > clip_length:
                start = random.randint(0, images.shape[-1] - clip_length)
                end = start + clip_length
                images = images[:, :, start:end]
                contours = contours[..., start:end]
            else:
                expand_idx = [int(i*images.shape[-1]/clip_length) for i in range(clip_length)]
                images = images[:,:,expand_idx]
                contours = contours[...,expand_idx]

            h,w,_ = images.shape
            masks = []
            # plt.imshow(contours[:, :, 0])
            # plt.show()
            all_cls = list(set(list(contours.reshape(-1))))
            all_cls.remove(0)
            print(f'view:{view} class:{all_cls}')
            for i in range(clip_length):
                contour = contours[:,:,i]
                mask = torch.zeros((h,w))
                for cls in range(1,organ_num[view]+1):
                    # print(cls)
                    if cls>len(all_cls):
                        break
                    contour_xy = np.argwhere(contour==all_cls[cls-1])
                    img = np.zeros((h, w, 3), np.uint8)
                    if list(contour_xy)!=[]:
                        cv2.fillPoly(img, [contour_xy], (255, 255, 255))
                        mask_xy = np.argwhere(img[:,:,0]==255)
                        for idx in mask_xy:
                            # print(idx)
                            mask[idx[1],idx[0]] = cls
                mask = torch.tensor(mask).unsqueeze(-1)
                masks.append(mask)
            masks = torch.cat(masks,dim=-1)
            plt.imshow(masks[:,:,0])
            plt.show()
            current_input_dir = transform({'images': images, 'masks': masks})
            plt.imshow(images[:,:,0])
            plt.savefig(os.path.join('/home/listu/zyzheng/PAH/test_data/rmyy/visual',f'{id}_{view}_before_transform.png'))
            plt.close()
            plt.imshow(contours[:, :, 0])
            plt.savefig(os.path.join('/home/listu/zyzheng/PAH/test_data/rmyy/visual', f'{id}_{view}_contour.png'))
            plt.close()
            plt.imshow(current_input_dir['images'][0, :, :, 0])
            plt.savefig(os.path.join('/home/listu/zyzheng/PAH/test_data/rmyy/visual', f'{id}_{view}_after_transform.png'))
            plt.close()
            plt.imshow(current_input_dir['masks'][0,:,:,0])
            plt.savefig(os.path.join('/home/listu/zyzheng/PAH/test_data/rmyy/visual', f'{id}_{view}_masks.png'))
            plt.close()
            if view   == '1':
                LV = torch.where(current_input_dir['masks'] == 1, 1, 0)
                RV = torch.where(current_input_dir['masks'] == 2, 1, 0)
                masks = torch.cat([LV, RV], dim=0)
            elif view == '2':
                PA = torch.where(current_input_dir['masks'] == 1, 1, 0)
                masks = PA
            elif view == '3':
                LV = torch.where(current_input_dir['masks'] == 1, 1, 0)
                RV = torch.where(current_input_dir['masks'] == 2, 1, 0)
                masks = torch.cat([LV, RV], dim=0)
            elif view == '4':
                LV = torch.where(current_input_dir['masks'] == 1, 1, 0)
                LA = torch.where(current_input_dir['masks'] == 2, 1, 0)
                RA = torch.where(current_input_dir['masks'] == 3, 1, 0)
                RV = torch.where(current_input_dir['masks'] == 4, 1, 0)
                masks = torch.cat([LV, LA, RA, RV], dim=0)
            masks = mask_to_allclass(masks, view)
            images = current_input_dir['images']
            # masks = current_input_dir['masks']
            new_images_path = os.path.join(root,infos[id]['views_images'][view].split('/')[-1])
            new_masks_path = os.path.join(root,infos[id]['views_labels'][view].split('/')[-1])
            new_infos[id]['views_images'][view] = new_images_path
            new_infos[id]['views_labels'][view] = new_masks_path
            images = nib.Nifti1Image(np.array(images), np.eye(4))
            masks = nib.Nifti1Image(np.array(masks), np.eye(4))
            nib.save(images,new_images_path)
            nib.save(masks, new_masks_path)
    print(f'{id} finish')


np.save(os.path.join(root,'infos.npy'),new_infos)