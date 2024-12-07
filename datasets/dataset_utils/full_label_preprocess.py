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

from PIL import Image
from torch.utils.data import Dataset


class Seg_PAHDataset(Dataset):
    def __init__(self, root, view_num=['2'], fill_mask=False):
        self.rort = root
        self.view_num = view_num
        self.fill_mask = fill_mask

        self.data_dict = self.get_dict(root)
        self.id_list = self.data_dict['image']
        self.num_data = len(self.id_list)

    def __getitem__(self, index):

        def get_info_dict(index):
            images = self.id_list[index]
            if images is not None:
                image_list = np.array(nib.load(images).dataobj)
                
                if len(image_list.shape) == 3:
                    image_path = images.split('/')
                    image_path_nill = images.split('/')

                    image_path[4] = 'dataset_pa_iltrasound_nill_files_clean_image'
                    image_path_nill[4] = 'dataset_pa_iltrasound_nill_files_clean'
                    
                    image_name =  image_path[-1][:-7]
                    image_dir = '/'.join(image_path[:-1]) + '/' + image_name + '/image/'
                    image_path = '/'.join(image_path)
                    image_path_nill = '/'.join(image_path_nill[:-1]) + '/' + image_name + '.nii.gz'

                    nill_file = nib.load(images)
                    image_list = np.array(nill_file.dataobj)

                    print(image_list.shape)
                    image_list = self.compute_pixel_discrepancy(image_list)
                    #new_imgs = nib.Nifti1Image(image_list, nill_file.affine, nill_file.header)
                    #nib.save(new_imgs, image_path_nill)
                    
                    print(os.path.exists(image_dir))
                    if not os.path.exists(image_dir):
                        os.makedirs(image_dir)
                    w, h, t = image_list.shape
                    for i in range(t):
                        frame = Image.fromarray(np.uint8(image_list[:, :, i]))
                        frame.save(image_dir+str(i).zfill(3)+'.jpg')
        
        get_info_dict(index)
        
        return index

    def __len__(self):
        return self.num_data

    def get_dict(self, root):
        # cases = ['normal-23', 'normal-27', 'patient-5', 'patient-17', 'patient-35', 
        #          'patient-39', 'patient-44', 'patient-60', 'patient-64', 'patient-67']
        all_image_file = []
        all_mask_file = []
        for r, ds, fs in os.walk(root):
            for f in fs:
                if "image" in f:
                    all_image_file.append(r + '/' + f)
                if "mask" in f:
                    all_mask_file.append(r + '/' + f)
        return {'image':all_image_file, 'mask':all_mask_file}
            

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


if __name__ == '__main__':
    data_dict = dict()
    root = '/home/jyangcu/Dataset/dataset_pa_iltrasound_nii_files_3rdcenters/label_all_frame'
    
    from monai.data import DataLoader
    train_ds = Seg_PAHDataset(root, view_num=['2'])
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=1)

    for _ in tqdm(train_loader):
        pass