#jyangcu@connect.ust.hk
import os
import random
import numpy as np

from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image, ImageFilter

import nibabel as nib

import torch
from torch.utils.data import Dataset
import datasets.dataset_utils.transform as transform

from monai.transforms import (
    AddChanneld,
    Compose,
    LoadImaged,
    RandSpatialCropd,
    CenterSpatialCropd,
    EnsureChannelFirstd,
    RandFlipd,
    Resized,
    ScaleIntensityRangePercentilesd,
    NormalizeIntensityd,
    Identity,
    EnsureTyped
)

np.set_printoptions(threshold=np.inf)
random.seed(7777)
np.random.seed(7777)

problem_data = [
                '/home/jyangcu/Dataset/dataset_pa_iltrasound_nill_files_clean_image/dataset_pa_nii_rmyy_center_20210903_size_126/patient-73-4_image/image/',
               ]

class RandomMaskingGenerator:
    def __init__(self, input_size, mask_size, mask_ratio, train_mode):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        if not isinstance(mask_size, tuple):
            mask_size = (mask_size,) * 2

        self.height, self.width, self.length = input_size
        self.mask_h_size, self.mask_w_size = mask_size
        self.num_patches = (self.height//self.mask_h_size) * (self.width//self.mask_w_size) * self.length
        self.empty_image = np.ones((self.num_patches, self.mask_h_size, self.mask_w_size))
        self.num_mask = int(mask_ratio * self.num_patches)
        self.train_mode = train_mode

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def __call__(self):
        if self.train_mode:
            select_num_mask = int(self.num_mask * (np.random.randint(50, 80) / 100))
        else:
            select_num_mask = int(self.num_mask * 0.75)

        mask = np.hstack([
            np.ones(self.num_patches - select_num_mask),
            np.zeros(select_num_mask),
        ])
        np.random.shuffle(mask)
        mask = np.expand_dims(mask, (1,2))
        mask = np.repeat(mask, self.mask_h_size, axis=(1))
        mask = np.repeat(mask, self.mask_w_size, axis=(2))
        mask = self.empty_image * mask
        return mask # [196]


class Seg_PAHDataset(Dataset):
    def __init__(self, args, infos, data_type=['normal'], is_train=False, is_test=False, set_select=['gy','rmyy','szfw'], view_num=['4'], is_video=True, single_frame=True, seg_parts=True):
        self.args = args
        self.root = args.dataset_path
        self.set_select = set_select
        self.view_num = view_num
        self.is_video = is_video
        self.single_frame = single_frame
        self.seg_parts = seg_parts
        self.crop_length = args.image_size[2] if not single_frame else 1 

        self.is_train = is_train
        self.is_test  = is_test

        self.transform = self.get_transform()
    
        # self.masked_position_generator = RandomMaskingGenerator(input_size=args.image_size, mask_size=args.mask_size, mask_ratio=args.mask_ratio, train_mode=is_train)

        self.all_image, self.all_video = self.get_dict(infos, data_type)
        if self.is_video:
            self.all_data = list(self.all_video['all'].keys())
        else:
            self.all_data = list(self.all_image['all'].keys())

        self.num_data = len(self.all_data)

        self.train_list = self.all_data[:int(self.num_data * 0.9)]
        self.valid_list = list(set(self.all_data).difference(set(self.train_list)))

        self.data_list = self.train_list if is_train else self.valid_list
        self.data_list.sort()

    def __getitem__(self, index):
        def get_frame_list(index):
            video_name = self.data_list[index]
            frame_list, mpap, pasp = self.all_video['all'][video_name]['images'], self.all_video['all'][video_name]['mPAP'], self.all_video['all'][video_name]['pasp']
            
            name, patient = video_name.split('/')[5:7]
            mask_name = self.args.mask_path + '/' + os.path.join(name, patient[:-5]+'label.nii.gz')
            masks = np.array(nib.load(mask_name).dataobj)

            mask_frames_ = np.sum(masks, axis=(0,1))
            mask_frames_ = np.where(mask_frames_ > 1, 1, 0)
            mask_frames_ = np.argwhere(mask_frames_ == 1)

            return frame_list, mpap, pasp, masks, mask_frames_, video_name

        index = index // 10 if self.is_train else index
        
        while self.data_list[index] in problem_data:
            index = random.randint(0, len(self.data_list)-1)
        
        frame_list, mpap, pasp, masks, mask_frames_, video_name = get_frame_list(index)
        while mask_frames_.shape[0] == 0:
            index = random.randint(0, len(self.data_list)-1)
            frame_list, mpap, pasp, masks, mask_frames_, video_name = get_frame_list(index)

        if self.single_frame:

            if len(masks.shape) == 3:
                index = random.choice(mask_frames_)[0]
                frame = Image.open(os.path.join(video_name+frame_list[index])).convert('L')
                masks = masks[:, :, index]
            
            video = np.expand_dims(frame, axis=-1)
            masks = np.expand_dims(masks, axis=-1)
            
        elif self.is_video:
            while len(frame_list) < self.crop_length:
                frame_list, mpap, pasp = get_frame_list(index)
                if len(frame_list) < self.crop_length:
                    index = random.randint(0, len(self.data_list)-1)
                    index = index if self.is_train else index
                new_index = index
                video_name = self.data_list[new_index]
            
            frame_list.sort()

            if len(masks.shape) == 3:
                index = random.choice(mask_frames_)[0]

                if masks.shape[-1] == 3:
                    frame = Image.open(os.path.join(video_name+frame_list[1])).convert('L')
                    video = np.tile(frame,(1,1,self.crop_length))
                    masks  = np.tile(masks[:, :, 1:2], (1,1,self.crop_length))

                else:     
                    r_index = random.randint(0, index if index < self.crop_length-1 else self.crop_length-1)
                    start = index - r_index
                    end = start + self.crop_length
                    frame_list = frame_list[start:end]
                    
                    video = list()
                    for frame in frame_list:
                        frame = Image.open(os.path.join(video_name+frame)).convert('L')
                        video.append(np.array(frame))
                    
                    video = np.stack(video, axis=0)
                    masks  = masks[:, :, start:end]
                    index = r_index

        out = self.transform({'images':video, 'masks':masks})

        trans_video = out['images']
        trans_masks = out['masks']

        if self.seg_parts:
            if self.view_num   == ['1']:
                BG = torch.where(trans_masks == 0, 1, 0)
                LV = torch.where(trans_masks == 1, 1, 0)
                RV = torch.where(trans_masks == 2, 1, 0)
                trans_masks = torch.cat([BG, LV, RV], dim=0)
            elif self.view_num == ['2']:
                BG = torch.where(trans_masks == 0, 1, 0)
                PA = torch.where(trans_masks == 1, 1, 0)
                trans_masks = torch.cat([BG, PA], dim=0)
            elif self.view_num == ['3']:
                BG = torch.where(trans_masks == 0, 1, 0)
                LV = torch.where(trans_masks == 1, 1, 0)
                RV = torch.where(trans_masks == 2, 1, 0)
                trans_masks = torch.cat([BG, LV, RV], dim=0)
            elif self.view_num == ['4']:
                BG = torch.where(trans_masks == 0, 1, 0)
                LV = torch.where(trans_masks == 1, 1, 0)
                LA = torch.where(trans_masks == 2, 1, 0)
                RA = torch.where(trans_masks == 3, 1, 0)
                RV = torch.where(trans_masks == 4, 1, 0)
                trans_masks = torch.cat([BG, LV, LA, RA, RV], dim=0)
        else:
            trans_masks = torch.where(trans_masks > 0, 1, 0)

        return trans_video / 127.5 - 1, trans_masks, float(mpap), float(pasp)

    def __len__(self):
        return len(self.data_list) * 10 if self.is_train else len(self.data_list)

    def mask_find_bboxs(self, mask):
        bbox = np.min(mask[0]), np.max(mask[0]), np.min(mask[1]), np.max(mask[1])
        return bbox

    def get_dict(self, infos, data_type):
        
        def is_number(s):
            if isinstance(s, str):
                return False

            if s == 'nan':
                return False
            
            if np.isnan(s):
                return False
            
            try:
                float(s)
                return True
            except ValueError:
                pass
        
            try:
                import unicodedata
                unicodedata.numeric(s)
                return True
            except (TypeError, ValueError):
                pass
        
            return False
        
        def detech_none_digit_value(value):
            if is_number(value):
                return value
            else:
                return 0

        selected_dict = dict()
        all_images = {'normal':{}, 'middle':{}, 'slight':{}, 'severe':{}, 'ASD':{}, 'None_ASD': {}, 'ASD-severe': {}}
        all_videos = {'normal':{}, 'middle':{}, 'slight':{}, 'severe':{}, 'ASD':{}, 'None_ASD': {}, 'ASD-severe': {}}
        for k, v in infos.items():
            if v['dataset_name'] in self.set_select:
                selected_dict[k] = {}
                selected_dict[k]['images'] = v['views_images']
                selected_dict[k]['masks']  = v['views_labels']
                selected_dict[k]['fold'] = v['fold']
                selected_dict[k]['dataset_name'] = v['dataset_name']
        
                for id in list(selected_dict.keys()):
                    view_path = selected_dict[id]['images']

                    for k in self.view_num:
                        if k in selected_dict[id]['masks'].keys():
                            if selected_dict[id]['masks'][k] is None:
                                pass
                            else:
                                if k in view_path.keys():
                                    if view_path[k] is None:
                                        image_path = ''
                                    else:
                                        image_fold, image_name = view_path[k].split('/')[-2:]
                                        image_name = image_name[:-7]
                                        image_path = self.root + '/' + image_fold + '/' + image_name + '/image/'
                            
                                image_dict = dict()
                                video_dict = dict()
                                if os.path.exists(image_path):
                                    for _, _, images in os.walk(image_path):
                                        video_dict[image_path] = {'images':images, 'mPAP':0, 'pasp':0, 'ASD': 0}
                                        for i in images:
                                            image_dict[image_path+i]={'mPAP':0, 'pasp':0}
                                    all_images['normal'].update(image_dict)
                                    all_videos['normal'].update(video_dict)
        
        image_list = {'all':{}}
        video_list = {'all':{}}
        for selected_type in data_type:
            image_list['all'].update(all_images[selected_type])
            video_list['all'].update(all_videos[selected_type])
        return image_list, video_list

    def get_transform(self):
        all_keys = ['images', 'masks']
        crop_size = (128, 128, self.crop_length)
        resize_size = (144, 144, self.crop_length)
        first_resize_spatial_size = (620, 460, self.crop_length)
        first_crop_spatial_size = (460, 460, self.crop_length)

        if self.is_train:
            
            rf0 = RandFlipd(keys=all_keys, prob=0.5, spatial_axis=0)
            rf1 = RandFlipd(keys=all_keys, prob=0.5, spatial_axis=1)
            rf2 = RandFlipd(keys=all_keys, prob=0.5, spatial_axis=2)

            rf0.set_random_state(0)
            rf1.set_random_state(0)
            if rf2 is not None:
                rf2.set_random_state(0)

            transform = Compose([
                    AddChanneld(keys=all_keys, allow_missing_keys=True),
                    Resized(keys=all_keys, spatial_size=first_resize_spatial_size, allow_missing_keys=True),
                    CenterSpatialCropd(keys=all_keys, roi_size=first_crop_spatial_size, allow_missing_keys=True),
                    Resized(keys=all_keys, spatial_size=resize_size, allow_missing_keys=True),
                    RandSpatialCropd(keys=all_keys, roi_size=crop_size, random_size=False, allow_missing_keys=True),
                    EnsureTyped(keys=all_keys, allow_missing_keys=True),
                ])
        else:
            transform = Compose([
                    AddChanneld(keys=all_keys, allow_missing_keys=True),
                    Resized(keys=all_keys, spatial_size=first_resize_spatial_size, allow_missing_keys=True),
                    CenterSpatialCropd(keys=all_keys, roi_size=first_crop_spatial_size, allow_missing_keys=True),
                    Resized(keys=all_keys, spatial_size=resize_size, allow_missing_keys=True),
                    CenterSpatialCropd(keys=all_keys, roi_size=crop_size, allow_missing_keys=True),
                    EnsureTyped(keys=all_keys, allow_missing_keys=True),
                ])
        return transform

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--image-size', type=int, default=(112,112,16), help='Image height and width (default: 256)')
    parser.add_argument('--mask-size', type=int, default=8, help='The size of mask patch (default: 16)')
    parser.add_argument('--mask-ratio', type=float, default=0.6, help='The ratio of masking area in an image (default: 0.75)')
    args = parser.parse_args()
    args.dataset_path = r'/home/jyangcu/Dataset/dataset_pa_iltrasound_nill_files_clean_image'
    args.mask_path = r'/home/jyangcu/Dataset/dataset_pa_iltrasound_nii_files_3rdcenters'


    data_dict = dict()
    infos = np.load(f'/home/jyangcu/Dataset/dataset_pa_iltrasound_nii_files_3rdcenters/save_infos_reg_v3.npy', allow_pickle=True).item()
    
    from monai.data import DataLoader
    from torchvision import utils as vutils
    train_ds = Seg_PAHDataset(args, infos, set_select=['gy','rmyy','szfw'], view_num=['4'])
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=1)
    from einops import rearrange
    for img, mask, mpap, pasp in tqdm(train_loader):
        print(img.shape)
        # masked_pos = rearrange(mask, 'b (l h w) p1 p2 -> b l (h p1) (w p2)', 
        #                               h=args.image_size[0]//args.mask_size, w=args.image_size[1]//args.mask_size, 
        #                               l=args.image_size[2], p1=args.mask_size, p2=args.mask_size).unsqueeze(1)
        # imgs = img.add(1.0).mul(0.5) * masked_pos.float()
        vutils.save_image(img[0,:,:,:,0].float(), "data_sample.jpg", nrow=4)