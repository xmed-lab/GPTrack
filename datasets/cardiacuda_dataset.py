#jyangcu@connect.ust.hk
import os
import random
import numpy as np

import cv2
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image, ImageFilter

from torch.utils.data import Dataset
import datasets.dataset_utils.transform as transform

np.set_printoptions(threshold=np.inf)
random.seed(7777)
np.random.seed(7777)


class CardiacUDA_Dataset(Dataset):
    def __init__(self, args, is_train=True, is_test=False, set_select=['gy'], view_num=['3'], is_video=True):
        self.args = args
        self.root = args.dataset_path
        self.set_select = set_select
        self.view_num = view_num
        self.is_train = is_train
        self.is_video = is_video
        self.is_test  = is_test
        self.crop_length = args.image_size[2]

        if is_train:
            self.transform = transforms.Compose([
                                transform.ToTensorVideo(),
                                transform.CenterCropVideo((460, 460)),
                                transform.ResizedVideo((360, 360)),
                                transform.RandomCropVideo((256, 256)),
                                transform.RandomHorizontalFlipVideo(p=0.5),
                                transform.Rotate(p=0.5),
                                # transform.Color_jitter(p=0.5),
                                # transform.Equalize(p=0.3),
                                ])
        else:
            self.transform = transforms.Compose([
                                transform.ToTensorVideo(),
                                transform.CenterCropVideo((460, 460)),
                                transform.ResizedVideo((360, 360)),
                                transform.CenterCropVideo((256, 256)),
                                ])

        self.all_cases = self.get_dict()
        self.data_list = list(self.all_cases.keys())
        
        self.train_list = self.all_data[:int(self.num_data * 0.9)]
        self.valid_list = list(set(self.all_data).difference(set(self.train_list)))
        self.data_list = self.train_list if is_train else self.valid_list
        self.data_list.sort()

    def __getitem__(self, index):
        def get_frame_list(index):
            video_name = self.data_list[index]
            frame_list = self.all_cases[video_name]
            return frame_list

        index = index

        if self.is_video:
            frame_list = list()
            while len(frame_list) < self.crop_length:
                frame_list = get_frame_list(index)
                if len(frame_list) < self.crop_length:
                    index = random.randint(0, len(self.data_list)-1)
                    index = index if self.is_train else index
                new_index = index
            
            frame_list.sort()
            video_length = len(frame_list)
            if self.is_train:
                max_sample_rate = video_length // self.crop_length
                if max_sample_rate > self.args.max_sample_rate:
                    max_sample_rate = self.args.max_sample_rate

                if max_sample_rate > 4:
                    sample_rate = random.randint(2, 4)
                elif 2 < max_sample_rate <= 4:
                    sample_rate = random.randint(2, max_sample_rate)
                elif max_sample_rate <= 2 and max_sample_rate >= 1:
                    sample_rate = random.randint(1, max_sample_rate)
                else:
                    sample_rate = 1
            else:
                sample_rate = 1
            start_idx = random.randint(0, (video_length-sample_rate*self.crop_length))

            frame_list = frame_list[start_idx:start_idx+sample_rate*self.crop_length:sample_rate]

            video_name = self.data_list[new_index]
            video = list()
            transformed_video = list()
            radius_ratio = random.uniform(0.1, 2.0)
            for frame in frame_list:
                frame = Image.open(os.path.join(video_name+'/'+frame)).convert('L')
                frame = frame.resize((620, 460))
                frame_weak = frame.filter(ImageFilter.GaussianBlur(radius = radius_ratio))
                video.append(np.array(frame_weak))

                frame_strong = frame.filter(ImageFilter.GaussianBlur(radius = 2))
                frame_strong = frame_strong.filter(ImageFilter.SMOOTH_MORE)
                frame_strong = np.array(frame_strong)
                frame_strong = Image.fromarray(np.where(frame_strong>frame_strong.mean() * 3.0, frame_strong, 0.0)).convert('L')
                # frame_strong = frame_strong.filter(ImageFilter.GaussianBlur(radius = 2))
                frame_strong = np.array(frame_strong)
                
                # area = []
                # enhanced_mask = np.zeros(np.array(frame).shape, np.uint8)
                # contours, _  = cv2.findContours(frame_strong.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                # for k in range(len(contours)):
                #     area.append(cv2.contourArea(contours[k]))
                # mean_area = np.array(area).mean()
                # max_area = np.array(area).max()
                # if len(contours) > 0:
                #     for contour in contours:
                #         if cv2.contourArea(contour) < 500:
                #             cv2.drawContours(frame_strong, [contour], 0, 0, -1)
                transformed_video.append(frame_strong)

            video = np.stack(video, axis=-1)
            video = np.expand_dims(video, axis=-1)
            video = np.transpose(video, (2, 0, 1, 3))
            video = self.transform(video)

            transformed_video = np.stack(transformed_video, axis=-1)
            transformed_video = np.expand_dims(transformed_video, axis=-1)
            transformed_video = np.transpose(transformed_video, (2, 0, 1, 3))
            transformed_video = self.transform(transformed_video)

        return video.transpose(0,1) / 255.0, transformed_video.transpose(0,1) / 255.0

    def __len__(self):
        return len(self.data_list)

    def mask_find_bboxs(self, mask):
        bbox = np.min(mask[0]), np.max(mask[0]), np.min(mask[1]), np.max(mask[1])
        return bbox

    def get_dict(self):
        selected_cases = {}
        files = os.listdir(self.root)
        for file in files:
            if file == 'label_all_frame':
                pass
            else:
                data_path = self.root + '/' + file
                cases = os.listdir(data_path)
            for case in cases:
                if '-4_' in case:
                    images = os.listdir(data_path + '/' + case + '/image')
                    images.sort()
                    if len(images) > 10:
                        selected_cases.update({data_path + '/' + case + '/image':images})
                    else:
                        continue
        return selected_cases
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--image-size', type=int, default=(112,112,16), help='Image height and width (default: 256)')
    parser.add_argument('--mask-size', type=int, default=8, help='The size of mask patch (default: 16)')
    parser.add_argument('--mask-ratio', type=float, default=0.6, help='The ratio of masking area in an image (default: 0.75)')
    parser.add_argument('--max_sample_rate', type=int, default=1, help='The sampling rate for the video')
    parser.add_argument('--blurring', type=bool, default=True, help='Whether blur the image')
    args = parser.parse_args()
    args.dataset_path = r'/home/jyangcu/Dataset/dataset_pa_iltrasound_nill_files_clean_image'

    data_dict = dict()
    from monai.data import DataLoader
    from torchvision import utils as vutils
    train_ds = CardiacUDA_Dataset(args, set_select=['gy','szfw','rmyy','shph'], view_num=['1','2','3','4'])
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=1)
    from einops import rearrange
    for img in tqdm(train_loader):
        print(img.shape)
