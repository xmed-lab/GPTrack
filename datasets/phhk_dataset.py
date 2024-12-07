#jyangcu@connect.ust.hk
import os
import random
import numpy as np
import pandas as pd

from tqdm import tqdm
from PIL import Image, ImageFilter

from torch.utils.data import Dataset
import torchvision.transforms as transforms

import datasets.dataset_utils.transform as vid_transform

np.set_printoptions(threshold=np.inf)
random.seed(8888)
np.random.seed(8888)


class RandomMaskingGenerator:
    def __init__(self, input_size, mask_size, mask_ratio, train_mode):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        if not isinstance(mask_size, tuple):
            mask_size = (mask_size,) * 2

        self.height, self.width, _ = input_size
        self.length = 1
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
        if self.train_mode == 'train':
            select_num_mask = int(self.num_mask * (np.random.randint(20, 80) / 100))
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


class PHHKDataset(Dataset):
    def __init__(self, args, mode='train', select_set=['GY'], view_num=['3'], is_video=True):
        self.args = args
        self.mode = mode
        self.select_set = select_set
        self.view_num = view_num
        self.is_video = is_video

        data_all = self.mapping_data()
        self.selected_data, splited_set, _ = self.split_data(data_all)

        if mode in ['train', 'valid', 'test']:
            self.data_list = splited_set[mode]
        else:
            raise RuntimeError('The mode should be one of the "trian", "valid" or "test".')

        if mode == 'train':
            self.transform = transforms.Compose([
                                vid_transform.ToTensorVideo(),
                                vid_transform.ResizedVideo((620, 460)),
                                vid_transform.CenterCropVideo((460, 460)),
                                # vid_transform.ColorJitterVideo(bri_con_sat=[0.5,0.5,0.5], hue=0.5),
                                vid_transform.ResizedVideo((int(args.image_size[0] * 1.3), int(args.image_size[1] * 1.3))),
                                vid_transform.RandomCropVideo((args.image_size[0], args.image_size[1])),
                                vid_transform.RandomHorizontalFlipVideo(),
                                ])
        else:
            self.transform = transforms.Compose([
                                vid_transform.ToTensorVideo(),
                                vid_transform.ResizedVideo((620, 460)),
                                vid_transform.CenterCropVideo((460, 460)),
                                vid_transform.ResizedVideo((int(args.image_size[0] * 1.3), int(args.image_size[1] * 1.3))),
                                vid_transform.CenterCropVideo((args.image_size[0], args.image_size[1])),
                                ])
            
        self.masked_position_generator = RandomMaskingGenerator(input_size=args.image_size, mask_size=args.mask_size, mask_ratio=args.mask_ratio, train_mode=mode)

    def __getitem__(self, index):
        if self.mode == 'train':
            index = index // 1
        def get_case_attr(index):
            case_dir = self.data_list[index]
            case_attr = self.selected_data[case_dir]
            return case_dir, case_attr

        view_frame_list = list()
        if self.is_video:
            case_dir, case_attr = get_case_attr(index)
            while len(case_attr[self.view_num[0]]) < self.args.image_size[-1]:
                index = random.randint(0, len(self.data_list)-1)
                case_dir, case_attr = get_case_attr(index)

            acquired_video = dict()
            for selected_view in self.view_num:
                view_frame_list = case_attr[selected_view]
                view_frame_list.sort()
                
                video_length = len(view_frame_list)
                if self.mode == 'train':
                    if self.args.max_sample_rate is not None:
                        max_sample_rate = self.args.max_sample_rate
                        if max_sample_rate > video_length // self.args.image_size[-1]:
                            max_sample_rate = video_length // self.args.image_size[-1]
                    else:
                        max_sample_rate = video_length // self.args.image_size[-1]

                    if self.args.min_sample_rate is not None:
                        if self.args.min_sample_rate < max_sample_rate:
                            min_sample_rate = self.args.min_sample_rate
                    else:
                        min_sample_rate = max_sample_rate // 2

                    if max_sample_rate >= 8:
                        sample_rate = random.randint(min_sample_rate, 8)
                    elif max_sample_rate > 4 and max_sample_rate <= 8:
                        sample_rate = random.randint(min_sample_rate, max_sample_rate)
                    elif max_sample_rate > 2 and max_sample_rate <= 4:
                        sample_rate = random.randint(min_sample_rate, max_sample_rate)
                    elif max_sample_rate <= 2 and max_sample_rate > 1:
                        sample_rate = random.randint(1, max_sample_rate)
                    else:
                        sample_rate = 1
                elif self.mode != 'train':
                    if video_length // self.args.image_size[-1] > 2:
                        max_sample_rate = 2
                        sample_rate = random.randint(1, max_sample_rate)
                    else:
                        sample_rate = 1
                else:
                    sample_rate = 1

                start_idx = random.randint(0, (video_length-sample_rate * self.args.image_size[-1]))
                view_frame_list = view_frame_list[start_idx : start_idx + sample_rate * self.args.image_size[-1] : sample_rate]

                video = list()
                for frame_id in view_frame_list:
                    frame = Image.open(os.path.join(case_dir, selected_view, frame_id)).convert('L')
                    if self.args.blurring:
                        frame = frame.filter(ImageFilter.GaussianBlur(radius = random.uniform(0.1, 2.0)))
                    frame = np.array(frame)
                    video.append(np.expand_dims(frame, axis=-1))
                bboxs = self.mask_find_bboxs(np.where(video[0] != 0))
                video = np.stack(video, axis=0)
                video = video[:, bboxs[0]:bboxs[1], bboxs[2]+15:bboxs[3]]

                video = self.transform(video)
                acquired_video[selected_view] = video

        mpap = float(case_attr['mpap'])
        pvr = float(case_attr['pvr'])
        cls_reg = case_attr['class_reg']
        if mpap == 0.0:
            mpap = random.randint(1, 15) 
            cls_reg = cls_reg / 15

        if pvr == 0.0:
            pvr = random.randint(1, 25) / 10 

        return acquired_video, mpap, pvr, int(case_attr['class']), cls_reg, self.masked_position_generator()

    def __len__(self):
        if self.mode == 'train':
            return len(self.data_list) * 1
        else:
            return len(self.data_list)
    
    def mapping_data(self):
        data_dict = dict()
        # Mapping All the Data According to the Hospital Center
        for dir in os.listdir(self.args.dataset_path):
            dir_path = os.path.join(self.args.dataset_path, dir)
            if os.path.isdir(dir_path):
                data_dict[dir] = {}
                # Mapping All the Data for a Hospital Center According to the Device
                for sub_dir in os.listdir(dir_path):
                    sub_dir_path = os.path.join(dir_path, sub_dir)
                    if os.path.isdir(sub_dir_path):
                        data_dict[dir][sub_dir] = {}
                        annotation_file = sub_dir_path + '.xlsx'
                        annos = self.aquire_annotation(os.path.join(self.args.dataset_path, sub_dir, annotation_file))
                        for idx in annos:
                            mpap = annos[idx]['mpap']
                            pvr = annos[idx]['pvr']
                            if mpap <= 20:
                                data_dict[dir][sub_dir][annos[idx]['id']] = {'mpap':mpap, 'pvr':pvr, 'class':0, 'class_reg':mpap/20}
                            elif 20 < mpap <= 35:
                                data_dict[dir][sub_dir][annos[idx]['id']] = {'mpap':mpap, 'pvr':pvr, 'class':1, 'class_reg':((mpap-20)/15)+1}
                            elif 35 < mpap <= 50:
                                data_dict[dir][sub_dir][annos[idx]['id']] = {'mpap':mpap, 'pvr':pvr, 'class':2, 'class_reg':((mpap-35)/15)+2}
                            elif 50 < mpap:
                                data_dict[dir][sub_dir][annos[idx]['id']] = {'mpap':mpap, 'pvr':pvr, 'class':3, 'class_reg':((mpap-50)/15)+3}
                            
                            case_path = os.path.join(sub_dir_path, annos[idx]['id'])
                            for selected_view in self.view_num:
                                case_view_path =  os.path.join(case_path, selected_view)
                                for _, _, frame_list in os.walk(case_view_path):
                                    data_dict[dir][sub_dir][annos[idx]['id']].update({selected_view: frame_list})

        return data_dict

    def aquire_annotation(self, file):
        annos = pd.read_excel(file, sheet_name = 0)
        annos = annos.rename(str.lower, axis='columns')
        return annos.to_dict('index')

    def split_data(self, data):
        normal_cases = dict()
        mild_cases = dict()
        moderate_cases = dict()
        severe_cases = dict()
        selected_data = dict()
        all_dict = dict()
        normal_value, mild_value, moderate_value, severe_value = list(), list(), list(), list()
        # This is for Selecting the Medical Center and Device types;
        # Currently, only Medical Center selection is avaliable.
        for set_name in self.select_set:
            set_data = data[set_name]
            set_path = os.path.join(self.args.dataset_path, set_name)
            selected_data[set_path] = dict()
            for subdir, v in set_data.items():
                selected_data[set_path][subdir] = dict()
                selected_data[set_path][subdir].update(v)
                for case, attrs in v.items():
                    all_dict.update({os.path.join(set_path, subdir, case):attrs})
                    if attrs['mpap'] <= 20:
                        normal_cases.update({os.path.join(set_path, subdir, case):attrs})
                        normal_value.append(attrs['mpap'])
                    elif 20 < attrs['mpap'] <= 35:
                        mild_cases.update({os.path.join(set_path, subdir, case):attrs})
                        mild_value.append(attrs['mpap'])
                    elif 35 < attrs['mpap'] <= 50:
                        moderate_cases.update({os.path.join(set_path, subdir, case):attrs})
                        moderate_value.append(attrs['mpap'])
                    elif 50 < attrs['mpap']:
                        severe_cases.update({os.path.join(set_path, subdir, case):attrs})
                        severe_value.append(attrs['mpap'])

        # Split Date to Train, Test and Validation Dataset.
        train_list = list()
        valid_list = list()
        test_list  = list()
        
        selected_cases = list(normal_cases.keys()) + list(mild_cases.keys()) + list(moderate_cases.keys()) + list(severe_cases.keys())
        selected_cases.sort()

        train_list = random.sample(selected_cases, int(len(selected_cases) * 0.8))
        remain_list = list(set(selected_cases).difference(set(train_list)))
        valid_list = random.sample(remain_list, int(len(remain_list) * 0.5))
        test_list = list(set(remain_list).difference(set(valid_list)))

        # print(len(selected_cases))
        # print(len(train_list))
        # print(len(test_list))

        all_cases_dict = {'Normal':normal_cases, 'Mild':mild_cases, 'Moderate':moderate_cases, 'Severe':severe_cases}

        num_normal_cases = len(list(normal_cases.keys()))
        num_mild_cases = len(list(mild_cases.keys()))
        num_moderate_cases = len(list(moderate_cases.keys()))
        num_severe_cases = len(list(severe_cases.keys()))

        if self.mode == 'train':
            print("Total Normal Cases   --->>> {a:.0f} -  | mean = {b:.4f} | var = {c:.4f} | std = {d:.4f} | max = {e:.4f} | min = {f:.4f} |".\
                format(a=num_normal_cases, b=np.mean(normal_value), c=np.var(normal_value), d=np.std(normal_value), e=max(normal_value), f=min(normal_value)))
            print("Total Mild Cases     --->>> {a:.0f} - | mean = {b:.4f} | var = {c:.4f} | std = {d:.4f} | max = {e:.4f} | min = {f:.4f} |".\
                format(a=num_mild_cases, b=np.mean(mild_value), c=np.var(mild_value), d=np.std(mild_value), e=max(mild_value), f=min(mild_value)))
            print("Total Moderate Cases --->>> {a:.0f} - | mean = {b:.4f} | var = {c:.4f} | std = {d:.4f} | max = {e:.4f} | min = {f:.4f} |".\
                format(a=num_moderate_cases, b=np.mean(moderate_value), c=np.var(moderate_value), d=np.std(moderate_value), e=max(moderate_value), f=min(moderate_value)))
            print("Total Severe Cases   --->>> {a:.0f} - | mean = {b:.4f} | var = {c:.4f} | std = {d:.4f} | max = {e:.4f} | min = {f:.4f} |".\
                format(a=num_severe_cases, b=np.mean(severe_value), c=np.var(severe_value), d=np.std(severe_value), e=max(severe_value), f=min(severe_value)))

        return all_dict, {'train':train_list, 'valid':valid_list, 'test':test_list, 'all':selected_cases}, all_cases_dict

    def mask_find_bboxs(self, mask):
        bbox = np.min(mask[0]), np.max(mask[0]), np.min(mask[1]), np.max(mask[1])
        return bbox


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--image-size', type=int, default=(112,112,16), help='Image height and width (default: 256)')
    parser.add_argument('--crop_length', type=int, default=16, help='Video length (default: 256)')
    parser.add_argument('--max_sample_rate', type=int, default=8, help='The sampling rate for the video')
    parser.add_argument('--blurring', type=bool, default=True, help='Whether blur the image')
    parser.add_argument('--mask-size', type=int, default=8, help='The size of mask patch (default: 16)')
    parser.add_argument('--mask-ratio', type=float, default=0.6, help='The ratio of masking area in an image (default: 0.75)')
    args = parser.parse_args()
    args.dataset_path = r'/home/jyangcu/Dataset/PH_HK_image'

    data_dict = dict()

    from monai.data import DataLoader
    from torchvision import utils as vutils
    train_ds = PHHKDataset(args, select_set=['PADN-HK','GDPH','GY','SZ','SH'], view_num=['4'])
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=1)
    from einops import rearrange
    for img, mpap, pasp, _ in tqdm(train_loader):
        pass