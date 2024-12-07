#jyangcu@connect.ust.hk
import os
import random
import numpy as np
import pandas as pd

from tqdm import tqdm
from torch.utils.data import Dataset

import pydicom
from pydicom.pixel_data_handlers.util import convert_color_space
from PIL import Image

import nibabel as nib

np.set_printoptions(threshold=np.inf)
random.seed(7777)
np.random.seed(7777)


class PHHKDataset(Dataset):
    def __init__(self, args, infos, mode='all', select_set=['GY'], select_view=['3'], is_video=True):
        self.args = args
        self.mode = mode
        self.select_set = select_set
        self.select_view = select_view
        self.is_video = is_video

        data_all = self.mapping_data()
        self.all_cases, splited_set = self.split_data(data_all)

        if mode in ['train', 'valid', 'test', 'all']:
            self.data_list = splited_set[mode]
        else:
            raise RuntimeError('The mode should be one of the "trian", "valid" or "test".')

    def __getitem__(self, index):
        case_path = self.data_list[index]
        case_attr = self.all_cases[case_path]
        frames = self.acquire_dicom(case_path, self.select_view)
        return frames, case_attr['mpap'], case_attr['pvr']

    def __len__(self):
        return len(self.data_list)
    
    def mapping_data(self):
        data_dict = dict()
        # Mapping All the Data According to the Hospital Center
        for dir in os.listdir(args.dataset_path):
            dir_path = os.path.join(args.dataset_path, dir)
            if os.path.isdir(dir_path):
                data_dict[dir] = {}
                # Mapping All the Data for a Hospital Center According to the Device
                for sub_dir in os.listdir(dir_path):
                    sub_dir_path = os.path.join(dir_path, sub_dir)
                    if os.path.isdir(sub_dir_path):
                        data_dict[dir][sub_dir] = {}
                        annotation_file = sub_dir_path + '.xlsx'
                        annos = self.aquire_annotation(os.path.join(args.dataset_path, sub_dir, annotation_file))
                        for idx in annos:
                            data_dict[dir][sub_dir][sub_dir+'/'+annos[idx]['id']] = {'mpap':annos[idx]['mpap'], 'pvr':annos[idx]['pvr']}

        return data_dict

    def aquire_annotation(self, file):
        annos = pd.read_excel(file, sheet_name = 0)
        annos = annos.rename(str.lower, axis='columns')
        return annos.to_dict('index')

    def split_data(self, data):
        selected_data = dict()
        # This is for Selecting the Medical Center and Device types;
        # Currently, only Medical Center selection is avaliable.
        for set_name in self.select_set:
            set_data = data[set_name]
            set_path = os.path.join(args.dataset_path, set_name)
            selected_data[set_path] = dict()
            for k, v in set_data.items():
                selected_data[set_path].update(v)
    
        # Split Date to Train, Test and Validation Dataset.
        all_dict = dict()
        train_list = list()
        valid_list = list()
        test_list  = list()
        for k, _ in selected_data.items():
            for name, attr in selected_data[k].items():
                case_path = os.path.join(k, name)
                all_dict[case_path] = attr
        all_cases = list(all_dict.keys())
        all_cases.sort()

        train_list = random.sample(all_cases, int(len(all_cases) * 0.8))
        remain_list = list(set(all_cases).difference(set(train_list)))
        valid_list = random.sample(remain_list, int(len(remain_list) * 0.5))
        test_list = list(set(remain_list).difference(set(valid_list)))

        return all_dict, {'train':train_list, 'valid':valid_list, 'test':test_list, 'all':all_cases}

    def acquire_dicom(self, path, select_view):
        frames_dict = dict()
        for view in select_view:
            new_image_dir = args.new_dataset_path + '/' + '/'.join(path.split('/')[4:]) + '/' + view
            print(new_image_dir)
            if not os.path.exists(new_image_dir):
                os.makedirs(new_image_dir)
                if os.path.exists(os.path.join(path, view+'.dcm')):
                    frames_dict[view] = pydicom.dcmread(os.path.join(path,view+'.dcm')).pixel_array
                elif os.path.exists(os.path.join(path, view)):
                    frames_dict[view] = pydicom.dcmread(os.path.join(path,view)).pixel_array
                elif os.path.exists(os.path.join(path, view + '.nii.gz')):
                    frames_dict[view] = np.flip(np.transpose(np.array(nib.load(os.path.join(path, view + '.nii.gz')).dataobj), axes=(2,1,0)),axis=(1,2))
                else:
                    continue
                if len(frames_dict[view].shape) == 4:
                    frames_dict[view] = convert_color_space(frames_dict[view], 'YBR_FULL', 'RGB')
                    t, w, h, c = frames_dict[view].shape
                elif len(frames_dict[view].shape) == 3:
                    t, w, h = frames_dict[view].shape
                elif len(frames_dict[view].shape) == 2:
                    frames_dict[view] = np.expand_dims(frames_dict[view], axis=0)
                    t, w, h = frames_dict[view].shape

                frames_dict[view] = self.compute_pixel_discrepancy(frames_dict[view])
                for i in range(t):
                    frame = Image.fromarray(np.uint8(frames_dict[view][i, ...]))
                    frame.save(new_image_dir+'/' + str(i).zfill(3)+'.jpg')

        return frames_dict

    def compute_pixel_discrepancy(self, image_list):
        if len(image_list.shape) == 4:
            t, w, h, c = image_list.shape
            # image_list_mean = np.mean(image_list, axis=-1)
            discrepancy_image = np.zeros((w, h, c))
            for i in range(w):
                for j in range(h):
                    for k in range(c):
                        pixel_array =  image_list[:, i, j, k]
                        pixel_mean = np.full_like(pixel_array.shape, np.mean(pixel_array))
                        pixel_discrepancy = ((pixel_array - pixel_mean)**2).mean()
                        discrepancy_image[i, j, k] = pixel_discrepancy
        elif len(image_list.shape) == 3:
            t, w, h = image_list.shape
            # image_list_mean = np.mean(image_list, axis=-1)
            discrepancy_image = np.zeros((w, h))
            for i in range(w):
                for j in range(h):
                    pixel_array =  image_list[:, i, j]
                    pixel_mean = np.full_like(pixel_array.shape, np.mean(pixel_array))
                    pixel_discrepancy = ((pixel_array - pixel_mean)**2).mean()
                    discrepancy_image[i, j] = pixel_discrepancy
        discrepancy_image = np.where(discrepancy_image > 0.01, 1, 0)
        # discrepancy_image = np.repeat(np.expand_dims(discrepancy_image, axis=2), c, axis=2)
        discrepancy_image = np.repeat(np.expand_dims(discrepancy_image, axis=0), t, axis=0)

        return image_list * discrepancy_image


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--image-size', type=int, default=(112,112,16), help='Image height and width (default: 256)')
    parser.add_argument('--mask-size', type=int, default=8, help='The size of mask patch (default: 16)')
    parser.add_argument('--mask-ratio', type=float, default=0.6, help='The ratio of masking area in an image (default: 0.75)')
    args = parser.parse_args()
    args.dataset_path = r'/home/jyangcu/Dataset/PH_HK'
    args.new_dataset_path = r'/home/jyangcu/Dataset/PH_HK_image'

    data_dict = dict()
    infos = np.load(f'/home/jyangcu/Dataset/dataset_pa_iltrasound_nii_files_3rdcenters/save_infos_reg_v3.npy', allow_pickle=True).item()
    
    from monai.data import DataLoader
    from torchvision import utils as vutils
    train_ds = PHHKDataset(args, infos, select_set=['Other'], select_view=['1','2','3','4'])
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=1)
    from einops import rearrange
    count = 0
    for _, _, _ in tqdm(train_loader):
        count += 1
        # if count == 1:
        #     break
        # else:
        #     pass