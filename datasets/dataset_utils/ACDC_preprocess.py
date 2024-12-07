#jyangcu@connect.ust.hk
import os
import random
import numpy as np
import pandas as pd

from tqdm import tqdm
from torch.utils.data import Dataset
import configparser

import pydicom
import SimpleITK as sitk
from PIL import Image


np.set_printoptions(threshold=np.inf)
random.seed(7777)
np.random.seed(7777)


class ACDC_Dataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.data_all = self.mapping_data()

    def __getitem__(self, index):
        pass
        return None

    def __len__(self):
        return len(self.data_list)
    
    def mapping_data(self):
        datainfo = {}
        def mkdirs(path):
            if os.path.exists(path):
                pass
            else:
                os.makedirs(path)
        
        configParser = configparser.RawConfigParser()
        for num, path in enumerate(self.args.dataset_path):
            if 'train' in path:
                data_type = 'train'
                datainfo['train'] = {}
            elif 'test' in path:
                data_type = 'test'
                datainfo['test'] = {}

            for root, dirs, files in os.walk(path):
                if len(dirs) > 0:
                    dirs.sort()
                    for dir in tqdm(dirs):
                        mkdirs(self.args.new_dataset_path[num] + '/' + dir)
                        for _, _, files in os.walk(path + '/' + dir):
                            files.sort()
                            if data_type == 'test':
                                del files[1]
                            conf = {}
                            with open(path + '/' + dir + '/' + files[0]) as stream:
                                for line in stream:
                                    if line.startswith('#'):
                                        continue
                                    key, val = line.strip().split(':')
                                    conf[key] = val
                                # configParser.read_string("[top]\n" + stream.read())  # This line does the trick.

                            image_name = files[1]
                            case_name = image_name.split('.')[0][:-3]
                            image_path = path + '/' + dir + '/' + image_name
                            mkdirs(self.args.new_dataset_path[num] + '/' + dir + '/' + case_name)
                            data=sitk.ReadImage(image_path)
                            data=sitk.GetArrayFromImage(data).astype(np.float32)
                            length, depth, width, hight = data.shape
                            datainfo[data_type][case_name] = {'name':case_name,
                                                         'nill_path':path + '/' + dir + '/' + image_name,
                                                         'image_path':self.args.new_dataset_path[num] + '/' + dir + '/' + case_name,
                                                         'start_path':path + '/' + dir + '/' + files[2],
                                                         'end_path':path + '/' + dir + '/' + files[4],
                                                         'anno_start_path':path + '/' + dir + '/' + files[3],
                                                         'anno_end_path':path + '/' + dir + '/' + files[5],
                                                         'ED_f':conf['ED'],
                                                         'ES_f':conf['ES'],
                                                         'shape':data.shape, 
                                                         'type':data_type}
                            # for idx_t in range(length):
                            #     t_frame = data[idx_t, ...]
                            #     mkdirs(self.args.new_dataset_path[num] + '/' + dir + '/' + case_name + '/' + str(idx_t))
                            #     for idx_d in range(depth):
                            #         d_frame = t_frame[idx_d, ...]
                            #         im = Image.fromarray(d_frame)
                            #         if im.mode != 'RGB':
                            #             im = im.convert('RGB')
                            #         im.save(self.args.new_dataset_path[num] + '/' + dir + '/' + case_name + '/' + str(idx_t) + '/' + str(idx_d) + '.jpg')
        np.save('ACDC_info.npy', datainfo)
        return None


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--image-size', type=int, default=(112,112,16), help='Image height and width (default: 256)')
    parser.add_argument('--mask-size', type=int, default=8, help='The size of mask patch (default: 16)')
    parser.add_argument('--mask-ratio', type=float, default=0.6, help='The ratio of masking area in an image (default: 0.75)')
    args = parser.parse_args()
    args.dataset_path = ['/home/jyangcu/Dataset/ACDC/train', '/home/jyangcu/Dataset/ACDC/test']
    args.new_dataset_path = ['/home/jyangcu/Dataset/ACDC_image/train', '/home/jyangcu/Dataset/ACDC_image/test']
    
    from monai.data import DataLoader
    from torchvision import utils as vutils
    train_ds = ACDC_Dataset(args)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=1)
    from einops import rearrange
    count = 0
    for _, _, _ in tqdm(train_loader):
        count += 1
        # if count == 1:
        #     break
        # else:
        #     pass