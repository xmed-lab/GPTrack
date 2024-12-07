# CUDA_VISIBLE_DEVICES=1,2,4,5 
import io
import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import utils as vutils

from models.GPTrack_2D import GPTrack2D
from einops import rearrange
from utils.tools import get_world_size, get_global_rank, get_local_rank, get_master_ip
from datasets.pah_dataset_all import Seg_PAHDataset
from datasets.pah_all_labelled import Seg_PAHDataset_ALL_Label
from monai.data import DataLoader
from monai.metrics import compute_hausdorff_distance

import cv2
import wandb
import matplotlib.pyplot as plt


class Train:
    def __init__(self, args):
        self.RViT = GPTrack2D(image_size = args.image_size[0],
                              patch_size = args.patch_size,
                              length = args.image_size[2],
                              depth = args.num_layers,
                              heads = args.num_heads,
                              mlp_dim = args.latent_dim,
                              dropout = 0.1,).to(args.device)
        
        # pretrain_params = torch.load('/home/jyangcu/Pulmonary_Arterial_Hypertension/results/checkpoints/checkpoint_1_best.pth', map_location='cpu')
        # pretrain_params = {k.replace('module.', ''): v for k, v in pretrain_params.items() if k.replace('module.', '') in self.RViT.state_dict()}
        # self.RViT.load_state_dict(pretrain_params)
        self.optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, self.RViT.parameters()), lr=args.learning_rate, betas=[args.beta1, args.beta2])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.1)

        print("---- Finish the Model Loading ----")

        if args.distributed:
            self.RViT = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.RViT)
            self.RViT = torch.nn.parallel.DistributedDataParallel(self.RViT, broadcast_buffers=True, find_unused_parameters=True,)
        
        elif len(args.enable_GPUs_id) > 1:
            # For normal
            self.RViT = torch.nn.DataParallel(self.RViT, device_ids=args.enable_GPUs_id, output_device=args.enable_GPUs_id[0])
        
        self.mseloss = torch.nn.MSELoss()
        self.l1loss = torch.nn.L1Loss()

        self.prepare_training()
        train_dataset = Seg_PAHDataset(args, set_select=['gy','rmyy'], view_num=['4'])
        self.train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
        valid_dataset = Seg_PAHDataset_ALL_Label('/home/jyangcu/Dataset/dataset_pa_iltrasound_nill_files_clean_image/label_all_frame', view_num=['4'], length = args.image_size[-1])
        self.valid_loader  = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=1)

        print("---- Finish the Dataset Loading ----")

        self.train(args)

    @staticmethod
    def prepare_training():
        os.makedirs("results", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)

    def train(self, args):
        self.mean_dice = 0
        for self.epoch in range(args.epochs):
            pbar = tqdm(self.train_loader)
            self.RViT.train()
            for step, (vids, transfromed_vids) in enumerate(pbar):
                # if step > 9:
                #     break
                hidden = torch.zeros(args.batch_size, (args.image_size[0] // args.patch_size) ** 2, args.latent_dim).to(args.device)
                loss, inf_flow_all, neg_inf_flow_all, lag_flow, lag_register, forward_regsiter, backward_regsiter = \
                    self.RViT(vids.transpose(1,2).to(args.device), hidden, transfromed_vids.transpose(1,2).to(args.device))

                (kl_param_loss, recon_loss_tgt, recon_loss_src, smooth_loss_inf, smooth_loss_neg, lag_gradient_loss, recon_loss_lag, recon_loss_lag_l1, total_loss) = loss

                self.optimizer.zero_grad()
                total_loss.backward(retain_graph=False)
                self.optimizer.step()

                if args.clip_grad:
                    torch.nn.utils.clip_grad_norm_(self.RViT.parameters(), max_norm=1, norm_type=2)

                if args.wandb:
                    if args.local_rank == args.enable_GPUs_id[0]:
                        wandb.log({'Loss/Total Loss': total_loss,
                                   'Loss/KL Param Loss': kl_param_loss,
                                   'Loss/Recon target Loss': recon_loss_tgt,
                                   'Loss/Recon Source Loss': recon_loss_src,
                                   'Loss/Smooth Inf Loss': smooth_loss_inf,
                                   'Loss/Smooth Neg Loss': smooth_loss_neg,
                                   'Loss/Lag Gardient Loss': lag_gradient_loss,
                                   'Loss/Recon Lag Loss': recon_loss_lag,
                                   'Loss/Recon Lag Loss L1': recon_loss_lag_l1,
                                  })

            self.scheduler.step()
        
    def validation(self, args):
        self.RViT.eval()
        pbar = tqdm(self.valid_loader)
        all_dice, all_hd95 = [], []
        for step, (vids, vids_org, masks) in enumerate(pbar):
            # b, c, t, h, w = vids.size
            hidden = torch.zeros(1, (args.image_size[0] // args.patch_size) ** 2, args.latent_dim).to(args.device)
            inf_flow_all, neg_inf_flow_all, lag_flow, neg_lag_flow, lag_register, forward_regsiter, backward_regsiter = self.RViT(vids.transpose(1,2).to(args.device), hidden, train=False)

            track_segments = []
            for idx in range(len(inf_flow_all)):
                # For Masks Evaluation
                if idx < 1:
                    c_mask = masks[:, 0, idx, ...].transpose(1,2)
                c_mask = self.plot_seg_warpgrid(c_mask, inf_flow_all[idx][0, ...])
                track_segments.append(c_mask)
        
            track_segments = torch.stack(track_segments, dim = 0)
            gt_segments = masks[:, 0, 1:,...].permute(1,0,3,2)

            _, dice, _, _, _ = self._calculate_overlap_metrics(torch.where(gt_segments > 0, 1, 0), torch.where(track_segments > 0, 1, 0))

            # track_segments = self.transfor_label(track_segments)
            # gt_segments = self.transfor_label(gt_segments)

            all_dice.append(dice.detach().cpu().numpy())
            # all_hd95.append(hd.detach().cpu().numpy())

        if args.wandb:
            if args.local_rank == args.enable_GPUs_id[0]:
                wandb.log({'Validation/DICE': np.mean(all_dice),
                        #    'Validation/HD': np.mean(all_hd95),
                          })

        orginial_imgs = vids[0, :, 1:, ...].transpose(0, 1)
        forward_imgs = torch.stack(forward_regsiter, dim=0)[:, 0, ...].detach().cpu()
        backward_imgs = torch.stack(backward_regsiter, dim=0)[:, 0, ...].detach().cpu()
        lag_imgs = lag_register[0].detach().cpu()
        combine_imgs = torch.cat([orginial_imgs, forward_imgs, backward_imgs, lag_imgs], dim=0)
        vutils.save_image(combine_imgs, os.path.join("results/example_result", f"example_{self.epoch}.jpg"), nrow=len(inf_flow_all))
        
        new_mean_dice = np.mean(all_dice)
        if new_mean_dice > self.mean_dice:
            torch.save(self.RViT.state_dict(), f'./results/checkpoints/checkpoint_{self.epoch}_best.pth')
            self.mean_dice = new_mean_dice

        torch.save(self.RViT.state_dict(), f'./results/checkpoints/checkpoint_{self.epoch}_{new_mean_dice:.4f}.pth')
        print('Validation Dice Score : ', new_mean_dice)
        self.RViT.train()

    def plot_seg_warpgrid(self, mask, warp, wrap_seg=True):
        """
        plots the given warpgrid
        @param warp: array, H x W x 2, the transformation
        @param interval: int, The interval between grid-lines
        @param show_axis: Bool, should axes be included?
        @return: matplotlib plot. Show with plt.show()
        """
        vectors = [torch.arange(0, s) for s in (args.image_size[0], args.image_size[1])]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)

        warp = warp.unsqueeze(0).detach().cpu()
        warp = grid + warp
        shape = warp.shape[2:]

        for i in range(len(shape)):
            warp[:, i, ...] = 2 * (warp[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            warp = warp.permute(0, 2, 3, 1)
            warp = warp[..., [1, 0]]

        mask = mask.float()
        # Get the warpping img
        if wrap_seg:
            seg_warp = torch.nn.functional.grid_sample(mask.unsqueeze(0), warp, align_corners=True, mode='nearest')[0]

        return seg_warp
    
    def _calculate_overlap_metrics(self, gt, pred, eps=1e-5):
        output = pred.reshape(-1, )
        target = gt.reshape(-1, ).float()

        tp = torch.sum(output * target)  # TP
        fp = torch.sum(output * (1 - target))  # FP
        fn = torch.sum((1 - output) * target)  # FN
        tn = torch.sum((1 - output) * (1 - target))  # TN

        # hd = compute_hausdorff_distance(pred.squeeze(2), gt.squeeze(2))
        # hd = torch.where(torch.isinf(hd),0, hd)
        # hd = torch.where(torch.isnan(hd),0, hd)
        pixel_acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)
        dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        precision = (tp + eps) / (tp + fp + eps)
        recall = (tp + eps) / (tp + fn + eps)
        specificity = (tn + eps) / (tn + fp + eps)

        return pixel_acc, dice, precision, specificity, recall

    def transfor_label(self, seg):
        LV = torch.where(seg == 1, 1, 0)
        LA = torch.where(seg == 2, 1, 0)
        RA = torch.where(seg == 3, 1, 0)
        RV = torch.where(seg == 4, 1, 0)
        return torch.stack([LV, LA, RA, RV], dim=0)

def main(rank, args):

    def wandb_init():
        wandb.init(
            project='Unsupervised Echocardiogram Segmentation',
            entity='jiewen-yang66',
            name='PHHK-Dataset-Deep-Tag',
            notes='Ver 1.0',
            save_code=True
        )
        wandb.config.update(args)

    try:
        args.local_rank
    except AttributeError:
            args.global_rank = rank
            args.local_rank = args.enable_GPUs_id[rank]
    else:
        if args.distributed:
            args.global_rank = rank
            args.local_rank = args.enable_GPUs_id[rank]

    if args.distributed:
        torch.cuda.set_device(int(args.local_rank))
        torch.distributed.init_process_group(backend='nccl',
                                             init_method=args.init_method,
                                             world_size=args.world_size,
                                             rank=args.global_rank,
                                             group_name='mtorch'
                                             )
        print('using GPU {}-{} for training'.format(
            int(args.global_rank), int(args.local_rank)
            ))

        if args.wandb:
            if args.local_rank == args.enable_GPUs_id[0]:
                wandb_init()

    else:
        if args.wandb:
            wandb_init()

    if torch.cuda.is_available(): 
        args.device = torch.device("cuda:{}".format(args.local_rank))
    else: 
        args.device = 'cpu'

    Train(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EchoNet")
    parser.add_argument('--latent-dim', type=int, default=64, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=tuple, default=(256, 256, 33), help='Image height and width (default: (112, 112 ,16))')
    parser.add_argument('--image-channels', type=int, default=1, help='Number of channels of images (default: 3)')
    parser.add_argument('--patch-size', type=int, default=16, help='Patch height and width (default: 8)')
    parser.add_argument('--blurring', type=bool, default=False, help='Whether blur the image')
    parser.add_argument('--max_sample_rate', type=int, default=1, help='The sampling rate for the video')
    parser.add_argument('--num-heads', type=int, default=8, help='The number of head of multiscale attention (default: 8)')
    parser.add_argument('--num-layers', type=int, default=2, help='The number of transformer layers')

    parser.add_argument('--mask-size', type=int, default=8, help='The size of mask patch (default: 16)')
    parser.add_argument('--mask-ratio', type=float, default=0.7, help='The ratio of masking area in an image (default: 0.75)')
    parser.add_argument('--selected-view', type=list, default=['4'], help='The selected view from dataset')
    parser.add_argument('--dataset-path', type=str, default='/home/jyangcu/Dataset/dataset_pa_iltrasound_nill_files_clean_image', help='Path to data (default: /data)')
    parser.add_argument('--batch-size', type=int, default=4, help='Input batch size for training (default: 6)')
    
    parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=5e-4, help='Learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.99, help='Adam beta param (default: 0.999)')
    parser.add_argument('--clip-grad', type=bool, default=False, help='perform gradient clipping in training (default: False)')

    parser.add_argument('--enable_GPUs_id', type=list, default=[1], help='The number and order of the enable gpus')
    parser.add_argument('--wandb', type=bool, default=False, help='Enable Wandb')

    args = parser.parse_args()

    # setting distributed configurations
    # args.world_size = 1
    args.world_size = len(args.enable_GPUs_id)
    args.init_method = f"tcp://{get_master_ip()}:{23455}"
    args.distributed = True if args.world_size > 1 else False

    # setup distributed parallel training environments
    if get_master_ip() == "127.0.0.1" and args.distributed:
        # manually launch distributed processes 
        torch.multiprocessing.spawn(main, nprocs=args.world_size, args=(args,))
    else:
        # multiple processes have been launched by openmpi
        args.local_rank = args.enable_GPUs_id[0]
        args.global_rank = args.enable_GPUs_id[0]
    
        main(args.local_rank, args)
