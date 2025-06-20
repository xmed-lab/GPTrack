# CUDA_VISIBLE_DEVICES=1,2,4,5 
import io
import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import utils as vutils

from models.GPTrack_3D import RViT
from einops import rearrange
from utils.tools import get_world_size, get_global_rank, get_local_rank, get_master_ip
from datasets.ACDC import ACDC_Dataset
from datasets.ACDC_test import ACDC_Dataset as ACDC_Dataset_test
from monai.data import DataLoader

from utils.SSIM_metric import SSIM
from utils.PSNR_metric import PSNR

import cv2
import wandb
import matplotlib.pyplot as plt


class Train:
    def __init__(self, args):
        self.RViT = RViT(image_size = args.image_size,
                         patch_size = args.patch_size,
                         length = args.image_size[2],
                         depth = args.num_layers,
                         heads = args.num_heads,
                         mlp_dim = args.latent_dim,
                         dropout = 0.1,).to(args.device)
        
        # pretrain_params = torch.load('./results/checkpoints/checkpoint_0.pth', map_location='cpu')
        # pretrain_params = {k.replace('module.', ''): v for k, v in pretrain_params.items() if k.replace('module.', '') in self.RViT.state_dict()}
        # self.RViT.load_state_dict(pretrain_params)
        self.optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, self.RViT.parameters()), lr=args.learning_rate, betas=[args.beta1, args.beta2])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1)

        print("---- Finish the Model Loading ----")

        if args.distributed:
            self.RViT = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.RViT)
            self.RViT = torch.nn.parallel.DistributedDataParallel(self.RViT, broadcast_buffers=True, find_unused_parameters=True,)
        
        elif len(args.enable_GPUs_id) > 1:
            # For normal
            self.RViT = torch.nn.DataParallel(self.RViT, device_ids=args.enable_GPUs_id, output_device=args.enable_GPUs_id[0])

        self.SSIM_metric = SSIM(window_size = 8)
        self.PSNR_metric = PSNR()


        self.prepare_training()
        infos = np.load('/home/jyangcu/Pulmonary_Arterial_Hypertension/datasets/dataset_utils/ACDC_info.npy', allow_pickle=True).item()
        train_dataset = ACDC_Dataset(args, infos)
        self.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
        infos = np.load('./datasets/dataset_utils/ACDC_info.npy', allow_pickle=True).item()
        valid_dataset = ACDC_Dataset_test(args, infos)
        self.valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=1)

        print("---- Finish the Dataset Loading ----")

        self.train(args)

    @staticmethod
    def prepare_training():
        os.makedirs("results", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)

    def train(self, args):
        for epoch in range(args.epochs):
            pbar = tqdm(self.train_loader)
            for step, (vids) in enumerate(pbar):
                b, t, c, h, w, d = vids.shape
                input_vids = vids
                hidden = torch.zeros(1, ((args.image_size[0] * args.image_size[1] * args.image_size[2]) // (args.patch_size[0] * args.patch_size[1] * args.patch_size[2])), args.latent_dim).to(args.device)
                loss, inf_flow_all, neg_inf_flow_all, lag_flow, lag_register, forward_regsiter, backward_regsiter = self.RViT(input_vids.to(args.device), hidden)

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

            # for idx in range(len(inf_flow_all)-1):
            #     inf_flow_plt = self.plot_warpgrid(args, input_vids[0, :, idx, ...], inf_flow_all[idx][0, ...], interval=8)
            #     inf_flow_plt.savefig(f'./results/flow_result/out_inf_flow_translation_warp_{idx}.png')
            #     inf_flow_plt.clf()
                
            #     lag_flow_plt = self.plot_warpgrid(args, lag_register[0][idx], lag_flow[0, :, idx, ...], interval=8)
            #     lag_flow_plt.savefig(f'./results/flow_result/out_lag_flow_translation_warp_{idx}.png')
            #     lag_flow_plt.clf()
            
            # orginial_imgs = input_vids[0, :, 1:, ...].transpose(0, 1)
            # forward_imgs = torch.stack(forward_regsiter, dim=0)[:, 0, ...].detach().cpu()
            # backward_imgs = torch.stack(backward_regsiter, dim=0)[:, 0, ...].detach().cpu()
            # lag_imgs = lag_register[0].detach().cpu()
            # combine_imgs = torch.cat([orginial_imgs, forward_imgs, backward_imgs, lag_imgs], dim=0)
            # vutils.save_image(combine_imgs.add(1.0).mul(0.5), os.path.join("results/example_result", f"example_{epoch}_{step}.jpg"), nrow=len(inf_flow_all))
            
            self.eval(args)
            self.scheduler.step()
            torch.save(self.RViT.state_dict(), f'./results/checkpoints/checkpoint_{epoch}.pth')
    
    def eval(self, args):
        self.RViT.eval()
        record_steps = 0
        all_info = []
        all_psnr, all_dice, all_ssim, lv_dice, rv_dice, myo_dice, lv_dice = [], [], [], [], [], [], []
        pbar = tqdm(self.valid_loader)
        for step, (vids, start_anno, end_anno, es_f) in enumerate(pbar):
            hidden = torch.zeros(1, ((args.image_size[0] * args.image_size[1] * args.image_size[2]) // (args.patch_size[0] * args.patch_size[1] * args.patch_size[2])), args.latent_dim).to(args.device)
            _, inf_flow_all, neg_inf_flow_all, lag_flow, neg_lag_flow, lag_register, forward_regsiter, backward_regsiter = self.RViT(vids.to(args.device), hidden, train=False)

            com_input_vids = vids.squeeze().permute(0,3,1,2)[1:, ...].cpu().mul(255)
            com_lag_register = lag_register.squeeze(0).cpu().mul(255)
            com_forward_regsiter = torch.stack(forward_regsiter, dim=1).squeeze().permute(0,3,1,2).cpu().mul(255)

            ssim_score = self.SSIM_metric(com_forward_regsiter, com_input_vids)
            psnr_score = self.PSNR_metric(com_forward_regsiter, com_input_vids)
            all_psnr.append(psnr_score.detach().cpu().numpy())
            all_ssim.append(ssim_score.detach().cpu().numpy())

            print("lag-reg: SSIM ---> {} , PSNR ---> {}".format(ssim_score, psnr_score))

            for idx in tqdm(range(len(inf_flow_all))):
                # inf_flow_plt = self.plot_warpgrid(vids_org[0, :, idx, ...], inf_flow_all[idx][0, ...], segment_result, interval=4, mark='r')
                # inf_flow_plt.savefig(f'./results/flow_result_eval/inf_flow_img_warp_{idx}.png')
                # inf_flow_plt.clf()
                
                # vutils.save_image(input_vids[0, :, idx, ...].add(1.0).mul(0.5), f'./results/flow_result_eval/org_img_{idx}.png')

                # lag_flow_plt = self.plot_warpgrid(lag_register[0][idx], lag_flow[0, :, idx, ...], interval=4, mark='c')
                # lag_flow_plt.savefig(f'./results/flow_result_eval/lag_flow_img_warp_{idx}.png')
                # lag_flow_plt.clf()

                # inf_flow_plt = self.plot_warpgrid(vids[0, :, idx, ...], inf_flow_all[idx][0, ...], interval=8, mark='w', heatmap=False)
                # inf_flow_plt.savefig(f'./results/flow_result_eval/inf_flow_heatmap_warp_{idx}.png', pad_inches=0.0)
                # inf_flow_plt.clf()

                # For Masks Evaluation
                if idx == 0:
                    c_mask = start_anno
                inf_flow_seg_plt, c_mask = self.plot_seg_warpgrid(vids[0, idx+1, ...], c_mask, end_anno, inf_flow_all[idx][0, ...], mark='w')

                if idx == int(es_f[0]) - 2:
                    track_segments = c_mask

                # inf_flow_seg_plt.savefig(f'./results/flow_result_eval/inf_flow_seg_warp_{idx}.png',pad_inches=0.0)
                # inf_flow_seg_plt.clf()

            gt_segments = end_anno
            gt_segments = self.transfor_label(gt_segments)
            track_segments = self.transfor_label(track_segments)
            pixel_acc, dice, precision, specificity, recall = self._calculate_overlap_metrics(torch.where(gt_segments > 0, 1, 0), torch.where(track_segments > 0, 1, 0))

            all_dice.append(dice.detach().cpu().numpy().item())

            for i in range(3):
                i_segments_track = track_segments[i, ...]
                i_segments_gt = gt_segments[i, ...]

                _, i_dice, _, _, _ = self._calculate_overlap_metrics(i_segments_gt, i_segments_track)
                if i == 0:
                    rv_dice.append(i_dice.detach().cpu().numpy())
                elif i == 1:
                    myo_dice.append(i_dice.detach().cpu().numpy())
                elif i == 2:
                    lv_dice.append(i_dice.detach().cpu().numpy())
            
            print("Pixel Acc is : ", pixel_acc)
            print("Dice Score is : ", dice)
            print("Precision is : ", precision)
            print("Specificity is : ", specificity)
            print("Recall is : ", recall)
            print("ES Frame Number : ", int(es_f[0])-1)
            # orginial_imgs = input_vids[0, :, 1:, ...].transpose(0, 1)
            # forward_imgs = torch.stack(forward_regsiter, dim=0)[:, 0, ...].detach().cpu()
            # backward_imgs = torch.stack(backward_regsiter, dim=0)[:, 0, ...].detach().cpu()
            # lag_imgs = lag_register[0].detach().cpu()
            # combine_imgs = torch.cat([orginial_imgs, forward_imgs, backward_imgs, lag_imgs], dim=0)
            # vutils.save_image(combine_imgs.add(1.0).mul(0.5), os.path.join("results/example_result_eval", f"example_{step}.jpg"), nrow=len(inf_flow_all))

        print("SSIM: Mean:{}, Std:{}".format(np.mean(all_ssim), np.std(all_ssim)))
        print("PSNR: Mean:{}, Std:{}".format(np.mean(all_psnr), np.std(all_psnr)))
        print("DICE: Mean:{}, Std:{}".format(np.mean(all_dice), np.std(all_dice)))
        print("RV_DICE: Mean:{}, Std:{}".format(np.mean(rv_dice), np.std(rv_dice)))
        print("MYO_DICE: Mean:{}, Std:{}".format(np.mean(myo_dice), np.std(myo_dice)))
        print("LV_DICE: Mean:{}, Std:{}".format(np.mean(lv_dice), np.std(lv_dice)))


    def plot_seg_warpgrid(self, img, mask, mask_tgt, warp, segment_result=None, 
                          interval=2, show_axis=False, mark='k', wrap_seg=True):
        """
        plots the given warpgrid
        @param warp: array, H x W x 2, the transformation
        @param interval: int, The interval between grid-lines
        @param show_axis: Bool, should axes be included?
        @return: matplotlib plot. Show with plt.show()
        """
        vectors = [torch.arange(0, s) for s in (args.image_size[0], args.image_size[1], args.image_size[2])]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)

        warp = warp.unsqueeze(0).detach().cpu()
        warp = grid + warp
        warp_save = grid + warp
        shape = warp.shape[2:]

        for i in range(len(shape)):
            warp[:, i, ...] = 2 * (warp[:, i, ...] / (shape[i] - 1) - 0.5)
            warp_save[:, i, ...] = (2 * (warp[:, i, ...] / (shape[i] - 1) - 0.5) - 2 * (grid[:, i] / (shape[i] - 1) - 0.5)) * shape[i]

        if len(shape) == 2:
            warp = warp.permute(0, 2, 3, 1)
            warp = warp[..., [1, 0]]

            warp_save = warp_save.permute(0, 2, 3, 1)
            warp_save = warp_save[..., [1, 0]]

        elif len(shape) == 3:
            warp = warp.permute(0, 2, 3, 4, 1)
            warp = warp[..., [2, 1, 0]]

            warp_save = warp_save.permute(0, 2, 3, 4, 1)
            warp_save = warp_save[..., [2, 1, 0]]

        warp_save = warp_save[0, ...]

        if img is not None:
            img = img
            mask = mask.float()
            mask_tgt = mask_tgt.float()

            # Get the warpping img
            if wrap_seg:
                seg_warp = torch.nn.functional.grid_sample(mask, warp, align_corners=True, mode='nearest')
                seg_warp_a = torch.where(seg_warp == 1, 10,  seg_warp)
                seg_warp_b = torch.where(seg_warp == 1, 10,  seg_warp)
                seg_warp_c = torch.where(seg_warp == 1, 255, seg_warp)

                seg_warp_a = torch.where(seg_warp_a == 2, 255, seg_warp_a)
                seg_warp_b = torch.where(seg_warp_b == 2, 10,  seg_warp_b)
                seg_warp_c = torch.where(seg_warp_c == 2, 10,  seg_warp_c)

                seg_warp_a = torch.where(seg_warp_a == 3, 10,  seg_warp_a)
                seg_warp_b = torch.where(seg_warp_b == 3, 255, seg_warp_b)
                seg_warp_c = torch.where(seg_warp_c == 3, 10,  seg_warp_c)

                # seg_warp_rgb = torch.cat([seg_warp_a, seg_warp_b, seg_warp_c], dim=0)

                seg_mask_a = torch.where(seg_warp == mask_tgt, 0, seg_warp_a)
                seg_mask_b = torch.where(seg_warp == mask_tgt, 0, seg_warp_b)
                seg_mask_c = torch.where(seg_warp == mask_tgt, 0, seg_warp_c)
                seg_mask = torch.cat([seg_mask_a, seg_mask_b, seg_mask_c], dim=0)

                seg_mask_gt_a = torch.where(mask_tgt == 1, 10,  mask_tgt)
                seg_mask_gt_b = torch.where(mask_tgt == 1, 10, mask_tgt)
                seg_mask_gt_c = torch.where(mask_tgt == 1, 255, mask_tgt)

                seg_mask_gt_a = torch.where(seg_mask_gt_a == 2, 255,  seg_mask_gt_a)
                seg_mask_gt_b = torch.where(seg_mask_gt_b == 2, 10, seg_mask_gt_b)
                seg_mask_gt_c = torch.where(seg_mask_gt_c == 2, 10, seg_mask_gt_c)

                seg_mask_gt_a = torch.where(seg_mask_gt_a == 3, 10,  seg_mask_gt_a)
                seg_mask_gt_b = torch.where(seg_mask_gt_b == 3, 255, seg_mask_gt_b)
                seg_mask_gt_c = torch.where(seg_mask_gt_c == 3, 10,  seg_mask_gt_c)

                seg_mask_gt = torch.cat([seg_mask_gt_a, seg_mask_gt_b, seg_mask_gt_c], dim=0)

            if show_axis is False:
                plt.axis('off')

            n, w, h, d = img.shape
            seg_mask = seg_mask[..., d//2].squeeze()
            seg_mask_gt = seg_mask_gt[..., d//2].squeeze()
            # seg_warp_rgb = seg_warp_rgb[..., d//2].squeeze()
            img = img[..., d//2].expand(3,-1,-1).add(1.0).mul(127.5)
            img = torch.where(seg_mask > 0, seg_mask, img) * 0.7 + img * 0.3
            # img = torch.where(seg_mask_gt > 0, seg_mask_gt, img) * 0.7 + img * 0.3
            plt.imshow(img.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8), cmap='viridis', vmin=0, vmax=255)

        return plt, seg_warp
    
    def draw_grid(self, img, grid_height = 6, grid_width = 6, line_width=5):
        height, width, _ = img.shape
        img = (img + 1) * 127.5
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        for x in range(0, width-1, grid_width):
            cv2.line(img, (x, 0), (x, height), (255))
        for y in range(0, height-1, grid_height):
            cv2.line(img, (0, y), (width, y), (255))

        img = img / 127.5 - 1

        return img

    def _calculate_overlap_metrics(self, gt, pred, eps=1e-5):
        output = pred.reshape(-1, )
        target = gt.reshape(-1, ).float()

        tp = torch.sum(output * target)  # TP
        fp = torch.sum(output * (1 - target))  # FP
        fn = torch.sum((1 - output) * target)  # FN
        tn = torch.sum((1 - output) * (1 - target))  # TN

        pixel_acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)
        dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        precision = (tp + eps) / (tp + fp + eps)
        recall = (tp + eps) / (tp + fn + eps)
        specificity = (tn + eps) / (tn + fp + eps)

        return pixel_acc, dice, precision, specificity, recall

    def transfor_label(self, seg):
        RV  = torch.where(seg == 1, 1, 0)
        MYO = torch.where(seg == 2, 1, 0)
        LV  = torch.where(seg == 3, 1, 0)
        return torch.stack([RV, MYO, LV], dim=0)

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
    parser.add_argument('--image-size', type=tuple, default=(128, 128, 32), help='Image height and width (default: (112, 112 ,16))')
    parser.add_argument('--image-channels', type=int, default=1, help='Number of channels of images (default: 3)')
    parser.add_argument('--patch-size', type=int, default=(16, 16, 16), help='Patch height and width (default: 8)')
    parser.add_argument('--blurring', type=bool, default=False, help='Whether blur the image')
    parser.add_argument('--max_sample_rate', type=int, default=1, help='The sampling rate for the video')
    parser.add_argument('--num-heads', type=int, default=8, help='The number of head of multiscale attention (default: 8)')
    parser.add_argument('--num-layers', type=int, default=2, help='The number of transformer layers')

    parser.add_argument('--mask-size', type=int, default=8, help='The size of mask patch (default: 16)')
    parser.add_argument('--mask-ratio', type=float, default=0.7, help='The ratio of masking area in an image (default: 0.75)')
    parser.add_argument('--dataset-path', type=str, default='/home/jyangcu/Dataset/PH_HK_image', help='Path to data (default: /data)')
    parser.add_argument('--batch-size', type=int, default=1, help='Input batch size for training (default: 6)')
    
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs to train (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=5e-4, help='Learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.99, help='Adam beta param (default: 0.999)')
    parser.add_argument('--clip-grad', type=bool, default=True, help='perform gradient clipping in training (default: False)')

    parser.add_argument('--enable_GPUs_id', type=list, default=[0], help='The number and order of the enable gpus')
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
