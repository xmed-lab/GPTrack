# CUDA_VISIBLE_DEVICES=1,2,4,5 
import io
import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
from torchvision.transforms import GaussianBlur

from models.GPTrack_3D import RViT
# from models.RViT_BidTag import RViT
from utils.tools import get_world_size, get_global_rank, get_local_rank, get_master_ip
from utils.SSIM_metric import SSIM
from utils.PSNR_metric import PSNR
# from datasets.pah_dataset_test import Seg_PAHDataset
from datasets.ACDC_test import ACDC_Dataset
from monai.data import DataLoader

import cv2
import wandb
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm as cm

norm = matplotlib.colors.Normalize()
Gaussian = GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))

class Eval:
    def __init__(self, args):
        self.RViT = RViT(image_size = args.image_size,
                         patch_size = args.patch_size,
                         length = args.image_size[2],
                         depth = args.num_layers,
                         heads = args.num_heads,
                         mlp_dim = args.latent_dim,
                         dropout = 0.1,).to(args.device)
        
        pretrain_params = torch.load('./results/checkpoints/checkpoint.pth', map_location='cpu')
        pretrain_params = {k.replace('module.', ''): v for k, v in pretrain_params.items() if k.replace('module.', '') in self.RViT.state_dict()}
        self.RViT.load_state_dict(pretrain_params)
        
        infos = np.load('./datasets/dataset_utils/ACDC_info.npy', allow_pickle=True).item()
        valid_dataset = ACDC_Dataset(args, infos)
        self.valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=1)
        self.SSIM_metric = SSIM(window_size = 8)
        self.PSNR_metric = PSNR()

        self.eval(args)

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


    def plot_warpgrid(self, img, warp, segment_result=None, 
                      interval=2, show_axis=False, mark='k', next_frame=None, 
                      heatmap=False, get_warp_img=False):
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
        warp_save = warp
        velocity_field = torch.sqrt(torch.mul(torch.pow(warp[:, 0], 2), torch.pow(warp[:, 1], 2)))
        velocity_field = velocity_field - torch.min(velocity_field)
        velocity_field = velocity_field / torch.max(velocity_field)

        # warp[:, 0] = torch.where(velocity_field > 0.2, warp[:, 0], 0)
        # warp[:, 1] = torch.where(velocity_field > 0.2, warp[:, 1], 0)
        warp = grid + warp
        warp_save = grid + warp_save
        shape = warp.shape[2:]

        for i in range(len(shape)):
            warp[:, i, ...] = 2 * (warp[:, i, ...] / (shape[i] - 1) - 0.5)
            warp_save[:, i, ...] = (2 * (warp_save[:, i, ...] / (shape[i] - 1) - 0.5) - 2 * (grid[:, i] / (shape[i] - 1) - 0.5)) * shape[i]

        if len(shape) == 2:
            warp = warp.permute(0, 2, 3, 1)
            warp = warp[..., [1, 0]]
        elif len(shape) == 3:
            warp = warp.permute(0, 2, 3, 4, 1)
            warp = warp[..., [2, 1, 0]]

        if len(shape) == 2:
            warp_save = warp_save.permute(0, 2, 3, 1)
            warp_save = warp_save[..., [1, 0]]
        elif len(shape) == 3:
            warp_save = warp_save.permute(0, 2, 3, 4, 1)
            warp_save = warp_save[..., [2, 1, 0]]

        warp_save = warp_save[0, ...]

        if img is not None:
            img = img
            
            # Get the warpping img
            if get_warp_img:
                new_locs = torch.zeros_like(warp)
                for i in range(len(shape)):
                    new_locs[:, i, ...] = 2 * (warp[:, i, ...] / (shape[i] - 1) - 0.5)
                img_warp = torch.nn.functional.grid_sample(img.unsqueeze(0), new_locs, align_corners=True, mode='bilinear')[0]
            
            # Get the heatmap according to the velocity fields
            if heatmap:
                lengths = np.sqrt(np.square(warp_save[:, :, 0]) + np.square(warp_save[:, :, 1]))
                img_heat = lengths - torch.min(lengths)
                img_heat = img_heat / torch.max(img_heat)
                img_heat = torch.where(img_heat > 0.2, 0.2, img_heat)
                img_heat = img_heat * 4

                img_heat = torch.where(img.permute(2, 1, 0)[:,:,0]> -0.5, img_heat, 0)
                img_heat = img_heat.mul(255)

            # Get the segmentation Result
            if segment_result is not None:
                # Display : mask with grey scale, mask with color 1, mask with color 2 and mask with outside contours
                filterd_masked_bw = np.zeros((256, 256, 1))
                filterd_masked_c1 = np.zeros((256, 256, 3))
                filterd_masked_c2 = np.zeros((256, 256, 3))
                filterd_masked_expand = np.zeros((256, 256, 3))

                # Here for each segmented part
                # for part in range(1, 5):
                seg_part = torch.where(torch.nn.Sigmoid()(segment_result[:, 1, ...]) > 0.5, 1, 0).permute(1, 2, 0)

                # Find the contours from segmentation results
                seg_part_cv2 = np.where(seg_part.numpy() > 0, 255, 0)
                _, threshold = cv2.threshold(np.uint8(seg_part_cv2), 127, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # countours is a python list
                
                # Find the largest countours
                if len(contours) > 0:
                    area = []
                    for k in range(len(contours)):
                        area.append(cv2.contourArea(contours[k]))
                    max_idx = np.argmax(np.array(area))

                    cv2.drawContours(filterd_masked_c1, contours, max_idx, (83, 253, 254), cv2.FILLED)
                    cv2.drawContours(filterd_masked_c2, contours, max_idx, (253, 83, 254), cv2.FILLED)
                    cv2.drawContours(filterd_masked_bw, contours, max_idx, 255, cv2.FILLED)
                    
                    filterd_masked_bw = self.dilate_mask(filterd_masked_bw, 10)
                    expanded_contours, _ = cv2.findContours(np.uint8(filterd_masked_bw), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    if len(expanded_contours) > 0:
                        area = []
                        for k in range(len(expanded_contours)):
                            area.append(cv2.contourArea(expanded_contours[k]))
                        max_idx_expanded_contours = np.argmax(np.array(area))
                        cv2.drawContours(filterd_masked_expand, expanded_contours, max_idx_expanded_contours, (83, 253, 254), cv2.FILLED)

                        # cnt_max = expanded_contours[max_idx_expanded_contours]
                        # rect = cv2.minAreaRect(cnt_max)
                        # box = cv2.boxPoints(rect)
                        # box = [np.int0(box)]
                        # # left top / left bottom / right bottom / right top : formulated as Y, X
                        # box[0][0] = [box[0][0][0]+10, box[0][0][1]-10]
                        # box[0][1] = [box[0][1][0]+10, box[0][1][1]-10]
                        # box[0][2] = [box[0][2][0]+10, box[0][2][1]+10]
                        # box[0][3] = [box[0][3][0]+10, box[0][3][1]+10]
                        # cv2.drawContours(filterd_masked_expand_square, box, -1, (255, 255, 255), cv2.FILLED)

                # img_ref = img.permute(2, 1, 0)
                # warp_save[:,:,0] = torch.where(img_ref[:,:,0]<-0.998, 0, warp_save[:,:,0])
                # warp_save[:,:,1] = torch.where(img_ref[:,:,0]<-0.998, 0, warp_save[:,:,1])
                # velocity_lengths = torch.sqrt(torch.square(warp_save[:, :, 0]) + torch.square(warp_save[:, :, 1])).transpose(0, 1).unsqueeze(-1).repeat(1, 1, 3)
                
                filterd_masked_expand = torch.from_numpy(filterd_masked_expand)
                filterd_masked_all = (filterd_masked_expand-filterd_masked_c1+filterd_masked_c2).transpose(0, 1).numpy().astype(np.uint8)

            if heatmap:
                plt.imshow(img.permute(1, 2, 0).add(1.0).mul(127.5).detach().cpu().numpy(), cmap='gray', vmin=0, vmax=255)
                plt.imshow(img_heat.unsqueeze(0).permute(2, 1, 0).detach().cpu().numpy(), cmap='jet', vmin=0, vmax=255, alpha=0.4)
            else:
                # img = torch.flip(img, dims=[2]) 
                plt.imshow(img.permute(1, 2, 0).add(1.0).mul(127.5).detach().cpu().numpy(), cmap='gray', vmin=0, vmax=255)
                if segment_result is not None:
                    plt.imshow(filterd_masked_all, vmin=0, vmax=255, alpha=0.4)

        if show_axis is False:
            plt.axis('off')
        ax = plt.gca()
        ax.set_aspect('equal')

        # This code is for the mesh drawing
        #
        # warp = warp[0, ...].numpy()
        # for row in range(0, warp.shape[0], interval):
        #     plt.plot(warp[row, :, 1], warp[row, :, 0], mark)
        # for col in range(0, warp.shape[1], interval):
        #     plt.plot(warp[:, col, 1], warp[:, col, 0], mark)

        warp_save = warp_save.numpy()
        if segment_result is not None:
            filterd_masked_expand = np.transpose(np.sum((filterd_masked_expand).transpose(0, 1).numpy(), axis=2))
        if img is not None:
            # img_ref = img.permute(2, 1, 0)
            # warp_save[:,:,0] = np.where(img_ref[:,:,0]<-0.99, 0, warp_save[:,:,0])
            # warp_save[:,:,1] = np.where(img_ref[:,:,0]<-0.99, 0, warp_save[:,:,1])
            if segment_result is not None:
                warp_save[:,:,0] = np.where(filterd_masked_expand[:,:] > 0, warp_save[:,:,0], 0)
                warp_save[:,:,1] = np.where(filterd_masked_expand[:,:] > 0, warp_save[:,:,1], 0)

            img_ref = img.permute(2, 1, 0)
            warp_save[:,:,0] = np.where(img_ref[:,:,0]<0.01, 0, warp_save[:,:,0])
            warp_save[:,:,1] = np.where(img_ref[:,:,0]<0.01, 0, warp_save[:,:,1])

        plt.quiver(grids[0][::4, ::4], grids[1][::4, ::4], warp_save[::4, ::4, 0], warp_save[::4, ::4, 1], units='xy', scale_units='xy', angles='xy', color='r', scale=1/2)

        return plt
    
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
    
    def dilate_mask(self, mask, kernel_size):
        kernel  = np.ones((kernel_size, kernel_size), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=1)
        return dilated

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

    def vector_to_rgb(self, angle, absolute):
        """Get the rgb value for the given `angle` and the `absolute` value

        Parameters
        ----------
        angle : float
            The angle in radians
        absolute : float
            The absolute value of the gradient
        
        Returns
        -------
        array_like
            The rgb value as a tuple with values [0..1]
        """
        max_abs = np.max(absolute)
        # normalize angle
        angle = angle % (2 * np.pi)
        if angle < 0:
            angle += 2 * np.pi

        return matplotlib.colors.hsv_to_rgb((angle / 2 / np.pi, 
                                             absolute / max_abs, 
                                             absolute / max_abs))

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
            name='PHHK-Dataset-Deep-Tag-original',
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

    Eval(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EchoNet")
    parser.add_argument('--latent-dim', type=int, default=64, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=tuple, default=(128, 128, 16), help='Image height and width (default: (112, 112 ,16))')
    parser.add_argument('--image-channels', type=int, default=1, help='Number of channels of images (default: 3)')
    parser.add_argument('--patch-size', type=int, default=(16, 16, 16), help='Patch height and width (default: 8)')
    parser.add_argument('--blurring', type=bool, default=False, help='Whether blur the image')
    parser.add_argument('--max_sample_rate', type=int, default=1, help='The sampling rate for the video')
    parser.add_argument('--num-heads', type=int, default=8, help='The number of head of multiscale attention (default: 8)')
    parser.add_argument('--num-layers', type=int, default=2, help='The number of transformer layers')

    parser.add_argument('--mask-size', type=int, default=8, help='The size of mask patch (default: 16)')
    parser.add_argument('--mask-ratio', type=float, default=0.7, help='The ratio of masking area in an image (default: 0.75)')
    parser.add_argument('--selected-view', type=list, default=['4'], help='The selected view from dataset')
    parser.add_argument('--dataset-path', type=str, default='/home/jyangcu/Dataset/PH_HK_image', help='Path to data (default: /data)')
    parser.add_argument('--batch-size', type=int, default=1, help='Input batch size for training (default: 6)')
    
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs to train (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=5e-4, help='Learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.99, help='Adam beta param (default: 0.999)')
    parser.add_argument('--clip-grad', type=bool, default=True, help='perform gradient clipping in training (default: False)')

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
