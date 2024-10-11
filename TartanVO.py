import torch
import torch.nn as nn
import os

import cv2
import numpy as np
import pypose as pp

from Network.VONet import VONet

from dense_ba import scale_from_disp_flow
from Datasets.transformation import tartan2kitti_pypose, cvtSE3_pypose

np.set_printoptions(precision=4, suppress=True, threshold=10000)


class TartanVO(nn.Module):
    def __init__(self, vo_model_name=None, pose_model_name=None, flow_model_name=None, stereo_model_name=None,
                    device_id=0, correct_scale=True, fix_parts=(), use_kitti_coord=True):
        
        super(TartanVO, self).__init__()
        
        self.device_id = device_id
        self.correct_scale = correct_scale
        self.use_kitti_coord = use_kitti_coord
        # the output scale factor
        self.pose_std = torch.tensor([0.13, 0.13, 0.13, 0.013, 0.013, 0.013]).cuda(self.device_id)

        self.vonet = VONet(fix_parts=fix_parts)

        # load the whole model
        if vo_model_name is not None and vo_model_name != "":
            # print('Loading vo network...')
            self.load_model(self.vonet, vo_model_name)
        # can override part of the model
        if flow_model_name is not None and flow_model_name != "" and flow_model_name != 'GIM':
            # print('Loading flow network...')
            self.load_model(self.vonet.flowNet, flow_model_name)
        if pose_model_name is not None and pose_model_name != "":
            # print('Loading pose network...')
            self.load_model(self.vonet.flowPoseNet, pose_model_name)
        if stereo_model_name is not None and stereo_model_name != "":
            # print('Loading stereo network...')
            self.load_model(self.vonet.stereoNet, stereo_model_name)

        self.flowNet_model = flow_model_name

        if self.flowNet_model == 'GIM':
            self.vonet.update_GIM()

            # load state dict
            # weights path
            ckpt = 'gim_dkm_100h.ckpt'
            checkpoints_path = os.path.join('models', ckpt)
            state_dict = torch.load(checkpoints_path, map_location='cpu', weights_only=False)
            if 'state_dict' in state_dict.keys(): state_dict = state_dict['state_dict']
            for k in list(state_dict.keys()):
                if k.startswith('model.'):
                    state_dict[k.replace('model.', '', 1)] = state_dict.pop(k)
                if 'encoder.net.fc' in k:
                    state_dict.pop(k)
            self.vonet.GIM.load_state_dict(state_dict)
        
        self.vonet = self.vonet.cuda(self.device_id)
        # self.vonet = torch.compile(self.vonet).cuda(self.device_id)


    def load_model(self, model, modelname):
        pretrain_dict = torch.load(modelname, map_location='cuda:%d'%self.device_id)
        model_dict = model.state_dict()

        loadin_dict = {}
        for k, v in pretrain_dict.items():
            for kk, vv in model_dict.items():
                if (k.endswith(kk) or kk.endswith(k)) and v.size () == vv.size():
                    loadin_dict[kk] = v

        # print('model_dict:')
        # for kk in model_dict:
        #     print(kk, model_dict[kk].shape)
        # print('pretrain:')
        # for k in pretrain_dict:
        #     print(k, pretrain_dict[k].shape)

        if 0 == len(loadin_dict):
            raise Exception("Could not load model from %s." % (modelname), "load_model")

        for kk in model_dict.keys():
            if kk not in loadin_dict:
                print("! [load_model] Key {} in model but not in {}!".format(k, modelname))
                # if k.endswith('weight'):
                #     print('\tinit with kaiming_normal_')
                #     w = torch.rand_like(model_dict[k])
                #     nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
                # else:
                #     print('\tinit to zeros')
                #     w = torch.zeros_like(model_dict[k])
                # loadin_dict[k] = w

        model_dict.update(loadin_dict)
        model.load_state_dict(model_dict)

        del pretrain_dict
        del loadin_dict

        return model


    def forward(self, sample, is_train=True, given_scale=None):
        self.vonet.train() if is_train else self.vonet.eval()
        with torch.set_grad_enabled(is_train):

            ############################## init inputs ######################################################################   
            nb = True
            img0 = sample['img0'].cuda(self.device_id, non_blocking=nb)
            img1 = sample['img1'].cuda(self.device_id, non_blocking=nb)
            intrinsic = sample['intrinsic'].cuda(self.device_id, non_blocking=nb)

            img0_norm = sample['img0_norm'].cuda(self.device_id, non_blocking=nb)
            img0_r_norm = sample['img0_r_norm'].cuda(self.device_id, non_blocking=nb)
            intrinsic_calib = sample['intrinsic_calib']
            baseline = torch.linalg.norm(sample['extrinsic'][:, :3], dim=1)
            precalc_flow = sample['flow'] if 'flow' in sample else None

            ############################## forward vonet ######################################################################   
            flow, disp, pose = self.vonet(img0, img1, img0_norm, img0_r_norm, intrinsic, flowNet_model=self.flowNet_model)
            pose = pose * self.pose_std # The output is normalized during training, now scale it back
            flow = flow.detach()
            disp = disp.detach()

            res = {}

            if given_scale is not None:
                trans = torch.nn.functional.normalize(pose[:, :3], dim=1) * given_scale.view(-1, 1)
                pose = torch.cat([trans, pose[:, 3:]], dim=1)

            elif not self.correct_scale:
                ############################## recover scale from stereo ######################################################################   

                if precalc_flow is None:
                    flow *= 5   # scale flow pridiction to pixel level
                else:
                    flow = precalc_flow
                
                disp *= 50/4    # scale disparity pridiction to pixel level

                # from Datasets.utils import save_images, warp_images

                # img0_warp = warp_images('temp', img1, flow)
                # save_images('temp', img0, prefix='', suffix='_orig', fx=1/4, fy=1/4)
                # save_images('temp', img1, prefix='', suffix='_x', fx=1/4, fy=1/4)

                # img0_r = sample['img0_r']
                # disp_warp = warp_images('temp2', img0_r, -disp)
                # save_images('temp2', img0, prefix='', suffix='_orig', fx=1/4, fy=1/4)
                # save_images('temp2', img0_r, prefix='', suffix='_x', fx=1/4, fy=1/4)

                # print('flow', torch.min(flow), torch.max(flow), torch.mean(flow))
                # print('disp', torch.min(disp), torch.max(disp), torch.mean(disp), torch.median(disp))
                
                pose_ENU_SE3 = tartan2kitti_pypose(pose)    # convert to ENU coordinate

                ############################## detect edges as mask ######################################################################   
                img0_np = img0.cpu().numpy()
                img0_np = img0_np.transpose(0, 2, 3, 1)
                img0_np = (img0_np*255).astype(np.uint8)
                edge = []
                for i in range(img0_np.shape[0]):
                    im = cv2.resize(img0_np[i], None, fx=1/4, fy=1/4)
                    e = cv2.Canny(im, 50, 100)
                    e = cv2.dilate(e, np.ones((5,5), np.uint8))
                    e = e > 0
                    edge.append(e)
                edge = torch.from_numpy(np.stack(edge)).cuda(self.device_id)

                ############################## calculate scale ######################################################################   
                scale, mask, depth, depth_mask = [], [], [], []
                for i in range(pose.shape[0]):
                    fx, fy, cx, cy = intrinsic_calib[i] / 4
                    disp_th_dict = {'kitti':5, 'tartanair':1, 'euroc':1}
                    s, z, m, dm = scale_from_disp_flow(disp[i], flow[i], pose_ENU_SE3[i], fx, fy, cx, cy, baseline[i], 
                                                    mask=edge[i], disp_th=disp_th_dict[sample['datatype'][i]])
                    scale.append(s)
                    mask.append(m)
                    depth.append(z)
                    depth_mask.append(dm)
                scale = torch.stack(scale)
                mask = torch.stack(mask)
                depth = torch.stack(depth)
                depth_mask = torch.stack(depth_mask)

                res['flow'] = flow
                res['disp'] = disp
                res['mask'] = mask
                res['depth'] = depth
                res['depth_mask'] = depth_mask
                res['baseline'] = baseline[0]
                res['intrinsic'] = intrinsic_calib[0] / 4
                
                trans = torch.nn.functional.normalize(pose[:, :3], dim=1) * scale.view(-1, 1)
                pose = torch.cat([trans, pose[:, 3:]], dim=1)
                
            else:
                ############################## recover scale from GT ######################################################################   
                motion_tar = sample['motion']
                scale = torch.norm(motion_tar[:, :3], dim=1).cuda(self.device_id)

                trans = torch.nn.functional.normalize(pose[:, :3], dim=1) * scale.view(-1, 1)
                pose = torch.cat([trans, pose[:, 3:]], dim=1)

            if self.use_kitti_coord:
                pose = tartan2kitti_pypose(pose)
            else:
                pose = cvtSE3_pypose(pose)
            res['motion'] = pose

            return res


    def pred_flow(self, img0, img1):
        img0 = img0.cuda(self.device_id)
        img1 = img1.cuda(self.device_id)

        batched = len(img0.shape) == 4
        if not batched:
            img0 = img0.unsqueeze(0)
            img1 = img1.unsqueeze(0)

        flow, _ = self.vonet.flowNet(torch.cat([img0, img1], dim=1))
        flow = flow[0] * 5

        if not batched:
            flow = flow.squeeze(0)

        return flow


    def join_flow(self, flow_to_join):
        height, width = flow_to_join[0].shape[-2:]

        u_lin = torch.linspace(0, width-1, width)
        v_lin = torch.linspace(0, height-1, height)
        u, v = torch.meshgrid(u_lin, v_lin, indexing='xy')
        uv = torch.stack([u, v]).cuda(self.device_id)

        x = uv.unsqueeze(0)
        flow_to_join.reverse()
        for f in flow_to_join:
            grid = (f + uv).permute(1, 2, 0).unsqueeze(0)
            grid[..., 0] = grid[..., 0] / width * 2 - 1
            grid[..., 1] = grid[..., 1] / height * 2 - 1
            x = torch.nn.functional.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

        x = x.squeeze(0)
        zero_mask = torch.logical_and(x[0]==0, x[1]==0).repeat(2, 1, 1)
        x = torch.where(zero_mask, -1, x)

        return x - uv
