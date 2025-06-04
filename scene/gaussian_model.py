#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        # def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
        #     L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        #     actual_covariance = L @ L.transpose(1, 2)
        #     symm = strip_symmetric(actual_covariance)
        #     return symm
        def build_covariance_from_scaling_rotation(diag:torch.Tensor, offdiag:torch.Tensor) -> torch.Tensor:
            # input : diag (N, 6), offdiag (N, 15)
            assert diag.shape[1] == 6 and offdiag.shape[1] == 15, "diag must have shape (N, 6)"
            L = torch.diag_embed(diag) # (N, 6, 6)
            dim = diag.shape[1] # 6
            tril_indices = torch.tril_indices(dim, dim, offset=-1, device=diag.device) # (2, 15)
            L[:, tril_indices[0], tril_indices[1]] = offdiag
            
            cov = torch.bmm(L, L.transpose(-1, -2)) # 6D covariance LL^T (N, 6, 6)
        
            cov_p = cov[:, :3, :3] # (N, 3, 3)
            cov_pd = cov[:, :3, 3:] # (N, 3, 3)
            cov_d = cov[:, 3:, 3:] # (N, 3, 3)
            cov_d_inv = torch.linalg.inv(cov_d) # (N, 3, 3)
            
            cov_cond_full = cov_p - torch.bmm(torch.bmm(cov_pd, cov_d_inv), cov_pd.transpose(1, 2)) # 3D conditional covariance (N, 3, 3)
            cov_cond = torch.stack([cov_cond_full[:, 0, 0].abs(), cov_cond_full[:, 0, 1], cov_cond_full[:, 0, 2], cov_cond_full[:, 1, 1].abs(), cov_cond_full[:, 1, 2], cov_cond_full[:, 2, 2].abs()], dim=1) # (N, 6)
            return cov_cond, cov
        
        self.scaling_activation = lambda x: torch.exp(x)
        self.scaling_inverse_activation = lambda x: torch.log(torch.abs(x+1e-6))

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = lambda x: torch.sigmoid(x) * 2.0 - 1.0 # torch.nn.functional.normalize
        self.rotation_inverse_activation = lambda x: inverse_sigmoid(torch.clip((x+1.0)/2.0, min=1e-6, max=1.0-1e-6))


    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._normal = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0) # diag
        self._rotation = torch.empty(0) # offdiag
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._normal,
            self._features_dc,
            self._features_rest,
            self._scaling, # diag
            self._rotation, # offdiag
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._normal,
        self._features_dc, 
        self._features_rest,
        self._scaling, # diag
        self._rotation, # offdiag
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum,
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
    
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling) # diag
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation) # offdiag
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_normal(self):
        return self._normal
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        # return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
        return self.covariance_activation(self.get_scaling, self.get_rotation) # diag, offdiag -> 3D conditional covariance (N, 6), 6D covariance (N, 6, 6)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
    
    ### modified
    def slice_gaussian(self, d, lambda_opa=0.35):
        dim = 3
        cov_cond, cov = self.get_covariance() # (N, 6), (N, 6, 6)
        cov_p = cov[:, :dim, :dim] # (N, 3, 3)
        cov_pd = cov[:, :dim, dim:] # (N, 3, 3)
        cov_d = cov[:, dim:, dim:] # (N, 3, 3)
        cov_d_inv = torch.linalg.inv(cov_d) # (N, 3, 3)
        
        mu_p = self.get_xyz
        mu_d = self.get_normal
        diff_d = d - mu_d # (N, 3)
        
        mu_cond = mu_p + torch.bmm(torch.bmm(cov_pd, cov_d_inv), diff_d.unsqueeze(-1)).squeeze(-1) # (N, 3)
        alpha_cond = self.get_opacity * torch.exp(- lambda_opa * torch.einsum('bi,bij,bj->b', diff_d, cov_d_inv, diff_d).unsqueeze(-1)) # (N, 1)
        
        return mu_cond, cov_cond, alpha_cond # (N, 3), (N, 6), (N, 1)

    ### cov_cond -> s, R
    def extract_SR(self, cov_cond):
        cov_mat = torch.zeros(cov_cond.shape[0], 3, 3, device=cov_cond.device)
        cov_mat[:, 0, 0] = cov_cond[:, 0]
        cov_mat[:, 0, 1] = cov_mat[:, 1, 0] = cov_cond[:, 1]
        cov_mat[:, 0, 2] = cov_mat[:, 2, 0] = cov_cond[:, 2]
        cov_mat[:, 1, 1] = cov_cond[:, 3]
        cov_mat[:, 1, 2] = cov_mat[:, 2, 1] = cov_cond[:, 4]
        cov_mat[:, 2, 2] = cov_cond[:, 5]
        U, D, _ = torch.linalg.svd(cov_mat)
    
        scale = torch.diag_embed(torch.sqrt(D))
        rotation = U.clone()
        det_R = torch.det(rotation)
        rotation[:, :, 2] = rotation[:, :, 2] * det_R.sign().unsqueeze(-1)

        return scale, rotation # (N, 3, 3), (N, 3, 3)
    
    ### (N, 3, 3) -> (N, 4)
    def rotation_to_quaternion(self, rotation):
        quat = torch.zeros((rotation.shape[0], 4), device=rotation.device, dtype=rotation.dtype) # (N, 4)

        trace = rotation[:, 0, 0] + rotation[:, 1, 1] + rotation[:, 2, 2]  # (N,)

        # Case 1: trace > 0
        cond1 = trace > 0
        s1 = torch.sqrt(trace[cond1] + 1.0) * 2
        quat[cond1, 0] = 0.25 * s1
        quat[cond1, 1] = (rotation[cond1, 2, 1] - rotation[cond1, 1, 2]) / s1
        quat[cond1, 2] = (rotation[cond1, 0, 2] - rotation[cond1, 2, 0]) / s1
        quat[cond1, 3] = (rotation[cond1, 1, 0] - rotation[cond1, 0, 1]) / s1

        # Case 2: rotation[0,0] is the largest diagonal term
        cond2 = (rotation[:, 0, 0] > rotation[:, 1, 1]) & (rotation[:, 0, 0] > rotation[:, 2, 2]) & ~cond1
        s2 = torch.sqrt(1.0 + rotation[cond2, 0, 0] - rotation[cond2, 1, 1] - rotation[cond2, 2, 2]) * 2
        quat[cond2, 0] = (rotation[cond2, 2, 1] - rotation[cond2, 1, 2]) / s2
        quat[cond2, 1] = 0.25 * s2
        quat[cond2, 2] = (rotation[cond2, 0, 1] + rotation[cond2, 1, 0]) / s2
        quat[cond2, 3] = (rotation[cond2, 0, 2] + rotation[cond2, 2, 0]) / s2

        # Case 3: rotation[1,1] is the largest diagonal term
        cond3 = (rotation[:, 1, 1] > rotation[:, 2, 2]) & ~cond1 & ~cond2
        s3 = torch.sqrt(1.0 + rotation[cond3, 1, 1] - rotation[cond3, 0, 0] - rotation[cond3, 2, 2]) * 2
        quat[cond3, 0] = (rotation[cond3, 0, 2] - rotation[cond3, 2, 0]) / s3
        quat[cond3, 1] = (rotation[cond3, 0, 1] + rotation[cond3, 1, 0]) / s3
        quat[cond3, 2] = 0.25 * s3
        quat[cond3, 3] = (rotation[cond3, 1, 2] + rotation[cond3, 2, 1]) / s3

        # Case 4: rotation[2,2] is the largest diagonal term
        cond4 = ~cond1 & ~cond2 & ~cond3
        s4 = torch.sqrt(1.0 + rotation[cond4, 2, 2] - rotation[cond4, 0, 0] - rotation[cond4, 1, 1]) * 2
        quat[cond4, 0] = (rotation[cond4, 1, 0] - rotation[cond4, 0, 1]) / s4
        quat[cond4, 1] = (rotation[cond4, 0, 2] + rotation[cond4, 2, 0]) / s4
        quat[cond4, 2] = (rotation[cond4, 1, 2] + rotation[cond4, 2, 1]) / s4
        quat[cond4, 3] = 0.25 * s4

        return quat

    ### (N, 3, 3), (N, 3, 3) -> (N, 3), (N, 4)
    def matrix_to_vector(self, scale, rotation):
        # rotation = torch.tensor([[ 0.9773, -0.1745,  0.1194], [ 0.1794,  0.9837, -0.0112], [-0.1161,  0.0323,  0.9927]], device=scale.device).unsqueeze(0)
        scale_vec = torch.stack([scale[:, 0, 0], scale[:, 1, 1], scale[:, 2, 2]], dim=1) # (N, 3)
        rotation_vec = self.rotation_to_quaternion(rotation) # (N, 4)
        # print("original rotation from SVD", rotation[0])
        # print("original rotation > quaternion > rotation", build_rotation(rotation_vec)[0])
        return scale_vec, rotation_vec

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001) # (N,)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3) # (N, 3)
        # rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda") # (N, 4)
        # rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")) # (N, 1)

        # self.diag (N, 6) 
        # self.diag[:, :3] : diagonal of positional covariance, so intialize with distance-based scales
        # self.diag[:, 3:] : diagonal of directional covariance, so initialize with one
        diag = self.scaling_inverse_activation(torch.cat([torch.ones([fused_point_cloud.shape[0], 3], device="cuda") * scales, torch.ones([fused_point_cloud.shape[0], 3], device="cuda")], dim=-1)) # (N, 6)
        # self.offdiag (N, 15) : initialize with zero
        offdiag = self.rotation_inverse_activation(torch.zeros([fused_point_cloud.shape[0], 15], device="cuda")) # (N, 15)
        
        normal = torch.randn(fused_point_cloud.shape[0], 3, device="cuda")
        normal = normal / normal.norm(dim=1, keepdim=True)
        
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._normal = nn.Parameter(normal.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        # Not to modify original 3DGS code much, 
        # set self._scaling as diag of covariance (N, 6)
        # set self._rotation as offdiag of covariance (N, 15)
        self._scaling = nn.Parameter(diag.requires_grad_(True))
        self._rotation = nn.Parameter(offdiag.requires_grad_(True))
        # self._scaling = nn.Parameter(scales.requires_grad_(True))
        # self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._normal], 'lr': training_args.direction_lr_init, "name": "normal"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"}, # diag
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"} # offdiag
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.normal_scheduler_args = get_expon_lr_func(lr_init=training_args.direction_lr_init,
                                                    lr_final=training_args.direction_lr_final,
                                                    max_steps=training_args.direction_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
            elif param_group["name"] == "normal":
                lr = self.normal_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]): # diag
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]): # offdiag
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        # normals = np.zeros_like(xyz)
        normal = self._normal.detach().cpu().numpy()
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy() # diag
        rotation = self._rotation.detach().cpu().numpy() # offdiag

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normal, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01)) # 6DGS # 3DGS: 0.01
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        normal = np.stack((np.asarray(plydata.elements[0]["nx"]),
                        np.asarray(plydata.elements[0]["ny"]),
                        np.asarray(plydata.elements[0]["nz"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._normal = nn.Parameter(torch.tensor(normal, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)) # diag
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)) # offdiag
        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._normal = optimizable_tensors["normal"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"] # diag
        self._rotation = optimizable_tensors["rotation"] # offdiag

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_normal, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        d = {"xyz": new_xyz,
        "normal": new_normal,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d) # add new attribute to existing attribute
        self._xyz = optimizable_tensors["xyz"]
        self._normal = optimizable_tensors["normal"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"] # diag
        self._rotation = optimizable_tensors["rotation"] # offdiag

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, scale, rotation, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(scale, dim=1).values > self.percent_dense*scene_extent)

        stds = scale[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_normal = self.get_normal[selected_pts_mask].repeat(N, 1) # do not change direction after densification
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N)) # scale for scale, self._scaling for diag
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1) # offdiag
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_normal, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool))) # remove existing selected points and maintain new N * selected points
        self.prune_points(prune_filter)

    def densify_and_clone(self, scale, rotation, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(scale, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_normal = self._normal[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask] # diag
        new_rotation = self._rotation[selected_pts_mask] # offdiag

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz, new_normal, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        cov_cond = self.get_covariance()[0] # (N, 6)
        scale_mat, rotation_mat = self.extract_SR(cov_cond) # (N, 3, 3), (N, 3, 3)
        scale, rotation = self.matrix_to_vector(scale_mat, rotation_mat) # (N, 3), (N, 4)

        grads = self.xyz_gradient_accum / self.denom # do not reflect self.normal_gradient_accum into grads 
        grads[grads.isnan()] = 0.0  

        self.tmp_radii = radii
        self.densify_and_clone(scale, rotation, grads, max_grad, extent) # grads (N',1)
        cov_cond = self.get_covariance()[0] # (N', 6)
        scale_mat, rotation_mat = self.extract_SR(cov_cond) # (N', 3, 3), (N', 3, 3)
        scale, rotation = self.matrix_to_vector(scale_mat, rotation_mat) # (N', 3), (N', 4)

        self.densify_and_split(scale, rotation, grads, max_grad, extent) # (N", 1)
        cov_cond = self.get_covariance()[0] # (N", 6)
        scale_mat, rotation_mat = self.extract_SR(cov_cond) # (N", 3, 3), (N", 3, 3)
        scale, rotation = self.matrix_to_vector(scale_mat, rotation_mat) # (N", 3), (N", 4)

        prune_mask = (self.get_opacity < min_opacity).squeeze()

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = scale.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
