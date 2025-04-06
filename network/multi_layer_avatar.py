import platform
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch3d.ops
import pytorch3d.transforms
import cv2 as cv
from sympy import rotations

import config
from network.avatar import AvatarNet
from network.styleunet.dual_styleunet import DualStyleUNet
from gaussians.gaussian_model import GaussianModel
from gaussians.gaussian_renderer import render3
from utils.net_util import read_map_mask
class MultiGaussianModel:
    def __init__(self, model_dict, layers):
        self.cano_gaussian_dict = {}
        self.layers = layers
        for layer in layers:
            self.cano_gaussian_dict[layer] = model_dict[layer].cano_gaussian_model
        

    @property
    def get_scaling(self):
        scaling_l = []
        for layer in self.layers:
            scaling_l.append(self.cano_gaussian_dict[layer].get_scaling)
        return torch.concat(scaling_l, dim=0)

    @property
    def get_rotation(self):
        rotation_l = []
        for layer in self.layers:
            rotation_l.append(self.cano_gaussian_dict[layer].get_rotation)
        return torch.concat(rotation_l, dim=0)

    @property
    def get_xyz(self):
        xyz_l = []
        for layer in self.layers:
            xyz_l.append(self.cano_gaussian_dict[layer].get_xyz)
        return torch.concat(xyz_l, dim=0)

    @property
    def get_opacity(self):
        opacity_l = []
        for layer in self.layers:
            opacity_l.append(self.cano_gaussian_dict[layer].get_opacity)
        return torch.concat(opacity_l, dim=0)


class MultiLAvatarNet(nn.Module):
    def __init__(self, opt, layers, data_dir=None):
        super(MultiLAvatarNet, self).__init__()
        self.layers = layers
        self.layers_nn = nn.ModuleDict()
        self.init_points = []
        self.lbs = []
        for layer in layers:
            self.layers_nn[layer] = AvatarNet(opt, layer, data_dir=data_dir)
            self.init_points.append(self.layers_nn[layer].cano_smpl_map[self.layers_nn[layer].cano_smpl_mask])
            self.lbs.append(self.layers_nn[layer].lbs)
        self.mode = config.opt["mode"]
        self.init_points = torch.concat(self.init_points, dim=0)
        self.lbs = torch.concat(self.lbs, dim=0)
        self.max_sh_degree = 0
        self.cano_gaussian_model = MultiGaussianModel(self.layers_nn, self.layers)
        self.upper_body_mask = self.layers_nn["body"].cano_smpl_mask & (~self.layers_nn["cloth"].cano_smpl_mask)
        self.selected_body_gaussian = self.upper_body_mask[self.layers_nn["body"].cano_smpl_mask]
        self.upper_cloth_mask = self.layers_nn["cloth"].cano_smpl_mask & self.layers_nn["body"].cano_smpl_mask
        self.selected_cloth_gaussian = self.upper_cloth_mask[self.layers_nn["cloth"].cano_smpl_mask]
        self.cover = opt.get("cover", False)
        self.data_dir = config.opt[self.mode]['data']['data_dir'] if data_dir is None else data_dir
        _, cano_smpl_hand_mask = read_map_mask(self.data_dir + f'/{self.layers_nn["body"].smpl_pos_map}/cano_smpl_hand_map.exr')
        
        self.body_hand_idx = cano_smpl_hand_mask[self.layers_nn["body"].cano_smpl_mask]
        self.original_body_cover_scales = None
        self.original_body_cover_rotations = None
        # set up select gaussian for body
        self.layers_nn["body"].selected_gaussian = self.selected_body_gaussian

    def transform_cano2live(self, gaussian_vals, lbs, items):
        pt_mats = torch.einsum('nj,jxy->nxy', lbs, items['cano2live_jnt_mats'])
        gaussian_vals['positions'] = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], gaussian_vals['positions']) + pt_mats[..., :3, 3]
        rot_mats = pytorch3d.transforms.quaternion_to_matrix(gaussian_vals['rotations'])
        rot_mats = torch.einsum('nxy,nyz->nxz', pt_mats[..., :3, :3], rot_mats)
        gaussian_vals['rotations'] = pytorch3d.transforms.matrix_to_quaternion(rot_mats)
        return gaussian_vals

    def get_lbs(self, layers=None):
        if layers is None:
            return self.lbs
        layer_lbs = []
        for layer in layers:
            layer_lbs.append(self.layers_nn[layer].lbs)
        return torch.concat(layer_lbs, dim=0)
    
    def get_init_points(self, layers=None):
        if layers is None:
            return self.init_points
        layer_init_points = []
        for layer in layers:
            layer_init_points.append(self.layers_nn[layer].cano_smpl_map[self.layers_nn[layer].cano_smpl_mask])
        return torch.concat(layer_init_points, dim=0)
    def upscale_opacity(self, gaussian_vals):
        gaussian_vals["opacity"][gaussian_vals["opacity"] >= 0.8] = 1

    def filter_gaussian(self, gaussian_vals, mask):
        gaussian_vals_filtered = {}
        for key in gaussian_vals.keys():
            if key == "max_sh_degree":
                gaussian_vals_filtered[key] = gaussian_vals[key]
            # elif key == "opacity":
            #     gaussian_vals_filtered[key] = torch.ones_like(gaussian_vals[key]).to(config.device)
            else:
                gaussian_vals_filtered[key] = gaussian_vals[key][mask]
        return gaussian_vals_filtered
    def render_filtered(self, items, bg_color = (0., 0., 0.), use_pca = -1, layers=None, upscale_opacity=False, only_gaussian=False):
        bg_color = torch.from_numpy(np.asarray(bg_color)).to(torch.float32).to(config.device)
        for layer in self.layers:
            items["smpl_pos_map"][layer] = items["smpl_pos_map"][layer].squeeze(0)

        if len(self.layers) == 2:
            gaussian_body_vals = self.layers_nn["body"].render(items, only_gaussian=True)
            gaussian_cloth_vals = self.layers_nn["cloth"].render(items, only_gaussian=True)
            gaussian_vals = {}

            if layers == ["body"]:
                gaussian_vals = gaussian_body_vals
            elif layers == ["cloth"]:
                gaussian_vals = gaussian_cloth_vals
            elif layers == ["lower"]:
                _, upper_mask = read_map_mask(self.data_dir + f'/{self.layers_nn["cloth"].smpl_pos_map}/cano_smpl_segment_map.exr')
                selected_cloth = upper_mask[self.layers_nn["cloth"].cano_smpl_mask]
                gaussian_vals = self.filter_gaussian(gaussian_cloth_vals, selected_cloth)
            elif layers == ["upper"]:
                _, upper_mask = read_map_mask(self.data_dir + f'/{self.layers_nn["cloth"].smpl_pos_map}/cano_smpl_segment_map.exr')
                lower_mask = ~upper_mask
                selected_cloth = lower_mask[self.layers_nn["cloth"].cano_smpl_mask]
                gaussian_vals = self.filter_gaussian(gaussian_cloth_vals, selected_cloth)
            else:
                render_body_ret = render3(
                    gaussian_body_vals,
                    bg_color,
                    items['extr'],
                    items['intr'],
                    items['img_w'],
                    items['img_h']
                )
                render_cloth_ret = render3(
                    gaussian_cloth_vals,
                    bg_color,
                    items['extr'],
                    items['intr'],
                    items['img_w'],
                    items['img_h']
                )


                render_body_ret["depth"][render_body_ret["depth"] == 0] = 10
                render_cloth_ret["depth"][render_cloth_ret["depth"] == 0] = 10
                depth_map = torch.zeros_like(render_cloth_ret["mask"])
                depth_map[:, :, :] = render_cloth_ret["depth"].clone() 
                depth_map -= torch.min(depth_map)
                depth_map /= torch.max(depth_map)
                render_body_mask = (render_body_ret["mask"] > 0.8)[0, :, :] & (render_body_ret["depth"] < (render_cloth_ret["depth"] - 0.05)).squeeze(0) 
                # kernel = np.ones((4, 4), np.uint8)
                # mask_dilate= cv.dilate(render_body_mask.detach().float().cpu().numpy().copy(), kernel)
                # mask_closed = cv.erode(mask_dilate.copy(), kernel)
                # render_body_mask = torch.from_numpy(render_body_mask).to(config.device).squeeze(0) > 0
                rgb_map = render_cloth_ret["render"].clone()
                rgb_map[:, render_body_mask] = render_body_ret["render"][:, render_body_mask]
                mask_map = render_cloth_ret["mask"].clone()
                mask_map[:, render_body_mask] = render_body_ret["mask"][:, render_body_mask]
            
                
                ret = {
                    'rgb_map': rgb_map.permute(1, 2, 0),
                    'mask_map': mask_map.permute(1, 2, 0),
                    'depth_map': depth_map.permute(1, 2, 0),
                    'body_mask': (render_body_ret["mask"] > 0.8)[0, :, :],
                    'cloth_mask_080': (render_cloth_ret["mask"] > 0.8)[0, :, :],
                    'cloth_mask_098': render_cloth_ret["mask"] ,
                    'body_visible_mask': render_body_mask,
                    'cloth_rgb': render_cloth_ret["render"].permute(1, 2, 0),
                    'body_rgb': render_body_ret["render"].permute(1, 2, 0),
                }
                return ret

            render_ret = render3(
                gaussian_vals,
                bg_color,
                items['extr'],
                items['intr'],
                items['img_w'],
                items['img_h']
            )
            rgb_map = render_ret['render'].permute(1, 2, 0)
            mask_map = render_ret['mask'].permute(1, 2, 0)

            
            ret = {
                'rgb_map': rgb_map,
                'mask_map': mask_map,
            }
            
        else:
            with torch.no_grad():
                gaussian_body_vals = self.layers_nn["body"].render(items, only_gaussian=True)
            gaussian_cloth_vals = self.layers_nn["cloth"].render(items, only_gaussian=True)
            gaussian_outer_vals = self.layers_nn["outer"].render(items, only_gaussian=True)
            gaussian_vals = {}

            # use constant color for label image
            body_color = torch.zeros_like(gaussian_body_vals["colors"]).to(config.device)
            gaussian_body_vals["label_colors"] = body_color
            cloth_color = torch.zeros_like(gaussian_cloth_vals["colors"]).to(config.device)
            gaussian_cloth_vals["label_colors"] = cloth_color
            outer_color = torch.full_like(gaussian_outer_vals["colors"], 1.0).to(config.device)
            gaussian_outer_vals["label_colors"] = outer_color


            for key in gaussian_cloth_vals.keys():
                if key == "max_sh_degree":
                    gaussian_vals[key] = gaussian_cloth_vals[key]
                else:
                    gaussian_vals[key] = torch.concat([gaussian_body_vals[key], gaussian_cloth_vals[key], gaussian_outer_vals[key]], dim=0)

            if layers == ["body"]:
                gaussian_vals = gaussian_body_vals
            elif layers == ["cloth"]:
                gaussian_vals = gaussian_cloth_vals
            elif layers == ["outer"]:
                gaussian_vals = gaussian_outer_vals
                
            
            render_ret = render3(
                gaussian_vals,
                bg_color,
                items['extr'],
                items['intr'],
                items['img_w'],
                items['img_h']
            )
            rgb_map = render_ret['render'].permute(1, 2, 0)
            mask_map = render_ret['mask'].permute(1, 2, 0)

            # render again the label image
            gaussian_vals["colors"] = gaussian_vals["label_colors"]
            render_ret_label = render3(
                gaussian_vals,
                bg_color,
                items['extr'],
                items['intr'],
                items['img_w'],
                items['img_h']
            )
            label_rgb_map = render_ret_label['render'].permute(1, 2, 0)

            ret = {
                'rgb_map': rgb_map,
                'mask_map': mask_map,
                'cloth_offset': gaussian_cloth_vals["offset"],
                'outer_offset': gaussian_outer_vals["offset"],
                "label_rgb_map": label_rgb_map,
            }
        return ret


    def render(self, items, bg_color = (0., 0., 0.), use_pca = -1, layers=None, upscale_opacity=False, only_gaussian=False):
        """
        Note that no batch index in items.
        """
        bg_color = torch.from_numpy(np.asarray(bg_color)).to(torch.float32).to(config.device)
        for layer in self.layers:
            items["smpl_pos_map"][layer] = items["smpl_pos_map"][layer].squeeze(0)

        if len(self.layers) == 2:
            gaussian_body_vals = self.layers_nn["body"].render(items, only_gaussian=True)
            gaussian_cloth_vals = self.layers_nn["cloth"].render(items, only_gaussian=True)
            gaussian_vals = {}

            # use constant color for label image
            body_color = torch.zeros_like(gaussian_body_vals["colors"]).to(config.device)
            gaussian_body_vals["label_colors"] = body_color
            cloth_color = torch.full_like(gaussian_cloth_vals["colors"], 1.0).to(config.device)
            gaussian_cloth_vals["label_colors"] = cloth_color


            for key in gaussian_cloth_vals.keys():
                if key == "max_sh_degree":
                    gaussian_vals[key] = gaussian_cloth_vals[key]
                else:
                    gaussian_vals[key] = torch.concat([gaussian_body_vals[key], gaussian_cloth_vals[key]], dim=0)

            if layers == ["body"]:
                gaussian_vals = gaussian_body_vals
            elif layers == ["cloth"]:
                gaussian_vals = gaussian_cloth_vals
            elif layers == ["lower"]:
                _, upper_mask = read_map_mask(self.data_dir + f'/{self.layers_nn["cloth"].smpl_pos_map}/cano_smpl_segment_map.exr')
                selected_cloth = upper_mask[self.layers_nn["cloth"].cano_smpl_mask]
                gaussian_vals = self.filter_gaussian(gaussian_cloth_vals, selected_cloth)
            elif layers == ["upper"]:
                _, upper_mask = read_map_mask(self.data_dir + f'/{self.layers_nn["cloth"].smpl_pos_map}/cano_smpl_segment_map.exr')
                lower_mask = ~upper_mask
                selected_cloth = lower_mask[self.layers_nn["cloth"].cano_smpl_mask]
                gaussian_vals = self.filter_gaussian(gaussian_cloth_vals, selected_cloth)
            if only_gaussian:
                return gaussian_vals
            render_ret = render3(
                gaussian_vals,
                bg_color,
                items['extr'],
                items['intr'],
                items['img_w'],
                items['img_h']
            )
            rgb_map = render_ret['render'].permute(1, 2, 0)
            mask_map = render_ret['mask'].permute(1, 2, 0)

            # render again the label image
            gaussian_vals["colors"] = gaussian_vals["label_colors"]
            render_ret_label = render3(
                gaussian_vals,
                bg_color,
                items['extr'],
                items['intr'],
                items['img_w'],
                items['img_h']
            )
            label_rgb_map = render_ret_label['render'].permute(1, 2, 0)
            gaussian_body_vals["colors"] = torch.full_like(gaussian_body_vals["colors"], 1.0).to(config.device)
            render_ret_body = render3(
                gaussian_body_vals,
                bg_color,
                items['extr'],
                items['intr'],
                items['img_w'],
                items['img_h']
            )
            label_body_rgb_map = render_ret_body['render'].permute(1, 2, 0)
            # gaussian_body_vals["colors"] = gaussian_body_vals["gaussian_norm"]
            # render_ret_normal = render3(
            #     gaussian_body_vals,
            #     bg_color,
            #     items['extr'],
            #     items['intr'],
            #     items['img_w'],
            #     items['img_h']
            # )
            # normal_map = render_ret_normal['render'].permute(1, 2, 0)
            
            visible_hand_idx = (gaussian_body_vals["opacity"] > 0.5).squeeze(1) & self.body_hand_idx
            ret = {
                'rgb_map': rgb_map,
                'mask_map': mask_map,
                'offset': gaussian_vals["offset"],
                "label_body_rgb_map": label_body_rgb_map,
                "gaussian_body_norm": gaussian_body_vals["gaussian_norm"],
                "label_rgb_map": label_rgb_map,
                # "normal_map": normal_map,
                "gaussian_cloth_pos": gaussian_cloth_vals["cano_positions"][self.selected_cloth_gaussian],
                "gaussian_cloth_opacity": gaussian_cloth_vals["opacity"],
            }
            
        else:
            with torch.no_grad():
                gaussian_body_vals = self.layers_nn["body"].render(items, only_gaussian=True)
            gaussian_cloth_vals = self.layers_nn["cloth"].render(items, only_gaussian=True)
            gaussian_outer_vals = self.layers_nn["outer"].render(items, only_gaussian=True)
            gaussian_vals = {}

            # use constant color for label image
            body_color = torch.zeros_like(gaussian_body_vals["colors"]).to(config.device)
            gaussian_body_vals["label_colors"] = body_color
            cloth_color = torch.zeros_like(gaussian_cloth_vals["colors"]).to(config.device)
            gaussian_cloth_vals["label_colors"] = cloth_color
            outer_color = torch.full_like(gaussian_outer_vals["colors"], 1.0).to(config.device)
            gaussian_outer_vals["label_colors"] = outer_color


            for key in gaussian_cloth_vals.keys():
                if key == "max_sh_degree":
                    gaussian_vals[key] = gaussian_cloth_vals[key]
                else:
                    gaussian_vals[key] = torch.concat([gaussian_body_vals[key], gaussian_cloth_vals[key], gaussian_outer_vals[key]], dim=0)

            if layers == ["body"]:
                gaussian_vals = gaussian_body_vals
            elif layers == ["cloth"]:
                gaussian_vals = gaussian_cloth_vals
            elif layers == ["lower"]:
                _, upper_mask = read_map_mask(self.data_dir + f'/{self.layers_nn["cloth"].smpl_pos_map}/cano_smpl_segment_map.exr')
                selected_cloth = upper_mask[self.layers_nn["cloth"].cano_smpl_mask]
                gaussian_vals = self.filter_gaussian(gaussian_cloth_vals, selected_cloth)
            elif layers == ["upper"]:
                _, upper_mask = read_map_mask(self.data_dir + f'/{self.layers_nn["cloth"].smpl_pos_map}/cano_smpl_segment_map.exr')
                lower_mask = ~upper_mask
                selected_cloth = lower_mask[self.layers_nn["cloth"].cano_smpl_mask]
                gaussian_vals = self.filter_gaussian(gaussian_cloth_vals, selected_cloth)
            elif layers == ["outer"]:
                gaussian_vals = gaussian_outer_vals
            if only_gaussian:
                return gaussian_vals
                
            
            render_ret = render3(
                gaussian_vals,
                bg_color,
                items['extr'],
                items['intr'],
                items['img_w'],
                items['img_h']
            )
            rgb_map = render_ret['render'].permute(1, 2, 0)
            mask_map = render_ret['mask'].permute(1, 2, 0)

            # render again the label image
            gaussian_vals["colors"] = gaussian_vals["label_colors"]
            render_ret_label = render3(
                gaussian_vals,
                bg_color,
                items['extr'],
                items['intr'],
                items['img_w'],
                items['img_h']
            )
            label_rgb_map = render_ret_label['render'].permute(1, 2, 0)

            ret = {
                'rgb_map': rgb_map,
                'mask_map': mask_map,
                'offset': gaussian_vals["offset"],
                "label_rgb_map": label_rgb_map,
            }
        return ret



    def get_positions(self, pose_map, return_map = False, layers=None):
        if layers is None:
            layers = self.layers
        if return_map:
            positions_l = []
            position_map_l = []
            for layer in layers:
                pose_map_layer = pose_map[layer]
                positions, position_map = self.layers_nn[layer].get_positions(pose_map_layer, return_map)
                positions_l.append(positions)
                position_map_l.append(position_map)
            return torch.concat(positions_l, dim=0), torch.concat(position_map_l, dim=0)
        else:
            positions_l = []
            for layer in layers:
                pose_map_layer = pose_map[layer]
                positions = self.layers_nn[layer].get_positions(pose_map_layer, return_map)
                positions_l.append(positions)

            return torch.concat(positions_l, dim=0)

    def get_others(self, pose_map, layers=None):
        opacity_l = []
        scales_l = []
        rotations_l = []
        if layers is None:
            layers = self.layers
        for layer in layers:
            opacity, scales, rotations = self.layers_nn[layer].get_others(pose_map[layer])
            opacity_l.append(opacity)
            scales_l.append(scales)
            rotations_l.append(rotations)
        return torch.concat(opacity_l, dim=0), torch.concat(scales_l, dim=0), torch.concat(rotations_l, dim=0)

    def get_colors(self, pose_map, items, layers=None):
        colors_l = []
        color_map_l = []
        if layers is None:
            layers = self.layers
        for layer in layers:
            if self.layers_nn[layer].with_viewdirs:
                front_viewdirs, back_viewdirs = self.layers_nn[layer].get_viewdir_feat(items)
            else:
                front_viewdirs, back_viewdirs = None, None
            colors, color_map = self.layers_nn[layer].get_colors(pose_map[layer],
                                                                 front_viewdirs, back_viewdirs)
            colors_l.append(colors)
            color_map_l.append(color_map)
        return torch.concat(colors_l, dim=0), torch.concat(color_map_l, dim=0)


