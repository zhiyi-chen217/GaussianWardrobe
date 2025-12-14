from typing import Any, Mapping
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d import ops
from network.network import ImplicitNetwork as ImplicitNet
import trimesh
import os
import config
class DeformationGraph(nn.Module):
    def __init__(self, opt=None):
        super().__init__()
        self.deformation_graph = ImplicitNet(**opt)
        mode = config.opt["mode"]
        self.simplified_mesh = trimesh.load_mesh(os.path.join(config.opt[mode]["data"]["data_dir"], 
                                                            config.opt.get("smpl_pos_map", "smpl_pos_map") + "_cloth", "simplified_lower.ply"))
        self.update_deformation_nodes(self.simplified_mesh)

    def update_deformation_nodes(self, simplified_mesh):
        """
        Here the deformation nodes should be progressively updated based on the new cloth template
        """
        self.deformation_graph_verts = torch.from_numpy(simplified_mesh.vertices).float().cuda().detach().unsqueeze(0)
        self.cano_mesh = simplified_mesh

    def forward(self, cond, nodes=None):
        if nodes == None:
            nodes = self.deformation_graph_verts
        n_batch = cond.shape[0]
        _, n_joint, n_dim = nodes.shape
        y = torch.zeros((n_batch, n_joint, n_dim))
        nodes_input, _ = torch.broadcast_tensors(nodes, y)
        transformation = self.deformation_graph(nodes_input, cond)  # predict the 6 DoF
        rot = transformation[:, :, :3] * np.pi # TODO try to use the quaternion representation
        trans = transformation[:, :, 3:]

        rot_mat = batch_rodrigues(rot.reshape(n_batch*n_joint, 3)) # from axis-angle to rotation matrix
        # rot_mat = rot_mat.reshape(n_batch, n_joint, 3, 3)
        # rot_mat = euler2rotmat(rot)
        transform_mat = to_transform_mat(rot_mat, trans.reshape(n_batch*n_joint, 3, 1))
        transform_mat = transform_mat.reshape(n_batch, n_joint, 4, 4)
        return transform_mat


def batch_rodrigues(axisang):
    # This function is borrowed from https://github.com/MandyMo/pytorch_HMR/blob/master/src/util.py#L37
    # axisang N x 3
    axisang_norm = torch.norm(axisang + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(axisang_norm, -1)
    axisang_normalized = torch.div(axisang, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * axisang_normalized], dim=1)
    rot_mat = quat2mat(quat)
    # rot_mat = rot_mat.view(rot_mat.shape[0], 9)
    return rot_mat


def quat2mat(quat):
    """
    This function is borrowed from https://github.com/MandyMo/pytorch_HMR/blob/master/src/util.py#L50
    Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [batch_size, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [batch_size, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:,
                                                   2], norm_quat[:,
                                                       3]

    batch_size = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy,
        w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz,
        w2 - x2 - y2 + z2
    ],
        dim=1).view(batch_size, 3, 3)
    return rotMat


def skinning(x, w, tfs, inverse=False, return_T=False):
    """Linear blend skinning
    Args:
        x (tensor): canonical points. shape: [B, N, D]
        w (tensor): conditional input. [B, N, J]
        tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
    Returns:
        x (tensor): skinned points. shape: [B, N, D]
    """
    x_h = F.pad(x, (0, 1), value=1.0)

    if inverse:
        # p:n_point, n:n_bone, i,k: n_dim+1
        w_tf = torch.einsum("bpn,bnij->bpij", w, tfs)
        x_h = torch.einsum("bpij,bpj->bpi", w_tf.inverse(), x_h)
    else:
        x_h = torch.einsum("bpn,bnij,bpj->bpi", w, tfs, x_h)
        w_tf = None
    if return_T:
        return x_h[:, :, :3], w_tf
    else:
        return x_h[:, :, :3]


def to_transform_mat(R, t):
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def euler2rotmat(euler):
    # the similar function in pytorch3d.transforms has significant math error compared to this one
    # don't know why
    sx = torch.sin(euler[:, 0]).view((-1, 1))
    sy = torch.sin(euler[:, 1]).view((-1, 1))
    sz = torch.sin(euler[:, 2]).view((-1, 1))
    cx = torch.cos(euler[:, 0]).view((-1, 1))
    cy = torch.cos(euler[:, 1]).view((-1, 1))
    cz = torch.cos(euler[:, 2]).view((-1, 1))

    mat_flat = torch.hstack([cy * cz,
                             sx * sy * cz - sz * cx,
                             sy * cx * cz + sx * sz,
                             sz * cy,
                             sx * sy * sz + cx * cz,
                             sy * sz * cx - sx * cz,
                             -sy,
                             sx * cy,
                             cx * cy])
    return mat_flat.view((-1, 3, 3))