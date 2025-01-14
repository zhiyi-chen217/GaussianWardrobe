import glob
import os
import pickle
import shutil

import numpy
import numpy as np
import torch
import trimesh
import cv2 as cv

from pytorch3d.renderer import PerspectiveCameras, RasterizationSettings, MeshRasterizer

from pytorch3d.structures import Meshes
from tqdm import tqdm

from utils.renderer import Renderer

SUBJECT = 127
CAMERA = 76
TAKES = [i for i in range(17, 19)]
LAYER = "Inner"
def read_all_png_camera(data_dir, png_type="images", takes=TAKES, camera=CAMERA, layer=LAYER):
    png_list = []
    for take in takes:
        png_list += glob.glob(
            os.path.join(data_dir, '%05d' % SUBJECT, layer, f'Take{take}', "Capture", png_type, '%04d' % camera, '*.png'),
            recursive=True)
    png_list.sort()
    return png_list
def read_all_pose(data_dir):
    pose_list = []
    for take in TAKES:
        pose_list += glob.glob(
            os.path.join(data_dir, '%05d' % SUBJECT, LAYER, f'Take{take}', "SMPLX", '*.pkl'),
            recursive=True)
    pose_list.sort()
    return pose_list
def copy_png_to_folder(new_data_dir, png_list, png_type="images", camera=CAMERA, layer=LAYER):
    if not os.path.exists(os.path.join(new_data_dir, '%05d' % SUBJECT,  layer, '%04d' % camera, png_type)):
        os.makedirs(os.path.join(new_data_dir, '%05d' % SUBJECT, layer, '%04d' % camera, png_type))
    for ind in range(len(png_list)):
        new_path = os.path.join(new_data_dir, '%05d' % SUBJECT, layer, '%04d' % camera, png_type, f'{ind:05d}.png')
        shutil.copy(png_list[ind], new_path)

def copy_pose_to_folder(new_data_dir, pose_list):
    if not os.path.exists(os.path.join(new_data_dir, '%05d' % SUBJECT, LAYER, "pose")):
        os.makedirs(os.path.join(new_data_dir, '%05d' % SUBJECT, LAYER, "pose"))
    for ind in range(len(pose_list)):
        new_path = os.path.join(new_data_dir, '%05d' % SUBJECT, LAYER, "pose", f'{ind:05d}.pkl')
        shutil.copy(pose_list[ind], new_path)
def combine_pose_to_npz(new_data_dir):
    regstr_list = []
    regstr_list += glob.glob(
        os.path.join(new_data_dir, '%05d' % SUBJECT, LAYER, "pose", '*.pkl'),
        recursive=True)
    regstr_list.sort()
    smpl_data = None
    for scan in regstr_list:
        try:
            with open(scan, 'rb') as f:
                regstr = pickle.load(f)
        except:
            print('corrupted npz')

        if smpl_data is None:
            del regstr['leye_pose']
            del regstr['reye_pose']
            smpl_data = regstr
            # expand dimension to concat
            for k in smpl_data.keys():
                smpl_data[k] = np.expand_dims(smpl_data[k], 0)
        else:
            for k in smpl_data.keys():
                if k == "betas":
                    continue
                smpl_data[k] = np.concatenate((smpl_data[k], np.expand_dims(regstr[k], 0)), axis=0)
    np.savez(os.path.join(new_data_dir, '%05d' % SUBJECT, LAYER, 'smpl_params.npz'), **smpl_data)
# load pytorch3d cameras from parameters: intrinsics, extrinsics
def load_pytorch_cameras(camera_params, camera_list, image_shape):
    # init camera_dict
    camera_dict = dict()
    # process all camera within camera_list
    for camera_id in camera_list:
        # assign camera intrinsic and extrinsic matrices
        intrinsic = torch.tensor((camera_params[camera_id]["intrinsics"]), dtype=torch.float32).cuda()
        extrinsic = torch.tensor(camera_params[camera_id]["extrinsics"], dtype=torch.float32).cuda()
        # assign camera image size
        image_size = torch.tensor([image_shape[0], image_shape[1]], dtype=torch.float32).unsqueeze(0).cuda()

        # assign camera parameters
        f_xy = torch.cat([intrinsic[0:1, 0], intrinsic[1:2, 1]], dim=0).unsqueeze(0)
        p_xy = intrinsic[:2, 2].unsqueeze(0)
        R = extrinsic[:, :3].unsqueeze(0)
        T = extrinsic[:, 3].unsqueeze(0)
        # coordinate system adaption to PyTorch3D
        R[:, :2, :] *= -1.0
        # camera position in world space -> world position in camera space
        T[:, :2] *= -1.0
        R = torch.transpose(R, 1, 2)  # row-major
        # assign Pytorch3d PerspectiveCameras
        camera_dict[camera_id] = PerspectiveCameras(focal_length=f_xy, principal_point=p_xy, R=R, T=T, in_ndc=False, image_size=image_size).cuda()
    # assign Pytorch3d RasterizationSettings
    raster_settings = RasterizationSettings(image_size=image_shape, blur_radius=0.0, faces_per_pixel=1, max_faces_per_bin=80000)
    return camera_dict, raster_settings

def render_mesh(mesh_path, camera_path, new_data_dir):
    mesh_list = glob.glob(
        os.path.join(mesh_path, '*.ply'),
        recursive=True)
    os.makedirs(os.path.join(new_data_dir, '%05d' % SUBJECT, LAYER, '%04d' % CAMERA, "body_masks"), exist_ok=True)
    mesh_list.sort()
    cameras = pickle.load(open(camera_path, 'rb'))
    camera_dict, raster_settings = load_pytorch_cameras(cameras, ['%04d' % CAMERA], (1280, 940))
    capture_rasts = MeshRasterizer(cameras=camera_dict['%04d' % CAMERA], raster_settings=raster_settings)
    for ind in  tqdm(range(len(mesh_list))):
        mesh_path = mesh_list[ind]
        mesh = trimesh.load(mesh_path)
        scan_mesh = Meshes(torch.tensor(mesh.vertices, dtype=torch.float32).unsqueeze(0),
                           torch.tensor(mesh.faces, dtype=torch.long).unsqueeze(0)).cuda()
        # render capture_mask(h, w) and capture_mask_image(h, w)
        capture_mask = capture_rasts(scan_mesh).pix_to_face[0, :, :, 0] > -1
        capture_mask = capture_mask.unsqueeze(-1).expand(-1, -1, 3)
        cv.imwrite(os.path.join(new_data_dir, '%05d' % SUBJECT, LAYER, '%04d' % CAMERA, "body_masks", f'{ind:05d}.png'), (capture_mask * 255).detach().cpu().numpy())
if __name__ == '__main__':
    data_dir = "/home/Mocap2/Clothes4D/SMLCHumans/"
    new_data_dir = "/home/zhiychen/Desktop/train_data/multiviewRGC/4d_dress/"
    mesh_path = "/home/zhiychen/Desktop/train_data/multiviewRGC/4d_dress/00127/Inner/body_mesh"
    camera_path = "/home/zhiychen/Desktop/train_data/multiviewRGC/4d_dress/00127/Inner/cameras.pkl"
    render_mesh(mesh_path, camera_path, new_data_dir)
    # image_list = read_all_png_camera(data_dir)
    # mask_list = read_all_png_camera(data_dir, png_type="masks")
    # pose_list = read_all_pose(data_dir)
    # copy_png_to_folder(new_data_dir, image_list)
    # copy_png_to_folder(new_data_dir, mask_list, png_type="masks")
    # copy_pose_to_folder(new_data_dir, pose_list)
    # combine_pose_to_npz(new_data_dir)
    print("end")