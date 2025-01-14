import glob
import os
import pickle
import shutil
import numpy as np
import cv2 as cv
import sys
sys.path.append("/local/home/zhiychen/AnimatableGaussain")
from utils.sh_utils import RGB2GRAY

SUBJECT = 127
CAMERA = 76
TAKES = [2, 4, 5, 6, 7, 8]
LAYER = "Inner"
SURFACE_LABEL = ['skin', 'upper', 'lower', 'hair', 'shoe', 'outer']
SURFACE_LABEL_COLOR = np.array([[128, 128, 128], [255, 128, 0], [128, 0, 255], [180, 50, 50], [50, 180, 50], [0, 128, 255]])
SURFACE_LABEL_GRAY = RGB2GRAY(SURFACE_LABEL_COLOR)
MASK_LABEL = dict(zip(SURFACE_LABEL, SURFACE_LABEL_GRAY))

def read_all_png_camera(data_dir, png_type="images", png_prefix="", takes=TAKES, camera=CAMERA):
    png_list = []
    for take in takes:
        if png_type != "labels":
            png_list += glob.glob(
                os.path.join(data_dir, '%05d' % SUBJECT, LAYER, f'Take{take}', "Capture", png_type, '%04d' % camera, f'{png_prefix}*.png'),
                recursive=True)
        else:
            png_list += glob.glob(
                os.path.join(data_dir, '%05d' % SUBJECT, LAYER, f'Take{take}', "Capture", '%04d' % camera, png_type, f'{png_prefix}*.png'),
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
def copy_png_to_folder(new_data_dir, png_list, png_type="images", camera=CAMERA):
    if not os.path.exists(os.path.join(new_data_dir, '%05d' % SUBJECT, LAYER, '%04d' % camera, png_type)):
        os.makedirs(os.path.join(new_data_dir, '%05d' % SUBJECT, LAYER, '%04d' % camera, png_type))
    for ind in range(len(png_list)):
        new_path = os.path.join(new_data_dir, '%05d' % SUBJECT, LAYER, '%04d' % camera, png_type, f'{ind:05d}.png')
        # img = cv.imread(png_list[ind], cv.IMREAD_GRAYSCALE)
        # new_img = np.full((img.shape[0], img.shape[1], 3), 128)
        # new_img[img != 255] = 0
        # new_img[img == MASK_LABEL["lower"]] = 255
        # new_img[img == MASK_LABEL["upper"]] = 255
        
        # cv.imwrite(new_path, new_img) 
        shutil.copy(png_list[ind], new_path)

def copy_filter_label(new_data_dir, png_list, camera=CAMERA, png_type="labels"):
    if not os.path.exists(os.path.join(new_data_dir, '%05d' % SUBJECT, LAYER, '%04d' % camera, png_type)):
        os.makedirs(os.path.join(new_data_dir, '%05d' % SUBJECT, LAYER, '%04d' % camera, png_type))
    for ind in range(len(png_list)):
        new_path = os.path.join(new_data_dir, '%05d' % SUBJECT, LAYER, '%04d' % camera, png_type, f'{ind:05d}.png')
        img = cv.imread(png_list[ind], cv.IMREAD_GRAYSCALE)
        new_img = np.full((img.shape[0], img.shape[1], 3), 0)
        # new_img[img != 255] = 0
        new_img[img == MASK_LABEL["lower"]] = 255
        new_img[img == MASK_LABEL["upper"]] = 255
        
        cv.imwrite(new_path, new_img) 

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


if __name__ == '__main__':
    data_dir = "/mnt/work/Clothes4D/SMLCHumans/"
    new_data_dir = "/data/zhiychen/AnimatableGaussain/train_data/multiviewRGC/4d_dress/"
    label_data_dir = "/data/zhiychen/4ddress/"
    # image_list = read_all_png_camera(data_dir, png_prefix="capture")
    # mask_list = read_all_png_camera(data_dir, png_type="masks", png_prefix="mask")
    # pose_list = read_all_pose(data_dir)
    # copy_png_to_folder(new_data_dir, image_list)
    # copy_png_to_folder(new_data_dir, mask_list, png_type="masks")
    # copy_pose_to_folder(new_data_dir, pose_list)
    # combine_pose_to_npz(new_data_dir)
    label_list = read_all_png_camera(label_data_dir, png_type="labels", png_prefix="label")
    copy_filter_label(new_data_dir, label_list, png_type="labels_cloth")
    print("end")