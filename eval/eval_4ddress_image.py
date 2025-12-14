# To compute FID, first install pytorch_fid
# pip install pytorch-fid

import os
import cv2 as cv
from tqdm import tqdm
import shutil

from eval.score import *

cam_id = 18
no_reg_dir = '/local/home/zhiychen/AnimatableGaussain/test_results/00185_no_virtual_bone/layered_full_stage_1_less_opacity_loss/cam_0000/batch_239040/both/rgb_map'
no_seg_dir = '/local/home/zhiychen/AnimatableGaussain/test_results/00185_no_virtual_bone/layered_full_stage_1_less_opacity_loss/cam_0000/batch_239040/both/rgb_map'
ours_dir = '/local/home/zhiychen/AnimatableGaussain/test_results/00185_virtual_bones/virtual_bone_body_loss/cam_0000/batch_358560/both/rgb_map'
# posevocab_dir = './test_results/subject00/posevocab/testing__cam_%03d/rgb_map' % cam_id
# tava_dir = './test_results/subject00/tava/cam_%03d' % cam_id
no_pene_dir = '/local/home/zhiychen/AnimatableGaussain/test_results/00185_no_virtual_bone/layered_full_stage_1_less_opacity_loss/cam_0000/batch_239040/both/rgb_map' 
# slrf_dir = './test_results/subject00/slrf/cam_%03d' % cam_id
gt_dir = '/data/zhiychen/AnimatableGaussain/test_data/multiviewRGC/4d_dress/00185/Inner/0004/images' 
mask_dir = '/data/zhiychen/AnimatableGaussain/test_data/multiviewRGC/4d_dress/00185/Inner/0004/masks'

frame_list = list(range(0, 310, 1))

if __name__ == '__main__':
    ours_metrics = Metrics()
    # posevocab_metrics = Metrics()
    # slrf_metrics = Metrics()
    no_reg_metrics = Metrics()
    # tava_metrics = Metrics()
    no_pene_metrics = Metrics()
    no_seg_metrics = Metrics()


    os.makedirs('./tmp_quant', exist_ok=True)
    shutil.rmtree('./tmp_quant')
    os.makedirs('./tmp_quant/ours', exist_ok = True)
    os.makedirs('./tmp_quant/AG', exist_ok=True)
    # os.makedirs('./tmp_quant/posevocab', exist_ok = True)
    # os.makedirs('./tmp_quant/slrf', exist_ok = True)
    os.makedirs('./tmp_quant/layga', exist_ok = True)
    # os.makedirs('./tmp_quant/tava', exist_ok = True)
    os.makedirs('./tmp_quant/gt', exist_ok = True)

    for frame_id in tqdm(frame_list):
        ours_img = (cv.imread(ours_dir + '/%08d.png' % frame_id, cv.IMREAD_UNCHANGED) / 255.).astype(np.float32)
        # posevocab_img = (cv.imread(posevocab_dir + '/%08d.jpg' % frame_id, cv.IMREAD_UNCHANGED) / 255.).astype(np.float32)
        # slrf_img = (cv.imread(slrf_dir + '/%08d.png' % frame_id, cv.IMREAD_UNCHANGED) / 255.).astype(np.float32)
        # tava_img = (cv.imread(tava_dir + '/%d.jpg' % frame_id, cv.IMREAD_UNCHANGED) / 255.).astype(np.float32)
        no_pene_img = (cv.imread(no_pene_dir + '/%08d.png' % frame_id, cv.IMREAD_UNCHANGED) / 255.).astype(np.float32)
        no_reg_img = (cv.imread(no_reg_dir + '/%08d.png' % frame_id, cv.IMREAD_UNCHANGED) / 255.).astype(np.float32)
        no_seg_img = (cv.imread(no_seg_dir + '/%08d.png' % frame_id, cv.IMREAD_UNCHANGED) / 255.).astype(np.float32)
        gt_img = (cv.imread(gt_dir + '/%05d.png' % frame_id, cv.IMREAD_UNCHANGED) / 255.).astype(np.float32)
        mask_img = cv.imread(mask_dir + '/%05d.png' % frame_id, cv.IMREAD_UNCHANGED) > 128
        gt_img[~mask_img] = 0
        ours_img[~mask_img] = 0
        no_reg_img[~mask_img] = 0
        no_pene_img[~mask_img] = 0
        no_seg_img[~mask_img] = 0
        ours_img_cropped, no_reg_img_cropped, no_pene_img_cropped, no_seg_img_cropped, gt_img_cropped = \
            crop_image(
                mask_img[:, :, 0],
                512,
                ours_img,
                # posevocab_img,
                # slrf_img,
                # tava_img,
                no_reg_img,
                no_pene_img,
                no_seg_img,
                gt_img
            )

        cv.imwrite('./tmp_quant/ours/%08d.png' % frame_id, (ours_img_cropped * 255).astype(np.uint8))
        cv.imwrite('./tmp_quant/no_pene/%08d.png' % frame_id, (no_pene_img_cropped * 255).astype(np.uint8))
        # cv.imwrite('./tmp_quant/posevocab/%08d.png' % frame_id, (posevocab_img_cropped * 255).astype(np.uint8))
        # cv.imwrite('./tmp_quant/slrf/%08d.png' % frame_id, (slrf_img_cropped * 255).astype(np.uint8))
        # cv.imwrite('./tmp_quant/tava/%08d.png' % frame_id, (tava_img_cropped * 255).astype(np.uint8))
        cv.imwrite('./tmp_quant/no_reg/%08d.png' % frame_id, (no_reg_img_cropped * 255).astype(np.uint8))
        cv.imwrite('./tmp_quant/no_seg/%08d.png' % frame_id, (no_seg_img_cropped * 255).astype(np.uint8))
        cv.imwrite('./tmp_quant/gt/%08d.png' % frame_id, (gt_img_cropped * 255).astype(np.uint8))
        if ours_img is not None:
            ours_psnr = compute_psnr(ours_img, gt_img)
            ours_metrics.psnr += ours_psnr
            ours_metrics.ssim += compute_ssim(ours_img, gt_img)
            ours_lpip = compute_lpips(ours_img_cropped, gt_img_cropped)
            
            ours_metrics.lpips += ours_lpip
            # ours_metrics.iou += compute_iou(ours_img, gt_img)
            # ours_metrics.acc += compute_accuracy(ours_img, gt_img)
            ours_metrics.count += 1

            # ours_lpips.append(ours_lpip)
            # ours_psnrs.append(ours_psnr)

        if no_pene_img is not None:
            no_pene_psnr = compute_psnr(no_pene_img, gt_img)
            no_pene_metrics.psnr += no_pene_psnr
            no_pene_metrics.ssim += compute_ssim(no_pene_img, gt_img)
            ag_lpip = compute_lpips(no_pene_img_cropped, gt_img_cropped)
            # AG_metrics.iou += compute_iou(AG_img, gt_img)
            # AG_metrics.acc += compute_accuracy(AG_img, gt_img)
            no_pene_metrics.lpips += ag_lpip
            no_pene_metrics.count += 1


            # ag_psnrs.append(ag_psnr)
            # ag_lpips.append(ag_lpip)
        # if posevocab_img is not None:
        #     posevocab_metrics.psnr += compute_psnr(posevocab_img, gt_img)
        #     posevocab_metrics.ssim += compute_ssim(posevocab_img, gt_img)
        #     posevocab_metrics.lpips += compute_lpips(posevocab_img_cropped, gt_img_cropped)
        #     posevocab_metrics.count += 1
        #
        # if slrf_img is not None:
        #     slrf_metrics.psnr += compute_psnr(slrf_img, gt_img)
        #     slrf_metrics.ssim += compute_ssim(slrf_img, gt_img)
        #     slrf_metrics.lpips += compute_lpips(slrf_img_cropped, gt_img_cropped)
        #     slrf_metrics.count += 1
        #
        if no_reg_img is not None:
            no_reg_psnr = compute_psnr(no_reg_img, gt_img)
            no_reg_metrics.psnr += no_reg_psnr
            no_reg_metrics.ssim += compute_ssim(no_reg_img, gt_img)
            # layga_metrics.iou += compute_iou(layga_img, gt_img)
            # layga_metrics.acc += compute_accuracy(layga_img, gt_img)
            no_reg_lpip = compute_lpips(no_reg_img_cropped, gt_img_cropped)
            no_reg_metrics.lpips += no_reg_lpip
            # layga_lpips.append(layga_lpip)
            # layga_psnrs.append(layag_psnr)
            no_reg_metrics.count += 1
        if no_seg_img is not None:
            no_seg_psnr = compute_psnr(no_seg_img, gt_img)
            no_seg_metrics.psnr += no_seg_psnr
            no_seg_metrics.ssim += compute_ssim(no_seg_img, gt_img)
            # layga_metrics.iou += compute_iou(layga_img, gt_img)
            # layga_metrics.acc += compute_accuracy(layga_img, gt_img)
            no_seg_lpip = compute_lpips(no_seg_img_cropped, gt_img_cropped)
            no_seg_metrics.lpips += no_seg_lpip
            # layga_lpips.append(layga_lpip)
            # layga_psnrs.append(layag_psnr)
            no_seg_metrics.count += 1
        #
        # if tava_img is not None:
        #     tava_metrics.psnr += compute_psnr(tava_img, gt_img)
        #     tava_metrics.ssim += compute_ssim(tava_img, gt_img)
        #     tava_metrics.lpips += compute_lpips(tava_img_cropped, gt_img_cropped)
        #     tava_metrics.count += 1
        #
    print('Ours metrics: ', ours_metrics)
    print('no_pene metrics: ', no_pene_metrics)
    # print('PoseVocab metrics: ', posevocab_metrics)
    # print('SLRF metrics: ', slrf_metrics)
    print('no_seg metrics: ', no_seg_metrics)
    print('no_reg metrics: ', no_reg_metrics)
    # print('TAVA metrics: ', tava_metrics)

    print('--- Ours ---')
    os.system('python -m pytorch_fid --device cuda {} {}'.format('./tmp_quant/ours', './tmp_quant/gt'))
    print('--- AG ---')
    os.system('python -m pytorch_fid --device cuda {} {}'.format('./tmp_quant/AG', './tmp_quant/gt'))
    # print('--- PoseVocab ---')
    # os.system('python -m pytorch_fid --device cuda {} {}'.format('./tmp_quant/posevocab', './tmp_quant/gt'))
    # print('--- SLRF ---')
    # os.system('python -m pytorch_fid --device cuda {} {}'.format('./tmp_quant/slrf', './tmp_quant/gt'))
    print('--- ARAH ---')
    os.system('python -m pytorch_fid --device cuda {} {}'.format('./tmp_quant/layga', './tmp_quant/gt'))
    # print('--- TAVA ---')
    # os.system('python -m pytorch_fid --device cuda {} {}'.format('./tmp_quant/tava', './tmp_quant/gt'))
    #
    #
