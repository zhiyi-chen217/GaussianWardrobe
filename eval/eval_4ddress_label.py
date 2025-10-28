# To compute FID, first install pytorch_fid
# pip install pytorch-fid

import os
import cv2 as cv
from tqdm import tqdm
import shutil

from eval.score import *

cam_id = 18
no_pene_dir = '/local/home/zhiychen/AnimatableGaussain/test_results/00134_test/no_body_loss/cam_0009/batch_131223/both/label_pixel_map'
ours_dir = '/local/home/zhiychen/AnimatableGaussain/test_results/00134_test/with_body_loss/cam_0009/batch_131223/both/label_pixel_map'
# posevocab_dir = './test_results/subject00/posevocab/testing__cam_%03d/rgb_map' % cam_id
# tava_dir = './test_results/subject00/tava/cam_%03d' % cam_id
no_reg_dir = '/local/home/zhiychen/AnimatableGaussain/test_results/00134_test/no_reg/cam_0009/batch_124602/both/label_pixel_map' 
no_seg_dir = '/local/home/zhiychen/AnimatableGaussain/test_results/00134_test/no_seg/cam_0009/batch_122402/both/label_pixel_map'
gt_dir = '/data/zhiychen/AnimatableGaussain/test_data/multiviewRGC/4d_dress/00134/Inner/0076/labels_pixel' 
mask_dir = '/data/zhiychen/AnimatableGaussain/test_data/multiviewRGC/4d_dress/00134/Inner/0076/masks'

frame_list = list(range(0, 300, 1))

ours_lpips = []
layga_lpips = []
ag_lpips = []

ours_psnrs = []
layga_psnrs = []
ag_psnrs = []
if __name__ == '__main__':
    ours_metrics = Metrics()
    # posevocab_metrics = Metrics()
    # slrf_metrics = Metrics()
    no_pene_metrics = Metrics()
    # tava_metrics = Metrics()
    no_reg_metrics = Metrics()
    no_seg_metrics = Metrics()


    os.makedirs('./tmp_quant', exist_ok=True)
    shutil.rmtree('./tmp_quant')
    os.makedirs('./tmp_quant/ours', exist_ok = True)
    os.makedirs('./tmp_quant/no_pene', exist_ok=True)
    # os.makedirs('./tmp_quant/posevocab', exist_ok = True)
    # os.makedirs('./tmp_quant/slrf', exist_ok = True)
    os.makedirs('./tmp_quant/no_seg', exist_ok = True)
    os.makedirs('./tmp_quant/no_reg', exist_ok = True)
    os.makedirs('./tmp_quant/gt', exist_ok = True)

    for frame_id in tqdm(frame_list):
        ours_img = (cv.imread(ours_dir + '/label_pixel_map_%08d.png' % frame_id, cv.IMREAD_UNCHANGED) / 255.).astype(np.float32)
        # posevocab_img = (cv.imread(posevocab_dir + '/%08d.jpg' % frame_id, cv.IMREAD_UNCHANGED) / 255.).astype(np.float32)
        # slrf_img = (cv.imread(slrf_dir + '/%08d.png' % frame_id, cv.IMREAD_UNCHANGED) / 255.).astype(np.float32)
        # tava_img = (cv.imread(tava_dir + '/%d.jpg' % frame_id, cv.IMREAD_UNCHANGED) / 255.).astype(np.float32)
        no_pene_img = (cv.imread(no_pene_dir + '/label_pixel_map_%08d.png' % frame_id, cv.IMREAD_UNCHANGED) / 255.).astype(np.float32)
        no_reg_img = (cv.imread(no_reg_dir + '/label_pixel_map_%08d.png' % frame_id, cv.IMREAD_UNCHANGED) / 255.).astype(np.float32)
        no_seg_img = (cv.imread(no_seg_dir + '/label_pixel_map_%08d.png' % frame_id, cv.IMREAD_UNCHANGED) / 255.).astype(np.float32)
        gt_img = (cv.imread(gt_dir + '/%05d.png' % frame_id, cv.IMREAD_UNCHANGED) / 255.).astype(np.float32)
        mask_img = cv.imread(mask_dir + '/%05d.png' % frame_id, cv.IMREAD_UNCHANGED) > 128
        mask_img = mask_img[:,:,0]
        gt_img[~mask_img] = 1
        ours_img[~mask_img] = 1
        no_pene_img[~mask_img] = 1
        no_reg_img[~mask_img] = 1
        no_seg_img[~mask_img] = 1
        # ours_img_cropped, layga_img_cropped, AG_img_cropped, gt_img_cropped = \
        #     crop_image(
        #         mask_img[:, :, 0],
        #         512,
        #         ours_img,
        #         # posevocab_img,
        #         # slrf_img,
        #         # tava_img,
        #         layga_img,
        #         AG_img,
        #         gt_img
        #     )

        cv.imwrite('./tmp_quant/ours/%08d.png' % frame_id, (ours_img * 255).astype(np.uint8))
        cv.imwrite('./tmp_quant/no_reg/%08d.png' % frame_id, (no_reg_img* 255).astype(np.uint8))
        # cv.imwrite('./tmp_quant/posevocab/%08d.png' % frame_id, (posevocab_img_cropped * 255).astype(np.uint8))
        # cv.imwrite('./tmp_quant/slrf/%08d.png' % frame_id, (slrf_img_cropped * 255).astype(np.uint8))
        # cv.imwrite('./tmp_quant/tava/%08d.png' % frame_id, (tava_img_cropped * 255).astype(np.uint8))
        cv.imwrite('./tmp_quant/no_pene/%08d.png' % frame_id, (no_pene_img * 255).astype(np.uint8))
        cv.imwrite('./tmp_quant/no_seg/%08d.png' % frame_id, (no_pene_img * 255).astype(np.uint8))
        cv.imwrite('./tmp_quant/gt/%08d.png' % frame_id, (gt_img * 255).astype(np.uint8))
        ours_img = np.rint(ours_img * 3).astype(np.int32)
        no_pene_img = np.rint(no_pene_img * 3).astype(np.int32)
        no_reg_img = np.rint(no_reg_img * 3).astype(np.int32)
        no_seg_img = np.rint(no_seg_img * 3).astype(np.int32)
        gt_img = np.rint(gt_img * 3).astype(np.int32)
        if ours_img is not None:
            # ours_psnr = compute_psnr(ours_img, gt_img)
            # ours_metrics.psnr += ours_psnr
            # ours_metrics.ssim += compute_ssim(ours_img, gt_img)
            # ours_lpip = compute_lpips(ours_img_cropped, gt_img_cropped)
            
            # ours_metrics.lpips += ours_lpip
            ours_metrics.iou += compute_iou(ours_img, gt_img, classes=[0, 1,2])
            ours_precision = compute_precision(ours_img, gt_img, classes=[0, 1,2])
            ours_recall = compute_recall(ours_img, gt_img,  classes=[0, 1,2])
            ours_metrics.precision += ours_precision
            ours_metrics.recall += ours_recall
            ours_metrics.f1 += compute_f1(precision=ours_precision, recall=ours_recall)
            ours_metrics.count += 1

            # ours_lpips.append(ours_lpip)
            # ours_psnrs.append(ours_psnr)

        if no_pene_img is not None:
            # ag_psnr = compute_psnr(AG_img, gt_img)
            # AG_metrics.psnr += ag_psnr
            # AG_metrics.ssim += compute_ssim(AG_img, gt_img)
            # ag_lpip = compute_lpips(AG_img_cropped, gt_img_cropped)
            no_pene_metrics.iou += compute_iou(no_pene_img, gt_img,)
            no_pene_precision = compute_precision(no_pene_img, gt_img,)
            no_pene_recall = compute_recall(no_pene_img, gt_img,)
            no_pene_metrics.recall += no_pene_recall
            no_pene_metrics.precision += no_pene_precision
            no_pene_metrics.f1 += compute_f1(precision=no_pene_precision, recall=no_pene_recall)
            # AG_metrics.lpips += ag_lpip
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
            # layag_psnr = compute_psnr(layga_img, gt_img)
            # layga_metrics.psnr += layag_psnr
            # layga_metrics.ssim += compute_ssim(layga_img, gt_img)
            no_reg_metrics.iou += compute_iou(no_reg_img, gt_img,)
            no_reg_precision = compute_precision(no_reg_img, gt_img,)
            no_reg_recall = compute_recall(no_reg_img, gt_img, )
            no_reg_metrics.f1 += compute_f1(no_reg_precision, no_reg_recall)
            no_reg_metrics.precision += no_reg_precision
            no_reg_metrics.recall += no_reg_recall
            # layga_lpip = compute_lpips(layga_img_cropped, gt_img_cropped)
            # layga_metrics.lpips += layga_lpip
            # layga_lpips.append(layga_lpip)
            # layga_psnrs.append(layag_psnr)
            no_reg_metrics.count += 1

        if no_seg_img is not None:
            # layag_psnr = compute_psnr(layga_img, gt_img)
            # layga_metrics.psnr += layag_psnr
            # layga_metrics.ssim += compute_ssim(layga_img, gt_img)
            no_seg_metrics.iou += compute_iou(no_seg_img, gt_img,)
            no_seg_precision = compute_precision(no_seg_img, gt_img,)
            no_seg_recall = compute_recall(no_seg_img, gt_img, )
            no_seg_metrics.f1 += compute_f1(no_seg_precision, no_seg_recall)
            no_seg_metrics.precision += no_seg_precision
            no_seg_metrics.recall += no_seg_recall
            # layga_lpip = compute_lpips(layga_img_cropped, gt_img_cropped)
            # layga_metrics.lpips += layga_lpip
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
    print('no_reg metrics: ', no_reg_metrics)
    # print('PoseVocab metrics: ', posevocab_metrics)
    # print('SLRF metrics: ', slrf_metrics)
    print('no_pene metrics: ', no_pene_metrics)
    print('no_seg metrics: ', no_seg_metrics)
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
