# -*- coding: utf-8 -*-
# @Time    : 20/5/16 21:47
# @Author  : qgking
# @Email   : qgking@tju.edu.cn
# @Software: PyCharm
# @Desc    : lung_inwondow_utils.py
import matplotlib
import sys

sys.path.extend(["../../", "../"])
from common.base_utls import *
from common.data_utils import *
import torch
from skimage.transform import resize
from torch.cuda import empty_cache
from torchvision.utils import make_grid


def process_20_covid_lung_inwindow(seg_help, model_seg_lung, covid_test_data,thres_hold=0.6):
    model_seg_lung.eval()
    img_deps = 256
    img_rows = 256
    img_cols = 32
    # pred_dir = 'predict_gz'
    # if isdir(join(seg_help.config.submission_dir, pred_dir)) and len(
    #         os.listdir(join(seg_help.config.submission_dir, pred_dir))) == 20:
    #     seg_help._submit_predict(join(seg_help.config.submission_dir, pred_dir), test_type=3)
    #     return
    for file_path in covid_test_data:
        # print(file_path)
        file_name = basename(file_path)
        scan = np.load(file_path)
        current_test = scan[0]
        # scan_nib = nib.load(
        #     join('../../../medical_data/COVID193DCT/3DCOVIDCT/COVID-19-CT-Seg_20cases', file_name[:-4] + '.nii.gz'))
        x = current_test.shape[0]
        y = current_test.shape[1]
        z = current_test.shape[2]
        score = np.zeros((2, img_deps, img_rows, z), dtype='float32')
        score_num = np.zeros((2, img_deps, img_rows, z), dtype='int16')
        over = math.ceil(z / img_cols)
        h_overlap = math.ceil((over * img_cols - z) / (over - 1 if over > 1 else 1))
        for i in range(0, z - img_cols + 1, img_cols - h_overlap):
            patch = current_test[:, :, i:i + img_cols]
            cropp_img = resize(patch, (img_deps, img_rows, img_cols), order=3, mode='constant',
                               cval=0, clip=True, preserve_range=True, anti_aliasing=False)
            # cropp_tumor = resize(scan[1, :, :, i:i + img_cols], (img_deps, img_rows, img_cols), order=0, mode='edge',
            #                      cval=0, clip=True, preserve_range=True, anti_aliasing=False)

            box_test = torch.from_numpy(np.expand_dims(cropp_img, 0))
            box_test = torch.unsqueeze(box_test, dim=0)
            box_test = box_test.to(seg_help.equipment).float()
            with torch.no_grad():
                patch_test_mask = model_seg_lung(box_test)
            patch_test_mask = torch.softmax(patch_test_mask, dim=1)
            score[:, :, :, i:i + img_cols] += torch.squeeze(patch_test_mask).detach().cpu().numpy()
            score_num[:, :, :, i:i + img_cols] += 1

            # image_save = box_test.transpose(1, 4).contiguous().view(
            #     (box_test.size(0) * box_test.size(-1), box_test.size(1), box_test.size(2), box_test.size(3)))
            # grid = make_grid(image_save, nrow=8, padding=2)
            # save_img = grid.detach().cpu().numpy()
            # visualize(np.transpose(save_img, (1, 2, 0)), join(seg_help.config.tmp_dir, str(i) + "_images"))
            #
            # final_data = torch.from_numpy(cropp_tumor).unsqueeze(dim=0).unsqueeze(dim=0)
            # prob_save = final_data.float().transpose(1, 4).contiguous().view(
            #     (final_data.size(0) * final_data.size(-1), final_data.size(1), final_data.size(2),
            #      final_data.size(3))) / cropp_tumor.max()
            # grid = make_grid(prob_save, nrow=8, padding=2)
            # save_img = grid.detach().cpu().numpy()
            # visualize(np.transpose(save_img, (1, 2, 0)),
            #           join(seg_help.config.tmp_dir, str(i) + "_label"))
            #
            # prob_max = 1 - patch_test_mask[:, 0]
            # prob_save = prob_max.float().unsqueeze(1).transpose(1, 4).contiguous().view(
            #     (prob_max.size(0) * prob_max.size(-1), prob_max.size(0), prob_max.size(1),
            #      prob_max.size(2)))
            # grid = make_grid(prob_save, nrow=8, padding=2)
            # save_img = grid.detach().cpu().numpy()
            # visualize(np.transpose(save_img, (1, 2, 0)), join(seg_help.config.tmp_dir, str(i) + "_predict"))

        score_num = np.where(score_num == 0, 1, score_num)
        score_ = score / (score_num)
        new_shape = np.array([img_deps, img_rows, z], dtype=np.float32)
        resize_factor = new_shape / [x, y, z]
        score = nd.interpolation.zoom(score_[0], 1 / resize_factor, mode='nearest')
        # score = resize(score_[0], (x, y, z), order=3, mode='constant', cval=0, clip=True, preserve_range=True,
        #                anti_aliasing=False)
        score = np.clip(score, 0, 1)
        pred = 1 - score

        coarse_pred = np.where(pred > thres_hold, 1, 0)
        [lung_labels, num] = measure.label(coarse_pred, return_num=True)
        region = measure.regionprops(lung_labels)
        box = []
        for i in range(num):
            box.append(region[i].area)
        label_num_left = box.index(sorted(box, reverse=True)[0]) + 1
        if num == 1:
            lung_labels = np.where((lung_labels == label_num_left), 1, 0)
        else:
            label_num_right = box.index(sorted(box, reverse=True)[1]) + 1
            lung_labels = np.where((lung_labels == label_num_left) | (lung_labels == label_num_right), 1, 0)
        lung_labels = ndimage.binary_fill_holes(lung_labels).astype(int)

        # save_dir = join(seg_help.config.submission_dir, pred_dir)
        # if not isdir(save_dir):
        #     makedirs(save_dir)
        # nib.save(nib.Nifti1Image(lung_labels, affine=scan_nib.get_affine()),
        #          join(save_dir, file_name[:-4] + '.nii.gz'))

        # -------------visualization start------------------
        save_dir = join(seg_help.config.tmp_dir, '20_case_lung')
        if not isdir(save_dir):
            makedirs(save_dir)

        final_data = torch.from_numpy(lung_labels).unsqueeze(dim=0).unsqueeze(dim=0)
        prob_save = final_data.float().transpose(1, 4).contiguous().view(
            (final_data.size(0) * final_data.size(-1), final_data.size(1), final_data.size(2),
             final_data.size(3)))
        grid = make_grid(prob_save, nrow=16, padding=2)
        save_img = grid.detach().cpu().numpy()
        visualize(np.transpose(save_img, (1, 2, 0)),
                  join(save_dir, file_name[:-4] + "_predict_fusion_struct_largest"))

        final_data = torch.from_numpy(scan[1]).unsqueeze(dim=0).unsqueeze(dim=0)
        prob_save = final_data.float().transpose(1, 4).contiguous().view(
            (final_data.size(0) * final_data.size(-1), final_data.size(1), final_data.size(2),
             final_data.size(3))) / np.max(scan[1])
        grid = make_grid(prob_save, nrow=16, padding=2)
        save_img = grid.detach().cpu().numpy()
        visualize(np.transpose(save_img, (1, 2, 0)),
                  join(save_dir, file_name[:-4] + "_label_covid"))
        # final_data = torch.from_numpy(current_test).unsqueeze(dim=0).unsqueeze(
        #     dim=0)
        # prob_save = final_data.float().transpose(1, 4).contiguous().view(
        #     (final_data.size(0) * final_data.size(-1), final_data.size(1), final_data.size(2),
        #      final_data.size(3)))
        # grid = make_grid(prob_save, nrow=16, padding=2)
        # save_img = grid.detach().cpu().numpy()
        # visualize(np.transpose(save_img, (1, 2, 0)), join(save_dir, file_name[:-4] + "_raw_covid"))

        # final_data = torch.from_numpy(pred).unsqueeze(dim=0).unsqueeze(dim=0)
        # prob_save = final_data.float().transpose(1, 4).contiguous().view(
        #     (final_data.size(0) * final_data.size(-1), final_data.size(1), final_data.size(2),
        #      final_data.size(3)))
        # grid = make_grid(prob_save, nrow=16, padding=2)
        # save_img = grid.detach().cpu().numpy()
        # visualize(np.transpose(save_img, (1, 2, 0)),
        #           join(save_dir, file_name[:-4] + "_predict_fusion_struct_prob"))

        # -------------visualization end------------------
    # seg_help._submit_predict(join(seg_help.config.submission_dir, pred_dir), test_type=3)


def process_50_covid_lung_inwindow(seg_help, model_seg_lung,covid_test_data, thres_hold=0.6):
    model_seg_lung.eval()
    img_deps = 256
    img_rows = 256
    img_cols = 32
    # pred_dir = '50_predict_gz'
    # if isdir(join(seg_help.config.submission_dir, pred_dir)) and len(
    #         os.listdir(join(seg_help.config.submission_dir, pred_dir))) == 20:
    #     seg_help._submit_predict(join(seg_help.config.submission_dir, pred_dir), test_type=3)
    #     return
    for file_path in covid_test_data:
        print(file_path)
        file_name = basename(file_path)
        scan = np.load(file_path)
        current_test = scan[0]
        # scan_nib = nib.load(
        #     join('../../../medical_data/COVID193DCT/MosMedData/MosMedData_Processed_250', file_name[:-4] + '.nii.gz'))
        x = current_test.shape[0]
        y = current_test.shape[1]
        z = current_test.shape[2]
        score = np.zeros((3, img_deps, img_rows, z), dtype='float32')
        score_num = np.zeros((3, img_deps, img_rows, z), dtype='int16')
        over = math.ceil(z / img_cols)
        h_overlap = math.ceil((over * img_cols - z) / (over - 1))
        for i in range(0, z - img_cols + 1, img_cols - h_overlap):
            patch = current_test[:, :, i:i + img_cols]
            cropp_img = resize(patch, (img_deps, img_rows, img_cols), order=3, mode='constant',
                               cval=0, clip=True, preserve_range=True, anti_aliasing=False)
            # cropp_tumor = resize(scan[1, :, :, i:i + img_cols], (img_deps, img_rows, img_cols), order=0, mode='edge',
            #                      cval=0, clip=True, preserve_range=True, anti_aliasing=False)

            box_test = torch.from_numpy(np.expand_dims(cropp_img, 0))
            box_test = torch.unsqueeze(box_test, dim=0)
            box_test = box_test.to(seg_help.equipment).float()
            with torch.no_grad():
                patch_test_mask = model_seg_lung(box_test)
            patch_test_mask = torch.softmax(patch_test_mask, dim=1)
            score[:, :, :, i:i + img_cols] += torch.squeeze(patch_test_mask).detach().cpu().numpy()
            score_num[:, :, :, i:i + img_cols] += 1

            # image_save = box_test.transpose(1, 4).contiguous().view(
            #     (box_test.size(0) * box_test.size(-1), box_test.size(1), box_test.size(2), box_test.size(3)))
            # grid = make_grid(image_save, nrow=8, padding=2)
            # save_img = grid.detach().cpu().numpy()
            # visualize(np.transpose(save_img, (1, 2, 0)), join(seg_help.config.tmp_dir, str(i) + "_images"))
            #
            # final_data = torch.from_numpy(cropp_tumor).unsqueeze(dim=0).unsqueeze(dim=0)
            # prob_save = final_data.float().transpose(1, 4).contiguous().view(
            #     (final_data.size(0) * final_data.size(-1), final_data.size(1), final_data.size(2),
            #      final_data.size(3))) / cropp_tumor.max()
            # grid = make_grid(prob_save, nrow=8, padding=2)
            # save_img = grid.detach().cpu().numpy()
            # visualize(np.transpose(save_img, (1, 2, 0)),
            #           join(seg_help.config.tmp_dir, str(i) + "_label"))
            #
            # prob_max = 1 - patch_test_mask[:, 0]
            # prob_save = prob_max.float().unsqueeze(1).transpose(1, 4).contiguous().view(
            #     (prob_max.size(0) * prob_max.size(-1), prob_max.size(0), prob_max.size(1),
            #      prob_max.size(2)))
            # grid = make_grid(prob_save, nrow=8, padding=2)
            # save_img = grid.detach().cpu().numpy()
            # visualize(np.transpose(save_img, (1, 2, 0)), join(seg_help.config.tmp_dir, str(i) + "_predict"))

        score_num = np.where(score_num == 0, 1, score_num)
        score_ = score / (score_num)
        new_shape = np.array([img_deps, img_rows, z], dtype=np.float32)
        resize_factor = new_shape / [x, y, z]
        score = nd.interpolation.zoom(score_[0], 1 / resize_factor, mode='nearest')
        # score = resize(score_[0], (x, y, z), order=3, mode='constant', cval=0, clip=True, preserve_range=True,
        #                anti_aliasing=False)
        score = np.clip(score, 0, 1)
        pred = 1 - score

        coarse_pred = np.where(pred > thres_hold, 1, 0)
        [lung_labels, num] = measure.label(coarse_pred, return_num=True)
        region = measure.regionprops(lung_labels)
        box = []
        for i in range(num):
            box.append(region[i].area)
        label_num_left = box.index(sorted(box, reverse=True)[0]) + 1
        if num == 1:
            lung_labels = np.where((lung_labels == label_num_left), 1, 0)
        else:
            label_num_right = box.index(sorted(box, reverse=True)[1]) + 1
            lung_labels = np.where((lung_labels == label_num_left) | (lung_labels == label_num_right), 1, 0)
        lung_labels = ndimage.binary_fill_holes(lung_labels).astype(int)

        # save_dir = join(seg_help.config.submission_dir, pred_dir)
        # if not isdir(save_dir):
        #     makedirs(save_dir)
        # nib.save(nib.Nifti1Image(lung_labels, affine=scan_nib.get_affine()),
        #          join(save_dir, file_name[:-4] + '.nii.gz'))

        # -------------visualization start------------------
        save_dir = join(seg_help.config.tmp_dir, '50_case_lung')
        if not isdir(save_dir):
            makedirs(save_dir)

        final_data = torch.from_numpy(lung_labels).unsqueeze(dim=0).unsqueeze(dim=0)
        prob_save = final_data.float().transpose(1, 4).contiguous().view(
            (final_data.size(0) * final_data.size(-1), final_data.size(1), final_data.size(2),
             final_data.size(3)))
        grid = make_grid(prob_save, nrow=16, padding=2)
        save_img = grid.detach().cpu().numpy()
        visualize(np.transpose(save_img, (1, 2, 0)),
                  join(save_dir, file_name[:-4] + "_predict_fusion_struct_largest"))

        final_data = torch.from_numpy(scan[1]).unsqueeze(dim=0).unsqueeze(dim=0)
        prob_save = final_data.float().transpose(1, 4).contiguous().view(
            (final_data.size(0) * final_data.size(-1), final_data.size(1), final_data.size(2),
             final_data.size(3))) / np.max(scan[1])
        grid = make_grid(prob_save, nrow=16, padding=2)
        save_img = grid.detach().cpu().numpy()
        visualize(np.transpose(save_img, (1, 2, 0)),
                  join(save_dir, file_name[:-4] + "_label_covid"))

        # final_data = torch.from_numpy(current_test).unsqueeze(dim=0).unsqueeze(
        #     dim=0)
        # prob_save = final_data.float().transpose(1, 4).contiguous().view(
        #     (final_data.size(0) * final_data.size(-1), final_data.size(1), final_data.size(2),
        #      final_data.size(3)))
        # grid = make_grid(prob_save, nrow=16, padding=2)
        # save_img = grid.detach().cpu().numpy()
        # visualize(np.transpose(save_img, (1, 2, 0)), join(save_dir, file_name[:-4] + "_raw_covid"))

        # final_data = torch.from_numpy(pred).unsqueeze(dim=0).unsqueeze(dim=0)
        # prob_save = final_data.float().transpose(1, 4).contiguous().view(
        #     (final_data.size(0) * final_data.size(-1), final_data.size(1), final_data.size(2),
        #      final_data.size(3)))
        # grid = make_grid(prob_save, nrow=16, padding=2)
        # save_img = grid.detach().cpu().numpy()
        # visualize(np.transpose(save_img, (1, 2, 0)),
        #           join(save_dir, file_name[:-4] + "_predict_fusion_struct_prob"))

        # -------------visualization end------------------
    # seg_help._submit_predict(join(seg_help.config.submission_dir, pred_dir), test_type=3)
