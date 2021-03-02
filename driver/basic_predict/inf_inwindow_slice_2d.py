# -*- coding: utf-8 -*-
# @Time    : 20/5/23 21:31
# @Author  : qgking
# @Email   : qgking@tju.edu.cn
# @Software: PyCharm
# @Desc    : inf_inwindow_slice_2d.py
import matplotlib
import sys

sys.path.extend(["../../", "../", "./"])
from common.base_utls import *
from common.data_utils import *
import torch
from skimage.transform import resize
import torch.nn.functional as F
from torch.cuda import empty_cache


def impaintshow(img, seg, preds, output_dir, fname):
    """Takes raw image img, seg in range 0-2, list of predictions in range 0-2"""
    img = np.squeeze(img)
    seg = np.squeeze(seg)
    preds = np.squeeze(preds)
    fig = plt.figure()
    ALPHA = 0.8
    n_plots = 1
    plt.set_cmap('gray')
    pre = preds == 1
    diffinf = np.logical_xor(seg == 1, pre)
    title = ""
    plt.title(title)
    plt.imshow(img)
    # plt.hold(True)
    # Liver prediction
    plt.imshow(np.ma.masked_where(pre == 0, pre), cmap="Greens", vmin=0.1, vmax=1.2, alpha=ALPHA)
    # plt.hold(True)
    # Lesion prediction
    plt.imshow(np.ma.masked_where(diffinf == 0, diffinf), cmap="Reds", vmin=0.1, vmax=1.2, alpha=ALPHA)
    # plt.hold(True)
    plt.axis('off')
    fig.set_size_inches(img.shape[0] / 100.0 / 3.0, img.shape[0] / 100.0 / 3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(join(output_dir, fname + '.png'), transparent=True, dpi=300, pad_inches=0, bbox_inches='tight')
    plt.close()


# crop x and y
def predict_inf_inwindow_slide_2d(seg_help, model, covid_test_data, thres=0.7):
    seg_help.model.eval()
    model.eval()
    img_cols = seg_help.config.patch_x
    img_rows = seg_help.config.patch_y
    img_deps = seg_help.config.patch_z

    segmentation_metrics = {
        'Jaccard': 0,
        'F1': 0, 'ACCURACY': 0, 'SENSITIVITY': 0, 'SPECIFICITY': 0,
        'DICESCORE': 0}
    lesion_segmentation_scores = {}
    output_dir = join(seg_help.config.tmp_dir, 'infer')
    mkdir_if_not_exist([output_dir])
    for file_path in covid_test_data:
        # print(file_path)
        file_name = basename(file_path)
        print(file_name)
        scans = np.load(file_path)
        img = scans[0]
        current_test = img.copy()
        if 'MosMedData' in file_path:
            lung = scans[2]
            infection = scans[1]
            nrow = 8
        elif 'COVID-19-CT' in file_path:
            lung = scans[1]
            infection = scans[2]
            nrow = 16
        minx, maxx, miny, maxy, minz, maxz = min_max_voi(lung, superior=3, inferior=3)
        x = current_test.shape[0]
        y = current_test.shape[1]
        z = current_test.shape[2]

        score = np.zeros((seg_help.config.classes, maxx - minx, maxy - miny, maxz - minz), dtype='float32')
        score_num = np.zeros((seg_help.config.classes, maxx - minx, maxy - miny, maxz - minz), dtype='int16')
        current_test_cut = current_test[minx:maxx, miny:maxy, minz:maxz]
        # normalization
        # current_test_cut = (current_test_cut - seg_help.mean) / seg_help.std
        current_inf_cut = infection[minx:maxx, miny:maxy, minz:maxz]
        num = 0
        flo = int(np.floor(img_deps / 2))
        over = math.ceil(score.shape[1] / img_cols) + 1
        xstep = math.ceil((over * img_cols - score.shape[1]) / (over - 1))
        xstep = max(img_cols // 2, xstep)
        over = math.ceil(score.shape[2] / img_rows) + 1
        ystep = math.ceil((over * img_rows - score.shape[2]) / (over - 1))
        ystep = max(img_rows // 2, ystep)
        x_slices = np.arange(0, score.shape[1], xstep)
        y_slices = np.arange(0, score.shape[2], ystep)
        # print('x step %d, y step %d' % (xstep, ystep))
        # print('total slices %d' % (len(x_slices) * len(x_slices) * score.shape[-1]))
        for i in range(len(x_slices)):
            deep = x_slices[i]
            cols = deep if deep + img_cols < score.shape[1] else score.shape[1] - img_cols
            for j in range(len(y_slices)):
                height = y_slices[j]
                rows = height if height + img_rows < score.shape[2] else score.shape[2] - img_rows
                for c in range(flo, score.shape[-1] - flo, 1):
                    cropp_img = current_test_cut[cols:cols + img_cols, rows:rows + img_rows,
                                c - flo: c + img_deps - flo]
                    cropp_infection = current_inf_cut[cols:cols + img_cols, rows:rows + img_rows,
                                      c - flo: c + img_deps - flo]
                    assert cropp_img.shape == (img_cols, img_rows, img_deps)
                    num += 1
                    box_test = torch.from_numpy(np.transpose(cropp_img, (2, 0, 1)))
                    box_test = torch.unsqueeze(box_test, dim=0)
                    box_test = box_test.to(seg_help.equipment).float()

                    cropp_infection = torch.from_numpy(np.transpose(cropp_infection, (2, 0, 1)))
                    cropp_infection = torch.unsqueeze(cropp_infection, dim=0)
                    cropp_infection = cropp_infection.to(seg_help.equipment).float()

                    with torch.no_grad():
                        patch_test_mask, _, _ = model(box_test)
                        if isinstance(patch_test_mask, (tuple, list)):
                            patch_test_mask = patch_test_mask[0]
                    patch_test_mask = torch.softmax(patch_test_mask, dim=1)
                    score[:, cols:cols + img_cols, rows:rows + img_rows,
                    c] += patch_test_mask.squeeze().detach().cpu().numpy()
                    score_num[:, cols:cols + img_cols, rows:rows + img_rows,
                    c] += 1
                    # visual_batch(box_test, output_dir, file_name[:-4] + '_' + str(num) + "_images",
                    #              channel=1,
                    #              nrow=nrow)
                    # visual_batch(cropp_infection, output_dir, file_name[:-4] + '_' + str(num) + "_label",
                    #              channel=1, nrow=nrow)
                    # prob_max = patch_test_mask[:, 1, :, :]
                    # image_save = prob_max.unsqueeze(1).contiguous()
                    # visual_batch(image_save, output_dir, file_name[:-4] + '_' + str(num) + "_predict",
                    #              channel=1,
                    #              nrow=nrow)
        score_ = score.copy()
        score_num = np.where(score_num == 0, 1, score_num)
        score_ = (score_ / (score_num)).copy()
        predict = score_[seg_help.config.classes - 1]
        score_final = np.zeros((x, y, z), dtype='float32')
        score_final[minx:maxx, miny:maxy, minz:maxz] = predict
        fusion_final_predict = np.where(score_final > thres, 1, 0)
        fine_pred = ndimage.binary_dilation(fusion_final_predict, iterations=1).astype(fusion_final_predict.dtype)

        score_final_torch = torch.from_numpy(score_final[minx:maxx, miny:maxy, minz:maxz]).unsqueeze(0).unsqueeze(
            0).contiguous()
        fine_pred_torch = torch.from_numpy(fine_pred[minx:maxx, miny:maxy, minz:maxz]).unsqueeze(0).unsqueeze(
            0).contiguous()
        imm = img[minx: maxx, miny: maxy, minz: maxz]
        img_torch = torch.from_numpy(imm).unsqueeze(0).unsqueeze(0).contiguous()
        infection_torch = torch.from_numpy(infection[minx:maxx, miny:maxy, minz:maxz]).unsqueeze(0).unsqueeze(
            0).contiguous()
        visual_batch(img_torch, output_dir, file_name[:-4] + "_images", channel=1, nrow=nrow)
        visual_batch(infection_torch, output_dir, file_name[:-4] + "_label", channel=1, nrow=nrow)
        visual_batch(fine_pred_torch, output_dir, file_name[:-4] + "_predict", channel=1, nrow=nrow)
        visual_batch(score_final_torch, output_dir, file_name[:-4] + "_predict_score", channel=1,
                     nrow=nrow)

        scores = compute_all_metric_for_single_seg(infection, fine_pred)
        for metric in segmentation_metrics:
            if metric not in lesion_segmentation_scores:
                lesion_segmentation_scores[metric] = []
            lesion_segmentation_scores[metric].extend(scores[metric])
    lesion_segmentation_metrics = {}
    info = ''
    for m in lesion_segmentation_scores:
        lesion_segmentation_metrics[m] = np.mean(lesion_segmentation_scores[m])
        info += ('\n' + m + ': {val:.9f}   '.format(val=lesion_segmentation_metrics[m]))
    print(info)
    zipDir(output_dir, output_dir + '.zip')
    shutil.rmtree(output_dir)
    return lesion_segmentation_metrics['DICESCORE']


# total resize x and y. not crop x and y
def predict_inf_inwindow_2d(seg_help, model, covid_test_data, thres=0.7):
    seg_help.model.eval()
    model.eval()
    resize_x = seg_help.config.patch_x
    resize_y = seg_help.config.patch_y
    resize_z = seg_help.config.patch_z
    segmentation_metrics = {
        'Jaccard': 0, 'HD': 0,
        'F1': 0, 'ACCURACY': 0, 'SENSITIVITY': 0, 'SPECIFICITY': 0,
        'DICESCORE': 0}
    lesion_segmentation_scores = {}
    output_dir = join(seg_help.config.tmp_dir, 'infer')
    mkdir_if_not_exist([output_dir])
    for file_path in covid_test_data:
        # print(file_path)
        file_name = basename(file_path)
        print(file_name)
        scans = np.load(file_path)
        img = scans[0]
        if 'MosMedData' in file_path:
            lung = scans[2]
            infection = scans[1]
            nrow = 8
        elif 'COVID-19-CT' in file_path:
            lung = scans[1]
            infection = scans[2]
            nrow = 16
        minx, maxx, miny, maxy, minz, maxz = min_max_voi(lung, superior=3, inferior=3)
        print((minx, maxx, miny, maxy, minz, maxz))
        cropped_im = img[minx: maxx, miny: maxy, :]

        cropped_if = infection[minx: maxx, miny: maxy, :]
        flo = int(np.floor(resize_z / 2))
        score = np.zeros((seg_help.config.classes, resize_x, resize_y, img.shape[-1]), dtype='float32')
        resized_img = resize(cropped_im, (resize_x, resize_y, img.shape[-1]), order=3, mode='constant',
                             cval=0, clip=True, preserve_range=True, anti_aliasing=False)
        resized_infection = resize(cropped_if, (resize_x, resize_y, img.shape[-1]), order=0, mode='edge',
                                   cval=0, clip=True, preserve_range=True, anti_aliasing=False)
        # normalization
        # resized_img = (resized_img - seg_help.mean) / seg_help.std
        minz = max(flo, minz)
        maxz = min(cropped_if.shape[-1] - flo, maxz)
        for c in range(minz, maxz, 1):
            cropp_img = resized_img[:, :, c - flo: c + resize_z - flo].copy()
            cropp_infection = resized_infection[:, :, c - flo: c + resize_z - flo].copy()
            cropp_img = np.transpose(cropp_img, (2, 0, 1))
            cropp_infection = np.transpose(cropp_infection, (2, 0, 1))
            box_test = torch.from_numpy(np.expand_dims(cropp_img, 0))
            cropp_infection_test = torch.from_numpy(np.expand_dims(cropp_infection, 0))
            box_test = box_test.to(seg_help.equipment).float()
            with torch.no_grad():
                logits, _, _ = seg_help.model(box_test)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
            patch_test_mask = torch.softmax(logits, dim=1)
            score[:, :, :, c] += patch_test_mask.squeeze().detach().cpu().numpy()
            del box_test
            # visual_batch(box_test, output_dir, file_name[:-4] + '_' + str(c) + "_images", channel=1,
            #              nrow=nrow)
            # visual_batch(cropp_infection_test, output_dir, file_name[:-4] + '_' + str(c) + "_label",
            #              channel=1, nrow=nrow)
            # prob_max = patch_test_mask[:, 1, :, :]
            # image_save = prob_max.unsqueeze(1).contiguous()
            # visual_batch(image_save, output_dir, file_name[:-4] + '_' + str(c) + "_predict", channel=1,
            #              nrow=nrow)

        score_ = score
        predict = score_[1]
        sc = np.zeros((1, 1, predict.shape[0], predict.shape[1], predict.shape[2]), dtype='float32')
        sc[0, 0, :, :, :] = predict
        up_predict = F.interpolate(torch.from_numpy(sc),
                                   size=(cropped_im.shape[0], cropped_im.shape[1], cropped_im.shape[2]),
                                   mode='trilinear',
                                   align_corners=True)
        up_predict = torch.squeeze(up_predict)
        score = np.clip(up_predict, 0, 1)
        score_final = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype='int16')
        score_final[minx: maxx, miny: maxy, :] = score
        fine_pred = np.where(score_final > thres, 1, 0)
        # fine_pred = ndimage.binary_dilation(fusion_final_predict, iterations=1).astype(fusion_final_predict.dtype)

        score_final_torch = torch.from_numpy(score_final[minx: maxx, miny: maxy, minz: maxz]).unsqueeze(0).unsqueeze(
            0).contiguous()
        fine_pred_torch = torch.from_numpy(fine_pred[minx: maxx, miny: maxy, minz: maxz]).unsqueeze(0).unsqueeze(
            0).contiguous()
        imm = img[minx: maxx, miny: maxy, minz: maxz]
        img_torch = torch.from_numpy(imm).unsqueeze(0).unsqueeze(0).contiguous()
        infection_torch = torch.from_numpy(infection[minx: maxx, miny: maxy, minz: maxz]).unsqueeze(0).unsqueeze(
            0).contiguous()
        visual_batch(img_torch, output_dir, file_name[:-4] + "_images", channel=1, nrow=nrow)
        visual_batch(infection_torch, output_dir, file_name[:-4] + "_label", channel=1, nrow=nrow)
        visual_batch(fine_pred_torch, output_dir, file_name[:-4] + "_predict", channel=1, nrow=nrow)
        visual_batch(score_final_torch, output_dir, file_name[:-4] + "_predict_score", channel=1,
                     nrow=nrow)

        scores = compute_all_metric_for_single_seg(infection, fine_pred)
        for metric in segmentation_metrics:
            if metric not in lesion_segmentation_scores:
                lesion_segmentation_scores[metric] = []
            lesion_segmentation_scores[metric].extend(scores[metric])
        empty_cache()
    lesion_segmentation_metrics = {}
    info = ''
    for m in lesion_segmentation_scores:
        print(lesion_segmentation_scores[m])
        lesion_segmentation_metrics[m] = np.mean(lesion_segmentation_scores[m])
        info += ('\n' + m + ': {val:.9f}   '.format(val=lesion_segmentation_metrics[m]))
    print(info)
    zipDir(output_dir, output_dir + '.zip')
    shutil.rmtree(output_dir)
    return lesion_segmentation_metrics['DICESCORE']


# total resize x and y. not crop x and y, sequense
def predict_inf_inwindow_2d_seq(seg_help, model, covid_test_data, thres=0.7):
    seg_help.model.eval()
    model.eval()
    resize_x = seg_help.config.patch_x
    resize_y = seg_help.config.patch_y
    segmentation_metrics = {
        'Jaccard': 0, 'HD': 0,
        'F1': 0, 'ACCURACY': 0, 'SENSITIVITY': 0, 'SPECIFICITY': 0,
        'DICESCORE': 0}
    lesion_segmentation_scores = {}
    output_dir = join(seg_help.config.tmp_dir, 'infer')
    mkdir_if_not_exist([output_dir])
    for file_path in covid_test_data:
        # print(file_path)
        file_name = basename(file_path)
        scans = np.load(file_path)
        img = scans[0]
        if 'MosMedData' in file_path:
            lung = scans[2]
            infection = scans[1]
            nrow = 8
        elif 'COVID-19-CT' in file_path:
            lung = scans[1]
            infection = scans[2]
            nrow = 16
        minx, maxx, miny, maxy, minz, maxz = min_max_voi(lung, superior=3, inferior=3)
        cropped_im = img[minx: maxx, miny: maxy, minz: maxz]
        cropped_if = infection[minx: maxx, miny: maxy, minz: maxz]
        resized_img = resize(cropped_im, (resize_x, resize_y, maxz - minz), order=3, mode='constant',
                             cval=0, clip=True, preserve_range=True, anti_aliasing=False)
        resized_infection = resize(cropped_if, (resize_x, resize_y, maxz - minz), order=0, mode='edge',
                                   cval=0, clip=True, preserve_range=True, anti_aliasing=False)
        su_data = torch.from_numpy(resized_img)
        su_label = torch.from_numpy(resized_infection)
        su_data = su_data.permute(2, 0, 1).squeeze(0)
        su_label = su_label.permute(2, 0, 1).squeeze(0)
        total_slices = su_data.size(0)
        patches_img = []
        cam_total = []
        seg_total = []
        for s in range(total_slices):
            cropp_img = su_data[s, :, :].unsqueeze(0)
            cropp_img = cropp_img.unsqueeze(0)
            patches_img.append(cropp_img)
            if s % seg_help.config.test_batch_size == 0 or s == total_slices - 1:
                patches_img = torch.cat(patches_img, dim=0)
                patches_img = patches_img.to(seg_help.equipment).float()
                with torch.no_grad():
                    bb, cc, xx, yy = patches_img.size()
                    patches_img = patches_img.view(bb * cc, 1, xx, yy)
                    out, feature, cam = model(patches_img)
                    cam_total.append(cam)
                    seg_total.append(out)
                del patches_img
                patches_img = []
        cam_total = torch.cat(cam_total, dim=0)
        seg_total = torch.cat(seg_total, dim=0)
        if 'u2net' in seg_help.config.model:
            prob = torch.softmax(seg_total[0], dim=1)
        else:
            prob = torch.softmax(seg_total, dim=1)

        prob_max = prob[:, seg_help.config.classes - 1, :, :]
        prob_max_up = prob_max.permute(1, 2, 0).unsqueeze(0).unsqueeze(0)
        prob_max_up = F.interpolate(prob_max_up,
                                    size=(cropped_im.shape[0], cropped_im.shape[1], maxz - minz),
                                    mode='trilinear',
                                    align_corners=True)
        prob_max_ = torch.squeeze(prob_max).cpu().numpy()
        fine_pred = np.where(prob_max_ > thres, 1, 0)
        prob_max_up = torch.squeeze(prob_max_up).cpu().numpy()
        fine_pred_up = np.where(prob_max_up > thres, 1, 0)
        fine_pred_final = ndimage.binary_dilation(fine_pred, iterations=1).astype(fine_pred.dtype)
        fine_pred_up_final = ndimage.binary_dilation(fine_pred_up, iterations=1).astype(fine_pred_up.dtype)

        su_cam_g = cam_total.detach()[:, seg_help.config.classes - 1, :, :].unsqueeze(1)
        su_cam_g = seg_help.max_norm_cam(su_cam_g)
        su_cam_g = F.interpolate(su_cam_g, size=su_data.size()[1:], mode='bilinear', align_corners=True)
        su_cam_g = make_grid(su_cam_g, nrow=nrow, padding=2, pad_value=1)
        su_cam_g = su_cam_g[0, :, :].unsqueeze(0).unsqueeze(0)
        su_data_g = su_data.unsqueeze(1).detach()
        su_data_g = make_grid(su_data_g, nrow=nrow, padding=2, pad_value=1)
        su_data_g = su_data_g.unsqueeze(0)
        _, result_pp = seg_help.visualize_cam(su_cam_g.cpu(), su_data_g.cpu())
        visualize(np.clip(np.transpose(result_pp.detach().cpu().numpy(), (1, 2, 0)), 0, 1),
                  join(output_dir, file_name[:-4] + "_cam"))

        image_save = prob_max.unsqueeze(1).contiguous()
        visual_batch(image_save, output_dir, file_name[:-4] + "_prob", channel=1, nrow=nrow)

        visual_batch(su_data, output_dir, file_name[:-4] + "_images", channel=1, nrow=nrow)

        visual_batch(su_label, output_dir, file_name[:-4] + "_label", channel=1, nrow=nrow)

        fine_pred_torch = torch.from_numpy(fine_pred_final).unsqueeze(1).contiguous()
        visual_batch(fine_pred_torch, output_dir, file_name[:-4] + "_predict", channel=1, nrow=nrow)

        fine_pred_orgi_res = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype='float32')
        fine_pred_orgi_res[minx: maxx, miny: maxy, minz: maxz] = fine_pred_up_final

        scores = compute_all_metric_for_single_seg(infection, fine_pred_orgi_res)
        for metric in segmentation_metrics:
            if metric not in lesion_segmentation_scores:
                lesion_segmentation_scores[metric] = []
            lesion_segmentation_scores[metric].extend(scores[metric])
        empty_cache()
    lesion_segmentation_metrics = {}
    info = ''
    for m in lesion_segmentation_scores:
        # print(lesion_segmentation_scores[m])
        lesion_segmentation_metrics[m] = np.mean(lesion_segmentation_scores[m])
        info += ('\n' + m + ': {val:.9f}   '.format(val=lesion_segmentation_metrics[m]))
    print(info)
    zipDir(output_dir, output_dir + '.zip')
    shutil.rmtree(output_dir)
    return lesion_segmentation_metrics['DICESCORE']


# total 2d slice without seq
def predict_inf_inwindow_2d_out_seq(seg_help, model, covid_test_data, thres=0.7, epoch=0):
    seg_help.model.eval()
    model.eval()
    resize_x = seg_help.config.patch_x
    resize_y = seg_help.config.patch_y
    segmentation_metrics = {
        'Jaccard': 0, 'HD': 0,
        'F1': 0, 'ACCURACY': 0, 'SENSITIVITY': 0, 'SPECIFICITY': 0,
        'DICESCORE': 0}
    lesion_segmentation_scores = {}
    img_list_ = covid_test_data[0]
    lung_list = covid_test_data[1]
    inf_list = covid_test_data[2]
    output_dir = join(seg_help.config.tmp_dir, 'infer_thres%.2f_%d' % (thres, epoch))
    mkdir_if_not_exist([output_dir])
    for iii in range(len(img_list_)):
        # print(file_path)
        img = img_list_[iii].copy()
        lung = lung_list[iii].copy()
        infection = inf_list[iii].copy()
        if np.sum(infection) == 0:
            continue
        nrow = 8
        minx, maxx, miny, maxy = min_max_voi_2d(lung, superior=10, inferior=10)
        cropped_im = img[minx: maxx, miny: maxy]
        cropped_if = infection[minx: maxx, miny: maxy]
        resized_img = resize(cropped_im, (resize_x, resize_y), order=3, mode='constant',
                             cval=0, clip=True, preserve_range=True, anti_aliasing=False)
        resized_infection = resize(cropped_if, (resize_x, resize_y), order=0, mode='edge',
                                   cval=0, clip=True, preserve_range=True, anti_aliasing=False)
        su_data = torch.from_numpy(resized_img)
        su_label = torch.from_numpy(resized_infection)
        su_data = su_data.unsqueeze(0).unsqueeze(0)
        su_label = su_label.unsqueeze(0).unsqueeze(0)
        cropp_img = su_data
        cropp_img = cropp_img
        patches_img = cropp_img.to(seg_help.equipment).float()
        with torch.no_grad():
            bb, cc, xx, yy = patches_img.size()
            patches_img = patches_img.view(bb * cc, 1, xx, yy)
            out, feature, cam = model(patches_img)
        if 'u2net' in seg_help.config.model:
            prob = torch.softmax(out[0], dim=1)
        else:
            prob = torch.softmax(out, dim=1)

        prob_max = prob[:, seg_help.config.classes - 1, :, :]
        prob_max_up = prob_max.unsqueeze(1)
        prob_max_up = F.interpolate(prob_max_up,
                                    size=(cropped_im.shape[0], cropped_im.shape[1]),
                                    mode='bilinear',
                                    align_corners=True)
        prob_max_ = torch.squeeze(prob_max).cpu().numpy()
        fine_pred = np.where(prob_max_ > thres, 1, 0)
        prob_max_up = torch.squeeze(prob_max_up).cpu().numpy()
        fine_pred_up = np.where(prob_max_up > thres, 1, 0)
        fine_pred_final = ndimage.binary_dilation(fine_pred, iterations=1).astype(fine_pred.dtype)
        fine_pred_up_final = ndimage.binary_dilation(fine_pred_up, iterations=1).astype(fine_pred_up.dtype)

        su_cam_g = cam.detach()[:, seg_help.config.classes - 1, :, :].unsqueeze(1)
        su_cam_g = F.interpolate(su_cam_g, size=su_data.size()[2:], mode='bilinear', align_corners=True)
        su_cam_g = make_grid(su_cam_g, nrow=nrow, pad_value=1)
        su_cam_g = su_cam_g[0, :, :].unsqueeze(0).unsqueeze(0)
        su_data_g = su_data.detach().clone()
        su_data_g = make_grid(su_data_g, nrow=nrow, pad_value=1)
        su_data_g = su_data_g.unsqueeze(0)
        su_cam_g = seg_help.max_norm_cam(su_cam_g)
        heatmap, result_pp = seg_help.visualize_cam(su_cam_g.cpu(), su_data_g.cpu())
        visualize(np.clip(np.transpose(result_pp.detach().cpu().numpy(), (1, 2, 0)), 0, 1),
                  join(output_dir,
                       "test_" + str(iii) + '_cam'))

        image_save = prob_max.unsqueeze(0).contiguous()
        visual_batch(image_save, output_dir, "test_" + str(iii) + "_prob", channel=1, nrow=nrow)

        visual_batch(su_data, output_dir, "test_" + str(iii) + "_images", channel=1, nrow=nrow)

        visual_batch(su_label, output_dir, "test_" + str(iii) + "_label", channel=1, nrow=nrow)

        # fine_pred_torch = torch.from_numpy(fine_pred_final).unsqueeze(0).contiguous()
        # visual_batch(fine_pred_torch, output_dir, str(iii) + "_test_predict", channel=1, nrow=nrow)

        fine_pred_orgi_res = np.zeros((img.shape[0], img.shape[1]), dtype='float32')
        fine_pred_orgi_res[minx: maxx, miny: maxy] = fine_pred_up_final

        scores = compute_all_metric_for_single_seg(infection, fine_pred_orgi_res)
        for metric in segmentation_metrics:
            if metric not in lesion_segmentation_scores:
                lesion_segmentation_scores[metric] = []
            lesion_segmentation_scores[metric].extend(scores[metric])
        empty_cache()
    lesion_segmentation_metrics = {}
    info = ''
    for m in lesion_segmentation_scores:
        # print(lesion_segmentation_scores[m])
        lesion_segmentation_metrics[m] = np.mean(lesion_segmentation_scores[m])
        info += ('\n' + m + ': {val:.9f}   '.format(val=lesion_segmentation_metrics[m]))
    # print(info)
    print('Test Dice : {val:.9f}  Thres {thre:.2f}  '.format(val=lesion_segmentation_metrics['DICESCORE'], thre=thres))
    zipDir(output_dir, output_dir + '.zip')
    shutil.rmtree(output_dir)
    return lesion_segmentation_metrics['DICESCORE']


# total 2d slice without seq
def predict_inf_inwindow_2d_out_seq_weight(seg_help, model, covid_test_data, thres=0.7, epoch=0):
    seg_help.model.eval()
    model.eval()
    resize_x = seg_help.config.patch_x
    resize_y = seg_help.config.patch_y
    segmentation_metrics = {
        'Jaccard': 0, 'HD': 0,
        'F1': 0, 'ACCURACY': 0, 'SENSITIVITY': 0, 'SPECIFICITY': 0,
        'DICESCORE': 0}
    lesion_segmentation_scores = {}
    img_list_ = covid_test_data[0]
    lung_list = covid_test_data[1]
    inf_list = covid_test_data[2]
    output_dir = join(seg_help.config.tmp_dir, 'infer_thres%.2f_%d' % (thres, epoch))
    mkdir_if_not_exist([output_dir])
    for iii in range(len(img_list_)):
        # print(file_path)
        img = img_list_[iii].copy()
        lung = lung_list[iii].copy()
        infection = inf_list[iii].copy()
        if np.sum(infection) == 0:
            continue
        nrow = 8
        minx, maxx, miny, maxy = min_max_voi_2d(lung, superior=10, inferior=10)
        cropped_im = img[minx: maxx, miny: maxy]
        cropped_if = infection[minx: maxx, miny: maxy]
        resized_img = resize(cropped_im, (resize_x, resize_y), order=3, mode='constant',
                             cval=0, clip=True, preserve_range=True, anti_aliasing=False)
        resized_infection = resize(cropped_if, (resize_x, resize_y), order=0, mode='edge',
                                   cval=0, clip=True, preserve_range=True, anti_aliasing=False)
        su_data = torch.from_numpy(resized_img)
        su_label = torch.from_numpy(resized_infection)
        su_data = su_data.unsqueeze(0).unsqueeze(0)
        su_label = su_label.unsqueeze(0).unsqueeze(0)
        cropp_img = su_data
        cropp_img = cropp_img
        patches_img = cropp_img.to(seg_help.equipment).float()
        with torch.no_grad():
            bb, cc, xx, yy = patches_img.size()
            patches_img = patches_img.view(bb * cc, 1, xx, yy)
            out1, out2, cam = model(patches_img)
            output = None
            for pred in out1:
                if output is None:
                    output = torch.zeros(pred.size()).to(seg_help.equipment)
                output += pred
            prob = torch.softmax(output, dim=1)

        prob_max = prob[:, seg_help.config.classes - 1, :, :]
        prob_max_up = prob_max.unsqueeze(1)
        prob_max_up = F.interpolate(prob_max_up,
                                    size=(cropped_im.shape[0], cropped_im.shape[1]),
                                    mode='bilinear',
                                    align_corners=True)
        prob_max_ = torch.squeeze(prob_max).cpu().numpy()
        fine_pred = np.where(prob_max_ > thres, 1, 0)
        prob_max_up = torch.squeeze(prob_max_up).cpu().numpy()
        fine_pred_up = np.where(prob_max_up > thres, 1, 0)
        fine_pred_final = ndimage.binary_dilation(fine_pred, iterations=1).astype(fine_pred.dtype)
        fine_pred_up_final = ndimage.binary_dilation(fine_pred_up, iterations=1).astype(fine_pred_up.dtype)

        if cam is not None:
            su_cam_g = cam.detach()[:, seg_help.config.classes - 1, :, :].unsqueeze(1)
            su_cam_g = F.interpolate(su_cam_g, size=su_data.size()[2:], mode='bilinear', align_corners=True)
            su_cam_g = make_grid(su_cam_g, nrow=nrow, pad_value=1)
            su_cam_g = su_cam_g[0, :, :].unsqueeze(0).unsqueeze(0)
            su_data_g = su_data.detach().clone()
            su_data_g = make_grid(su_data_g, nrow=nrow, pad_value=1)
            su_data_g = su_data_g.unsqueeze(0)
            su_cam_g = seg_help.max_norm_cam(su_cam_g)
            heatmap, result_pp = seg_help.visualize_cam(su_cam_g.cpu(), su_data_g.cpu())
            visualize(np.clip(np.transpose(result_pp.detach().cpu().numpy(), (1, 2, 0)), 0, 1),
                      join(output_dir,
                           "test_" + str(iii) + '_cam'))

        image_save = prob_max.unsqueeze(0).contiguous()
        visual_batch(image_save, output_dir, "test_" + str(iii) + "_prob", channel=1, nrow=nrow)

        visual_batch(su_data, output_dir, "test_" + str(iii) + "_images", channel=1, nrow=nrow)

        visual_batch(su_label, output_dir, "test_" + str(iii) + "_label", channel=1, nrow=nrow)

        # fine_pred_torch = torch.from_numpy(fine_pred_final).unsqueeze(0).contiguous()
        # visual_batch(fine_pred_torch, output_dir, str(iii) + "_test_predict", channel=1, nrow=nrow)

        fine_pred_orgi_res = np.zeros((img.shape[0], img.shape[1]), dtype='float32')
        fine_pred_orgi_res[minx: maxx, miny: maxy] = fine_pred_up_final

        scores = compute_all_metric_for_single_seg(infection, fine_pred_orgi_res)
        for metric in segmentation_metrics:
            if metric not in lesion_segmentation_scores:
                lesion_segmentation_scores[metric] = []
            lesion_segmentation_scores[metric].extend(scores[metric])
        empty_cache()
    lesion_segmentation_metrics = {}
    info = ''
    for m in lesion_segmentation_scores:
        # print(lesion_segmentation_scores[m])
        lesion_segmentation_metrics[m] = np.mean(lesion_segmentation_scores[m])
        info += ('\n' + m + ': {val:.9f}   '.format(val=lesion_segmentation_metrics[m]))
    # print(info)
    print('Test Dice : {val:.9f}   '.format(val=lesion_segmentation_metrics['DICESCORE']))
    zipDir(output_dir, output_dir + '.zip')
    shutil.rmtree(output_dir)
    return lesion_segmentation_metrics['DICESCORE']


# total 2d slice without seq
def predict_inf_inwindow_2d_out_seq_weight_cam(seg_help, model, model_cam, covid_test_data, thres=0.7, epoch=0):
    seg_help.model.eval()
    model.eval()
    model_cam.eval()
    resize_x = seg_help.config.patch_x
    resize_y = seg_help.config.patch_y
    segmentation_metrics = {
        'Jaccard': 0, 'HD': 0,
        'F1': 0, 'ACCURACY': 0, 'SENSITIVITY': 0, 'SPECIFICITY': 0,
        'DICESCORE': 0}
    lesion_segmentation_scores = {}
    img_list_ = covid_test_data[0]
    lung_list = covid_test_data[1]
    inf_list = covid_test_data[2]
    output_dir = join(seg_help.config.tmp_dir, 'infer_thres%.2f_%d' % (thres, epoch))
    mkdir_if_not_exist([output_dir])
    for iii in range(len(img_list_)):
        # print(file_path)
        img = img_list_[iii].copy()
        lung = lung_list[iii].copy()
        infection = inf_list[iii].copy()
        if np.sum(infection) == 0:
            continue
        nrow = 8
        minx, maxx, miny, maxy = min_max_voi_2d(lung, superior=10, inferior=10)
        cropped_im = img[minx: maxx, miny: maxy]
        cropped_if = infection[minx: maxx, miny: maxy]
        resized_img = resize(cropped_im, (resize_x, resize_y), order=3, mode='constant',
                             cval=0, clip=True, preserve_range=True, anti_aliasing=False)
        resized_infection = resize(cropped_if, (resize_x, resize_y), order=0, mode='edge',
                                   cval=0, clip=True, preserve_range=True, anti_aliasing=False)
        su_data = torch.from_numpy(resized_img)
        su_label = torch.from_numpy(resized_infection)
        su_data = su_data.unsqueeze(0).unsqueeze(0)
        su_label = su_label.unsqueeze(0).unsqueeze(0)
        cropp_img = su_data
        cropp_img = cropp_img
        patches_img = cropp_img.to(seg_help.equipment).float()
        with torch.no_grad():
            bb, cc, xx, yy = patches_img.size()
            patches_img = patches_img.view(bb * cc, 1, xx, yy)
            cam_p = model_cam.forward_cam(patches_img)
            out1, out2, cam = model(patches_img, cam_p)
            output = None
            xxx = 0
            for pred in out1:
                if output is None:
                    output = torch.zeros(pred.size()).to(seg_help.equipment)
                output += pred
                xxx += 1
                pred = torch.softmax(pred, dim=1)
                pred_max = pred[:, seg_help.config.classes - 1, :, :]
                image_save = pred_max.unsqueeze(0).contiguous()
                visual_batch(image_save, output_dir, "test_" + str(iii) + "_prob_" + str(xxx), channel=1, nrow=nrow)

            output_0 = output
            cam_p = model_cam.forward_cam(seg_help.fliplr(patches_img))
            out1, out2, cam = model(seg_help.fliplr(patches_img), cam_p)
            output = None
            for pred in out1:
                if output is None:
                    output = torch.zeros(pred.size()).to(seg_help.equipment)
                output += seg_help.fliplr(pred)
            output_1 = output
            cam_p = model_cam.forward_cam(seg_help.flipvr(patches_img))
            out1, out2, cam = model(seg_help.flipvr(patches_img), cam_p)
            output = None
            for pred in out1:
                if output is None:
                    output = torch.zeros(pred.size()).to(seg_help.equipment)
                output += seg_help.flipvr(pred)
            output_2 = output
            prob = torch.softmax(output_0 + output_1 + output_2, dim=1)
            # prob = torch.softmax(output_0, dim=1)

        prob_max = prob[:, seg_help.config.classes - 1, :, :]
        prob_max_up = prob_max.unsqueeze(1)
        prob_max_up = F.interpolate(prob_max_up,
                                    size=(cropped_im.shape[0], cropped_im.shape[1]),
                                    mode='bilinear',
                                    align_corners=True)
        prob_max_ = torch.squeeze(prob_max).cpu().numpy()
        fine_pred = np.where(prob_max_ > thres, 1, 0)
        prob_max_up = torch.squeeze(prob_max_up).cpu().numpy()
        fine_pred_up = np.where(prob_max_up > thres, 1, 0)
        fine_pred_up_final = np.where(prob_max_up > thres, 1, 0)
        # fine_pred_final = ndimage.binary_dilation(fine_pred, iterations=1).astype(fine_pred.dtype)
        # fine_pred_up_final = ndimage.binary_dilation(fine_pred_up, iterations=1).astype(fine_pred_up.dtype)

        if cam is not None:
            su_cam_g = cam.detach()[:, seg_help.config.classes - 1, :, :].unsqueeze(1)
            su_cam_g = F.interpolate(su_cam_g, size=su_data.size()[2:], mode='bilinear', align_corners=True)
            su_cam_g = make_grid(su_cam_g, nrow=nrow, pad_value=1)
            su_cam_g = su_cam_g[0, :, :].unsqueeze(0).unsqueeze(0)
            su_data_g = su_data.detach().clone()
            su_data_g = make_grid(su_data_g, nrow=nrow, pad_value=1)
            su_data_g = su_data_g.unsqueeze(0)
            su_cam_g = seg_help.max_norm_cam(su_cam_g)
            heatmap, result_pp = seg_help.visualize_cam(su_cam_g.cpu(), su_data_g.cpu())
            visualize(np.clip(np.transpose(result_pp.detach().cpu().numpy(), (1, 2, 0)), 0, 1),
                      join(output_dir,
                           "test_" + str(iii) + '_cam'))

        image_save = prob_max.unsqueeze(0).contiguous()
        visual_batch(image_save, output_dir, "test_" + str(iii) + "_prob", channel=1, nrow=nrow)

        visual_batch(su_data, output_dir, "test_" + str(iii) + "_images", channel=1, nrow=nrow)

        visual_batch(su_label, output_dir, "test_" + str(iii) + "_label", channel=1, nrow=nrow)

        # fine_pred_torch = torch.from_numpy(fine_pred_final).unsqueeze(0).contiguous()
        # visual_batch(fine_pred_torch, output_dir, str(iii) + "_test_predict", channel=1, nrow=nrow)

        fine_pred_orgi_res = np.zeros((img.shape[0], img.shape[1]), dtype='float32')
        fine_pred_orgi_res[minx: maxx, miny: maxy] = fine_pred_up_final
        # if np.sum(fine_pred_orgi_res)==0:
        #     continue
        scores = compute_all_metric_for_single_seg(infection, fine_pred_orgi_res)
        for metric in segmentation_metrics:
            if metric not in lesion_segmentation_scores:
                lesion_segmentation_scores[metric] = []
            lesion_segmentation_scores[metric].extend(scores[metric])
        empty_cache()
    lesion_segmentation_metrics = {}
    info = ''
    for m in lesion_segmentation_scores:
        # print(lesion_segmentation_scores[m])
        lesion_segmentation_metrics[m] = np.mean(lesion_segmentation_scores[m])
        info += (m + ': {val:.9f} \t  '.format(val=lesion_segmentation_metrics[m]))
    print(info)
    print('Test Dice : {val:.9f}  Thres {thre:.2f}  '.format(val=lesion_segmentation_metrics['DICESCORE'], thre=thres))
    zipDir(output_dir, output_dir + '.zip')
    shutil.rmtree(output_dir)
    return lesion_segmentation_metrics['DICESCORE']


# total 2d slice without seq
def predict_inf_inwindow_2d_out_seq_weight_cam_50(seg_help, model, model_cam, covid_test_data, thres=0.7, epoch=0):
    seg_help.model.eval()
    model.eval()
    model_cam.eval()
    resize_x = seg_help.config.patch_x
    resize_y = seg_help.config.patch_y
    segmentation_metrics = {
        'Jaccard': 0, 'HD': 0,
        'F1': 0, 'ACCURACY': 0, 'SENSITIVITY': 0, 'SPECIFICITY': 0,
        'DICESCORE': 0}
    lesion_segmentation_scores = {}
    img_list_ = covid_test_data[0]
    lung_list = covid_test_data[1]
    inf_list = covid_test_data[2]
    output_dir = join(seg_help.config.tmp_dir, 'infer_thres%.2f_%d_40s' % (thres, epoch))
    mkdir_if_not_exist([output_dir])
    semi_inf = os.listdir('../../log/3DCOVIDCT/Semi-Inf-Net/')
    for iii in range(len(img_list_)):
        # print(file_path)
        img = img_list_[iii].copy()
        lung = lung_list[iii].copy()
        infection = inf_list[iii].copy()
        if np.sum(infection) == 0:
            continue
        if str(iii) + '.png' not in semi_inf:
            continue

        img_semi = Image.open('../../log/3DCOVIDCT/Semi-Inf-Net/' + str(iii) + '.png')
        img_semi = np.array(img_semi)
        img_semi = np.where(img_semi > 128, 1, 0)
        img_semi = resize(img_semi, (resize_x, resize_y), order=0, mode='edge',
                          cval=0, clip=True, preserve_range=True, anti_aliasing=False)

        nrow = 8
        minx, maxx, miny, maxy = min_max_voi_2d(lung, superior=0, inferior=0)
        cropped_im = img[minx: maxx, miny: maxy]
        cropped_if = infection[minx: maxx, miny: maxy]
        resized_img = resize(cropped_im, (resize_x, resize_y), order=3, mode='constant',
                             cval=0, clip=True, preserve_range=True, anti_aliasing=False)
        resized_infection = resize(cropped_if, (resize_x, resize_y), order=0, mode='edge',
                                   cval=0, clip=True, preserve_range=True, anti_aliasing=False)
        su_data = torch.from_numpy(resized_img)
        su_label = torch.from_numpy(resized_infection)
        su_data = su_data.unsqueeze(0).unsqueeze(0)
        su_label = su_label.unsqueeze(0).unsqueeze(0)
        cropp_img = su_data
        cropp_img = cropp_img
        patches_img = cropp_img.to(seg_help.equipment).float()
        with torch.no_grad():
            bb, cc, xx, yy = patches_img.size()
            patches_img = patches_img.view(bb * cc, 1, xx, yy)
            cam_p = model_cam.forward_cam(patches_img)
            out1, out2, cam = model(patches_img, cam_p)
            xxx = 0
            output = None
            for pred in out1:
                if output is None:
                    output = torch.zeros(pred.size()).to(seg_help.equipment)
                output += pred
                xxx += 1
                pred = torch.softmax(pred, dim=1)
                pred_max = pred[:, seg_help.config.classes - 1, :, :]
                image_save = pred_max.unsqueeze(0).contiguous()
                # visual_batch(image_save, output_dir, "test_" + str(iii) + "_prob_" + str(xxx), channel=1, nrow=nrow)
            output_0 = output
            cam_p = model_cam.forward_cam(seg_help.fliplr(patches_img))
            out1, out2, cam = model(seg_help.fliplr(patches_img), cam_p)
            output = None
            for pred in out1:
                if output is None:
                    output = torch.zeros(pred.size()).to(seg_help.equipment)
                output += seg_help.fliplr(pred)
            output_1 = output
            cam_p = model_cam.forward_cam(seg_help.flipvr(patches_img))
            out1, out2, cam = model(seg_help.flipvr(patches_img), cam_p)
            output = None
            for pred in out1:
                if output is None:
                    output = torch.zeros(pred.size()).to(seg_help.equipment)
                output += seg_help.flipvr(pred)
            output_2 = output
            prob = torch.softmax(output_0 + output_1 + output_2, dim=1)
            # prob = torch.softmax(output_0, dim=1)

        prob_max = prob[:, seg_help.config.classes - 1, :, :]
        prob_max_up = prob_max.unsqueeze(1)
        prob_max_up = F.interpolate(prob_max_up,
                                    size=(cropped_im.shape[0], cropped_im.shape[1]),
                                    mode='bilinear',
                                    align_corners=True)
        prob_max_ = torch.squeeze(prob_max).cpu().numpy()
        fine_pred = np.where(prob_max_ > thres, 1, 0)
        prob_max_up = torch.squeeze(prob_max_up).cpu().numpy()
        fine_pred_up = np.where(prob_max_up > thres, 1, 0)
        fine_pred_up_final = np.where(prob_max_up > thres, 1, 0)
        # fine_pred_final = ndimage.binary_dilation(fine_pred, iterations=1).astype(fine_pred.dtype)
        # fine_pred_up_final = ndimage.binary_dilation(fine_pred_up, iterations=1).astype(fine_pred_up.dtype)

        if cam is not None:
            su_cam_g = cam.detach()[:, seg_help.config.classes - 1, :, :].unsqueeze(1)
            su_cam_g = F.interpolate(su_cam_g, size=su_data.size()[2:], mode='bilinear', align_corners=True)
            su_cam_g = make_grid(su_cam_g, nrow=nrow, pad_value=1)
            su_cam_g = su_cam_g[0, :, :].unsqueeze(0).unsqueeze(0)
            su_data_g = su_data.detach().clone()
            su_data_g = make_grid(su_data_g, nrow=nrow, pad_value=1)
            su_data_g = su_data_g.unsqueeze(0)
            su_cam_g = seg_help.max_norm_cam(su_cam_g)
            heatmap, result_pp = seg_help.visualize_cam(su_cam_g.cpu(), su_data_g.cpu())
            visualize(np.clip(np.transpose(result_pp.detach().cpu().numpy(), (1, 2, 0)), 0, 1),
                      join(output_dir,
                           "test_" + str(iii) + '_cam'))

        image_save = prob_max.unsqueeze(0).contiguous()
        visual_batch(image_save, output_dir, "test_" + str(iii) + "_prob", channel=1, nrow=nrow)

        visual_batch(su_data, output_dir, "test_" + str(iii) + "_images", channel=1, nrow=nrow)

        visual_batch(su_label, output_dir, "test_" + str(iii) + "_label", channel=1, nrow=nrow)

        impaintshow(su_data.cpu().numpy(), su_label.cpu().numpy(), fine_pred, output_dir,
                    "test_" + str(iii) + "_overlay")

        # fine_pred_torch = torch.from_numpy(fine_pred_final).unsqueeze(0).contiguous()
        # visual_batch(fine_pred_torch, output_dir, str(iii) + "_test_predict", channel=1, nrow=nrow)

        fine_pred_orgi_res = np.zeros((img.shape[0], img.shape[1]), dtype='float32')
        fine_pred_orgi_res[minx: maxx, miny: maxy] = fine_pred_up_final

        scores = compute_all_metric_for_single_seg(infection, fine_pred_orgi_res)
        for metric in segmentation_metrics:
            if metric not in lesion_segmentation_scores:
                lesion_segmentation_scores[metric] = []
            lesion_segmentation_scores[metric].extend(scores[metric])
        empty_cache()
    lesion_segmentation_metrics = {}
    info = ''
    for m in lesion_segmentation_scores:
        # print(lesion_segmentation_scores[m])
        lesion_segmentation_metrics[m] = np.mean(lesion_segmentation_scores[m])
        info += (m + ': {val:.9f} \t  '.format(val=lesion_segmentation_metrics[m]))
    print(info)
    print('50 Slice Test Dice : {val:.9f}  Thres {thre:.2f}  '.format(val=lesion_segmentation_metrics['DICESCORE'],
                                                                      thre=thres))
    zipDir(output_dir, output_dir + '.zip')
    shutil.rmtree(output_dir)
    return lesion_segmentation_metrics['DICESCORE']


# total 2d slice without seq
def predict_inf_inwindow_2d_out_seq_cam(seg_help, model, covid_test_data, thres=0.7, epoch=0):
    seg_help.model.eval()
    model.eval()
    resize_x = seg_help.config.patch_x
    resize_y = seg_help.config.patch_y
    segmentation_metrics = {
        'Jaccard': 0, 'HD': 0,
        'F1': 0, 'ACCURACY': 0, 'SENSITIVITY': 0, 'SPECIFICITY': 0,
        'DICESCORE': 0}
    lesion_segmentation_scores = {}
    img_list_ = covid_test_data[0]
    lung_list = covid_test_data[1]
    inf_list = covid_test_data[2]
    output_dir = join(seg_help.config.tmp_dir, 'infer_thres%.2f_%d' % (thres, epoch))
    mkdir_if_not_exist([output_dir])
    for iii in range(len(img_list_)):
        # print(file_path)
        img = img_list_[iii].copy()
        lung = lung_list[iii].copy()
        infection = inf_list[iii].copy()
        nrow = 8
        minx, maxx, miny, maxy = min_max_voi_2d(lung, superior=10, inferior=10)
        cropped_im = img[minx: maxx, miny: maxy]
        cropped_if = infection[minx: maxx, miny: maxy]
        resized_img = resize(cropped_im, (resize_x, resize_y), order=3, mode='constant',
                             cval=0, clip=True, preserve_range=True, anti_aliasing=False)
        resized_infection = resize(cropped_if, (resize_x, resize_y), order=0, mode='edge',
                                   cval=0, clip=True, preserve_range=True, anti_aliasing=False)
        su_data = torch.from_numpy(resized_img)
        su_label = torch.from_numpy(resized_infection)
        su_data = su_data.unsqueeze(0).unsqueeze(0)
        su_label = su_label.unsqueeze(0).unsqueeze(0)
        cropp_img = su_data
        cropp_img = cropp_img
        patches_img = cropp_img.to(seg_help.equipment).float()
        with torch.no_grad():
            bb, cc, xx, yy = patches_img.size()
            patches_img = patches_img.view(bb * cc, 1, xx, yy)
            _, _, cam = model(patches_img)

        su_cam_g = cam.detach()[:, seg_help.config.classes - 1, :, :].unsqueeze(1)
        su_cam_g = F.interpolate(su_cam_g, size=su_data.size()[2:], mode='bilinear', align_corners=True)
        su_cam_g = make_grid(su_cam_g, nrow=nrow, pad_value=1)
        su_cam_g = su_cam_g[0, :, :].unsqueeze(0).unsqueeze(0)
        su_data_g = su_data.detach().clone()
        su_data_g = make_grid(su_data_g, nrow=nrow, pad_value=1)
        su_data_g = su_data_g.unsqueeze(0)
        su_cam_g = seg_help.max_norm_cam(su_cam_g)
        heatmap, result_pp = seg_help.visualize_cam(su_cam_g.cpu(), su_data_g.cpu())
        visualize(np.clip(np.transpose(result_pp.detach().cpu().numpy(), (1, 2, 0)), 0, 1),
                  join(output_dir,
                       "test_" + str(iii) + '_cam'))
        visual_batch(su_label, output_dir, "test_" + str(iii) + "_label", channel=1, nrow=nrow)

    zipDir(output_dir, output_dir + '.zip')
    shutil.rmtree(output_dir)


# total 2d slice without seq
def predict_inf_inwindow_2d_out_seq_weight_cam_infer(seg_help, model, model_cam, covid_test_data, thres=0.7, epoch=0):
    seg_help.model.eval()
    model.eval()
    model_cam.eval()
    resize_x = seg_help.config.patch_x
    resize_y = seg_help.config.patch_y
    segmentation_metrics = {
        'Jaccard': 0, 'HD': 0,
        'F1': 0, 'ACCURACY': 0, 'SENSITIVITY': 0, 'SPECIFICITY': 0,
        'DICESCORE': 0}
    lesion_segmentation_scores = {}
    img_list_ = covid_test_data[0]
    lung_list = covid_test_data[1]
    inf_list = covid_test_data[2]
    output_dir = join(seg_help.config.tmp_dir, 'infer_thres%.2f_%d' % (thres, epoch))
    mkdir_if_not_exist([output_dir])
    for iii in range(len(img_list_)):
        # print(file_path)
        img = img_list_[iii].copy()
        lung = lung_list[iii].copy()
        infection = inf_list[iii].copy()
        if np.sum(infection) == 0:
            continue
        nrow = 8
        minx, maxx, miny, maxy = min_max_voi_2d(lung, superior=10, inferior=10)
        cropped_im = img[minx: maxx, miny: maxy]
        cropped_if = infection[minx: maxx, miny: maxy]
        resized_img = resize(cropped_im, (resize_x, resize_y), order=3, mode='constant',
                             cval=0, clip=True, preserve_range=True, anti_aliasing=False)
        resized_infection = resize(cropped_if, (resize_x, resize_y), order=0, mode='edge',
                                   cval=0, clip=True, preserve_range=True, anti_aliasing=False)
        su_data = torch.from_numpy(resized_img)
        su_label = torch.from_numpy(resized_infection)
        su_data = su_data.unsqueeze(0).unsqueeze(0)
        su_label = su_label.unsqueeze(0).unsqueeze(0)
        cropp_img = su_data
        cropp_img = cropp_img
        patches_img = cropp_img.to(seg_help.equipment).float()
        with torch.no_grad():
            bb, cc, xx, yy = patches_img.size()
            patches_img = patches_img.view(bb * cc, 1, xx, yy)
            cam_p = model_cam.forward_cam(patches_img)
            out1, out2, cam = model.forward_seg(patches_img, cam_p)
            output = None
            xxx = 0
            for pred in out1:
                if output is None:
                    output = torch.zeros(pred.size()).to(seg_help.equipment)
                output += pred
                xxx += 1
                pred = torch.softmax(pred, dim=1)
                pred_max = pred[:, seg_help.config.classes - 1, :, :]
                image_save = pred_max.unsqueeze(0).contiguous()
                visual_batch(image_save, output_dir, "test_" + str(iii) + "_prob_" + str(xxx), channel=1, nrow=nrow)

            output_0 = output
            cam_p = model_cam.forward_cam(seg_help.fliplr(patches_img))
            out1, out2, cam = model.forward_seg(seg_help.fliplr(patches_img), cam_p)
            output = None
            for pred in out1:
                if output is None:
                    output = torch.zeros(pred.size()).to(seg_help.equipment)
                output += seg_help.fliplr(pred)
            output_1 = output
            cam_p = model_cam.forward_cam(seg_help.flipvr(patches_img))
            out1, out2, cam = model.forward_seg(seg_help.flipvr(patches_img), cam_p)
            output = None
            for pred in out1:
                if output is None:
                    output = torch.zeros(pred.size()).to(seg_help.equipment)
                output += seg_help.flipvr(pred)
            output_2 = output
            prob = torch.softmax(output_0 + output_1 + output_2, dim=1)
            # prob = torch.softmax(output_0 + output_1, dim=1)

        prob_max = prob[:, seg_help.config.classes - 1, :, :]
        prob_max_up = prob_max.unsqueeze(1)
        prob_max_up = F.interpolate(prob_max_up,
                                    size=(cropped_im.shape[0], cropped_im.shape[1]),
                                    mode='bilinear',
                                    align_corners=True)
        prob_max_ = torch.squeeze(prob_max).cpu().numpy()
        fine_pred = np.where(prob_max_ > thres, 1, 0)
        prob_max_up = torch.squeeze(prob_max_up).cpu().numpy()
        fine_pred_up = np.where(prob_max_up > thres, 1, 0)
        fine_pred_up_final = np.where(prob_max_up > thres, 1, 0)
        # fine_pred_final = ndimage.binary_dilation(fine_pred, iterations=1).astype(fine_pred.dtype)
        # fine_pred_up_final = ndimage.binary_dilation(fine_pred_up, iterations=1).astype(fine_pred_up.dtype)

        if cam is not None:
            su_cam_g = cam.detach()[:, seg_help.config.classes - 1, :, :].unsqueeze(1)
            su_cam_g = F.interpolate(su_cam_g, size=su_data.size()[2:], mode='bilinear', align_corners=True)
            su_cam_g = make_grid(su_cam_g, nrow=nrow, pad_value=1)
            su_cam_g = su_cam_g[0, :, :].unsqueeze(0).unsqueeze(0)
            su_data_g = su_data.detach().clone()
            su_data_g = make_grid(su_data_g, nrow=nrow, pad_value=1)
            su_data_g = su_data_g.unsqueeze(0)
            su_cam_g = seg_help.max_norm_cam(su_cam_g)
            heatmap, result_pp = seg_help.visualize_cam(su_cam_g.cpu(), su_data_g.cpu())
            visualize(np.clip(np.transpose(result_pp.detach().cpu().numpy(), (1, 2, 0)), 0, 1),
                      join(output_dir,
                           "test_" + str(iii) + '_cam'))

        image_save = prob_max.unsqueeze(0).contiguous()
        visual_batch(image_save, output_dir, "test_" + str(iii) + "_prob", channel=1, nrow=nrow)

        visual_batch(su_data, output_dir, "test_" + str(iii) + "_images", channel=1, nrow=nrow)

        visual_batch(su_label, output_dir, "test_" + str(iii) + "_label", channel=1, nrow=nrow)

        # fine_pred_torch = torch.from_numpy(fine_pred_final).unsqueeze(0).contiguous()
        # visual_batch(fine_pred_torch, output_dir, str(iii) + "_test_predict", channel=1, nrow=nrow)

        fine_pred_orgi_res = np.zeros((img.shape[0], img.shape[1]), dtype='float32')
        fine_pred_orgi_res[minx: maxx, miny: maxy] = fine_pred_up_final

        scores = compute_all_metric_for_single_seg(infection, fine_pred_orgi_res)
        for metric in segmentation_metrics:
            if metric not in lesion_segmentation_scores:
                lesion_segmentation_scores[metric] = []
            lesion_segmentation_scores[metric].extend(scores[metric])
        empty_cache()
    lesion_segmentation_metrics = {}
    info = ''
    for m in lesion_segmentation_scores:
        # print(lesion_segmentation_scores[m])
        lesion_segmentation_metrics[m] = np.mean(lesion_segmentation_scores[m])
        info += (m + ': {val:.9f} \t  '.format(val=lesion_segmentation_metrics[m]))
    # print(info)
    print('Test Dice : {val:.9f}  Thres {thre:.2f}  '.format(val=lesion_segmentation_metrics['DICESCORE'], thre=thres))
    zipDir(output_dir, output_dir + '.zip')
    shutil.rmtree(output_dir)
    return lesion_segmentation_metrics['DICESCORE']


# total 2d slice without seq
def predict_inf_inwindow_2d_out_seq_weight_cam_50_infer(seg_help, model, model_cam, covid_test_data, thres=0.7,
                                                        epoch=0):
    seg_help.model.eval()
    model.eval()
    model_cam.eval()
    resize_x = seg_help.config.patch_x
    resize_y = seg_help.config.patch_y
    segmentation_metrics = {
        'Jaccard': 0, 'HD': 0,
        'F1': 0, 'ACCURACY': 0, 'SENSITIVITY': 0, 'SPECIFICITY': 0,
        'DICESCORE': 0}
    lesion_segmentation_scores = {}
    img_list_ = covid_test_data[0]
    lung_list = covid_test_data[1]
    inf_list = covid_test_data[2]
    output_dir = join(seg_help.config.tmp_dir, 'infer_thres%.2f_%d_40s' % (thres, epoch))
    mkdir_if_not_exist([output_dir])
    semi_inf = os.listdir('../../log/3DCOVIDCT/Semi-Inf-Net/')
    for iii in range(len(img_list_)):
        # print(file_path)
        img = img_list_[iii].copy()
        lung = lung_list[iii].copy()
        infection = inf_list[iii].copy()
        if np.sum(infection) == 0:
            continue
        if str(iii) + '.png' not in semi_inf:
            continue
        nrow = 8
        minx, maxx, miny, maxy = min_max_voi_2d(lung, superior=10, inferior=10)
        cropped_im = img[minx: maxx, miny: maxy]
        cropped_if = infection[minx: maxx, miny: maxy]
        resized_img = resize(cropped_im, (resize_x, resize_y), order=3, mode='constant',
                             cval=0, clip=True, preserve_range=True, anti_aliasing=False)
        resized_infection = resize(cropped_if, (resize_x, resize_y), order=0, mode='edge',
                                   cval=0, clip=True, preserve_range=True, anti_aliasing=False)
        su_data = torch.from_numpy(resized_img)
        su_label = torch.from_numpy(resized_infection)
        su_data = su_data.unsqueeze(0).unsqueeze(0)
        su_label = su_label.unsqueeze(0).unsqueeze(0)
        cropp_img = su_data
        cropp_img = cropp_img
        patches_img = cropp_img.to(seg_help.equipment).float()
        with torch.no_grad():
            bb, cc, xx, yy = patches_img.size()
            patches_img = patches_img.view(bb * cc, 1, xx, yy)
            cam_p = model_cam.forward_cam(patches_img)
            out1, out2, cam = model.forward_seg(patches_img, cam_p)
            xxx = 0
            output = None
            for pred in out1:
                if output is None:
                    output = torch.zeros(pred.size()).to(seg_help.equipment)
                output += pred
                xxx += 1
                pred = torch.softmax(pred, dim=1)
                pred_max = pred[:, seg_help.config.classes - 1, :, :]
                image_save = pred_max.unsqueeze(0).contiguous()
                visual_batch(image_save, output_dir, "test_" + str(iii) + "_prob_" + str(xxx), channel=1, nrow=nrow)
            output_0 = output
            cam_p = model_cam.forward_cam(seg_help.fliplr(patches_img))
            out1, out2, cam = model.forward_seg(seg_help.fliplr(patches_img), cam_p)
            output = None
            for pred in out1:
                if output is None:
                    output = torch.zeros(pred.size()).to(seg_help.equipment)
                output += seg_help.fliplr(pred)
            output_1 = output
            cam_p = model_cam.forward_cam(seg_help.flipvr(patches_img))
            out1, out2, cam = model.forward_seg(seg_help.flipvr(patches_img), cam_p)
            output = None
            for pred in out1:
                if output is None:
                    output = torch.zeros(pred.size()).to(seg_help.equipment)
                output += seg_help.flipvr(pred)
            output_2 = output
            prob = torch.softmax(output_0 + output_1 + output_2, dim=1)
            # prob = torch.softmax(output_0 + output_1, dim=1)
        prob_max = prob[:, seg_help.config.classes - 1, :, :]
        prob_max_up = prob_max.unsqueeze(1)
        prob_max_up = F.interpolate(prob_max_up,
                                    size=(cropped_im.shape[0], cropped_im.shape[1]),
                                    mode='bilinear',
                                    align_corners=True)
        prob_max_ = torch.squeeze(prob_max).cpu().numpy()
        fine_pred = np.where(prob_max_ > thres, 1, 0)
        prob_max_up = torch.squeeze(prob_max_up).cpu().numpy()
        fine_pred_up = np.where(prob_max_up > thres, 1, 0)
        fine_pred_up_final = np.where(prob_max_up > thres, 1, 0)
        # fine_pred_final = ndimage.binary_dilation(fine_pred, iterations=1).astype(fine_pred.dtype)
        # fine_pred_up_final = ndimage.binary_dilation(fine_pred_up, iterations=1).astype(fine_pred_up.dtype)

        if cam is not None:
            su_cam_g = cam.detach()[:, seg_help.config.classes - 1, :, :].unsqueeze(1)
            su_cam_g = F.interpolate(su_cam_g, size=su_data.size()[2:], mode='bilinear', align_corners=True)
            su_cam_g = make_grid(su_cam_g, nrow=nrow, pad_value=1)
            su_cam_g = su_cam_g[0, :, :].unsqueeze(0).unsqueeze(0)
            su_data_g = su_data.detach().clone()
            su_data_g = make_grid(su_data_g, nrow=nrow, pad_value=1)
            su_data_g = su_data_g.unsqueeze(0)
            su_cam_g = seg_help.max_norm_cam(su_cam_g)
            heatmap, result_pp = seg_help.visualize_cam(su_cam_g.cpu(), su_data_g.cpu())
            visualize(np.clip(np.transpose(result_pp.detach().cpu().numpy(), (1, 2, 0)), 0, 1),
                      join(output_dir,
                           "test_" + str(iii) + '_cam'))

        image_save = prob_max.unsqueeze(0).contiguous()
        visual_batch(image_save, output_dir, "test_" + str(iii) + "_prob", channel=1, nrow=nrow)

        visual_batch(su_data, output_dir, "test_" + str(iii) + "_images", channel=1, nrow=nrow)

        visual_batch(su_label, output_dir, "test_" + str(iii) + "_label", channel=1, nrow=nrow)

        # fine_pred_torch = torch.from_numpy(fine_pred_final).unsqueeze(0).contiguous()
        # visual_batch(fine_pred_torch, output_dir, str(iii) + "_test_predict", channel=1, nrow=nrow)

        fine_pred_orgi_res = np.zeros((img.shape[0], img.shape[1]), dtype='float32')
        fine_pred_orgi_res[minx: maxx, miny: maxy] = fine_pred_up_final

        scores = compute_all_metric_for_single_seg(infection, fine_pred_orgi_res)
        for metric in segmentation_metrics:
            if metric not in lesion_segmentation_scores:
                lesion_segmentation_scores[metric] = []
            lesion_segmentation_scores[metric].extend(scores[metric])
        empty_cache()
    lesion_segmentation_metrics = {}
    info = ''
    for m in lesion_segmentation_scores:
        # print(lesion_segmentation_scores[m])
        lesion_segmentation_metrics[m] = np.mean(lesion_segmentation_scores[m])
        info += (m + ': {val:.9f} \t  '.format(val=lesion_segmentation_metrics[m]))
    # print(info)
    print('50 Slice Test Dice : {val:.9f}  Thres {thre:.2f}  '.format(val=lesion_segmentation_metrics['DICESCORE'],
                                                                      thre=thres))
    zipDir(output_dir, output_dir + '.zip')
    shutil.rmtree(output_dir)
    return lesion_segmentation_metrics['DICESCORE']


# total 2d slice without seq
def predict_inf_inwindow_2d_out_seq_weight_cam_infer_stu(seg_help, model, model_seg, model_cam, covid_test_data,
                                                         thres=0.7,
                                                         epoch=0):
    seg_help.model.eval()
    model.eval()
    model_cam.eval()
    model_seg.eval()
    resize_x = seg_help.config.patch_x
    resize_y = seg_help.config.patch_y
    segmentation_metrics = {
        'Jaccard': 0, 'HD': 0,
        'F1': 0, 'ACCURACY': 0, 'SENSITIVITY': 0, 'SPECIFICITY': 0,
        'DICESCORE': 0}
    lesion_segmentation_scores = {}
    img_list_ = covid_test_data[0]
    lung_list = covid_test_data[1]
    inf_list = covid_test_data[2]
    output_dir = join(seg_help.config.tmp_dir, 'infer_thres%.2f_%d' % (thres, epoch))
    mkdir_if_not_exist([output_dir])
    for iii in range(len(img_list_)):
        # print(file_path)
        img = img_list_[iii].copy()
        lung = lung_list[iii].copy()
        infection = inf_list[iii].copy()
        if np.sum(infection) == 0:
            continue
        nrow = 8
        minx, maxx, miny, maxy = min_max_voi_2d(lung, superior=10, inferior=10)
        cropped_im = img[minx: maxx, miny: maxy]
        cropped_if = infection[minx: maxx, miny: maxy]
        resized_img = resize(cropped_im, (resize_x, resize_y), order=3, mode='constant',
                             cval=0, clip=True, preserve_range=True, anti_aliasing=False)
        resized_infection = resize(cropped_if, (resize_x, resize_y), order=0, mode='edge',
                                   cval=0, clip=True, preserve_range=True, anti_aliasing=False)
        su_data = torch.from_numpy(resized_img)
        su_label = torch.from_numpy(resized_infection)
        su_data = su_data.unsqueeze(0).unsqueeze(0)
        su_label = su_label.unsqueeze(0).unsqueeze(0)
        cropp_img = su_data
        cropp_img = cropp_img
        patches_img = cropp_img.to(seg_help.equipment).float()
        with torch.no_grad():
            bb, cc, xx, yy = patches_img.size()
            patches_img = patches_img.view(bb * cc, 1, xx, yy)
            # 1
            cam_p = model_cam.forward_cam(patches_img)
            out1 = model_seg.forward_seg(patches_img, cam_p)
            output = None
            for pred in out1:
                if output is None:
                    output = torch.zeros(pred.size()).to(seg_help.equipment)
                output += pred
            output_o = output

            out1 = model.forward_seg(patches_img, cam_p)
            output = None
            xxx = 0
            for pred in out1:
                if output is None:
                    output = torch.zeros(pred.size()).to(seg_help.equipment)
                output += pred
                xxx += 1
                pred = torch.softmax(pred, dim=1)
                pred_max = pred[:, seg_help.config.classes - 1, :, :]
                image_save = pred_max.unsqueeze(0).contiguous()
                visual_batch(image_save, output_dir, "test_" + str(iii) + "_prob_" + str(xxx), channel=1, nrow=nrow)
            output_0 = output_o + output
            # 2
            cam_p = model_cam.forward_cam(seg_help.fliplr(patches_img))
            out1 = model_seg.forward_seg(seg_help.fliplr(patches_img), cam_p)
            output = None
            for pred in out1:
                if output is None:
                    output = torch.zeros(pred.size()).to(seg_help.equipment)
                output += seg_help.fliplr(pred)
            output_o = output

            out1 = model.forward_seg(seg_help.fliplr(patches_img), cam_p)
            output = None
            for pred in out1:
                if output is None:
                    output = torch.zeros(pred.size()).to(seg_help.equipment)
                output += seg_help.fliplr(pred)
            output_1 = output_o + output

            # 3
            cam_p = model_cam.forward_cam(seg_help.flipvr(patches_img))
            out1 = model_seg.forward_seg(seg_help.flipvr(patches_img), cam_p)
            output = None
            for pred in out1:
                if output is None:
                    output = torch.zeros(pred.size()).to(seg_help.equipment)
                output += seg_help.flipvr(pred)
            output_o = output

            out1 = model.forward_seg(seg_help.flipvr(patches_img), cam_p)
            output = None
            for pred in out1:
                if output is None:
                    output = torch.zeros(pred.size()).to(seg_help.equipment)
                output += seg_help.flipvr(pred)
            output_2 = output_o + output
            # patches_img = patches_img.view(bb * cc, 1, xx, yy)
            # cam_p = model_cam.forward_cam(patches_img)
            # out1, out2, cam = model_seg(patches_img, cam_p)
            # xxx = 0
            # output = None
            # for pred in out1:
            #     if output is None:
            #         output = torch.zeros(pred.size()).to(seg_help.equipment)
            #     output += pred
            #     xxx += 1
            #     pred = torch.softmax(pred, dim=1)
            #     pred_max = pred[:, seg_help.config.classes - 1, :, :]
            #     image_save = pred_max.unsqueeze(0).contiguous()
            #     visual_batch(image_save, output_dir, "test_" + str(iii) + "_prob_" + str(xxx), channel=1, nrow=nrow)
            # output_0 = output
            # cam_p = model_cam.forward_cam(seg_help.fliplr(patches_img))
            # out1, out2, cam = model_seg(seg_help.fliplr(patches_img), cam_p)
            # output = None
            # for pred in out1:
            #     if output is None:
            #         output = torch.zeros(pred.size()).to(seg_help.equipment)
            #     output += seg_help.fliplr(pred)
            # output_1 = output
            # cam_p = model_cam.forward_cam(seg_help.flipvr(patches_img))
            # out1, out2, cam = model_seg(seg_help.flipvr(patches_img), cam_p)
            # output = None
            # for pred in out1:
            #     if output is None:
            #         output = torch.zeros(pred.size()).to(seg_help.equipment)
            #     output += seg_help.flipvr(pred)
            # output_2 = output

            prob = torch.softmax(output_0 + output_1 + output_2, dim=1)

        prob_max = prob[:, seg_help.config.classes - 1, :, :]
        prob_max_up = prob_max.unsqueeze(1)
        prob_max_up = F.interpolate(prob_max_up,
                                    size=(cropped_im.shape[0], cropped_im.shape[1]),
                                    mode='bilinear',
                                    align_corners=True)
        prob_max_ = torch.squeeze(prob_max).cpu().numpy()
        fine_pred = np.where(prob_max_ > thres, 1, 0)
        prob_max_up = torch.squeeze(prob_max_up).cpu().numpy()
        fine_pred_up = np.where(prob_max_up > thres, 1, 0)
        fine_pred_up_final = np.where(prob_max_up > thres, 1, 0)
        # fine_pred_final = ndimage.binary_dilation(fine_pred, iterations=1).astype(fine_pred.dtype)
        # fine_pred_up_final = ndimage.binary_dilation(fine_pred_up, iterations=1).astype(fine_pred_up.dtype)

        # if cam is not None:
        #     su_cam_g = cam.detach()[:, seg_help.config.classes - 1, :, :].unsqueeze(1)
        #     su_cam_g = F.interpolate(su_cam_g, size=su_data.size()[2:], mode='bilinear', align_corners=True)
        #     su_cam_g = make_grid(su_cam_g, nrow=nrow, pad_value=1)
        #     su_cam_g = su_cam_g[0, :, :].unsqueeze(0).unsqueeze(0)
        #     su_data_g = su_data.detach().clone()
        #     su_data_g = make_grid(su_data_g, nrow=nrow, pad_value=1)
        #     su_data_g = su_data_g.unsqueeze(0)
        #     su_cam_g = seg_help.max_norm_cam(su_cam_g)
        #     heatmap, result_pp = seg_help.visualize_cam(su_cam_g.cpu(), su_data_g.cpu())
        #     visualize(np.clip(np.transpose(result_pp.detach().cpu().numpy(), (1, 2, 0)), 0, 1),
        #               join(output_dir,
        #                    "test_" + str(iii) + '_cam'))

        image_save = prob_max.unsqueeze(0).contiguous()
        visual_batch(image_save, output_dir, "test_" + str(iii) + "_prob", channel=1, nrow=nrow)

        visual_batch(su_data, output_dir, "test_" + str(iii) + "_images", channel=1, nrow=nrow)

        visual_batch(su_label, output_dir, "test_" + str(iii) + "_label", channel=1, nrow=nrow)

        # fine_pred_torch = torch.from_numpy(fine_pred_final).unsqueeze(0).contiguous()
        # visual_batch(fine_pred_torch, output_dir, str(iii) + "_test_predict", channel=1, nrow=nrow)

        fine_pred_orgi_res = np.zeros((img.shape[0], img.shape[1]), dtype='float32')
        fine_pred_orgi_res[minx: maxx, miny: maxy] = fine_pred_up_final

        scores = compute_all_metric_for_single_seg(infection, fine_pred_orgi_res)
        for metric in segmentation_metrics:
            if metric not in lesion_segmentation_scores:
                lesion_segmentation_scores[metric] = []
            lesion_segmentation_scores[metric].extend(scores[metric])
        empty_cache()
    lesion_segmentation_metrics = {}
    info = ''
    for m in lesion_segmentation_scores:
        # print(lesion_segmentation_scores[m])
        lesion_segmentation_metrics[m] = np.mean(lesion_segmentation_scores[m])
        info += (m + ': {val:.9f} \t  '.format(val=lesion_segmentation_metrics[m]))
    # print(info)
    print('Test Dice : {val:.9f}  Thres {thre:.2f}  '.format(val=lesion_segmentation_metrics['DICESCORE'], thre=thres))
    zipDir(output_dir, output_dir + '.zip')
    shutil.rmtree(output_dir)
    return lesion_segmentation_metrics['DICESCORE']


# total 2d slice without seq
def predict_inf_inwindow_2d_out_seq_weight_cam_50_infer_stu(seg_help, model, model_seg, model_cam, covid_test_data,
                                                            thres=0.7,
                                                            epoch=0):
    seg_help.model.eval()
    model.eval()
    model_cam.eval()
    model_seg.eval()
    resize_x = seg_help.config.patch_x
    resize_y = seg_help.config.patch_y
    segmentation_metrics = {
        'Jaccard': 0, 'HD': 0,
        'F1': 0, 'ACCURACY': 0, 'SENSITIVITY': 0, 'SPECIFICITY': 0,
        'DICESCORE': 0}
    lesion_segmentation_scores = {}
    img_list_ = covid_test_data[0]
    lung_list = covid_test_data[1]
    inf_list = covid_test_data[2]
    output_dir = join(seg_help.config.tmp_dir, 'infer_thres%.2f_%d_40s' % (thres, epoch))
    mkdir_if_not_exist([output_dir])
    semi_inf = os.listdir('../../log/3DCOVIDCT/Semi-Inf-Net/')
    for iii in range(len(img_list_)):
        # print(file_path)
        img = img_list_[iii].copy()
        lung = lung_list[iii].copy()
        infection = inf_list[iii].copy()
        if np.sum(infection) == 0:
            continue
        if str(iii) + '.png' not in semi_inf:
            continue
        nrow = 8
        minx, maxx, miny, maxy = min_max_voi_2d(lung, superior=10, inferior=10)
        cropped_im = img[minx: maxx, miny: maxy]
        cropped_if = infection[minx: maxx, miny: maxy]
        resized_img = resize(cropped_im, (resize_x, resize_y), order=3, mode='constant',
                             cval=0, clip=True, preserve_range=True, anti_aliasing=False)
        resized_infection = resize(cropped_if, (resize_x, resize_y), order=0, mode='edge',
                                   cval=0, clip=True, preserve_range=True, anti_aliasing=False)
        su_data = torch.from_numpy(resized_img)
        su_label = torch.from_numpy(resized_infection)
        su_data = su_data.unsqueeze(0).unsqueeze(0)
        su_label = su_label.unsqueeze(0).unsqueeze(0)
        cropp_img = su_data
        cropp_img = cropp_img
        patches_img = cropp_img.to(seg_help.equipment).float()
        with torch.no_grad():
            bb, cc, xx, yy = patches_img.size()
            patches_img = patches_img.view(bb * cc, 1, xx, yy)
            # 1
            cam_p = model_cam.forward_cam(patches_img)
            out1 = model_seg.forward_seg(patches_img, cam_p)
            output = None
            for pred in out1:
                if output is None:
                    output = torch.zeros(pred.size()).to(seg_help.equipment)
                output += pred
            output_o = output

            out1 = model.forward_seg(patches_img, cam_p)
            output = None
            xxx = 0
            for pred in out1:
                if output is None:
                    output = torch.zeros(pred.size()).to(seg_help.equipment)
                output += pred
                xxx += 1
                pred = torch.softmax(pred, dim=1)
                pred_max = pred[:, seg_help.config.classes - 1, :, :]
                image_save = pred_max.unsqueeze(0).contiguous()
                visual_batch(image_save, output_dir, "test_" + str(iii) + "_prob_" + str(xxx), channel=1, nrow=nrow)
            output_0 = output_o + output
            # 2
            cam_p = model_cam.forward_cam(seg_help.fliplr(patches_img))
            out1 = model_seg.forward_seg(seg_help.fliplr(patches_img), cam_p)
            output = None
            for pred in out1:
                if output is None:
                    output = torch.zeros(pred.size()).to(seg_help.equipment)
                output += seg_help.fliplr(pred)
            output_o = output

            out1 = model.forward_seg(seg_help.fliplr(patches_img), cam_p)
            output = None
            for pred in out1:
                if output is None:
                    output = torch.zeros(pred.size()).to(seg_help.equipment)
                output += seg_help.fliplr(pred)
            output_1 = output_o + output

            # 3
            cam_p = model_cam.forward_cam(seg_help.flipvr(patches_img))
            out1 = model_seg.forward_seg(seg_help.flipvr(patches_img), cam_p)
            output = None
            for pred in out1:
                if output is None:
                    output = torch.zeros(pred.size()).to(seg_help.equipment)
                output += seg_help.flipvr(pred)
            output_o = output

            out1 = model.forward_seg(seg_help.flipvr(patches_img), cam_p)
            output = None
            for pred in out1:
                if output is None:
                    output = torch.zeros(pred.size()).to(seg_help.equipment)
                output += seg_help.flipvr(pred)
            output_2 = output_o + output
            # patches_img = patches_img.view(bb * cc, 1, xx, yy)
            # cam_p = model_cam.forward_cam(patches_img)
            # out1, out2, cam = model_seg(patches_img, cam_p)
            # xxx = 0
            # output = None
            # for pred in out1:
            #     if output is None:
            #         output = torch.zeros(pred.size()).to(seg_help.equipment)
            #     output += pred
            #     xxx += 1
            #     pred = torch.softmax(pred, dim=1)
            #     pred_max = pred[:, seg_help.config.classes - 1, :, :]
            #     image_save = pred_max.unsqueeze(0).contiguous()
            #     visual_batch(image_save, output_dir, "test_" + str(iii) + "_prob_" + str(xxx), channel=1, nrow=nrow)
            # output_0 = output
            # cam_p = model_cam.forward_cam(seg_help.fliplr(patches_img))
            # out1, out2, cam = model_seg(seg_help.fliplr(patches_img), cam_p)
            # output = None
            # for pred in out1:
            #     if output is None:
            #         output = torch.zeros(pred.size()).to(seg_help.equipment)
            #     output += seg_help.fliplr(pred)
            # output_1 = output
            # cam_p = model_cam.forward_cam(seg_help.flipvr(patches_img))
            # out1, out2, cam = model_seg(seg_help.flipvr(patches_img), cam_p)
            # output = None
            # for pred in out1:
            #     if output is None:
            #         output = torch.zeros(pred.size()).to(seg_help.equipment)
            #     output += seg_help.flipvr(pred)
            # output_2 = output
            prob = torch.softmax(output_0 + output_1 + output_2, dim=1)
        prob_max = prob[:, seg_help.config.classes - 1, :, :]
        prob_max_up = prob_max.unsqueeze(1)
        prob_max_up = F.interpolate(prob_max_up,
                                    size=(cropped_im.shape[0], cropped_im.shape[1]),
                                    mode='bilinear',
                                    align_corners=True)
        prob_max_ = torch.squeeze(prob_max).cpu().numpy()
        fine_pred = np.where(prob_max_ > thres, 1, 0)
        prob_max_up = torch.squeeze(prob_max_up).cpu().numpy()
        fine_pred_up = np.where(prob_max_up > thres, 1, 0)
        fine_pred_up_final = np.where(prob_max_up > thres, 1, 0)
        # fine_pred_final = ndimage.binary_dilation(fine_pred, iterations=1).astype(fine_pred.dtype)
        # fine_pred_up_final = ndimage.binary_dilation(fine_pred_up, iterations=1).astype(fine_pred_up.dtype)

        # if cam is not None:
        #     su_cam_g = cam.detach()[:, seg_help.config.classes - 1, :, :].unsqueeze(1)
        #     su_cam_g = F.interpolate(su_cam_g, size=su_data.size()[2:], mode='bilinear', align_corners=True)
        #     su_cam_g = make_grid(su_cam_g, nrow=nrow, pad_value=1)
        #     su_cam_g = su_cam_g[0, :, :].unsqueeze(0).unsqueeze(0)
        #     su_data_g = su_data.detach().clone()
        #     su_data_g = make_grid(su_data_g, nrow=nrow, pad_value=1)
        #     su_data_g = su_data_g.unsqueeze(0)
        #     su_cam_g = seg_help.max_norm_cam(su_cam_g)
        #     heatmap, result_pp = seg_help.visualize_cam(su_cam_g.cpu(), su_data_g.cpu())
        #     visualize(np.clip(np.transpose(result_pp.detach().cpu().numpy(), (1, 2, 0)), 0, 1),
        #               join(output_dir,
        #                    "test_" + str(iii) + '_cam'))

        image_save = prob_max.unsqueeze(0).contiguous()
        visual_batch(image_save, output_dir, "test_" + str(iii) + "_prob", channel=1, nrow=nrow)

        visual_batch(su_data, output_dir, "test_" + str(iii) + "_images", channel=1, nrow=nrow)

        visual_batch(su_label, output_dir, "test_" + str(iii) + "_label", channel=1, nrow=nrow)

        # fine_pred_torch = torch.from_numpy(fine_pred_final).unsqueeze(0).contiguous()
        # visual_batch(fine_pred_torch, output_dir, str(iii) + "_test_predict", channel=1, nrow=nrow)

        fine_pred_orgi_res = np.zeros((img.shape[0], img.shape[1]), dtype='float32')
        fine_pred_orgi_res[minx: maxx, miny: maxy] = fine_pred_up_final

        scores = compute_all_metric_for_single_seg(infection, fine_pred_orgi_res)
        for metric in segmentation_metrics:
            if metric not in lesion_segmentation_scores:
                lesion_segmentation_scores[metric] = []
            lesion_segmentation_scores[metric].extend(scores[metric])
        empty_cache()
    lesion_segmentation_metrics = {}
    info = ''
    for m in lesion_segmentation_scores:
        # print(lesion_segmentation_scores[m])
        lesion_segmentation_metrics[m] = np.mean(lesion_segmentation_scores[m])
        info += (m + ': {val:.9f} \t  '.format(val=lesion_segmentation_metrics[m]))
    # print(info)
    print('50 Slice Test Dice : {val:.9f}  Thres {thre:.2f}  '.format(val=lesion_segmentation_metrics['DICESCORE'],
                                                                      thre=thres))
    zipDir(output_dir, output_dir + '.zip')
    shutil.rmtree(output_dir)
    return lesion_segmentation_metrics['DICESCORE']
