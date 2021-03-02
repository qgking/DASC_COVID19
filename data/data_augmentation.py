# -*- coding: utf-8 -*-
# @Time    : 20/5/21 10:51
# @Author  : qgking
# @Email   : qgking@tju.edu.cn
# @Software: PyCharm
# @Desc    : data_augmentation.py
from skimage.transform import resize
from torch.utils.data import Dataset
from common.base_utls import *
from common.data_utils import *
import torch
from torchvision import transforms
from torchvision.utils import make_grid
import random


def rescale_crop(image, num, output_size=(224, 224)):
    image_list = []
    trans = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.RandomCrop((int(h * scale), int(w * scale))),
        transforms.Resize(output_size),
        # transforms.RandomAffine([-30, 30], translate=[0.01, 0.01], shear=[-10, 10],
        #                         scale=[0.9, 1.1]),
        transforms.RandomAffine([-45, 45], translate=[0.01, 0.01], shear=[-10, 10],
                                scale=[0.8, 1.2]),
        transforms.CenterCrop(output_size),
        transforms.RandomRotation((-90, 90)),
        transforms.Resize(output_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomChoice([
            transforms.ColorJitter(brightness=0.1),
            transforms.ColorJitter(contrast=0.2),
            transforms.ColorJitter(saturation=0.1),
            transforms.ColorJitter(hue=0.15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        ]),
    ])
    for i in range(num):
        img = trans(image)
        image_list.append(img)
    return image_list

def rescale_crop_seg(image, num, output_size=(224, 224)):
    image_list = []
    trans = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.RandomCrop((int(h * scale), int(w * scale))),
        transforms.Resize(output_size),
        # transforms.RandomAffine([-30, 30], translate=[0.01, 0.01], shear=[-10, 10],
        #                         scale=[0.9, 1.1]),
        transforms.RandomAffine([-45, 45], translate=[0.01, 0.01], shear=[-10, 10],
                                scale=[0.8, 1.2]),
        transforms.CenterCrop(output_size),
        transforms.RandomRotation((-90, 90)),
        transforms.Resize(output_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ])
    for i in range(num):
        img = trans(image)
        image_list.append(img)
    return image_list

def rescale(image, num, output_size=(224, 224)):
    image_list = []
    trans = transforms.Compose([
        transforms.Resize(output_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ])
    for i in range(num):
        img = trans(image)
        image_list.append(img)
    return image_list


def agumentation_img_inf_2d(cropp_img, cropp_infection, input_x, input_y, mean, std, num=2):
    image_list = []
    gt_list = []
    x, y, z = cropp_img.shape
    idxx = int(np.floor(z / 2))
    # save_dir = '/home/qgking/COVID3DSeg/log/3DCOVIDCT/deeplab2d/inf_da_1_run_dapt_from_50_to_20_reszie_v1_tttt/tmp'
    # cropp_pppp = np.expand_dims(np.transpose(cropp_img, (2, 0, 1)), axis=0)
    # visual_batch(torch.from_numpy(cropp_pppp), save_dir, "test_img_process_back_crop_cc", channel=1, nrow=8)
    # cropp_pppp = np.expand_dims(np.transpose(cropp_infection, (2, 0, 1)), axis=0)
    # visual_batch(torch.from_numpy(cropp_pppp), save_dir, "test_img_process_back_crop_gt", channel=1, nrow=8)
    cropp_img = (cropp_img * 255.).astype('uint8')
    cropp_infection = (cropp_infection * 255.).astype('uint8')
    for ii in range(num):
        seed = np.random.randint(2147483647)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        image_list0_img = rescale_crop(Image.fromarray(cropp_img), 1, output_size=(input_x, input_y))
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        image_list0_gt = rescale_crop_seg(Image.fromarray(cropp_infection), 1, output_size=(input_x, input_y))
        image_list += image_list0_img
        gt_list += image_list0_gt
    # seed = np.random.randint(2147483647)
    # np.random.seed(seed)
    # random.seed(seed)
    # torch.manual_seed(seed)
    # image_list += rescale(Image.fromarray(cropp_img), 1, output_size=(input_x, input_y))
    # np.random.seed(seed)
    # random.seed(seed)
    # torch.manual_seed(seed)
    # gt_list += rescale(Image.fromarray(cropp_infection), 1, output_size=(input_x, input_y))

    nomalize_img = transforms.Lambda(lambda crops: torch.stack(
        [transforms.Compose([transforms.ToTensor()])(crop) for crop
         in
         crops]))
    nomalize_gt = transforms.Lambda(lambda crops: torch.stack(
        [transforms.Compose([transforms.ToTensor()])(crop) for crop
         in
         crops]))
    image_list = nomalize_img(image_list)
    gt_list = nomalize_gt(gt_list)
    # TODO only one slice
    image_list = image_list[:, idxx, :, :].unsqueeze(1)
    gt_list = gt_list[:, idxx, :, :].unsqueeze(1)
    gt_list[gt_list > 0] = 1

    # visual_batch(image_list, save_dir, "test_img_process_back_crop_image_list", channel=1, nrow=8)
    # visual_batch(gt_list, save_dir, "test_img_process_back_crop_gt_list", channel=1, nrow=8)
    # print('aaa')
    # image_list = image_list.permute(1, 2, 3, 0)
    # image_list = (image_list - mean) / std
    # gt_list = gt_list.permute(1, 2, 3, 0)
    return {
        "image_patch": image_list,
        "image_segment": gt_list,
    }


def agumentation_img_3d(cropp_img, input_x, input_y, input_z):
    flip_num = np.random.randint(0, 8)
    if flip_num == 1:
        cropp_img = np.flipud(cropp_img)
    elif flip_num == 2:
        cropp_img = np.fliplr(cropp_img)
    elif flip_num == 3:
        cropp_img = np.rot90(cropp_img, k=1, axes=(1, 0))
    elif flip_num == 4:
        cropp_img = np.rot90(cropp_img, k=3, axes=(1, 0))
    elif flip_num == 5:
        cropp_img = np.fliplr(cropp_img)
        cropp_img = np.rot90(cropp_img, k=1, axes=(1, 0))
    elif flip_num == 6:
        cropp_img = np.fliplr(cropp_img)
        cropp_img = np.rot90(cropp_img, k=3, axes=(1, 0))
    elif flip_num == 7:
        cropp_img = np.flipud(cropp_img)
        cropp_img = np.fliplr(cropp_img)
    cropp_img = resize(cropp_img, (input_x, input_y, input_z), order=3, mode='constant',
                       cval=0, clip=True, preserve_range=True, anti_aliasing=False)
    return cropp_img


def agumentation_img_inf_3d(cropp_img, cropp_infection, input_x, input_y, input_z):
    flip_num = np.random.randint(0, 8)
    # save_dir = '/home/qgking/COVID3DSeg/log/3DCOVIDCT/deeplab2d/inf_da_1_run_dapt_from_50_to_20_reszie_v1_tttt/tmp'
    # cropp_img_t = torch.from_numpy(np.expand_dims(cropp_img, 0))
    # cropp_infection_t = torch.from_numpy(cropp_infection)
    # cropp_pppp = cropp_img_t.unsqueeze(0)
    # cropp_iiii = cropp_infection_t.unsqueeze(0).unsqueeze(0)
    # visual_batch(cropp_pppp, save_dir, "test_img_aug_a", channel=1, nrow=8)
    # visual_batch(cropp_iiii, save_dir, "test_gt_aug_a", channel=1, nrow=8)
    if flip_num == 1:
        cropp_img = np.flipud(cropp_img)
        cropp_infection = np.flipud(cropp_infection)
    elif flip_num == 2:
        cropp_img = np.fliplr(cropp_img)
        cropp_infection = np.fliplr(cropp_infection)
    elif flip_num == 3:
        cropp_img = np.rot90(cropp_img, k=1, axes=(1, 0))
        cropp_infection = np.rot90(cropp_infection, k=1, axes=(1, 0))
    elif flip_num == 4:
        cropp_img = np.rot90(cropp_img, k=3, axes=(1, 0))
        cropp_infection = np.rot90(cropp_infection, k=3, axes=(1, 0))
    elif flip_num == 5:
        cropp_img = np.fliplr(cropp_img)
        cropp_infection = np.fliplr(cropp_infection)
        cropp_img = np.rot90(cropp_img, k=1, axes=(1, 0))
        cropp_infection = np.rot90(cropp_infection, k=1, axes=(1, 0))
    elif flip_num == 6:
        cropp_img = np.fliplr(cropp_img)
        cropp_infection = np.fliplr(cropp_infection)
        cropp_img = np.rot90(cropp_img, k=3, axes=(1, 0))
        cropp_infection = np.rot90(cropp_infection, k=3, axes=(1, 0))
    elif flip_num == 7:
        cropp_img = np.flipud(cropp_img)
        cropp_infection = np.flipud(cropp_infection)
        cropp_img = np.fliplr(cropp_img)
        cropp_infection = np.fliplr(cropp_infection)

    cropp_infection = resize(cropp_infection, (input_x, input_y, input_z), order=0, mode='edge',
                             cval=0, clip=True, preserve_range=True, anti_aliasing=False)
    cropp_img = resize(cropp_img, (input_x, input_y, input_z), order=3, mode='constant',
                       cval=0, clip=True, preserve_range=True, anti_aliasing=False)
    # print(np.unique(cropp_infection))
    # cropp_img_t = torch.from_numpy(np.expand_dims(cropp_img, 0))
    # cropp_infection_t = torch.from_numpy(cropp_infection)
    # cropp_pppp = cropp_img_t.unsqueeze(0)
    # cropp_iiii = cropp_infection_t.unsqueeze(0).unsqueeze(0)
    # visual_batch(cropp_pppp, save_dir, "test_img_aug_c", channel=1, nrow=8)
    # visual_batch(cropp_iiii, save_dir, "test_gt_aug_c", channel=1, nrow=8)

    return {
        "image_patch": cropp_img,
        "image_segment": cropp_infection,
    }


class seg_bagging_aug(object):
    def __init__(self, generate_bag):
        self.generate_bag = generate_bag

    def __call__(self, img, pseudo, generate_bag, input_x, input_y, input_z):
        minx, maxx, miny, maxy, minz, maxz = min_max_voi(pseudo, superior=5, inferior=5)
        final_imgs = []
        for idx in range(generate_bag):
            #  randomly scale
            scale = np.random.uniform(0.8, 1.2)
            deps = int(input_x * scale)
            rows = int(input_y * scale)
            cols = int(input_z)

            a = np.random.randint(minx + deps // 2, maxx - deps // 2 - 1)
            b = np.random.randint(miny + rows // 2, maxy - rows // 2 - 1)
            c = np.random.randint(minz + cols // 2, maxz - cols // 2 - 1)
            # print(c)c
            a = a if a - deps // 2 >= 0 else deps // 2
            a = a if a + deps // 2 < img.shape[0] else img.shape[0] - deps // 2 - 1
            b = b if b - rows // 2 >= 0 else rows // 2
            b = b if b + rows // 2 < img.shape[1] else img.shape[1] - rows // 2 - 1
            c = c if c - cols // 2 >= 0 else cols // 2
            c = c if c + cols // 2 < img.shape[2] else img.shape[2] - cols // 2 - 1
            # print(c)
            # print(minindex)
            # print(maxindex)
            # print(cen)
            cropp_img = img[a - deps // 2:a + deps // 2, b - rows // 2:b + rows // 2,
                        c - cols // 2: c + cols // 2].copy()
            augment_img = agumentation_img_3d(cropp_img, input_x, input_y, input_z)
            final_imgs.append(augment_img)
        image_list = torch.Tensor(final_imgs)
        image_list = torch.unsqueeze(image_list, dim=1)
        # image_save = image_list.transpose(1, 4).contiguous().view(
        #     (image_list.size(0) * image_list.size(-1), image_list.size(1), image_list.size(2), image_list.size(3)))
        # grid = make_grid(image_save, nrow=16, padding=2)
        # save_img = grid.detach().cpu().numpy()
        # visualize(np.transpose(save_img, (1, 2, 0)), join('./tmp', "aug_images"))
        return image_list


def agumentation_img_inf_2d_for_infer(cropp_img, mean, std, transform):
    image_list = []
    x, y, z = cropp_img.shape
    # save_dir = '/home/qgking/COVID3DSeg/log/3DCOVIDCT/u2net_multi_cam/inf_da_2_run_dapt_from_20_to_50_reszie_v1_ttttt/tmp'
    # cropp_pppp = np.expand_dims(np.transpose(cropp_img, (2, 0, 1)), axis=0)
    # visual_batch(torch.from_numpy(cropp_pppp), save_dir, "test_img_process_back_crop_cc", channel=1, nrow=8)
    # cropp_pppp = np.expand_dims(np.transpose(cropp_infection, (2, 0, 1)), axis=0)
    # visual_batch(torch.from_numpy(cropp_pppp), save_dir, "test_img_process_back_crop_gt", channel=1, nrow=8)
    cropp_img = (cropp_img * 255.).astype('uint8')
    for ii in range(z):
        image_list0_img = transform(Image.fromarray(cropp_img[:, :, ii]))
        image_list += image_list0_img
    nomalize_img = transforms.Lambda(lambda crops: torch.stack(
        [transforms.Compose([transforms.ToTensor()])(crop) for crop
         in
         crops]))
    image_list = nomalize_img(image_list)
    image_list = image_list.permute(1, 2, 3, 0)
    image_list = (image_list - mean) / std
    return image_list[0, :, :, :]
