# -*- coding: utf-8 -*-
# @Time    : 20/5/1 16:58
# @Author  : qgking
# @Email   : qgking@tju.edu.cn
# @Software: PyCharm
# @Desc    : LungSegDataLoader.py
from skimage.transform import resize
from torch.utils.data import Dataset
from common.base_utls import *
from common.data_utils import *
import torch


class LungCoarseSegDataset(Dataset):
    def __init__(self, root_dir, split_pickle=None, input_size=(256, 256, 64), generate_each=6):
        self.input_x = input_size[0]
        self.input_y = input_size[1]
        self.input_z = input_size[2]
        self.root_dir = root_dir
        self.generate_each = generate_each
        self.img_list = []
        self.minindex_list = []
        self.maxindex_list = []
        self.lungidx = []
        self.lunglines = []
        img_list = sorted(glob(join(root_dir, '*.npy')), reverse=True)
        for idx in range(len(img_list)):
            file_name = basename(img_list[idx])[:-4]
            if file_name not in split_pickle:
                continue
            self.img_list.append(img_list[idx])
        print(self.img_list)

    def __len__(self):
        return int(self.generate_each * len(self.img_list))

    def __getitem__(self, index):
        # while True:
        scans = np.load(self.img_list[index // self.generate_each])
        # print(self.img_list[index // self.generate_each])
        img = scans[0]
        tumor = scans[1]
        #  randomly scale
        scale = np.random.uniform(0.8, 1.2)
        cols = int(self.input_z * scale)
        c = int(np.random.randint(cols // 2, img.shape[-1] - cols / 2 - 1))
        # print('start co %d, end co %d, im shape z %d' % (c - cols // 2, c + cols // 2, img.shape[-1]))
        cropp_img = img[:, :, c - cols // 2: c + cols // 2].copy()
        cropp_tumor = tumor[:, :, c - cols // 2:c + cols // 2].copy()
        return self.__agumentation__(cropp_img, cropp_tumor)

    def __agumentation__(self, cropp_img, cropp_tumor):
        flip_num = np.random.randint(0, 8)
        if flip_num == 1:
            cropp_img = np.flipud(cropp_img)
            cropp_tumor = np.flipud(cropp_tumor)
        elif flip_num == 2:
            cropp_img = np.fliplr(cropp_img)
            cropp_tumor = np.fliplr(cropp_tumor)
        elif flip_num == 3:
            cropp_img = np.rot90(cropp_img, k=1, axes=(1, 0))
            cropp_tumor = np.rot90(cropp_tumor, k=1, axes=(1, 0))
        elif flip_num == 4:
            cropp_img = np.rot90(cropp_img, k=3, axes=(1, 0))
            cropp_tumor = np.rot90(cropp_tumor, k=3, axes=(1, 0))
        elif flip_num == 5:
            cropp_img = np.fliplr(cropp_img)
            cropp_tumor = np.fliplr(cropp_tumor)
            cropp_img = np.rot90(cropp_img, k=1, axes=(1, 0))
            cropp_tumor = np.rot90(cropp_tumor, k=1, axes=(1, 0))
        elif flip_num == 6:
            cropp_img = np.fliplr(cropp_img)
            cropp_tumor = np.fliplr(cropp_tumor)
            cropp_img = np.rot90(cropp_img, k=3, axes=(1, 0))
            cropp_tumor = np.rot90(cropp_tumor, k=3, axes=(1, 0))
        elif flip_num == 7:
            cropp_img = np.flipud(cropp_img)
            cropp_tumor = np.flipud(cropp_tumor)
            cropp_img = np.fliplr(cropp_img)
            cropp_tumor = np.fliplr(cropp_tumor)
        cropp_tumor = resize(cropp_tumor, (self.input_x, self.input_y, self.input_z), order=0, mode='edge',
                             cval=0, clip=True, preserve_range=True, anti_aliasing=False)
        cropp_img = resize(cropp_img, (self.input_x, self.input_y, self.input_z), order=3, mode='constant',
                           cval=0, clip=True, preserve_range=True, anti_aliasing=False)
        cropp_tumor = np.where(cropp_tumor > 0, 1, 0)
        cropp_img = torch.from_numpy(np.expand_dims(cropp_img, 0))
        cropp_tumor = torch.from_numpy(cropp_tumor)
        return {
            "image_patch": cropp_img,
            "image_segment": cropp_tumor,
        }


class LungSegDataset(Dataset):
    def __init__(self, root_dir, split_pickle=None, input_size=(256, 256, 64), generate_each=6):
        self.input_x = input_size[0]
        self.input_y = input_size[1]
        self.input_z = input_size[2]
        self.root_dir = root_dir
        self.generate_each = generate_each
        self.img_list = []
        self.minindex_list = []
        self.maxindex_list = []
        self.lungidx = []
        self.lunglines = []
        img_list = sorted(glob(join(root_dir, '*.npy')), reverse=True)
        for idx in range(len(img_list)):
            file_name = basename(img_list[idx])[:-4]
            if file_name not in split_pickle:
                continue
            # if idx > 3:
            #     break
            self.img_list.append(img_list[idx])
            scans = np.load(img_list[idx])
            values = np.loadtxt(join(root_dir, file_name + '_lung.txt'), delimiter=' ')
            minindex = np.min(values, axis=0)
            maxindex = np.max(values, axis=0)
            minindex = np.array(minindex, dtype='int')
            maxindex = np.array(maxindex, dtype='int')
            minindex[0] = max(minindex[0] - 3, 0)
            minindex[1] = max(minindex[1] - 3, 0)
            minindex[2] = max(minindex[2] - 3, 0)
            maxindex[0] = min(scans[0].shape[0], maxindex[0] + 3)
            maxindex[1] = min(scans[0].shape[1], maxindex[1] + 3)
            maxindex[2] = min(scans[0].shape[2], maxindex[2] + 3)
            self.minindex_list.append(minindex)
            self.maxindex_list.append(maxindex)
            f2 = open(join(root_dir, file_name + '_lung.txt'), 'r')
            lungline = f2.readlines()
            self.lunglines.append(lungline)
            self.lungidx.append(len(lungline))
            f2.close()
            del scans
        print(self.img_list)

    def __len__(self):
        return int(self.generate_each * len(self.img_list))

    def __getitem__(self, index):
        # while True:
        count = index // self.generate_each
        scans = np.load(self.img_list[count])
        img = scans[0]
        tumor = scans[1]
        minindex = self.minindex_list[count]
        maxindex = self.maxindex_list[count]
        lines = self.lunglines[count]
        numid = self.lungidx[count]
        #  randomly scale
        scale = np.random.uniform(0.8, 1.2)
        deps = int(self.input_x * scale)
        rows = int(self.input_y * scale)
        cols = int(self.input_z * scale)

        sed = np.random.randint(1, numid)
        # print('index %d, img_num %d, minindex_list %d, sed %d, lungidx %d' % (
        #     index, count, len(self.minindex_list), sed, len(self.lungidx)))
        cen = lines[sed - 1]
        cen = np.fromstring(cen, dtype=int, sep=' ')
        a = int(min(max(minindex[0] + deps / 2, cen[0]), maxindex[0] - deps / 2 - 1))
        b = int(min(max(minindex[1] + rows / 2, cen[1]), maxindex[1] - rows / 2 - 1))
        c = int(min(max(minindex[2] + cols / 2, cen[2]), maxindex[2] - cols / 2 - 1))
        cropp_img = img[a - deps // 2:a + deps // 2, b - rows // 2:b + rows // 2,
                    c - cols // 2: c + cols // 2].copy()
        cropp_tumor = tumor[a - deps // 2:a + deps // 2, b - rows // 2:b + rows // 2,
                      c - cols // 2:c + cols // 2].copy()

        return self.__agumentation__(cropp_img, cropp_tumor)

    def __agumentation__(self, cropp_img, cropp_tumor):
        flip_num = np.random.randint(0, 8)
        if flip_num == 1:
            cropp_img = np.flipud(cropp_img)
            cropp_tumor = np.flipud(cropp_tumor)
        elif flip_num == 2:
            cropp_img = np.fliplr(cropp_img)
            cropp_tumor = np.fliplr(cropp_tumor)
        elif flip_num == 3:
            cropp_img = np.rot90(cropp_img, k=1, axes=(1, 0))
            cropp_tumor = np.rot90(cropp_tumor, k=1, axes=(1, 0))
        elif flip_num == 4:
            cropp_img = np.rot90(cropp_img, k=3, axes=(1, 0))
            cropp_tumor = np.rot90(cropp_tumor, k=3, axes=(1, 0))
        elif flip_num == 5:
            cropp_img = np.fliplr(cropp_img)
            cropp_tumor = np.fliplr(cropp_tumor)
            cropp_img = np.rot90(cropp_img, k=1, axes=(1, 0))
            cropp_tumor = np.rot90(cropp_tumor, k=1, axes=(1, 0))
        elif flip_num == 6:
            cropp_img = np.fliplr(cropp_img)
            cropp_tumor = np.fliplr(cropp_tumor)
            cropp_img = np.rot90(cropp_img, k=3, axes=(1, 0))
            cropp_tumor = np.rot90(cropp_tumor, k=3, axes=(1, 0))
        elif flip_num == 7:
            cropp_img = np.flipud(cropp_img)
            cropp_tumor = np.flipud(cropp_tumor)
            cropp_img = np.fliplr(cropp_img)
            cropp_tumor = np.fliplr(cropp_tumor)
        cropp_tumor = resize(cropp_tumor, (self.input_x, self.input_y, self.input_z), order=0, mode='edge',
                             cval=0, clip=True, preserve_range=True, anti_aliasing=False)
        cropp_img = resize(cropp_img, (self.input_x, self.input_y, self.input_z), order=3, mode='constant',
                           cval=0, clip=True, preserve_range=True, anti_aliasing=False)
        cropp_img = torch.from_numpy(np.expand_dims(cropp_img, 0))
        cropp_tumor = torch.from_numpy(cropp_tumor)
        return {
            "image_patch": cropp_img,
            "image_segment": cropp_tumor,
        }
