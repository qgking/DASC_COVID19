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
from torchvision import transforms
from torchvision.utils import make_grid
from data.data_augmentation import *


# --------------5fold start-------------
class CovidInf5foldDatasetBase(Dataset):
    def __init__(self, root_dir, img_list, input_size, generate_each, mean, std, pos):
        self.input_x = input_size[0]
        self.input_y = input_size[1]
        self.input_z = input_size[2]
        self.root_dir = root_dir
        self.pos = pos
        self.generate_each = generate_each
        self.img_list = []
        self.minindex_list = []
        self.maxindex_list = []
        self.infidx = []
        self.inflines = []
        self.mean = mean
        self.std = std
        print('mean %.8f  std %.8f' % (self.mean, self.std))
        for idx in range(len(img_list)):
            # if idx > 1:
            #     break
            file_name = basename(img_list[idx])[:-4]
            print(img_list[idx])
            self.img_list.append(img_list[idx])
            scans = np.load(img_list[idx])
            txt_path = join(root_dir, file_name + '_inf.txt')
            if not exists(txt_path):
                txt_path = join(root_dir, file_name[1:] + '_inf.txt')
            values = np.loadtxt(txt_path, delimiter=' ')
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
            f2 = open(txt_path, 'r')
            liverline = f2.readlines()
            self.inflines.append(liverline)
            self.infidx.append(len(liverline))
            f2.close()
            del scans

    def __len__(self):
        return int(self.generate_each * len(self.img_list))

    def __getitem__(self, index):
        return None


# resize 5 fold
class CovidInf5fold2dAugSegDataset(CovidInf5foldDatasetBase):
    def __init__(self, root_dir, img_list, input_size, generate_each, mean, std, pos):
        super(CovidInf5fold2dAugSegDataset, self).__init__(root_dir, img_list, input_size,
                                                           generate_each, mean, std, pos)

    def __getitem__(self, index):
        # while True:
        count = index // self.generate_each
        scans = np.load(self.img_list[count])
        img = scans[0].copy()
        if 'MosMedData' in self.root_dir:
            lung = scans[2]
            infection = scans[1]
        elif 'COVID-19-CT' in self.root_dir:
            lung = scans[1]
            infection = scans[2]

        minx, maxx, miny, maxy, minz, maxz = min_max_voi(lung, superior=3, inferior=3)
        # tmp = img[minx: maxx, miny: maxy, minz: maxz].copy()
        # tmp = (tmp - self.mean) / self.std
        # img[minx: maxx, miny: maxy, minz: maxz] = tmp
        pos = np.random.random()
        if pos > self.pos:
            # only inf region selected
            # print('only inf region selected')
            minindex = self.minindex_list[count]
            maxindex = self.maxindex_list[count]
            lines = self.inflines[count]
            numid = self.infidx[count]
            scale = np.random.uniform(0.8, 1.2)
            cols = int(self.input_z)

            sed = np.random.randint(1, numid)
            cen = lines[sed - 1]
            cen = np.fromstring(cen, dtype=int, sep=' ')
            c = int(min(max(minindex[2] + cols / 2, cen[2]), maxindex[2] - cols / 2 - 1))
            c = c if c - cols // 2 >= 0 else cols // 2
            c = c if c + cols // 2 < img.shape[-1] else img.shape[-1] - cols // 2 - 1
        else:
            # inf region and none inf region selected
            # print('inf region and none inf region selected')
            scale = np.random.uniform(0.8, 1.2)
            cols = int(self.input_z)
            x = np.random.randint(minx, maxx)
            y = np.random.randint(miny, maxy)
            z = np.random.randint(minz, maxz)
            cen = [x, y, z]
            c = int(min(max(minz + cols / 2, cen[2]), maxz - cols / 2 - 1))
            c = c if c - cols // 2 >= 0 else cols // 2
            c = c if c + cols // 2 < img.shape[-1] else img.shape[-1] - cols // 2 - 1
        flo = int(np.floor(cols / 2))
        cropp_img = img[minx: maxx, miny: maxy, c - flo: c + cols - flo].copy()
        cropp_infection = infection[minx: maxx, miny: maxy, c - flo: c + cols - flo].copy()
        return agumentation_img_inf_2d(cropp_img, cropp_infection, self.input_x, self.input_y, self.mean, self.std)


# resize 5 fold
class CovidInf5fold2dResizeSegDataset(CovidInf5foldDatasetBase):
    def __init__(self, root_dir, img_list, input_size, generate_each, mean, std, pos):
        super(CovidInf5fold2dResizeSegDataset, self).__init__(root_dir, img_list, input_size,
                                                              generate_each, mean, std, pos)

    def __getitem__(self, index):
        # while True:
        count = index // self.generate_each
        scans = np.load(self.img_list[count])
        img = scans[0].copy()
        if 'MosMedData' in self.root_dir:
            lung = scans[2]
            infection = scans[1]
        elif 'COVID-19-CT' in self.root_dir:
            lung = scans[1]
            infection = scans[2]

        minx, maxx, miny, maxy, minz, maxz = min_max_voi(lung, superior=3, inferior=3)
        # tmp = img[minx: maxx, miny: maxy, minz: maxz].copy()
        # tmp = (tmp - self.mean) / self.std
        # img[minx: maxx, miny: maxy, minz: maxz] = tmp
        pos = np.random.random()
        if pos > self.pos:
            # only inf region selected
            # print('only inf region selected')
            minindex = self.minindex_list[count]
            maxindex = self.maxindex_list[count]
            lines = self.inflines[count]
            numid = self.infidx[count]
            scale = np.random.uniform(0.8, 1.2)
            cols = int(self.input_z)

            sed = np.random.randint(1, numid)
            cen = lines[sed - 1]
            cen = np.fromstring(cen, dtype=int, sep=' ')
            c = int(min(max(minindex[2] + cols / 2, cen[2]), maxindex[2] - cols / 2 - 1))
            c = c if c - cols // 2 >= 0 else cols // 2
            c = c if c + cols // 2 < img.shape[-1] else img.shape[-1] - cols // 2 - 1
        else:
            # inf region and none inf region selected
            # print('inf region and none inf region selected')
            scale = np.random.uniform(0.8, 1.2)
            cols = int(self.input_z)
            x = np.random.randint(minx, maxx)
            y = np.random.randint(miny, maxy)
            z = np.random.randint(minz, maxz)
            cen = [x, y, z]
            c = int(min(max(minz + cols / 2, cen[2]), maxz - cols / 2 - 1))
            c = c if c - cols // 2 >= 0 else cols // 2
            c = c if c + cols // 2 < img.shape[-1] else img.shape[-1] - cols // 2 - 1
        flo = int(np.floor(cols / 2))
        cropp_img = img[minx: maxx, miny: maxy, c - flo: c + cols - flo].copy()
        cropp_infection = infection[minx: maxx, miny: maxy, c - flo: c + cols - flo].copy()
        return agumentation_img_inf_3d(cropp_img, cropp_infection, self.input_x, self.input_y, self.input_z)


# crop 5 fold
class CovidInf5fold2dSegDataset(CovidInf5foldDatasetBase):
    def __init__(self, root_dir, img_list, input_size, generate_each, mean, std, pos):
        super(CovidInf5fold2dSegDataset, self).__init__(root_dir, img_list, input_size, generate_each, mean, std, pos)

    def __getitem__(self, index):
        # while True:
        count = index // self.generate_each
        scans = np.load(self.img_list[count])
        img = scans[0].copy()
        if 'MosMedData' in self.root_dir:
            lung = scans[2]
            infection = scans[1]
        elif 'COVID-19-CT' in self.root_dir:
            lung = scans[1]
            infection = scans[2]
        minx, maxx, miny, maxy, minz, maxz = min_max_voi(lung, superior=3, inferior=3)
        # tmp = img[minx: maxx, miny: maxy, minz: maxz].copy()
        # tmp = (tmp - self.mean) / self.std
        # img[minx: maxx, miny: maxy, minz: maxz] = tmp

        minindex = self.minindex_list[count]
        maxindex = self.maxindex_list[count]
        lines = self.inflines[count]
        numid = self.infidx[count]

        scale = np.random.uniform(0.8, 1.2)
        deps = int(self.input_x * scale)
        rows = int(self.input_y * scale)
        cols = int(self.input_z)

        sed = np.random.randint(1, numid)
        cen = lines[sed - 1]
        cen = np.fromstring(cen, dtype=int, sep=' ')
        a = int(min(max(minindex[0] + deps / 2, cen[0]), maxindex[0] - deps / 2 - 1))
        b = int(min(max(minindex[1] + rows / 2, cen[1]), maxindex[1] - rows / 2 - 1))
        c = int(min(max(minindex[2] + cols / 2, cen[2]), maxindex[2] - cols / 2 - 1))
        c = c if c - cols // 2 >= 0 else cols // 2
        c = c if c + cols // 2 < img.shape[-1] else img.shape[-1] - cols // 2 - 1
        flo = int(np.floor(cols / 2))
        cropp_img = img[a - deps // 2:a + deps // 2, b - rows // 2:b + rows // 2,
                    c - flo: c + cols - flo].copy()
        cropp_infection = infection[a - deps // 2:a + deps // 2, b - rows // 2:b + rows // 2,
                          c - flo: c + cols - flo].copy()
        return agumentation_img_inf_3d(cropp_img, cropp_infection, self.input_x, self.input_y, self.input_z)


# --------------5fold end-------------


# --------------UnsuData start-------------
class CovidInfUnsuDatasetBase(Dataset):
    def __init__(self, root_dir, split, input_size, generate_each, mean, std, pos):
        self.input_x = input_size[0]
        self.input_y = input_size[1]
        self.input_z = input_size[2]
        self.root_dir = root_dir
        self.generate_each = generate_each
        self.pos = pos
        self.img_list = []
        self.minindex_list = []
        self.maxindex_list = []
        self.infidx = []
        self.inflines = []
        self.mean = mean
        self.std = std
        print('mean %.8f  std %.8f' % (self.mean, self.std))
        if 'MosMedData' in root_dir:
            img_list = sorted(glob(join(root_dir, 'm*.npy')), reverse=True)
            idx = []
            np.random.seed(666)
            indx = np.random.choice(range(len(img_list)), size=int(len(img_list) * 0.2), replace=False)
            idx.extend(indx)
            if split == 'train':
                img_list = [img_list[ii] for ii in range(len(img_list)) if ii not in idx]
            elif split == 'valid':
                img_list = [img_list[ii] for ii in range(len(img_list)) if ii in idx]
            elif split == None:
                img_list
        elif 'COVID-19-CT' in root_dir:
            img_list = sorted(glob(join(root_dir, '*.npy')), reverse=True)
            idx = []
            np.random.seed(666)
            indx = np.random.choice(range(10), size=2, replace=False)
            idx.extend(indx)
            np.random.seed(666)
            indx = np.random.choice(range(10, 20), size=2, replace=False)
            idx.extend(indx)
            if split == 'train':
                img_list = [img_list[ii] for ii in range(len(img_list)) if ii not in idx]
            elif split == 'valid':
                img_list = [img_list[ii] for ii in range(len(img_list)) if ii in idx]
            elif split == None:
                img_list

        for idx in range(len(img_list)):
            # if idx > 1:
            #     break
            file_name = basename(img_list[idx])[:-4]
            print(img_list[idx])
            self.img_list.append(img_list[idx])
            scans = np.load(img_list[idx])
            txt_path = join(root_dir, file_name + '_inf.txt')
            if not exists(txt_path):
                txt_path = join(root_dir, file_name[1:] + '_inf.txt')
            values = np.loadtxt(txt_path, delimiter=' ')
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
            f2 = open(txt_path, 'r')
            liverline = f2.readlines()
            self.inflines.append(liverline)
            self.infidx.append(len(liverline))
            f2.close()
            del scans

    def __len__(self):
        return int(self.generate_each * len(self.img_list))

    def __getitem__(self, index):
        # while True:
        return None


# resize unsupervised
class CovidInfUnsu2dResizeSegDataset(CovidInfUnsuDatasetBase):
    def __init__(self, root_dir, split, input_size, generate_each, mean, std, pos):
        super(CovidInfUnsu2dResizeSegDataset, self).__init__(root_dir, split, input_size,
                                                             generate_each, mean, std, pos)

    def __getitem__(self, index):
        # while True:
        count = index // self.generate_each
        scans = np.load(self.img_list[count])
        img = scans[0].copy()
        if 'MosMedData' in self.root_dir:
            lung = scans[2]
            infection = scans[1]
        elif 'COVID-19-CT' in self.root_dir:
            lung = scans[1]
            infection = scans[2]

        minx, maxx, miny, maxy, minz, maxz = min_max_voi(lung, superior=3, inferior=3)
        # tmp = img[minx: maxx, miny: maxy, minz: maxz].copy()
        # tmp = (tmp - self.mean) / self.std
        # img[minx: maxx, miny: maxy, minz: maxz] = tmp

        # save_dir = '/home/qgking/COVID3DSeg/log/3DCOVIDCT/deeplab2d/inf_da_0_run_dapt_from_50_to_20_reszie_eeetest/tmp'
        #
        # minx, maxx, miny, maxy, minz, maxz = min_max_voi(lung, superior=3, inferior=3)
        # tmp = img[minx: maxx, miny: maxy, minz: maxz].copy()
        # cropp_pppp = torch.from_numpy(tmp)
        # cropp_pppp = cropp_pppp.unsqueeze(0).unsqueeze(0)
        # visual_batch(cropp_pppp, save_dir, "test_img_tmp", channel=1, nrow=8)
        #
        # tmp = (tmp - self.mean) / self.std
        # img[minx: maxx, miny: maxy, minz: maxz] = tmp
        #
        # cropp_pppp = torch.from_numpy(tmp)
        # cropp_pppp = cropp_pppp.unsqueeze(0).unsqueeze(0)
        # visual_batch(cropp_pppp, save_dir, "test_img", channel=1, nrow=8)
        #
        # cropp_pppp = torch.from_numpy((img * self.std + self.mean)[minx: maxx, miny: maxy, minz: maxz])
        # cropp_pppp = cropp_pppp.unsqueeze(0).unsqueeze(0)
        # visual_batch(cropp_pppp, save_dir, "test_img_process_back_crop", channel=1, nrow=8)

        pos = np.random.random()
        if pos > self.pos:
            # only inf region selected
            # print('only inf region selected')
            minindex = self.minindex_list[count]
            maxindex = self.maxindex_list[count]
            lines = self.inflines[count]
            numid = self.infidx[count]
            scale = np.random.uniform(0.8, 1.2)
            cols = int(self.input_z)

            sed = np.random.randint(1, numid)
            cen = lines[sed - 1]
            cen = np.fromstring(cen, dtype=int, sep=' ')
            c = int(min(max(minindex[2] + cols / 2, cen[2]), maxindex[2] - cols / 2 - 1))
            c = c if c - cols // 2 >= 0 else cols // 2
            c = c if c + cols // 2 < img.shape[-1] else img.shape[-1] - cols // 2 - 1
        else:
            # inf region and none inf region selected
            # print('inf region and none inf region selected')
            scale = np.random.uniform(0.8, 1.2)
            cols = int(self.input_z)
            x = np.random.randint(minx, maxx)
            y = np.random.randint(miny, maxy)
            z = np.random.randint(minz, maxz)
            cen = [x, y, z]
            c = int(min(max(minz + cols / 2, cen[2]), maxz - cols / 2 - 1))
            c = c if c - cols // 2 >= 0 else cols // 2
            c = c if c + cols // 2 < img.shape[-1] else img.shape[-1] - cols // 2 - 1
        flo = int(np.floor(cols / 2))
        cropp_img = img[minx: maxx, miny: maxy, c - flo: c + cols - flo].copy()
        cropp_infection = infection[minx: maxx, miny: maxy, c - flo: c + cols - flo].copy()
        # nbb = agumentation_img_inf_3d(cropp_img, cropp_infection, self.input_x, self.input_y, self.input_z)
        # cropp_pppp = np.expand_dims(np.transpose(nbb['image_patch'], (2, 0, 1)), axis=0)
        # visual_batch(torch.from_numpy(cropp_pppp), save_dir, "test_img_process_back_crop_cc", channel=1, nrow=8)
        return agumentation_img_inf_3d(cropp_img, cropp_infection, self.input_x, self.input_y, self.input_z)


# resize unsupervised slice
class CovidInfUnsu2dAugSegDataset(CovidInfUnsuDatasetBase):
    def __init__(self, root_dir, split, input_size, generate_each, mean, std, pos):
        super(CovidInfUnsu2dAugSegDataset, self).__init__(root_dir, split, input_size,
                                                          generate_each, mean, std, pos)

    def __getitem__(self, index):
        # while True:
        count = index // self.generate_each
        scans = np.load(self.img_list[count])
        img = scans[0].copy()
        if 'MosMedData' in self.root_dir:
            lung = scans[2]
            infection = scans[1]
        elif 'COVID-19-CT' in self.root_dir:
            lung = scans[1]
            infection = scans[2]

        minx, maxx, miny, maxy, minz, maxz = min_max_voi(lung, superior=3, inferior=3)
        # tmp = img[minx: maxx, miny: maxy, minz: maxz].copy()
        # tmp = (tmp - self.mean) / self.std
        # img[minx: maxx, miny: maxy, minz: maxz] = tmp

        # save_dir = '/home/qgking/COVID3DSeg/log/3DCOVIDCT/deeplab2d/inf_da_0_run_dapt_from_50_to_20_reszie_eeetest/tmp'
        #
        # minx, maxx, miny, maxy, minz, maxz = min_max_voi(lung, superior=3, inferior=3)
        # tmp = img[minx: maxx, miny: maxy, minz: maxz].copy()
        # cropp_pppp = torch.from_numpy(tmp)
        # cropp_pppp = cropp_pppp.unsqueeze(0).unsqueeze(0)
        # visual_batch(cropp_pppp, save_dir, "test_img_tmp", channel=1, nrow=8)
        #
        # tmp = (tmp - self.mean) / self.std
        # img[minx: maxx, miny: maxy, minz: maxz] = tmp
        #
        # cropp_pppp = torch.from_numpy(tmp)
        # cropp_pppp = cropp_pppp.unsqueeze(0).unsqueeze(0)
        # visual_batch(cropp_pppp, save_dir, "test_img", channel=1, nrow=8)
        #
        # cropp_pppp = torch.from_numpy((img * self.std + self.mean)[minx: maxx, miny: maxy, minz: maxz])
        # cropp_pppp = cropp_pppp.unsqueeze(0).unsqueeze(0)
        # visual_batch(cropp_pppp, save_dir, "test_img_process_back_crop", channel=1, nrow=8)

        pos = np.random.random()
        if pos > self.pos:
            # only inf region selected
            # print('only inf region selected')
            minindex = self.minindex_list[count]
            maxindex = self.maxindex_list[count]
            lines = self.inflines[count]
            numid = self.infidx[count]
            scale = np.random.uniform(0.8, 1.2)
            cols = int(self.input_z)

            sed = np.random.randint(1, numid)
            cen = lines[sed - 1]
            cen = np.fromstring(cen, dtype=int, sep=' ')
            c = int(min(max(minindex[2] + cols / 2, cen[2]), maxindex[2] - cols / 2 - 1))
            c = c if c - cols // 2 >= 0 else cols // 2
            c = c if c + cols // 2 < img.shape[-1] else img.shape[-1] - cols // 2 - 1
        else:
            # inf region and none inf region selected
            # print('inf region and none inf region selected')
            scale = np.random.uniform(0.8, 1.2)
            cols = int(self.input_z)
            x = np.random.randint(minx, maxx)
            y = np.random.randint(miny, maxy)
            z = np.random.randint(minz, maxz)
            cen = [x, y, z]
            c = int(min(max(minz + cols / 2, cen[2]), maxz - cols / 2 - 1))
            c = c if c - cols // 2 >= 0 else cols // 2
            c = c if c + cols // 2 < img.shape[-1] else img.shape[-1] - cols // 2 - 1
        flo = int(np.floor(cols / 2))
        cropp_img = img[minx: maxx, miny: maxy, c - flo: c + cols - flo].copy()
        cropp_infection = infection[minx: maxx, miny: maxy, c - flo: c + cols - flo].copy()
        # nbb = agumentation_img_inf_3d(cropp_img, cropp_infection, self.input_x, self.input_y, self.input_z)
        # cropp_pppp = np.expand_dims(np.transpose(nbb['image_patch'], (2, 0, 1)), axis=0)
        # visual_batch(torch.from_numpy(cropp_pppp), save_dir, "test_img_process_back_crop_cc", channel=1, nrow=8)
        return agumentation_img_inf_2d(cropp_img, cropp_infection, self.input_x, self.input_y, self.mean, self.std)


# resize unsupervised slice
class CovidInfValidUnsu2dAugSegDataset(CovidInfUnsuDatasetBase):
    def __init__(self, root_dir, split, input_size, generate_each, mean, std, pos):
        super(CovidInfValidUnsu2dAugSegDataset, self).__init__(root_dir, split, input_size,
                                                               generate_each, mean, std, pos)

    def __len__(self):
        return int(len(self.img_list))

    def __getitem__(self, index):
        # while True:
        count = index
        scans = np.load(self.img_list[count])
        img = scans[0].copy()
        if 'MosMedData' in self.root_dir:
            lung = scans[2]
            infection = scans[1]
        elif 'COVID-19-CT' in self.root_dir:
            lung = scans[1]
            infection = scans[2]

        minx, maxx, miny, maxy, minz, maxz = min_max_voi(lung, superior=3, inferior=3)
        cropp_img = img[minx: maxx, miny: maxy, minz: maxz].copy()
        cropp_infection = infection[minx: maxx, miny: maxy, minz: maxz].copy()
        return agumentation_img_inf_3d(cropp_img, cropp_infection, self.input_x, self.input_y, (maxz - minz))


# crop unsupervised
class CovidInfUnsu2dSegDataset(CovidInfUnsuDatasetBase):
    def __init__(self, root_dir, split, input_size, generate_each, mean, std, pos):
        super(CovidInfUnsu2dSegDataset, self).__init__(root_dir, split, input_size, generate_each, mean, std, pos)

    def __getitem__(self, index):
        # while True:
        count = index // self.generate_each
        scans = np.load(self.img_list[count])
        img = scans[0].copy()
        if 'MosMedData' in self.root_dir:
            lung = scans[2]
            infection = scans[1]
        elif 'COVID-19-CT' in self.root_dir:
            lung = scans[1]
            infection = scans[2]

        # print(cen)
        # cropp_img = img[a - deps // 2:a + deps // 2, b - rows // 2:b + rows // 2,
        #             c - cols // 2: c + cols // 2].copy()
        # cropp_infection = infection[a - deps // 2:a + deps // 2, b - rows // 2:b + rows // 2,
        #                   c - cols // 2:c + cols // 2].copy()
        #
        # minx, maxx, miny, maxy, minz, maxz = min_max_voi(lung, superior=3, inferior=3)
        # cropped_im = img[minx: maxx, miny: maxy, minz: maxz]
        # cropped_if = infection[minx: maxx, miny: maxy, minz: maxz]
        # sed = np.random.randint(1, numid)
        # cen = lines[sed - 1]
        # cen = np.fromstring(cen, dtype=int, sep=' ')
        # c = cen[2] - minz
        # cols = int(self.input_z)
        # maxz = cropped_if.shape[2]
        # minz = 0
        # # c = np.random.randint(minz, maxz - cols - 1)
        # flo = int(np.floor(cols / 2))
        # cel = int(np.ceil(cols // 2))
        # c = int(min(max(minz + flo, c), maxz - cel - 1))
        # cropp_img = cropped_im[:, :, c - flo: c + cols - cel].copy()
        # cropp_infection = cropped_if[:, :, c - flo: c + cols - cel].copy()
        # if not (c >= minz and c < maxz):
        #     print('shape:', img.shape)
        #     print('min max:', (minx, maxx, miny, maxy, minz, maxz))
        #     print('cropped shape:', cropp_img.shape)
        #     print(self.img_list[count])
        #     print('min c %d, max c %d' % (c - flo, c + cols - cel))
        #     print(cen)
        #     exit(0)
        # save_dir = '/home/qgking/COVID3DSeg/log/3DCOVIDCT/deeplabdilate2d/inf_seg_0_run_unsu_mos_covid_0/tmp'
        # cropp_pppp = torch.from_numpy(cropp_img)
        # cropp_pppp=cropp_pppp.unsqueeze(0).unsqueeze(0)
        # cropp_iiii = torch.from_numpy(cropp_infection)
        # cropp_iiii = cropp_iiii.unsqueeze(0).unsqueeze(0)
        # visual_batch(cropp_pppp, save_dir, "test_img", channel=1, nrow=8)
        # visual_batch(cropp_iiii, save_dir, "test_gt", channel=1, nrow=8)
        minx, maxx, miny, maxy, minz, maxz = min_max_voi(lung, superior=3, inferior=3)
        # tmp = img[minx: maxx, miny: maxy, minz: maxz].copy()
        # tmp = (tmp - self.mean) / self.std
        # img[minx: maxx, miny: maxy, minz: maxz] = tmp
        pos = np.random.random()
        if pos > self.pos:
            # only inf region selected
            # print('only inf region selected')
            minindex = self.minindex_list[count]
            maxindex = self.maxindex_list[count]
            lines = self.inflines[count]
            numid = self.infidx[count]

            scale = np.random.uniform(0.8, 1.2)
            deps = int(self.input_x * scale)
            rows = int(self.input_y * scale)
            cols = int(self.input_z)

            sed = np.random.randint(1, numid)
            cen = lines[sed - 1]
            cen = np.fromstring(cen, dtype=int, sep=' ')
            a = int(min(max(minindex[0] + deps / 2, cen[0]), maxindex[0] - deps / 2 - 1))
            b = int(min(max(minindex[1] + rows / 2, cen[1]), maxindex[1] - rows / 2 - 1))
            c = int(min(max(minindex[2] + cols / 2, cen[2]), maxindex[2] - cols / 2 - 1))
            c = c if c - cols // 2 >= 0 else cols // 2
            c = c if c + cols // 2 < img.shape[-1] else img.shape[-1] - cols // 2 - 1
        else:
            # inf region and none inf region selected
            # print('inf region and none inf region selected')
            minx, maxx, miny, maxy, minz, maxz = min_max_voi(lung, superior=3, inferior=3)
            scale = np.random.uniform(0.8, 1.2)
            deps = int(self.input_x * scale)
            rows = int(self.input_y * scale)
            cols = int(self.input_z)
            x = np.random.randint(minx, maxx)
            y = np.random.randint(miny, maxy)
            z = np.random.randint(minz, maxz)
            cen = [x, y, z]
            a = int(min(max(minx + deps / 2, cen[0]), maxx - deps / 2 - 1))
            b = int(min(max(miny + rows / 2, cen[1]), maxy - rows / 2 - 1))
            c = int(min(max(minz + cols / 2, cen[2]), maxz - cols / 2 - 1))
            c = c if c - cols // 2 >= 0 else cols // 2
            c = c if c + cols // 2 < img.shape[-1] else img.shape[-1] - cols // 2 - 1
        flo = int(np.floor(cols / 2))
        cropp_img = img[a - deps // 2:a + deps // 2, b - rows // 2:b + rows // 2,
                    c - flo: c + cols - flo].copy()
        cropp_infection = infection[a - deps // 2:a + deps // 2, b - rows // 2:b + rows // 2,
                          c - flo: c + cols - flo].copy()
        return agumentation_img_inf_3d(cropp_img, cropp_infection, self.input_x, self.input_y, self.input_z)


# --------------UnsuData end-------------


# --------------2D slice start-------------
class CovidInfUnsu2dDatasetBase(Dataset):
    def __init__(self, root_dir, split, input_size, generate_each, mean, std, pos):
        self.input_x = input_size[0]
        self.input_y = input_size[1]
        self.input_z = input_size[2]
        self.root_dir = root_dir
        self.generate_each = generate_each
        self.pos = pos
        self.img_list = []
        self.lung_list = []
        self.inf_list = []
        self.total_slices = 0
        self.mean = mean
        self.std = std
        print('mean %.8f  std %.8f' % (self.mean, self.std))
        if 'MosMedData' in root_dir:
            img_list = sorted(glob(join(root_dir, 'm*.npy')), reverse=True)
        elif 'COVID-19-CT' in root_dir:
            img_list = sorted(glob(join(root_dir, '*.npy')), reverse=True)
        elif 'Italy' in root_dir:
            # TODO need to be modified
            img_list = sorted(glob(join(root_dir, '*.npy')), reverse=True)

        for idx in range(len(img_list)):
            print(img_list[idx])
            scans = np.load(img_list[idx])
            img = scans[0].copy()
            if 'MosMedData' in img_list[idx]:
                lung = scans[2].copy()
                infection = scans[1].copy()
                # use lung and inf
                # sums = np.sum(lung, axis=(0, 1))
                # if np.sum(lung) == 0:
                #     continue
                sums = np.sum(infection, axis=(0, 1))
                inf_sli = np.where(sums > 1)[0]
            elif 'COVID-19-CT' in img_list[idx]:
                lung = scans[1].copy()
                infection = scans[2].copy()
                # use lung and inf
                # sums = np.sum(lung, axis=(0, 1))
                sums = np.sum(infection, axis=(0, 1))
                inf_sli = np.where(sums > 1)[0]
            elif 'Italy' in img_list[idx]:
                lung = scans[1].copy()
                infection = scans[2].copy()
                # GGO and Consolidation
                infection[np.where(infection == 3)] = 0
                infection[np.where(infection > 0)] = 1
                # use inf
                sums = np.sum(infection, axis=(0, 1))
                inf_sli = np.where(sums > 1)[0]
            s_img = img[:, :, inf_sli]
            s_lung = lung[:, :, inf_sli]
            s_infection = infection[:, :, inf_sli]
            for ii in range(s_img.shape[-1]):
                # if 'Italy' in img_list[idx]:
                #     semi_inf = os.listdir('../../log/3DCOVIDCT/Semi-Inf-Net/')
                #     if str(ii) + '.png' not in semi_inf:
                #         continue
                self.img_list.append(s_img[:, :, ii])
                self.lung_list.append(s_lung[:, :, ii])
                self.inf_list.append(s_infection[:, :, ii])
            del scans

        # if 'MosMedData' in root_dir:
        #     idx = []
        #     np.random.seed(666)
        #     indx = np.random.choice(range(len(self.img_list)), size=int(len(self.img_list) * 0.2), replace=False)
        #     idx.extend(indx)
        #     if split == 'train':
        #         self.img_list = [self.img_list[ii] for ii in range(len(self.img_list)) if ii not in idx]
        #         self.lung_list = [self.lung_list[ii] for ii in range(len(self.lung_list)) if ii not in idx]
        #         self.inf_list = [self.inf_list[ii] for ii in range(len(self.inf_list)) if ii not in idx]
        #     elif split == 'valid':
        #         self.img_list = [self.img_list[ii] for ii in range(len(self.img_list)) if ii in idx]
        #         self.lung_list = [self.lung_list[ii] for ii in range(len(self.lung_list)) if ii in idx]
        #         self.inf_list = [self.inf_list[ii] for ii in range(len(self.inf_list)) if ii in idx]
        # elif 'COVID-19-CT' in root_dir:
        #     # img_list = sorted(glob(join(root_dir, '*.npy')), reverse=True)
        #     idx = []
        #     np.random.seed(666)
        #     indx = np.random.choice(range(10), size=2, replace=False)
        #     idx.extend(indx)
        #     np.random.seed(666)
        #     indx = np.random.choice(range(10, 20), size=2, replace=False)
        #     idx.extend(indx)
        #     if split == 'train':
        #         self.img_list = [self.img_list[ii] for ii in range(len(self.img_list)) if ii not in idx]
        #         self.lung_list = [self.lung_list[ii] for ii in range(len(self.lung_list)) if ii not in idx]
        #         self.inf_list = [self.inf_list[ii] for ii in range(len(self.inf_list)) if ii not in idx]
        #     elif split == 'valid':
        #         self.img_list = [self.img_list[ii] for ii in range(len(self.img_list)) if ii in idx]
        #         self.lung_list = [self.lung_list[ii] for ii in range(len(self.lung_list)) if ii in idx]
        #         self.inf_list = [self.inf_list[ii] for ii in range(len(self.inf_list)) if ii in idx]
        # elif 'Italy' in root_dir:
        #     idx = []
        #     np.random.seed(666)
        #     indx = np.random.choice(range(len(self.img_list)), size=int(len(self.img_list) * 0.2), replace=False)
        #     idx.extend(indx)
        #     # TODO need to be modified
        #     if split == 'train':
        #         self.img_list = [self.img_list[ii] for ii in range(len(self.img_list)) if ii not in idx]
        #         self.lung_list = [self.lung_list[ii] for ii in range(len(self.lung_list)) if ii not in idx]
        #         self.inf_list = [self.inf_list[ii] for ii in range(len(self.inf_list)) if ii not in idx]
        #     elif split == 'valid':
        #         self.img_list = [self.img_list[ii] for ii in range(len(self.img_list)) if ii in idx]
        #         self.lung_list = [self.lung_list[ii] for ii in range(len(self.lung_list)) if ii in idx]
        #         self.inf_list = [self.inf_list[ii] for ii in range(len(self.inf_list)) if ii in idx]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        # while True:
        return None


# resize unsupervised slice
class CovidInfValidUnsu2dDatasetBase(CovidInfUnsu2dDatasetBase):
    def __init__(self, root_dir, split, input_size, generate_each, mean, std, pos):
        super(CovidInfValidUnsu2dDatasetBase, self).__init__(root_dir, split, input_size,
                                                             generate_each, mean, std, pos)

    def __len__(self):
        return int(len(self.img_list))

    def __getitem__(self, index):
        im = self.img_list[index]
        lung = self.lung_list[index]
        inf = self.inf_list[index]
        minx, maxx, miny, maxy = min_max_voi_2d(lung, superior=5, inferior=5)
        cropp_img = im[minx: maxx, miny: maxy].copy()
        cropp_infection = inf[minx: maxx, miny: maxy].copy()
        cropp_img = np.tile(np.expand_dims(cropp_img, axis=-1), self.input_z)
        cropp_infection = np.tile(np.expand_dims(cropp_infection, axis=-1), self.input_z)
        return agumentation_img_inf_2d(cropp_img, cropp_infection, self.input_x, self.input_y, self.mean, self.std,
                                       num=4)


# simple 2D slices
class CovidInfUnsu2dSliceSegDataset(CovidInfUnsu2dDatasetBase):
    def __init__(self, root_dir, split, input_size, generate_each, mean, std, pos):
        super(CovidInfUnsu2dSliceSegDataset, self).__init__(root_dir, split, input_size,
                                                            generate_each, mean, std, pos)

    def __getitem__(self, index):
        im = self.img_list[index]
        lung = self.lung_list[index]
        # print(np.unique(lung))
        # print(index)
        inf = self.inf_list[index]
        minx, maxx, miny, maxy = min_max_voi_2d(lung, superior=5, inferior=5)
        cropp_img = im[minx: maxx, miny: maxy].copy()
        cropp_infection = inf[minx: maxx, miny: maxy].copy()
        cropp_img = np.tile(np.expand_dims(cropp_img, axis=-1), self.input_z)
        cropp_infection = np.tile(np.expand_dims(cropp_infection, axis=-1), self.input_z)
        return agumentation_img_inf_2d(cropp_img, cropp_infection, self.input_x, self.input_y, self.mean, self.std,
                                       num=4)


# --------------2D slice end-------------


class CovidInf20SegDataset(Dataset):
    def __init__(self, root_dir, split='train', input_size=(256, 256, 64), generate_each=6):
        self.input_x = input_size[0]
        self.input_y = input_size[1]
        self.input_z = input_size[2]
        self.root_dir = root_dir
        self.generate_each = generate_each
        self.img_list = []
        self.minindex_list = []
        self.maxindex_list = []
        self.infidx = []
        self.inflines = []
        img_list = sorted(glob(join(root_dir, '*.npy')), reverse=True)
        idx = []
        np.random.seed(666)
        indx = np.random.choice(range(10), size=2, replace=False)
        idx.extend(indx)
        np.random.seed(666)
        indx = np.random.choice(range(10, 20), size=2, replace=False)
        idx.extend(indx)
        if split == 'train':
            img_list = [img_list[ii] for ii in range(len(img_list)) if ii not in idx]
        elif split == 'valid':
            img_list = [img_list[ii] for ii in range(len(img_list)) if ii in idx]

        for idx in range(len(img_list)):
            file_name = basename(img_list[idx])[:-4]
            # if idx > 3:
            #     break
            print(img_list[idx])
            self.img_list.append(img_list[idx])
            scans = np.load(img_list[idx])
            values = np.loadtxt(join(root_dir, file_name + '_inf.txt'), delimiter=' ')
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
            f2 = open(join(root_dir, file_name + '_inf.txt'), 'r')
            liverline = f2.readlines()
            self.inflines.append(liverline)
            self.infidx.append(len(liverline))
            f2.close()
            del scans

    def __len__(self):
        return int(self.generate_each * len(self.img_list))

    def __getitem__(self, index):
        # while True:
        count = index // self.generate_each
        scans = np.load(self.img_list[count])
        img = scans[0]
        infection = scans[2]

        minindex = self.minindex_list[count]
        maxindex = self.maxindex_list[count]
        lines = self.inflines[count]
        numid = self.infidx[count]

        #  randomly scale
        scale = np.random.uniform(0.8, 1.2)
        deps = int(self.input_x * scale)
        rows = int(self.input_y * scale)
        cols = int(self.input_z)

        sed = np.random.randint(1, numid)
        cen = lines[sed - 1]
        cen = np.fromstring(cen, dtype=int, sep=' ')
        a = int(min(max(minindex[0] + deps / 2, cen[0]), maxindex[0] - deps / 2 - 1))
        b = int(min(max(minindex[1] + rows / 2, cen[1]), maxindex[1] - rows / 2 - 1))
        c = int(min(max(minindex[2] + cols / 2, cen[2]), maxindex[2] - cols / 2 - 1))
        # print(c)
        c = c if c - cols // 2 >= 0 else cols // 2
        c = c if c + cols // 2 < img.shape[-1] else img.shape[-1] - cols // 2 - 1
        # print(c)
        # print(minindex)
        # print(maxindex)
        # print(cen)
        cropp_img = img[a - deps // 2:a + deps // 2, b - rows // 2:b + rows // 2,
                    c - cols // 2: c + cols // 2].copy()
        cropp_infection = infection[a - deps // 2:a + deps // 2, b - rows // 2:b + rows // 2,
                          c - cols // 2:c + cols // 2].copy()
        # print(img.shape)
        # print(cropp_infection.shape)
        # print('a %d,b %d,c %d' % (a, b, c))
        return agumentation_img_inf_3d(cropp_img, cropp_infection, self.input_x, self.input_y, self.input_z)


class CovidInfDegDataset(Dataset):
    def __init__(self, img_list, split='train', input_size=(256, 256, 64)):
        self.input_x = input_size[0]
        self.input_y = input_size[1]
        self.input_z = input_size[2]
        self.img_list = []
        self.minindex_list = []
        self.maxindex_list = []
        self.infidx = []
        self.inflines = []
        self.img_list = []
        self.split = split
        for img_path in img_list:
            st_index = img_path.rfind('_')
            end_index = img_path.rfind('.')
            label = int(img_path[st_index + 1:end_index])
            if label >= 3:
                self.img_list.extend([img_path, img_path, img_path, img_path, img_path])
            else:
                self.img_list.append(img_path)
        # print('Total dataset %d: ' % (len(self.img_list)))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        # while True:
        img_path = self.img_list[index]
        # print(img_path)
        scans = np.load(img_path)
        img = scans[0]
        coarse_seg = scans[1]
        minx, maxx, miny, maxy, minz, maxz = min_max_voi(coarse_seg, superior=3, inferior=3)
        patch = img[minx: maxx, miny: maxy, minz: maxz]
        bagging_imgs = agumentation_img_3d(patch, self.input_x, self.input_y, self.input_z)
        bagging_imgs = torch.from_numpy(np.expand_dims(bagging_imgs, 0))
        st_index = img_path.rfind('_')
        end_index = img_path.rfind('.')
        image_label = int(img_path[st_index + 1:end_index])
        # if image_label >= 3:
        #     l = 1
        # else:
        #     l = 0
        if image_label >= 3:
            l = 2
        else:
            l = image_label - 1
        return {
            "image_patch": bagging_imgs,
            'image_label': l,
        }


class CovidInfDegDatasetMIL(Dataset):
    def __init__(self, img_list, input_size=(256, 256, 64), generate_bag=6, seg_bagging_aug=None):
        self.input_x = input_size[0]
        self.input_y = input_size[1]
        self.input_z = input_size[2]
        self.generate_bag = generate_bag
        self.img_list = []
        self.minindex_list = []
        self.maxindex_list = []
        self.infidx = []
        self.inflines = []
        self.img_list = img_list
        self.seg_bagging_aug = seg_bagging_aug

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        # while True:
        img_path = self.img_list[index]
        # print(img_path)
        scans = np.load(img_path)
        img = scans[0]
        coarse_seg = scans[1]
        bagging_imgs = self.seg_bagging_aug(img.copy(), coarse_seg, self.generate_bag, self.input_x, self.input_y,
                                            self.input_z)
        st_index = img_path.rfind('_')
        end_index = img_path.rfind('.')
        image_label = int(img_path[st_index + 1:end_index])
        # print(img.shape)
        # print(cropp_infection.shape)
        # print('a %d,b %d,c %d' % (a, b, c))
        return {
            "image_patch": bagging_imgs,
            'image_label': image_label,
        }


class CovidInf50CoarseSegDataset(Dataset):
    def __init__(self, root_dir, input_size=(256, 256, 64), generate_each=6):
        self.input_x = input_size[0]
        self.input_y = input_size[1]
        self.input_z = input_size[2]
        self.root_dir = root_dir
        self.generate_each = generate_each
        self.img_list = []
        self.minindex_list = []
        self.maxindex_list = []
        self.infidx = []
        self.inflines = []
        img_list = sorted(glob(join(root_dir, '*.npy')), reverse=True)
        for idx in range(len(img_list)):
            file_name = basename(img_list[idx])[:-4]
            # if idx > 3:
            #     break
            print(img_list[idx])
            self.img_list.append(img_list[idx])
            scans = np.load(img_list[idx])
            values = np.loadtxt(join(root_dir, file_name + '_inf.txt'), delimiter=' ')
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
            f2 = open(join(root_dir, file_name + '_inf.txt'), 'r')
            liverline = f2.readlines()
            self.inflines.append(liverline)
            self.infidx.append(len(liverline))
            f2.close()
            del scans

    def __len__(self):
        return int(self.generate_each * len(self.img_list))

    def __getitem__(self, index):
        # while True:
        count = index // self.generate_each
        scans = np.load(self.img_list[count])
        img = scans[0]
        infection = scans[1]

        minindex = self.minindex_list[count]
        maxindex = self.maxindex_list[count]
        lines = self.inflines[count]
        numid = self.infidx[count]

        #  randomly scale
        scale = np.random.uniform(0.8, 1.2)
        deps = int(self.input_x * scale)
        rows = int(self.input_y * scale)
        cols = int(self.input_z)

        sed = np.random.randint(1, numid)
        cen = lines[sed - 1]
        cen = np.fromstring(cen, dtype=int, sep=' ')
        a = int(min(max(minindex[0] + deps / 2, cen[0]), maxindex[0] - deps / 2 - 1))
        b = int(min(max(minindex[1] + rows / 2, cen[1]), maxindex[1] - rows / 2 - 1))
        c = int(min(max(minindex[2] + cols / 2, cen[2]), maxindex[2] - cols / 2 - 1))
        # print(c)
        c = c if c - cols // 2 >= 0 else cols // 2
        c = c if c + cols // 2 < img.shape[-1] else img.shape[-1] - cols // 2 - 1
        # print(c)
        # print(minindex)
        # print(maxindex)
        # print(cen)
        cropp_img = img[a - deps // 2:a + deps // 2, b - rows // 2:b + rows // 2,
                    c - cols // 2: c + cols // 2].copy()
        cropp_infection = infection[a - deps // 2:a + deps // 2, b - rows // 2:b + rows // 2,
                          c - cols // 2:c + cols // 2].copy()
        # print(img.shape)
        # print(cropp_infection.shape)
        # print('a %d,b %d,c %d' % (a, b, c))
        return agumentation_img_inf_3d(cropp_img, cropp_infection, self.input_x, self.input_y, self.input_z)


def CovidInfDegData(root_dir, npy_prefix='mstudy*'):
    img_list = sorted(glob(join(root_dir, npy_prefix + '.npy')), reverse=False)
    labels = []
    imgs = []
    for img_path in img_list:
        st_index = img_path.rfind('_')
        end_index = img_path.rfind('.')
        label = int(img_path[st_index + 1:end_index])
        if label == 0:
            continue
        # if label >= 3:
        #     l = 1
        # else:
        #     l = 0
        if label >= 3:
            l = 2
        else:
            l = label - 1
        labels.append(l)
        imgs.append(img_path)
    print('total imgs %d' % (len(imgs)))
    return imgs, labels
