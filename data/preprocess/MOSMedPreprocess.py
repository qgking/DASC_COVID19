# -*- coding: utf-8 -*-
# @Time    : 20/5/1 12:01
# @Author  : qgking
# @Email   : qgking@tju.edu.cn
# @Software: PyCharm
# @Desc    : StructSegPreprocess.py
import sys

sys.path.extend(["../../", "../", "./"])
from common.base_utls import *
from common.data_utils import *
import scipy.io as sio
import torch
from torchvision.utils import make_grid

MIN_IMG_BOUND = -1250.0  # Everything below: Water  -62   -200   ,-1024
MAX_IMG_BOUND = 250.0  # Everything above corresponds to bones  238    3071
THRES = -40


def process_moscow_nii_file(new_img_dir='studies_Processed', truncate=[MIN_IMG_BOUND, MAX_IMG_BOUND],
                            root_dir='', resample=False, new_voxel_dim=[1, 1, 1],
                            min_max=False):
    studies_dir = join(root_dir, 'studies')
    sub_dirs = os.listdir(studies_dir)
    new_img_dir = new_img_dir + '_%d' % (truncate[1])
    new_file_dir = join(root_dir, new_img_dir)
    if resample:
        new_file_dir = new_file_dir + '_Resample'
    if min_max:
        new_file_dir = new_file_dir + '_MM'
    if not exists(new_file_dir):
        makedirs(new_file_dir)
    # shutil.rmtree(new_file_dir)
    # makedirs(new_file_dir)
    sub_dirs.sort(reverse=True)
    for sub_dir in sub_dirs:
        # label = sub_dir[3:]
        # if int(label) != 4:
        #     continue
        dir_images = glob(join(studies_dir, sub_dir, '*.nii.gz'))
        dir_images.sort(reverse=True)
        for path in dir_images:
            scan = nib.load(path)
            img_name = basename(path)
            voxel_dim = np.array(scan.header.structarr["pixdim"][1:4], dtype=np.float32)
            print(path)
            print(voxel_dim)
            img = scan.get_data()
            print(img.shape)
            print('min hu %d, max hu %d' % (np.min(img), np.max(img)))
            thres = -100
            if np.min(img) >= 0 and np.max(img) <= 255:
                thres = 200
            slices = fill_holes(img, thres)
            inf_msk = []
            if resample:
                img, inf_msk, _ = nibResample(img=img, seg=inf_msk, scan=scan,
                                              new_voxel_dim=new_voxel_dim)
            truncate_low = truncate[0]
            truncate_high = truncate[1]
            if np.min(img) > 0:
                truncate_low = 0
                truncate_high = 255
            print('min hu %.3f, max hu %.3f' % (truncate_low, truncate_high))
            img = set_bounds(img, truncate_low, truncate_high)
            # if np.min(img) >= 0 and np.max(img) <= 255:
            #     truncate_low = (truncate[0] - (-1024.)) * (255. - 0.) / (3071. - (-1024.)) + 0.
            #     truncate_high = (truncate[1] - (-1024.)) * (255. - 0.) / (3071. - (-1024.)) + 0.
            # print('min hu %.3f, max hu %.3f' % (truncate_low, truncate_high))
            # img = set_bounds(img, truncate_low, truncate_high)
            if min_max:
                minx, maxx, miny, maxy, minz, maxz = min_max_voi(inf_msk, superior=30, inferior=30)
                img = img[minx: maxx, miny: maxy, minz: maxz]
                inf_msk = inf_msk[minx: maxx, miny: maxy, minz: maxz]
            img = normalize_scale(img)
            img = img * slices

            # final_data = torch.from_numpy(img).unsqueeze(dim=0).unsqueeze(dim=0)
            # prob_save = final_data.float().transpose(1, 4).contiguous().view(
            #     (final_data.size(0) * final_data.size(-1), final_data.size(1), final_data.size(2),
            #      final_data.size(3)))
            # grid = make_grid(prob_save, nrow=8, padding=2)
            # save_img = grid.detach().cpu().numpy()
            # visualize(np.transpose(save_img, (1, 2, 0)),
            #           join('./tmp', img_name[:-7] + "_3"))
            save_img = slices[:, :, slices.shape[-1] // 2]
            visualize(np.expand_dims(save_img, axis=-1),
                      join('./tmp', img_name[:-7] + "_4"))

            # remove all non lung organs
            print('processed mask label unique:')
            print(np.unique(inf_msk))
            final = np.stack([img])
            label = sub_dir[3:]
            # np.save(join(new_file_dir, img_name[:-7] + '_%s.npy' % (label)), final)


def process_covid_nii_file(new_img_dir='MosMedData_Processed', truncate=[MIN_IMG_BOUND, MAX_IMG_BOUND],
                           root_dir='', resample=False, new_voxel_dim=[1, 1, 1],
                           min_max=False):
    raw_file_dir = join(root_dir, 'image')
    inf_file_dir = join(root_dir, 'masks')
    new_img_dir = new_img_dir + '_%d' % (truncate[1])
    new_file_dir = join(root_dir, new_img_dir)
    if resample:
        new_file_dir = new_file_dir + '_Resample'
    if min_max:
        new_file_dir = new_file_dir + '_MM'
    if not exists(new_file_dir):
        makedirs(new_file_dir)
    # shutil.rmtree(new_file_dir)
    # makedirs(new_file_dir)
    files = os.listdir(raw_file_dir)
    files.sort(reverse=True)
    pixels = []
    num_slices = 0
    for ff in files:
        print(join(raw_file_dir, ff))
        scan = nib.load(join(raw_file_dir, ff))
        voxel_dim = np.array(scan.header.structarr["pixdim"][1:4], dtype=np.float32)
        print(voxel_dim)
        img = scan.get_data()
        print(img.shape)
        print('min hu %d, max hu %d' % (np.min(img), np.max(img)))
        thres = THRES
        if np.min(img) >= 0 and np.max(img) <= 255:
            thres = 200
        slices = fill_holes(img, thres)

        inf_msk = nib.load(join(inf_file_dir, ff[:-7] + '_mask.nii.gz')).get_data()
        sums = np.sum(inf_msk, axis=(0, 1))
        num_slices += np.sum(np.where(sums > 1, 1, 0))
        print(num_slices)
        continue
        if resample:
            img, inf_msk, _ = nibResample(img=img, seg=inf_msk, scan=scan,
                                          new_voxel_dim=new_voxel_dim)
        truncate_low = truncate[0]
        truncate_high = truncate[1]
        if np.min(img) > 0:
            truncate_low = 0
            truncate_high = 255
        print('min hu %.3f, max hu %.3f' % (truncate_low, truncate_high))
        img = set_bounds(img, truncate_low, truncate_high)
        # if np.min(img) >= 0 and np.max(img) <= 255:
        #     truncate_low = (truncate[0] - (-1024.)) * (255. - 0.) / (3071. - (-1024.)) + 0.
        #     truncate_high = (truncate[1] - (-1024.)) * (255. - 0.) / (3071. - (-1024.)) + 0.
        # print('min hu %.3f, max hu %.3f' % (truncate_low, truncate_high))
        # img = set_bounds(img, truncate_low, truncate_high)
        minx, maxx, miny, maxy, minz, maxz = min_max_voi(inf_msk, superior=30, inferior=30)
        if min_max:
            # minx, maxx, miny, maxy, minz, maxz = min_max_voi(inf_msk, superior=30, inferior=30)
            img = img[minx: maxx, miny: maxy, minz: maxz]
            inf_msk = inf_msk[minx: maxx, miny: maxy, minz: maxz]
        img = normalize_scale(img)
        img = img * slices
        # final_data = torch.from_numpy(img).unsqueeze(dim=0).unsqueeze(dim=0)
        # prob_save = final_data.float().transpose(1, 4).contiguous().view(
        #     (final_data.size(0) * final_data.size(-1), final_data.size(1), final_data.size(2),
        #      final_data.size(3)))
        # grid = make_grid(prob_save, nrow=8, padding=2)
        # save_img = grid.detach().cpu().numpy()
        # visualize(np.transpose(save_img, (1, 2, 0)),
        #           join('./tmp', ff[:-7] + "_3"))
        save_img = slices[:, :, slices.shape[-1] // 2]
        visualize(np.expand_dims(save_img, axis=-1),
                  join('./tmp', ff[:-7] + "_4"))
        print('processed mask label unique:')
        print(np.unique(inf_msk))
        final = np.stack([img, inf_msk])
        # np.save(join(new_file_dir, ff[:-7] + '.npy'), final)
        # generate_txt(inf_msk, join(new_file_dir, ff[:-7] + '_inf'))
        # pixels.extend(img[minx: maxx, miny: maxy, minz: maxz].flatten())
    # print(np.mean(pixels))
    # print(np.std(pixels))
    exit(0)

def process_covid_nii_file_2d(new_img_dir='MosMedData_Processed', truncate=[MIN_IMG_BOUND, MAX_IMG_BOUND],
                           root_dir='', resample=False, new_voxel_dim=[1, 1, 1],
                           min_max=False):
    raw_file_dir = join(root_dir, 'image')
    inf_file_dir = join(root_dir, 'masks')
    new_img_dir = new_img_dir + '_%d' % (truncate[1])
    new_file_dir = join(root_dir, new_img_dir)
    if resample:
        new_file_dir = new_file_dir + '_Resample'
    if min_max:
        new_file_dir = new_file_dir + '_MM'
    if not exists(new_file_dir):
        makedirs(new_file_dir)
    # shutil.rmtree(new_file_dir)
    # makedirs(new_file_dir)
    files = os.listdir(raw_file_dir)
    files.sort(reverse=True)
    pixels = []
    num_slices = 0
    for ff in files:
        print(join(raw_file_dir, ff))
        scan = nib.load(join(raw_file_dir, ff))
        voxel_dim = np.array(scan.header.structarr["pixdim"][1:4], dtype=np.float32)
        print(voxel_dim)
        img = scan.get_data()
        print(img.shape)
        print('min hu %d, max hu %d' % (np.min(img), np.max(img)))
        thres = THRES
        if np.min(img) >= 0 and np.max(img) <= 255:
            thres = 200
        slices = fill_holes(img, thres)

        inf_msk = nib.load(join(inf_file_dir, ff[:-7] + '_mask.nii.gz')).get_data()
        sums = np.sum(inf_msk, axis=(0, 1))
        num_slices += np.sum(np.where(sums > 1, 1, 0))
        print(num_slices)
        continue
        if resample:
            img, inf_msk, _ = nibResample(img=img, seg=inf_msk, scan=scan,
                                          new_voxel_dim=new_voxel_dim)
        truncate_low = truncate[0]
        truncate_high = truncate[1]
        if np.min(img) > 0:
            truncate_low = 0
            truncate_high = 255
        print('min hu %.3f, max hu %.3f' % (truncate_low, truncate_high))
        img = set_bounds(img, truncate_low, truncate_high)
        # if np.min(img) >= 0 and np.max(img) <= 255:
        #     truncate_low = (truncate[0] - (-1024.)) * (255. - 0.) / (3071. - (-1024.)) + 0.
        #     truncate_high = (truncate[1] - (-1024.)) * (255. - 0.) / (3071. - (-1024.)) + 0.
        # print('min hu %.3f, max hu %.3f' % (truncate_low, truncate_high))
        # img = set_bounds(img, truncate_low, truncate_high)
        minx, maxx, miny, maxy, minz, maxz = min_max_voi(inf_msk, superior=30, inferior=30)
        if min_max:
            # minx, maxx, miny, maxy, minz, maxz = min_max_voi(inf_msk, superior=30, inferior=30)
            img = img[minx: maxx, miny: maxy, minz: maxz]
            inf_msk = inf_msk[minx: maxx, miny: maxy, minz: maxz]
        img = normalize_scale(img)
        img = img * slices
        # final_data = torch.from_numpy(img).unsqueeze(dim=0).unsqueeze(dim=0)
        # prob_save = final_data.float().transpose(1, 4).contiguous().view(
        #     (final_data.size(0) * final_data.size(-1), final_data.size(1), final_data.size(2),
        #      final_data.size(3)))
        # grid = make_grid(prob_save, nrow=8, padding=2)
        # save_img = grid.detach().cpu().numpy()
        # visualize(np.transpose(save_img, (1, 2, 0)),
        #           join('./tmp', ff[:-7] + "_3"))
        save_img = slices[:, :, slices.shape[-1] // 2]
        visualize(np.expand_dims(save_img, axis=-1),
                  join('./tmp', ff[:-7] + "_4"))
        print('processed mask label unique:')
        print(np.unique(inf_msk))
        final = np.stack([img, inf_msk])
        # np.save(join(new_file_dir, ff[:-7] + '.npy'), final)
        # generate_txt(inf_msk, join(new_file_dir, ff[:-7] + '_inf'))
        # pixels.extend(img[minx: maxx, miny: maxy, minz: maxz].flatten())
    # print(np.mean(pixels))
    # print(np.std(pixels))
    exit(0)


def generate_txt(msk, save_folder):
    f = open(save_folder + '.txt', 'w')
    index = np.where(msk > 0)
    x = index[0]
    y = index[1]
    z = index[2]
    np.savetxt(f, np.c_[x, y, z], fmt="%d")
    f.write("\n")
    f.close()


if __name__ == '__main__':
    root = '../../../medical_data/COVID193DCT/'
    process_covid_nii_file_2d(new_img_dir='MosMedData_Processed_2d', truncate=[MIN_IMG_BOUND, MAX_IMG_BOUND],
                           root_dir=join(root, 'MosMedData'), resample=False,
                           new_voxel_dim=[1, 1, 1],
                           min_max=False)

    # process_covid_nii_file(new_img_dir='MosMedData_Processed', truncate=[MIN_IMG_BOUND, MAX_IMG_BOUND],
    #                        root_dir=join(root, 'MosMedData'), resample=False,
    #                        new_voxel_dim=[1, 1, 1],
    #                        min_max=False)
    # root = '../../../medical_data/COVID193DCT/'
    # process_moscow_nii_file(new_img_dir='studies_Processed', truncate=[MIN_IMG_BOUND, MAX_IMG_BOUND],
    #                         root_dir=join(root, 'MosMedData'), resample=False,
    #                         new_voxel_dim=[1, 1, 1],
    #                         min_max=False)
