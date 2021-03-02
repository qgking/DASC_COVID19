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
import random
import torch
from torchvision.utils import make_grid
MIN_IMG_BOUND = -1250.0  # Everything below: Water  -62   -200
MAX_IMG_BOUND = 250.0  # Everything above corresponds to bones  238    200


def process_lung_nii_file(new_img_dir='MSD_Processed', truncate=[MIN_IMG_BOUND, MAX_IMG_BOUND],
                          root_dir='../../medical_data/COVID193DCT/MSD', resample=False, new_voxel_dim=[1, 1, 1],
                          min_max=False):
    raw_file_dir = join(root_dir, 'Task06_Lung')
    new_img_dir = new_img_dir + '_%d' % (truncate[1])
    new_file_dir = join(root_dir, new_img_dir)
    if resample:
        new_file_dir = new_file_dir + '_Resample'
    if min_max:
        new_file_dir = new_file_dir + '_MM'
    if not exists(new_file_dir):
        makedirs(new_file_dir)
    raw_img_file_dir = join(raw_file_dir, 'imagesTr')
    gt_img_file_dir = join(raw_file_dir, 'labelsTr')
    files = os.listdir(raw_img_file_dir)
    for ff in files:
        print(join(raw_file_dir, ff))
        scan = nib.load(join(raw_img_file_dir, ff))
        img = scan.get_data()
        print(img.shape)
        print('min hu %d, max hu %d' % (np.min(img), np.max(img)))
        # preprocess for remove the scan steel
        thres = 0
        if np.min(img) >= 0 and np.max(img) <= 255:
            thres = 200
        slices = fill_holes(img, thres)

        msk = nib.load(join(gt_img_file_dir, ff)).get_data()
        if resample:
            img, msk, new_voxel_dim = nibResample(img=img, seg=msk, scan=scan,
                                                  new_voxel_dim=new_voxel_dim)
        truncate_low = truncate[0]
        truncate_high = truncate[1]
        if np.min(img) >= 0 and np.max(img) <= 255:
            truncate_low = (truncate[0] - (-1024.)) * (255. - 0.) / (3071. - (-1024.)) + 0.
            truncate_high = (truncate[1] - (-1024.)) * (255. - 0.) / (3071. - (-1024.)) + 0.
        print('min hu %.3f, max hu %.3f' % (truncate_low, truncate_high))
        img = set_bounds(img, truncate_low, truncate_high)
        if min_max:
            minx, maxx, miny, maxy, minz, maxz = min_max_voi(msk, superior=30, inferior=30)
            img = img[minx: maxx, miny: maxy, minz: maxz]
            msk = msk[minx: maxx, miny: maxy, minz: maxz]
        img = normalize_scale(img)
        img = img * slices
        final_data = torch.from_numpy(img).unsqueeze(dim=0).unsqueeze(dim=0)
        prob_save = final_data.float().transpose(1, 4).contiguous().view(
            (final_data.size(0) * final_data.size(-1), final_data.size(1), final_data.size(2),
             final_data.size(3)))
        grid = make_grid(prob_save, nrow=8, padding=2)
        save_img = grid.detach().cpu().numpy()
        visualize(np.transpose(save_img, (1, 2, 0)),
                  join('./tmp',ff[:-7] + "_4"))
        msk[msk > 1] = 0
        print('processed mask label unique:')
        print(np.unique(msk))
        final = np.stack([img, msk])
        np.save(join(new_file_dir, 'train_' + ff[:-7] + '.npy'), final)
        generate_lungtxt(msk, join(new_file_dir, 'train_' + ff[:-7] + '_abnomal'))


def generate_lungtxt(msk, save_folder):
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
    process_lung_nii_file(new_img_dir='MSD_Processed', truncate=[MIN_IMG_BOUND, MAX_IMG_BOUND],
                          root_dir=join(root, 'MSD'), resample=False,
                          new_voxel_dim=[1, 1, 1],
                          min_max=False)
