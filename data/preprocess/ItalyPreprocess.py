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


def process_covid_nii_file_2d(new_img_dir='MosMedData_Processed', truncate=[MIN_IMG_BOUND, MAX_IMG_BOUND],
                              root_dir='', resample=False, new_voxel_dim=[1, 1, 1],
                              min_max=False):
    raw_file = join(root_dir, 'tr_im.nii.gz')
    inf_file = join(root_dir, 'tr_mask.nii.gz')
    lung_file = join(root_dir, 'tr_lungmasks_updated.nii.gz')
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
    raw = nib.load(raw_file)
    inf = nib.load(inf_file)
    lung = nib.load(lung_file)

    voxel_dim = np.array(raw.header.structarr["pixdim"][1:4], dtype=np.float32)
    print(voxel_dim)
    img_data = raw.get_data()
    infection_data = inf.get_data()
    lung_data = lung.get_data()

    print(img_data.shape)
    print('min hu %d, max hu %d' % (np.min(img_data), np.max(img_data)))
    thres = THRES
    if np.min(img_data) >= 0 and np.max(img_data) <= 255:
        thres = 200
    truncate_low = truncate[0]
    truncate_high = truncate[1]
    if np.min(img_data) > 0:
        truncate_low = 0
        truncate_high = 255
    print('min hu %.3f, max hu %.3f' % (truncate_low, truncate_high))
    img = set_bounds(img_data, truncate_low, truncate_high)
    img = normalize_scale(img)
    # final_data = torch.from_numpy(img).unsqueeze(dim=0).unsqueeze(dim=0)
    # prob_save = final_data.float().transpose(1, 4).contiguous().view(
    #     (final_data.size(0) * final_data.size(-1), final_data.size(1), final_data.size(2),
    #      final_data.size(3)))
    # grid = make_grid(prob_save, nrow=8, padding=2)
    # save_img = grid.detach().cpu().numpy()
    # visualize(np.transpose(save_img, (1, 2, 0)),
    #           join('./tmp', ff[:-7] + "_3"))
    save_img = img[:, :, img.shape[-1] // 2]
    visualize(np.expand_dims(save_img, axis=-1),
              join('./tmp', "ttt_4"))
    print('processed mask label unique:')
    print(np.unique(infection_data))
    final = np.stack([img, lung_data, infection_data])
    np.save(join(new_file_dir, 'tr_im.npy'), final)
    # pixels.extend(img[minx: maxx, miny: maxy, minz: maxz].flatten())
    # print(np.mean(pixels))
    # print(np.std(pixels))


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
    process_covid_nii_file_2d(new_img_dir='Italy_Processed_2d', truncate=[MIN_IMG_BOUND, MAX_IMG_BOUND],
                              root_dir=join(root, 'Italy_COVID'), resample=False,
                              new_voxel_dim=[1, 1, 1],
                              min_max=False)
