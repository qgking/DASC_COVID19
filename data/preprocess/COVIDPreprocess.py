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


# first_patient = load_scan(INPUT_FOLDER + patients[0])
# first_patient_pixels = get_pixels_hu(first_patient)
# plt.hist(first_patient_pixels.flatten(), bins=80, color='c')
# plt.xlabel("Hounsfield Units (HU)")
# plt.ylabel("Frequency")
# plt.show()
def process_covid_nii_file(new_img_dir='COVID-19-CT-Seg_Processed', truncate=[MIN_IMG_BOUND, MAX_IMG_BOUND],
                           root_dir='../../medical_data/COVID193DCT/3DCOVIDCT', resample=False, new_voxel_dim=[1, 1, 1],
                           min_max=False):
    raw_file_dir = join(root_dir, 'COVID-19-CT-Seg_20cases')
    inf_file_dir = join(root_dir, 'Infection_Mask')
    lung_file_dir = join(root_dir, 'Lung_Mask')
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
    iii = 0
    total_slices=0
    for ff in files:
        # if iii >= 10:
        #     break
        iii += 1
        print(join(raw_file_dir, ff))
        scan = nib.load(join(raw_file_dir, ff))
        voxel_dim = np.array(scan.header.structarr["pixdim"][1:4], dtype=np.float32)
        print(voxel_dim)
        img = scan.get_data()
        # inf_msk = nib.load(join(inf_file_dir, ff)).get_data()
        # sums = np.sum(inf_msk, axis=(0, 1))
        # inf_sli = np.where(sums > 1)[0]
        # total_slices+=len(inf_sli)
        # continue
        # preprocess for remove the scan steel
        thres = 0
        if np.min(img) >= 0 and np.max(img) <= 255:
            thres = 200
        slices = fill_holes(img, thres)

        # final_data = torch.from_numpy(img).unsqueeze(dim=0).unsqueeze(dim=0)
        # prob_save = final_data.float().transpose(1, 4).contiguous().view(
        #     (final_data.size(0) * final_data.size(-1), final_data.size(1), final_data.size(2),
        #      final_data.size(3)))
        # grid = make_grid(prob_save, nrow=8, padding=2)
        # save_img = grid.detach().cpu().numpy()
        # visualize(np.transpose(save_img, (1, 2, 0)),
        #           join('./tmp',
        #                ff[:-7] + "_0"))
        # final_data = torch.from_numpy(coarse_pred).unsqueeze(dim=0).unsqueeze(dim=0)
        # prob_save = final_data.float().transpose(1, 4).contiguous().view(
        #     (final_data.size(0) * final_data.size(-1), final_data.size(1), final_data.size(2),
        #      final_data.size(3)))
        # grid = make_grid(prob_save, nrow=8, padding=2)
        # save_img = grid.detach().cpu().numpy()
        # visualize(np.transpose(save_img, (1, 2, 0)),
        #           join('./tmp',
        #                ff[:-7] + "_1"))
        # final_data = torch.from_numpy(human_body).unsqueeze(dim=0).unsqueeze(dim=0)
        # prob_save = final_data.float().transpose(1, 4).contiguous().view(
        #     (final_data.size(0) * final_data.size(-1), final_data.size(1), final_data.size(2),
        #      final_data.size(3)))
        # grid = make_grid(prob_save, nrow=8, padding=2)
        # save_img = grid.detach().cpu().numpy()
        # visualize(np.transpose(save_img, (1, 2, 0)),
        #           join('./tmp',
        #                ff[:-7] + "_2"))

        final_data = torch.from_numpy(slices).unsqueeze(dim=0).unsqueeze(dim=0)
        prob_save = final_data.float().transpose(1, 4).contiguous().view(
            (final_data.size(0) * final_data.size(-1), final_data.size(1), final_data.size(2),
             final_data.size(3)))
        grid = make_grid(prob_save, nrow=8, padding=2)
        save_img = grid.detach().cpu().numpy()
        visualize(np.transpose(save_img, (1, 2, 0)),
                  join('./tmp',
                       ff[:-7] + "_3"))
        print(img.shape)
        print('min hu %d, max hu %d' % (np.min(img), np.max(img)))
        lung_msk = nib.load(join(lung_file_dir, ff)).get_data()
        inf_msk = nib.load(join(inf_file_dir, ff)).get_data()
        if resample:
            img, lung_msk, _ = nibResample(img=img, seg=lung_msk, scan=scan,
                                           new_voxel_dim=new_voxel_dim)
            _, inf_msk, _ = nibResample(img=img, seg=inf_msk, scan=scan,
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
            minx, maxx, miny, maxy, minz, maxz = min_max_voi(lung_msk, superior=30, inferior=30)
            img = img[minx: maxx, miny: maxy, minz: maxz]
            lung_msk = lung_msk[minx: maxx, miny: maxy, minz: maxz]
            inf_msk = inf_msk[minx: maxx, miny: maxy, minz: maxz]
        img = normalize_scale(img)

        img = img * slices

        save_img = slices[:, :, slices.shape[-1] // 2]
        visualize(np.expand_dims(save_img, axis=-1),
                  join('./tmp', ff[:-7] + "_3"))
        final_data = torch.from_numpy(img).unsqueeze(dim=0).unsqueeze(dim=0)
        prob_save = final_data.float().transpose(1, 4).contiguous().view(
            (final_data.size(0) * final_data.size(-1), final_data.size(1), final_data.size(2),
             final_data.size(3)))
        grid = make_grid(prob_save, nrow=8, padding=2)
        save_img = grid.detach().cpu().numpy()
        visualize(np.transpose(save_img, (1, 2, 0)),
                  join('./tmp',ff[:-7] + "_4"))

        # remove all non lung organs
        print('processed mask label unique:')
        print(np.unique(lung_msk))
        print(np.unique(inf_msk))
        final = np.stack([img, lung_msk, inf_msk])
        # sio.savemat(join(new_file_dir, ff[:-7] + '.mat'), mdict={'COVID_LUNG_INF': final})
        # np.save(join(new_file_dir, ff[:-7] + '.npy'), final)
        # generate_txt(lung_msk, join(new_file_dir, ff[:-7] + '_lung'))
        # generate_txt(inf_msk, join(new_file_dir, ff[:-7] + '_inf'))
    print(total_slices)

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
    process_covid_nii_file(new_img_dir='COVID-19-CT-Seg_Processed', truncate=[MIN_IMG_BOUND, MAX_IMG_BOUND],
                           root_dir=join(root, '3DCOVIDCT'), resample=False,
                           new_voxel_dim=[1, 1, 1],
                           min_max=False)
