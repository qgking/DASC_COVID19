import sys
import os

from os.path import join, exists, basename, isdir, isfile
from os import makedirs
import scipy.misc
from scipy import ndimage as nd
from glob import glob
from skimage import measure
import numpy as np
from os import fsync, makedirs
from PIL import Image
import nibabel as nib
import SimpleITK as sitk
from torchvision.utils import make_grid
import numpy as np
import natsort
import re
from scipy.ndimage import zoom as cpzoom
import scipy.ndimage.morphology as morph
import pandas as pd
from scipy.ndimage import rotate as cprotate
import scipy.ndimage.filters as filters
import math
import pickle
from medpy import metric
from scipy import ndimage
from hausdorff import hausdorff_distance
import SimpleITK as sitk
import shutil
import torch
from common.evaluation import compute_all_metric_for_single_seg
import zipfile
from datetime import datetime

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.pyplot import cm
from matplotlib.ticker import NullFormatter
import time

IMG_DTYPE = np.float
SEG_DTYPE = np.uint8
TMP_DIR = "./tmp"
if not os.path.isdir(TMP_DIR):
    os.makedirs(TMP_DIR)


class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class Averagvalue(object):
    """Computes and stores the average and current value"""

    def __init__(self, shape=1):
        self.shape = shape
        self.reset()

    def reset(self):
        self.val = np.zeros(self.shape)
        self.avg = np.zeros(self.shape)
        self.sum = np.zeros(self.shape)
        self.count = 0

    def update(self, val, n=1):
        val = np.array(val)
        assert val.shape == self.val.shape
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# visualize image (as PIL image, NOT as matplotlib!)
def visualize(data, filename):
    assert (len(data.shape) == 3)  # height*width*channels
    if data.shape[2] == 1:  # in case it is black and white
        data = np.reshape(data, (data.shape[0], data.shape[1]))
    if np.max(data) > 1:
        img = Image.fromarray(data.astype(np.uint8))  # the image is already 0-255
    else:
        img = Image.fromarray((data * 255).astype(np.uint8))  # the image is between 0-1
    img.save(filename + '.png')
    return img


def histeq_processor(img):
    """Histogram equalization"""
    nbr_bins = 256
    # get image histogram
    imhist, bins = np.histogram(img.flatten(), nbr_bins, normed=True)
    cdf = imhist.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]  # normalize
    # use linear interpolation of cdf to find new pixel values
    original_shape = img.shape
    img = np.interp(img.flatten(), bins[:-1], cdf)
    img = img / 256.0
    return img.reshape(original_shape)


def normalize_scale(img):
    imgs_normalized = (img - np.min(img)) / (np.max(img) - np.min(img))
    # imgs_normalized = (img - np.mean(imgs_normalized)) / np.std(imgs_normalized)
    return imgs_normalized


def file_name_path(file_dir, dir=True, file=False):
    """
    get root path,sub_dirs,all_sub_files
    :param file_dir:
    :return:
    """
    for root, dirs, files in os.walk(file_dir):
        if len(dirs) and dir:
            print("sub_dirs:", dirs)
            return dirs
        if len(files) and file:
            print("files:", files)
            return files


def cal_ssim(im1, im2):
    assert im1.shape == im2.shape
    mu1 = im1.mean()
    mu2 = im2.mean()
    sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
    k1, k2, L = 0.01, 0.03, 255
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    C3 = C2 / 2
    l12 = (2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2 * sigma1 * sigma2 + C2) / (sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3) / (sigma1 * sigma2 + C3)
    ssim = l12 * c12 * s12
    return ssim


def cal_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def visual_batch(batch, dir, name, channel=1, nrow=8):
    batch_len = len(batch.size())
    if batch_len == 3:
        image_save = batch.detach().contiguous()
        image_save = image_save.unsqueeze(1)
        grid = make_grid(image_save, nrow=nrow, padding=2, pad_value=1)
        save_img = grid.detach().cpu().numpy()
        visualize(np.transpose(save_img, (1, 2, 0)), join(dir, name))
    if batch_len == 4:
        if channel == 3:
            image_save = batch.detach().contiguous()
            grid = make_grid(image_save, nrow=nrow, padding=2, pad_value=1)
            save_img = grid.detach().cpu().numpy()
            visualize(np.transpose(save_img, (1, 2, 0)), join(dir, name))
        else:
            image_save = batch.detach().contiguous()
            image_save = image_save.view(
                (image_save.size(0) * image_save.size(1), image_save.size(2), image_save.size(3)))
            image_save = image_save.unsqueeze(1)
            grid = make_grid(image_save, nrow=nrow, padding=2, pad_value=1)
            save_img = grid.detach().cpu().numpy()
            visualize(np.transpose(save_img, (1, 2, 0)), join(dir, name))
    if batch_len == 5:
        image_save = batch.transpose(1, 4).contiguous()
        image_save = image_save.view(
            (image_save.size(0) * image_save.size(1), image_save.size(4), image_save.size(2), image_save.size(3)))
        grid = make_grid(image_save, nrow=nrow, padding=2, pad_value=1)
        save_img = grid.detach().cpu().numpy()
        visualize(np.transpose(save_img, (1, 2, 0)), join(dir, name))


def zipDir(dirpath, outFullName):
    """
    压缩指定文件夹
    :param dirpath: 目标文件夹路径
    :param outFullName: 压缩文件保存路径+xxxx.zip
    :return: 无
    """
    zip = zipfile.ZipFile(outFullName, "w", zipfile.ZIP_DEFLATED)
    for path, dirnames, filenames in os.walk(dirpath):
        # 去掉目标跟路径，只对目标文件夹下边的文件及文件夹进行压缩
        fpath = path.replace(dirpath, '')
        for filename in filenames:
            zip.write(os.path.join(path, filename), os.path.join(fpath, filename))
    zip.close()


def mkdir_if_not_exist(dir_list):
    for directory in dir_list:
        if not os.path.exists(directory):
            os.makedirs(directory)

# def t_sne(latent_vecs, final_label, out_dir, label_names=['Source Data', 'Target Data 1', 'Target Data 2']):
#     num_classes = len(label_names)
#     fname = "tsne_" + str(time.time()) + '.png'
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     colors = cm.Spectral(np.linspace(0, 1, 9))
#     embeddings = TSNE(n_components=2, random_state=0).fit_transform(latent_vecs)
#     xx = embeddings[:, 0]
#     yy = embeddings[:, 1]
#
#     # plot the 2D data points
#     for i in range(num_classes):
#         ax.scatter(xx[final_label == i], yy[final_label == i], color=colors[i * 3 + 2], label=label_names[i], s=20)
#
#     ax.xaxis.set_major_formatter(NullFormatter())
#     ax.set_xticks([])
#     ax.yaxis.set_major_formatter(NullFormatter())
#     ax.set_yticks([])
#     plt.axis('tight')
#     # plt.xticks([])
#     # plt.yticks([])
#     plt.axis('off')
#     plt.legend(loc='best', scatterpoints=1, fontsize=10)
#     plt.savefig(out_dir + '/' + fname, format='png', dpi=600, pad_inches=0, bbox_inches='tight')
#

def t_sne(latent_vecs, final_label, out_dir, label_names=['Source Data', 'Target Data']):
    num_classes = len(label_names)
    fname = "tsne_" + str(time.time()) + '.png'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = cm.Spectral(np.linspace(0, 1, 20))
    embeddings = TSNE(n_components=2, random_state=0).fit_transform(latent_vecs)
    xx = embeddings[:, 0]
    yy = embeddings[:, 1]

    # plot the 2D data points
    for i in range(num_classes):
        ax.scatter(xx[final_label == i], yy[final_label == i], color=colors[i * 3 + 2], label=label_names[i], s=20)

    ax.xaxis.set_major_formatter(NullFormatter())
    ax.set_xticks([])
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.set_yticks([])
    plt.axis('tight')
    # plt.xticks([])
    # plt.yticks([])
    plt.axis('off')
    plt.legend(loc='best', scatterpoints=1, fontsize=10)
    plt.savefig(out_dir + '/' + fname, format='png', dpi=600, pad_inches=0, bbox_inches='tight')
    # plt.show()
