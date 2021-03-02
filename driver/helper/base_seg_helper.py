# -*- coding: utf-8 -*-
# @Time    : 20/1/2 21:43
# @Author  : qgking
# @Email   : qgking@tju.edu.cn
# @Software: PyCharm
# @Desc    : base_seg_helper.py
from common.base_utls import *
from common.data_utils import *
from torch import nn
from models import MODELS
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
from tensorboardX import SummaryWriter
from data import *
from collections import OrderedDict
import pandas as pd
from common.SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient
from sklearn.model_selection import train_test_split
import cv2
import torch.nn.functional as F

plt.rcParams.update({'figure.max_open_warning': 20})


class BaseTrainHelper(object):
    def __init__(self, model, criterions, config):
        self.model = model
        self.criterions = criterions
        self.config = config
        # p = next(filter(lambda p: p.requires_grad, generator.parameters()))
        self.use_cuda = config.use_cuda
        # self.device = p.get_device() if self.use_cuda else None
        self.device = config.gpu if self.use_cuda else None
        if self.config.train:
            self.make_dirs()
        self.define_log()
        self.out_put_summary()
        self.FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if self.use_cuda else torch.LongTensor

    def make_dirs(self):
        if not isdir(self.config.tmp_dir):
            os.makedirs(self.config.tmp_dir)
        if not isdir(self.config.save_dir):
            os.makedirs(self.config.save_dir)
        if not isdir(self.config.save_model_path):
            os.makedirs(self.config.save_model_path)
        if not isdir(self.config.tensorboard_dir):
            os.makedirs(self.config.tensorboard_dir)
        if not isdir(self.config.submission_dir):
            os.makedirs(self.config.submission_dir)

    def merge_batch(self, batch):
        image_patch = [torch.unsqueeze(image, dim=0) for inst in batch for image in inst["image_patch"]]
        orders = np.random.permutation(len(image_patch))
        image_patch = [image_patch[o] for o in orders]
        image_patch = torch.cat(image_patch, dim=0)
        image_segment = [torch.unsqueeze(image, dim=0) for inst in batch for image in inst["image_segment"]]
        image_segment = [image_segment[o] for o in orders]
        image_segment = torch.cat(image_segment, dim=0)

        return {"image_patch": image_patch,
                "image_segment": image_segment,
                }

    def save_checkpoint(self, state, filename='checkpoint.pth'):
        torch.save(state, filename)

    def define_log(self):
        now = datetime.now()  # current date and time
        date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
        if self.config.train:
            log_s = self.config.log_file[:self.config.log_file.rfind('.txt')]
            self.log = Logger(log_s + '_' + str(date_time) + '.txt')
        else:
            self.log = Logger(join(self.config.save_dir, 'test_log_%s.txt' % (str(date_time))))
        sys.stdout = self.log

    def move_to_cuda(self):
        if self.use_cuda and self.model:
            torch.cuda.set_device(self.config.gpu)
            self.equipment = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.equipment)
            for key in self.criterions.keys():
                print(key)
                self.criterions[key].to(self.equipment)
            if len(self.config.gpu_count) > 1 and self.config.train:
                self.model = torch.nn.DataParallel(self.model, device_ids=self.config.gpu_count)
        else:
            self.equipment = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def save_model_checkpoint(self, epoch, optimizer):
        save_file = join(self.config.save_model_path, 'checkpoint_%d_latest.pth' % (epoch))
        self.save_checkpoint({
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, filename=save_file)

    def load_hist_model_optim(self, optimizer):
        load_file = join(self.config.save_model_path, 'checkpoint_epoch_latest.pth')
        if isfile(load_file):
            state_dict = torch.load(load_file, map_location=('cuda:' + str(self.device)))
            epoch = state_dict['epoch']
            self.model.load_state_dict(state_dict['state_dict'])
            optimizer.load_state_dict(state_dict['optimizer'])
            print('load history model and optim at epoch %d' % (epoch))
            return optimizer, epoch
        else:
            print('no history model and optim founded!')
            return optimizer, 0

    def save_best_checkpoint(self, **kwargs):
        opti_file_path = join(self.config.save_model_path, "iter_best_optim.opt")
        save_model_path = join(self.config.save_model_path, "iter_best_model.pt")
        if kwargs['save_model']:
            torch.save(self.model.state_dict(), save_model_path)
        if kwargs['model_optimizer'] is not None:
            torch.save(kwargs['model_optimizer'].state_dict(), opti_file_path)

    def get_best_checkpoint(self, **kwargs):
        state_dict = None
        if kwargs['model_optimizer']:
            load_file = join(self.config.save_model_path, "iter_best_optim.opt")
            if isfile(load_file):
                print('load file ' + load_file)
                state_dict = torch.load(load_file, map_location=('cuda:' + str(self.device)))
            else:
                print('no file exist, optim weight load fail')
        if kwargs['load_model']:
            # load_file = join(self.config.save_model_path, "M_9_iter_best_model.pt")
            load_file = join(self.config.save_model_path, "iter_best_model.pt")
            if isfile(load_file):
                print('load file ' + load_file)
                state_dict = self.load_model_history_weight(load_file)
            else:
                print('no file exist, model weight load fail')
        return state_dict

    def load_best_state(self):
        state_dict_file = self.get_best_checkpoint(load_model=True, model_optimizer=False)
        if state_dict_file is not None:
            print("-------------------load model-------------------")
            self.model.load_state_dict(state_dict_file)
        else:
            print('model is not loaded!!!!')
            exit(0)

    def load_best_optim(self, optim):
        state_dict_file = self.get_best_checkpoint(load_model=False, model_optimizer=True)
        if state_dict_file:
            optim.load_state_dict(state_dict_file)
        return optim

    def out_put_summary(self):
        self.summary_writer = SummaryWriter(self.config.tensorboard_dir)

    def write_summary(self, epoch, criterions):
        for key in criterions.keys():
            self.summary_writer.add_scalar(
                key, criterions[key], epoch)

    def load_model_history_weight(self, weight_file):
        if not self.config.use_cuda:
            return torch.load(weight_file, map_location={'cuda:0': 'cpu', 'cuda:1': 'cpu',
                                                         'cuda:2': 'cpu', 'cuda:3': 'cpu'})
        else:
            state_dict = torch.load(weight_file, map_location=('cuda:' + str(self.device)))
            state_dict_model = self.model.state_dict()
            if ('module' not in list(state_dict_model.keys())[0][:7] and 'module' not in list(state_dict.keys())[0][
                                                                                         :7]) or (
                    'module' in list(state_dict_model.keys())[0][:7] and 'module' in list(state_dict.keys())[0][:7]):
                return torch.load(weight_file, map_location=('cuda:' + str(self.device)))
            elif 'module' in list(state_dict.keys())[0][:7]:
                unParalled_state_dict = {}
                for key in state_dict.keys():
                    unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
                return unParalled_state_dict
            elif 'module' in list(state_dict_model.keys())[0][:7]:
                unParalled_state_dict = {}
                for key in state_dict_model.keys():
                    unParalled_state_dict[key] = state_dict[key.replace("module.", "")]
                return unParalled_state_dict

    def plot_vali_loss(self, epoch, criterions, type='vali'):
        if epoch == 0:
            self.seg_loss_cal = []
            self.seg_acc_cal = []
        plt.figure(figsize=(16, 10), dpi=100)
        self.seg_loss_cal.append(criterions[type + '/seg_loss'])
        self.seg_acc_cal.append(criterions[type + '/seg_acc'])
        epochs = range(len(self.seg_loss_cal))
        plt.subplot(2, 1, 1)
        plt.plot(epochs, self.seg_loss_cal, color='blue', marker='s', linestyle='-', label='seg loss')
        max_ylim = 3 if type == 'vali' else 3
        plt.ylim(0, max_ylim)
        plt.xlim(-2, self.config.epochs + 5)
        plt.title(type + ' loss vs. epoches')
        plt.ylabel('loss')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(epochs, self.seg_acc_cal, color='orange', marker='D', linestyle='-', label='seg accuracy')
        plt.ylim(0.5, 1)
        plt.xlim(-2, self.config.epochs + 5)
        plt.xlabel(type + ' accuracy vs. epoches')
        plt.ylabel(type + ' accuracy')
        plt.legend()
        plt.savefig(join(self.config.submission_dir, "iter_" + type + "_accuracy_loss.jpg"))
        plt.close()

    def plot_train_loss(self, epoch, criterions, type='train'):
        if epoch == 0:
            self.train_seg_loss_cal = []
            self.train_seg_acc_cal = []
        plt.figure(figsize=(16, 10), dpi=100)
        self.train_seg_loss_cal.append(criterions[type + '/seg_loss'])
        self.train_seg_acc_cal.append(criterions[type + '/seg_acc'])
        epochs = range(len(self.train_seg_loss_cal))
        plt.subplot(2, 1, 1)
        plt.plot(epochs, self.train_seg_loss_cal, color='blue', marker='s', linestyle='-', label='seg loss')
        plt.ylim(0, 2)
        plt.xlim(-2, self.config.epochs + 5)
        plt.title(type + ' loss vs. epoches')
        plt.ylabel('loss')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(epochs, self.train_seg_acc_cal, color='orange', marker='D', linestyle='-', label='seg accuracy')
        plt.ylim(0.5, 1)
        plt.xlim(-2, self.config.epochs + 5)
        plt.xlabel(type + ' accuracy vs. epoches')
        plt.ylabel(type + ' accuracy')
        plt.legend()
        plt.savefig(join(self.config.submission_dir, "iter_" + type + "_accuracy_loss.jpg"))
        plt.close()

    def generate_batch(self, batch):
        images = batch['image_patch'].to(self.equipment).float()
        segment = batch['image_segment'].to(self.equipment).long()
        return images, segment

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def _submit_predict(self, pred_dir, test_type=2):
        if test_type == 2:
            gt_path = '../../../medical_data/COVID193DCT/3DCOVIDCT/Lung_Mask'
            save_name = 'Task_lung_SegMetric.csv'
        if test_type == 1:
            gt_path = '../../../medical_data/COVID193DCT/3DCOVIDCT/Infection_Mask'
            save_name = 'Task_infection_SegMetric.csv'
        if test_type == 3:
            gt_path = '../../../medical_data/COVID193DCT/3DCOVIDCT/Lung_Mask'
            save_name = 'Task_Lung_SegMetric.csv'
        if test_type == 4:
            gt_path = '../../../medical_data/COVID193DCT/3DCOVIDCT/Lung_and_Infection_Mask'
            save_name = 'Task_lf_SegMetric.csv'
        filenames = sorted(glob(join(pred_dir, '*.nii.gz')), reverse=True)
        filenames.sort()
        num_labels = test_type
        seg_metrics = OrderedDict()
        seg_metrics['Name'] = list()
        if num_labels == 1:
            seg_metrics['LesionDSC'] = list()
            seg_metrics['LesionNSD-3mm'] = list()
        elif num_labels == 2:
            seg_metrics['L-lungDSC'] = list()
            seg_metrics['L-lung-1mm'] = list()

            seg_metrics['R-lungDSC'] = list()
            seg_metrics['R-lung-1mm'] = list()
        elif num_labels == 3:
            seg_metrics['lungDSC'] = list()
            seg_metrics['lung-1mm'] = list()
        else:
            seg_metrics['L-lungDSC'] = list()
            seg_metrics['L-lung-1mm'] = list()

            seg_metrics['R-lungDSC'] = list()
            seg_metrics['R-lung-1mm'] = list()

            seg_metrics['LesionDSC'] = list()
            seg_metrics['LesionNSD-3mm'] = list()

        for path in filenames:
            name = basename(path)
            seg_metrics['Name'].append(name)
            # load grond truth and segmentation
            gt_nii = nib.load(join(gt_path, name))
            case_spacing = gt_nii.header.get_zooms()
            gt_data = np.uint8(gt_nii.get_fdata())
            seg_data = nib.load(join(pred_dir, name)).get_fdata()
            print(path)
            if num_labels == 1:  # Lesion
                surface_distances = compute_surface_distances(gt_data == 1, seg_data == 1, case_spacing)
                seg_metrics['LesionDSC'].append(compute_dice_coefficient(gt_data == 1, seg_data == 1))
                seg_metrics['LesionNSD-3mm'].append(compute_surface_dice_at_tolerance(surface_distances, 3))
            elif num_labels == 2:  # left and right lung
                surface_distances = compute_surface_distances(gt_data == 1, seg_data == 1, case_spacing)
                seg_metrics['L-lungDSC'].append(compute_dice_coefficient(gt_data == 1, seg_data == 1))
                seg_metrics['L-lung-1mm'].append(compute_surface_dice_at_tolerance(surface_distances, 1))

                surface_distances = compute_surface_distances(gt_data == 2, seg_data == 2, case_spacing)
                seg_metrics['R-lungDSC'].append(compute_dice_coefficient(gt_data == 2, seg_data == 2))
                seg_metrics['R-lung-1mm'].append(compute_surface_dice_at_tolerance(surface_distances, 1))
            elif num_labels == 3:  # lung
                surface_distances = compute_surface_distances(gt_data >= 1, seg_data == 1, case_spacing)
                seg_metrics['lungDSC'].append(compute_dice_coefficient(gt_data >= 1, seg_data == 1))
                seg_metrics['lung-1mm'].append(compute_surface_dice_at_tolerance(surface_distances, 1))
            else:  # left lung, right lung and infections
                surface_distances = compute_surface_distances(gt_data == 1, seg_data == 1, case_spacing)
                seg_metrics['L-lungDSC'].append(compute_dice_coefficient(gt_data == 1, seg_data == 1))
                seg_metrics['L-lung-1mm'].append(compute_surface_dice_at_tolerance(surface_distances, 1))

                surface_distances = compute_surface_distances(gt_data == 2, seg_data == 2, case_spacing)
                seg_metrics['R-lungDSC'].append(compute_dice_coefficient(gt_data == 2, seg_data == 2))
                seg_metrics['R-lung-1mm'].append(compute_surface_dice_at_tolerance(surface_distances, 1))

                surface_distances = compute_surface_distances(gt_data == 3, seg_data == 3, case_spacing)
                seg_metrics['LesionDSC'].append(compute_dice_coefficient(gt_data == 3, seg_data == 3))
                seg_metrics['LesionNSD-3mm'].append(compute_surface_dice_at_tolerance(surface_distances, 3))

        dataframe = pd.DataFrame(seg_metrics)
        dataframe.to_csv(join(self.config.submission_dir, save_name), index=False)

    def visualize_cam(self, mask, img):
        """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
        Args:
            mask (torch.tensor): mask shape of (1, 1, H, W) and each element has value in range [0, 1]
            img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]

        Return:
            heatmap (torch.tensor): heatmap img shape of (3, H, W)
            result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
        """
        heatmap = cv2.applyColorMap(np.uint8(255 * mask.squeeze()), cv2.COLORMAP_JET)
        heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
        b, g, r = heatmap.split(1)
        heatmap = torch.cat([r, g, b])
        img = torch.squeeze(img.cpu()).permute(1, 2, 0).double()
        image = torch.unsqueeze(img, dim=0).permute(0, 3, 1, 2).double()
        result = heatmap.double() + image.double()
        result = result.div(result.max()).squeeze()

        return heatmap, result

    def generate_cam_batch(self, model, batch_image, save_name='image_result_batch_origin'):
        from grad_cam_resnet.gradcam import GradCAMpp
        from grad_cam_resnet.utils import visualize_cam

        model.eval()
        camp_model_dict = dict(type='resnet', arch=model, layer_name='layer4',
                               input_size=(self.config.patch_x, self.config.patch_y))
        grandcamp = GradCAMpp(camp_model_dict, input_channel=1, verbose=True)
        # image_grid = make_grid(batch_image, nrow=4, padding=2)
        # visualize(np.clip(np.transpose(image_grid.detach().cpu().numpy(), (1, 2, 0)) * std + mean, 0, 1),
        #           join(self.config.tmp_dir,
        #                'image_batch_origin'))
        cam_masks = []
        results_pp = []
        saliency_maps_out = []
        for idx in range(batch_image.size(0)):
            torch_image = torch.unsqueeze(batch_image[idx], dim=0)
            mask_pp, _, saliency_map_out = grandcamp(torch_image,
                                                     retain_graph=True)
            heatmap_pp, result_pp = visualize_cam(mask_pp.cpu(), torch_image.cpu())
            cam_masks.append(mask_pp)
            results_pp.append(torch.unsqueeze(result_pp, dim=0))
            saliency_maps_out.append(saliency_map_out)
        cam_masks = torch.cat(cam_masks, dim=0)
        saliency_maps_out = torch.cat(saliency_maps_out, dim=0)
        results_pp = torch.cat(results_pp, dim=0)
        visual_batch(results_pp, self.config.tmp_dir, save_name, channel=3,
                     nrow=8)
        return cam_masks, results_pp, saliency_maps_out

    def max_norm_cam(self, cam_g, e=1e-5):
        saliency_map = F.relu(cam_g)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        cam_out = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min + e).data
        cam_out = F.relu(cam_out)
        return cam_out

    def adjust_learning_rate(self, optimizer, i_iter, num_steps):
        if i_iter < num_steps // 20:
            lr = self.lr_warmup(self.config.learning_rate, i_iter, num_steps // 20)
        else:
            lr = self.lr_poly(self.config.learning_rate, i_iter, num_steps, 0.9)
        # for g in optimizer.param_groups:
        #     current_lr = max(lr, self.config.min_lrate)
        # print("Decaying the learning ratio to %.8f" % (current_lr))
        # g['lr'] = current_lr
        optimizer.param_groups[0]['lr'] = lr
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]['lr'] = lr * 10

    def adjust_learning_rate_ms(self, optimizer, i_iter, num_steps):
        if i_iter < num_steps // 20:
            lr = self.lr_warmup(self.config.learning_rate, i_iter, num_steps // 20)
        else:
            lr = self.lr_poly(self.config.learning_rate, i_iter, num_steps, 0.9)
        # for g in optimizer.param_groups:
        #     current_lr = max(lr, self.config.min_lrate)
        # print("Decaying the learning ratio to %.8f" % (current_lr))
        # g['lr'] = current_lr
        optimizer.param_groups[0]['lr'] = lr
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]['lr'] = lr

    def adjust_learning_rate_se(self, optimizer, i_iter, num_steps, cur_lr):
        if i_iter < num_steps // 20:
            lr = self.lr_warmup(cur_lr, i_iter, num_steps // 20)
        else:
            lr = self.lr_poly(cur_lr, i_iter, num_steps, 0.9)
        optimizer.param_groups[0]['lr'] = lr
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]['lr'] = lr

    def lr_poly(self, base_lr, iter, max_iter, power):
        return base_lr * ((1 - float(iter) / max_iter) ** (power))

    def lr_warmup(self, base_lr, iter, warmup_iter):
        return base_lr * (float(iter) / warmup_iter)

    def adjust_learning_rate_D(self, optimizer, i_iter, num_steps):
        if i_iter < num_steps // 20:
            lr = self.lr_warmup(self.config.learning_rate_d, i_iter, num_steps // 20)
        else:
            lr = self.lr_poly(self.config.learning_rate_d, i_iter, num_steps, 0.9)
        # for g in optimizer.param_groups:
        #     current_lr = max(lr, self.config.min_lrate)
        # print("Learning ratio to %.8f" % (current_lr))
        # g['lr'] = current_lr
        optimizer.param_groups[0]['lr'] = lr
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]['lr'] = lr * 10

    def fliplr(self, img):
        '''flip horizontal'''
        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()  # N x C x H x W
        img_flip = img.index_select(3, inv_idx)
        return img_flip

    def flipvr(self, img):
        '''flip horizontal'''
        inv_idx = torch.arange(img.size(2) - 1, -1, -1).long().cuda()  # N x C x H x W
        img_flip = img.index_select(2, inv_idx)
        return img_flip
