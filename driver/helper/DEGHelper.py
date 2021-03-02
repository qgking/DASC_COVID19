# -*- coding: utf-8 -*-
# @Time    : 20/5/18 11:33
# @Author  : qgking
# @Email   : qgking@tju.edu.cn
# @Software: PyCharm
# @Desc    : DEGHelper.py

from common.base_utls import *
from common.data_utils import *
from tensorboardX import SummaryWriter
from driver.helper.base_seg_helper import BaseTrainHelper
from driver import OPTIM
from torch.utils.data import DataLoader
import torch
from tensorboardX import SummaryWriter
from sklearn.model_selection import train_test_split


class DEGHelper(BaseTrainHelper):
    def __init__(self, model, criterions, config):
        super(DEGHelper, self).__init__(model, criterions, config)

    def out_put_shape(self):
        self.summary_writer = SummaryWriter(self.config.tensorboard_dir)
        print(self.model)

    def reset_optim(self):
        # optimizer = OPTIM[self.config.learning_algorithm](
        #     params=filter(lambda p: p.requires_grad, self.model.parameters()),
        #     lr=self.config.learning_rate, weight_decay=1e-4)
        # return optimizer
        backbone_layer_id = [ii for m in self.model.get_backbone_layers() for ii in list(map(id, m.parameters()))]
        backbone_layer = filter(lambda p: id(p) in backbone_layer_id, self.model.parameters())
        rest_layer = filter(lambda p: id(p) not in backbone_layer_id, self.model.parameters())
        optimizer = OPTIM[self.config.learning_algorithm](
            [{'params': backbone_layer, 'lr': self.config.backbone_learning_rate},
             {'params': rest_layer, 'lr': self.config.learning_rate},
             ],
            lr=self.config.learning_rate, weight_decay=1e-4)
        return optimizer

    def load_pretrained_coarse_inf_seg_model(self, model_seg_coarse):
        file_dir = '../../log/'
        save_model_path = join(file_dir, "coarse_inf_seg_model.pt")
        if not os.path.exists(save_model_path):
            raise FileExistsError('coarse seg model file %s not exits' % (save_model_path))
        else:
            print("loaded %s" % (save_model_path))
            if not self.config.use_cuda:
                self.model.load_state_dict(torch.load(save_model_path,
                                                      map_location={'cuda:0': 'cpu', 'cuda:1': 'cpu',
                                                                    'cuda:2': 'cpu', 'cuda:3': 'cpu'})['g_state_dict'])
            else:
                state_dict = torch.load(save_model_path, map_location=('cuda:' + str(self.device)))
                if 'module' not in list(state_dict.keys())[0][:7]:
                    model_seg_coarse.load_state_dict(
                        torch.load(save_model_path, map_location=('cuda:' + str(self.device))))
                else:
                    unParalled_state_dict = {}
                    for key in state_dict.keys():
                        unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
                    model_seg_coarse.load_state_dict(unParalled_state_dict)
        return model_seg_coarse

    def get_covid_infection_data_mil_deg_loader(self):
        from data.AnomalDataLoader import CovidInfDegData, CovidInfDegDatasetMIL, seg_bagging_aug
        img_list, labels = CovidInfDegData(self.config.data_root, npy_prefix='mstudy*')
        X_train, _, y_train, _ = train_test_split(
            img_list, labels, test_size=0.5, stratify=labels, random_state=666)
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train, y_train, test_size=0.1, stratify=y_train, random_state=666)

        train_dataset = CovidInfDegDatasetMIL(X_train, seg_bagging_aug=seg_bagging_aug(self.config.patch_each),
                                              input_size=(
                                                  self.config.patch_x, self.config.patch_y, self.config.patch_z),
                                              generate_bag=self.config.patch_each)
        train_loader = DataLoader(train_dataset, batch_size=self.config.train_batch_size,
                                  # collate_fn=self.merge_batch,
                                  shuffle=True, num_workers=self.config.workers)
        vali_dataset = CovidInfDegDatasetMIL(X_valid, seg_bagging_aug=seg_bagging_aug(self.config.patch_each),
                                             input_size=(self.config.patch_x, self.config.patch_y, self.config.patch_z),
                                             generate_bag=self.config.patch_each)
        test_loader = DataLoader(vali_dataset, batch_size=self.config.test_batch_size,
                                 # collate_fn=self.merge_batch,
                                 num_workers=self.config.workers,
                                 shuffle=False)
        return train_loader, test_loader

    def get_covid_infection_data_deg_loader(self):
        from data.AnomalDataLoader import CovidInfDegData, CovidInfDegDataset, seg_bagging_aug
        img_list, labels = CovidInfDegData(self.config.data_root, npy_prefix='mstudy*')
        X_train, _, y_train, _ = train_test_split(
            img_list, labels, test_size=0.5, stratify=labels, random_state=666)
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train, y_train, test_size=0.1, stratify=y_train, random_state=666)
        X_train = sorted(X_train)
        train_dataset = CovidInfDegDataset(X_train, split='train',
                                           input_size=(self.config.patch_x, self.config.patch_y, self.config.patch_z))
        train_loader = DataLoader(train_dataset, batch_size=self.config.train_batch_size,
                                  # collate_fn=self.merge_batch,
                                  shuffle=True, num_workers=self.config.workers)
        vali_dataset = CovidInfDegDataset(X_valid, split='valid',
                                          input_size=(self.config.patch_x, self.config.patch_y, self.config.patch_z))
        test_loader = DataLoader(vali_dataset, batch_size=self.config.test_batch_size,
                                 # collate_fn=self.merge_batch,
                                 num_workers=self.config.workers,
                                 shuffle=False)
        return train_loader, test_loader

    def get_moscow_deg_test_data(self):
        from data.AnomalDataLoader import CovidInfDegData, CovidInfDegDataset
        img_list, labels = CovidInfDegData(self.config.data_root)
        _, X_test, _, y_test = train_test_split(
            img_list, labels, test_size=0.5, stratify=labels, random_state=666)
        return X_test

    def generate_batch(self, batch):
        images = batch['image_patch'].to(self.equipment).float()
        labels = batch['image_label'].to(self.equipment).long()
        return images, labels
