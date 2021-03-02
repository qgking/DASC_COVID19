from common.base_utls import *
from common.data_utils import *
from tensorboardX import SummaryWriter
from driver.helper.base_seg_helper import BaseTrainHelper
from driver import OPTIM
from torch.utils.data import DataLoader
import torch
from tensorboardX import SummaryWriter
from driver import mean as gmean
from driver import std as gstd
from models import MODELS, DISCRIMINATOR


class DASEGHelper(BaseTrainHelper):
    def __init__(self, model, criterions, config):
        super(DASEGHelper, self).__init__(model, criterions, config)

    def out_put_shape(self):
        self.summary_writer = SummaryWriter(self.config.tensorboard_dir)
        print(self.model)

    def reset_optim(self):
        optimizer = OPTIM[self.config.learning_algorithm](
            self.model.optim_parameters(self.config.learning_rate),
            lr=self.config.learning_rate, weight_decay=5e-4)
        return optimizer
        # backbone_layer_id = [ii for m in self.model.get_backbone_layers() for ii in list(map(id, m.parameters()))]
        # backbone_layer = filter(lambda p: id(p) in backbone_layer_id, self.model.parameters())
        # rest_layer = filter(lambda p: id(p) not in backbone_layer_id, self.model.parameters())
        # optimizer = OPTIM[self.config.learning_algorithm](
        #     [{'params': backbone_layer, 'lr': self.config.backbone_learning_rate},
        #      {'params': rest_layer, 'lr': self.config.learning_rate},
        #      ],
        #     lr=self.config.learning_rate, weight_decay=1e-4)
        # return optimizer

    def reset_model(self):
        if hasattr(self, 'model'):
            del self.model
        self.model = MODELS[self.config.model](backbone=self.config.backbone, n_channels=self.config.channel,
                                               n_classes=self.config.classes)
        self.model.to(self.equipment)

    def read_data(self, train_helper, dataloader):
        while True:
            for batch in dataloader:
                images, labels = train_helper.generate_batch(batch)
                # images.permute((0,3,1,2))
                # images = images.unsqueeze(1).transpose(1, 4).squeeze()
                # labels = labels.unsqueeze(1).transpose(1, 4).squeeze()
                yield images, labels

    def load_pretrained_cam_seg_model(self, model_cam):
        # save_model_path = join(self.config.save_model_path, "coarse_seg_model.pt")
        # TODO change to upper code in formal env
        file_dir = '../../log/' + self.config.data_name
        save_model_path = join(file_dir, "checkpoint_0_latest.pth")
        if not os.path.exists(save_model_path):
            raise FileExistsError('coarse seg model file %s not exits' % (save_model_path))
        else:
            print("loaded %s" % (save_model_path))
            weight_file = torch.load(save_model_path, map_location=('cuda:' + str(self.device)))['state_dict']
            model_cam.load_state_dict(weight_file)
            del weight_file
        return model_cam

    def load_pretrained_da_seg_model(self, model_seg):
        # data_dir = self.config.unsu_root
        # if 'MosMedData' in data_dir:
        #     file_name= "MosMed_iter_best_model_%d.pt" % (int(self.config.backbone[6:]))
        # elif 'Italy' in data_dir:
        #     # file_name = "Italy_iter_best_model_%d.pt" % (int(self.config.backbone[6:]))
        #     file_name= "Italy_iter_best_model_%d_1.pt" % (int(self.config.backbone[6:]))
        # # save_model_path = join(self.config.save_model_path, "coarse_seg_model.pt")
        # # TODO change to upper code in formal env
        # file_dir = '../../log/' + self.config.data_name
        # save_model_path = join(file_dir, "iter_best_model_%d_1.pt" % (int(self.config.backbone[6:])))
        save_model_path = self.config.sc_model_path
        if not os.path.exists(save_model_path):
            raise FileExistsError('coarse seg model file %s not exits' % (save_model_path))
        else:
            print("loaded pretrained DA model %s" % (save_model_path))
            weight_file = torch.load(save_model_path, map_location=('cuda:' + str(self.device)))
            model_seg.load_state_dict(weight_file)
            del weight_file
        return model_seg

    def load_pretrained_lung_seg_model(self, model_seg_coarse):
        file_dir = '../../log/'
        save_model_path = join(file_dir, "lung_seg_model.pt")
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

    def load_pretrained_coarse_lung_seg_model(self, model_seg_coarse):
        file_dir = '../../log/'
        save_model_path = join(file_dir, "coarse_lung_seg_model.pt")
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

    # --------------UnsuData start-------------
    def get_covid_infection_seg_data_loader(self, data_root='', pos=-1):
        from data.AnomalDataLoader import CovidInfUnsu2dSegDataset, CovidInfValidUnsu2dAugSegDataset
        from data.AnomalDataLoader import CovidInfUnsu2dResizeSegDataset, CovidInfUnsu2dAugSegDataset
        # using mean and std of train data
        if 'MosMedData' in self.config.data_root:
            mean = gmean['MosMedData']
            std = gstd['MosMedData']
        elif 'COVID-19-CT' in self.config.data_root:
            mean = gmean['COVID-19-CT']
            std = gstd['COVID-19-CT']
        self.mean = mean
        self.std = std
        train_dataset = CovidInfUnsu2dAugSegDataset(root_dir=data_root, split='train', mean=mean, std=std,
                                                    input_size=(
                                                        self.config.patch_x, self.config.patch_y,
                                                        self.config.patch_z),
                                                    generate_each=self.config.patch_each, pos=pos)
        train_loader = DataLoader(train_dataset, batch_size=self.config.train_batch_size,
                                  collate_fn=self.merge_batch,
                                  shuffle=True, num_workers=self.config.workers)
        vali_dataset = CovidInfValidUnsu2dAugSegDataset(root_dir=data_root, split='valid', mean=mean, std=std,
                                                        input_size=(
                                                            self.config.patch_x, self.config.patch_y,
                                                            self.config.patch_z),
                                                        generate_each=self.config.patch_each, pos=pos)
        test_loader = DataLoader(vali_dataset, batch_size=1,
                                 num_workers=self.config.workers,
                                 shuffle=False)
        return train_loader, test_loader

    def get_covid_infection_seg_unsu_data_loader(self, data_root='', pos=-1):
        from data.AnomalDataLoader import CovidInfUnsu2dSegDataset
        from data.AnomalDataLoader import CovidInfUnsu2dResizeSegDataset, CovidInfUnsu2dAugSegDataset
        # using mean and std of train data
        if 'MosMedData' in self.config.data_root:
            mean = gmean['MosMedData']
            std = gstd['MosMedData']
        elif 'COVID-19-CT' in self.config.data_root:
            mean = gmean['COVID-19-CT']
            std = gstd['COVID-19-CT']
        self.mean = mean
        self.std = std
        unsu_dataset = CovidInfUnsu2dAugSegDataset(root_dir=data_root, split=None, mean=mean, std=std,
                                                   input_size=(
                                                       self.config.patch_x, self.config.patch_y,
                                                       self.config.patch_z),
                                                   generate_each=self.config.patch_each, pos=pos)
        unsu_loader = DataLoader(unsu_dataset, batch_size=self.config.train_batch_size,
                                 collate_fn=self.merge_batch,
                                 num_workers=self.config.workers,
                                 shuffle=True)

        return unsu_loader

    # --------------UnsuData end-------------
    # --------------2d slice loader start-------------
    def get_covid_infection_seg_data_loader_2d_slice(self, data_root='', pos=-1):
        from data.AnomalDataLoader import CovidInfUnsu2dSliceSegDataset, CovidInfValidUnsu2dDatasetBase
        mean = 0
        std = 0
        if 'MosMedData' in self.config.data_root:
            mean = gmean['MosMedData']
            std = gstd['MosMedData']
        elif 'COVID-19-CT' in self.config.data_root:
            mean = gmean['COVID-19-CT']
            std = gstd['COVID-19-CT']
        self.mean = mean
        self.std = std
        train_dataset = CovidInfUnsu2dSliceSegDataset(root_dir=data_root, split='train', mean=mean, std=std,
                                                      input_size=(
                                                          self.config.patch_x, self.config.patch_y,
                                                          self.config.patch_z),
                                                      generate_each=self.config.patch_each, pos=pos)
        train_loader = DataLoader(train_dataset, batch_size=self.config.train_batch_size,
                                  collate_fn=self.merge_batch,
                                  shuffle=True, num_workers=self.config.workers)
        # vali_dataset = CovidInfValidUnsu2dDatasetBase(root_dir=data_root, split='valid', mean=mean, std=std,
        #                                               input_size=(
        #                                                   self.config.patch_x, self.config.patch_y,
        #                                                   self.config.patch_z),
        #                                               generate_each=self.config.patch_each, pos=pos)
        # test_loader = DataLoader(vali_dataset, batch_size=self.config.test_batch_size,
        #                          num_workers=self.config.workers,
        #                          collate_fn=self.merge_batch,
        #                          shuffle=False)
        test_loader = None
        return train_loader, test_loader

    def get_covid_infection_seg_unsu_data_loader_2d(self, data_root='', pos=-1):
        from data.AnomalDataLoader import CovidInfUnsu2dSliceSegDataset
        mean = 0
        std = 0
        if 'MosMedData' in self.config.data_root:
            mean = gmean['MosMedData']
            std = gstd['MosMedData']
        elif 'COVID-19-CT' in self.config.data_root:
            mean = gmean['COVID-19-CT']
            std = gstd['COVID-19-CT']
        self.mean = mean
        self.std = std
        unsu_dataset = CovidInfUnsu2dSliceSegDataset(root_dir=data_root, split=None, mean=mean, std=std,
                                                     input_size=(
                                                         self.config.patch_x, self.config.patch_y,
                                                         self.config.patch_z),
                                                     generate_each=self.config.patch_each, pos=pos)
        unsu_loader = DataLoader(unsu_dataset, batch_size=self.config.train_batch_size,
                                 collate_fn=self.merge_batch,
                                 num_workers=self.config.workers,
                                 shuffle=True)

        return unsu_loader

    # --------------2d slice loader end-------------

    def get_covid_data(self, data_root):
        # using train data mean std
        if 'MosMedData' in self.config.data_root:
            mean = gmean['MosMedData']
            std = gstd['MosMedData']
        elif 'COVID-19-CT' in self.config.data_root:
            mean = gmean['COVID-19-CT']
            std = gstd['COVID-19-CT']
        if 'MosMedData' in data_root:
            img_list = sorted(glob(join(data_root, 'm*.npy')), reverse=True)
        elif 'COVID-19-CT' in data_root:
            img_list = sorted(glob(join(data_root, '*.npy')), reverse=True)
        self.mean = mean
        self.std = std
        # pixels = []
        # for img_path in img_list:
        #     scans = np.load(img_path)
        #     img = scans[0]
        #     if 'MosMedData' in img_path:
        #         lung = scans[2]
        #     elif 'COVID-19-CT' in img_path:
        #         lung = scans[1]
        #     minx, maxx, miny, maxy, minz, maxz = min_max_voi(lung, superior=3, inferior=3)
        #     px = img[minx: maxx, miny: maxy, minz:maxz]
        #     pixels.extend(px.flatten())
        # print(data_root)
        # print('mean %.8f' % (np.mean(pixels)))
        # print('std %.8f' % (np.std(pixels)))
        return img_list

    def get_covid_data_2d(self, data_root):
        # using train data mean std
        img_list_ = []
        lung_list = []
        inf_list = []
        mean = 0
        std = 0
        if 'MosMedData' in self.config.data_root:
            mean = gmean['MosMedData']
            std = gstd['MosMedData']
        elif 'COVID-19-CT' in self.config.data_root:
            mean = gmean['COVID-19-CT']
            std = gstd['COVID-19-CT']
        if 'MosMedData' in data_root:
            img_list = sorted(glob(join(data_root, 'm*.npy')), reverse=True)
        elif 'COVID-19-CT' in data_root:
            img_list = sorted(glob(join(data_root, '*.npy')), reverse=True)
        elif 'Italy' in data_root:
            img_list = sorted(glob(join(data_root, '*.npy')), reverse=True)
        for idx in range(len(img_list)):
            print(img_list[idx])
            scans = np.load(img_list[idx])
            img = scans[0].copy()
            if 'MosMedData' in img_list[idx]:
                lung = scans[2].copy()
                infection = scans[1].copy()
                # use lung and inf
                sums = np.sum(infection, axis=(0, 1))
                inf_sli = np.where(sums > 1)[0]
            elif 'COVID-19-CT' in img_list[idx]:
                lung = scans[1].copy()
                infection = scans[2].copy()
                # use lung and inf
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
                inf_sli = np.where(sums >= 0)[0]
            s_img = img[:, :, inf_sli]
            s_lung = lung[:, :, inf_sli]
            s_infection = infection[:, :, inf_sli]
            for ii in range(s_img.shape[-1]):
                img_list_.append(s_img[:, :, ii])
                lung_list.append(s_lung[:, :, ii])
                inf_list.append(s_infection[:, :, ii])
        self.mean = mean
        self.std = std
        # pixels = []
        # for img_path in img_list:
        #     scans = np.load(img_path)
        #     img = scans[0]
        #     if 'MosMedData' in img_path:
        #         lung = scans[2]
        #     elif 'COVID-19-CT' in img_path:
        #         lung = scans[1]
        #     minx, maxx, miny, maxy, minz, maxz = min_max_voi(lung, superior=3, inferior=3)
        #     px = img[minx: maxx, miny: maxy, minz:maxz]
        #     pixels.extend(px.flatten())
        # print(data_root)
        # print('mean %.8f' % (np.mean(pixels)))
        # print('std %.8f' % (np.std(pixels)))
        return img_list_, lung_list, inf_list

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

    def prob_2_entropy(self, prob):
        """ convert probabilistic prediction maps to weighted self-information maps
        """
        n, c, h, w = prob.size()
        return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)

    def weightmap(self, pred1, pred2):
        output = 1.0 - torch.sum((pred1 * pred2), 1).view(pred1.size(0), 1, pred1.size(2), pred1.size(3)) / \
                 (torch.norm(pred1, 2, 1) * torch.norm(pred2, 2, 1)).view(pred1.size(0), 1, pred1.size(2),
                                                                          pred1.size(3))
        return output
