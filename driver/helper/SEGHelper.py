from common.base_utls import *
from common.data_utils import *
from tensorboardX import SummaryWriter
from driver.helper.base_seg_helper import BaseTrainHelper
from driver import OPTIM
from torch.utils.data import DataLoader
import torch
from tensorboardX import SummaryWriter
from models import MODELS
from driver import mean as gmean
from driver import std as gstd


class SEGHelper(BaseTrainHelper):
    def __init__(self, model, criterions, config):
        super(SEGHelper, self).__init__(model, criterions, config)

    def out_put_shape(self):
        self.summary_writer = SummaryWriter(self.config.tensorboard_dir)
        print(self.model)

    def reset_optim(self):
        optimizer = OPTIM[self.config.learning_algorithm](
            params=filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.learning_rate, weight_decay=1e-4)
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

    def read_data(self, train_helper, dataloader):
        while True:
            for batch in dataloader:
                images, labels = train_helper.generate_batch(batch)
                images = images.unsqueeze(1).transpose(1, 4).squeeze()
                labels = labels.unsqueeze(1).transpose(1, 4).squeeze()
                yield images, labels

    def save_best_checkpoint(self, **kwargs):
        opti_file_path = join(self.config.save_model_path, "%d_iter_best_optim.opt" % (kwargs['fold']))
        save_model_path = join(self.config.save_model_path, "%d_iter_best_model.pt" % (kwargs['fold']))
        if kwargs['save_model']:
            torch.save(self.model.state_dict(), save_model_path)
        if kwargs['model_optimizer'] is not None:
            torch.save(kwargs['model_optimizer'].state_dict(), opti_file_path)

    def get_best_checkpoint(self, **kwargs):
        state_dict = None
        if kwargs['model_optimizer']:
            load_file = join(self.config.save_model_path, "%d_iter_best_optim.opt" % (kwargs['fold']))
            if isfile(load_file):
                print('load file ' + load_file)
                state_dict = torch.load(load_file, map_location=('cuda:' + str(self.device)))
            else:
                print('no file exist, optim weight load fail')
        if kwargs['load_model']:
            load_file = join(self.config.save_model_path, "%d_iter_best_model.pt" % (kwargs['fold']))
            if isfile(load_file):
                print('load file ' + load_file)
                state_dict = self.load_model_history_weight(load_file)
            else:
                print('no file exist, model weight load fail')
        return state_dict

    def load_best_state(self, fold):
        state_dict_file = self.get_best_checkpoint(load_model=True, model_optimizer=False, fold=fold)
        if state_dict_file is not None:
            print("-------------------load model-------------------")
            self.model.load_state_dict(state_dict_file)
        else:
            print('model is not loaded!!!!')
            exit(0)

    def load_best_optim(self, optim, fold):
        state_dict_file = self.get_best_checkpoint(load_model=False, model_optimizer=True, fold=fold)
        if state_dict_file:
            optim.load_state_dict(state_dict_file)
        return optim

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

    def get_lung_coarse_data_loader(self):
        from data.LungSegDataLoader import LungCoarseSegDataset
        pkl_file = open(self.config.split_pickle, 'rb')
        split_pickle = pickle.load(pkl_file)[0]
        train_dataset = LungCoarseSegDataset(root_dir=self.config.data_root, split_pickle=split_pickle['train'],
                                             input_size=(self.config.patch_x, self.config.patch_y, self.config.patch_z),
                                             generate_each=self.config.patch_each)
        train_loader = DataLoader(train_dataset, batch_size=self.config.train_batch_size ,
                                  # collate_fn=self.merge_batch,
                                  shuffle=True, num_workers=self.config.workers)
        vali_dataset = LungCoarseSegDataset(root_dir=self.config.data_root, split_pickle=split_pickle['val'],
                                            input_size=(self.config.patch_x, self.config.patch_y, self.config.patch_z),
                                            generate_each=4)
        test_loader = DataLoader(vali_dataset, batch_size=self.config.test_batch_size ,
                                 # collate_fn=self.merge_batch,
                                 num_workers=self.config.workers,
                                 shuffle=False)
        return train_loader, test_loader

    # def get_lung_data_loader(self):
    #     from data.LungSegDataLoader import LungSegDataset
    #     pkl_file = open(self.config.split_pickle, 'rb')
    #     split_pickle = pickle.load(pkl_file)[0]
    #     train_dataset = LungSegDataset(root_dir=self.config.data_root, split_pickle=split_pickle['train'],
    #                                    input_size=(self.config.patch_x, self.config.patch_y, self.config.patch_z),
    #                                    generate_each=self.config.patch_each)
    #     train_loader = DataLoader(train_dataset, batch_size=self.config.train_batch_size ,
    #                               # collate_fn=self.merge_batch,
    #                               shuffle=True, num_workers=self.config.workers)
    #     vali_dataset = LungSegDataset(root_dir=self.config.data_root, split_pickle=split_pickle['val'],
    #                                   input_size=(self.config.patch_x, self.config.patch_y, self.config.patch_z),
    #                                   generate_each=self.config.patch_each)
    #     test_loader = DataLoader(vali_dataset, batch_size=self.config.test_batch_size ,
    #                              # collate_fn=self.merge_batch,
    #                              num_workers=self.config.workers,
    #                              shuffle=False)
    #     return train_loader, test_loader

    def get_lung_test_data_loader(self):
        from data.LungSegDataLoader import LungSegDataset
        pkl_file = open(self.config.split_pickle, 'rb')
        split_pickle = pickle.load(pkl_file)[0]
        vali_dataset = LungSegDataset(root_dir=self.config.data_root, split_pickle=split_pickle['val'],
                                      input_size=(self.config.patch_x, self.config.patch_y, self.config.patch_z),
                                      generate_each=self.config.patch_each)
        test_loader = DataLoader(vali_dataset, batch_size=self.config.test_batch_size ,
                                 # collate_fn=self.merge_batch,
                                 shuffle=False, num_workers=self.config.workers)
        return test_loader

    def get_moscow_covid_infection_total_1108(self,
                                              data_root='../../../medical_data/COVID193DCT/MosMedData/studies_Processed_250'):
        from data.AnomalDataLoader import CovidInfDegData
        img_list, labels = CovidInfDegData(data_root, npy_prefix='study*')
        return img_list

    def get_covid_infection_20_data_loader(self):
        from data.AnomalDataLoader import CovidInf20SegDataset
        train_dataset = CovidInf20SegDataset(root_dir=self.config.data_root, split='train',
                                             input_size=(self.config.patch_x, self.config.patch_y, self.config.patch_z),
                                             generate_each=self.config.patch_each)
        train_loader = DataLoader(train_dataset, batch_size=self.config.train_batch_size ,
                                  # collate_fn=self.merge_batch,
                                  shuffle=True, num_workers=self.config.workers)
        vali_dataset = CovidInf20SegDataset(root_dir=self.config.data_root, split='valid',
                                            input_size=(self.config.patch_x, self.config.patch_y, self.config.patch_z),
                                            generate_each=4)
        test_loader = DataLoader(vali_dataset, batch_size=self.config.test_batch_size ,
                                 # collate_fn=self.merge_batch,
                                 num_workers=self.config.workers,
                                 shuffle=False)
        return train_loader, test_loader

    def get_covid_infection_total_50_data_loader(self):
        from data.AnomalDataLoader import CovidInf50CoarseSegDataset
        train_dataset = CovidInf50CoarseSegDataset(root_dir=self.config.data_root, input_size=(
            self.config.patch_x, self.config.patch_y, self.config.patch_z),
                                                   generate_each=self.config.patch_each)
        train_loader = DataLoader(train_dataset, batch_size=self.config.train_batch_size ,
                                  # collate_fn=self.merge_batch,
                                  shuffle=True, num_workers=self.config.workers)
        return train_loader

    # def get_infection_data_loader(self):
    #     from data.AnomalDataLoader import InfSegDataset
    #     pkl_file = open(self.config.split_pickle, 'rb')
    #     split_pickle = pickle.load(pkl_file)[0]
    #     train_dataset = InfSegDataset(root_dir=self.config.data_root, split_pickle=split_pickle['train'],
    #                                   input_size=(self.config.patch_x, self.config.patch_y, self.config.patch_z),
    #                                   generate_each=self.config.patch_each)
    #     train_loader = DataLoader(train_dataset, batch_size=self.config.train_batch_size ,
    #                               # collate_fn=self.merge_batch,
    #                               shuffle=True, num_workers=self.config.workers)
    #     vali_dataset = InfSegDataset(root_dir=self.config.data_root, split_pickle=split_pickle['val'],
    #                                  input_size=(self.config.patch_x, self.config.patch_y, self.config.patch_z),
    #                                  generate_each=self.config.patch_each)
    #     test_loader = DataLoader(vali_dataset, batch_size=self.config.test_batch_size ,
    #                              # collate_fn=self.merge_batch,
    #                              num_workers=self.config.workers,
    #                              shuffle=False)
    #     return train_loader, test_loader

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

    def get_struct_data(self):
        pkl_file = open(self.config.split_pickle, 'rb')
        split_pickle = pickle.load(pkl_file)[0]
        img_list = sorted(glob(join(self.config.data_root, '*.npy')), reverse=True)
        img_list_final = []
        for idx in range(len(img_list)):
            file_name = basename(img_list[idx])[:-4]
            if file_name not in split_pickle['val']:
                continue
            img_list_final.append(img_list[idx])
        return img_list_final


    def get_covid_infection_5fold_seg_data_loader(self, data_root='', kfold=5, train_vali_rate=1 / 8, test=False):
        print('load data %s' % (data_root))
        # using mean and std of train data
        if 'MosMedData' in self.config.data_root:
            mean = gmean['MosMedData']
            std = gstd['MosMedData']
        elif 'COVID-19-CT' in self.config.data_root:
            mean = gmean['COVID-19-CT']
            std = gstd['COVID-19-CT']
        self.mean=mean
        self.std=std
        data_list = self.get_covid_data(data_root=data_root)
        np.random.seed(666)
        np.random.shuffle(data_list)
        n = len(data_list)
        all_index = range(n)
        each_fold_data = n // kfold
        fold_data = {}
        for i in range(kfold):
            from_index = i * each_fold_data
            to_index = (i + 1) * each_fold_data
            if i == kfold - 1:
                to_index = n
            test_index = range(from_index, to_index)
            fold_data[str(i) + '_fold_test'] = [data_list[idx] for idx in test_index]
            train_vali_index = np.setdiff1d(all_index, test_index)
            vali_index = train_vali_index[0:int(len(train_vali_index) * train_vali_rate)]
            train_index = train_vali_index[int(len(train_vali_index) * train_vali_rate):]
            fold_data[str(i) + '_fold_train'] = [data_list[idx] for idx in train_index]
            fold_data[str(i) + '_fold_vali'] = [data_list[idx] for idx in vali_index]

        from data.AnomalDataLoader import CovidInf5fold2dSegDataset,CovidInf5fold2dAugSegDataset
        from data.AnomalDataLoader import CovidInf5fold2dResizeSegDataset
        train_data_loader_list = []
        vali_data_loader_list = []
        test_data_list = []
        for fold in range(kfold):
            # print('fold %d' % (fold))
            # -------------train-------------
            fold_train_images = fold_data[str(fold) + '_fold_train']
            # -------------vali-------------
            fold_vali_images = fold_data[str(fold) + '_fold_vali']
            # -------------test-------------
            fold_test_images = fold_data[str(fold) + '_fold_test']
            if not test:
                train_dataset = CovidInf5fold2dAugSegDataset(root_dir=data_root, img_list=fold_train_images, mean=mean,
                                                          std=std,
                                                          input_size=(
                                                              self.config.patch_x, self.config.patch_y,
                                                              self.config.patch_z),
                                                          generate_each=self.config.patch_each,pos=-1)
                vali_dataset = CovidInf5fold2dAugSegDataset(root_dir=data_root, img_list=fold_vali_images, mean=mean,
                                                         std=std,
                                                         input_size=(
                                                             self.config.patch_x, self.config.patch_y,
                                                             self.config.patch_z),
                                                         generate_each=self.config.patch_each,pos=-1)

                train_loader = DataLoader(train_dataset,
                                          batch_size=self.config.train_batch_size ,
                                          # collate_fn=self.merge_batch,
                                          shuffle=True, num_workers=self.config.workers)

                vali_loader = DataLoader(vali_dataset,
                                         batch_size=self.config.test_batch_size ,
                                         # collate_fn=self.merge_batch,
                                         num_workers=self.config.workers,
                                         shuffle=False)
                train_data_loader_list.append(train_loader)
                vali_data_loader_list.append(vali_loader)
            test_data_list.append(fold_test_images)
        return train_data_loader_list, vali_data_loader_list, test_data_list

    def reset_model(self):
        if hasattr(self, 'model'):
            del self.model
        self.model = MODELS[self.config.model](backbone=self.config.backbone, n_channels=self.config.patch_z,
                                               n_classes=self.config.classes)
        self.model.to(self.equipment)
        if len(self.config.gpu_count) > 1 and self.config.train:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config.gpu_count)
