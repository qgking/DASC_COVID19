import matplotlib
import sys

sys.path.extend(["../../", "../", "./"])
from common.base_utls import *
from common.data_utils import *
from models import MODELS, DISCRIMINATOR
import torch
from torch.cuda import empty_cache
import random
from driver.helper.DASEGHelper import DASEGHelper
from driver.Config import Configurable
from common.evaluation import correct_predictions, accuracy_check
from torch.nn.utils import clip_grad_norm_
from torch.nn import CrossEntropyLoss
from module.Critierion import DiceLoss, DC_and_FC_loss, EntropyLoss, EMLoss, Multi_scale_DC_FC_loss, \
    WeightedBCEWithLogitsLoss
from torchvision.utils import make_grid
from driver import OPTIM
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss, KLDivLoss, MultiLabelSoftMarginLoss
import torch.nn.functional as F
from driver.basic_predict.inf_inwindow_slice_2d import *

matplotlib.use("Agg")
import configparser

config = configparser.ConfigParser()
import argparse


def main(config):
    model = MODELS['deeplab_tsne'](backbone='resnet34', n_channels=config.channel,
                                   n_classes=config.classes)
    criterion = {
        'loss': CrossEntropyLoss(),
        'bceloss': BCEWithLogitsLoss(),
        'wbceloss': WeightedBCEWithLogitsLoss(),
        'klloss': KLDivLoss(),
        'emloss': EMLoss(),
        'entropyloss': EntropyLoss(),
        'celoss': CrossEntropyLoss()
    }
    seg_help = DASEGHelper(model, criterion,
                           config)
    seg_help.move_to_cuda()

    # optimizer, epoch_start = seg_help.load_hist_model_optim(optimizer)
    print("data name ", seg_help.config.data_name)
    train_loader, _ = seg_help.get_covid_infection_seg_data_loader_2d_slice(
        data_root='../../../medical_data/COVID193DCT/3DCOVIDCT/COVID-19-CT-Seg_Processed_250', pos=0.6)
    unsu_loader = seg_help.get_covid_infection_seg_unsu_data_loader_2d(
        data_root='../../../medical_data/COVID193DCT/Italy_COVID/Italy_Processed_2d_250', pos=0.6)
    unsu_loader_1 = seg_help.get_covid_infection_seg_unsu_data_loader_2d(
        data_root='../../../medical_data/COVID193DCT/MosMedData/MosMedData_Processed_250', pos=0.6)

    train(seg_help, train_loader, unsu_loader, unsu_loader_1)

    print("\n-----------load best state of model -----------")
    seg_help.load_best_state()
    seg_help.log.flush()

    seg_help.summary_writer.close()


def train(seg_help, train_loader, unsu_loader, unsu_loader_1):
    seg_help.model.eval()
    batch_num_su = int(np.ceil(len(train_loader.dataset) / float(seg_help.config.train_batch_size)))
    batch_num_un = int(np.ceil(len(unsu_loader.dataset) / float(seg_help.config.train_batch_size)))
    batch_num_un_1 = int(np.ceil(len(unsu_loader_1.dataset) / float(seg_help.config.train_batch_size)))

    print('train with train loader %d' % (len(train_loader.dataset)))
    print('batch_num_su %d,batch_num_un %d' % (batch_num_su, batch_num_un))
    su_data_reader = seg_help.read_data(seg_help, train_loader)
    un_data_reader = seg_help.read_data(seg_help, unsu_loader)
    un_data_reader_1 = seg_help.read_data(seg_help, unsu_loader_1)

    features_label_list = []
    features_s_list = []
    for i_iter in range(batch_num_un):
        su_data, su_label = next(su_data_reader)
        bb, cc, xx, yy = su_data.size()
        su_data = su_data.view(bb * cc, 1, xx, yy)
        visual_batch(su_data, seg_help.config.tmp_dir, str(i_iter) + "_su_data", channel=1, nrow=8)
        features_s = seg_help.model(su_data)
        features = features_s.detach().cpu().numpy()
        features_s_list.extend(features)
    features_s_label_list = np.zeros(len(features_s_list))
    features_label_list.extend(features_s_label_list)

    # features_t_list = []
    # for i_iter in range(batch_num_un):
    #     un_data, un_label = next(un_data_reader)
    #     bb, cc, xx, yy = un_data.size()
    #     un_data = un_data.view(bb * cc, 1, xx, yy)
    #     visual_batch(un_data, seg_help.config.tmp_dir, str(i_iter) + "_un_data", channel=1, nrow=8)
    #     features_t = seg_help.model(un_data)
    #     features = features_t.detach().cpu().numpy()
    #     features_t_list.extend(features)
    # features_t_label_list = np.ones(len(features_t_list))
    # features_label_list.extend(features_t_label_list)

    features_t_1_list = []
    for i_iter in range(batch_num_un):
        un_data, un_label = next(un_data_reader_1)
        bb, cc, xx, yy = un_data.size()
        un_data = un_data.view(bb * cc, 1, xx, yy)
        visual_batch(un_data, seg_help.config.tmp_dir, str(i_iter) + "_un_data_1", channel=1, nrow=8)
        features_t = seg_help.model(un_data)
        features = features_t.detach().cpu().numpy()
        features_t_1_list.extend(features)
    features_t_1_label_list = np.ones(len(features_t_1_list))
    features_label_list.extend(features_t_1_label_list)
    # features_list = features_s_list + features_t_list + features_t_1_list
    features_list = features_s_list + features_t_1_list
    t_sne(features_list, np.array(features_label_list), seg_help.config.tmp_dir)

    empty_cache()
    exit(0)


if __name__ == '__main__':
    torch.manual_seed(6666)
    torch.cuda.manual_seed(6666)
    random.seed(6666)
    np.random.seed(6666)
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    torch.backends.cudnn.benchmark = True  # cudn
    print("CuDNN: \n", torch.backends.cudnn.enabled)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='da_seg_configuration.txt')
    argparser.add_argument('--use-cuda', action='store_true', default=True)
    argparser.add_argument('--train', help='test not need write', default=True)

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args, isTrain=args.train)
    torch.set_num_threads(config.workers + 1)

    config.train = args.train
    config.use_cuda = False
    if gpu and args.use_cuda: config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)
    main(config)
