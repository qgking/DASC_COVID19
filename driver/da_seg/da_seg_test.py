import matplotlib
import sys

sys.path.extend(["../../", "../", "./"])
from common.base_utls import *
from common.data_utils import *
from models import MODELS
import torch
import random
from driver.helper.DASEGHelper import DASEGHelper
from driver.Config import Configurable
from driver.basic_predict.inf_inwindow_slice_2d import *

matplotlib.use("Agg")
import configparser

config = configparser.ConfigParser()
import argparse

log_template = "[Epoch %d/%d] [seg loss: %f] [seg acc: %f]"


def main(config):
    model = MODELS[config.model](backbone=config.backbone, n_channels=config.channel,
                                 n_classes=config.classes)
    model_cam = MODELS['deeplabdilate2d_cam'](backbone='resnet34', n_channels=config.channel,
                                              n_classes=config.classes)

    criterion = {
    }
    seg_help = DASEGHelper(model, criterion,
                           config)
    seg_help.move_to_cuda()
    try:
        model_cam = seg_help.load_pretrained_cam_seg_model(model_cam)
    except FileExistsError as e:
        raise ValueError('file not exist')
    if seg_help.use_cuda:
        model_cam.to(seg_help.equipment)
    print("data name ", seg_help.config.data_name)
    seg_help.load_best_state()
    covid_test_data = seg_help.get_covid_data_2d(data_root=seg_help.config.unsu_root)
    # predict_inf_inwindow_2d_out_seq_weight_cam(seg_help, seg_help.model, model_cam,
    #                                                   covid_test_data,
    #                                                   thres=0.3,
    #                                                   epoch=6666)
    predict_inf_inwindow_2d_out_seq_weight_cam_50(seg_help, seg_help.model, model_cam,
                                                         covid_test_data,
                                                         thres=0.3,
                                                         epoch=6666)
    # predict_inf_inwindow_2d_out_seq_weight_cam(seg_help, seg_help.model, model_cam,
    #                                                   covid_test_data,
    #                                                   thres=0.5,
    #                                                   epoch=6666)
    # predict_inf_inwindow_2d_out_seq_weight_cam_50(seg_help, seg_help.model, model_cam,
    #                                                      covid_test_data,
    #                                                      thres=0.5,
    #                                                      epoch=6666)
    # predict_inf_inwindow_2d_out_seq_weight_cam(seg_help, seg_help.model, model_cam,
    #                                                   covid_test_data,
    #                                                   thres=0.7,
    #                                                   epoch=6666)
    # predict_inf_inwindow_2d_out_seq_weight_cam_50(seg_help, seg_help.model, model_cam,
    #                                                      covid_test_data,
    #                                                      thres=0.7,
    #                                                      epoch=6666)
    seg_help.log.flush()
    seg_help.summary_writer.close()


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
    argparser.add_argument('--train', help='test not need write', default=False)

    args, extra_args = argparser.parse_known_args()
    file = 'configuration.txt'
    root = '../../log/3DCOVIDCT/deeplabdilate2d_camv19/inf_da_0_run_dapt_ms_ft_v7_lr_ajust_2_new_m_1_final'
    config = Configurable(join(root, file), extra_args, isTrain=args.train)
    torch.set_num_threads(config.workers + 1)

    config.train = args.train
    config.use_cuda = False
    if gpu and args.use_cuda: config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)
    main(config)
