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
from copy import deepcopy

matplotlib.use("Agg")
import configparser

config = configparser.ConfigParser()
import argparse


def main(config):
    model = MODELS[config.model](backbone=config.backbone, n_channels=config.channel,
                                 n_classes=config.classes)
    model_cam = MODELS['deeplabdilate2d_cam'](backbone='resnet34', n_channels=config.channel,
                                              n_classes=config.classes)
    model_seg = MODELS['deeplabdilate2d_camv19'](backbone=config.backbone, n_channels=config.channel,
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
    optimizer = seg_help.reset_optim()
    seg_help.move_to_cuda()
    try:
        model_cam = seg_help.load_pretrained_cam_seg_model(model_cam)
    except FileExistsError as e:
        raise ValueError('file not exist')
    try:
        model_seg = seg_help.load_pretrained_da_seg_model(model_seg)
    except FileExistsError as e:
        raise ValueError('file not exist')
    if len(seg_help.config.gpu_count) > 1:
        seg_help.model.module.load_state_dict(model_seg.state_dict())
    else:
        seg_help.model.load_state_dict(model_seg.state_dict())
    model_seg.eval()
    if seg_help.use_cuda:
        model_cam.to(seg_help.equipment)
        model_seg.to(seg_help.equipment)
        if len(seg_help.config.gpu_count) > 1 and seg_help.config.train:
            model_cam = torch.nn.DataParallel(model_cam, device_ids=seg_help.config.gpu_count)
            model_seg = torch.nn.DataParallel(model_seg, device_ids=seg_help.config.gpu_count)

    # optimizer, epoch_start = seg_help.load_hist_model_optim(optimizer)
    print("data name ", seg_help.config.data_name)
    train_loader, _ = seg_help.get_covid_infection_seg_data_loader_2d_slice(
        data_root=seg_help.config.data_root, pos=0.6)
    unsu_loader = seg_help.get_covid_infection_seg_unsu_data_loader_2d(
        data_root=seg_help.config.unsu_root, pos=0.6)
    train(seg_help, model_cam, model_seg, train_loader, unsu_loader, optimizer)
    seg_help.log.flush()

    seg_help.summary_writer.close()


def train(seg_help, model_cam, model_seg, train_loader, unsu_loader, optimizer):
    results = None
    model_cam.eval()
    model_seg.eval()
    batch_num_su = int(np.ceil(len(train_loader.dataset) / float(seg_help.config.train_batch_size)))
    batch_num_un = int(np.ceil(len(unsu_loader.dataset) / float(seg_help.config.train_batch_size)))
    total_iter = batch_num_su * seg_help.config.epochs
    print('train with train loader %d' % (len(train_loader.dataset)))
    print('batch_num_su %d,batch_num_un %d' % (batch_num_su, batch_num_un))
    print('total iter %d' % (total_iter))
    su_data_reader = seg_help.read_data(seg_help, train_loader)
    un_data_reader = seg_help.read_data(seg_help, unsu_loader)
    covid_test_data = seg_help.get_covid_data_2d(data_root=seg_help.config.unsu_root)
    Lambda_seg = 3
    cycle = 10
    best_acc = 0
    seg_help.model.train()
    w0 = model_seg.state_dict()
    model_former = deepcopy(model_seg)
    for c in range(1, cycle):
        m = c
        w_next = {}
        w_former = model_former.state_dict()
        for key in w_former.keys():
            w_next[key] = w_former[key] * m / (m + 1) + w0[key] / (m + 1)
        seg_help.model.load_state_dict(w_next)
        lr_min = seg_help.config.min_lrate
        lr_max = seg_help.config.learning_rate
        # cur_lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(m * math.pi / cycle))
        cur_lr = seg_help.config.learning_rate
        print('change lr to %.6f' % (cur_lr))
        optimizer = OPTIM[seg_help.config.learning_algorithm](
            seg_help.model.optim_parameters(cur_lr),
            lr=cur_lr, weight_decay=5e-4)
        for i_iter in range(total_iter):
            optimizer.zero_grad()
            seg_help.adjust_learning_rate_se(optimizer, i_iter, total_iter, cur_lr)
            result = []
            su_data, su_label = next(su_data_reader)
            bb, cc, xx, yy = su_data.size()
            su_data = su_data.view(bb * cc, 1, xx, yy)
            bb, cc, xx, yy = su_label.size()
            su_label = su_label.view(bb * cc, xx, yy)
            # train supervised

            un_data, un_label = next(un_data_reader)
            bb, cc, xx, yy = un_data.size()
            un_data = un_data.view(bb * cc, 1, xx, yy)
            bb, cc, xx, yy = un_label.size()
            un_label = un_label.view(bb * cc, xx, yy)
            with torch.no_grad():
                cam_p = model_cam.forward_cam(un_data)
                (x1, pred_source1, pred_source2) = model_seg.forward_seg(un_data, cam_p)
                pseudo_label = x1 + pred_source1 + pred_source2

                cam_p_ = model_cam.forward_cam(seg_help.fliplr(un_data))
                (x1, pred_source1, pred_source2) = model_seg.forward_seg(seg_help.fliplr(un_data),
                                                                         cam_p_)
                pseudo_label += (seg_help.fliplr(x1) + seg_help.fliplr(pred_source1) + seg_help.fliplr(pred_source2))

                cam_p_ = model_cam.forward_cam(seg_help.flipvr(un_data))
                (x1, pred_source1, pred_source2) = model_seg.forward_seg(seg_help.flipvr(un_data),
                                                                         cam_p_)
                pseudo_label += (seg_help.flipvr(x1) + seg_help.flipvr(pred_source1) + seg_help.flipvr(pred_source2))
                pseudo_label_0 = pseudo_label

                cam_p = model_cam.forward_cam(un_data)
                (x1, pred_source1, pred_source2) = model_former.forward_seg(un_data, cam_p)
                pseudo_label = x1 + pred_source1 + pred_source2

                cam_p_ = model_cam.forward_cam(seg_help.fliplr(un_data))
                (x1, pred_source1, pred_source2) = model_former.forward_seg(seg_help.fliplr(un_data),
                                                                            cam_p_)
                pseudo_label += (seg_help.fliplr(x1) + seg_help.fliplr(pred_source1) + seg_help.fliplr(pred_source2))

                cam_p_ = model_cam.forward_cam(seg_help.flipvr(un_data))
                (x1, pred_source1, pred_source2) = model_former.forward_seg(seg_help.flipvr(un_data),
                                                                            cam_p_)
                pseudo_label += (seg_help.flipvr(x1) + seg_help.flipvr(pred_source1) + seg_help.flipvr(pred_source2))
                pseudo_label_former = pseudo_label

                pseudo_label_fusion = pseudo_label_former * m / (m + 1) + pseudo_label_0 / (m + 1)
                pseudo_label = torch.softmax(pseudo_label_fusion, dim=1)

                visual_batch(un_data, seg_help.config.tmp_dir, "train_" + str(i_iter) + "_image", channel=1, nrow=4)
                visual_batch(un_label, seg_help.config.tmp_dir, "train_" + str(i_iter) + "_image_label", channel=1,
                             nrow=4)

            # train supervised
            pseudo_label = torch.argmax(pseudo_label, dim=1)
            visual_batch(pseudo_label.view(bb * cc, 1, xx, yy), seg_help.config.tmp_dir,
                         "train_" + str(i_iter) + "_image_pseudo", channel=1, nrow=4)
            real_pseudo_label = torch.cat([su_label, pseudo_label], dim=0)
            su_un_data = torch.cat([su_data, un_data], dim=0)

            image_patch = [torch.unsqueeze(image, dim=0) for image in su_un_data]
            orders = np.random.permutation(len(su_un_data))
            image_patch = [image_patch[o] for o in orders]
            image_patch = torch.cat(image_patch, dim=0)

            image_segment = [torch.unsqueeze(ll, dim=0) for ll in real_pseudo_label]
            image_segment = [image_segment[o] for o in orders]
            image_segment = torch.cat(image_segment, dim=0)

            with torch.no_grad():
                cam_p = model_cam.forward_cam(image_patch)

            visual_batch(image_patch, seg_help.config.tmp_dir, "train_" + str(i_iter) + "_image", channel=1, nrow=4)
            visual_batch(image_segment, seg_help.config.tmp_dir, "train_" + str(i_iter) + "_image_label", channel=1,
                         nrow=4)

            (x1, pred_source1, pred_source2), _, su_cam = seg_help.model(image_patch, cam_p)
            loss_su_seg = seg_help.criterions['loss'](pred_source1, image_segment) \
                          + seg_help.criterions['loss'](pred_source2, image_segment) \
                          + Lambda_seg * seg_help.criterions['loss'](x1, image_segment)
            loss_cls_and_seg = loss_su_seg
            loss_cls_and_seg.backward()
            result.append(loss_su_seg.item())

            if results is None:
                results = Averagvalue(len(result))
            results.update(result)

            if (i_iter + 1) % seg_help.config.update_every == 0:
                clip_grad_norm_(filter(lambda p: p.requires_grad, seg_help.model.parameters()), \
                                max_norm=seg_help.config.clip)
                optimizer.step()
                optimizer.zero_grad()
                empty_cache()

            if (i_iter + 1) % 20 == 0 or i_iter == total_iter - 1:
                print(
                    "[Iter %d/%d] [seg loss: %f]" % (
                        i_iter,
                        total_iter,
                        results.avg[0]
                    )
                )
                dice_sc = predict_inf_inwindow_2d_out_seq_weight_cam(seg_help, seg_help.model,
                                                                        model_cam,
                                                                        covid_test_data,
                                                                        thres=0.3,
                                                                        epoch=i_iter)
                save_model_iter = join(seg_help.config.save_model_path, "M_%d_iter_%d_model.pt" % (m, i_iter))
                torch.save(seg_help.model.state_dict(), save_model_iter)

                if dice_sc >= best_acc:
                    print(" * Best vali acc: history = %.4f, current = %.4f" % (best_acc, dice_sc))
                    best_acc = dice_sc
                    save_model_path = join(seg_help.config.save_model_path, "M_%d_iter_best_model.pt" % (m))
                    torch.save(seg_help.model.state_dict(), save_model_path)
                    # predict_inf_inwindow_2d_out_seq_weight_cam_50(seg_help, seg_help.model, model_cam,
                    #                                            covid_test_data,
                    #                                            thres=0.3,
                    #                                            epoch=i_iter)
                for g in optimizer.param_groups:
                    print("optimizer current_lr to %.8f" % (g['lr']))
                seg_help.log.flush()

        # load_file = join(seg_help.config.save_model_path, "M_%d_iter_best_model.pt" % (m))
        state_dict = seg_help.load_model_history_weight(save_model_path)
        del model_former
        model_former = deepcopy(seg_help.model)
        model_former.load_state_dict(state_dict)
        model_former.eval()
        seg_help.reset_model()
        # optimizer = seg_help.reset_optim()
        del optimizer

    empty_cache()
    return {
        'train/seg_acc': results.avg[0]
    }


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
