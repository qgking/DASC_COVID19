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
    model = MODELS['adaptSegNet'](backbone=config.backbone, n_channels=config.channel,
                                  n_classes=config.classes)
    model_cam = MODELS['deeplabdilate2d_cam'](backbone='resnet34', n_channels=config.channel,
                                              n_classes=config.classes)

    latent_discriminator = DISCRIMINATOR['discriminator'](in_channels=config.classes, filters=64)
    seg_discriminator = DISCRIMINATOR['discriminator'](in_channels=config.classes, filters=64)
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
    latent_optimizer = OPTIM['adam'](
        params=filter(lambda p: p.requires_grad, latent_discriminator.parameters()),
        lr=seg_help.config.learning_rate_d, betas=(seg_help.config.beta_1, seg_help.config.beta_2))
    seg_optimizer = OPTIM['adam'](
        params=filter(lambda p: p.requires_grad, seg_discriminator.parameters()),
        lr=seg_help.config.learning_rate_d, betas=(seg_help.config.beta_1, seg_help.config.beta_2))
    seg_help.move_to_cuda()
    try:
        model_cam = seg_help.load_pretrained_cam_seg_model(model_cam)
    except FileExistsError as e:
        raise ValueError('file not exist')

    if seg_help.use_cuda:
        latent_discriminator.to(seg_help.equipment)
        seg_discriminator.to(seg_help.equipment)
        model_cam.to(seg_help.equipment)
        if len(seg_help.config.gpu_count) > 1 and seg_help.config.train:
            latent_discriminator = torch.nn.DataParallel(latent_discriminator, device_ids=seg_help.config.gpu_count)
            seg_discriminator = torch.nn.DataParallel(seg_discriminator, device_ids=seg_help.config.gpu_count)
            model_cam = torch.nn.DataParallel(model_cam, device_ids=seg_help.config.gpu_count)

    # optimizer, epoch_start = seg_help.load_hist_model_optim(optimizer)
    print("data name ", seg_help.config.data_name)
    train_loader, _ = seg_help.get_covid_infection_seg_data_loader_2d_slice(
        data_root=seg_help.config.data_root, pos=0.6)
    unsu_loader = seg_help.get_covid_infection_seg_unsu_data_loader_2d(
        data_root=seg_help.config.unsu_root, pos=0.6)
    train(seg_help, model_cam, train_loader, unsu_loader, latent_discriminator,
          seg_discriminator, optimizer, latent_optimizer, seg_optimizer)

    print("\n-----------load best state of model -----------")
    seg_help.load_best_state()
    seg_help.log.flush()

    seg_help.summary_writer.close()


def train(seg_help, model_cam, train_loader, unsu_loader, latent_discriminator,
          seg_discriminator, optimizer, latent_optimizer, seg_optimizer):
    results = None
    model_cam.eval()
    best_acc = 0
    batch_num_su = int(np.ceil(len(train_loader.dataset) / float(seg_help.config.train_batch_size)))
    batch_num_un = int(np.ceil(len(unsu_loader.dataset) / float(seg_help.config.train_batch_size)))
    total_iter = batch_num_su * seg_help.config.epochs
    print('train with train loader %d' % (len(train_loader.dataset)))
    print('batch_num_su %d,batch_num_un %d' % (batch_num_su, batch_num_un))
    print('total iter %d' % (total_iter))
    su_data_reader = seg_help.read_data(seg_help, train_loader)
    un_data_reader = seg_help.read_data(seg_help, unsu_loader)

    covid_test_data = seg_help.get_covid_data_2d(data_root=seg_help.config.unsu_root)
    su_label_tag = 0
    un_label_tag = 1
    Epsilon = 0.4
    Lambda_local = 10
    Lambda_adv = 0.001
    # Lambda_adv = 0.01
    Lambda_fea = 0.001
    # Lambda_fea = 0.01
    Lambda_weight = 0.01
    Lambda_seg = 3

    def get_y_truth_tensor(d_out_seg, fill):
        y_truth_tensor = torch.FloatTensor(d_out_seg.size())
        y_truth_tensor.fill_(fill)
        y_truth_tensor = y_truth_tensor.to(d_out_seg.get_device())
        return y_truth_tensor

    for i_iter in range(total_iter):
        seg_help.model.train()
        latent_discriminator.train()
        seg_discriminator.train()
        optimizer.zero_grad()
        latent_optimizer.zero_grad()
        seg_optimizer.zero_grad()

        seg_help.adjust_learning_rate(optimizer, i_iter, total_iter)
        seg_help.adjust_learning_rate_D(seg_optimizer, i_iter, total_iter)
        seg_help.adjust_learning_rate_D(latent_optimizer, i_iter, total_iter)
        damping = (1 - i_iter / total_iter)
        result = []
        # only train seg. Don't accumulate grads in disciminators
        seg_help.set_requires_grad(latent_discriminator, False)
        seg_help.set_requires_grad(seg_discriminator, False)
        # train on source
        # generate seg and cls data& label
        su_data, su_label = next(su_data_reader)
        with torch.no_grad():
            cam_p = model_cam.forward_cam(su_data)
        bb, cc, xx, yy = su_data.size()
        su_data = su_data.view(bb * cc, 1, xx, yy)
        bb, cc, xx, yy = su_label.size()
        su_label = su_label.view(bb * cc, xx, yy)

        (pred_source1, pred_source2), _, _ = seg_help.model(su_data, cam_p)

        loss_su_seg = 0.1 * seg_help.criterions['loss'](pred_source1, su_label) \
                      + seg_help.criterions['loss'](pred_source2, su_label)
        loss_cls_and_seg = loss_su_seg
        loss_cls_and_seg.backward()
        result.append(0)
        result.append(loss_su_seg.item())

        # adversarial training to fool the discriminator
        # generate seg and cls data without label
        un_data, un_label = next(un_data_reader)
        with torch.no_grad():
            cam_p = model_cam.forward_cam(un_data)
        (pred_target1, pred_target2), _, _ = seg_help.model(un_data, cam_p)
        # D entropy

        d_out_seg_1 = seg_discriminator(F.softmax(pred_target1, dim=1))

        d_out_seg_2 = latent_discriminator(F.softmax(pred_target2, dim=1))

        y_truth_tensor_1 = get_y_truth_tensor(d_out_seg_1, su_label_tag)
        y_truth_tensor_2 = get_y_truth_tensor(d_out_seg_2, su_label_tag)

        loss_adv = seg_help.criterions['bceloss'](d_out_seg_1, y_truth_tensor_1)
        loss_aux = seg_help.criterions['bceloss'](d_out_seg_2, y_truth_tensor_2)
        loss_adv_t = loss_adv * 0.0002 + loss_aux * 0.001
        loss_adv_t.backward()
        result.append(loss_aux.item())
        result.append(loss_adv.item())

        # Train discriminator networks
        # enable training mode on discriminator networks
        seg_help.set_requires_grad(latent_discriminator, True)
        seg_help.set_requires_grad(seg_discriminator, True)

        # Train with Source
        pred_source1 = pred_source1.detach()
        pred_source2 = pred_source2.detach()

        D_out_s_1 = seg_discriminator(F.softmax(pred_source1, dim=1))
        D_out_s_2 = latent_discriminator(F.softmax(pred_source2, dim=1))
        y_truth_tensor_1 = get_y_truth_tensor(D_out_s_1, su_label_tag)
        y_truth_tensor_2 = get_y_truth_tensor(D_out_s_2, su_label_tag)

        loss_D_s = seg_help.criterions['bceloss'](D_out_s_1, y_truth_tensor_1)
        loss_D_sf = seg_help.criterions['bceloss'](D_out_s_2, y_truth_tensor_2)
        loss_D_s.backward()
        loss_D_sf.backward()

        # Train with Target
        pred_target1 = pred_target1.detach()
        pred_target2 = pred_target2.detach()

        D_out_t_1 = seg_discriminator(F.softmax(pred_target1, dim=1))
        D_out_t_2 = latent_discriminator(F.softmax(pred_target2, dim=1))
        y_truth_tensor_1 = get_y_truth_tensor(D_out_t_1, un_label_tag)
        y_truth_tensor_2 = get_y_truth_tensor(D_out_t_2, un_label_tag)

        loss_D_t = seg_help.criterions['bceloss'](D_out_t_1, y_truth_tensor_1)
        loss_D_tf = seg_help.criterions['bceloss'](D_out_t_2, y_truth_tensor_2)
        loss_D_t.backward()
        loss_D_tf.backward()

        result.append(loss_D_s.item())
        result.append(loss_D_t.item())
        result.append(loss_D_sf.item())
        result.append(loss_D_tf.item())

        if results is None:
            results = Averagvalue(len(result))
        results.update(result)

        if (i_iter + 1) % seg_help.config.update_every == 0:
            clip_grad_norm_(filter(lambda p: p.requires_grad, seg_help.model.parameters()), \
                            max_norm=seg_help.config.clip)
            optimizer.step()
            latent_optimizer.step()
            seg_optimizer.step()
            optimizer.zero_grad()
            latent_optimizer.zero_grad()
            seg_optimizer.zero_grad()
            empty_cache()

        if (i_iter + 1) % batch_num_su == 0 or i_iter == total_iter - 1:
            print(
                "[Iter %d/%d] [cls loss: %f] [seg loss: %f] [adv_f loss: %f] [adv loss: %f] [loss_D_s: %f] [loss_D_t: %f] [loss_D_sf: %f] [loss_D_tf: %f]" % (
                    i_iter,
                    total_iter,
                    results.avg[0], results.avg[1], results.avg[2], results.avg[3], results.avg[4], results.avg[5],
                    results.avg[6], results.avg[7]
                )
            )
            dice_sc = predict_inf_inwindow_2d_out_seq_weight_cam(seg_help, seg_help.model, model_cam, covid_test_data,
                                                                 thres=0.3,
                                                                 epoch=i_iter)
            if dice_sc >= best_acc:
                print(" * Best vali acc: history = %.4f, current = %.4f" % (best_acc, dice_sc))
                best_acc = dice_sc
                seg_help.save_best_checkpoint(save_model=True, model_optimizer=None)
                predict_inf_inwindow_2d_out_seq_weight_cam_50(seg_help, seg_help.model, model_cam, covid_test_data,
                                                              thres=0.3,
                                                              epoch=i_iter)
            for g in optimizer.param_groups:
                print("optimizer current_lr to %.8f" % (g['lr']))
            for g in latent_optimizer.param_groups:
                print("latent_optimizer current_lr to %.8f" % (g['lr']))
            for g in seg_optimizer.param_groups:
                print("seg_optimizer current_lr to %.8f" % (g['lr']))

            seg_help.log.flush()
        if (i_iter + 1) % (seg_help.config.infer_epoch * 10) == 0 or i_iter == total_iter - 1:
            seg_help.save_model_checkpoint(i_iter, optimizer)

    empty_cache()
    return {
        'train/seg_loss': results.avg[0],
        'train/seg_acc': results.avg[1],
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
