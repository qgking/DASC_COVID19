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
from module.Critierion import DiceLoss, DC_and_FC_loss, EntropyLoss, EMLoss, Multi_scale_DC_FC_loss
from torchvision.utils import make_grid
from driver import OPTIM
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss

matplotlib.use("Agg")
import configparser

config = configparser.ConfigParser()
import argparse

log_template = "[Epoch %d/%d] [seg loss: %f] [seg acc: %f]"


def main(config):
    model = MODELS[config.model](backbone=config.backbone, n_channels=config.patch_z, n_classes=config.classes)
    seg_discriminator = DISCRIMINATOR['discriminator'](in_channels=config.classes, filters=64)
    criterion = {
        'loss': DC_and_FC_loss(),
        'bceloss': BCEWithLogitsLoss(),
    }
    if 'u2net' in config.model:
        criterion['loss'] = Multi_scale_DC_FC_loss()
    seg_help = DASEGHelper(model, criterion,
                           config)
    optimizer = seg_help.reset_optim()
    seg_optimizer = OPTIM[seg_help.config.learning_algorithm](
        params=filter(lambda p: p.requires_grad, seg_discriminator.parameters()),
        lr=seg_help.config.learning_rate, weight_decay=1e-4)
    seg_help.move_to_cuda()

    if seg_help.use_cuda:
        seg_discriminator.to(seg_help.equipment)
        if len(seg_help.config.gpu_count) > 1 and seg_help.config.train:
            seg_discriminator = torch.nn.DataParallel(seg_discriminator, device_ids=seg_help.config.gpu_count)

    optimizer, epoch_start = seg_help.load_hist_model_optim(optimizer)
    print("data name ", seg_help.config.data_name)
    train_loader, valid_loader = seg_help.get_covid_infection_seg_data_loader(
        data_root=seg_help.config.data_root, pos=0.3)
    unsu_loader = seg_help.get_covid_infection_seg_unsu_data_loader(
        data_root=seg_help.config.unsu_root, pos=0.3)
    best_acc = 0
    bad_step = 0
    for epoch in range(epoch_start, seg_help.config.epochs):
        train_critics = {
            'train/seg_loss': 0,
            'train/seg_acc': 0,
        }
        train_critics_seg = train(seg_help, train_loader, unsu_loader, seg_discriminator, optimizer, seg_optimizer,
                                  epoch)
        train_critics.update(train_critics_seg)
        vali_critics = {
            'vali/seg_loss': 0,
            'vali/seg_acc': 0,
        }
        vali_critics_seg = test(seg_help, valid_loader, epoch)
        vali_critics.update(vali_critics_seg)

        seg_help.save_model_checkpoint(epoch, optimizer)
        seg_help.log.flush()
        if vali_critics['vali/seg_acc'] >= best_acc:
            print(" * Best vali acc: history = %.4f, current = %.4f" % (best_acc, vali_critics['vali/seg_acc']))
            best_acc = vali_critics['vali/seg_acc']
            bad_step = 0
            seg_help.write_summary(epoch, vali_critics)
            seg_help.save_best_checkpoint(save_model=True, model_optimizer=None)
        else:
            bad_step += 1
            if bad_step == 1:
                seg_help.save_best_checkpoint(save_model=False, model_optimizer=optimizer)
            if bad_step >= seg_help.config.bad_step:
                bad_step = 0
                seg_help.load_best_state()
                optimizer = seg_help.load_best_optim(optimizer)
                for g in optimizer.param_groups:
                    current_lr = max(g['lr'] * 0.1, seg_help.config.min_lrate)
                    print("Decaying the learning ratio to %.8f" % (current_lr))
                    g['lr'] = current_lr

    print("\n-----------load best state of model -----------")
    seg_help.load_best_state()
    seg_help.log.flush()

    seg_help.summary_writer.close()


def train(seg_help, train_loader, unsu_loader, seg_discriminator, optimizer, seg_optimizer, epoch):
    seg_help.model.train()
    seg_discriminator.train()
    results = Averagvalue(5)
    optimizer.zero_grad()
    seg_optimizer.zero_grad()
    su_label_tag = 0
    un_label_tag = 1
    middle_slice = int(np.floor(seg_help.config.patch_z // 2))
    batch_num_su = int(np.ceil(len(train_loader.dataset) / float(seg_help.config.train_batch_size)))
    batch_num_un = int(np.ceil(len(unsu_loader.dataset) / float(seg_help.config.train_batch_size)))
    batch_num = max(batch_num_su, batch_num_un) * 2
    print('train with train loader %d' % (len(train_loader.dataset)))
    print('batch_num_su %d,batch_num_un %d' % (batch_num_su, batch_num_un))
    su_data_reader = seg_help.read_data(seg_help, train_loader)
    un_data_reader = seg_help.read_data(seg_help, unsu_loader)

    def get_y_truth_tensor(d_out_seg, fill):
        y_truth_tensor = torch.FloatTensor(d_out_seg.size())
        y_truth_tensor.fill_(fill)
        y_truth_tensor = y_truth_tensor.to(d_out_seg.get_device())
        return y_truth_tensor

    for i in range(batch_num):
        result = []
        # only train seg. Don't accumulate grads in disciminators
        seg_help.set_requires_grad(seg_discriminator, False)

        # train on source
        su_data, su_label = next(su_data_reader)
        su_logits, _, _ = seg_help.model(su_data)
        loss_su_seg = seg_help.criterions['loss'](su_logits, su_label[:, middle_slice, :, :])
        loss_su_seg.backward()
        if 'u2net' in seg_help.config.model:
            prob_su = torch.softmax(su_logits[0], dim=1)
        else:
            prob_su = torch.softmax(su_logits, dim=1)
        acc = correct_predictions(prob_su, su_label[:, middle_slice, :, :])
        result.append(loss_su_seg.item())
        result.append(acc.item())
        del loss_su_seg
        del acc
        # adversarial training ot fool the discriminator
        un_data, un_label = next(un_data_reader)
        un_logits, un_aspp, _ = seg_help.model(un_data)
        if 'u2net' in seg_help.config.model:
            prob_un = torch.softmax(un_logits[0], dim=1)
        else:
            prob_un = torch.softmax(un_logits, dim=1)
        d_out_seg = seg_discriminator(seg_help.prob_2_entropy(prob_un))
        y_truth_tensor = get_y_truth_tensor(d_out_seg, su_label_tag)
        d_out_seg = seg_help.criterions['bceloss'](d_out_seg, y_truth_tensor)
        d_out_seg.backward()
        result.append(d_out_seg.item())

        # Train discriminator networks
        # enable training mode on discriminator networks
        seg_help.set_requires_grad(seg_discriminator, True)
        # train with source
        if 'u2net' in seg_help.config.model:
            su_logits = su_logits[0].detach()
        else:
            su_logits = su_logits.detach()
        prob_su = torch.softmax(su_logits, dim=1)
        d_out_seg = seg_discriminator(seg_help.prob_2_entropy(prob_su))
        y_truth_tensor = get_y_truth_tensor(d_out_seg, su_label_tag)
        d_out_seg = seg_help.criterions['bceloss'](d_out_seg, y_truth_tensor)
        d_out_seg = d_out_seg / 2
        d_out_seg.backward()
        result.append(d_out_seg.item())

        # train with target
        if 'u2net' in seg_help.config.model:
            un_logits = un_logits[0].detach()
        else:
            un_logits = un_logits.detach()
        prob_un = torch.softmax(un_logits, dim=1)
        d_out_seg = seg_discriminator(seg_help.prob_2_entropy(prob_un))
        y_truth_tensor = get_y_truth_tensor(d_out_seg, un_label_tag)
        d_out_seg = seg_help.criterions['bceloss'](d_out_seg, y_truth_tensor)
        d_out_seg = d_out_seg / 2
        d_out_seg.backward()
        result.append(d_out_seg.item())
        results.update(result)

        # visual_batch(su_data, seg_help.config.tmp_dir, str(i) + "_su_data", channel=1, nrow=8)
        #
        # visual_batch(su_label, seg_help.config.tmp_dir, str(i) + "_su_label", channel=1, nrow=8)
        #
        # prob_max = prob_su.max(1)[1]
        # image_save = prob_max.unsqueeze(1).contiguous()
        # visual_batch(image_save, seg_help.config.tmp_dir, str(i) + "_su_predict", channel=1, nrow=8)
        #
        # visual_batch(un_data, seg_help.config.tmp_dir, str(i) + "_un_data", channel=1, nrow=8)
        #
        # visual_batch(un_label, seg_help.config.tmp_dir, str(i) + "_un_label", channel=1, nrow=8)
        #
        # prob_max = prob_un.max(1)[1]
        # image_save = prob_max.unsqueeze(1).contiguous()
        # visual_batch(image_save, seg_help.config.tmp_dir, str(i) + "_un_predict", channel=1, nrow=8)

        if (i + 1) % seg_help.config.update_every == 0 or i == batch_num - 1:
            clip_grad_norm_(filter(lambda p: p.requires_grad, seg_help.model.parameters()), \
                            max_norm=seg_help.config.clip)
            optimizer.step()
            seg_optimizer.step()
            optimizer.zero_grad()
            seg_optimizer.zero_grad()
        empty_cache()
        # Print log
    print(
        log_template % (
            epoch,
            seg_help.config.epochs,
            results.avg[0], results.avg[1]
        )
    )
    empty_cache()
    return {
        'train/seg_loss': results.avg[0],
        'train/seg_acc': results.avg[1],
    }


def test(seg_help, test_loader, epoch):
    seg_help.model.eval()
    results = Averagvalue(3)
    dice_score = DiceLoss()
    middle_slice = int(np.floor(seg_help.config.patch_z // 2))
    batch_num_su = int(np.ceil(len(test_loader.dataset) / float(seg_help.config.test_batch_size)))
    batch_num = batch_num_su
    print('train with train loader %d' % (len(test_loader.dataset)))
    su_data_reader = seg_help.read_data(seg_help, test_loader)
    for i in range(batch_num):
        su_data, su_label = next(su_data_reader)
        with torch.no_grad():
            logits, _, _ = seg_help.model(su_data)
        loss = seg_help.criterions['loss'](logits, su_label[:, middle_slice, :, :])
        if 'u2net' in seg_help.config.model:
            prob = torch.softmax(logits[0], dim=1)
            dice = 1 - dice_score(logits[0], su_label[:, middle_slice, :, :])
        else:
            prob = torch.softmax(logits, dim=1)
            dice = 1 - dice_score(logits, su_label[:, middle_slice, :, :])
        acc = correct_predictions(prob, su_label[:, middle_slice, :, :])
        result = [loss.item(), acc.item(), dice.item()]
        results.update(result)

        # visual_batch(su_data * seg_help.std + seg_help.mean, seg_help.config.tmp_dir, str(i) + "_images", channel=1,
        #              nrow=8)
        visual_batch(su_data , seg_help.config.tmp_dir, str(i) + "_images", channel=1,
                     nrow=8)
        visual_batch(su_label, seg_help.config.tmp_dir, str(i) + "_label", channel=1, nrow=8)

        prob_max = prob.max(1)[1]
        image_save = prob_max.unsqueeze(1).contiguous()
        visual_batch(image_save, seg_help.config.tmp_dir, str(i) + "_predict", channel=1, nrow=8)

        empty_cache()
        # measure elapsed time
    info = 'Vali Epoch: [{0}/{1}]'.format(epoch, seg_help.config.epochs) + \
           ' Loss {loss:f} '.format(loss=results.avg[0]) + \
           ' Acc {acc:f} '.format(acc=results.avg[1]) + \
           ' Dice {dice:f} '.format(dice=results.avg[2])
    print(info)
    return {
        'vali/seg_loss': results.avg[0],
        'vali/seg_acc': results.avg[1],
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
