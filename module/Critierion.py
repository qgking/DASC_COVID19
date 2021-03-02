import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Function
from torch.nn import BCELoss, MSELoss, CrossEntropyLoss
import numpy as np
from torch.autograd import Variable


def make_one_hot(labels, classes):
    if len(labels.size()) == 4:
        one_hot = torch.cuda.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_()
    elif len(labels.size()) == 5:
        one_hot = torch.cuda.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3],
                                         labels.size()[4]).zero_()
    target = one_hot.scatter_(1, labels.data, 1)
    return target


class BCELoss(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.loss_fn = nn.BCELoss()

    def forward(self, model_output, data_input):
        masks_probs_flat = model_output.view(-1)
        true_masks_flat = data_input.float().view(-1)
        loss = self.loss_fn(masks_probs_flat, true_masks_flat)
        return loss


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + 1) / (m1.sum(1) + m2.sum(1) + 1)
        score = 1 - score.sum() / num
        return score


class WeightedBCEWithLogitsLoss(nn.Module):

    def __init__(self, size_average=True):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.size_average = size_average

    def weighted(self, input, target, weight, alpha, beta):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

        if weight is not None:
            loss = alpha * loss + beta * loss * weight

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

    def forward(self, input, target, weight, alpha, beta):
        if weight is not None:
            return self.weighted(input, target, weight, alpha, beta)
        else:
            return self.weighted(input, target, None, alpha, beta)


class CrossEntropy2d(nn.Module):

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target):
        N, C, H, W = predict.size()
        sm = nn.Softmax2d()

        P = sm(predict)
        P = torch.clamp(P, min=1e-9, max=1 - (1e-9))

        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask].view(1, -1)
        predict = P[target_mask.view(N, 1, H, W).repeat(1, C, 1, 1)].view(C, -1)
        probs = torch.gather(predict, dim=0, index=target)
        log_p = probs.log()
        batch_loss = -(torch.pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:

            loss = batch_loss.sum()
        return loss


class EntropyLoss(nn.Module):
    def __init__(self, ):
        super(EntropyLoss, self).__init__()

    def forward(self, v):
        """
            Entropy loss for probabilistic prediction vectors
            input: batch_size x channels x h x w
            output: batch_size x 1 x h x w
        """
        assert v.dim() == 4
        n, c, h, w = v.size()
        return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * h * w * np.log2(c))


class EMLoss(nn.Module):
    def __init__(self, ):
        super(EMLoss, self).__init__()

    def forward(self, d_real, d_fake):
        """
            Wasserstein distance
            input: batch_size x channels x h x w
            output: batch_size x 1 x h x w
        """
        assert d_real.dim() == 4
        return -torch.mean(d_real) + torch.mean(d_fake)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1., ignore_index=255):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, output, target):
        if self.ignore_index not in range(target.min(), target.max()):
            if (target == self.ignore_index).sum() > 0:
                target[target == self.ignore_index] = target.min()
        target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
        output = F.softmax(output, dim=1)
        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (output_flat * target_flat).sum()
        loss = 1 - ((2. * intersection + self.smooth) /
                    (output_flat.sum() + target_flat.sum() + self.smooth))
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, ignore_index=255, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.CE_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=alpha)

    def forward(self, output, target):
        logpt = self.CE_loss(output, target)
        pt = torch.exp(-logpt)
        loss = ((1 - pt) ** self.gamma) * logpt
        if self.size_average:
            return loss.mean()
        return loss.sum()


class KLDivLossWithLogits(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__(reduction)
        self.kl_div_loss = nn.KLDivLoss(reduction=reduction)

    def forward(self, logits1, logits2):
        return self.kl_div_loss(F.log_softmax(logits1, dim=1), F.softmax(logits2, dim=1))


class Multi_scale_DC_FC_loss(nn.Module):
    def __init__(self, ):
        super(Multi_scale_DC_FC_loss, self).__init__()
        self.ce = FocalLoss()
        self.dc = DiceLoss()

    def forward(self, net_output, target):
        result = 0
        for out in net_output:
            dc_loss = self.dc(out, target)
            ce_loss = self.ce(out, target)
            result += (ce_loss + dc_loss)
        return result


class DC_and_FC_loss(nn.Module):
    def __init__(self, ):
        super(DC_and_FC_loss, self).__init__()
        self.ce = FocalLoss()
        self.dc = DiceLoss()
        # self.gamma1 = Parameter(torch.ones(1))

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target)
        ce_loss = self.ce(net_output, target)
        # flb_loss_value = flb_loss.item()
        # dc_loss_value = dc_loss.item()
        # result = dc_loss_value * flb_loss + dc_loss * flb_loss_value
        result = ce_loss + dc_loss
        # result = self.gamma1 * flb_loss + dc_loss
        return result


class BinaryFocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param reduction: `none`|`mean`|`sum`
    :param **kwargs
        balance_index: (int) balance class index, should be specific when alpha is float
    """

    def __init__(self, alpha=[1.0, 1.0], gamma=2, ignore_index=None, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        if alpha is None:
            alpha = [0.25, 0.75]
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = 1e-6
        self.ignore_index = ignore_index
        self.reduction = reduction

        assert self.reduction in ['none', 'mean', 'sum']

        if self.alpha is None:
            self.alpha = torch.ones(2)
        elif isinstance(self.alpha, (list, np.ndarray)):
            self.alpha = np.asarray(self.alpha)
            self.alpha = np.reshape(self.alpha, (2))
            assert self.alpha.shape[0] == 2, \
                'the `alpha` shape is not match the number of class'
        elif isinstance(self.alpha, (float, int)):
            self.alpha = np.asarray([self.alpha, 1.0 - self.alpha], dtype=np.float).view(2)

        else:
            raise TypeError('{} not supported'.format(type(self.alpha)))

    def forward(self, output, target):
        prob = output
        prob = torch.clamp(prob, self.smooth, 1.0 - self.smooth)

        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()

        pos_loss = -self.alpha[0] * torch.pow(torch.sub(1.0, prob), self.gamma) * torch.log(prob) * pos_mask
        neg_loss = -self.alpha[1] * torch.pow(prob, self.gamma) * \
                   torch.log(torch.sub(1.0, prob)) * neg_mask

        neg_loss = neg_loss.sum()
        pos_loss = pos_loss.sum()
        num_pos = pos_mask.view(pos_mask.size(0), -1).sum()
        num_neg = neg_mask.view(neg_mask.size(0), -1).sum()

        if num_pos == 0:
            loss = neg_loss
        else:
            loss = pos_loss / num_pos + neg_loss / num_neg
        return loss


class DC_and_Focal_loss(nn.Module):
    def __init__(self, ):
        super(DC_and_Focal_loss, self).__init__()
        self.flb = BinaryFocalLoss()
        self.dc = SoftDiceLoss()
        # self.gamma1 = Parameter(torch.ones(1))

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target)
        flb_loss = self.flb(net_output, target)
        # flb_loss_value = flb_loss.item()
        # dc_loss_value = dc_loss.item()
        # result = dc_loss_value * flb_loss + dc_loss * flb_loss_value
        result = flb_loss + dc_loss
        # result = self.gamma1 * flb_loss + dc_loss
        return result


class DC_and_CE_loss(nn.Module):
    def __init__(self, ):
        super(DC_and_CE_loss, self).__init__()
        self.ce = CrossEntropyLoss()
        self.dc = DiceLoss()
        # self.gamma1 = Parameter(torch.ones(1))

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target)
        ce_loss = self.ce(net_output, target)
        # flb_loss_value = flb_loss.item()
        # dc_loss_value = dc_loss.item()
        # result = dc_loss_value * flb_loss + dc_loss * flb_loss_value
        result = ce_loss + dc_loss
        # result = self.gamma1 * flb_loss + dc_loss
        return result


#
# # https://github.com/pytorch/pytorch/issues/1249
# def dice_coeff(pred, target):
#     smooth = 1.
#     num = pred.size(0)
#     m1 = pred.view(num, -1)  # Flatten
#     m2 = target.view(num, -1)  # Flatten
#     intersection = (m1 * m2).sum()
#
#     return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)


device = 0


class ReconLoss(nn.Module):
    def __init__(self, reduction='mean', masked=False):
        super().__init__()
        self.loss_fn = nn.L1Loss(reduction=reduction)
        self.masked = masked

    def forward(self, data_input, model_output):
        outputs = model_output['outputs']
        targets = data_input['targets']
        if self.masked:
            masks = data_input['masks']
            return self.loss_fn(outputs * (1 - masks), targets * (1 - masks))
        else:
            return self.loss_fn(outputs, targets)


class VGGLoss(nn.Module):
    def __init__(self, vgg):
        super().__init__()
        self.vgg = vgg
        self.l1_loss = nn.L1Loss()

    def vgg_loss(self, output, target):
        output_feature = self.vgg(output.repeat(1, 3, 1, 1))
        target_feature = self.vgg(target.repeat(1, 3, 1, 1))
        loss = (
                self.l1_loss(output_feature.relu2_2, target_feature.relu2_2)
                + self.l1_loss(output_feature.relu3_3, target_feature.relu3_3)
                + self.l1_loss(output_feature.relu4_3, target_feature.relu4_3)
        )
        return loss

    def forward(self, targets, outputs):
        # Note: It can be batch-lized
        mean_image_loss = []
        for frame_idx in range(targets.size(-1)):
            mean_image_loss.append(
                self.vgg_loss(outputs[..., frame_idx], targets[..., frame_idx])
            )

        mean_image_loss = torch.stack(mean_image_loss, dim=0).mean(dim=0)
        return mean_image_loss


class StyleLoss(nn.Module):
    def __init__(self, vgg, original_channel_norm=True):
        super().__init__()
        self.vgg = vgg
        self.l1_loss = nn.L1Loss()
        self.original_channel_norm = original_channel_norm

    # From https://github.com/pytorch/tutorials/blob/master/advanced_source/neural_style_tutorial.py
    def gram_matrix(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)

    # Implement "Image Inpainting for Irregular Holes Using Partial Convolutions", Liu et al., 2018
    def style_loss(self, output, target):
        output_features = self.vgg(output.repeat(1, 3, 1, 1))
        target_features = self.vgg(target.repeat(1, 3, 1, 1))
        layers = ['relu2_2', 'relu3_3', 'relu4_3']  # n_channel: 128 (=2 ** 7), 256 (=2 ** 8), 512 (=2 ** 9)
        loss = 0
        for i, layer in enumerate(layers):
            output_feature = getattr(output_features, layer)
            target_feature = getattr(target_features, layer)
            B, C_P, H, W = output_feature.shape
            output_gram_matrix = self.gram_matrix(output_feature)
            target_gram_matrix = self.gram_matrix(target_feature)
            if self.original_channel_norm:
                C_P_square_divider = 2 ** (i + 1)  # original design (avoid too small loss)
            else:
                C_P_square_divider = C_P ** 2
                assert C_P == 128 * 2 ** i
            loss += self.l1_loss(output_gram_matrix, target_gram_matrix) / C_P_square_divider
        return loss

    def forward(self, targets, outputs):
        # Note: It can be batch-lized
        mean_image_loss = []
        for frame_idx in range(targets.size(-1)):
            mean_image_loss.append(
                self.style_loss(outputs[..., frame_idx], targets[..., frame_idx])
            )

        mean_image_loss = torch.stack(mean_image_loss, dim=0).mean(dim=0)
        return mean_image_loss


class L1LossMaskedMean(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss(reduction='sum')

    def forward(self, x, y, masked):
        l1_sum = self.l1(x * masked, y * masked)
        return l1_sum / torch.sum(masked)


class L2LossMaskedMean(nn.Module):
    def __init__(self, reduction='sum'):
        super().__init__()
        self.l2 = nn.MSELoss(reduction=reduction)

    def forward(self, x, y, mask):
        l2_sum = self.l2(x * mask, y * mask)
        return l2_sum / torch.sum(mask)


# From https://github.com/jxgu1016/Total_Variation_Loss.pytorch
class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, data_input, model_output):
        # View 3D data as 2D
        outputs = model_output['outputs']
        B, L, C, H, W = outputs.shape
        x = outputs.view([B * L, C, H, W])

        masks = data_input['masks']
        masks = masks.view([B * L, -1])
        mask_areas = masks.sum(dim=1)

        h_x = x.size()[2]
        w_x = x.size()[3]
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum(1).sum(1).sum(1)
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum(1).sum(1).sum(1)
        return ((h_tv + w_tv) / mask_areas).mean()


# Based on https://github.com/knazeri/edge-connect/blob/master/src/loss.py
class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='lsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge | l1
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label).to(device))
        self.register_buffer('fake_label', torch.tensor(target_fake_label).to(device))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

        elif type == 'l1':
            self.criterion = nn.L1Loss()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss


class KLMSELoss(nn.Module):
    def __init__(self):
        super(KLMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(size_average=False)

    def forward(self, recon_x, x, mu, logvar):
        MSE = self.mse_loss(recon_x, x)
        # see Appendix B from VAE paper:    https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE + KLD
