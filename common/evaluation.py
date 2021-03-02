import numpy as np
from common.base_utls import *
from common.data_utils import *
import pandas as pd
from scipy import interp
import torch
from hausdorff import hausdorff_distance
import torch
import torch.nn as nn


def jaccard(y_true, y_pred):
    intersect = np.sum(y_true * y_pred)  # Intersection points
    union = np.sum(y_true) + np.sum(y_pred)  # Union points
    return (float(intersect)) / (union - intersect + 1e-7)


def dice_coef(y_true, y_pred):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(y_true).astype(np.bool)
    im2 = np.asarray(y_pred).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())


def compute_jaccard(y_true, y_pred):
    mean_jaccard = 0.
    thresholded_jaccard = 0.

    for im_index in range(y_pred.shape[0]):
        current_jaccard = jaccard(y_true=y_true[im_index], y_pred=y_pred[im_index])

        mean_jaccard += current_jaccard
        thresholded_jaccard += 0 if current_jaccard < 0.65 else current_jaccard

    mean_jaccard = mean_jaccard / y_pred.shape[0]
    thresholded_jaccard = thresholded_jaccard / y_pred.shape[0]

    return mean_jaccard, thresholded_jaccard


def get_accuracy(SR, GT, threshold=0.5):
    SR = (SR > threshold).int()
    GT = (GT == torch.max(GT)).int()
    corr = torch.sum(SR == GT)
    tensor_size = torch.prod(torch.tensor(SR.size()))
    acc = float(corr) / float(tensor_size)
    return acc


def get_sensitivity(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SR = (SR > threshold).int()
    GT = (GT == torch.max(GT)).int()

    # TP : True Positive
    # FN : False Negative
    TP = (((SR == 1).int() + (GT == 1).int()).int() == 2).int()
    FN = (((SR == 0).int() + (GT == 1).int()).int() == 2).int()

    SE = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)

    return SE


def get_specificity(SR, GT, threshold=0.5):
    SR = (SR > threshold).int()
    GT = (GT == torch.max(GT)).int()

    # TN : True Negative
    # FP : False Positive
    TN = (((SR == 0).int() + (GT == 0).int()).int() == 2).int()
    FP = (((SR == 1).int() + (GT == 0).int()).int() == 2).int()

    SP = float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-6)

    return SP


def get_precision(SR, GT, threshold=0.5):
    SR = (SR > threshold).int()
    GT = (GT == torch.max(GT)).int()

    # TP : True Positive
    # FP : False Positive
    TP = (((SR == 1).int() + (GT == 1).int()).int() == 2).int()
    FP = (((SR == 1).int() + (GT == 0).int()).int() == 2).int()

    PC = float(torch.sum(TP)) / (float(torch.sum(TP + FP)) + 1e-6)

    return PC


def get_F1(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity(SR, GT, threshold=threshold)
    PC = get_precision(SR, GT, threshold=threshold)

    F1 = 2 * SE * PC / (SE + PC + 1e-6)

    return F1


def get_JS(SR, GT, threshold=0.5):
    # JS : Jaccard similarity
    SR = (SR > threshold).int()
    GT = (GT == torch.max(GT)).int()

    Inter = torch.sum((SR + GT) == 2).int()
    Union = torch.sum((SR + GT) >= 1).int()

    JS = float(Inter) / (float(Union) + 1e-6)

    return JS


def diceCoeff(input, target):
    eps = 1e-6
    inter = torch.dot(input.view(-1), target.view(-1))
    union = torch.sum(input).int() + torch.sum(target) + eps
    t = (2 * inter.float() + eps) / union.float()
    return float(t)


def get_DC(SR, GT, threshold=0.5):
    # DC : Dice Coefficient
    SR = (SR > threshold).int()
    GT = (GT == torch.max(GT)).int()

    Inter = torch.sum((SR + GT) == 2).int()
    DC = float(2 * Inter) / (float(torch.sum(SR) + torch.sum(GT)) + 1e-6)

    return DC


def correct_predictions(output_probabilities, targets):
    """
    Compute the number of predictions that match some target classes in the
    output of a model.

    Args:
        output_probabilities: A tensor of probabilities for different output
            classes.
        targets: The indices of the actual target classes.

    Returns:
        The number of correct predictions in 'output_probabilities'.
    """
    _, out_classes = output_probabilities.max(dim=1)
    correct = (out_classes == targets).sum()
    total_num = torch.prod(torch.tensor(targets.size())).float()
    return (correct.item() / total_num).float()


def accuracy_check(mask, prediction):
    ims = [mask, prediction]
    np_ims = []
    for item in ims:
        if 'PIL' in str(type(item)):
            item = np.array(item)
        elif 'torch' in str(type(item)):
            item = item.cpu().detach().numpy()
        np_ims.append(item)
    compare = np.equal(np.where(np_ims[0] > 0.5, 1, 0), np_ims[1])
    accuracy = np.sum(compare)
    return accuracy / len(np_ims[0].flatten())


def compute_segmentation_scores(prediction_mask, reference_mask):
    """
    Calculates metrics scores from numpy arrays and returns an dict.

    Assumes that each object in the input mask has an integer label that
    defines object correspondence between prediction_mask and
    reference_mask.

    :param prediction_mask: numpy.array, int
    :param reference_mask: numpy.array, int
    :param voxel_spacing: list with x,y and z spacing
    :return: dict with dice, jaccard, voe, rvd, assd, rmsd, and msd
    """

    scores = {'dice': [],
              'jaccard': [],
              'voe': [],
              'rvd': []}

    for i, obj_id in enumerate(np.unique(prediction_mask)):
        if obj_id == 0:
            continue  # 0 is background, not an object; skip

        # Limit processing to the bounding box containing both the prediction
        # and reference objects.
        target_mask = (reference_mask == obj_id) + (prediction_mask == obj_id)
        bounding_box = ndimage.find_objects(target_mask)[0]
        p = (prediction_mask == obj_id)[bounding_box]
        r = (reference_mask == obj_id)[bounding_box]
        if np.any(p) and np.any(r):
            dice = metric.dc(p, r)
            jaccard = dice / (2. - dice)
            scores['dice'].append(dice)
            scores['jaccard'].append(jaccard)
            scores['voe'].append(1. - jaccard)
            scores['rvd'].append(metric.ravd(r, p))
    return scores


def border_map(binary_img, neigh):
    """
    Creates the border for a 3D image
    """
    binary_map = np.asarray(binary_img, dtype=np.uint8)
    neigh = neigh
    west = ndimage.shift(binary_map, [-1, 0, 0], order=0)
    east = ndimage.shift(binary_map, [1, 0, 0], order=0)
    north = ndimage.shift(binary_map, [0, 1, 0], order=0)
    south = ndimage.shift(binary_map, [0, -1, 0], order=0)
    top = ndimage.shift(binary_map, [0, 0, 1], order=0)
    bottom = ndimage.shift(binary_map, [0, 0, -1], order=0)
    cumulative = west + east + north + south + top + bottom
    border = ((cumulative < 6) * binary_map) == 1
    return border


def border_distance(ref, seg):
    """
    This functions determines the map of distance from the borders of the
    segmentation and the reference and the border maps themselves
    """
    neigh = 8
    border_ref = border_map(ref, neigh)
    border_seg = border_map(seg, neigh)
    oppose_ref = 1 - ref
    oppose_seg = 1 - seg
    # euclidean distance transform
    distance_ref = ndimage.distance_transform_edt(oppose_ref)
    distance_seg = ndimage.distance_transform_edt(oppose_seg)
    distance_border_seg = border_ref * distance_seg
    distance_border_ref = border_seg * distance_ref
    return distance_border_ref, distance_border_seg
    #  border_ref, border_seg


def Hausdorff_distance(ref, seg):
    """
    This functions calculates the average symmetric distance and the
    hausdorff distance between a segmentation and a reference image
    :return: hausdorff distance and average symmetric distance
    """
    ref_border_dist, seg_border_dist = border_distance(ref, seg)
    hausdorff_distance = np.max(
        [np.max(ref_border_dist), np.max(seg_border_dist)])
    return hausdorff_distance


from scipy.spatial.distance import directed_hausdorff as hausdorff


class HausdorffDistance(nn.Module):
    def __init__(self):
        super(HausdorffDistance, self).__init__()

    def forward(self, input, target):

        if isinstance(input, torch.Tensor):
            input = sitk.GetImageFromArray(input)
        if isinstance(target, torch.Tensor):
            target = sitk.GetImageFromArray(target)

        hausdorff_computer = sitk.HausdorffDistanceImageFilter()
        hausdorff_computer.Execute(input, target)

        return hausdorff_computer.GetHausdorffDistance()


def compute_all_metric_for_single_seg(y_true, y_pred):
    tensor_y_pred = torch.from_numpy(y_pred).cuda().float()
    tensor_y_true = torch.from_numpy(y_true).cuda().float()
    accuracy = get_accuracy(tensor_y_pred, tensor_y_true)
    sensitivity = get_sensitivity(tensor_y_pred, tensor_y_true)
    specificity = get_specificity(tensor_y_pred, tensor_y_true)
    # dice_score = diceCoeff(tensor_y_pred, tensor_y_true)
    dice_score = get_DC(tensor_y_pred, tensor_y_true)
    mean_jaccard = get_JS(tensor_y_pred, tensor_y_true)
    F1_score = get_F1(tensor_y_pred, tensor_y_true)
    # HD = HausdorffDistance()
    # hd = HD.forward(tensor_y_pred.cpu(), tensor_y_true.cpu())
    # hd =0
    # compute hausdorff distance
    seg_ind = np.argwhere(y_pred)
    gt_ind = np.argwhere(y_true)
    hd = max(hausdorff(seg_ind, gt_ind)[0], hausdorff(gt_ind, seg_ind)[0])
    scores = {  # 'ROC': [],
        # 'Precision-Recall': [],
        'Jaccard': [],
        'F1': [], 'ACCURACY': [], 'SENSITIVITY': [], 'SPECIFICITY': [],
        # 'PRECISION': [],
        # 'DICEDIST': [],
        'DICESCORE': [],
        'HD': []}
    # scores['ROC'].append(AUC_ROC)
    scores['Jaccard'].append(mean_jaccard)
    scores['F1'].append(F1_score)
    scores['ACCURACY'].append(accuracy)
    scores['SENSITIVITY'].append(sensitivity)
    scores['SPECIFICITY'].append(specificity)
    # scores['PRECISION'].append(precision)
    scores['DICESCORE'].append(dice_score)
    scores['HD'].append(hd)
    del tensor_y_pred
    del tensor_y_true
    return scores
