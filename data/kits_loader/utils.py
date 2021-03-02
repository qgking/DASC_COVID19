from pathlib import Path

import nibabel as nib


def get_full_case_id(cid):
    try:
        cid = int(cid)
        case_id = "case_{:05d}".format(cid)
    except ValueError:
        case_id = cid

    return case_id


def load_volume(case_path):
    vol = nib.load(str(case_path + "imaging.nii.gz"))
    return vol


def load_segmentation(case_path):
    seg = nib.load(str(case_path + "segmentation.nii.gz"))
    return seg


def load_case(cid):
    vol = load_volume(cid)
    seg = load_segmentation(cid)
    return vol, seg
