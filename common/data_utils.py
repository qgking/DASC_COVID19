from common.base_utls import *


def process_dicom_file(csv_path, min_img_bound, max_img_bound, min_msk_bound, max_msk_bound):
    imgs, msks = [], []
    df = pd.read_csv(csv_path)
    for i, item in df.iterrows():
        img = read_dicom_series(item[0])
        mask = read_liver_lesion_masks(item[1])
        print('train data select dicom file:' + item[0])
        img = set_bounds(img, min_img_bound, max_img_bound)
        mask = set_bounds(mask, min_msk_bound, max_msk_bound)
        imgs.append(img)
        msks.append(mask)
    return (imgs, msks)


def read_dicom_series(directory, filepattern="image_*"):
    """ Reads a DICOM Series files in the given directory.
    Only filesnames matching filepattern will be considered"""

    if not os.path.exists(directory) or not os.path.isdir(directory):
        raise ValueError("Given directory does not exist or is a file : " + str(directory))
    print('\tRead Dicom', directory)
    lstFilesDCM = natsort.natsorted(glob(os.path.join(directory, filepattern)))
    print('\tLength dicom series', len(lstFilesDCM))
    # Get ref file
    RefDs = dicom.read_file(lstFilesDCM[0])
    # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
    ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))
    # The array is sized based on 'ConstPixelDims'
    ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

    # loop through all the DICOM files
    for filenameDCM in lstFilesDCM:
        # read the file
        ds = dicom.read_file(filenameDCM)
        # store the raw image data
        ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array

    return ArrayDicom


def rotate(matrix):
    """
    :type matrix: List[List[int]]
    :rtype: void Do not return anything, modify matrix in-place instead.
    """
    matrix[:] = map(list, zip(*matrix[::-1]))


def generate_nii_from_dicom():
    # generate nii from dicom
    # import dicom2nifti
    for i in range(1, 21):
        lbl = read_liver_lesion_masks(
            'data/3Dircadb1/3Dircadb1.' + str(i) + '/MASKS_DICOM')
        imgnii = nib.load(
            'data/3Dircadb1/3Dircadb1.' + str(i) + '/3Dircadb1.' + str(
                i) + '.nii')
        img = imgnii.get_data()
        sp = img.shape

        nib.save(nib.Nifti1Image(lbl, affine=imgnii.get_affine()),
                 'data/3Dircadb1/3Dircadb1.' + str(
                     i) + '/3Dircadb_gt_1.' + str(i) + '.nii')


def read_liver_lesion_masks(masks_dirname):
    """Since 3DIRCAD provides an individual mask for each tissue type (in DICOM series format),
    we merge multiple tissue types into one Tumor mask, and merge this mask with the liver mask

    Args:
        masks_dirname : MASKS_DICOM directory containing multiple DICOM series directories,
                        one for each labelled mask
    Returns:
        numpy array with 0's for background pixels, 1's for liver pixels and 2's for tumor pixels
    """
    tumor_volume = None
    liver_volume = None

    # For each relevant organ in the current volume
    for organ in os.listdir(masks_dirname):
        organ_path = os.path.join(masks_dirname, organ)
        if not os.path.isdir(organ_path):
            continue

        organ = organ.lower()

        if organ.startswith("livertumor") or re.match("liver.yst.*", organ) or organ.startswith(
                "stone") or organ.startswith("metastasecto"):
            print('Organ', masks_dirname, organ)
            current_tumor = read_dicom_series(organ_path)
            current_tumor = np.clip(current_tumor, 0, 1)
            # Merge different tumor masks into a single mask volume
            tumor_volume = current_tumor if tumor_volume is None else np.logical_or(tumor_volume, current_tumor)
        elif organ == 'liver':
            print('Organ', masks_dirname, organ)
            liver_volume = read_dicom_series(organ_path)
            liver_volume = np.clip(liver_volume, 0, 1)

    # Merge liver and tumor into 1 volume with background=0, liver=1, tumor=2
    label_volume = np.zeros(liver_volume.shape)
    label_volume[liver_volume == 1] = 1
    label_volume[tumor_volume == 1] = 2
    label_final = np.zeros(label_volume.shape)
    for j in range(label_volume.shape[-1]):
        im = label_volume[:, :, j]
        rotate(im)
        label_final[:, :, j] = im
    return label_final


# ====================================================
# ======================volume preprocessing method===
# ====================================================
def to_scale(img, slice_shape, shape=None):
    if shape is None:
        shape = slice_shape

    height, width = shape
    if img.dtype == SEG_DTYPE:
        return scipy.misc.imresize(img, (height, width), interp="nearest").astype(SEG_DTYPE)
    elif img.dtype == IMG_DTYPE:
        factor = 256.0 / np.max(img)
        return (scipy.misc.imresize(img, (height, width), interp="nearest") / factor).astype(IMG_DTYPE)
    else:
        raise TypeError(
            'Error. To scale the image array, its type must be np.uint8 or np.float64. (' + str(img.dtype) + ')')


def sitkResample(sitk_image, new_space, is_label=False):
    '''
    sitk_image:
    new_spacing: x,y,z
    is_label: if True, using Interpolator `sitk.sitkNearestNeighbor`
    '''
    size = np.array(sitk_image.GetSize())
    spacing = np.array(sitk_image.GetSpacing())
    new_spacing = np.array(new_space)
    new_size = size * spacing / new_spacing
    new_spacing_refine = size * spacing / new_size
    new_spacing_refine = [float(s) for s in new_spacing_refine]
    new_size = [int(s) for s in new_size]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputDirection(sitk_image.GetDirection())
    resample.SetOutputOrigin(sitk_image.GetOrigin())
    resample.SetSize(new_size)
    resample.SetOutputSpacing(new_spacing_refine)

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        # resample.SetInterpolator(sitk.sitkBSpline)
        resample.SetInterpolator(sitk.sitkLinear)

    newimage = resample.Execute(sitk_image)
    return newimage


def nibResample(img, seg, scan, new_voxel_dim=[1, 1, 1]):
    # Get voxel size
    voxel_dim = np.array(scan.header.structarr["pixdim"][1:4], dtype=np.float32)
    print('old voxel dim')
    print(voxel_dim)
    # Resample to optimal [1,1,1] voxel size
    resize_factor = voxel_dim / new_voxel_dim
    scan_shape = np.array(scan.header.get_data_shape())
    new_scan_shape = scan_shape * resize_factor
    rounded_new_scan_shape = np.round(new_scan_shape)
    rounded_resize_factor = rounded_new_scan_shape / scan_shape  # Change resizing due to round off error
    new_voxel_dim = voxel_dim / rounded_resize_factor

    img = nd.interpolation.zoom(img, rounded_resize_factor, mode='nearest')
    seg = nd.interpolation.zoom(seg, rounded_resize_factor, mode='nearest')
    return img, seg, new_voxel_dim


def norm_hounsfield_dyn(arr, c_min=0.1, c_max=0.3):
    """ Converts from hounsfield units to float64 image with range 0.0 to 1.0 """
    # calc min and max
    min, max = np.amin(arr), np.amax(arr)
    if min <= 0:
        arr = np.clip(arr, min * c_min, max * c_max)
        # right shift to zero
        arr = np.abs(min * c_min) + arr
    else:
        arr = np.clip(arr, min, max * c_max)
        # left shift to zero
        arr = arr - min
    # normalization
    norm_fac = np.amax(arr)
    if norm_fac != 0:
        norm = np.divide(
            np.multiply(arr, 255),
            np.amax(arr))
    else:  # don't divide through 0
        norm = np.multiply(arr, 255)

    norm = np.clip(np.multiply(norm, 0.00390625), 0, 1)
    return norm


# Get majority label in image
def largest_label_volume(img, bg=0):
    vals, counts = np.unique(img, return_counts=True)
    counts = counts[vals != bg]
    vals = vals[vals != bg]
    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None


def label_connected_component(pred):
    seg = measure.label(pred, neighbors=8, background=0)
    return seg


# brain utils
def generate_patch_locations(patches, patch_size, im_size):
    nx = round((patches * 8 * im_size[0] * im_size[0] / im_size[1] / im_size[2]) ** (1.0 / 3))
    ny = round(nx * im_size[1] / im_size[0])
    nz = round(nx * im_size[2] / im_size[0])
    x = np.rint(np.linspace(patch_size, im_size[0] - patch_size, num=nx))
    y = np.rint(np.linspace(patch_size, im_size[1] - patch_size, num=ny))
    z = np.rint(np.linspace(patch_size, im_size[2] - patch_size, num=nz))
    return x, y, z


def perturb_patch_locations(patch_locations, radius):
    x, y, z = patch_locations
    x = np.rint(x + np.random.uniform(-radius, radius, len(x)))
    y = np.rint(y + np.random.uniform(-radius, radius, len(y)))
    z = np.rint(z + np.random.uniform(-radius, radius, len(z)))
    return x, y, z


def generate_patch_probs(path, patch_locations, patch_size, im_size, tagTumor=0):
    x, y, z = patch_locations
    seg = nib.load(glob(os.path.join(path, '*_seg.nii.gz'))[0]).get_data().astype(np.float32)
    p = []
    for i in range(len(x)):
        for j in range(len(y)):
            for k in range(len(z)):
                patch = seg[int(x[i] - patch_size / 2): int(x[i] + patch_size / 2),
                        int(y[j] - patch_size / 2): int(y[j] + patch_size / 2),
                        int(z[k] - patch_size / 2): int(z[k] + patch_size / 2)]
                patch = (patch > tagTumor).astype(np.float32)
                percent = np.sum(patch) / (patch_size * patch_size * patch_size)
                p.append((1 - np.abs(percent - 0.5)) * percent)
    p = np.asarray(p, dtype=np.float32)
    p[p == 0] = np.amin(p[np.nonzero(p)])
    p = p / np.sum(p)
    return p


def normalize_roi(im_input):
    x_start = im_input.shape[0] // 4
    x_range = im_input.shape[0] // 2
    y_start = im_input.shape[1] // 4
    y_range = im_input.shape[1] // 2
    z_start = im_input.shape[2] // 4
    z_range = im_input.shape[2] // 2
    roi = im_input[x_start: x_start + x_range, y_start: y_start + y_range, z_start: z_start + z_range]
    roi = (im_input - np.mean(roi)) / np.std(roi)
    # rescale to 0  1
    im_output = (roi - np.min(roi)) / (np.max(roi) - np.min(roi))
    return im_output


def read_image(path, is_training=True):
    t1 = nib.load(glob(os.path.join(path, '*_t1.nii.gz'))[0]).get_data().astype(np.float32)
    t1ce = nib.load(glob(os.path.join(path, '*_t1ce.nii.gz'))[0]).get_data().astype(np.float32)
    t2 = nib.load(glob(os.path.join(path, '*_t2.nii.gz'))[0]).get_data().astype(np.float32)
    flair = nib.load(glob(os.path.join(path, '*_flair.nii.gz'))[0]).get_data().astype(np.float32)
    assert t1.shape == t1ce.shape == t2.shape == flair.shape
    if is_training:
        seg = nib.load(glob(os.path.join(path, '*_seg.nii.gz'))[0]).get_data().astype(np.float32)
        assert t1.shape == seg.shape
        nchannel = 5
    else:
        nchannel = 4

    image = np.empty((t1.shape[0], t1.shape[1], t1.shape[2], nchannel), dtype=np.float32)

    # image[..., 0] = remove_low_high(t1)
    # image[..., 1] = remove_low_high(t1ce)
    # image[..., 2] = remove_low_high(t2)
    # image[..., 3] = remove_low_high(flair)
    image[..., 0] = normalize_roi(t1)
    image[..., 1] = normalize_roi(t1ce)
    image[..., 2] = normalize_roi(t2)
    image[..., 3] = normalize_roi(flair)

    if is_training:
        image[..., 4] = np.clip(seg, 0, 1)

    return image


def generate_test_locations(patch_size, stride, im_size):
    stride_size_x = patch_size[0] / stride
    stride_size_y = patch_size[1] / stride
    stride_size_z = patch_size[2] / stride
    pad_x = (
        int(patch_size[0] / 2),
        int(np.ceil(im_size[0] / stride_size_x) * stride_size_x - im_size[0] + patch_size[0] / 2))
    pad_y = (
        int(patch_size[1] / 2),
        int(np.ceil(im_size[1] / stride_size_y) * stride_size_y - im_size[1] + patch_size[1] / 2))
    pad_z = (
        int(patch_size[2] / 2),
        int(np.ceil(im_size[2] / stride_size_z) * stride_size_z - im_size[2] + patch_size[2] / 2))
    x = np.arange(patch_size[0] / 2, im_size[0] + pad_x[0] + pad_x[1] - patch_size[0] / 2 + 1, stride_size_x)
    y = np.arange(patch_size[1] / 2, im_size[1] + pad_y[0] + pad_y[1] - patch_size[1] / 2 + 1, stride_size_y)
    z = np.arange(patch_size[2] / 2, im_size[2] + pad_z[0] + pad_z[1] - patch_size[2] / 2 + 1, stride_size_z)
    return (x, y, z), (pad_x, pad_y, pad_z)


def min_max_voi(mask, superior=10, inferior=10):
    sp = mask.shape
    tp = np.transpose(np.nonzero(mask))
    minx, miny, minz = np.min(tp, axis=0)
    maxx, maxy, maxz = np.max(tp, axis=0)
    minz = 0 if minz - superior < 0 else minz - superior
    maxz = sp[-1] if maxz + inferior >= sp[-1] else maxz + inferior + 1
    miny = 0 if miny - superior < 0 else miny - superior
    maxy = sp[1] if maxy + inferior >= sp[1] else maxy + inferior + 1
    minx = 0 if minx - superior < 0 else minx - superior
    maxx = sp[0] if maxx + inferior >= sp[0] else maxx + inferior + 1
    return minx, maxx, miny, maxy, minz, maxz

def min_max_voi_2d(mask, superior=10, inferior=10):
    sp = mask.shape
    tp = np.transpose(np.nonzero(mask))
    minx, miny = np.min(tp, axis=0)
    maxx, maxy = np.max(tp, axis=0)
    miny = 0 if miny - superior < 0 else miny - superior
    maxy = sp[1] if maxy + inferior >= sp[1] else maxy + inferior + 1
    minx = 0 if minx - superior < 0 else minx - superior
    maxx = sp[0] if maxx + inferior >= sp[0] else maxx + inferior + 1
    return minx, maxx, miny, maxy



def overlay(volume_ims, segmentation_ims, segmentation, alpha=0.3, k_color=[255, 0, 0], t_color=[0, 0, 255]):
    im_volume = 255 * volume_ims
    volume_ims = np.stack((im_volume, im_volume, im_volume), axis=-1)
    segmentation_ims = class_to_color(segmentation_ims, k_color=k_color, t_color=t_color)
    # Get binary array for places where an ROI lives
    segbin = np.greater(segmentation, 0)
    repeated_segbin = np.stack((segbin, segbin, segbin), axis=-1)
    # Weighted sum where there's a value to overlay
    overlayed = np.where(
        repeated_segbin,
        np.round(alpha * segmentation_ims + (1 - alpha) * volume_ims).astype(np.uint8),
        np.round(volume_ims).astype(np.uint8)
    )
    return overlayed


def class_to_color(segmentation, k_color, t_color):
    # initialize output to zeros
    shp = segmentation.shape
    seg_color = np.zeros((shp[0], shp[1], shp[2], 3), dtype=np.float32)

    # set output to appropriate color at each location
    seg_color[np.equal(segmentation, 1)] = k_color
    # seg_color[np.equal(segmentation, 2)] = t_color
    return seg_color


def gen_patches(image, seg, seg_unique, patch_per_modality, padding, new_shape, rootdir, vol, thres=30, filter=2):
    rotate_angle = [0, 90, 180, 270]
    # rotate_axis = [(1, 0), (1, 2), (2, 0)]
    rotate_axis = [(1, 0)]

    for unique in seg_unique:
        seg_uni = seg.copy()
        # image_uni = image.copy()
        total_pixels = np.sum(seg_uni == unique)
        if total_pixels < thres:
            print("total pixels less skip")
            continue
        print("total pixels:" + str(total_pixels))
        # image_uni[np.where(seg_uni == unique)] = 1
        seg_uni = np.where(seg_uni != unique, 0, 1)
        minx, maxx, miny, maxy, minz, maxz = min_max_voi(seg_uni, superior=0, inferior=0)
        print((maxx - minx, maxy - miny, maxz - minz))

        # image[minx:maxx, miny:maxy, minz: maxz, 0:4] = 1
        # generate cube mask
        # imgtmp = image[minx:maxx, miny:maxy, minz: maxz, -2]
        # seg_uni = seg_uni[minx:maxx, miny:maxy, minz: maxz]
        # tsp = seg_uni.shape
        # x = np.linspace(0, tsp[0], tsp[0], endpoint=False)
        # y = np.linspace(0, tsp[1], tsp[1], endpoint=False)
        # z = np.linspace(0, tsp[2], tsp[2], endpoint=False)
        # # Manipulate x,y,z here to obtain the dimensions you are looking for
        #
        # center = np.array([tsp[0] // 2, tsp[1] // 2, tsp[2] // 2])
        #
        # # Generate grid of points
        # X, Y, Z = np.meshgrid(x, y, z)
        # data = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
        #
        # distance = sp.distance.cdist(data, center.reshape(1, -1)).ravel()
        # points_in_sphere = data[distance < np.max(center)]

        # from copy import deepcopy
        #
        # ''' size : size of original 3D numpy matrix A.
        #     radius : radius of circle inside A which will be filled with ones.
        # '''
        # size, radius = 5, 2
        #
        # ''' A : numpy.ndarray of shape size*size*size. '''
        # A = np.zeros((size, size, size))
        #
        # ''' AA : copy of A (you don't want the original copy of A to be overwritten.) '''
        # AA = deepcopy(A)
        #
        # ''' (x0, y0, z0) : coordinates of center of circle inside A. '''
        # x0, y0, z0 = int(np.floor(A.shape[0] / 2)), \
        #              int(np.floor(A.shape[1] / 2)), int(np.floor(A.shape[2] / 2))
        #
        # for x in range(x0 - radius, x0 + radius + 1):
        #     for y in range(y0 - radius, y0 + radius + 1):
        #         for z in range(z0 - radius, z0 + radius + 1):
        #             ''' deb: measures how far a coordinate in A is far from the center.
        #                     deb>=0: inside the sphere.
        #                     deb<0: outside the sphere.'''
        #             deb = radius - abs(x0 - x) - abs(y0 - y) - abs(z0 - z)
        #             if (deb) >= 0: AA[x, y, z] = 1

        for num in range(patch_per_modality):
            mminx = np.random.randint(minx - padding, minx)
            mmaxx = np.random.randint(maxx, maxx + padding)

            mminy = np.random.randint(miny - padding, miny)
            mmaxy = np.random.randint(maxy, maxy + padding)

            mminz = np.random.randint(minz - padding // 2, minz)
            mmaxz = np.random.randint(maxz, maxz + padding // 2)

            # mminx = minx - padding
            mminx = mminx if mminx > 0 else 0
            # mmaxx = maxx + padding
            # mminy = miny - padding
            mminy = mminy if mminy > 0 else 0
            # mmaxy = maxy + padding
            # mminz = minz - padding // 4
            mminz = mminz if mminz > 0 else 0
            # mmaxz = maxz + padding // 4
            mmaxz = mmaxz if mmaxz < image.shape[2] else image.shape[2] - 1

            # for fit the network, two downsample and concat
            # mmaxz = mmaxz - (mmaxz - mminz) % 4
            # mmaxy = mmaxy - (mmaxy - mminy) % 4
            # mmaxx = mmaxx - (mmaxx - mminx) % 4
            print((mminx, mmaxx, mminy, mmaxy, mminz, mmaxz))

            ori_npy = image[mminx:mmaxx, mminy:mmaxy, mminz:mmaxz]
            seg_npy = seg_uni[mminx:mmaxx, mminy:mmaxy, mminz:mmaxz]
            # boudray sigma 1.0      tumor sigma 5.0
            # boundary_npy = ((
            #                     filters.gaussian_filter(
            #                         255. * np.multiply(np.invert(morph.binary_erosion(seg_npy)), seg_npy),
            #                         1.0)) / 255.).reshape(seg_npy.shape)

            # boundary_npy = np.where(np.multiply(np.invert(morph.binary_erosion(seg_npy)), seg_npy) > 0, 1, 0)
            # tumor_npy = ((
            #                  filters.gaussian_filter(255. * np.multiply(morph.binary_erosion(seg_npy), seg_npy),
            #                                          3.0)) / 255).reshape(seg_npy.shape)
            # tumor_npy = np.multiply(morph.binary_erosion(seg_npy), seg_npy).reshape(seg_npy.shape)
            resize_factor = np.array(new_shape, dtype=np.float32) / [ori_npy.shape[0], ori_npy.shape[1],
                                                                     ori_npy.shape[2]]
            ori_npy = cpzoom(ori_npy, resize_factor, order=1)
            seg_npy = cpzoom(seg_npy, resize_factor, order=1)
            if np.random.random() > 0.5:
                rotate_a = np.random.randint(len(rotate_angle))
                rotate_b = np.random.randint(len(rotate_axis))
                angle = rotate_angle[rotate_a]
                raxis = rotate_axis[rotate_b]
                ori_npy = cprotate(ori_npy, angle, raxis, order=1)
                seg_npy = cprotate(seg_npy, angle, raxis, order=1)

            # ori_npy = nd.interpolation.zoom(ori_npy, resize_factor, mode='wrap')
            # seg_npy = nd.interpolation.zoom(seg_npy, resize_factor, mode='wrap')
            # create boundary
            seg_npy = np.where(seg_npy > 0, 1, 0)
            boundary_npy = ((
                                filters.gaussian_filter(
                                    255. * np.multiply(np.invert(morph.binary_erosion(seg_npy)), seg_npy),
                                    1.0)) / 255).reshape(seg_npy.shape)
            boundary_npy = np.where(boundary_npy > np.max(boundary_npy) / filter, 1, 0)

            # create tumor region
            tumor_npy = seg_npy.copy()

            # create mask,normalization
            # ori_npy = normalize_(ori_npy)
            msk_npy = ori_npy.copy()
            msk_npy[seg_npy > 0] = 1

            save_file = np.stack([ori_npy, msk_npy, tumor_npy, boundary_npy, seg_npy], -1)
            # for sss in range(save_file.shape[-2]):
            #     visualize(np.expand_dims(save_file[..., 0][:, :, sss], -1),
            #               join(rootdir, 'img_' + '{0:0>3}'.format(vol) + '_' + '{0:0>1}'.format(
            #                   unique) + '_' + '{0:0>2}'.format(
            #                   num) + str(sss) + "Araw"))
            #     visualize(np.expand_dims(save_file[..., 1][:, :, sss], -1),
            #               join(rootdir, 'img_' + '{0:0>3}'.format(vol) + '_' + '{0:0>1}'.format(
            #                   unique) + '_' + '{0:0>2}'.format(
            #                   num) + str(sss) + "Berase"))
            #     visualize(np.expand_dims(save_file[..., 2][:, :, sss], -1),
            #               join(rootdir, 'img_' + '{0:0>3}'.format(vol) + '_' + '{0:0>1}'.format(
            #                   unique) + '_' + '{0:0>2}'.format(
            #                   num) + str(sss)+ "Ctumor"))
            #     visualize(np.expand_dims(save_file[..., 3][:, :, sss], -1),
            #               join(rootdir, 'img_' + '{0:0>3}'.format(vol) + '_' + '{0:0>1}'.format(
            #                   unique) + '_' + '{0:0>2}'.format(
            #                   num) + str(sss) + "Dboundary"))
            #     visualize(np.expand_dims(save_file[..., 4][:, :, sss], -1),
            #               join(rootdir, 'img_' + '{0:0>3}'.format(vol) + '_' + '{0:0>1}'.format(
            #                   unique) + '_' + '{0:0>2}'.format(
            #                   num) + str(sss) + "Eseg"))
            np.save(
                rootdir + 'img_' + '{0:0>3}'.format(vol) + '_' + '{0:0>1}'.format(
                    unique) + '_' + '{0:0>2}'.format(
                    num) + '.npy', save_file)


def gen_patches_2d(image, seg, seg_unique, patch_per_modality, padding, new_shape, rootdir, vol, thres=30, filter=2):
    rotate_angle = [0, 90, 180, 270]
    rotate_axis = [(1, 0), (1, 2), (2, 0)]
    for unique in seg_unique:
        seg_uni = seg.copy()
        # image_uni = image.copy()
        total_pixels = np.sum(seg_uni == unique)
        if total_pixels < thres:
            print("total pixels less skip")
            continue
        print("total pixels:" + str(total_pixels))
        # image_uni[np.where(seg_uni == unique)] = 1
        seg_uni = np.where(seg_uni != unique, 0, 1)
        minx, maxx, miny, maxy, minz, maxz = min_max_voi(seg_uni, superior=0, inferior=0)
        print((maxx - minx, maxy - miny, maxz - minz))
        # TODO delete
        # seg_uni[minx:maxx, miny:maxy, minz:maxz] = 1

        # tmp = np.where(seg_uni > 0, 1, 0)
        # for sss in range(image_uni.shape[-1]):
        #     visualize(np.expand_dims(image[..., sss], -1),
        #               join(TMP_DIR, str(sss) + "Araw"))
        #     visualize(np.expand_dims(image_uni[..., sss], -1),
        #               join(TMP_DIR, str(sss) + "Berase"))
        #     visualize(np.expand_dims(tmp[..., sss], -1),
        #               join(TMP_DIR, str(sss) + "Cseg"))

        # image[minx:maxx, miny:maxy, minz: maxz, 0:4] = 1
        # generate cube mask
        # imgtmp = image[minx:maxx, miny:maxy, minz: maxz, -2]
        # seg_uni = seg_uni[minx:maxx, miny:maxy, minz: maxz]
        # tsp = seg_uni.shape
        # x = np.linspace(0, tsp[0], tsp[0], endpoint=False)
        # y = np.linspace(0, tsp[1], tsp[1], endpoint=False)
        # z = np.linspace(0, tsp[2], tsp[2], endpoint=False)
        # # Manipulate x,y,z here to obtain the dimensions you are looking for
        #
        # center = np.array([tsp[0] // 2, tsp[1] // 2, tsp[2] // 2])
        #
        # # Generate grid of points
        # X, Y, Z = np.meshgrid(x, y, z)
        # data = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
        #
        # distance = sp.distance.cdist(data, center.reshape(1, -1)).ravel()
        # points_in_sphere = data[distance < np.max(center)]

        # from copy import deepcopy
        #
        # ''' size : size of original 3D numpy matrix A.
        #     radius : radius of circle inside A which will be filled with ones.
        # '''
        # size, radius = 5, 2
        #
        # ''' A : numpy.ndarray of shape size*size*size. '''
        # A = np.zeros((size, size, size))
        #
        # ''' AA : copy of A (you don't want the original copy of A to be overwritten.) '''
        # AA = deepcopy(A)
        #
        # ''' (x0, y0, z0) : coordinates of center of circle inside A. '''
        # x0, y0, z0 = int(np.floor(A.shape[0] / 2)), \
        #              int(np.floor(A.shape[1] / 2)), int(np.floor(A.shape[2] / 2))
        #
        # for x in range(x0 - radius, x0 + radius + 1):
        #     for y in range(y0 - radius, y0 + radius + 1):
        #         for z in range(z0 - radius, z0 + radius + 1):
        #             ''' deb: measures how far a coordinate in A is far from the center.
        #                     deb>=0: inside the sphere.
        #                     deb<0: outside the sphere.'''
        #             deb = radius - abs(x0 - x) - abs(y0 - y) - abs(z0 - z)
        #             if (deb) >= 0: AA[x, y, z] = 1

        for num in range(patch_per_modality):
            # mminx = np.random.randint(minx - padding, minx)
            # mmaxx = np.random.randint(maxx, maxx + padding)
            # mminy = np.random.randint(miny - padding, miny)
            # mmaxy = np.random.randint(maxy, maxy + padding)
            # mminz = np.random.randint(minz - padding, minz)
            mminx = np.random.randint(minx - padding, minx)
            mmaxx = np.random.randint(maxx, maxx + padding)
            mminy = np.random.randint(miny - padding, miny)
            mmaxy = np.random.randint(maxy, maxy + padding)

            mminx = mminx if mminx > 0 else 0
            mminy = mminy if mminy > 0 else 0
            mminz = minz
            mmaxz = maxz

            # mminx = minx - padding
            # mminx = mminx if mminx > 0 else 0
            # mmaxx = maxx + padding
            # mminy = miny - padding
            # mminy = mminy if mminy > 0 else 0
            # mmaxy = maxy + padding
            # mminz = minz
            # mmaxz = maxz
            print((mminx, mmaxx, mminy, mmaxy, mminz, mmaxz))
            ori_npy = image[mminx:mmaxx, mminy:mmaxy, mminz:mmaxz]
            seg_npy = seg_uni[mminx:mmaxx, mminy:mmaxy, mminz:mmaxz]

            boundary_npy = ((
                                filters.gaussian_filter(
                                    255. * np.multiply(np.invert(morph.binary_erosion(seg_npy)), seg_npy),
                                    1.0)) / 255).reshape(seg_npy.shape)
            boundary_npy = np.where(boundary_npy > np.max(boundary_npy) / filter, 1, 0)
            tumor_npy = seg_npy.copy()
            new_shape[-1] = ori_npy.shape[2]
            resize_factor = np.array(new_shape, dtype=np.float32) / [ori_npy.shape[0], ori_npy.shape[1],
                                                                     ori_npy.shape[2]]
            ori_npy = cpzoom(ori_npy, resize_factor, order=1)
            seg_npy = cpzoom(seg_npy, resize_factor, order=1)
            boundary_npy = cpzoom(boundary_npy, resize_factor, order=1)
            tumor_npy = cpzoom(tumor_npy, resize_factor, order=1)

            # ori_npy = nd.interpolation.zoom(ori_npy, resize_factor, mode='wrap')
            # msk_npy = nd.interpolation.zoom(msk_npy, resize_factor, mode='wrap')
            # seg_npy = nd.interpolation.zoom(seg_npy, resize_factor, mode='wrap')
            msk_npy = ori_npy.copy()
            msk_npy[seg_npy > 0] = 1
            save_file = np.stack([ori_npy, msk_npy, tumor_npy, boundary_npy, seg_npy], -1)
            for sss in range(save_file.shape[-2]):
                total_pixels = np.sum(save_file[..., 2][:, :, sss] > 0)
                if total_pixels < 15:
                    print("total pixels less skip")
                    continue
                save_file_2d = np.stack(
                    [save_file[..., 0][:, :, sss], save_file[..., 1][:, :, sss], save_file[..., 2][:, :, sss],
                     save_file[..., 3][:, :, sss], save_file[..., 4][:, :, sss]], -1)
                np.save(
                    rootdir + 'img_' + '{0:0>3}'.format(vol) + '_' + '{0:0>1}'.format(
                        unique) + '_' + '{0:0>2}'.format(
                        sss) + '.npy', save_file_2d)
                # visualize(np.expand_dims(save_file[..., 0][:, :, sss], -1),
                #           join(TMP_DIR, str(sss) + "Araw"))
                # visualize(np.expand_dims(save_file[..., 1][:, :, sss], -1),
                #           join(TMP_DIR, str(sss) + "Berase"))
                # visualize(np.expand_dims(save_file[..., 2][:, :, sss], -1),
                #           join(TMP_DIR, str(sss) + "Ctumor"))
                # visualize(np.expand_dims(save_file[..., 3][:, :, sss], -1),
                #           join(TMP_DIR, str(sss) + "Dboundary"))
                # visualize(np.expand_dims(save_file[..., 4][:, :, sss], -1),
                #           join(TMP_DIR, str(sss) + "Eseg"))


def set_bounds(img, min_bound, max_bound):
    image = np.clip(img, min_bound, max_bound)
    image = image.astype(np.float32)
    return image


def fill_holes(img, thres):
    coarse_pred = np.where(img.copy() > thres, 1, 0)
    coarse_pred = ndimage.binary_erosion(coarse_pred, iterations=1).astype(coarse_pred.dtype)
    [lung_labels, num] = measure.label(coarse_pred, return_num=True)
    region = measure.regionprops(lung_labels)
    box = []
    for i in range(num):
        box.append(region[i].area)
    human_body_idx = box.index(sorted(box, reverse=True)[0]) + 1
    human_body = np.where((lung_labels == human_body_idx), 1, 0)
    slices = np.sum(human_body, axis=2)
    slices = np.where(slices > 0, 1, 0)
    # slices = ndimage.binary_dilation(slices, iterations=4).astype(slices.dtype)
    tp = np.transpose(np.nonzero(slices))
    min_x, _ = np.min(tp, axis=0)
    max_x, _ = np.max(tp, axis=0)
    for x in range(min_x, max_x):
        tp = np.transpose(np.nonzero(slices[x]))
        min_y = np.min(tp, axis=0)[0]
        max_y = np.max(tp, axis=0)[0]
        slices[x, min_y:max_y] = 1
    # slices = ndimage.binary_closing(slices).astype(coarse_pred.dtype)
    slices = np.expand_dims(slices, axis=-1)
    slices = np.tile(slices, human_body.shape[-1])
    return slices
