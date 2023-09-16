import numpy as np
import os
import torch
import SimpleITK as sitk
import nibabel as nib
from models import models
from scipy.interpolate import NearestNDInterpolator


def compute_landmark_accuracy(landmarks_pred, landmarks_gt, voxel_size):
    landmarks_pred = np.round(landmarks_pred)
    landmarks_gt = np.round(landmarks_gt)

    difference = landmarks_pred - landmarks_gt
    difference = np.abs(difference)
    difference = difference * voxel_size

    means = np.mean(difference, 0)
    stds = np.std(difference, 0)

    difference = np.square(difference)
    difference = np.sum(difference, 1)
    difference = np.sqrt(difference)

    means = np.append(means, np.mean(difference))
    stds = np.append(stds, np.std(difference))

    means = np.round(means, 2)
    stds = np.round(stds, 2)

    means = means[::-1]
    stds = stds[::-1]

    return means, stds


def compute_landmarks(network, landmarks_pre, image_size):
    scale_of_axes = [(0.5 * s) for s in image_size]

    coordinate_tensor = torch.FloatTensor(landmarks_pre / (scale_of_axes)) - 1.0

    output = network(coordinate_tensor.cuda())

    delta = output.cpu().detach().numpy() * (scale_of_axes)

    return landmarks_pre + delta, delta


def load_image_DIRLab(variation=1, folder=r"D:\Data\DIRLAB\Case", mode='train'):
    # Size of data, per image pair
    image_sizes = [
        0,
        [94, 256, 256],
        [112, 256, 256],
        [104, 256, 256],
        [99, 256, 256],
        [106, 256, 256],
        [128, 512, 512],
        [136, 512, 512],
        [128, 512, 512],
        [128, 512, 512],
        [120, 512, 512],
    ]

    # Scale of data, per image pair
    voxel_sizes = [
        0,
        [2.5, 0.97, 0.97],
        [2.5, 1.16, 1.16],
        [2.5, 1.15, 1.15],
        [2.5, 1.13, 1.13],
        [2.5, 1.1, 1.1],
        [2.5, 0.97, 0.97],
        [2.5, 0.97, 0.97],
        [2.5, 0.97, 0.97],
        [2.5, 0.97, 0.97],
        [2.5, 0.97, 0.97],
    ]

    shape = image_sizes[variation]

    folder = folder + str(variation) + r"Pack" + os.path.sep

    # Images
    dtype = np.dtype(np.int16)

    if mode == "train":
        insp_phase = "00"
        exp_phase = "50"
    elif mode == "finetune":
        insp_phase = "40"
        exp_phase = "60"
    with open(folder + "Images/case" + str(variation) + f"_T{insp_phase}_s.img", "rb") as f:
        data = np.fromfile(f, dtype)
    image_insp = data.reshape(shape)

    with open(folder + "Images/case" + str(variation) + f"_T{exp_phase}_s.img", "rb") as f:
        data = np.fromfile(f, dtype)
    image_exp = data.reshape(shape)

    # imgsitk_in = sitk.ReadImage(folder + r"Masks\case" + str(variation) + "_T00_s.mhd")

    # mask = np.clip(sitk.GetArrayFromImage(imgsitk_in), 0, 1)
    mask = np.ones(image_insp.shape)

    image_insp = torch.FloatTensor(image_insp)
    image_exp = torch.FloatTensor(image_exp)

    # Landmarks
    with open(
        folder + "ExtremePhases/Case" + str(variation) + "_300_T00_xyz.txt"
    ) as f:
        landmarks_insp = np.array(
            [list(map(int, line[:-1].split("\t")[:3])) for line in f.readlines()]
        )

    with open(
        folder + "ExtremePhases/Case" + str(variation) + "_300_T50_xyz.txt"
    ) as f:
        landmarks_exp = np.array(
            [list(map(int, line[:-1].split("\t")[:3])) for line in f.readlines()]
        )

    landmarks_insp[:, [0, 2]] = landmarks_insp[:, [2, 0]]
    landmarks_exp[:, [0, 2]] = landmarks_exp[:, [2, 0]]

    return (
        image_insp,
        image_exp,
        torch.zeros(image_insp.shape + (3,)).to(image_insp),
        torch.from_numpy(mask).to(image_insp),
        landmarks_insp,
        landmarks_exp,
        mask,
        voxel_sizes[variation],
    )


def load_image_liver_motion(variation=1, folder="/mnt/ibrixfs04-Kspace/motion_patients", mode='train'):
    patients = ["ARENDS_WILLIAM_100448365_2017NOV29_131132_EOVIST",
                "MURTON_ROBERT_18051141_2018MAY31_113605_EOVIST",
                # "TRUELOVE_CHARLES_101461481_2022JAN07_130825_EOVIST",
                # "GIETZEN_ANDREW_101410533_2021OCT26_114742_EOVIST",
                # "WILSON_PHILIP_040866620_2020SEP16_093357_EOVIST",
                "BROADWORTH_RICKEY_29938374_2017NOV01_101511_EOVIST",
                "MILLER_CAROL_100512736_2017JUN16_152234_MULTIHANCE",
                "HEINROTH_ROBERT_100583154_2018FEB21_094222_EOVIST"]
    patient = patients[variation]
    img_folder = folder + f"/{patient}" + "/reconstruction/Tornado5_SMIN90_SMAX180_STRIDE90_DMC"
    dvf_folder = folder + f"/{patient}" + "/reconstruction/Tornado5_SMIN90_SMAX180_STRIDE90_DMC_DEF"
    images = {}
    dvfs = {}
    first_phase_time = 10000
    temp_resolution = 90 * 150 / 1000  # 13.5s
    for file in os.listdir(img_folder):
        if file.endswith(".img") and file.startswith("Sequence"):
            phase_time = int(file[-10:-4]) * 150 / 1000
            img = nib.load(os.path.join(f"{img_folder}/", file))
            img_numpy = img.get_fdata().astype(np.float32)
            images[phase_time] = np.flip(img_numpy, axis=2).transpose(2, 1, 0)  # (i->s, a->p, r->l)
            if phase_time < first_phase_time:
                first_phase_time = phase_time
    for file in os.listdir(dvf_folder):
        if file.endswith(".img") and file.startswith("DeformationField"):
            state = int(file[-6:-4])
            dvf = nib.load(os.path.join(f"{dvf_folder}/", file))
            dvf_numpy = dvf.get_fdata().astype(np.float32)
            dvf_numpy = np.squeeze(np.flip(dvf_numpy, axis=2).transpose(2, 1, 0, 3, 4))   # (i->s, a->p, r->l)
            dvf_numpy = np.flip(dvf_numpy, axis=3)
            dvf_numpy[:, :, :, 0] = -dvf_numpy[:, :, :, 0]
            dvfs[state] = dvf_numpy
    mid_phase_time = 3680 * 150 / 1000  # 9.2 min, state 37
    last_phase_time = 6819 * 150 / 1000  # 17.0 min (1022.85s)
    voxel_size = [img.affine[2, 2], img.affine[1, 1], img.affine[0, 0]]

    if mode == "train":
        moving_phase_time = mid_phase_time + temp_resolution
        # moving_phase_time = 27.0
        fixed_phase_time = mid_phase_time + temp_resolution * 2
        # fixed_phase_time = mid_phase_time + temp_resolution * 35
        # fixed_phase_time = last_phase_time
        moving_state = 37
        fixed_state = 39
        # moving_state = 1
        # fixed_state = 72
    elif mode == "finetune":
        moving_phase_time = mid_phase_time + temp_resolution * 20  # 13.7 min
        fixed_phase_time = mid_phase_time + temp_resolution * 22  # 14.2 min
        moving_state = 57
        fixed_state = 59

    fixed_image = images[fixed_phase_time]
    moving_image = images[moving_phase_time]
    # moving_image = np.roll(fixed_image, 20, axis=2)
    # dvf = -(dvfs[moving_state] - dvfs[fixed_state]) * np.array(voxel_size).reshape(1, 1, 1, 3)
    dvf = dvfs[moving_state] - dvfs[fixed_state]  # It seems the unit of DVFs are in mm.
    dvf_72_to_moving = dvfs[moving_state] - dvfs[72]

    # mask = np.ones(fixed_image.shape)
    voi_paths = {"Liver": "/RadOnc-MRI1/Student_Folder/jiarenz/projects/NeRP_motion/data/voi/Sequence_0000_VOI.txt",
                 "Stomach": "/mnt/ibrixfs04-Kspace/motion_patients/Jiaren_test/test_vois/Stomach_OAR.txt",
                 "Duodenum": "/mnt/ibrixfs04-Kspace/motion_patients/Jiaren_test/test_vois/Duodenum_OAR.txt",
                 "Colon": "/mnt/ibrixfs04-Kspace/motion_patients/Jiaren_test/test_vois/Colon_OAR.txt",
                 "Bowel_small": "/mnt/ibrixfs04-Kspace/motion_patients/Jiaren_test/test_vois/Bowel_small_OAR.txt"}
    vois = {}
    for k, v in voi_paths.items():
        voi_txt = open(voi_paths[k]).read().split()
        voi_state_72 = np.zeros_like(fixed_image)
        for text in voi_txt:
            if text[:5] == "Slice":
                text = text.split("_")
                voi_state_72[int(text[1]) - 1, int(text[3]), int(text[4])] = 1
        voi_state_72 = np.flip(voi_state_72, axis=0).transpose(0, 2, 1)
        normalized_dvf = torch.from_numpy(dvf_72_to_moving).cuda() / torch.tensor(voxel_size).cuda().reshape(1, 1, 1, 3)
        normalized_dvf = normalized_dvf * 2 / torch.tensor(fixed_image.shape).cuda().reshape(1, 1, 1, 3)
        normalized_dvf = normalized_dvf.view(-1, 3)
        coordinate_tensor = [torch.linspace(-1, 1, fixed_image.shape[i]) for i in range(3)]
        X, Y, Z = torch.meshgrid(*coordinate_tensor)
        possible_coordinate_tensor = make_masked_coordinate_tensor(np.ones(fixed_image.shape),
                                                                   dims=fixed_image.shape)
        normalized_dvf = normalized_dvf + possible_coordinate_tensor
        interp = NearestNDInterpolator(normalized_dvf.cpu(), voi_state_72.flatten())
        voi_moving_state = interp(X, Y, Z)
        vois[k] = torch.from_numpy(voi_moving_state.copy()).cuda()

    mask = fixed_image > np.max(fixed_image) / 150
    # for i in range(fixed_image.shape[0]):
    #     for j in range(fixed_image.shape[1]):
    #         for k in range(fixed_image.shape[2]):
    #             if f"Slice_{i+1}_voxel_{j}_{k}" in voi_txt:
    #                 voi[i, j, k] = 1
    # roi_names = ['Bowel_small', 'Colon', 'Stomach', 'Duodenum']
    # vois = {}
    # for roi_name in roi_names:
    #     voi = nib.load(f"/mnt/ibrixfs01-FUNCI/Liver_2015_039/patients/BROADWORTH_RICKEY_29938374/2017NOV01_101511_EOVIST/nifti/OARcontour/{roi_name}_OAR.img")
    #     voi_numpy = voi.get_fdata().astype(np.bool_)
    #     vois[roi_name] = np.flip(voi_numpy, axis=2).transpose(2, 1, 0)

    fixed_image = torch.from_numpy(fixed_image.copy())
    moving_image = torch.from_numpy(moving_image.copy())
    dvf = torch.from_numpy(dvf.copy())
    # voi = torch.from_numpy(voi.copy())

    return (
        fixed_image,
        moving_image,
        dvf,
        vois,
        None,
        None,
        mask,
        voxel_size,
    )


def fast_trilinear_interpolation(input_array, x_indices, y_indices, z_indices):
    x_indices = (x_indices + 1) * (input_array.shape[0] - 1) * 0.5
    y_indices = (y_indices + 1) * (input_array.shape[1] - 1) * 0.5
    z_indices = (z_indices + 1) * (input_array.shape[2] - 1) * 0.5

    x0 = torch.floor(x_indices.detach()).to(torch.long)
    y0 = torch.floor(y_indices.detach()).to(torch.long)
    z0 = torch.floor(z_indices.detach()).to(torch.long)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    x0 = torch.clamp(x0, 0, input_array.shape[0] - 1)
    y0 = torch.clamp(y0, 0, input_array.shape[1] - 1)
    z0 = torch.clamp(z0, 0, input_array.shape[2] - 1)
    x1 = torch.clamp(x1, 0, input_array.shape[0] - 1)
    y1 = torch.clamp(y1, 0, input_array.shape[1] - 1)
    z1 = torch.clamp(z1, 0, input_array.shape[2] - 1)

    x = x_indices - x0
    y = y_indices - y0
    z = z_indices - z0

    output = (
        input_array[x0, y0, z0] * (1 - x) * (1 - y) * (1 - z)
        + input_array[x1, y0, z0] * x * (1 - y) * (1 - z)
        + input_array[x0, y1, z0] * (1 - x) * y * (1 - z)
        + input_array[x0, y0, z1] * (1 - x) * (1 - y) * z
        + input_array[x1, y0, z1] * x * (1 - y) * z
        + input_array[x0, y1, z1] * (1 - x) * y * z
        + input_array[x1, y1, z0] * x * y * (1 - z)
        + input_array[x1, y1, z1] * x * y * z
    )
    return output


def rotate_coordinates(theta, coordinate_tensor):
    new_coordinate_tensor = coordinate_tensor.clone()
    y_coordinate = coordinate_tensor[:, 1] * torch.cos(theta) - coordinate_tensor[:, 2] * torch.sin(theta)
    z_coordinate = coordinate_tensor[:, 1] * torch.sin(theta) + coordinate_tensor[:, 2] * torch.cos(theta)
    new_coordinate_tensor[:, 1] = y_coordinate
    new_coordinate_tensor[:, 2] = z_coordinate
    return new_coordinate_tensor


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_coordinate_slice(dims=(28, 28), dimension=0, slice_pos=0, gpu=True):
    """Make a coordinate tensor."""

    dims = list(dims)
    dims.insert(dimension, 1)

    coordinate_tensor = [torch.linspace(-1, 1, dims[i]) for i in range(3)]
    coordinate_tensor[dimension] = torch.linspace(slice_pos, slice_pos, 1)
    coordinate_tensor = torch.meshgrid(*coordinate_tensor)
    coordinate_tensor = torch.stack(coordinate_tensor, dim=3)
    coordinate_tensor = coordinate_tensor.view([np.prod(dims), 3])

    coordinate_tensor = coordinate_tensor.cuda()

    return coordinate_tensor


def make_coordinate_tensor(dims=(28, 28, 28), gpu=True):
    """Make a coordinate tensor."""

    coordinate_tensor = [torch.linspace(-1, 1, dims[i]) for i in range(3)]
    coordinate_tensor = torch.meshgrid(*coordinate_tensor)
    coordinate_tensor = torch.stack(coordinate_tensor, dim=3)
    coordinate_tensor = coordinate_tensor.view([np.prod(dims), 3])

    coordinate_tensor = coordinate_tensor.cuda()

    return coordinate_tensor


def make_masked_coordinate_tensor(mask, dims=(28, 28, 28)):
    """Make a coordinate tensor."""

    coordinate_tensor = [torch.linspace(-1, 1, dims[i]) for i in range(3)]
    coordinate_tensor = torch.meshgrid(*coordinate_tensor)
    coordinate_tensor = torch.stack(coordinate_tensor, dim=3)
    coordinate_tensor = coordinate_tensor.view([np.prod(dims), 3])
    coordinate_tensor = coordinate_tensor[mask.flatten() > 0, :]

    coordinate_tensor = coordinate_tensor.cuda()

    return coordinate_tensor