from utils import general
from models import models
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import wandb
import time
import argparse
import os
from shutil import copyfile

project_dir = "/RadOnc-MRI1/Student_Folder/jiarenz/projects/NeRP_motion"

parser_main = argparse.ArgumentParser()
parser_main.add_argument('--dataset', type=str, help='Choose which dataset for training')
parser_main.add_argument('--case_id', type=int, default=0, help='Choose which case for training')
args = parser_main.parse_args()

start_run_at = time.strftime("%Y%m%d_%H%M%S", time.localtime())

if args.dataset == "DIRLAB":
    data_dir = "/RadOnc-MRI1/Student_Folder/jiarenz/projects/NeRP_motion/data/DIRLAB/DIRLAB_clean"
    out_dir = "/RadOnc-MRI1/Student_Folder/jiarenz/projects/NeRP_motion/data/DIRLAB/outtest"
elif args.dataset == "liver_motion":
    data_dir = "/mnt/ibrixfs04-Kspace/motion_patients"
    out_dir = "/RadOnc-MRI1/Student_Folder/jiarenz/projects/NeRP_motion/data/liver_motion/training"
save_folder = out_dir + '_' + str(args.case_id) + f"_{start_run_at}"
os.makedirs(save_folder)
os.makedirs(f"{save_folder}/wandb")
copyfile(project_dir + '/run.py', save_folder + '/run.py')
copyfile(project_dir + '/models/models.py', save_folder + '/models.py')
copyfile(project_dir + '/utils/general.py', save_folder + '/general.py')

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="NeRP4motion",
    dir=f"{save_folder}",
    # track hyperparameters and run metadata
    config=args.__dict__,
    save_code=True
)

if args.dataset == "DIRLAB":
    (
        img_insp,
        img_exp,
        landmarks_insp,
        landmarks_exp,
        mask_exp,
        voxel_size,
    ) = general.load_image_DIRLab(args.case_id, f"{data_dir}/Case", mode='train')
elif args.dataset == "liver_motion":
    (
        img_insp,
        img_exp,
        landmarks_insp,
        landmarks_exp,
        mask_exp,
        voxel_size,
    ) = general.load_image_liver_motion(args.case_id, f"{data_dir}", mode='train')

kwargs = {}
kwargs["verbose"] = True
kwargs["hyper_regularization"] = False
kwargs["jacobian_regularization"] = False
kwargs["bending_regularization"] = True
kwargs["network_type"] = "SIREN"  # Options are "MLP" and "SIREN"
kwargs["save_folder"] = save_folder
kwargs["mask"] = mask_exp

ImpReg = models.ImplicitRegistrator(img_exp, img_insp, **kwargs)
ImpReg.fit()
# new_landmarks_orig, _ = general.compute_landmarks(
#     ImpReg.network, landmarks_insp, image_size=img_insp.shape
# )
#
# print(voxel_size)
# accuracy_mean, accuracy_std = general.compute_landmark_accuracy(
#     new_landmarks_orig, landmarks_exp, voxel_size=voxel_size
# )

# print("{} {} {}".format(case_id, accuracy_mean, accuracy_std))
