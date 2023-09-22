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
import torch.optim as optim


project_dir = "/RadOnc-MRI1/Student_Folder/jiarenz/projects/NeRP_motion"

parser_main = argparse.ArgumentParser()
parser_main.add_argument('--dataset', type=str, help='Choose which dataset for training')
parser_main.add_argument('--case_id', type=int, default=0, help='Choose which case for training')
parser_main.add_argument('--mode', type=str, default="train", help='Whether to train from scratch or finetune '
                                                                   'an existing model')
parser_main.add_argument('--n_proj', type=int, default=1, help='Number of projecitions for finetuning')
parser_main.add_argument('--finetune_lr', type=float, default=1e-5, help='Learning rate for finetuning')
parser_main.add_argument('--n_epoch_train', type=int, default=2500, help='number of training epochs')
parser_main.add_argument('--n_epoch_finetune', type=int, default=200, help='number of finetuning epochs')
args = parser_main.parse_args()

start_run_at = time.strftime("%Y%m%d_%H%M%S", time.localtime())

if args.dataset == "DIRLAB":
    data_dir = "/RadOnc-MRI1/Student_Folder/jiarenz/projects/NeRP_motion/data/DIRLAB/DIRLAB_clean"
    if args.mode == "train":
        out_dir = "/RadOnc-MRI1/Student_Folder/jiarenz/projects/NeRP_motion/data/DIRLAB/outtest"
    elif args.mode == "finetune":
        out_dir = "/RadOnc-MRI1/Student_Folder/jiarenz/projects/NeRP_motion/data/DIRLAB/finetuning_output"
elif args.dataset == "liver_motion":
    data_dir = "/mnt/ibrixfs04-Kspace/motion_patients"
    if args.mode == "train":
        out_dir = "/RadOnc-MRI1/Student_Folder/jiarenz/projects/NeRP_motion/data/liver_motion/training"
    elif args.mode == "finetune":
        out_dir = "/RadOnc-MRI1/Student_Folder/jiarenz/projects/NeRP_motion/data/liver_motion/finetuning"
save_folder = out_dir + '_' + str(args.case_id) + f"_{start_run_at}"
os.makedirs(save_folder)
os.makedirs(f"{save_folder}/wandb")
copyfile(project_dir + '/run.py', save_folder + '/run.py')
copyfile(project_dir + '/models/models.py', save_folder + '/models.py')
copyfile(project_dir + '/utils/general.py', save_folder + '/general.py')

# start a new wandb run to track this script
if args.mode == "train":
    run_name = f"patient {args.case_id}-{args.mode}-{start_run_at}"
else:
    run_name = f"patient {args.case_id}-{args.mode}-{args.n_proj} proj-{start_run_at}"
wandb.init(
    # set the wandb project where this run will be logged
    project="NeRP4motion",
    dir=f"{save_folder}",
    # track hyperparameters and run metadata
    config=args.__dict__,
    save_code=True,
    name=run_name
)

if args.dataset == "DIRLAB":
    (
        img_insp,
        img_exp,
        dvf,
        voi,
        landmarks_insp,
        landmarks_exp,
        mask_exp,
        voxel_size,
    ) = general.load_image_DIRLab(args.case_id, f"{data_dir}/Case", mode=args.mode)
elif args.dataset == "liver_motion":
    if args.mode == "train":
        (
            img_insp,
            img_exp,
            dvf,
            vois,
            landmarks_insp,
            landmarks_exp,
            mask_exp,
            voxel_size,
        ) = general.load_image_liver_motion(args.case_id, f"{data_dir}", mode=args.mode)
    else:
        image_series_data, fixed_states = general.load_image_series_liver_motion(args.case_id, f"{data_dir}", mode=args.mode)
        mask_exp = image_series_data[0][-2]
kwargs = {}
kwargs["verbose"] = True
kwargs["hyper_regularization"] = False
kwargs["jacobian_regularization"] = False
kwargs["bending_regularization"] = True
kwargs["alpha_bending"] = 100
kwargs["network_type"] = "SIREN"  # Options are "MLP" and "SIREN"
kwargs["save_folder"] = save_folder
kwargs["mask"] = mask_exp
kwargs["batch_size"] = 10000
# kwargs["voi"] = voi
kwargs["loss_function"] = 'ncc'
# checkpoints = ["training_0_20230907_231958", "training_1_20230908_000700", "training_2_20230910_232109"]
checkpoints = ["training_0_20230907_231958", "training_1_20230908_000700", "training_2_20230921_142043"]
if args.mode == "finetune":
    kwargs["checkpoint"] = f"/RadOnc-MRI1/Student_Folder/jiarenz/projects/NeRP_motion/data/liver_motion/{checkpoints[args.case_id]}/network.pt"
    # kwargs["loss_function"] = 'mse'
    kwargs["loss_function"] = 'ncc'
    kwargs["hyper_regularization"] = False
    kwargs["jacobian_regularization"] = False
    kwargs["bending_regularization"] = False
    # kwargs["alpha_bending"] = 10
    kwargs["epochs"] = args.n_epoch_finetune
    kwargs["finetune_lr"] = args.finetune_lr


if args.mode == "finetune":
    ImpReg = models.ImplicitRegistrator(image_series_data[0][1],
                                        image_series_data[0][0],
                                        image_series_data[0][2],
                                        image_series_data[0][3],
                                        image_series_data[0][-1], **kwargs)
    for i in range(len(image_series_data)):
        # ImpReg.moving_image = image_series_data[i][0].cuda()
        ImpReg.fixed_image = image_series_data[i][0].cuda()
        ImpReg.dvf = image_series_data[i][2]
        ImpReg.vois = image_series_data[i][3]
        ImpReg.voxel_size = image_series_data[i][-1]
        ImpReg.mask = image_series_data[i][-2]
        ImpReg.fixed_state = fixed_states[i]
        # ImpReg.optimizer = optim.Adam(ImpReg.network.parameters(), lr=args.finetune_lr)
        ImpReg.fit(mode='finetune', n_proj=args.n_proj)
else:
    ImpReg = models.ImplicitRegistrator(img_exp, img_insp, dvf, vois, voxel_size, **kwargs)
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