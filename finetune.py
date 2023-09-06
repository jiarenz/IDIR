from utils import general
from models import models
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import wandb
import time

wandb.init(
    # set the wandb project where this run will be logged
    project="NeRP4motion",

    # track hyperparameters and run metadata
    config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
    }
)
data_dir = "/RadOnc-MRI1/Student_Folder/jiarenz/projects/NeRP_motion/data/DIRLAB/DIRLAB_clean"
out_dir = "/RadOnc-MRI1/Student_Folder/jiarenz/projects/NeRP_motion/data/DIRLAB/finetuning_output"

case_id = 1

(
    img_insp,
    img_exp,
    landmarks_insp,
    landmarks_exp,
    mask_exp,
    voxel_size,
) = general.load_image_DIRLab(case_id, f"{data_dir}/Case", mode='finetune')

start_run_at = time.strftime("%Y%m%d_%H%M%S", time.localtime())
kwargs = {}
kwargs["verbose"] = True
kwargs["hyper_regularization"] = False
kwargs["jacobian_regularization"] = False
kwargs["bending_regularization"] = False
kwargs["network_type"] = "SIREN"  # Options are "MLP" and "SIREN"
kwargs["save_folder"] = out_dir + str(case_id) + f"_{start_run_at}"
kwargs["mask"] = mask_exp
kwargs["checkpoint"] = "/RadOnc-MRI1/Student_Folder/jiarenz/projects/NeRP_motion/data/DIRLAB/outtest1/network.pt"
kwargs["loss_function"] = 'mse'

ImpReg = models.ImplicitRegistrator(img_exp, img_insp, **kwargs)
ImpReg.fit(mode='finetune')
new_landmarks_orig, _ = general.compute_landmarks(
    ImpReg.network, landmarks_insp, image_size=img_insp.shape
)

print(voxel_size)
accuracy_mean, accuracy_std = general.compute_landmark_accuracy(
    new_landmarks_orig, landmarks_exp, voxel_size=voxel_size
)

print("{} {} {}".format(case_id, accuracy_mean, accuracy_std))
