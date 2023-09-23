#!/bin/bash
#CUDA_VISIBLE_DEVICES=1 python /RadOnc-MRI1/Student_Folder/jiarenz/projects/NeRP_motion/run.py \
#--dataset liver_motion --case_id 2 --mode train&
#wait
sleep 1 && CUDA_VISIBLE_DEVICES=0 python /RadOnc-MRI1/Student_Folder/jiarenz/projects/NeRP_motion/run.py \
--dataset liver_motion --case_id 2 --mode finetune --n_proj 1 --finetune_lr 1e-7 --n_epoch_finetune 20 \
--loss mse&
sleep 50 && CUDA_VISIBLE_DEVICES=1 python /RadOnc-MRI1/Student_Folder/jiarenz/projects/NeRP_motion/run.py \
--dataset liver_motion --case_id 2 --mode finetune --n_proj 5 --finetune_lr 1e-7 --n_epoch_finetune 20 \
--loss mse&
sleep 100 && CUDA_VISIBLE_DEVICES=2 python /RadOnc-MRI1/Student_Folder/jiarenz/projects/NeRP_motion/run.py \
--dataset liver_motion --case_id 2 --mode finetune --n_proj 10 --finetune_lr 1e-7 --n_epoch_finetune 20 \
--loss mse&
wait
sleep 1 && CUDA_VISIBLE_DEVICES=0 python /RadOnc-MRI1/Student_Folder/jiarenz/projects/NeRP_motion/run.py \
--dataset liver_motion --case_id 2 --mode finetune --n_proj 20 --finetune_lr 1e-7 --n_epoch_finetune 20 \
--loss mse&
sleep 50 && CUDA_VISIBLE_DEVICES=1 python /RadOnc-MRI1/Student_Folder/jiarenz/projects/NeRP_motion/run.py \
--dataset liver_motion --case_id 2 --mode finetune --n_proj 10 --finetune_lr 1e-5 --n_epoch_finetune 20 \
--loss mse&
sleep 100 && CUDA_VISIBLE_DEVICES=2 python /RadOnc-MRI1/Student_Folder/jiarenz/projects/NeRP_motion/run.py \
--dataset liver_motion --case_id 2 --mode finetune --n_proj 10 --finetune_lr 1e-6 --n_epoch_finetune 20 \
--loss mse&
wait
sleep 1 && CUDA_VISIBLE_DEVICES=0 python /RadOnc-MRI1/Student_Folder/jiarenz/projects/NeRP_motion/run.py \
--dataset liver_motion --case_id 2 --mode finetune --n_proj 10 --finetune_lr 1e-8 --n_epoch_finetune 20 \
--loss mse&
sleep 50 && CUDA_VISIBLE_DEVICES=1 python /RadOnc-MRI1/Student_Folder/jiarenz/projects/NeRP_motion/run.py \
--dataset liver_motion --case_id 2 --mode finetune --n_proj 10 --finetune_lr 1e-7 --n_epoch_finetune 10 \
--loss mse&
sleep 100 && CUDA_VISIBLE_DEVICES=2 python /RadOnc-MRI1/Student_Folder/jiarenz/projects/NeRP_motion/run.py \
--dataset liver_motion --case_id 2 --mode finetune --n_proj 10 --finetune_lr 1e-7 --n_epoch_finetune 50 \
--loss mse&
wait
sleep 1 && CUDA_VISIBLE_DEVICES=0 python /RadOnc-MRI1/Student_Folder/jiarenz/projects/NeRP_motion/run.py \
--dataset liver_motion --case_id 2 --mode finetune --n_proj 10 --finetune_lr 1e-7 --n_epoch_finetune 100 \
--loss mse&
sleep 50 && CUDA_VISIBLE_DEVICES=1 python /RadOnc-MRI1/Student_Folder/jiarenz/projects/NeRP_motion/run.py \
--dataset liver_motion --case_id 2 --mode finetune --n_proj 10 --finetune_lr 1e-7 --n_epoch_finetune 200 \
--loss mse&
#wait
#CUDA_VISIBLE_DEVICES=2 python /RadOnc-MRI1/Student_Folder/jiarenz/projects/NeRP_motion/run.py \
#--dataset liver_motion --case_id 2 --mode finetune --n_proj 5 --finetune_lr 1e-7&
#wait
#CUDA_VISIBLE_DEVICES=2 python /RadOnc-MRI1/Student_Folder/jiarenz/projects/NeRP_motion/run.py \
#--dataset liver_motion --case_id 2 --mode finetune --n_proj 10 --finetune_lr 1e-7&
#wait
#CUDA_VISIBLE_DEVICES=2 python /RadOnc-MRI1/Student_Folder/jiarenz/projects/NeRP_motion/run.py \
#--dataset liver_motion --case_id 2 --mode finetune --n_proj 20 --finetune_lr 1e-5&
#wait
#CUDA_VISIBLE_DEVICES=2 python /RadOnc-MRI1/Student_Folder/jiarenz/projects/NeRP_motion/run.py \
#--dataset liver_motion --case_id 2 --mode finetune --n_proj 30 --finetune_lr 1e-6&

#CUDA_VISIBLE_DEVICES=1 python /RadOnc-MRI1/Student_Folder/jiarenz/projects/NeRP_motion/run.py \
#--dataset DIRLAB --case_id 1 --mode train&

#CUDA_VISIBLE_DEVICES=0 python /RadOnc-MRI1/Student_Folder/jiarenz/projects/NeRP_motion/run.py \
#--dataset liver_motion --case_id 2 --mode finetune --n_epoch_finetune 20 --finetune_lr 1e-7 --n_proj 10