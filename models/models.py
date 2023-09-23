import torch
import torch.nn as nn
import torch.optim as optim
import os
import tqdm

from utils import general
from networks import networks
from objectives import ncc
from objectives import regularizers
import matplotlib.pyplot as plt
import numpy as np
import wandb
from skimage.metrics import structural_similarity as ssim
from torch.utils.checkpoint import checkpoint
import time
import medpy.metric


class ImplicitRegistrator:
    """This is a class for registrating implicitly represented images."""

    def __call__(
        self, coordinate_tensor=None, output_shape=(28, 28), dimension=0, slice_pos=0
    ):
        """Return the image-values for the given input-coordinates."""

        # Use standard coordinate tensor if none is given
        if coordinate_tensor is None:
            coordinate_tensor = self.make_coordinate_slice(
                output_shape, dimension, slice_pos
            )

        output = self.network(coordinate_tensor)

        # Shift coordinates by 1/n * v
        coord_temp = torch.add(output, coordinate_tensor)

        transformed_image = self.transform_no_add(coord_temp)
        return (
            transformed_image.cpu()
            .detach()
            .numpy()
            .reshape(output_shape[0], output_shape[1])
        )

    def __init__(self, moving_image, fixed_image, dvf, vois, voxel_size, **kwargs):
        """Initialize the learning model."""

        # Set all default arguments in a dict: self.args
        self.set_default_arguments()

        # Check if all kwargs keys are valid (this checks for typos)
        assert all(kwarg in self.args.keys() for kwarg in kwargs)

        # Parse important argument from kwargs
        self.epochs = kwargs["epochs"] if "epochs" in kwargs else self.args["epochs"]
        self.log_interval = (
            kwargs["log_interval"]
            if "log_interval" in kwargs
            else self.args["log_interval"]
        )
        self.gpu = kwargs["gpu"] if "gpu" in kwargs else self.args["gpu"]
        self.lr = kwargs["lr"] if "lr" in kwargs else self.args["lr"]
        self.finetune_lr = kwargs["finetune_lr"] if "finetune_lr" in kwargs else self.args["finetune_lr"]
        self.momentum = (
            kwargs["momentum"] if "momentum" in kwargs else self.args["momentum"]
        )
        self.optimizer_arg = (
            kwargs["optimizer"] if "optimizer" in kwargs else self.args["optimizer"]
        )
        self.loss_function_arg = (
            kwargs["loss_function"]
            if "loss_function" in kwargs
            else self.args["loss_function"]
        )
        self.layers = kwargs["layers"] if "layers" in kwargs else self.args["layers"]
        self.weight_init = (
            kwargs["weight_init"]
            if "weight_init" in kwargs
            else self.args["weight_init"]
        )
        self.omega = kwargs["omega"] if "omega" in kwargs else self.args["omega"]
        self.save_folder = (
            kwargs["save_folder"]
            if "save_folder" in kwargs
            else self.args["save_folder"]
        )

        # Parse other arguments from kwargs
        self.verbose = (
            kwargs["verbose"] if "verbose" in kwargs else self.args["verbose"]
        )

        # Make folder for output
        if not self.save_folder == "" and not os.path.isdir(self.save_folder):
            os.mkdir(self.save_folder)

        # Add slash to divide folder and filename
        self.save_folder += "/"

        # Make loss list to save losses
        self.loss_list = [0 for _ in range(self.epochs)]
        self.data_loss_list = [0 for _ in range(self.epochs)]

        # Set seed
        torch.manual_seed(self.args["seed"])

        # Load network
        self.network_from_file = (
            kwargs["network"] if "network" in kwargs else self.args["network"]
        )
        self.checkpoint = (
            kwargs["checkpoint"] if "checkpoint" in kwargs else self.args["checkpoint"]
        )
        self.network_type = (
            kwargs["network_type"]
            if "network_type" in kwargs
            else self.args["network_type"]
        )
        if self.network_from_file is None:
            if self.network_type == "MLP":
                self.network = networks.MLP(self.layers)
            else:
                self.network = networks.Siren(self.layers, self.weight_init, self.omega)
            if self.verbose:
                print(
                    "Network contains {} trainable parameters.".format(
                        general.count_parameters(self.network)
                    )
                )
        else:
            self.network = torch.load(self.network_from_file)
            if self.gpu:
                self.network.cuda()

        if self.checkpoint is not None:
            self.network.load_state_dict(torch.load(self.checkpoint, map_location='cpu'))

        # Choose the optimizer
        if self.optimizer_arg.lower() == "sgd":
            self.optimizer = optim.SGD(
                self.network.parameters(), lr=self.lr, momentum=self.momentum
            )

        elif self.optimizer_arg.lower() == "adam":
            if self.checkpoint is not None:
                self.optimizer = optim.Adam(self.network.parameters(), lr=self.finetune_lr)
            else:
                self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

        elif self.optimizer_arg.lower() == "adadelta":
            self.optimizer = optim.Adadelta(self.network.parameters(), lr=self.lr)

        else:
            self.optimizer = optim.SGD(
                self.network.parameters(), lr=self.lr, momentum=self.momentum
            )
            print(
                "WARNING: "
                + str(self.optimizer_arg)
                + " not recognized as optimizer, picked SGD instead"
            )

        # Choose the loss function
        if self.loss_function_arg.lower() == "mse":
            self.criterion = nn.MSELoss()

        elif self.loss_function_arg.lower() == "l1":
            self.criterion = nn.L1Loss()

        elif self.loss_function_arg.lower() == "ncc":
            self.criterion = ncc.NCC()

        elif self.loss_function_arg.lower() == "smoothl1":
            self.criterion = nn.SmoothL1Loss(beta=0.2)

        elif self.loss_function_arg.lower() == "huber":
            self.criterion = nn.HuberLoss()

        else:
            self.criterion = nn.MSELoss()
            print(
                "WARNING: "
                + str(self.loss_function_arg)
                + " not recognized as loss function, picked MSE instead"
            )

        # Move variables to GPU
        if self.gpu:
            self.network.cuda()

        # Parse arguments from kwargs
        self.mask = kwargs["mask"] if "mask" in kwargs else self.args["mask"]

        # Parse regularization kwargs
        self.jacobian_regularization = (
            kwargs["jacobian_regularization"]
            if "jacobian_regularization" in kwargs
            else self.args["jacobian_regularization"]
        )
        self.alpha_jacobian = (
            kwargs["alpha_jacobian"]
            if "alpha_jacobian" in kwargs
            else self.args["alpha_jacobian"]
        )

        self.hyper_regularization = (
            kwargs["hyper_regularization"]
            if "hyper_regularization" in kwargs
            else self.args["hyper_regularization"]
        )
        self.alpha_hyper = (
            kwargs["alpha_hyper"]
            if "alpha_hyper" in kwargs
            else self.args["alpha_hyper"]
        )

        self.bending_regularization = (
            kwargs["bending_regularization"]
            if "bending_regularization" in kwargs
            else self.args["bending_regularization"]
        )
        self.alpha_bending = (
            kwargs["alpha_bending"]
            if "alpha_bending" in kwargs
            else self.args["alpha_bending"]
        )

        # Set seed
        torch.manual_seed(self.args["seed"])

        # Parse arguments from kwargs
        self.image_shape = (
            kwargs["image_shape"]
            if "image_shape" in kwargs
            else self.args["image_shape"]
        )
        self.batch_size = (
            kwargs["batch_size"] if "batch_size" in kwargs else self.args["batch_size"]
        )

        self.finetune_batch_size = (
            kwargs["finetune_batch_size"] if "finetune_batch_size" in kwargs else self.args["finetune_batch_size"]
        )

        # Initialization
        self.moving_image = moving_image
        self.fixed_image = fixed_image
        self.dvf = dvf
        self.estimated_dvf = torch.zeros_like(dvf)
        self.fixed_state = 0
        self.voxel_size = voxel_size
        self.vois = vois

        self.possible_coordinate_tensor = general.make_masked_coordinate_tensor(
            np.ones(fixed_image.shape), self.fixed_image.shape
        )
        self.possible_masked_coordinate_tensor = general.make_masked_coordinate_tensor(
            self.mask, self.fixed_image.shape
        )

        if self.gpu:
            self.moving_image = self.moving_image.cuda()
            self.fixed_image = self.fixed_image.cuda()
            self.estimated_dvf = self.estimated_dvf.cuda()

    def cuda(self):
        """Move the model to the GPU."""

        # Standard variables
        self.network.cuda()

        # Variables specific to this class
        self.moving_image.cuda()
        self.fixed_image.cuda()

    def set_default_arguments(self):
        """Set default arguments."""

        # Inherit default arguments from standard learning model
        self.args = {}

        # Define the value of arguments
        self.args["mask"] = None
        self.args["mask_2"] = None

        self.args["method"] = 1

        self.args["lr"] = 0.00001
        self.args["finetune_lr"] = 1e-5
        self.args["batch_size"] = 10000
        self.args["finetune_batch_size"] = 10000
        self.args["layers"] = [3, 256, 256, 256, 3]
        self.args["velocity_steps"] = 1

        # Define argument defaults specific to this class
        self.args["output_regularization"] = False
        self.args["alpha_output"] = 0.2
        self.args["reg_norm_output"] = 1

        self.args["jacobian_regularization"] = False
        self.args["alpha_jacobian"] = 0.05

        self.args["hyper_regularization"] = False
        self.args["alpha_hyper"] = 0.25

        self.args["bending_regularization"] = False
        self.args["alpha_bending"] = 10.0

        self.args["image_shape"] = (200, 200)

        self.args["network"] = None
        self.args["checkpoint"] = None

        self.args["epochs"] = 2500
        self.args["log_interval"] = self.args["epochs"] // 4
        self.args["verbose"] = True
        self.args["save_folder"] = "output"

        self.args["network_type"] = "MLP"

        self.args["gpu"] = torch.cuda.is_available()
        self.args["optimizer"] = "Adam"
        self.args["loss_function"] = "ncc"
        self.args["momentum"] = 0.5

        self.args["positional_encoding"] = False
        self.args["weight_init"] = True
        self.args["omega"] = 32

        self.args["seed"] = 1

    def training_iteration(self, epoch, mode='train', n_proj=None):
        """Perform one iteration of training."""

        # Reset the gradient
        self.network.train()

        if mode == 'train':
            loss = 0
            indices = torch.randperm(
                self.possible_masked_coordinate_tensor.shape[0], device="cuda"
            )[: self.batch_size]
            coordinate_tensor = self.possible_masked_coordinate_tensor[indices, :]
            coordinate_tensor = coordinate_tensor.requires_grad_(True)

            output = self.network(coordinate_tensor)
            coord_temp = torch.add(output, coordinate_tensor)
            output = coord_temp

            transformed_image = self.transform_no_add(coord_temp)
            fixed_image = general.fast_trilinear_interpolation(
                self.fixed_image,
                coordinate_tensor[:, 0],
                coordinate_tensor[:, 1],
                coordinate_tensor[:, 2],
            )

            # Compute the loss
            loss += self.criterion(transformed_image, fixed_image)
        elif mode == 'finetune':
            nx, ny, nz = self.fixed_image.shape
            loss = 0
            coordinate_tensor = self.possible_coordinate_tensor
            coordinate_tensor = coordinate_tensor.requires_grad_(True)
            # coordinate_tensor = coordinate_tensor.reshape(self.fixed_image.shape + (3,))  # (nx, ny, nz, 3)
            # coordinate_tensor = coordinate_tensor.permute(0, 2, 1, 3).view(-1, ny, 3)  # (nx * nz, ny, 3)
            # indices = torch.randperm(coordinate_tensor.shape[0], device="cuda")[:self.batch_size]
            # coordinate_tensor = coordinate_tensor[indices, :, :].view(-1, 3)  # (n_lines * ny, 3)
            # theta = torch.tensor(torch.pi / 2).to(self.fixed_image)
            # coordinate_tensor = general.rotate_coordinates(theta, coordinate_tensor)
            output = []
            for i in range(0, coordinate_tensor.shape[0], self.finetune_batch_size):
                output.append(checkpoint(self.network,
                                         coordinate_tensor[i:i + self.finetune_batch_size],
                                         use_reentrant=False))
            output = torch.cat(output)  # (nx*ny*nz, 3)
            # indices = torch.randperm(
            #     self.possible_coordinate_tensor.shape[0], device="cuda"
            # )[: self.batch_size]
            # coordinate_tensor = self.possible_coordinate_tensor[indices, :]
            # coordinate_tensor = coordinate_tensor.requires_grad_(True)
            # output = self.network(coordinate_tensor)
            coord_temp = torch.add(output + self.estimated_dvf.reshape(-1, 3), coordinate_tensor)
            output = output + coordinate_tensor
            # torch.cuda.empty_cache()
            transformed_image = self.transform_no_add(coord_temp)   # (nx*ny*nz)
            # fixed_image = general.fast_trilinear_interpolation(
            #     self.fixed_image,
            #     coordinate_tensor[:, 0],
            #     coordinate_tensor[:, 1],
            #     coordinate_tensor[:, 2],
            # )
            transformed_image = transformed_image.reshape(self.moving_image.shape)
            # fixed_image = fixed_image.reshape(self.moving_image.shape)
            theta_list = []
            for i in range(n_proj):
                theta_list.append(torch.tensor(111.25 / 180 * torch.pi * i).to(transformed_image))
            for theta in theta_list:
                rotated_fixed_image = checkpoint(self.generate_proj, theta, self.fixed_image, use_reentrant=False)
                rotated_transformed_image = checkpoint(self.generate_proj, theta, transformed_image, use_reentrant=False)
                rotated_transformed_image = rotated_transformed_image.view(-1)
                rotated_fixed_image = rotated_fixed_image.view(-1)
                # Compute the loss
                loss += self.criterion(rotated_transformed_image, rotated_fixed_image)
            loss = loss / n_proj
        # Store the value of the data loss
        if self.verbose:
            self.data_loss_list[epoch] = loss.detach().cpu().numpy()

        # Relativation of output
        output_rel = torch.subtract(output, coordinate_tensor)

        # Regularization
        if self.jacobian_regularization:
            loss += self.alpha_jacobian * regularizers.compute_jacobian_loss(
                coordinate_tensor, output_rel, batch_size=self.batch_size
            )
        if self.hyper_regularization:
            loss += self.alpha_hyper * regularizers.compute_hyper_elastic_loss(
                coordinate_tensor, output_rel, batch_size=self.batch_size
            )
        if self.bending_regularization:
            if mode == 'finetune':
                for i in range(0, coordinate_tensor.shape[0], self.finetune_batch_size):
                    loss += self.alpha_bending * checkpoint(regularizers.compute_bending_energy,
                        coordinate_tensor[i:i + self.finetune_batch_size],
                        output_rel[i:i + self.finetune_batch_size],
                        batch_size=self.finetune_batch_size, use_reentrant=False
                    )
            elif mode == 'train':
                loss += self.alpha_bending * regularizers.compute_bending_energy(
                    coordinate_tensor, output_rel, batch_size=self.batch_size
                )
            else:
                print("Not accepting mode other than finetune or train!")

        # print(f"epoch: {epoch}, loss: {loss}")
        torch.save(self.network.state_dict(), self.save_folder + '/network.pt')
        if (((epoch + 1) % 10 == 0 or epoch + 1 == 1) and mode == 'train') or (((epoch + 1) % 10 == 0 or epoch + 1 == 1) and mode == 'finetune'):
            end_time = time.time()
            time_elapsed = end_time - self.start_time
            with torch.no_grad():
                output = []
                for i in range(0, self.possible_coordinate_tensor.shape[0], self.batch_size):
                    output.append(self.network(self.possible_coordinate_tensor[i:i + self.batch_size]))
                output = torch.cat(output)
                if mode == "finetune":
                    coord_temp = torch.add(output + self.estimated_dvf.reshape(-1, 3),
                                           self.possible_coordinate_tensor)
                else:
                    coord_temp = torch.add(output, self.possible_coordinate_tensor)
                transformed_image = self.transform_no_add(coord_temp)
                transformed_image = transformed_image.reshape(self.moving_image.shape)
                plt.imshow(np.concatenate((np.concatenate((self.fixed_image.cpu().numpy()[:, 110, :],
                           self.moving_image.cpu().numpy()[:, 110, :],
                           transformed_image.cpu().numpy()[:, 110, :])),
                           np.concatenate((abs(self.fixed_image.cpu().numpy()[:, 110, :] - self.fixed_image.cpu().numpy()[:, 110, :]),
                           abs(self.moving_image.cpu().numpy()[:, 110, :] - self.fixed_image.cpu().numpy()[:, 110, :]) * 10,
                           abs(transformed_image.cpu().numpy()[:, 110, :] - self.fixed_image.cpu().numpy()[:, 110, :]) * 10))),
                                          axis=1),
                           cmap='gray',
                           vmax=0.1)
                plt.text(0.05, 0.05, f'epoch {epoch + 1}\n{time_elapsed:.1f}s', fontsize=20)
                plt.colorbar()
                plt.axis('off')
                plt.savefig(self.save_folder + f'/epoch_{epoch + 1}.png', bbox_inches='tight')
                plt.close()

                if mode == "finetune":
                    output = output + self.estimated_dvf.reshape(-1, 3)
                output = output.reshape(self.moving_image.shape + (3,))
                output = output * 0.5 * torch.tensor(self.fixed_image.shape).to(output).reshape(1, 1, 1, 3)
                output = output * torch.tensor(self.voxel_size).to(output).reshape(1, 1, 1, 3)
                plt.figure()
                plt.imshow(np.concatenate(
                    (np.concatenate((self.dvf.cpu().numpy()[:, 110, :, 0],
                                   self.dvf.cpu().numpy()[:, 110, :, 1],
                                   self.dvf.cpu().numpy()[:, 110, :, 2])),
                    np.concatenate((output.cpu().numpy()[:, 110, :, 0],
                                output.cpu().numpy()[:, 110, :, 1],
                                output.cpu().numpy()[:, 110, :, 2]))),
                    axis=1), cmap='seismic', vmax=2, vmin=-2)
                plt.text(0.05, 0.05, f'epoch {epoch + 1}\n{time_elapsed:.1f}s', fontsize=20)
                plt.colorbar()

                plt.axis('off')
                plt.savefig(self.save_folder + f'/dvf_epoch_{epoch + 1}.png', bbox_inches='tight')
                plt.close()

                hd95 = {}
                for organ, voi in self.vois.items():
                    NeRP_transformed_voi = self.transform_no_add(coord_temp, moving_image=voi)
                    NeRP_transformed_voi = NeRP_transformed_voi.reshape(self.moving_image.shape)
                    # medpy.metric.binary.hd95(NeRP_transformed_voi.cpu().numpy(),
                    #                          self.voi.cpu().numpy(),
                    #                          voxelspacing=self.voxel_size)
                    normalized_dvf = self.dvf.to(output) / torch.tensor(self.voxel_size).to(output).reshape(1, 1, 1, 3)
                    normalized_dvf = normalized_dvf * 2 / torch.tensor(self.fixed_image.shape).to(output).reshape(1, 1, 1, 3)
                    normalized_dvf = normalized_dvf.view(-1, 3)
                    ground_truth_transformed_voi = self.transform_no_add(normalized_dvf + self.possible_coordinate_tensor,
                                                                         moving_image=voi)
                    ground_truth_transformed_voi = ground_truth_transformed_voi.reshape(self.moving_image.shape)
                    ground_truth_transformed_voi[ground_truth_transformed_voi < 0.5] = 0
                    NeRP_transformed_voi[NeRP_transformed_voi < 0.5] = 0
                    ground_truth_transformed_voi[ground_truth_transformed_voi >= 0.5] = 1
                    NeRP_transformed_voi[NeRP_transformed_voi >= 0.5] = 1
                    hd95[organ] = medpy.metric.binary.hd95(NeRP_transformed_voi.cpu().numpy(),
                                                    ground_truth_transformed_voi.cpu().numpy(),
                                                    voxelspacing=self.voxel_size, connectivity=3)

                    plt.figure()
                    plt.imshow(np.concatenate((ground_truth_transformed_voi.cpu().numpy()[40, :, :],
                                              NeRP_transformed_voi.cpu().numpy()[40, :, :],
                                              NeRP_transformed_voi.cpu().numpy()[40, :, :] - ground_truth_transformed_voi.cpu().numpy()[40, :, :])),
                               cmap='seismic', vmax=1, vmin=-1)
                    plt.text(0.05, 0.05, f'epoch {epoch + 1}\n{time_elapsed:.1f}s', fontsize=20)
                    plt.colorbar()
                    plt.axis('off')
                    plt.savefig(self.save_folder + f'/{organ}_mask_epoch_{epoch + 1}.png', bbox_inches='tight')
                    plt.close()

                mean_abs_dvf_error = np.mean(abs(self.dvf.cpu().numpy() - output.cpu().numpy())[self.mask], 0)

        # Perform the backpropagation and update the parameters accordingly
                wandb.log({"MSE": torch.nn.functional.mse_loss(self.fixed_image, transformed_image),
                           "SSIM": ssim(transformed_image.cpu().numpy(), self.fixed_image.cpu().numpy(),
                  data_range=transformed_image.cpu().numpy().max() - transformed_image.cpu().numpy().min()),
                           "SI_error (mm)": mean_abs_dvf_error[0],
                           "AP_error (mm)": mean_abs_dvf_error[1],
                           "LR_error (mm)": mean_abs_dvf_error[2],
                           "HD95 (mm)": hd95,
                           "loss": loss,
                           "epoch": epoch,
                           "Reference state": self.fixed_state})
        for param in self.network.parameters():
            param.grad = None
        loss.backward()
        self.optimizer.step()
        self.scheduler.step(loss)
        if mode == "finetune":
            if epoch == self.epochs - 1:
                with torch.no_grad():
                    output = []
                    for i in range(0, self.possible_coordinate_tensor.shape[0], self.batch_size):
                        output.append(self.network(self.possible_coordinate_tensor[i:i + self.batch_size]))
                    output = torch.cat(output)
                    output = output.reshape(self.moving_image.shape + (3,))
                # self.moving_image = transformed_image.detach()
                    self.estimated_dvf = self.estimated_dvf + output

        # Store the value of the total loss
        if self.verbose:
            self.loss_list[epoch] = loss.detach().cpu().numpy()

    def generate_proj(self, theta, image):
        """Transform moving image given a transformation."""
        rotated_possible_coordinates = general.rotate_coordinates(theta, self.possible_coordinate_tensor)
        rotated_image = general.fast_trilinear_interpolation(
            image,
            rotated_possible_coordinates[:, 0],
            rotated_possible_coordinates[:, 1],
            rotated_possible_coordinates[:, 2],
        )
        # rotated_fixed_image = rotated_fixed_image.reshape(self.moving_image.shape)
        # rotated_transformed_image = rotated_transformed_image.reshape(self.moving_image.shape)
        rotated_image = torch.sum(rotated_image.reshape(image.shape),
                                         dim=1, keepdim=True)  # (nx, 1, nz)
        return rotated_image

    def transform(
        self, transformation, coordinate_tensor=None, moving_image=None, reshape=False
    ):
        """Transform moving image given a transformation."""

        # If no specific coordinate tensor is given use the standard one of 28x28
        if coordinate_tensor is None:
            coordinate_tensor = self.coordinate_tensor

        # If no moving image is given use the standard one
        if moving_image is None:
            moving_image = self.moving_image

        # From relative to absolute
        transformation = torch.add(transformation, coordinate_tensor)
        return general.fast_trilinear_interpolation(
            moving_image,
            transformation[:, 0],
            transformation[:, 1],
            transformation[:, 2],
        )

    def transform_no_add(self, transformation, moving_image=None, reshape=False):
        """Transform moving image given a transformation."""

        # If no moving image is given use the standard one
        if moving_image is None:
            moving_image = self.moving_image
        # print('GET MOVING')
        return general.fast_trilinear_interpolation(
            moving_image,
            transformation[:, 0],
            transformation[:, 1],
            transformation[:, 2],
        )

    def fit(self, epochs=None, red_blue=False, mode='train', n_proj=None):
        """Train the network."""
        scaler = torch.cuda.amp.GradScaler()
        # Determine epochs
        if epochs is None:
            epochs = self.epochs

        # Set seed
        torch.manual_seed(self.args["seed"])

        # Extend lost_list if necessary
        if not len(self.loss_list) == epochs:
            self.loss_list = [0 for _ in range(epochs)]
            self.data_loss_list = [0 for _ in range(epochs)]

        # Perform training iterations
        self.start_time = time.time()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min',
                                                               patience=50, factor=0.5)
        for i in tqdm.tqdm(range(epochs)):
            self.training_iteration(i, mode=mode, n_proj=n_proj)


    # def projection_mse(self, moving_image, fixed_image):
    #     """Train the network."""
    #     for angle in self.projection_angles: