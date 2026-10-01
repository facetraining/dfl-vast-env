import multiprocessing
import operator

import numpy as np

import os
import shutil

# from psutil import cpu_count

from core.interact import interact as io
from core.leras import nn
from facelib import FaceType
from models import ModelBase
from samplelib import *

from pathlib import Path

from utils.label_face import label_face_filename

from utils.train_status_export import data_format_change, prepare_sample
import cv2
from core.cv2ex import cv2_imwrite
from tqdm import tqdm


class MEModel(ModelBase):
    # Override the on_initialize_options method of the parent class
    def on_initialize_options(self):
        # Get the current device configuration
        device_config = nn.getCurrentDeviceConfig()

        # Recommended batch size based on device's VRAM capacity
        lowest_vram = 4
        if len(device_config.devices) != 0:
            lowest_vram = device_config.devices.get_worst_device().total_mem_gb
        if lowest_vram >= 13:
            suggest_batch_size = 8
        else:
            suggest_batch_size = 4

        # Define minimum and maximum resolution
        min_res = 64
        max_res = 640
        
        self.batch_size = suggest_batch_size
        
        default_archi = self.options["archi"] = self.load_or_def_option(
            "archi", "df-ud"
        )
        default_models_opt_on_gpu = self.options["models_opt_on_gpu"] = (
            self.load_or_def_option("models_opt_on_gpu", True)
        )

        default_uniform_yaw = self.options["uniform_yaw"] = self.load_or_def_option(
            "uniform_yaw", False
        )

        default_adabelief = self.options["adabelief"] = True

        lr_dropout = self.load_or_def_option("lr_dropout", "n")
        lr_dropout = {True: "y", False: "n"}.get(
            lr_dropout, lr_dropout
        )  # backward comp
        default_lr_dropout = self.options["lr_dropout"] = lr_dropout

        default_loss_function = self.options["loss_function"] = "SSIM"
        
        default_random_warp = self.options["random_warp"] = self.load_or_def_option(
            "random_warp", True
        )
        default_random_hsv_power = self.options["random_hsv_power"] = 0
        
        default_random_downsample = self.options["random_downsample"] = False
        default_random_noise = self.options["random_noise"] = False
        default_random_blur = self.options["random_blur"] = False
        default_random_jpeg = self.options["random_jpeg"] = False
        default_super_warp = self.options["super_warp"] = False
        
        default_rotation_range = self.rotation_range = [-3, 3]
        default_scale_range = self.scale_range = [-0.15, 0.15]

        # Load or define other training-related default options

        default_true_face_power = self.options["true_face_power"] = 0.0

        default_ct_mode = self.options["ct_mode"] = self.load_or_def_option(
            "ct_mode", "none"
        )
        default_random_color = self.options["random_color"] = False
        
        default_clipgrad = self.options["clipgrad"] = True
        
        default_pretrain = self.options["pretrain"] = False

        default_cpu_cap = self.options["cpu_cap"] = suggest_batch_size
        
        default_preview_samples = self.options["preview_samples"] = 2

        default_full_preview = self.options["force_full_preview"] = False

        default_lr = self.options["lr"] = self.load_or_def_option("lr", 4e-5)

        default_quick_opt = self.options["quick_opt"] = self.load_or_def_option("quick_opt", True)
        
        # Determine if model settings need to be overridden
        ask_override = self.ask_override()
        self.quick_opt=default_quick_opt
        if ask_override:
            # If this is the first time it is run or the settings need to be overridden, the user is asked to enter various configurations
            self.ask_random_src_flip()
            self.ask_quick_opt()
        if self.is_first_run():
            self.random_src_flip = True
            self.quick_opt = True
            
    # Override the on_initialize method of the parent class
    def on_initialize(self):
        # Get the current device configuration and initialization data format
        device_config = nn.getCurrentDeviceConfig()
        devices = device_config.devices
        self.model_data_format = (
            "NCHW" if len(devices) != 0 and not self.is_debug() else "NHWC"
        )
        nn.initialize(data_format=self.model_data_format)
        tf = nn.tf  # TensorFlow references

        # Setting resolution and face type
        resolution = self.resolution = 512  # Set resolution to 512
        self.face_type = FaceType.WHOLE_FACE  # Set face type to whole face
        ae_dims = 224  # Set the ae dimension to 224
        e_dims = 64  # Set the e dimension to 64
        d_dims = 64  # Set the d dimension to 64
        d_mask_dims = 22  # Set the d_mask dimension to 16
        if self.quick_opt:
            io.log_info("Currently training eyes and mouth")
        else:
            io.log_info("Current Training Skin")
        # Setting Eye and Mouth Priorities
        eyes_prio = True if self.quick_opt else False
        mouth_prio = True if self.quick_opt else False

        # Parsing Architecture Types
        archi_split = self.options["archi"].split("-")
        if len(archi_split) == 2:
            archi_type, archi_opts = archi_split
        elif len(archi_split) == 1:
            archi_type, archi_opts = archi_split[0], None
        self.archi_type = archi_type        
        model_archi = nn.DeepFakeArchi(resolution, use_fp16=False, opts=archi_opts)  # Creating a Model Architecture

        # Setting whether to pre-train or not
        self.pretrain = False  # False for whether to pre-train
        self.pretrain_just_disabled = False  # False if pre-training has just been disabled.
        masked_training = True

        # Set whether or not to use the AdaBelief optimizer
        adabelief = self.options["adabelief"]

        # Set whether to use half-precision floating point numbers
        if self.is_exporting:
            use_fp16 = io.input_bool(
                "Export quantized?",
                False,
                help_message="Makes exported models faster. If you encounter problems, disable this option.",
            )

        # Setting the relevant parameters (all locks of the pre-training have been unlocked, except GAN)
        self.gan_power = gan_power = 0.0 if self.quick_opt else 0.02
        random_warp = True if self.quick_opt else False
        random_src_flip = self.random_src_flip
        random_dst_flip = True
        random_hsv_power = 0.05
        blur_out_mask = False
        
        if np.random.randint(3) > 0:
            ct_mode = "fs-aug"
        else:
            ct_mode = "cc-aug"


        # Setting Model Optimization Options
        models_opt_on_gpu = (
            False if len(devices) == 0 else self.options["models_opt_on_gpu"]
        )
        models_opt_device = (
            nn.tf_default_device_name
            if models_opt_on_gpu and self.is_training
            else "/CPU:0"
        )
        optimizer_vars_on_cpu = models_opt_device == "/CPU:0"

        # Setting input channels and shapes
        input_ch = 3
        bgr_shape = self.bgr_shape = nn.get4Dshape(resolution, resolution, input_ch)
        mask_shape = nn.get4Dshape(resolution, resolution, 1)
        self.model_filename_list = []

        with tf.device("/CPU:0"):
            # Initialize placeholders on the CPU
            self.warped_src = tf.placeholder(nn.floatx, bgr_shape, name="warped_src")
            self.warped_dst = tf.placeholder(nn.floatx, bgr_shape, name="warped_dst")

            self.target_src = tf.placeholder(nn.floatx, bgr_shape, name="target_src")
            self.target_dst = tf.placeholder(nn.floatx, bgr_shape, name="target_dst")

            self.target_srcm = tf.placeholder(nn.floatx, mask_shape, name="target_srcm")
            self.target_srcm_em = tf.placeholder(
                nn.floatx, mask_shape, name="target_srcm_em"
            )
            self.target_dstm = tf.placeholder(nn.floatx, mask_shape, name="target_dstm")
            self.target_dstm_em = tf.placeholder(
                nn.floatx, mask_shape, name="target_dstm_em"
            )

        # Initializing the Model Architecture
        model_archi = nn.DeepFakeArchi(resolution, use_fp16=False, opts="ud")

        # Continue initializing the other components of the model
        with tf.device(models_opt_device):
            # Initialize different parts of the model according to the type of architecture
            if "df" in archi_type:
                # DF Architecture
                self.encoder = model_archi.Encoder(
                    in_ch=input_ch, e_ch=e_dims, name="encoder"
                )
                encoder_out_ch = (
                    self.encoder.get_out_ch()
                    * self.encoder.get_out_res(resolution) ** 2
                )

                self.inter = model_archi.Inter(
                    in_ch=encoder_out_ch, ae_ch=ae_dims, ae_out_ch=ae_dims, name="inter"
                )
                inter_out_ch = self.inter.get_out_ch()

                self.decoder_src = model_archi.Decoder(
                    in_ch=inter_out_ch,
                    d_ch=d_dims,
                    d_mask_ch=d_mask_dims,
                    name="decoder_src",
                )
                self.decoder_dst = model_archi.Decoder(
                    in_ch=inter_out_ch,
                    d_ch=d_dims,
                    d_mask_ch=d_mask_dims,
                    name="decoder_dst",
                )

                self.model_filename_list += [
                    [self.encoder, "encoder.npy"],
                    [self.inter, "inter.npy"],
                    [self.decoder_src, "decoder_src.npy"],
                    [self.decoder_dst, "decoder_dst.npy"],
                ]

                # Initialize code discriminator if training is in progress
                if self.is_training:
                    if self.options["true_face_power"] != 0:
                        self.code_discriminator = nn.CodeDiscriminator(
                            ae_dims, code_res=self.inter.get_out_res(), name="dis"
                        )
                        self.model_filename_list += [
                            [self.code_discriminator, "code_discriminator.npy"]
                        ]

            elif "liae" in archi_type:
                # LIAE Architecture
                self.encoder = model_archi.Encoder(
                    in_ch=input_ch, e_ch=e_dims, name="encoder"
                )
                encoder_out_ch = (
                    self.encoder.get_out_ch()
                    * self.encoder.get_out_res(resolution) ** 2
                )

                self.inter_AB = model_archi.Inter(
                    in_ch=encoder_out_ch,
                    ae_ch=ae_dims,
                    ae_out_ch=ae_dims * 2,
                    name="inter_AB",
                )
                self.inter_B = model_archi.Inter(
                    in_ch=encoder_out_ch,
                    ae_ch=ae_dims,
                    ae_out_ch=ae_dims * 2,
                    name="inter_B",
                )

                inter_out_ch = self.inter_AB.get_out_ch()
                inters_out_ch = inter_out_ch * 2
                self.decoder = model_archi.Decoder(
                    in_ch=inters_out_ch,
                    d_ch=d_dims,
                    d_mask_ch=d_mask_dims,
                    name="decoder",
                )

                self.model_filename_list += [
                    [self.encoder, "encoder.npy"],
                    [self.inter_AB, "inter_AB.npy"],
                    [self.inter_B, "inter_B.npy"],
                    [self.decoder, "decoder.npy"],
                ]

            if self.is_training:
                if gan_power != 0:
                    self.D_src = nn.UNetPatchDiscriminator(
                        patch_size=64,
                        in_ch=input_ch,
                        base_ch=16,
                        use_fp16=False,
                        name="D_src",
                    )
                    self.model_filename_list += [[self.D_src, "GAN.npy"]]

                # Initializing the Optimizer
                lr = self.options["lr"]

                if self.options["lr_dropout"] in ["y", "cpu"] and not self.pretrain:
                    lr_cos = 500
                    lr_dropout = 0.3
                else:
                    lr_cos = 0
                    lr_dropout = 1.0
                OptimizerClass = nn.AdaBelief
                clipnorm = 1.0

                # Setting the optimizer according to the architecture type
                if "df" in archi_type:
                    self.src_dst_saveable_weights = (
                        self.encoder.get_weights()
                        + self.inter.get_weights()
                        + self.decoder_src.get_weights()
                        + self.decoder_dst.get_weights()
                    )
                    self.src_dst_trainable_weights = self.src_dst_saveable_weights
                elif "liae" in archi_type:
                    self.src_dst_saveable_weights = (
                        self.encoder.get_weights()
                        + self.inter_AB.get_weights()
                        + self.inter_B.get_weights()
                        + self.decoder.get_weights()
                    )
                    if random_warp:
                        self.src_dst_trainable_weights = self.src_dst_saveable_weights
                    else:
                        self.src_dst_trainable_weights = (
                            self.encoder.get_weights()
                            + self.inter_B.get_weights()
                            + self.decoder.get_weights()
                        )

                # Optimizer for initializing sources and targets
                self.src_dst_opt = OptimizerClass(
                    lr=lr,
                    lr_dropout=lr_dropout,
                    lr_cos=lr_cos,
                    clipnorm=clipnorm,
                    name="src_dst_opt",
                )
                self.src_dst_opt.initialize_variables(
                    self.src_dst_saveable_weights,
                    vars_on_cpu=optimizer_vars_on_cpu,
                    lr_dropout_on_cpu=self.options["lr_dropout"] == "cpu",
                )
                self.model_filename_list += [(self.src_dst_opt, "src_dst_opt.npy")]

                # If using a GAN, initialize the GAN discriminator optimizer
                if gan_power != 0:
                    self.D_src_dst_opt = OptimizerClass(
                        lr=lr,
                        lr_dropout=lr_dropout,
                        lr_cos=lr_cos,
                        clipnorm=clipnorm,
                        name="GAN_opt",
                    )
                    self.D_src_dst_opt.initialize_variables(
                        self.D_src.get_weights(),
                        vars_on_cpu=optimizer_vars_on_cpu,
                        lr_dropout_on_cpu=self.options["lr_dropout"] == "cpu",
                    )  # +self.D_src_x2.get_weights()
                    self.model_filename_list += [(self.D_src_dst_opt, "GAN_opt.npy")]

        if self.is_training:
            # Resizing batches in multi-GPU environments
            gpu_count = max(1, len(devices))  # Get the number of GPUs, at least 1
            bs_per_gpu = max(
                1, self.get_batch_size() // gpu_count
            )  # Batch size per GPU, at least 1
            self.set_batch_size(gpu_count * bs_per_gpu)  # Setting the total batch size

            # Calculate the loss per GPU
            gpu_pred_src_src_list = []  # GPU prediction source-to-source list
            gpu_pred_dst_dst_list = []  # GPU predicts target-to-target list
            gpu_pred_src_dst_list = []  # GPU predicts source-to-target list
            gpu_pred_src_srcm_list = []  # List of GPU prediction source-to-source masks
            gpu_pred_dst_dstm_list = []  # List of GPU predicted target-to-target masks
            gpu_pred_src_dstm_list = []  # List of GPU prediction source-to-target masks

            gpu_src_losses = []  # GPU Source Loss List
            gpu_dst_losses = []  # GPU target loss list
            gpu_G_loss_gvs = []  # List of GPU generator loss gradients
            gpu_D_code_loss_gvs = []  # List of GPU discriminator coding loss gradients
            gpu_D_src_dst_loss_gvs = []  # List of GPU discriminator source-to-target loss gradients

            for gpu_id in range(gpu_count):
                with tf.device(
                    f"/{devices[gpu_id].tf_dev_type}:{gpu_id}"
                    if len(devices) != 0
                    else f"/CPU:0"
                ):
                    with tf.device(f"/CPU:0"):
                        # Slicing operations are performed on the CPU to avoid all batch data being transferred to the GPU first
                        batch_slice = slice(
                            gpu_id * bs_per_gpu, (gpu_id + 1) * bs_per_gpu
                        )
                        gpu_warped_src = self.warped_src[
                            batch_slice, :, :, :
                        ]  # Deformation source image after slicing
                        gpu_warped_dst = self.warped_dst[
                            batch_slice, :, :, :
                        ]  # Deformed target image after slicing
                        gpu_target_src = self.target_src[
                            batch_slice, :, :, :
                        ]  # Target source image after slicing
                        gpu_target_dst = self.target_dst[
                            batch_slice, :, :, :
                        ]  # Sliced Target Target Image
                        gpu_target_srcm_all = self.target_srcm[
                            batch_slice, :, :, :
                        ]  # Target source mask after slicing
                        gpu_target_srcm_em = self.target_srcm_em[
                            batch_slice, :, :, :
                        ]  # Target source emergency mask after slicing
                        gpu_target_dstm_all = self.target_dstm[
                            batch_slice, :, :, :
                        ]  # Target destination mask after slicing
                        gpu_target_dstm_em = self.target_dstm_em[
                            batch_slice, :, :, :
                        ]  # Sliced Target Target Emergency Mask

                    gpu_target_srcm_anti = 1 - gpu_target_srcm_all  # source reverse mask
                    gpu_target_dstm_anti = 1 - gpu_target_dstm_all  # Target reverse mask


                    # Processing the model tensor
                    if "df" in archi_type:
                        # Using the 'df' architecture type
                        gpu_src_code = self.inter(self.encoder(gpu_warped_src))
                        gpu_dst_code = self.inter(self.encoder(gpu_warped_dst))
                        gpu_pred_src_src, gpu_pred_src_srcm = self.decoder_src(
                            gpu_src_code
                        )
                        gpu_pred_dst_dst, gpu_pred_dst_dstm = self.decoder_dst(
                            gpu_dst_code
                        )
                        gpu_pred_src_dst, gpu_pred_src_dstm = self.decoder_src(
                            gpu_dst_code
                        )
                        gpu_pred_src_dst_no_code_grad, _ = self.decoder_src(
                            tf.stop_gradient(gpu_dst_code)
                        )

                    elif "liae" in archi_type:
                        # Using the 'liae' architecture type
                        gpu_src_code = self.encoder(gpu_warped_src)
                        gpu_src_inter_AB_code = self.inter_AB(gpu_src_code)
                        gpu_src_code = tf.concat(
                            [gpu_src_inter_AB_code, gpu_src_inter_AB_code],
                            nn.conv2d_ch_axis,
                        )
                        gpu_dst_code = self.encoder(gpu_warped_dst)
                        gpu_dst_inter_B_code = self.inter_B(gpu_dst_code)
                        gpu_dst_inter_AB_code = self.inter_AB(gpu_dst_code)
                        gpu_dst_code = tf.concat(
                            [gpu_dst_inter_B_code, gpu_dst_inter_AB_code],
                            nn.conv2d_ch_axis,
                        )
                        gpu_src_dst_code = tf.concat(
                            [gpu_dst_inter_AB_code, gpu_dst_inter_AB_code],
                            nn.conv2d_ch_axis,
                        )

                        gpu_pred_src_src, gpu_pred_src_srcm = self.decoder(gpu_src_code)
                        gpu_pred_dst_dst, gpu_pred_dst_dstm = self.decoder(gpu_dst_code)
                        gpu_pred_src_dst, gpu_pred_src_dstm = self.decoder(
                            gpu_src_dst_code
                        )
                        gpu_pred_src_dst_no_code_grad, _ = self.decoder(
                            tf.stop_gradient(gpu_src_dst_code)
                        )

                    gpu_pred_src_src_list.append(
                        gpu_pred_src_src
                    )  # Add source-to-source results of GPU predictions to the list
                    gpu_pred_dst_dst_list.append(
                        gpu_pred_dst_dst
                    )  # Add GPU-predicted target-to-target results to the list
                    gpu_pred_src_dst_list.append(
                        gpu_pred_src_dst
                    )  # Add source-to-target results of GPU predictions to the list

                    gpu_pred_src_srcm_list.append(
                        gpu_pred_src_srcm
                    )  # Add source-to-source masks for GPU predictions to the list
                    gpu_pred_dst_dstm_list.append(
                        gpu_pred_dst_dstm
                    )  # Add GPU-predicted target-to-target masks to the list
                    gpu_pred_src_dstm_list.append(
                        gpu_pred_src_dstm
                    )  # Add source-to-target masks for GPU predictions to the list

                    # Unpacking individual masks from a combined mask
                    gpu_target_srcm = tf.clip_by_value(
                        gpu_target_srcm_all, 0, 1
                    )  # GPU Source Mask
                    gpu_target_dstm = tf.clip_by_value(
                        gpu_target_dstm_all, 0, 1
                    )  # GPU target mask
                    gpu_target_srcm_eye_mouth = tf.clip_by_value(
                        gpu_target_srcm_em - 1, 0, 1
                    )  # GPU Source Eye Mouth Mask
                    gpu_target_dstm_eye_mouth = tf.clip_by_value(
                        gpu_target_dstm_em - 1, 0, 1
                    )  # GPU Target Eye Mouth Mask
                    gpu_target_srcm_mouth = tf.clip_by_value(
                        gpu_target_srcm_em - 2, 0, 1
                    )  # GPU Source Mouth Mask
                    gpu_target_dstm_mouth = tf.clip_by_value(
                        gpu_target_dstm_em - 2, 0, 1
                    )  # GPU Target Mouth Mask
                    gpu_target_srcm_eyes = tf.clip_by_value(
                        gpu_target_srcm_eye_mouth - gpu_target_srcm_mouth, 0, 1
                    )  # GPU Source Eye Mask
                    gpu_target_dstm_eyes = tf.clip_by_value(
                        gpu_target_dstm_eye_mouth - gpu_target_dstm_mouth, 0, 1
                    )  # GPU Target Eye Mask

                    gpu_target_srcm_blur = nn.gaussian_blur(
                        gpu_target_srcm, max(1, resolution // 32)
                    )  # Fuzzy Processing GPU Source Mask
                    gpu_target_srcm_blur = (
                        tf.clip_by_value(gpu_target_srcm_blur, 0, 0.5) * 2
                    )
                    gpu_target_srcm_anti_blur = (
                        1.0 - gpu_target_srcm_blur
                    )  # Reverse Fuzzy Processing GPU Source Masks

                    gpu_target_dstm_blur = nn.gaussian_blur(
                        gpu_target_dstm, max(1, resolution // 32)
                    )  # Fuzzy Processing GPU Target Mask
                    gpu_target_dstm_blur = (
                        tf.clip_by_value(gpu_target_dstm_blur, 0, 0.5) * 2
                    )

                    gpu_style_mask_blur = nn.gaussian_blur(
                        gpu_pred_src_dstm * gpu_pred_dst_dstm, max(1, resolution // 32)
                    )  # Fuzzy Processing Style Mask
                    gpu_style_mask_blur = tf.stop_gradient(
                        tf.clip_by_value(gpu_target_srcm_blur, 0, 1.0)
                    )
                    gpu_style_mask_anti_blur = (
                        1.0 - gpu_style_mask_blur
                    )  # Reverse Blur Processing Style Mask

                    gpu_target_dst_masked = (
                        gpu_target_dst * gpu_target_dstm_blur
                    )  # GPU Target Masks for Applying Fuzzy Processing

                    gpu_target_src_anti_masked = (
                        gpu_target_src * gpu_target_srcm_anti_blur
                    )  # GPU source image with inverse blurring applied
                    gpu_pred_src_src_anti_masked = (
                        gpu_pred_src_src * gpu_target_srcm_anti_blur
                    )  # GPU Source Prediction Images Applying Inverse Fuzzy Processing

                    gpu_target_src_masked_opt = (
                        gpu_target_src * gpu_target_srcm_blur
                        if masked_training
                        else gpu_target_src
                    )  # Selection of GPU source images based on whether or not they are masked for training
                    gpu_target_dst_masked_opt = (
                        gpu_target_dst_masked if masked_training else gpu_target_dst
                    )  # Selection of GPU target images based on whether or not they are masked for training
                    gpu_pred_src_src_masked_opt = (
                        gpu_pred_src_src * gpu_target_srcm_blur
                        if masked_training
                        else gpu_pred_src_src
                    )  # Selection of GPU source prediction images based on whether or not they are mask trained
                    gpu_pred_dst_dst_masked_opt = (
                        gpu_pred_dst_dst * gpu_target_dstm_blur
                        if masked_training
                        else gpu_pred_dst_dst
                    )  # Selection of GPU target prediction images based on whether or not they are mask trained



                    gpu_src_loss = tf.reduce_mean(
                        5
                        * nn.dssim(
                            gpu_target_src_masked_opt,
                            gpu_pred_src_src_masked_opt,
                            max_val=1.0,
                            filter_size=int(resolution / 11.6),
                        ),
                        axis=[1],
                    )
                    gpu_src_loss += tf.reduce_mean(
                        5
                        * nn.dssim(
                            gpu_target_src_masked_opt,
                            gpu_pred_src_src_masked_opt,
                            max_val=1.0,
                            filter_size=int(resolution / 23.2),
                        ),
                        axis=[1],
                    )
                    gpu_src_loss += tf.reduce_mean(
                        10
                        * tf.square(
                            gpu_target_src_masked_opt - gpu_pred_src_src_masked_opt
                        ),
                        axis=[1, 2, 3],
                    )

                    if eyes_prio or mouth_prio:
                        # If eye or mouth priority is set
                        if eyes_prio and mouth_prio:
                            gpu_target_part_mask = gpu_target_srcm_eye_mouth
                        elif eyes_prio:
                            gpu_target_part_mask = gpu_target_srcm_eyes
                        elif mouth_prio:
                            gpu_target_part_mask = gpu_target_srcm_mouth

                        gpu_src_loss += tf.reduce_mean(
                            300
                            * tf.abs(
                                gpu_target_src * gpu_target_part_mask
                                - gpu_pred_src_src * gpu_target_part_mask
                            ),
                            axis=[1, 2, 3],
                        )

                    gpu_src_loss += tf.reduce_mean(
                        10 * tf.square(gpu_target_srcm - gpu_pred_src_srcm),
                        axis=[1, 2, 3],
                    )  # Calculate the loss between the GPU source mask and the GPU source prediction mask
                    

                    gpu_dst_loss = tf.reduce_mean(
                        5
                        * nn.dssim(
                            gpu_target_dst_masked_opt,
                            gpu_pred_dst_dst_masked_opt,
                            max_val=1.0,
                            filter_size=int(resolution / 11.6),
                        ),
                        axis=[1],
                    )
                    gpu_dst_loss += tf.reduce_mean(
                        5
                        * nn.dssim(
                            gpu_target_dst_masked_opt,
                            gpu_pred_dst_dst_masked_opt,
                            max_val=1.0,
                            filter_size=int(resolution / 23.2),
                        ),
                        axis=[1],
                    )
                    gpu_dst_loss += tf.reduce_mean(
                        10
                        * tf.square(
                            gpu_target_dst_masked_opt - gpu_pred_dst_dst_masked_opt
                        ),
                        axis=[1, 2, 3],
                    )

                    if eyes_prio or mouth_prio:
                        if eyes_prio and mouth_prio:
                            gpu_target_part_mask = gpu_target_dstm_eye_mouth
                        elif eyes_prio:
                            gpu_target_part_mask = gpu_target_dstm_eyes
                        elif mouth_prio:
                            gpu_target_part_mask = gpu_target_dstm_mouth

                        gpu_dst_loss += tf.reduce_mean(
                            300
                            * tf.abs(
                                gpu_target_dst * gpu_target_part_mask
                                - gpu_pred_dst_dst * gpu_target_part_mask
                            ),
                            axis=[1, 2, 3],
                        )

                    gpu_dst_loss += tf.reduce_mean(
                        10 * tf.square(gpu_target_dstm - gpu_pred_dst_dstm),
                        axis=[1, 2, 3],
                    )

                    gpu_src_losses += [gpu_src_loss]
                    gpu_dst_losses += [gpu_dst_loss]

                    gpu_G_loss = gpu_src_loss + gpu_dst_loss

                    def DLoss(labels, logits):
                        return tf.reduce_mean(
                            tf.nn.sigmoid_cross_entropy_with_logits(
                                labels=labels, logits=logits
                            ),
                            axis=[1, 2, 3],
                        )


                    if gan_power != 0:
                        gpu_pred_src_src_d, gpu_pred_src_src_d2 = self.D_src(
                            gpu_pred_src_src_masked_opt
                        )

                        def get_smooth_noisy_labels(
                            label, tensor, smoothing=0.1, noise=0.05
                        ):
                            num_labels = self.batch_size
                            for d in tensor.get_shape().as_list()[1:]:
                                num_labels *= d

                            probs = (
                                tf.math.log([[noise, 1 - noise]])
                                if label == 1
                                else tf.math.log([[1 - noise, noise]])
                            )
                            x = tf.random.categorical(probs, num_labels)
                            x = tf.cast(x, tf.float32)
                            x = tf.math.scalar_mul(1 - smoothing, x)
                            # x = x + (smoothing/num_labels)
                            x = tf.reshape(
                                x,
                                (self.batch_size,)
                                + tuple(tensor.get_shape().as_list()[1:]),
                            )
                            return x

                        smoothing = 0.1
                        noise = 0.0

                        gpu_pred_src_src_d_ones = tf.ones_like(gpu_pred_src_src_d)
                        gpu_pred_src_src_d2_ones = tf.ones_like(gpu_pred_src_src_d2)

                        gpu_pred_src_src_d_smooth_zeros = get_smooth_noisy_labels(
                            0, gpu_pred_src_src_d, smoothing=smoothing, noise=noise
                        )
                        gpu_pred_src_src_d2_smooth_zeros = get_smooth_noisy_labels(
                            0, gpu_pred_src_src_d2, smoothing=smoothing, noise=noise
                        )

                        gpu_target_src_d, gpu_target_src_d2 = self.D_src(
                            gpu_target_src_masked_opt
                        )

                        gpu_target_src_d_smooth_ones = get_smooth_noisy_labels(
                            1, gpu_target_src_d, smoothing=smoothing, noise=noise
                        )
                        gpu_target_src_d2_smooth_ones = get_smooth_noisy_labels(
                            1, gpu_target_src_d2, smoothing=smoothing, noise=noise
                        )

                        gpu_D_src_dst_loss = (
                            DLoss(gpu_target_src_d_smooth_ones, gpu_target_src_d)
                            + DLoss(gpu_pred_src_src_d_smooth_zeros, gpu_pred_src_src_d)
                            + DLoss(gpu_target_src_d2_smooth_ones, gpu_target_src_d2)
                            + DLoss(
                                gpu_pred_src_src_d2_smooth_zeros, gpu_pred_src_src_d2
                            )
                        )

                        gpu_D_src_dst_loss_gvs += [
                            nn.gradients(gpu_D_src_dst_loss, self.D_src.get_weights())
                        ]  # +self.D_src_x2.get_weights()

                        gpu_G_loss += gan_power * (
                            DLoss(gpu_pred_src_src_d_ones, gpu_pred_src_src_d)
                            + DLoss(gpu_pred_src_src_d2_ones, gpu_pred_src_src_d2)
                        )

                        if masked_training:
                            # Minimal src-src-bg rec with total_variation_mse to suppress random bright dots from gan
                            gpu_G_loss += 0.000001 * nn.total_variation_mse(
                                gpu_pred_src_src
                            )
                            gpu_G_loss += 0.02 * tf.reduce_mean(
                                tf.square(
                                    gpu_pred_src_src_anti_masked
                                    - gpu_target_src_anti_masked
                                ),
                                axis=[1, 2, 3],
                            )

                    gpu_G_loss_gvs += [
                        nn.gradients(gpu_G_loss, self.src_dst_trainable_weights)
                    ]

            # Average losses and gradients, and create optimizer update ops
            with tf.device(f"/CPU:0"):
                pred_src_src = nn.concat(gpu_pred_src_src_list, 0)
                pred_dst_dst = nn.concat(gpu_pred_dst_dst_list, 0)
                pred_src_dst = nn.concat(gpu_pred_src_dst_list, 0)
                pred_src_srcm = nn.concat(gpu_pred_src_srcm_list, 0)
                pred_dst_dstm = nn.concat(gpu_pred_dst_dstm_list, 0)
                pred_src_dstm = nn.concat(gpu_pred_src_dstm_list, 0)

            with tf.device(models_opt_device):
                src_loss = tf.concat(gpu_src_losses, 0)
                dst_loss = tf.concat(gpu_dst_losses, 0)
                src_dst_loss_gv_op = self.src_dst_opt.get_update_op(
                    nn.average_gv_list(gpu_G_loss_gvs)
                )

                if self.options["true_face_power"] != 0:
                    D_loss_gv_op = self.D_code_opt.get_update_op(
                        nn.average_gv_list(gpu_D_code_loss_gvs)
                    )

                if gan_power != 0:
                    src_D_src_dst_loss_gv_op = self.D_src_dst_opt.get_update_op(
                        nn.average_gv_list(gpu_D_src_dst_loss_gvs)
                    )

            # Initializing training and view functions
            def src_dst_train(
                warped_src,
                target_src,
                target_srcm,
                target_srcm_em,
                warped_dst,
                target_dst,
                target_dstm,
                target_dstm_em,
            ):
                s, d = nn.tf_sess.run(
                    [src_loss, dst_loss, src_dst_loss_gv_op],
                    feed_dict={
                        self.warped_src: warped_src,
                        self.target_src: target_src,
                        self.target_srcm: target_srcm,
                        self.target_srcm_em: target_srcm_em,
                        self.warped_dst: warped_dst,
                        self.target_dst: target_dst,
                        self.target_dstm: target_dstm,
                        self.target_dstm_em: target_dstm_em,
                    },
                )[:2]
                return s, d

            self.src_dst_train = src_dst_train

            def get_src_dst_information(
                warped_src,
                target_src,
                target_srcm,
                target_srcm_em,
                warped_dst,
                target_dst,
                target_dstm,
                target_dstm_em,
            ):
                out_data = nn.tf_sess.run(
                    [
                        src_loss,
                        dst_loss,
                        pred_src_src,
                        pred_src_srcm,
                        pred_dst_dst,
                        pred_dst_dstm,
                        pred_src_dst,
                        pred_src_dstm,
                    ],
                    feed_dict={
                        self.warped_src: warped_src,
                        self.target_src: target_src,
                        self.target_srcm: target_srcm,
                        self.target_srcm_em: target_srcm_em,
                        self.warped_dst: warped_dst,
                        self.target_dst: target_dst,
                        self.target_dstm: target_dstm,
                        self.target_dstm_em: target_dstm_em,
                    },
                )

                return out_data

            self.get_src_dst_information = get_src_dst_information

            if self.options["true_face_power"] != 0:

                def D_train(warped_src, warped_dst):
                    nn.tf_sess.run(
                        [D_loss_gv_op],
                        feed_dict={
                            self.warped_src: warped_src,
                            self.warped_dst: warped_dst,
                        },
                    )

                self.D_train = D_train

            if gan_power != 0:

                def D_src_dst_train(
                    warped_src,
                    target_src,
                    target_srcm,
                    target_srcm_em,
                    warped_dst,
                    target_dst,
                    target_dstm,
                    target_dstm_em,
                ):
                    nn.tf_sess.run(
                        [src_D_src_dst_loss_gv_op],
                        feed_dict={
                            self.warped_src: warped_src,
                            self.target_src: target_src,
                            self.target_srcm: target_srcm,
                            self.target_srcm_em: target_srcm_em,
                            self.warped_dst: warped_dst,
                            self.target_dst: target_dst,
                            self.target_dstm: target_dstm,
                            self.target_dstm_em: target_dstm_em,
                        },
                    )

                self.D_src_dst_train = D_src_dst_train

            def AE_view(warped_src, warped_dst):
                return nn.tf_sess.run(
                    [
                        pred_src_src,
                        pred_src_srcm,
                        pred_dst_dst,
                        pred_dst_dstm,
                        pred_src_dst,
                        pred_src_dstm,
                    ],
                    feed_dict={
                        self.warped_src: warped_src,
                        self.warped_dst: warped_dst,
                    },
                )

            self.AE_view = AE_view
        else:
            # Initializing merge function
            with tf.device(
                nn.tf_default_device_name if len(devices) != 0 else f"/CPU:0"
            ):
                if "df" in archi_type:
                    gpu_dst_code = self.inter(self.encoder(self.warped_dst))
                    gpu_pred_src_dst, gpu_pred_src_dstm = self.decoder_src(gpu_dst_code)
                    _, gpu_pred_dst_dstm = self.decoder_dst(gpu_dst_code)

                elif "liae" in archi_type:
                    gpu_dst_code = self.encoder(self.warped_dst)
                    gpu_dst_inter_B_code = self.inter_B(gpu_dst_code)
                    gpu_dst_inter_AB_code = self.inter_AB(gpu_dst_code)
                    gpu_dst_code = tf.concat(
                        [gpu_dst_inter_B_code, gpu_dst_inter_AB_code], nn.conv2d_ch_axis
                    )
                    gpu_src_dst_code = tf.concat(
                        [gpu_dst_inter_AB_code, gpu_dst_inter_AB_code],
                        nn.conv2d_ch_axis,
                    )

                    gpu_pred_src_dst, gpu_pred_src_dstm = self.decoder(gpu_src_dst_code)
                    _, gpu_pred_dst_dstm = self.decoder(gpu_dst_code)

            def AE_merge(warped_dst):
                return nn.tf_sess.run(
                    [gpu_pred_src_dst, gpu_pred_dst_dstm, gpu_pred_src_dstm],
                    feed_dict={self.warped_dst: warped_dst},
                )

            self.AE_merge = AE_merge

        # Iterate over the model and the corresponding filenames
        for model, filename in io.progress_bar_generator(
            self.model_filename_list, "Initialization Model"
        ):

            # Trying to load model weights from a file
            do_init = not model.load_weights(
                self.get_strpath_storage_for_file(filename)
            )

            if do_init and self.pretrained_model_path is not None:
                pretrained_filepath = self.pretrained_model_path / filename
                # Check if the pre-training file path exists
                if pretrained_filepath.exists():
                    #print('Find preset weights')
                    # Attempts to load pre-trained model weights, and if it fails sets do_init to True for initialization
                    do_init = not model.load_weights(pretrained_filepath)
                    
            # If initialization is required, initialize the model weights
            if do_init:
                model.init_weights()

        ###############
        # Initialize the sample generator
        if self.is_training:
            # If it's in training mode
            training_data_src_path = (
                self.training_data_src_path  # Use the specified training data source path
                if not self.pretrain  # If not in pre-training mode
                else self.get_pretraining_data_path()  # Otherwise use pre-trained data paths
            )
            training_data_dst_path = (
                self.training_data_dst_path  # Use the specified target training data path
                if not self.pretrain  # If not in pre-training mode
                else self.get_pretraining_data_path()  # Otherwise use pre-trained data paths
            )
            # If ct_mode is specified and it is not a pre-training mode, the target training data path is used
            random_ct_samples_path = (
                training_data_dst_path
                if ct_mode is not None and not self.pretrain
                else None  # Otherwise no path is used
            )

            # Get the number of CPU cores up to a set limit
            cpu_count = min(multiprocessing.cpu_count(), self.options["cpu_cap"])
            src_generators_count = cpu_count // 2  # The number of source data generators is half the number of CPU cores
            dst_generators_count = (
                cpu_count // 2
            )  # The number of target data generators is also half the number of CPU cores
            if ct_mode is not None:
                src_generators_count = int(
                    src_generators_count * 1.5
                )  # If ct_mode is specified, increase the number of source data generators

            dst_aug = None  # Initialize target data enhancement to None
            allowed_dst_augs = ["fs-aug", "cc-aug"]  # Define allowed target data enhancement types
            if ct_mode in allowed_dst_augs:
                dst_aug = ct_mode  # If ct_mode is a permitted type, data enhancements of that type are used

            channel_type = (
                SampleProcessor.ChannelType.LAB_RAND_TRANSFORM  # If the random color option is turned on
                if self.options["random_color"]
                else SampleProcessor.ChannelType.BGR  # Otherwise use the BGR channel type
            )


            ignore_same_path = False
            if (
                self.src_pak_name != self.dst_pak_name
                and training_data_src_path == training_data_dst_path
                and not self.pretrain
            ):
                ignore_same_path = True
            elif self.pretrain:
                self.src_pak_name = self.dst_pak_name = "faceset"

            # print("test super warp",self.rotation_range,self.scale_range)
            self.set_training_data_generators(
                [
                    SampleGeneratorFace(
                        training_data_src_path,
                        pak_name=self.src_pak_name,
                        ignore_same_path=ignore_same_path,
                        random_ct_samples_path=random_ct_samples_path,
                        debug=self.is_debug(),
                        batch_size=self.get_batch_size(),
                        sample_process_options=SampleProcessor.Options(
                            rotation_range=self.rotation_range,
                            scale_range=self.scale_range,
                            random_flip=random_src_flip,
                        ),
                        output_sample_types=[
                            {
                                "sample_type": SampleProcessor.SampleType.FACE_IMAGE,
                                "warp": random_warp,
                                "random_downsample": self.options["random_downsample"],
                                "random_noise": self.options["random_noise"],
                                "random_blur": self.options["random_blur"],
                                "random_jpeg": self.options["random_jpeg"],
                                "random_hsv_shift_amount": random_hsv_power,
                                "transform": True,
                                "channel_type": channel_type,
                                "ct_mode": ct_mode,
                                "face_type": self.face_type,
                                "data_format": nn.data_format,
                                "resolution": resolution,
                            },
                            {
                                "sample_type": SampleProcessor.SampleType.FACE_IMAGE,
                                "warp": False,
                                "transform": True,
                                "channel_type": channel_type,
                                "ct_mode": ct_mode,
                                "random_hsv_shift_amount": random_hsv_power,
                                "face_type": self.face_type,
                                "data_format": nn.data_format,
                                "resolution": resolution,
                            },
                            {
                                "sample_type": SampleProcessor.SampleType.FACE_MASK,
                                "warp": False,
                                "transform": True,
                                "channel_type": SampleProcessor.ChannelType.G,
                                "face_mask_type": SampleProcessor.FaceMaskType.FULL_FACE,
                                "face_type": self.face_type,
                                "data_format": nn.data_format,
                                "resolution": resolution,
                            },
                            {
                                "sample_type": SampleProcessor.SampleType.FACE_MASK,
                                "warp": False,
                                "transform": True,
                                "channel_type": SampleProcessor.ChannelType.G,
                                "face_mask_type": SampleProcessor.FaceMaskType.FULL_FACE_EYES,
                                "face_type": self.face_type,
                                "data_format": nn.data_format,
                                "resolution": resolution,
                            },
                        ],
                        uniform_yaw_distribution=self.options["uniform_yaw"]
                        or self.pretrain,
                        generators_count=src_generators_count,
                    ),
                    SampleGeneratorFace(
                        training_data_dst_path,
                        pak_name=self.dst_pak_name,
                        ignore_same_path=ignore_same_path,
                        debug=self.is_debug(),
                        batch_size=self.get_batch_size(),
                        sample_process_options=SampleProcessor.Options(
                            rotation_range=self.rotation_range,
                            scale_range=self.scale_range,
                            random_flip=random_src_flip,
                        ),
                        output_sample_types=[
                            {
                                "sample_type": SampleProcessor.SampleType.FACE_IMAGE,
                                "warp": random_warp,
                                "random_downsample": self.options["random_downsample"],
                                "random_noise": self.options["random_noise"],
                                "random_blur": self.options["random_blur"],
                                "random_jpeg": self.options["random_jpeg"],
                                "transform": True,
                                "channel_type": channel_type,
                                "ct_mode": dst_aug,
                                "face_type": self.face_type,
                                "data_format": nn.data_format,
                                "resolution": resolution,
                            },
                            {
                                "sample_type": SampleProcessor.SampleType.FACE_IMAGE,
                                "warp": False,
                                "transform": True,
                                "channel_type": channel_type,
                                "ct_mode": dst_aug,
                                "random_hsv_shift_amount": random_hsv_power,
                                "face_type": self.face_type,
                                "data_format": nn.data_format,
                                "resolution": resolution,
                            },
                            {
                                "sample_type": SampleProcessor.SampleType.FACE_MASK,
                                "warp": False,
                                "transform": True,
                                "channel_type": SampleProcessor.ChannelType.G,
                                "face_mask_type": SampleProcessor.FaceMaskType.FULL_FACE,
                                "face_type": self.face_type,
                                "data_format": nn.data_format,
                                "resolution": resolution,
                            },
                            {
                                "sample_type": SampleProcessor.SampleType.FACE_MASK,
                                "warp": False,
                                "transform": True,
                                "channel_type": SampleProcessor.ChannelType.G,
                                "face_mask_type": SampleProcessor.FaceMaskType.FULL_FACE_EYES,
                                "face_type": self.face_type,
                                "data_format": nn.data_format,
                                "resolution": resolution,
                            },
                        ],
                        uniform_yaw_distribution=self.options["uniform_yaw"]
                        or self.pretrain,
                        generators_count=dst_generators_count,
                    ),
                ]
            )

            if self.pretrain_just_disabled:
                self.update_sample_for_preview(force_new=True)

    def export_dfm(self):
        output_path = self.get_strpath_storage_for_file("model.dfm")

        io.log_info(f"Export .dfm to {output_path}")

        tf = nn.tf
        nn.set_data_format("NCHW")

        with tf.device(nn.tf_default_device_name):
            warped_dst = tf.placeholder(
                nn.floatx, (None, self.resolution, self.resolution, 3), name="in_face"
            )
            warped_dst = tf.transpose(warped_dst, (0, 3, 1, 2))

            if "df" in self.archi_type:
                gpu_dst_code = self.inter(self.encoder(warped_dst))
                gpu_pred_src_dst, gpu_pred_src_dstm = self.decoder_src(gpu_dst_code)
                _, gpu_pred_dst_dstm = self.decoder_dst(gpu_dst_code)

            elif "liae" in self.archi_type:
                gpu_dst_code = self.encoder(warped_dst)
                gpu_dst_inter_B_code = self.inter_B(gpu_dst_code)
                gpu_dst_inter_AB_code = self.inter_AB(gpu_dst_code)
                gpu_dst_code = tf.concat(
                    [gpu_dst_inter_B_code, gpu_dst_inter_AB_code], nn.conv2d_ch_axis
                )
                gpu_src_dst_code = tf.concat(
                    [gpu_dst_inter_AB_code, gpu_dst_inter_AB_code], nn.conv2d_ch_axis
                )

                gpu_pred_src_dst, gpu_pred_src_dstm = self.decoder(gpu_src_dst_code)
                _, gpu_pred_dst_dstm = self.decoder(gpu_dst_code)

            gpu_pred_src_dst = tf.transpose(gpu_pred_src_dst, (0, 2, 3, 1))
            gpu_pred_dst_dstm = tf.transpose(gpu_pred_dst_dstm, (0, 2, 3, 1))
            gpu_pred_src_dstm = tf.transpose(gpu_pred_src_dstm, (0, 2, 3, 1))

        tf.identity(gpu_pred_dst_dstm, name="out_face_mask")
        tf.identity(gpu_pred_src_dst, name="out_celeb_face")
        tf.identity(gpu_pred_src_dstm, name="out_celeb_face_mask")

        output_graph_def = tf.graph_util.convert_variables_to_constants(
            nn.tf_sess,
            tf.get_default_graph().as_graph_def(),
            ["out_face_mask", "out_celeb_face", "out_celeb_face_mask"],
        )

        import tf2onnx

        with tf.device("/CPU:0"):
            model_proto, _ = tf2onnx.convert._convert_common(
                output_graph_def,
                name="DeepAscension",
                input_names=["in_face:0"],
                output_names=[
                    "out_face_mask:0",
                    "out_celeb_face:0",
                    "out_celeb_face_mask:0",
                ],
                opset=12,
                output_path=output_path,
            )

    # override
    def get_model_filename_list(self):
        return self.model_filename_list

    # override
    def onSave(self):
        for model, filename in io.progress_bar_generator(
            self.get_model_filename_list(), "Saved...", leave=False
        ):
            model.save_weights(self.get_strpath_storage_for_file(filename))

    # override
    def should_save_preview_history(self):
        return (
            not io.is_colab()
            and self.iter % (10 * (max(1, self.resolution // 64))) == 0
        ) or (io.is_colab() and self.iter % 100 == 0)

    # override
    def onTrainOneIter(self):
        if (
            self.is_first_run()
            and not self.pretrain
            and not self.pretrain_just_disabled
        ):
            io.log_info("You are training a model from scratch.\n")

        (
            (warped_src, target_src, target_srcm, target_srcm_em),
            (warped_dst, target_dst, target_dstm, target_dstm_em),
        ) = self.generate_next_samples()

        src_loss, dst_loss = self.src_dst_train(
            warped_src,
            target_src,
            target_srcm,
            target_srcm_em,
            warped_dst,
            target_dst,
            target_dstm,
            target_dstm_em,
        )

        if self.options["true_face_power"] != 0 and not self.pretrain:
            self.D_train(warped_src, warped_dst)

        if self.gan_power != 0:
            self.D_src_dst_train(
                warped_src,
                target_src,
                target_srcm,
                target_srcm_em,
                warped_dst,
                target_dst,
                target_dstm,
                target_dstm_em,
            )

        return (
            ("src_loss", np.mean(src_loss)),
            ("dst_loss", np.mean(dst_loss)),
        )

    # override
    def onGetPreview(self, samples, for_history=False, filenames=None):
        (
            (warped_src, target_src, target_srcm, target_srcm_em),
            (warped_dst, target_dst, target_dstm, target_dstm_em),
        ) = samples

        S, D, SS, SSM, DD, DDM, SD, SDM = [
            np.clip(nn.to_data_format(x, "NHWC", self.model_data_format), 0.0, 1.0)
            for x in ([target_src, target_dst] + self.AE_view(target_src, target_dst))
        ]
        SW, DW = [
            np.clip(nn.to_data_format(x, "NHWC", self.model_data_format), 0.0, 1.0)
            for x in ([warped_src, warped_dst])
        ]
        (
            SSM,
            DDM,
            SDM,
        ) = [np.repeat(x, (3,), -1) for x in [SSM, DDM, SDM]]

        target_srcm, target_dstm = [
            nn.to_data_format(x, "NHWC", self.model_data_format)
            for x in ([target_srcm, target_dstm])
        ]

        n_samples = min(self.get_batch_size(), self.options["preview_samples"])

        if filenames is not None and len(filenames) > 0:
            for i in range(n_samples):
                S[i] = label_face_filename(S[i], filenames[0][i])
                D[i] = label_face_filename(D[i], filenames[1][i])


        result = []

        st = []
        for i in range(n_samples):
            ar = S[i], SS[i], D[i], DD[i], SD[i]
            st.append(np.concatenate(ar, axis=1))
        result += [
            ("Quick512", np.concatenate(st, axis=0)),
        ]


        st_m = []
        for i in range(n_samples):
            SD_mask = DDM[i] * SDM[i] if self.face_type < FaceType.HEAD else SDM[i]
            SM = S[i] * target_srcm[i]
            DM = D[i] * target_dstm[i]
            if filenames is not None and len(filenames) > 0:
                SM = label_face_filename(SM, filenames[0][i])
                DM = label_face_filename(DM, filenames[1][i])
            ar = SM, SS[i] * SSM[i], DM, DD[i] * DDM[i], SD[i] * SD_mask
            st_m.append(np.concatenate(ar, axis=1))

        result += [
            ("Quick512 masked", np.concatenate(st_m, axis=0)),
        ]
        
        return result

    def predictor_func(self, face=None):
        face = nn.to_data_format(face[None, ...], self.model_data_format, "NHWC")

        bgr, mask_dst_dstm, mask_src_dstm = [
            nn.to_data_format(x, "NHWC", self.model_data_format).astype(np.float32)
            for x in self.AE_merge(face)
        ]

        return bgr[0], mask_src_dstm[0][..., 0], mask_dst_dstm[0][..., 0]

    # override
    def get_MergerConfig(self):
        import merger

        return (
            self.predictor_func,
            (512, 512, 3),
            merger.MergerConfigMasked(face_type=self.face_type, default_mode="overlay"),
        )



Model = MEModel
