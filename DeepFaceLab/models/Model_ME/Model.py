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
        lowest_vram = 2
        if len(device_config.devices) != 0:
            lowest_vram = device_config.devices.get_worst_device().total_mem_gb
        if lowest_vram >= 4:
            suggest_batch_size = 8
        else:
            suggest_batch_size = 4

        # Define minimum and maximum resolution
        min_res = 64
        max_res = 640
        default_usefp16 = self.options["use_fp16"] = self.load_or_def_option(
            "use_fp16", False
        )
        default_resolution = self.options["resolution"] = self.load_or_def_option(
            "resolution", 128
        )
        default_face_type = self.options["face_type"] = self.load_or_def_option(
            "face_type", "f"
        )
        default_models_opt_on_gpu = self.options["models_opt_on_gpu"] = (
            self.load_or_def_option("models_opt_on_gpu", True)
        )

        default_archi = self.options["archi"] = self.load_or_def_option(
            "archi", "liae-ud"
        )

        default_ae_dims = self.options["ae_dims"] = self.load_or_def_option(
            "ae_dims", 256
        )
        default_e_dims = self.options["e_dims"] = self.load_or_def_option("e_dims", 64)
        default_d_dims = self.options["d_dims"] = self.options.get("d_dims", None)
        default_d_mask_dims = self.options["d_mask_dims"] = self.options.get(
            "d_mask_dims", None
        )
        default_masked_training = self.options["masked_training"] = (
            self.load_or_def_option("masked_training", True)
        )

        default_retraining_samples = self.options["retraining_samples"] = (
            self.load_or_def_option("retraining_samples", False)
        )

        default_eyes_prio = self.options["eyes_prio"] = self.load_or_def_option(
            "eyes_prio", False
        )
        default_mouth_prio = self.options["mouth_prio"] = self.load_or_def_option(
            "mouth_prio", False
        )

        # Compatibility check
        eyes_mouth_prio = self.options.get("eyes_mouth_prio")
        if eyes_mouth_prio is not None:
            default_eyes_prio = self.options["eyes_prio"] = eyes_mouth_prio
            default_mouth_prio = self.options["mouth_prio"] = eyes_mouth_prio
            self.options.pop("eyes_mouth_prio")

        default_uniform_yaw = self.options["uniform_yaw"] = self.load_or_def_option(
            "uniform_yaw", False
        )
        default_blur_out_mask = self.options["blur_out_mask"] = self.load_or_def_option(
            "blur_out_mask", False
        )

        default_adabelief = self.options["adabelief"] = self.load_or_def_option(
            "adabelief", True
        )

        lr_dropout = self.load_or_def_option("lr_dropout", "n")
        lr_dropout = {True: "y", False: "n"}.get(
            lr_dropout, lr_dropout
        )  # backward comp
        default_lr_dropout = self.options["lr_dropout"] = lr_dropout

        default_loss_function = self.options["loss_function"] = self.load_or_def_option(
            "loss_function", "SSIM"
        )

        default_random_warp = self.options["random_warp"] = self.load_or_def_option(
            "random_warp", True
        )
        default_random_hsv_power = self.options["random_hsv_power"] = (
            self.load_or_def_option("random_hsv_power", 0.0)
        )
        default_random_downsample = self.options["random_downsample"] = (
            self.load_or_def_option("random_downsample", False)
        )
        default_random_noise = self.options["random_noise"] = self.load_or_def_option(
            "random_noise", False
        )
        default_random_blur = self.options["random_blur"] = self.load_or_def_option(
            "random_blur", False
        )
        default_random_jpeg = self.options["random_jpeg"] = self.load_or_def_option(
            "random_jpeg", False
        )
        default_super_warp = self.options["super_warp"] = self.load_or_def_option(
            "super_warp", False
        )
        default_rotation_range = self.rotation_range = [-3, 3]
        default_scale_range = self.scale_range = [-0.15, 0.15]

        # Load or define other training-related default options
        default_background_power = self.options["background_power"] = (
            self.load_or_def_option("background_power", 0.0)
        )
        default_true_face_power = self.options["true_face_power"] = (
            self.load_or_def_option("true_face_power", 0.0)
        )
        default_face_style_power = self.options["face_style_power"] = (
            self.load_or_def_option("face_style_power", 0.0)
        )
        default_bg_style_power = self.options["bg_style_power"] = (
            self.load_or_def_option("bg_style_power", 0.0)
        )
        default_ct_mode = self.options["ct_mode"] = self.load_or_def_option(
            "ct_mode", "none"
        )
        default_random_color = self.options["random_color"] = self.load_or_def_option(
            "random_color", False
        )
        default_clipgrad = self.options["clipgrad"] = self.load_or_def_option(
            "clipgrad", False
        )
        default_pretrain = self.options["pretrain"] = self.load_or_def_option(
            "pretrain", False
        )
        default_cpu_cap = self.options["cpu_cap"] = self.load_or_def_option(
            "cpu_cap", 8
        )
        default_preview_samples = self.options["preview_samples"] = (
            self.load_or_def_option("preview_samples", 4)
        )
        default_full_preview = self.options["force_full_preview"] = (
            self.load_or_def_option("force_full_preview", False)
        )
        default_lr = self.options["lr"] = self.load_or_def_option("lr", 5e-5)

        # Determine if model settings need to be overridden
        ask_override = False if self.read_from_conf else self.ask_override()
        if self.is_first_run() or ask_override:
            if (
                self.read_from_conf and not self.config_file_exists
            ) or not self.read_from_conf:
                # If this is the first time it is run or the settings need to be overridden, the user is asked to enter various configurations
                self.ask_autobackup_hour()
                self.ask_maximum_n_backups()
                self.ask_write_preview_history()
                self.options["preview_samples"] = np.clip(
                    io.input_int(
                        "Preview sample size (longitudinal)",
                        default_preview_samples,
                        add_info="1 - 6",
                        help_message="Typical fine values are 4",
                    ),
                    1,
                    16,
                )
                self.options["force_full_preview"] = io.input_bool(
                    "Forced non-separation preview", default_full_preview,
                    help_message="Five columns are also expanded for large resolutions",
                )

                # Get other training-related configurations
                self.ask_reset_training()
                self.ask_target_iter()
                self.ask_retraining_samples(default_retraining_samples)
                self.ask_random_src_flip()
                self.ask_random_dst_flip()
                self.ask_batch_size(suggest_batch_size)
                self.options["use_fp16"] = io.input_bool(
                    "Use of fp16 (test function)",
                    default_usefp16,
                    help_message="Increase training speed, reduce video memory usage, increase BS limit. Easy to crash in the early stage, not enough precision in the late stage, 5000~200000 iterations are recommended, make sure to backup first!",
                )
                self.options["cpu_cap"] = np.clip(
                    io.input_int(
                        "Maximum number of CPU cores used.",
                        default_cpu_cap,
                        add_info="1 - 256",
                        help_message="Typical fine values are 8",
                    ),
                    1,
                    256,
                )

        if self.is_first_run():
            if (
                self.read_from_conf and not self.config_file_exists
            ) or not self.read_from_conf:
                # Obtaining training discrimination
                resolution = io.input_int(
                    "Resolution",
                    default_resolution,
                    add_info="64-640",
                    help_message="Higher resolutions require more VRAM and training time. This value will be adjusted to multiples of 16 and 32 to accommodate different architectures.",
                )
                resolution = np.clip((resolution // 16) * 16, min_res, max_res)
                self.options["resolution"] = resolution
                self.options["face_type"] = io.input_str(
                    "Face_type",
                    default_face_type,
                    ["h", "mf", "f", "wf", "head", "custom"],
                    help_message="Half / mid face / full face / whole face / head / custom. Half Face/Mid Face/Full Face/Full Face/Head/Custom. Half face has a higher resolution but covers a smaller area of the cheeks. Mid face is 30% wider than half face. Full Face Covers the entire face including the forehead. Head covers the entire head, but requires XSeg to get the source and destination face sets",
                ).lower()

                # Get Training Architecture Configuration
                while True:
                    archi = io.input_str(
                        "AE architecture",
                        default_archi,
                        help_message="""
                            'df' Keep more identity in the face (more like SRC).
                            'liae' Can fix too different face shapes (more like DST).
                            '-u' Increasing the similarity to the source face (SRC) requires more VRAM.
                            '-d' Halves the computational cost. Longer training times are required and the use of pre-trained models is recommended. Resolution must be changed in multiples of 32
							'-t' Increase the similarity to the source face (SRC).
							'-c' (Experimental) Set the activation function to cosine units (default: Leaky ReLu).
                            typical example: df, liae-d, df-dt, liae-udt, ...
                            """,
                    ).lower()

                    archi_split = archi.split("-")

                    if len(archi_split) == 2:
                        archi_type, archi_opts = archi_split
                    elif len(archi_split) == 1:
                        archi_type, archi_opts = archi_split[0], None
                    else:
                        continue

                    if archi_type not in ["df", "liae"]:
                        continue

                    if archi_opts is not None:
                        if len(archi_opts) == 0:
                            continue
                        if (
                            len(
                                [
                                    1
                                    for opt in archi_opts
                                    if opt not in ["u", "d", "t", "c"]
                                ]
                            )
                            != 0
                        ):
                            continue

                        if "d" in archi_opts:
                            self.options["resolution"] = np.clip(
                                (self.options["resolution"] // 32) * 32,
                                min_res,
                                max_res,
                            )

                    break
                self.options["archi"] = archi

            default_d_dims = self.options["d_dims"] = self.load_or_def_option(
                "d_dims", 64
            )

            default_d_mask_dims = default_d_dims // 3
            default_d_mask_dims += default_d_mask_dims % 2
            default_d_mask_dims = self.options["d_mask_dims"] = self.load_or_def_option(
                "d_mask_dims", default_d_mask_dims
            )
            # Author's signature
            self.ask_author_name()

        # Getting AutoEncoder, Encoder, and Decoder dimension configurations on first run
        if self.is_first_run():
            if (
                self.read_from_conf and not self.config_file_exists
            ) or not self.read_from_conf:
                self.options["ae_dims"] = np.clip(
                    io.input_int(
                        "AutoEncoder dimensions",
                        default_ae_dims,
                        add_info="32-1024",
                        help_message="All face information will be compressed to the AE dimension. If the AE dimension is not enough, e.g. closed eyes may not be recognized. More dimensions means better, but requires more VRAM. the model can be resized according to the GPU.",
                    ),
                    32,
                    1024,
                )

                e_dims = np.clip(
                    io.input_int(
                        "Encoder dimensions",
                        default_e_dims,
                        add_info="16-256",
                        help_message="More dimensions help to recognize more facial features and get clearer results but require more VRAM. the model size can be adjusted according to the GPU.",
                    ),
                    16,
                    256,
                )
                self.options["e_dims"] = e_dims + e_dims % 2

                d_dims = np.clip(
                    io.input_int(
                        "Decoder dimensions",
                        default_d_dims,
                        add_info="16-256",
                        help_message="More dimensions help to recognize more facial features and get clearer results but require more VRAM. the model size can be adjusted according to the GPU.",
                    ),
                    16,
                    256,
                )
                self.options["d_dims"] = d_dims + d_dims % 2

                d_mask_dims = np.clip(
                    io.input_int(
                        "Decoder mask dimensions",
                        default_d_mask_dims,
                        add_info="16-256",
                        help_message="A typical mask dimension is one third of the decoder dimension. You can increase this parameter for better quality if you manually remove obstacles from the target mask.",
                    ),
                    16,
                    256,
                )

                self.options["adabelief"] = io.input_bool(
                    "Use AdaBelief optimizer?",
                    default_adabelief,
                    help_message="Use the AdaBelief optimizer. It requires more VRAM, but the model is more accurate and generalized",
                )

                self.options["d_mask_dims"] = d_mask_dims + d_mask_dims % 2

        # Configuration for the first time you run it or if you need to override settings
        if self.is_first_run() or ask_override:
            if (
                self.read_from_conf and not self.config_file_exists
            ) or not self.read_from_conf:
                # Additional configurations for specific face types
                if self.options["face_type"] in ["wf", "head", "custom"]:
                    self.options["masked_training"] = io.input_bool(
                        "Masked training",
                        default_masked_training,
                        help_message="This option is only available for 'whole_face' or 'head' types. Mask training clips the training region to a full face mask or XSeg mask to better train the face.",
                    )

                # Getting eye and mouth priority configurations
                self.options["eyes_prio"] = io.input_bool(
                    "Eyes priority",
                    default_eyes_prio,
                    help_message='Helps to solve eye problems in training, such as "alien eyes" and wrong eye orientation (especially in high-resolution training), by forcing the neural network to prioritize eye training.',
                )
                self.options["mouth_prio"] = io.input_bool(
                    "Mouth priority",
                    default_mouth_prio,
                    help_message="helps to solve the mouth-in-training problem by forcing the neural network to prioritize the mouth.",
                )

                # Get other training configurations
                self.options["uniform_yaw"] = io.input_bool(
                    "Uniform yaw distribution of samples",
                    default_uniform_yaw,
                    help_message="helps to solve the problem of blurring of side faces in the sample, due to the small number of side faces in the dataset.",
                )
                self.options["blur_out_mask"] = io.input_bool(
                    "Blur out mask",
                    default_blur_out_mask,
                    help_message="The peripheral regions of the applied face mask are blurred in the training samples. The result is a smooth background near the face and less noticeable during face changing. Exact xseg masks need to be used in both source and target datasets.",
                )

        # GAN-related configurations
        default_gan_power = self.options["gan_power"] = self.load_or_def_option(
            "gan_power", 0.0
        )
        default_gan_patch_size = self.options["gan_patch_size"] = (
            self.load_or_def_option("gan_patch_size", self.options["resolution"] // 8)
        )
        default_gan_dims = self.options["gan_dims"] = self.load_or_def_option(
            "gan_dims", 16
        )
        default_gan_smoothing = self.options["gan_smoothing"] = self.load_or_def_option(
            "gan_smoothing", 0.1
        )
        default_gan_noise = self.options["gan_noise"] = self.load_or_def_option(
            "gan_noise", 0.0
        )

        if self.is_first_run() or ask_override:
            if (
                self.read_from_conf and not self.config_file_exists
            ) or not self.read_from_conf:
                self.options["models_opt_on_gpu"] = io.input_bool(
                    "Place models and optimizer on GPU",
                    default_models_opt_on_gpu,
                    help_message="When training on one GPU, by default the model and optimizer weights are placed on the GPU to speed up the training process. You can place them on the CPU to free up additional VRAM to set larger dimensions",
                )

                self.options["lr_dropout"] = io.input_str(
                    f"Use learning rate dropout",
                    default_lr_dropout,
                    ["n", "y", "cpu"],
                    help_message="When the face is trained well enough, this option can be enabled to get extra sharpness and reduce sub-pixel jitter, thus reducing the number of iterations. Enable before disabling random warping and GAN. Enable on the CPU. This allows you to sacrifice 20% of the iteration time by not using extra VRAM",
                )

                self.options["loss_function"] = io.input_str(
                    f"Loss function",
                    default_loss_function,
                    ["SSIM", "MS-SSIM", "MS-SSIM+L1"],
                    help_message="Variation loss function for image quality assessment",
                )

                self.options["lr"] = np.clip(
                    io.input_number(
                        "Learning rate",
                        default_lr,
                        add_info="0.0 .. 1.0",
                        help_message="Learning rate: Typical fine values 5e-5",
                    ),
                    0.0,
                    1,
                )

                self.options["random_warp"] = io.input_bool(
                    "Enable random warp of samples",
                    default_random_warp,
                    help_message="To generalize the facial expressions of two faces, random warping is required. When faces are trained well enough, it can be disabled to gain extra sharpness and reduce sub-pixel jitter, thus reducing the number of iterations",
                )

                self.options["random_hsv_power"] = np.clip(
                    io.input_number(
                        "Random hue/saturation/light intensity",
                        default_random_hsv_power,
                        add_info="0.0 .. 0.3",
                        help_message="Random hue/saturation/light intensity is only applied to the src face set input by the neural network. Stabilizes color perturbations during face swapping. Reduce the quality of color conversion by selecting the closest faces in the original face set. Therefore the src face set must be sufficiently diverse. Typical refinement values are 0.05",
                    ),
                    0.0,
                    0.3,
                )

                self.options["random_downsample"] = io.input_bool(
                    "Enable random downsample of samples",
                    default_random_downsample,
                    help_message="Challenging the model by shrinking part of the sample",
                )
                self.options["random_noise"] = io.input_bool(
                    "Enable random noise added to samples",
                    default_random_noise,
                    help_message="Challenge the model by adding noise to some samples",
                )
                self.options["random_blur"] = io.input_bool(
                    "Enable random blur of samples",
                    default_random_blur,
                    help_message="Challenging the model by adding blurring effects to certain samples",
                )
                self.options["random_jpeg"] = io.input_bool(
                    "Enable random jpeg compression of samples",
                    default_random_jpeg,
                    help_message="Challenge the model by applying jpeg-compressed quality degradation to some samples",
                )

                self.options["super_warp"] = io.input_bool(
                    "Enable super warp of samples",
                    default_super_warp,
                    help_message="Don't turn it on most of the time, it takes up more time and space. Only if the dst has an exaggeratedly large expression and the src doesn't, try increasing the amount of computation in order to blend it in. Maybe with Mouth priority!",
                )

                # if self.options["super_warp"] == True:
                # self.rotation_range=[-15,15]
                # self.scale_range=[-0.25, 0.25]

                """
                self.options["random_shadow"] = io.input_str(
                    "Enable random shadows and highlights of samples",
                    default_random_shadow,
                    ["none", "src", "dst", "all"],
                    help_message="Helps to create dark light areas in the dataset. If your src dataset lacks shadows/different light situations; use dst to help generalize; or use all for both!",
                )
                """
                self.options["gan_power"] = np.clip(
                    io.input_number(
                        "GAN power",
                        default_gan_power,
                        add_info="0.0 .. 10.0",
                        help_message="Train the network in a generative adversarial manner. Force the neural network to learn small details about faces. Enable it only if the face is trained well enough, otherwise don't disable it. Typical value is 0.1",
                    ),
                    0.0,
                    10.0,
                )

                if self.options["gan_power"] != 0.0:
                    gan_patch_size = np.clip(
                        io.input_int(
                            "GAN patch size",
                            default_gan_patch_size,
                            add_info="3-640",
                            help_message="The larger the patch size, the higher the quality and the more video memory required. You will get sharper edges even at the lowest settings. Typical good values are resolution divided by 8",
                        ),
                        3,
                        640,
                    )
                    self.options["gan_patch_size"] = gan_patch_size

                    gan_dims = np.clip(
                        io.input_int(
                            "GAN dimensions",
                            default_gan_dims,
                            add_info="4-64",
                            help_message="Size of the GAN network. The larger the size, the more VRAM is required. Sharper edges are obtained even at the lowest settings. Typical fine values are 16",
                        ),
                        4,
                        64,
                    )
                    self.options["gan_dims"] = gan_dims

                    self.options["gan_smoothing"] = np.clip(
                        io.input_number(
                            "GAN label smoothing",
                            default_gan_smoothing,
                            add_info="0 - 0.5",
                            help_message="Regularization effect using soft labels whose values slightly deviate from 0/1 of the GAN",
                        ),
                        0,
                        0.5,
                    )
                    self.options["gan_noise"] = np.clip(
                        io.input_number(
                            "GAN noisy labels",
                            default_gan_noise,
                            add_info="0 - 0.5",
                            help_message="Labeling certain images with incorrect labels helps prevent collapse",
                        ),
                        0,
                        0.5,
                    )

                if "df" in self.options["archi"]:
                    self.options["true_face_power"] = np.clip(
                        io.input_number(
                            "True face (src) power.",
                            default_true_face_power,
                            add_info="0.0000 .. 1.0",
                            help_message="Experimental Options. Discriminate the resultant face to be more like the original face. The larger the value, the better the discrimination. Typical value is 0.01. Comparison - https://i.imgur.com/czScS9q.png",
                        ),
                        0.0,
                        1.0,
                    )
                else:
                    self.options["true_face_power"] = 0.0

                self.options["background_power"] = np.clip(
                    io.input_number(
                        "background (src) power",
                        default_background_power,
                        add_info="0.0..1.0",
                        help_message="Understand the area outside the mask. Helps smooth out areas near the mask boundary. Ready to use",
                    ),
                    0.0,
                    1.0,
                )

                self.options["face_style_power"] = np.clip(
                    io.input_number(
                        "Face style (dst) power",
                        default_face_style_power,
                        add_info="0.0..100.0",
                        help_message="Learns to predict the color of the face so that it is the same as the dst inside the mask. To use this option with whole_face, the XSeg training mask must be used. Warning: Enable this option only after 10k passes, when the predicted face is clear enough to start learning the style. Start with a value of 0.001 and check for historical changes. Enabling this option increases the chance of model crashes!",
                    ),
                    0.0,
                    100.0,
                )
                self.options["bg_style_power"] = np.clip(
                    io.input_number(
                        "Background style (dst) power",
                        default_bg_style_power,
                        add_info="0.0..100.0",
                        help_message="Learning to predict the area outside the face mask is the same as dst. To use this option for whole_face, you must use the XSeg training mask. For whole_face, you must use the XSeg training mask. This will make the face more like dst. enabling this option will increase the chance of model crashes. Typical value is 2.0",
                    ),
                    0.0,
                    100.0,
                )

                self.options["ct_mode"] = io.input_str(
                    f"Color transfer for src faceset",
                    default_ct_mode,
                    ["none", "rct", "lct", "mkl", "idt", "sot", "fs-aug", "cc-aug"],
                    help_message="Change the color distribution of the src sample close to the dst sample. Try all modes to find the best solution. cc and fs aug add random colors to dst and src",
                )
                self.options["random_color"] = io.input_bool(
                    "Random color",
                    default_random_color,
                    help_message="In LAB color space, the samples are randomly rotated around the L-axis, which helps train generalization. To put it in human terms, the brightness remains the same, but the hue changes more than hsv. hsv's brightness and contrast are really not recommended to be drastic, so this option is complementary, and the recommendation is to turn it on in rotation rather than at the same time!",
                )
                self.options["clipgrad"] = io.input_bool(
                    "Enable gradient clipping",
                    default_clipgrad,
                    help_message="Gradient trimming reduces the chance of model crashes, but sacrifices training speed",
                )

                self.options["pretrain"] = io.input_bool(
                    "Enable pretraining mode",
                    default_pretrain,
                    help_message="Use a large variety of face pre-training models that can be used to train fake data faster. Forces the use of random_warp=N, random_flips=Y, gan_power=0.0, lr_dropout=N, styles=0.0, uniform_yaw=Y",
                )

        if self.options["pretrain"] and self.get_pretraining_data_path() is None:
            raise Exception("pretraining_data_path is not defined")

        self.gan_model_changed = (
            default_gan_patch_size != self.options["gan_patch_size"]
        ) or (default_gan_dims != self.options["gan_dims"])
        # pre-training to regularization
        self.pretrain_just_disabled = (
            default_pretrain == True and self.options["pretrain"] == False
        )

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
        self.resolution = resolution = self.options["resolution"]
        self.face_type = {
            "h": FaceType.HALF,
            "mf": FaceType.MID_FULL,
            "f": FaceType.FULL,
            "wf": FaceType.WHOLE_FACE,
            "custom": FaceType.CUSTOM,
            "head": FaceType.HEAD,
        }[self.options["face_type"]]

        # Setting Eye and Mouth Priorities
        eyes_prio = self.options["eyes_prio"]
        mouth_prio = self.options["mouth_prio"]

        # Parsing Architecture Types
        archi_split = self.options["archi"].split("-")
        if len(archi_split) == 2:
            archi_type, archi_opts = archi_split
        elif len(archi_split) == 1:
            archi_type, archi_opts = archi_split[0], None
        self.archi_type = archi_type

        # Setting the dimensions of AutoEncoder, Encoder and Decoder
        ae_dims = self.options["ae_dims"]
        e_dims = self.options["e_dims"]
        d_dims = self.options["d_dims"]
        d_mask_dims = self.options["d_mask_dims"]

        # Setting whether to pre-train or not
        self.pretrain = self.options["pretrain"]
        if self.pretrain_just_disabled:
            ask_for_clean = input("Does it zero out the number of iterations? Please enter 'y' or 'n':")
            if ask_for_clean.lower() == "y":
                self.set_iter(0)
                print("The number of iterations has been reset!")
            else:
                print("Retain the number of iterations to end pre-training!")

        # Set whether or not to use the AdaBelief optimizer
        adabelief = self.options["adabelief"]

        # Set whether to use half-precision floating point numbers
        use_fp16 = self.options["use_fp16"]
        if self.is_exporting:
            use_fp16 = io.input_bool(
                "Export quantized?",
                False,
                help_message="Makes exported models faster. If you encounter problems, disable this option.",
            )

        # Setting the relevant parameters (all locks of the pre-training have been unlocked, except GAN)
        self.gan_power = gan_power = 0.0 if self.pretrain else self.options["gan_power"]
        random_warp = self.options["random_warp"]
        random_src_flip = self.random_src_flip
        random_dst_flip = self.random_dst_flip
        random_hsv_power = self.options["random_hsv_power"]
        blur_out_mask = self.options["blur_out_mask"]

        # If in the pre-training phase, adjust some parameter settings (RW\flip\hsv\blur, retain gan and style restrictions have been solved)
        if self.pretrain:
            self.options_show_override["gan_power"] = 0.0
            self.options_show_override["face_style_power"] = 0.0
            self.options_show_override["bg_style_power"] = 0.0

        # Setting whether to perform mask training and color conversion mode
        masked_training = self.options["masked_training"]
        ct_mode = self.options["ct_mode"]
        if ct_mode == "none":
            ct_mode = None

        """
        # Setting random shadow sources and targets based on profile usage
        if (
            self.read_from_conf and not self.config_file_exists
        ) or not self.read_from_conf:
            random_shadow_src = (
                True if self.options["random_shadow"] in ["all", "src"] else False
            )
            random_shadow_dst = (
                True if self.options["random_shadow"] in ["all", "dst"] else False
            )

            # Remove the random shading option if this is the first time a model is created using a profile
            if not self.config_file_exists and self.read_from_conf:
                del self.options["random_shadow"]
        else:
            random_shadow_src = self.options["random_shadow_src"]
            random_shadow_dst = self.options["random_shadow_dst"]
        """

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
        model_archi = nn.DeepFakeArchi(resolution, use_fp16=use_fp16, opts=archi_opts)

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
                        patch_size=self.options["gan_patch_size"],
                        in_ch=input_ch,
                        base_ch=self.options["gan_dims"],
                        use_fp16=self.options["use_fp16"],
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
                OptimizerClass = nn.AdaBelief if adabelief else nn.RMSprop
                clipnorm = 1.0 if self.options["clipgrad"] else 0.0

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

                # If using real face strength, initialize the code discriminator optimizer
                if self.options["true_face_power"] != 0:
                    self.D_code_opt = OptimizerClass(
                        lr=lr,
                        lr_dropout=lr_dropout,
                        lr_cos=lr_cos,
                        clipnorm=clipnorm,
                        name="D_code_opt",
                    )
                    self.D_code_opt.initialize_variables(
                        self.code_discriminator.get_weights(),
                        vars_on_cpu=optimizer_vars_on_cpu,
                        lr_dropout_on_cpu=self.options["lr_dropout"] == "cpu",
                    )
                    self.model_filename_list += [(self.D_code_opt, "D_code_opt.npy")]

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

                    if blur_out_mask:
                        sigma = resolution / 128

                        x = nn.gaussian_blur(
                            gpu_target_src * gpu_target_srcm_anti, sigma
                        )
                        y = 1 - nn.gaussian_blur(gpu_target_srcm_all, sigma)
                        y = tf.where(tf.equal(y, 0), tf.ones_like(y), y)
                        gpu_target_src = (
                            gpu_target_src * gpu_target_srcm_all
                            + (x / y) * gpu_target_srcm_anti
                        )

                        x = nn.gaussian_blur(
                            gpu_target_dst * gpu_target_dstm_anti, sigma
                        )
                        y = 1 - nn.gaussian_blur(gpu_target_dstm_all, sigma)
                        y = tf.where(tf.equal(y, 0), tf.ones_like(y), y)
                        gpu_target_dst = (
                            gpu_target_dst * gpu_target_dstm_all
                            + (x / y) * gpu_target_dstm_anti
                        )

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

                    if self.options["loss_function"] == "MS-SSIM":
                        # Using the MS-SSIM loss function
                        gpu_src_loss = 10 * nn.MsSsim(bs_per_gpu, input_ch, resolution)(
                            gpu_target_src_masked_opt,
                            gpu_pred_src_src_masked_opt,
                            max_val=1.0,
                        )
                        gpu_src_loss += tf.reduce_mean(
                            10
                            * tf.square(
                                gpu_target_src_masked_opt - gpu_pred_src_src_masked_opt
                            ),
                            axis=[1, 2, 3],
                        )
                    elif self.options["loss_function"] == "MS-SSIM+L1":
                        # Using MS-SSIM + L1 loss function
                        gpu_src_loss = 10 * nn.MsSsim(
                            bs_per_gpu, input_ch, resolution, use_l1=True
                        )(
                            gpu_target_src_masked_opt,
                            gpu_pred_src_src_masked_opt,
                            max_val=1.0,
                        )
                    else:
                        # Using other loss functions
                        if resolution < 256:
                            gpu_src_loss = tf.reduce_mean(
                                10
                                * nn.dssim(
                                    gpu_target_src_masked_opt,
                                    gpu_pred_src_src_masked_opt,
                                    max_val=1.0,
                                    filter_size=int(resolution / 11.6),
                                ),
                                axis=[1],
                            )
                        else:
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

                    if self.options["background_power"] > 0:
                        bg_factor = self.options["background_power"]

                        if self.options["loss_function"] == "MS-SSIM":
                            gpu_src_loss += (
                                bg_factor
                                * 10
                                * nn.MsSsim(bs_per_gpu, input_ch, resolution)(
                                    gpu_target_src, gpu_pred_src_src, max_val=1.0
                                )
                            )
                            gpu_src_loss += bg_factor * tf.reduce_mean(
                                10 * tf.square(gpu_target_src - gpu_pred_src_src),
                                axis=[1, 2, 3],
                            )
                        elif self.options["loss_function"] == "MS-SSIM+L1":
                            gpu_src_loss += (
                                bg_factor
                                * 10
                                * nn.MsSsim(
                                    bs_per_gpu, input_ch, resolution, use_l1=True
                                )(gpu_target_src, gpu_pred_src_src, max_val=1.0)
                            )
                        else:
                            if resolution < 256:
                                gpu_src_loss += bg_factor * tf.reduce_mean(
                                    10
                                    * nn.dssim(
                                        gpu_target_src,
                                        gpu_pred_src_src,
                                        max_val=1.0,
                                        filter_size=int(resolution / 11.6),
                                    ),
                                    axis=[1],
                                )
                            else:
                                gpu_src_loss += bg_factor * tf.reduce_mean(
                                    5
                                    * nn.dssim(
                                        gpu_target_src,
                                        gpu_pred_src_src,
                                        max_val=1.0,
                                        filter_size=int(resolution / 11.6),
                                    ),
                                    axis=[1],
                                )
                                gpu_src_loss += bg_factor * tf.reduce_mean(
                                    5
                                    * nn.dssim(
                                        gpu_target_src,
                                        gpu_pred_src_src,
                                        max_val=1.0,
                                        filter_size=int(resolution / 23.2),
                                    ),
                                    axis=[1],
                                )
                            gpu_src_loss += bg_factor * tf.reduce_mean(
                                10 * tf.square(gpu_target_src - gpu_pred_src_src),
                                axis=[1, 2, 3],
                            )

                    face_style_power = self.options["face_style_power"] / 100.0
                    if face_style_power != 0 and not self.pretrain:
                        gpu_src_loss += nn.style_loss(
                            gpu_pred_src_dst_no_code_grad
                            * tf.stop_gradient(gpu_pred_src_dstm),
                            tf.stop_gradient(gpu_pred_dst_dst * gpu_pred_dst_dstm),
                            gaussian_blur_radius=resolution // 8,
                            loss_weight=10000 * face_style_power,
                        )

                    bg_style_power = self.options["bg_style_power"] / 100.0
                    if bg_style_power != 0 and not self.pretrain:
                        gpu_target_dst_style_anti_masked = (
                            gpu_target_dst * gpu_style_mask_anti_blur
                        )
                        gpu_psd_style_anti_masked = (
                            gpu_pred_src_dst * gpu_style_mask_anti_blur
                        )

                        gpu_src_loss += tf.reduce_mean(
                            (10 * bg_style_power)
                            * nn.dssim(
                                gpu_psd_style_anti_masked,
                                gpu_target_dst_style_anti_masked,
                                max_val=1.0,
                                filter_size=int(resolution / 11.6),
                            ),
                            axis=[1],
                        )
                        gpu_src_loss += tf.reduce_mean(
                            (10 * bg_style_power)
                            * tf.square(
                                gpu_psd_style_anti_masked
                                - gpu_target_dst_style_anti_masked
                            ),
                            axis=[1, 2, 3],
                        )

                    if self.options["loss_function"] == "MS-SSIM":
                        gpu_dst_loss = 10 * nn.MsSsim(bs_per_gpu, input_ch, resolution)(
                            gpu_target_dst_masked_opt,
                            gpu_pred_dst_dst_masked_opt,
                            max_val=1.0,
                        )
                        gpu_dst_loss += tf.reduce_mean(
                            10
                            * tf.square(
                                gpu_target_dst_masked_opt - gpu_pred_dst_dst_masked_opt
                            ),
                            axis=[1, 2, 3],
                        )
                    elif self.options["loss_function"] == "MS-SSIM+L1":
                        gpu_dst_loss = 10 * nn.MsSsim(
                            bs_per_gpu, input_ch, resolution, use_l1=True
                        )(
                            gpu_target_dst_masked_opt,
                            gpu_pred_dst_dst_masked_opt,
                            max_val=1.0,
                        )
                    else:
                        if resolution < 256:
                            gpu_dst_loss = tf.reduce_mean(
                                10
                                * nn.dssim(
                                    gpu_target_dst_masked_opt,
                                    gpu_pred_dst_dst_masked_opt,
                                    max_val=1.0,
                                    filter_size=int(resolution / 11.6),
                                ),
                                axis=[1],
                            )
                        else:
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

                    if self.options["background_power"] > 0:
                        bg_factor = self.options["background_power"]

                        if self.options["loss_function"] == "MS-SSIM":
                            gpu_dst_loss += (
                                bg_factor
                                * 10
                                * nn.MsSsim(bs_per_gpu, input_ch, resolution)(
                                    gpu_target_dst, gpu_pred_dst_dst, max_val=1.0
                                )
                            )
                            gpu_dst_loss += bg_factor * tf.reduce_mean(
                                10 * tf.square(gpu_target_dst - gpu_pred_dst_dst),
                                axis=[1, 2, 3],
                            )
                        elif self.options["loss_function"] == "MS-SSIM+L1":
                            gpu_dst_loss += (
                                bg_factor
                                * 10
                                * nn.MsSsim(
                                    bs_per_gpu, input_ch, resolution, use_l1=True
                                )(gpu_target_dst, gpu_pred_dst_dst, max_val=1.0)
                            )
                        else:
                            if resolution < 256:
                                gpu_dst_loss += bg_factor * tf.reduce_mean(
                                    10
                                    * nn.dssim(
                                        gpu_target_dst,
                                        gpu_pred_dst_dst,
                                        max_val=1.0,
                                        filter_size=int(resolution / 11.6),
                                    ),
                                    axis=[1],
                                )
                            else:
                                gpu_dst_loss += bg_factor * tf.reduce_mean(
                                    5
                                    * nn.dssim(
                                        gpu_target_dst,
                                        gpu_pred_dst_dst,
                                        max_val=1.0,
                                        filter_size=int(resolution / 11.6),
                                    ),
                                    axis=[1],
                                )
                                gpu_dst_loss += bg_factor * tf.reduce_mean(
                                    5
                                    * nn.dssim(
                                        gpu_target_dst,
                                        gpu_pred_dst_dst,
                                        max_val=1.0,
                                        filter_size=int(resolution / 23.2),
                                    ),
                                    axis=[1],
                                )
                            gpu_dst_loss += bg_factor * tf.reduce_mean(
                                10 * tf.square(gpu_target_dst - gpu_pred_dst_dst),
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

                    if self.options["true_face_power"] != 0:
                        gpu_src_code_d = self.code_discriminator(gpu_src_code)
                        gpu_src_code_d_ones = tf.ones_like(gpu_src_code_d)
                        gpu_src_code_d_zeros = tf.zeros_like(gpu_src_code_d)
                        gpu_dst_code_d = self.code_discriminator(gpu_dst_code)
                        gpu_dst_code_d_ones = tf.ones_like(gpu_dst_code_d)

                        gpu_G_loss += self.options["true_face_power"] * DLoss(
                            gpu_src_code_d_ones, gpu_src_code_d
                        )

                        gpu_D_code_loss = (
                            DLoss(gpu_dst_code_d_ones, gpu_dst_code_d)
                            + DLoss(gpu_src_code_d_zeros, gpu_src_code_d)
                        ) * 0.5

                        gpu_D_code_loss_gvs += [
                            nn.gradients(
                                gpu_D_code_loss, self.code_discriminator.get_weights()
                            )
                        ]

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

                        smoothing = self.options["gan_smoothing"]
                        noise = self.options["gan_noise"]

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
            # Check if pre-training has just been disabled
            if self.pretrain_just_disabled:
                do_init = False
                # If the architecture type contains "df"
                if "df" in archi_type:
                    # For specific models, initialization is required
                    if model == self.inter:
                        print("Pre-training to regularization...")
                        ask_for_clean = input(
                            "Whether to reset the inter? (reset works better, but training is slower) Enter 'y' or 'n':"
                        )
                        if ask_for_clean.lower() == "y":
                            do_init = True
                            print("The inter has been reset!")
                        else:
                            do_init = False
                            print("Keep inter to continue training! It is recommended to turn on Random Twist!")

                # If the architecture type is "liae"
                elif "liae" in archi_type:
                    # For specific models, initialization is required
                    if model == self.inter_AB:
                        ask_for_clean = input(
                            "To reset inter_AB or not (reset works better, but training is slower) enter 'y' or 'n':"
                        )
                        if ask_for_clean.lower() == "y":
                            do_init = True
                            print("inter_AB has been reset!")
                        else:
                            do_init = False
                            print("Keep inter_AB to continue training! It is recommended to turn on Random Twist!")
            else:
                # Check if this is the first run
                do_init = self.is_first_run()
                # If it is a training mode and the GAN's capability is not 0, initialize the particular model
                if self.is_training and gan_power != 0 and model == self.D_src:
                    if self.gan_model_changed:
                        do_init = True

            # If no initialization is required, attempt to load model weights from file
            if not do_init:
                do_init = not model.load_weights(
                    self.get_strpath_storage_for_file(filename)
                )

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

            # Check for pak names
            # give priority to pak names in configuration file
            if self.read_from_conf and self.config_file_exists:
                conf_src_pak_name = self.options.get("src_pak_name", None)
                conf_dst_pak_name = self.options.get("dst_pak_name", None)
                if conf_src_pak_name is not None:
                    self.src_pak_name = conf_src_pak_name
                if conf_dst_pak_name is not None:
                    self.dst_pak_name = conf_dst_pak_name

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

            if self.options["retraining_samples"]:
                self.last_src_samples_loss = []
                self.last_dst_samples_loss = []

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
                name="SAEHD512",
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
            io.log_info("You are training a model from scratch. It is highly recommended to use a pre-trained model to increase efficiency.\n")

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

        if self.options["retraining_samples"]:
            bs = self.get_batch_size()

            for i in range(bs):
                self.last_src_samples_loss.append(
                    (target_src[i], target_srcm[i], target_srcm_em[i], src_loss[i])
                )
                self.last_dst_samples_loss.append(
                    (target_dst[i], target_dstm[i], target_dstm_em[i], dst_loss[i])
                )

            if len(self.last_src_samples_loss) >= bs * 16:
                src_samples_loss = sorted(
                    self.last_src_samples_loss, key=operator.itemgetter(3), reverse=True
                )
                dst_samples_loss = sorted(
                    self.last_dst_samples_loss, key=operator.itemgetter(3), reverse=True
                )

                target_src = np.stack([x[0] for x in src_samples_loss[:bs]])
                target_srcm = np.stack([x[1] for x in src_samples_loss[:bs]])
                target_srcm_em = np.stack([x[2] for x in src_samples_loss[:bs]])

                target_dst = np.stack([x[0] for x in dst_samples_loss[:bs]])
                target_dstm = np.stack([x[1] for x in dst_samples_loss[:bs]])
                target_dstm_em = np.stack([x[2] for x in dst_samples_loss[:bs]])

                src_loss, dst_loss = self.src_dst_train(
                    target_src,
                    target_src,
                    target_srcm,
                    target_srcm_em,
                    target_dst,
                    target_dst,
                    target_dstm,
                    target_dstm_em,
                )
                self.last_src_samples_loss = []
                self.last_dst_samples_loss = []

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

        if self.resolution <= 256 or self.options["force_full_preview"] == True:
            result = []

            st = []
            for i in range(n_samples):
                ar = S[i], SS[i], D[i], DD[i], SD[i]
                st.append(np.concatenate(ar, axis=1))
            result += [
                ("SN", np.concatenate(st, axis=0)),
            ]

            wt = []
            for i in range(n_samples):
                ar = SW[i], SS[i], DW[i], DD[i], SD[i]
                wt.append(np.concatenate(ar, axis=1))
            result += [
                ("SN warped", np.concatenate(wt, axis=0)),
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
                ("SN masked", np.concatenate(st_m, axis=0)),
            ]
        else:
            result = []

            st = []
            for i in range(n_samples):
                ar = S[i], SS[i]
                st.append(np.concatenate(ar, axis=1))
            result += [
                ("SN src-src", np.concatenate(st, axis=0)),
            ]

            st = []
            for i in range(n_samples):
                ar = D[i], DD[i]
                st.append(np.concatenate(ar, axis=1))
            result += [
                ("SN dst-dst", np.concatenate(st, axis=0)),
            ]

            st = []
            for i in range(n_samples):
                ar = D[i], SD[i]
                st.append(np.concatenate(ar, axis=1))
            result += [
                ("SN pred", np.concatenate(st, axis=0)),
            ]

            wt = []
            for i in range(n_samples):
                ar = SW[i], SS[i]
                wt.append(np.concatenate(ar, axis=1))
            result += [
                ("SN warped src-src", np.concatenate(wt, axis=0)),
            ]

            wt = []
            for i in range(n_samples):
                ar = DW[i], DD[i]
                wt.append(np.concatenate(ar, axis=1))
            result += [
                ("SN warped dst-dst", np.concatenate(wt, axis=0)),
            ]

            wt = []
            for i in range(n_samples):
                ar = DW[i], SD[i]
                wt.append(np.concatenate(ar, axis=1))
            result += [
                ("SN warped pred", np.concatenate(wt, axis=0)),
            ]

            st_m = []
            for i in range(n_samples):
                ar = S[i] * target_srcm[i], SS[i] * SSM[i]
                st_m.append(np.concatenate(ar, axis=1))
            result += [
                ("SN masked src-src", np.concatenate(st_m, axis=0)),
            ]

            st_m = []
            for i in range(n_samples):
                ar = D[i] * target_dstm[i], DD[i] * DDM[i]
                st_m.append(np.concatenate(ar, axis=1))
            result += [
                ("SN masked dst-dst", np.concatenate(st_m, axis=0)),
            ]

            st_m = []
            for i in range(n_samples):
                SD_mask = DDM[i] * SDM[i] if self.face_type < FaceType.HEAD else SDM[i]
                ar = D[i] * target_dstm[i], SD[i] * SD_mask
                st_m.append(np.concatenate(ar, axis=1))
            result += [
                ("SN masked pred", np.concatenate(st_m, axis=0)),
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
            (self.options["resolution"], self.options["resolution"], 3),
            merger.MergerConfigMasked(face_type=self.face_type, default_mode="overlay"),
        )

    # override
    def get_config_schema_path(self):
        config_path = Path(__file__).parent.absolute() / Path("config_schema.json")
        return config_path

    # override
    def get_formatted_configuration_path(self):
        config_path = Path(__file__).parent.absolute() / Path("formatted_config.yaml")
        return config_path

    # function is WIP
    def generate_training_state(self):
        # Import the required modules
        from tqdm import tqdm

        import datetime
        import json
        from itertools import zip_longest
        import multiprocessing as mp

        # Generate training status
        src_gen = self.generator_list[
            0
        ]  # Get the first generator object in the generator list and assign it to the variable src_gen
        dst_gen = self.generator_list[
            1
        ]  # Get the second generator object in the generator list and assign it to the variable dst_gen
        self.src_sample_state = []  # Initialize variable self.src_sample_state to empty list
        self.dst_sample_state = []  # Initialize variable self.dst_sample_state to empty list

        src_samples = src_gen.samples  # Get a sample of the source generator
        dst_samples = dst_gen.samples  # Get a sample of the target generator
        src_len = len(src_samples)  # Get the length of the source sample
        dst_len = len(dst_samples)  # Get the length of the target sample
        length = src_len  # Initialization length is the length of the source sample
        if length < dst_len:  # If the length of the source sample is less than the length of the target sample
            length = dst_len  # The update length is the length of the target sample

        # set paths
        # create core folder
        self.state_history_path = self.saved_models_path / (
            f"{self.get_model_name()}_state_history"
        )
        # The state history path is the specific model name under the Save Model path plus an underscore plus the state history.
        if not self.state_history_path.exists():
            # If the status history path does not exist
            self.state_history_path.mkdir(exist_ok=True)
            # Create a new directory under the status history path
        # create state folder
        idx_str = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")  # Get current timestamp
        idx_state_history_path = (
            self.state_history_path / idx_str
        )  # Get status history path
        idx_state_history_path.mkdir()  # Creating a Status History Path
        # create set folders
        self.src_state_path = (
            idx_state_history_path / "src"
        )  # Specify the source state path as the "src" folder in the index state history path.
        self.src_state_path.mkdir()  # Creating a source status folder
        self.dst_state_path = (
            idx_state_history_path / "dst"
        )  # Specify the target state path as the "dst" folder in the index state history path.
        self.dst_state_path.mkdir()  # Creating a destination status folder

        print("Generating dataset state snapshot\r")

        # doing batch 2 always since it is coded to always expect dst and src
        # if one is smaller reapeating the last sample as a placeholder

        # 0 means ignore and use dummy data
        # Pack the source and target samples in equal lengths and fill them with zeros
        data_list = list(zip_longest(src_samples, dst_samples, fillvalue=0))

        # Create an all-zero virtual input array of shape (self.resolution, self.resolution, 3) with data type np.float32
        self._dummy_input = np.zeros(
            (self.resolution, self.resolution, 3), dtype=np.float32
        )

        # Create an all-zero virtual mask array of shape (self.resolution, self.resolution, 1) with data type np.float32
        self._dummy_mask = np.zeros(
            (self.resolution, self.resolution, 1), dtype=np.float32
        )

        # For each sample tuple in the data list, iterate using the tqdm library
        for sample_tuple in tqdm(data_list, desc="Data download in progress", total=len(data_list)):
            # Call the processor function to process the sample tuple
            self._processor(sample_tuple)

        # save model state params
        # copy model summary
        # model_summary = self.options.copy()
        model_summary = {}  # Create an empty dictionary for storing model summary information
        model_summary["iter"] = (
            self.get_iter()
        )  # Get the current iteration count and store it in the model_summary dictionary as the value of the "iter" key.
        model_summary["name"] = (
            self.get_model_name()
        )  # Get the name of the model and store it in the model_summary dictionary as the value of the "name" key.

        # error with some types, need to double check
        with open(idx_state_history_path / "model_summary.json", "w") as outfile:
            # Use the open function to open the file model_summary.json and write the data to the file in write mode
            # The file pointer is automatically managed by the with statement
            json.dump(model_summary, outfile)
            # Write the data in model_summary to the outfile file in json format

        # training state, full loss stuff from .dat file - prolly should be global
        # state_history_json = self.loss_history

        # main config data
        # set name and full path
        config_dict = {
            "datasets": [
                {"name": "src", "path": str(self.training_data_src_path)},
                {"name": "dst", "path": str(self.training_data_dst_path)},
            ]
        }

        # Create a configuration dictionary containing the training data source path and the training data target path
        with open(self.state_history_path / "config.json", "w") as outfile:
            json.dump(config_dict, outfile)
        # save image loss data
        src_full_state_dict = {
            "data": self.src_sample_state,  # Define a dictionary src_full_state_dict where the key "data" is self.src_sample_state.
            "set": "src",  # The value of the key "set" is "src".
            "type": "set-state",  # The value of the key "type" is "set-state".
        }

        with open(
            idx_state_history_path / "src_state.json", "w"
        ) as outfile:  # Open the file "src_state.json" as a write file and assign the file object to outfile.
            json.dump(
                src_full_state_dict, outfile
            )  # Write src_full_state_dict in json format to outfile

        dst_full_state_dict = {
            "data": self.dst_sample_state,  # Assign self.dst_sample_state to key "data".
            "set": "dst",  # Assign the string "dst" to the key "set".
            "type": "set-state",  # Assign the string "set-state" to the key "type".
        }
        with open(idx_state_history_path / "dst_state.json", "w") as outfile:
            json.dump(dst_full_state_dict, outfile)

        print("fulfillment")

    def _get_formatted_image(self, raw_output):
        # Converts the raw output format to the specified data format and crops it so that the value is between 0 and 1
        formatted = np.clip(
            nn.to_data_format(raw_output, "NHWC", self.model_data_format), 0.0, 1.0
        )
        # Compress the number of dimensions of the first dimension to get the final output image
        formatted = np.squeeze(formatted, 0)

        return formatted

    # Export src dst Loss Map Logs=========================================================================
    def _processor(self, samples_tuple):
        """
        Processing of input sample tuples

        Args:
            samples_tuple: A tuple containing two samples, samples_tuple[0] is the source sample and samples_tuple[1] is the target sample

        Returns:
            None

        """
        if samples_tuple[0] != 0:
            src_sample_bgr, src_sample_mask, src_sample_mask_em = prepare_sample(
                samples_tuple[0], self.options, self.resolution, self.face_type
            )
        else:
            src_sample_bgr, src_sample_mask, src_sample_mask_em = (
                self._dummy_input,
                self._dummy_mask,
                self._dummy_mask,
            )
        if samples_tuple[1] != 0:
            dst_sample_bgr, dst_sample_mask, dst_sample_mask_em = prepare_sample(
                samples_tuple[1], self.options, self.resolution, self.face_type
            )
        else:
            dst_sample_bgr, dst_sample_mask, dst_sample_mask_em = (
                self._dummy_input,
                self._dummy_mask,
                self._dummy_mask,
            )

        (
            src_loss,  # Source Image Loss
            dst_loss,  # Target image loss
            pred_src_src,  # Source Image Predicted Source Image
            pred_src_srcm,  # Source image blending for source image prediction
            pred_dst_dst,  # Target image prediction of target image
            pred_dst_dstm,  # Target Image Intensity Change for Target Image Prediction
            pred_src_dst,  # Target image predicted from source image
            pred_src_dstm,  # Intensity change of the target image predicted by the source image
        ) = self.get_src_dst_information(  # Call the get_src_dst_information method to get information about the source and target images
            data_format_change(
                src_sample_bgr
            ),  # Call the data_format_change method to change the order of the source image color channels
            data_format_change(
                src_sample_bgr
            ),  # Call the data_format_change method to change the order of the source image color channels
            data_format_change(
                src_sample_mask
            ),  # Call the data_format_change method to change the channel order of the source image masks
            data_format_change(
                src_sample_mask_em
            ),  # Call the data_format_change method to change the channel order of the source image mask energy
            data_format_change(
                dst_sample_bgr
            ),  # Call the data_format_change method to change the order of the target image color channels
            data_format_change(
                dst_sample_bgr
            ),  # Call the data_format_change method to change the order of the target image color channels
            data_format_change(
                dst_sample_mask
            ),  # Call the data_format_change method to change the channel order of the target image mask
            data_format_change(
                dst_sample_mask_em
            ),  # Call the data_format_change method to change the channel order of the target image mask energy
        )

        if samples_tuple[0] != 0:
            # Get the sample file name of the .stem
            src_file_name = Path(samples_tuple[0].filename).stem

            # Save the processed image as a jpg file
            cv2_imwrite(
                self.src_state_path / f"{src_file_name}_output.jpg",
                self._get_formatted_image(pred_src_src) * 255,
                [int(cv2.IMWRITE_JPEG_QUALITY), 100],
            )  # output

            src_data = {
                # Convert the first element of src_loss to a floating point number and assign it to the loss key
                "loss": float(src_loss[0]),
                # Add suffix .jpg to src_file_name and assign to input key
                "input": f"{src_file_name}.jpg",
                # Add suffix _output.jpg to src_file_name and assign to output key
                "output": f"{src_file_name}_output.jpg",
            }
            # Add src_data to self.src_sample_state list
            self.src_sample_state.append(src_data)

        if samples_tuple[1] != 0:
            # Get the filename and remove the extension
            dst_file_name = Path(samples_tuple[1].filename).stem

            # Save predictions as images
            cv2_imwrite(
                self.dst_state_path / f"{dst_file_name}_output.jpg",
                self._get_formatted_image(pred_dst_dst) * 255,
                [int(cv2.IMWRITE_JPEG_QUALITY), 100],
            )  # Output Image
            cv2_imwrite(
                self.dst_state_path / f"{dst_file_name}_swap.jpg",
                self._get_formatted_image(pred_src_dst) * 255,
                [int(cv2.IMWRITE_JPEG_QUALITY), 100],
            )  # Exchange of pictures

            # Constructing the data dictionary that holds the results
            dst_data = {
                "loss": float(dst_loss[0]),
                "input": f"{dst_file_name}.jpg",
                "output": f"{dst_file_name}_output.jpg",
                "swap": f"{dst_file_name}_swap.jpg",
            }
            # Add result data to the list of target sample states
            self.dst_sample_state.append(dst_data)

            # Delete self.dst_state_path folder
            if os.path.exists(self.dst_state_path):
                shutil.rmtree(self.dst_state_path)

            # Delete self.src_state_path folder
            if os.path.exists(self.src_state_path):
                shutil.rmtree(self.src_state_path)


Model = MEModel
