"""Modified from https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
"""
#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import gc
import json
import logging
import math
import os
import pickle
import random
import shutil
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import accelerate
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms.functional as TF
import transformers
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import DDIMScheduler, FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (EMAModel,
                                      compute_density_for_timestep_sampling,
                                      compute_loss_weighting_for_sd3)
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module
from einops import rearrange
from omegaconf import OmegaConf
from packaging import version
from PIL import Image
from torch.utils.data import RandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers.utils import ContextManagers

import datasets

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.data.bucket_sampler import (ASPECT_RATIO_512,
                                            ASPECT_RATIO_RANDOM_CROP_512,
                                            ASPECT_RATIO_RANDOM_CROP_PROB,
                                            AspectRatioBatchImageVideoSampler,
                                            RandomSampler, get_closest_ratio)
from videox_fun.data.dataset_image_video import (ImageVideoDataset,
                                                 ImageVideoSampler,
                                                 get_random_mask,
                                                 process_pose_file,
                                                 process_pose_params)
from videox_fun.data.dataset_image_video_actionmap import ImageVideoControlDataset
from videox_fun.models import (AutoencoderKLWan, AutoencoderKLWan3_8,
                               CLIPModel, Wan2_2Transformer3DModel,
                               WanT5EncoderModel)
from videox_fun.pipeline import Wan2_2FunControlPipeline
from videox_fun.research.object_temporal_attention_bias import ObjectTemporalAttentionBiasController
from videox_fun.training.method1_focused_loss import (METHOD1_LOSS_VARIANTS,
                                                      method1_focused_flow_loss)
from videox_fun.utils.discrete_sampler import DiscreteSampling
from videox_fun.utils.utils import (calculate_dimensions, get_image_latent,
                                    get_image_to_video_latent,
                                    get_video_to_video_latent,
                                    save_videos_grid)

if is_wandb_available():
    import wandb


def filter_kwargs(cls, kwargs):
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs

def linear_decay(initial_value, final_value, total_steps, current_step):
    if current_step >= total_steps:
        return final_value
    current_step = max(0, current_step)
    step_size = (final_value - initial_value) / total_steps
    current_value = initial_value + step_size * current_step
    return current_value

def generate_timestep_with_lognorm(low, high, shape, device="cpu", generator=None):
    u = torch.normal(mean=0.0, std=1.0, size=shape, device=device, generator=generator)
    t = 1 / (1 + torch.exp(-u)) * (high - low) + low
    return torch.clip(t.to(torch.int32), low, high - 1)

def resize_mask(mask, latent, process_first_frame_only=True):
    latent_size = latent.size()
    batch_size, channels, num_frames, height, width = mask.shape

    if process_first_frame_only:
        target_size = list(latent_size[2:])
        target_size[0] = 1
        first_frame_resized = F.interpolate(
            mask[:, :, 0:1, :, :],
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
        
        target_size = list(latent_size[2:])
        target_size[0] = target_size[0] - 1
        if target_size[0] != 0:
            remaining_frames_resized = F.interpolate(
                mask[:, :, 1:, :, :],
                size=target_size,
                mode='trilinear',
                align_corners=False
            )
            resized_mask = torch.cat([first_frame_resized, remaining_frames_resized], dim=2)
        else:
            resized_mask = first_frame_resized
    else:
        target_size = list(latent_size[2:])
        resized_mask = F.interpolate(
            mask,
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
    return resized_mask

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

logger = get_logger(__name__, log_level="INFO")

def log_validation(vae, text_encoder, tokenizer, transformer3d, args, config, accelerator, weight_dtype, global_step):
    try:
        is_deepspeed = type(transformer3d).__name__ == 'DeepSpeedEngine'
        if is_deepspeed:
            origin_config = transformer3d.config
            transformer3d.config = accelerator.unwrap_model(transformer3d).config
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=accelerator.device):
            logger.info("Running validation... ")
            scheduler = FlowMatchEulerDiscreteScheduler(
                **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
            )
            if args.boundary_type == "full":
                transformer3d_1 = accelerator.unwrap_model(transformer3d) if type(transformer3d).__name__ == 'DistributedDataParallel' else transformer3d
                transformer3d_2 = None
            else:
                if args.boundary_type == "low":
                    transformer3d_1 = accelerator.unwrap_model(transformer3d) if type(transformer3d).__name__ == 'DistributedDataParallel' else transformer3d
                    
                    sub_path = config['transformer_additional_kwargs'].get('transformer_high_noise_model_subpath', 'transformer')
                    transformer3d_2 = Wan2_2Transformer3DModel.from_pretrained(
                        os.path.join(args.pretrained_model_name_or_path, sub_path),
                        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
                    ).to(weight_dtype)
                    
                else:
                    sub_path = config['transformer_additional_kwargs'].get('transformer_low_noise_model_subpath', 'transformer')
                    transformer3d_1 = Wan2_2Transformer3DModel.from_pretrained(
                        os.path.join(args.pretrained_model_name_or_path, sub_path),
                        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
                    ).to(weight_dtype)

                    transformer3d_2 = accelerator.unwrap_model(transformer3d) if type(transformer3d).__name__ == 'DistributedDataParallel' else transformer3d

            pipeline = Wan2_2FunControlPipeline(
                vae=vae, 
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                transformer=transformer3d_1,
                transformer_2=transformer3d_2,
                scheduler=scheduler,
            )
            pipeline = pipeline.to(accelerator.device)

            if args.seed is None:
                generator = None
            else:
                rank_seed = args.seed + accelerator.process_index
                generator = torch.Generator(device=accelerator.device).manual_seed(rank_seed)
                logger.info(f"Rank {accelerator.process_index} using seed: {rank_seed}")

            for i in range(len(args.validation_prompts)):
                import cv2
                cap = cv2.VideoCapture(args.validation_paths[i])
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()

                width, height = calculate_dimensions(args.image_sample_size * args.image_sample_size,  width / height)
                video_length = int((args.video_sample_n_frames - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if args.video_sample_n_frames != 1 else 1
                
                inpaint_video, inpaint_video_mask, clip_image = get_image_to_video_latent(None, None, video_length=video_length, sample_size=[height, width])
                input_video, input_video_mask, ref_image, clip_image = get_video_to_video_latent(args.validation_paths[i], video_length=video_length, sample_size=[height, width])
                sample = pipeline(
                    args.validation_prompts[i], 
                    num_frames = video_length,
                    negative_prompt = "bad detailed",
                    height      = height,
                    width       = width,
                    generator   = generator,

                    control_video   = input_video,
                    video           = inpaint_video,
                    mask_video      = inpaint_video_mask,
                    num_inference_steps = 25,
                    guidance_scale      = 4.5,
                    boundary            = config['transformer_additional_kwargs'].get('boundary', 0.900)
                ).videos
                os.makedirs(os.path.join(args.logging_dir, "sample"), exist_ok=True)
                save_videos_grid(
                    sample, 
                    os.path.join(
                        args.logging_dir, 
                        f"sample/sample-{global_step}-rank{accelerator.process_index}-image-{i}.gif"
                    )
                )

            del pipeline
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            vae.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
            if not args.enable_text_encoder_in_dataloader:
                text_encoder.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
        if is_deepspeed:
            transformer3d.config = origin_config
    except Exception as e:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print(f"Eval error on rank {accelerator.process_index} with info {e}")
        vae.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
        if not args.enable_text_encoder_in_dataloader:
            text_encoder.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. "
        ),
    )
    parser.add_argument(
        "--train_data_meta",
        type=str,
        default=None,
        help=(
            "A csv containing the training data. "
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--validation_paths",
        type=str,
        default=None,
        nargs="+",
        help=("A set of control videos evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--use_came",
        action="store_true",
        help="whether to use came",
    )
    parser.add_argument(
        "--multi_stream",
        action="store_true",
        help="whether to use cuda multi-stream",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--vae_mini_batch", type=int, default=32, help="mini batch size for vae."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_model_info", action="store_true", help="Whether or not to report more info about model (such as norm, grad)."
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=2000,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    
    parser.add_argument(
        "--snr_loss", action="store_true", help="Whether or not to use snr_loss."
    )
    parser.add_argument(
        "--uniform_sampling", action="store_true", help="Whether or not to use uniform_sampling."
    )
    parser.add_argument(
        "--enable_text_encoder_in_dataloader", action="store_true", help="Whether or not to use text encoder in dataloader."
    )
    parser.add_argument(
        "--enable_bucket", action="store_true", help="Whether enable bucket sample in datasets."
    )
    parser.add_argument(
        "--random_ratio_crop", action="store_true", help="Whether enable random ratio crop sample in datasets."
    )
    parser.add_argument(
        "--random_frame_crop", action="store_true", help="Whether enable random frame crop sample in datasets."
    )
    parser.add_argument(
        "--random_hw_adapt", action="store_true", help="Whether enable random adapt height and width in datasets."
    )
    parser.add_argument(
        "--training_with_video_token_length", action="store_true", help="The training stage of the model in training.",
    )
    parser.add_argument(
        "--auto_tile_batch_size", action="store_true", help="Whether to auto tile batch size.",
    )
    parser.add_argument(
        "--motion_sub_loss", action="store_true", help="Whether enable motion sub loss."
    )
    parser.add_argument(
        "--motion_sub_loss_ratio", type=float, default=0.25, help="The ratio of motion sub loss."
    )
    parser.add_argument(
        "--train_sampling_steps",
        type=int,
        default=1000,
        help="Run train_sampling_steps.",
    )
    parser.add_argument(
        "--keep_all_node_same_token_length",
        action="store_true", 
        help="Reference of the length token.",
    )
    parser.add_argument(
        "--token_sample_size",
        type=int,
        default=512,
        help="Sample size of the token.",
    )
    parser.add_argument(
        "--video_sample_size",
        type=int,
        default=512,
        help="Sample size of the video.",
    )
    parser.add_argument(
        "--image_sample_size",
        type=int,
        default=512,
        help="Sample size of the image.",
    )
    parser.add_argument(
        "--fix_sample_size", 
        nargs=2, type=int, default=None,
        help="Fix Sample size [height, width] when using bucket and collate_fn."
    )
    parser.add_argument(
        "--video_sample_stride",
        type=int,
        default=4,
        help="Sample stride of the video.",
    )
    parser.add_argument(
        "--video_sample_n_frames",
        type=int,
        default=17,
        help="Num frame of video.",
    )
    parser.add_argument(
        "--video_repeat",
        type=int,
        default=0,
        help="Num of repeat video.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help=(
            "The config of the model in training."
        ),
    )
    parser.add_argument(
        "--transformer_path",
        type=str,
        default=None,
        help=("If you want to load the weight from other transformers, input its path."),
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help=("If you want to load the weight from other vaes, input its path."),
    )

    parser.add_argument(
        '--trainable_modules', 
        nargs='+', 
        help='Enter a list of trainable modules'
    )
    parser.add_argument(
        '--trainable_modules_low_learning_rate', 
        nargs='+', 
        default=[],
        help='Enter a list of trainable modules with lower learning rate'
    )
    parser.add_argument(
        "--moe_mode",
        type=str,
        default="camera_kinematic",
        choices=["camera_kinematic", "control_expert"],
        help="MoE injection mode for transformer FFNs.",
    )
    parser.add_argument(
        "--disable_moe",
        action="store_true",
        help="Disable external MoE injection while keeping action map conditioning logic enabled.",
    )
    parser.add_argument(
        "--moe_all_blocks",
        action="store_true",
        help="Inject MoE into all transformer blocks instead of the default subset for the selected mode.",
    )
    parser.add_argument(
        "--moe_route_temperature",
        type=float,
        default=1.0,
        help="Softmax temperature used by the explicit control expert MoE routing weights.",
    )
    parser.add_argument(
        "--camera_moe_root",
        type=str,
        default=os.environ.get("CAMERA_MOE_ROOT", ""),
        help="Directory containing camera_moe_core.py for external MoE injection.",
    )
    parser.add_argument(
        '--tokenizer_max_length', 
        type=int,
        default=512,
        help='Max length of tokenizer'
    )
    parser.add_argument(
        "--use_deepspeed", action="store_true", help="Whether or not to use deepspeed."
    )
    parser.add_argument(
        "--use_fsdp", action="store_true", help="Whether or not to use fsdp."
    )
    parser.add_argument(
        "--low_vram", action="store_true", help="Whether enable low_vram mode."
    )
    parser.add_argument(
        "--freeze_control_adapter", action="store_true",
        help="Freeze the control_adapter (camera) module so it is not trained."
    )
    parser.add_argument(
        "--enable_arm_info", action="store_true",
        help="Enable robotic arm action conditioning from dataset metadata."
    )
    parser.add_argument(
        "--enable_action_map_info", action="store_true",
        help="Enable action map / pose-video conditioning from dataset metadata."
    )
    parser.add_argument(
        "--enable_object_temporal_attention_bias",
        action="store_true",
        help="Keep ordinary diffusion MSE training, but add object-token temporal cross-attention logit bias.",
    )
    parser.add_argument(
        "--object_temporal_attention_layers",
        type=str,
        default="all",
        help="Cross-attention layers patched for object-temporal logit bias, e.g. all, last4, or 8-17.",
    )
    parser.add_argument(
        "--object_temporal_attention_source_frames",
        type=int,
        default=81,
        help="Frame count used by metadata window_relative_frame_range.",
    )
    parser.add_argument(
        "--object_temporal_attention_require_token_match",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Raise if enabled but no object phrase matches prompt tokens in a batch.",
    )
    parser.add_argument(
        "--arm_action_stat_path",
        type=str,
        default=None,
        help="Path to robotic arm action normalization statistics."
    )
    parser.add_argument(
        "--arm_action_key",
        type=str,
        default="state",
        help="Key used to read robotic arm action values from JSON annotations."
    )
    parser.add_argument(
        "--arm_action_dim",
        type=int,
        default=14,
        help="Dimension of each robotic arm action vector."
    )
    parser.add_argument(
        "--arm_action_num_frames",
        type=int,
        default=None,
        help="Fixed number of frames used by the robotic arm action embedder."
    )
    parser.add_argument(
        "--zero_init_arm_action_output",
        action="store_true",
        help="Zero-initialize the arm adapter outputs while retaining trainable hidden layers.",
    )
    parser.add_argument(
        "--enable_method1_focused_loss",
        action="store_true",
        help="Enable Method1 action-effect focused flow-matching loss.",
    )
    parser.add_argument(
        "--method1_loss_variant",
        type=str,
        default="CAER",
        choices=METHOD1_LOSS_VARIANTS,
        help="Method1 weighting variant. LIBERO CAER uses rho=max(S/mean_future(S), 1).",
    )
    parser.add_argument(
        "--method1_action_dropout_prob",
        type=float,
        default=0.10,
        help="Per-sample probability of dropping the arm condition in the main training forward.",
    )
    parser.add_argument(
        "--method1_tau_s",
        type=float,
        default=0.50,
        help="Physical scheduler sigma used for the fixed action-effect diagnostic pair.",
    )
    parser.add_argument(
        "--method1_eps",
        type=float,
        default=1e-6,
        help="Numerical epsilon for per-sample Method1 normalization.",
    )
    parser.add_argument(
        "--method1_mse_threshold",
        type=float,
        default=0.0,
        help="Optional absolute residual cutoff; <=0 keeps the exact squared-error objective.",
    )
    parser.add_argument(
        "--method1_log_stats",
        action="store_true",
        help="Print lightweight Method1 rho and action-dropout statistics.",
    )
    parser.add_argument(
        "--boundary_type",
        type=str,
        default="low",
        help=(
            'The format of training data. Support `"low"` and `"high"`'
        ),
    )
    parser.add_argument(
        "--abnormal_norm_clip_start",
        type=int,
        default=1000,
        help=(
            'When do we start doing additional processing on abnormal gradients. '
        ),
    )
    parser.add_argument(
        "--initial_grad_norm_ratio",
        type=int,
        default=5,
        help=(
            'The initial gradient is relative to the multiple of the max_grad_norm. '
        ),
    )
    parser.add_argument(
        "--train_mode",
        type=str,
        default="control",
        help=(
            'The format of training data. Support `"control"`'
            ' (default), `"control_ref"`, `"control_camera_ref"`.'
        ),
    )
    parser.add_argument(
        "--control_ref_image",
        type=str,
        default="first_frame",
        help=(
            'The format of training data. Support `"first_frame"`'
            ' (default), `"random"`.'
        ),
    )
    parser.add_argument(
        "--add_full_ref_image_in_self_attention",
        action="store_true",
        help=(
            'Whether enable add full ref image in self attention.'
        ),
    )
    parser.add_argument(
        "--add_inpaint_info",
        action="store_true",
        help=(
            'Whether enable add inpaint info in self attention.'
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--benchmark_timing_path",
        type=str,
        default=None,
        help="Optional JSONL path for per-update benchmark timing.",
    )
    parser.add_argument(
        "--skip_sanity_check",
        action="store_true",
        help="Skip first-batch GIF/PNG sanity check generation.",
    )
    parser.add_argument(
        "--skip_final_checkpoint",
        action="store_true",
        help="Skip the final checkpoint save at training end.",
    )

    args = parser.parse_args()
    if args.enable_method1_focused_loss and not args.enable_arm_info:
        parser.error("--enable_method1_focused_loss requires --enable_arm_info")
    if args.zero_init_arm_action_output and not args.enable_arm_info:
        parser.error("--zero_init_arm_action_output requires --enable_arm_info")
    if args.zero_init_arm_action_output and args.transformer_path is not None:
        parser.error(
            "--zero_init_arm_action_output is only for TI2V initialization without --transformer_path"
        )
    if not 0.0 <= args.method1_action_dropout_prob <= 1.0:
        parser.error("--method1_action_dropout_prob must be in [0, 1]")
    if not 0.0 <= args.method1_tau_s <= 1.0:
        parser.error("--method1_tau_s must be in [0, 1]")
    if args.method1_eps <= 0:
        parser.error("--method1_eps must be positive")
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision
    if args.enable_object_temporal_attention_bias and args.enable_text_encoder_in_dataloader:
        raise ValueError("--enable_object_temporal_attention_bias requires tokenizer input_ids in the train loop; disable --enable_text_encoder_in_dataloader.")

    return args


def adapt_action_map_moe_state_dict(state_dict, model_state_dict):
    adapted_state_dict = dict(state_dict)

    for key, value in list(adapted_state_dict.items()):
        if not key.endswith(".ffn.control_moe.router.weight") or key not in model_state_dict:
            continue
        target_value = model_state_dict[key]
        if value.shape == target_value.shape:
            continue
        if value.ndim == 2 and value.shape[0] == 3 and target_value.ndim == 2 and target_value.shape[0] == 4 and value.shape[1] == target_value.shape[1]:
            new_value = torch.zeros_like(target_value, device=value.device, dtype=value.dtype)
            new_value[:3] = value
            adapted_state_dict[key] = new_value

    for key, target_value in model_state_dict.items():
        if ".ffn.control_moe.action_map_expert." not in key or key in adapted_state_dict:
            continue
        source_key = key.replace(".ffn.control_moe.action_map_expert.", ".ffn.control_moe.shared_expert.")
        source_value = adapted_state_dict.get(source_key, None)
        if source_value is not None and source_value.shape == target_value.shape:
            adapted_state_dict[key] = source_value

    return adapted_state_dict


def main():
    args = parse_args()

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    args.logging_dir = logging_dir

    config = OmegaConf.load(args.config_path)
    if args.arm_action_num_frames is None:
        args.arm_action_num_frames = args.video_sample_n_frames

    if "transformer_additional_kwargs" not in config or config["transformer_additional_kwargs"] is None:
        config["transformer_additional_kwargs"] = OmegaConf.create()

    if args.enable_arm_info:
        config["transformer_additional_kwargs"]["add_arm_action_embedder"] = True
        config["transformer_additional_kwargs"]["arm_action_dim"] = args.arm_action_dim
        config["transformer_additional_kwargs"]["arm_action_num_frames"] = args.arm_action_num_frames

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    fsdp_plugin = None
    if args.use_fsdp:
        fsdp_plugin = FullyShardedDataParallelPlugin(
            sharding_strategy="FULL_SHARD",
            backward_prefetch="BACKWARD_PRE",
            auto_wrap_policy="transformer_based_wrap",
            transformer_cls_names_to_wrap=["WanAttentionBlock"],
            state_dict_type="SHARDED_STATE_DICT",
            cpu_ram_efficient_loading=False,
            use_orig_params=True,
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        fsdp_plugin=fsdp_plugin,
    )

    deepspeed_plugin = accelerator.state.deepspeed_plugin if hasattr(accelerator.state, "deepspeed_plugin") else None
    fsdp_plugin = accelerator.state.fsdp_plugin if hasattr(accelerator.state, "fsdp_plugin") else None
    if deepspeed_plugin is not None:
        zero_stage = int(deepspeed_plugin.zero_stage)
        fsdp_stage = 0
        print(f"Using DeepSpeed Zero stage: {zero_stage}")

        args.use_deepspeed = True
        if zero_stage == 3:
            print(f"Auto set save_state to True because zero_stage == 3")
            args.save_state = True
    elif fsdp_plugin is not None:
        from torch.distributed.fsdp import ShardingStrategy
        zero_stage = 0
        if fsdp_plugin.sharding_strategy is ShardingStrategy.FULL_SHARD:
            fsdp_stage = 3
        elif fsdp_plugin.sharding_strategy is None: # The fsdp_plugin.sharding_strategy is None in FSDP 2.
            fsdp_stage = 3
        elif fsdp_plugin.sharding_strategy is ShardingStrategy.SHARD_GRAD_OP:
            fsdp_stage = 2
        else:
            fsdp_stage = 0
        print(f"Using FSDP stage: {fsdp_stage}")

        args.use_fsdp = True
        if fsdp_stage == 3:
            print(f"Auto set save_state to True because fsdp_stage == 3")
            args.save_state = True
    else:
        zero_stage = 0
        fsdp_stage = 0
        print("DeepSpeed is not enabled.")

    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=logging_dir)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        rng = np.random.default_rng(np.random.PCG64(args.seed + accelerator.process_index))
        torch_rng = torch.Generator(accelerator.device).manual_seed(args.seed + accelerator.process_index)
    else:
        rng = None
        torch_rng = None
    method1_torch_rng = None
    if args.enable_method1_focused_loss:
        method1_rng_seed = (
            (args.seed if args.seed is not None else torch.initial_seed())
            + accelerator.process_index
            + 1_000_003
        )
        method1_torch_rng = torch.Generator(accelerator.device).manual_seed(
            method1_rng_seed
        )
    index_rng = np.random.default_rng(np.random.PCG64(43))
    print(f"Init rng with seed {args.seed + accelerator.process_index}. Process_index is {accelerator.process_index}")

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        if args.logging_dir is not None:
            os.makedirs(args.logging_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer3d) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Load scheduler, tokenizer and models.
    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )

    # Get Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        # Get Text encoder
        text_encoder = WanT5EncoderModel.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
            additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
        text_encoder = text_encoder.eval()
        # Get Vae
        Chosen_AutoencoderKL = {
            "AutoencoderKLWan": AutoencoderKLWan,
            "AutoencoderKLWan3_8": AutoencoderKLWan3_8
        }[config['vae_kwargs'].get('vae_type', 'AutoencoderKLWan')]
        vae = Chosen_AutoencoderKL.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, config['vae_kwargs'].get('vae_subpath', 'vae')),
            additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
        )
        vae.eval()
            
    # Get Transformer
    if args.boundary_type == "low" or args.boundary_type == "full":
        sub_path = config['transformer_additional_kwargs'].get('transformer_low_noise_model_subpath', 'transformer')
    else:
        sub_path = config['transformer_additional_kwargs'].get('transformer_high_noise_model_subpath', 'transformer')
    transformer3d = Wan2_2Transformer3DModel.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, sub_path),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
    ).to(weight_dtype)

    if not args.disable_moe:
        # --- INJECT MOE HERE ---
        import sys
        if args.camera_moe_root and args.camera_moe_root not in sys.path:
            sys.path.append(args.camera_moe_root)
        from camera_moe_core import inject_moe_into_wan_model
        moe_target_block_indices = list(range(len(transformer3d.blocks))) if args.moe_all_blocks else None
        transformer3d = inject_moe_into_wan_model(
            transformer3d,
            target_block_indices=moe_target_block_indices,
            moe_mode=args.moe_mode,
            route_temperature=args.moe_route_temperature,
        )
        # -----------------------

    # Freeze vae and text_encoder and set transformer3d to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    transformer3d.requires_grad_(False)

    if args.transformer_path is not None:
        print(f"From checkpoint: {args.transformer_path}")
        if args.transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(args.transformer_path)
        else:
            state_dict = torch.load(args.transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        state_dict = adapt_action_map_moe_state_dict(state_dict, transformer3d.state_dict())

        m, u = transformer3d.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    if args.zero_init_arm_action_output:
        transformer3d.zero_init_arm_action_output()
        zero_init_parameters = {
            "arm_action_embedder.fc2.weight": transformer3d.arm_action_embedder.fc2.weight,
            "arm_action_embedder.fc2.bias": transformer3d.arm_action_embedder.fc2.bias,
            "arm_action_embedder_proj.fc2.weight": transformer3d.arm_action_embedder_proj.fc2.weight,
            "arm_action_embedder_proj.fc2.bias": transformer3d.arm_action_embedder_proj.fc2.bias,
            "arm_condition_mask_emb.weight": transformer3d.arm_condition_mask_emb.weight,
            "arm_condition_mask_emb_proj.weight": transformer3d.arm_condition_mask_emb_proj.weight,
        }
        max_abs = max(
            parameter.detach().float().abs().max().item()
            for parameter in zero_init_parameters.values()
            if parameter is not None
        )
        if max_abs != 0.0:
            raise RuntimeError(
                f"arm action zero initialization failed: max_abs={max_abs}"
            )
        accelerator.print(
            "arm_action_zero_init passed=1 mode=zero_output_adapter max_abs=0.0"
        )

    if args.vae_path is not None:
        print(f"From checkpoint: {args.vae_path}")
        if args.vae_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(args.vae_path)
        else:
            state_dict = torch.load(args.vae_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = vae.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
    
    # A good trainable modules is showed below now.
    # For 3D Patch: trainable_modules = ['ff.net', 'pos_embed', 'attn2', 'proj_out', 'timepositionalencoding', 'h_position', 'w_position']
    # For 2D Patch: trainable_modules = ['ff.net', 'attn2', 'timepositionalencoding', 'h_position', 'w_position']
    transformer3d.train()
    if accelerator.is_main_process:
        accelerator.print(
            f"Trainable modules '{args.trainable_modules}'."
        )
    for name, param in transformer3d.named_parameters():
        for trainable_module_name in args.trainable_modules + args.trainable_modules_low_learning_rate:
            if trainable_module_name in name:
                param.requires_grad = True
                break

    if args.freeze_control_adapter:
        frozen_count = 0
        for name, param in transformer3d.named_parameters():
            if "control_adapter" in name:
                param.requires_grad = False
                frozen_count += 1
        if accelerator.is_main_process:
            accelerator.print(f"Froze {frozen_count} control_adapter parameters.")

    # Create EMA for the transformer3d.
    if args.use_ema:
        if zero_stage == 3:
            raise NotImplementedError("FSDP does not support EMA.")

        ema_transformer3d = Wan2_2Transformer3DModel.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
            transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        ).to(weight_dtype)

        ema_transformer3d = EMAModel(ema_transformer3d.parameters(), model_cls=Wan2_2Transformer3DModel, model_config=ema_transformer3d.config)

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        if fsdp_stage != 0 or zero_stage == 3:
            def save_model_hook(models, weights, output_dir):
                accelerate_state_dict = accelerator.get_state_dict(models[-1], unwrap=True)
                if accelerator.is_main_process:
                    from safetensors.torch import save_file

                    safetensor_save_path = os.path.join(output_dir, f"diffusion_pytorch_model.safetensors")
                    accelerate_state_dict = {k: v.to(dtype=weight_dtype) for k, v in accelerate_state_dict.items()}
                    save_file(accelerate_state_dict, safetensor_save_path, metadata={"format": "pt"})

                    with open(os.path.join(output_dir, "sampler_pos_start.pkl"), 'wb') as file:
                        pickle.dump([batch_sampler.sampler._pos_start, first_epoch], file)

            def load_model_hook(models, input_dir):
                pkl_path = os.path.join(input_dir, "sampler_pos_start.pkl")
                if os.path.exists(pkl_path):
                    with open(pkl_path, 'rb') as file:
                        loaded_number, _ = pickle.load(file)
                        batch_sampler.sampler._pos_start = max(loaded_number - args.dataloader_num_workers * accelerator.num_processes * 2, 0)
                    print(f"Load pkl from {pkl_path}. Get loaded_number = {loaded_number}.")
        else:
            # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
            def save_model_hook(models, weights, output_dir):
                if accelerator.is_main_process:
                    if args.use_ema:
                        ema_transformer3d.save_pretrained(os.path.join(output_dir, "transformer_ema"))

                    models[0].save_pretrained(os.path.join(output_dir, "transformer"))
                    if not args.use_deepspeed:
                        weights.pop()

                    with open(os.path.join(output_dir, "sampler_pos_start.pkl"), 'wb') as file:
                        pickle.dump([batch_sampler.sampler._pos_start, first_epoch], file)

            def load_model_hook(models, input_dir):
                if args.use_ema:
                    ema_path = os.path.join(input_dir, "transformer_ema")
                    _, ema_kwargs = Wan2_2Transformer3DModel.load_config(ema_path, return_unused_kwargs=True)
                    load_model = Wan2_2Transformer3DModel.from_pretrained(
                        input_dir, subfolder="transformer_ema",
                        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs'])
                    )
                    load_model = EMAModel(load_model.parameters(), model_cls=Wan2_2Transformer3DModel, model_config=load_model.config)
                    load_model.load_state_dict(ema_kwargs)

                    ema_transformer3d.load_state_dict(load_model.state_dict())
                    ema_transformer3d.to(accelerator.device)
                    del load_model

                for i in range(len(models)):
                    # pop models so that they are not loaded again
                    model = models.pop()

                    # load diffusers style into model
                    load_model = Wan2_2Transformer3DModel.from_pretrained(
                        input_dir, subfolder="transformer"
                    )
                    model.register_to_config(**load_model.config)

                    model.load_state_dict(load_model.state_dict())
                    del load_model

                pkl_path = os.path.join(input_dir, "sampler_pos_start.pkl")
                if os.path.exists(pkl_path):
                    with open(pkl_path, 'rb') as file:
                        loaded_number, _ = pickle.load(file)
                        batch_sampler.sampler._pos_start = max(loaded_number - args.dataloader_num_workers * accelerator.num_processes * 2, 0)
                    print(f"Load pkl from {pkl_path}. Get loaded_number = {loaded_number}.")

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        transformer3d.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    elif args.use_came:
        try:
            from came_pytorch import CAME
        except Exception:
            raise ImportError(
                "Please install came_pytorch to use CAME. You can do so by running `pip install came_pytorch`"
            )

        optimizer_cls = CAME
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params = list(filter(lambda p: p.requires_grad, transformer3d.parameters()))
    trainable_params_optim = [
        {'params': [], 'lr': args.learning_rate},
        {'params': [], 'lr': args.learning_rate / 2},
    ]
    debug_lr_assignment = os.environ.get("WAN22_DEBUG_LR_ASSIGNMENT", "0") == "1"
    in_already = []
    for name, param in transformer3d.named_parameters():
        if not param.requires_grad:
            continue
        if name in in_already:
            continue
        low_lr_flag = False
        for trainable_module_name in args.trainable_modules_low_learning_rate:
            if trainable_module_name in name:
                in_already.append(name)
                low_lr_flag = True
                trainable_params_optim[1]['params'].append(param)
                if debug_lr_assignment and accelerator.is_main_process:
                    print(f"Set {name} to lr : {args.learning_rate / 2}")
                break
        if low_lr_flag:
            continue
        for trainable_module_name in args.trainable_modules:
            if trainable_module_name in name:
                in_already.append(name)
                trainable_params_optim[0]['params'].append(param)
                if debug_lr_assignment and accelerator.is_main_process:
                    print(f"Set {name} to lr : {args.learning_rate}")
                break

    if args.use_came:
        optimizer = optimizer_cls(
            trainable_params_optim,
            lr=args.learning_rate,
            # weight_decay=args.adam_weight_decay,
            betas=(0.9, 0.999, 0.9999), 
            eps=(1e-30, 1e-16)
        )
    else:
        optimizer = optimizer_cls(
            trainable_params_optim,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    # Get the training dataset
    sample_n_frames_bucket_interval = vae.config.temporal_compression_ratio
    spatial_compression_ratio = vae.config.spatial_compression_ratio
    
    if args.fix_sample_size is not None and args.enable_bucket:
        args.video_sample_size = max(max(args.fix_sample_size), args.video_sample_size)
        args.image_sample_size = max(max(args.fix_sample_size), args.image_sample_size)
        args.training_with_video_token_length = False
        args.random_hw_adapt = False

    skip_unused_control_pixel_values = args.train_mode == "control_camera_ref"

    # Get the dataset
    train_dataset = ImageVideoControlDataset(
        args.train_data_meta, args.train_data_dir,
        video_sample_size=args.video_sample_size, video_sample_stride=args.video_sample_stride, video_sample_n_frames=args.video_sample_n_frames, 
        video_repeat=args.video_repeat, 
        image_sample_size=args.image_sample_size,
        enable_bucket=args.enable_bucket, 
        enable_camera_info=args.train_mode == "control_camera_ref",
        enable_arm_info=args.enable_arm_info,
        enable_action_map_info=args.enable_action_map_info,
        skip_control_pixel_values=skip_unused_control_pixel_values,
        arm_action_stat_path=args.arm_action_stat_path,
        arm_action_key=args.arm_action_key,
        arm_action_dim=args.arm_action_dim,
        arm_action_num_frames=args.arm_action_num_frames,
    )

    def _get_control_type(example):
        return example.get("control_type", "control")

    def _sample_clip_index(num_frames):
        if args.control_ref_image == "first_frame":
            return 0

        def _create_special_list(length):
            if length == 1:
                return [1.0]
            if length >= 2:
                first_element = 0.40
                remaining_sum = 1.0 - first_element
                other_elements_value = remaining_sum / (length - 1)
                special_list = [first_element] + [other_elements_value] * (length - 1)
                return special_list

        number_list_prob = np.array(_create_special_list(num_frames))
        return int(np.random.choice(list(range(num_frames)), p=number_list_prob))

    def _prepare_arm_action(example):
        local_arm_action = torch.zeros((args.arm_action_num_frames, args.arm_action_dim), dtype=torch.float32)
        local_arm_mask = 0.0
        arm_action_values = example.get("arm_action_values", None)
        control_type = _get_control_type(example)

        if not args.enable_arm_info or control_type != "arm" or arm_action_values is None:
            return local_arm_action, local_arm_mask

        local_arm_action = torch.as_tensor(arm_action_values, dtype=torch.float32)
        if local_arm_action.ndim == 1:
            local_arm_action = local_arm_action.unsqueeze(0)
        elif local_arm_action.ndim > 2:
            local_arm_action = local_arm_action.reshape(local_arm_action.shape[0], -1)

        if local_arm_action.size(0) == 0:
            return torch.zeros((args.arm_action_num_frames, args.arm_action_dim), dtype=torch.float32), 0.0

        if local_arm_action.size(0) > args.arm_action_num_frames:
            frame_index = torch.linspace(0, local_arm_action.size(0) - 1, args.arm_action_num_frames).long()
            local_arm_action = local_arm_action[frame_index]
        elif local_arm_action.size(0) < args.arm_action_num_frames:
            pad_size = args.arm_action_num_frames - local_arm_action.size(0)
            local_arm_action = torch.cat([local_arm_action, local_arm_action[-1:].repeat(pad_size, 1)], dim=0)

        if local_arm_action.size(1) > args.arm_action_dim:
            local_arm_action = local_arm_action[:, :args.arm_action_dim]
        elif local_arm_action.size(1) < args.arm_action_dim:
            local_arm_action = torch.cat(
                [
                    local_arm_action,
                    local_arm_action.new_zeros(local_arm_action.size(0), args.arm_action_dim - local_arm_action.size(1)),
                ],
                dim=1,
            )

        return local_arm_action, 1.0

    def collate_fn_no_bucket(examples):
        new_examples = {}
        new_examples["pixel_values"] = torch.stack([example["pixel_values"] for example in examples])
        if not skip_unused_control_pixel_values:
            new_examples["control_pixel_values"] = torch.stack([example["control_pixel_values"] for example in examples])
        if args.enable_action_map_info:
            action_map_flags = [
                _get_control_type(example) == "action_map" and example.get("action_map_pixel_values", None) is not None
                for example in examples
            ]
            new_examples["action_map_mask"] = torch.tensor(action_map_flags, dtype=torch.float32)
            new_examples["action_map_pixel_values"] = torch.stack([
                example["action_map_pixel_values"] if example.get("action_map_pixel_values", None) is not None else torch.zeros_like(example["pixel_values"])
                for example in examples
            ])
        new_examples["text"] = [example["text"] for example in examples]
        if args.enable_object_temporal_attention_bias:
            new_examples["object_temporal_attention"] = [
                example.get("object_temporal_attention", []) for example in examples
            ]
            new_examples["object_temporal_attention_config"] = [
                example.get("object_temporal_attention_config", {}) for example in examples
            ]
        new_examples["idx"] = torch.tensor([example["idx"] for example in examples])
        new_examples["data_type"] = [example["data_type"] for example in examples]
        new_examples["control_type"] = [_get_control_type(example) for example in examples]
    
        if args.train_mode == "control_camera_ref":
            use_camera_flags = [
                _get_control_type(example) == "camera" and example.get("control_camera_values", None) is not None
                for example in examples
            ]
            new_examples["control_camera_mask"] = torch.tensor(use_camera_flags, dtype=torch.float32)
            if any(use_camera_flags):
                control_camera_values = []
                for example, use_camera in zip(examples, use_camera_flags):
                    if use_camera:
                        control_camera_values.append(example["control_camera_values"])
                    else:
                        example_pixel_values = example["pixel_values"]
                        control_camera_values.append(
                            torch.zeros(
                                (example_pixel_values.size(0), 6, example_pixel_values.size(2), example_pixel_values.size(3)),
                                dtype=example_pixel_values.dtype,
                            )
                        )
                new_examples["control_camera_values"] = torch.stack(control_camera_values)
            else:
                new_examples["control_camera_values"] = None

        if args.enable_arm_info:
            arm_action_values = []
            arm_action_mask = []
            for example in examples:
                local_arm_action, local_arm_mask = _prepare_arm_action(example)
                arm_action_values.append(local_arm_action)
                arm_action_mask.append(local_arm_mask)
            new_examples["arm_action_values"] = torch.stack(arm_action_values)
            new_examples["arm_action_mask"] = torch.tensor(arm_action_mask, dtype=torch.float32)

        if args.train_mode != "control":
            new_examples["ref_pixel_values"] = []
            new_examples["clip_pixel_values"] = []
            new_examples["clip_idx"] = []
            if args.add_inpaint_info:
                new_examples["mask_pixel_values"] = []
                new_examples["mask"] = []

            for pixel_values in new_examples["pixel_values"]:
                clip_index = _sample_clip_index(len(pixel_values))
                new_examples["clip_idx"].append(clip_index)

                ref_pixel_values = pixel_values[clip_index].unsqueeze(0)
                new_examples["ref_pixel_values"].append(ref_pixel_values)

                clip_pixel_values = pixel_values[clip_index].permute(1, 2, 0).contiguous()
                clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
                new_examples["clip_pixel_values"].append(clip_pixel_values)

                if args.add_inpaint_info:
                    mask = get_random_mask(pixel_values.size())
                    mask_pixel_values = pixel_values * (1 - mask) 
                    # Wan 2.1 use 0 for masked pixels
                    # + torch.ones_like(new_examples["pixel_values"][-1]) * -1 * mask
                    new_examples["mask_pixel_values"].append(mask_pixel_values)
                    new_examples["mask"].append(mask)

        if args.enable_text_encoder_in_dataloader:
            prompt_ids = tokenizer(
                new_examples['text'], 
                max_length=args.tokenizer_max_length, 
                padding="max_length", 
                add_special_tokens=True, 
                truncation=True, 
                return_tensors="pt"
            )
            encoder_hidden_states = text_encoder(
                prompt_ids.input_ids
            )[0]
            new_examples['encoder_attention_mask'] = prompt_ids.attention_mask
            new_examples['encoder_hidden_states'] = encoder_hidden_states

        return new_examples

    def worker_init_fn(_seed):
        _seed = _seed * 256
        def _worker_init_fn(worker_id):
            if os.environ.get("WAN22_DEBUG_WORKER_INIT", "0") == "1":
                print(f"worker_init_fn with {_seed + worker_id}")
            np.random.seed(_seed + worker_id)
            random.seed(_seed + worker_id)
        return _worker_init_fn
    
    if args.enable_bucket:
        aspect_ratio_sample_size = {key : [x / 512 * args.video_sample_size for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
        batch_sampler_generator = torch.Generator().manual_seed(args.seed)
        batch_sampler = AspectRatioBatchImageVideoSampler(
            sampler=RandomSampler(train_dataset, generator=batch_sampler_generator), dataset=train_dataset.dataset, 
            batch_size=args.train_batch_size, train_folder = args.train_data_dir, drop_last=True,
            aspect_ratios=aspect_ratio_sample_size,
        )

        def collate_fn(examples):
            def get_length_to_frame_num(token_length):
                if args.image_sample_size > args.video_sample_size:
                    sample_sizes = list(range(args.video_sample_size, args.image_sample_size + 1, 128))

                    if sample_sizes[-1] != args.image_sample_size:
                        sample_sizes.append(args.image_sample_size)
                else:
                    sample_sizes = [args.image_sample_size]
                
                length_to_frame_num = {
                    sample_size: min(token_length / sample_size / sample_size, args.video_sample_n_frames) // sample_n_frames_bucket_interval * sample_n_frames_bucket_interval + 1 for sample_size in sample_sizes
                }

                return length_to_frame_num

            def get_random_downsample_ratio(sample_size, image_ratio=[],
                                            all_choices=False, rng=None):
                def _create_special_list(length):
                    if length == 1:
                        return [1.0]
                    if length >= 2:
                        first_element = 0.90
                        remaining_sum = 1.0 - first_element
                        other_elements_value = remaining_sum / (length - 1)
                        special_list = [first_element] + [other_elements_value] * (length - 1)
                        return special_list
                        
                if sample_size >= 1536:
                    number_list = [1, 1.25, 1.5, 2, 2.5, 3] + image_ratio 
                elif sample_size >= 1024:
                    number_list = [1, 1.25, 1.5, 2] + image_ratio
                elif sample_size >= 768:
                    number_list = [1, 1.25, 1.5] + image_ratio
                elif sample_size >= 512:
                    number_list = [1] + image_ratio
                else:
                    number_list = [1]

                if all_choices:
                    return number_list

                number_list_prob = np.array(_create_special_list(len(number_list)))
                if rng is None:
                    return np.random.choice(number_list, p = number_list_prob)
                else:
                    return rng.choice(number_list, p = number_list_prob)

            # Get token length
            target_token_length = args.video_sample_n_frames * args.token_sample_size * args.token_sample_size
            length_to_frame_num = get_length_to_frame_num(target_token_length)

            # Create new output
            new_examples                 = {}
            new_examples["target_token_length"] = target_token_length
            new_examples["pixel_values"] = []
            new_examples["text"]         = []
            new_examples["control_type"] = []
            # Used in Control Mode
            if not skip_unused_control_pixel_values:
                new_examples["control_pixel_values"] = []
            # Used in Control Ref Mode
            if args.train_mode != "control":
                new_examples["ref_pixel_values"] = []
                new_examples["clip_pixel_values"] = []
                new_examples["clip_idx"] = []
            # Used in Control Camera Ref Mode
            if args.train_mode == "control_camera_ref":
                new_examples["control_camera_values"] = []
                new_examples["control_camera_mask"] = []
            if args.enable_arm_info:
                new_examples["arm_action_values"] = []
                new_examples["arm_action_mask"] = []
            if args.enable_action_map_info:
                new_examples["action_map_pixel_values"] = []
                new_examples["action_map_mask"] = []
            if args.enable_object_temporal_attention_bias:
                new_examples["object_temporal_attention"] = []
                new_examples["object_temporal_attention_config"] = []
                
            # Used in Inpaint mode 
            if args.add_inpaint_info:
                new_examples["mask_pixel_values"] = []
                new_examples["mask"] = []
                new_examples["clip_pixel_values"] = []

            # Get downsample ratio in image and videos
            pixel_value     = examples[0]["pixel_values"]
            data_type       = examples[0]["data_type"]
            f, h, w, c      = np.shape(pixel_value)
            if data_type == 'image':
                random_downsample_ratio = 1 if not args.random_hw_adapt else get_random_downsample_ratio(args.image_sample_size, image_ratio=[args.image_sample_size / args.video_sample_size])

                aspect_ratio_sample_size = {key : [x / 512 * args.image_sample_size / random_downsample_ratio for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
                aspect_ratio_random_crop_sample_size = {key : [x / 512 * args.image_sample_size / random_downsample_ratio for x in ASPECT_RATIO_RANDOM_CROP_512[key]] for key in ASPECT_RATIO_RANDOM_CROP_512.keys()}
                
                batch_video_length = args.video_sample_n_frames + sample_n_frames_bucket_interval
            else:
                if args.random_hw_adapt:
                    if args.training_with_video_token_length:
                        local_min_size = np.min(np.array([np.mean(np.array([np.shape(example["pixel_values"])[1], np.shape(example["pixel_values"])[2]])) for example in examples]))

                        def get_random_downsample_probability(choice_list, token_sample_size):
                            length = len(choice_list)
                            if length == 1:
                                return [1.0]  # If there's only one element, it gets all the probability
                            
                            # Find the index of the closest value to token_sample_size
                            closest_index = min(range(length), key=lambda i: abs(choice_list[i] - token_sample_size))
                            
                            # Assign 50% to the closest index
                            first_element = 0.50
                            remaining_sum = 1.0 - first_element
                            
                            # Distribute the remaining 50% evenly among the other elements
                            other_elements_value = remaining_sum / (length - 1) if length > 1 else 0.0
                            
                            # Construct the probability distribution
                            probability_list = [other_elements_value] * length
                            probability_list[closest_index] = first_element
                            
                            return probability_list

                        choice_list = [length for length in list(length_to_frame_num.keys()) if length < local_min_size * 1.25]
                        if len(choice_list) == 0:
                            choice_list = list(length_to_frame_num.keys())
                        probabilities = get_random_downsample_probability(choice_list, args.token_sample_size)
                        local_video_sample_size = np.random.choice(choice_list, p=probabilities)

                        random_downsample_ratio = args.video_sample_size / local_video_sample_size
                        batch_video_length = length_to_frame_num[local_video_sample_size]
                    else:
                        random_downsample_ratio = get_random_downsample_ratio(args.video_sample_size)
                        batch_video_length = args.video_sample_n_frames + sample_n_frames_bucket_interval
                else:
                    random_downsample_ratio = 1
                    batch_video_length = args.video_sample_n_frames + sample_n_frames_bucket_interval

                aspect_ratio_sample_size = {key : [x / 512 * args.video_sample_size / random_downsample_ratio for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
                aspect_ratio_random_crop_sample_size = {key : [x / 512 * args.video_sample_size / random_downsample_ratio for x in ASPECT_RATIO_RANDOM_CROP_512[key]] for key in ASPECT_RATIO_RANDOM_CROP_512.keys()}

            if args.fix_sample_size is not None:
                fix_sample_size = [int(x / spatial_compression_ratio / 2) * spatial_compression_ratio * 2 for x in args.fix_sample_size]
            elif args.random_ratio_crop:
                if rng is None:
                    random_sample_size = aspect_ratio_random_crop_sample_size[
                        np.random.choice(list(aspect_ratio_random_crop_sample_size.keys()), p = ASPECT_RATIO_RANDOM_CROP_PROB)
                    ]
                else:
                    random_sample_size = aspect_ratio_random_crop_sample_size[
                        rng.choice(list(aspect_ratio_random_crop_sample_size.keys()), p = ASPECT_RATIO_RANDOM_CROP_PROB)
                    ]
                random_sample_size = [int(x / spatial_compression_ratio / 2) * spatial_compression_ratio * 2 for x in random_sample_size]
            else:
                closest_size, closest_ratio = get_closest_ratio(h, w, ratios=aspect_ratio_sample_size)
                closest_size = [int(x / spatial_compression_ratio / 2) * spatial_compression_ratio * 2 for x in closest_size]

            min_example_length = min(
                [example["pixel_values"].shape[0] for example in examples]
            )
            batch_video_length = int(min(batch_video_length, min_example_length))

            # Magvae needs the number of frames to be 4n + 1.
            batch_video_length = (batch_video_length - 1) // sample_n_frames_bucket_interval * sample_n_frames_bucket_interval + 1

            if batch_video_length <= 0:
                batch_video_length = 1

            use_camera_flags = None
            batch_has_camera = False
            if args.train_mode == "control_camera_ref":
                use_camera_flags = [
                    _get_control_type(example) == "camera" and example.get("control_camera_values", None) is not None
                    for example in examples
                ]
                batch_has_camera = any(use_camera_flags)
                
            for example_idx, example in enumerate(examples):
                # To 0~1
                pixel_values = torch.from_numpy(example["pixel_values"]).permute(0, 3, 1, 2).contiguous()
                pixel_values = pixel_values / 255.

                if not skip_unused_control_pixel_values:
                    control_pixel_values = torch.from_numpy(example["control_pixel_values"]).permute(0, 3, 1, 2).contiguous()
                    control_pixel_values = control_pixel_values / 255.
                if args.enable_action_map_info:
                    if example.get("action_map_pixel_values", None) is not None:
                        action_map_pixel_values = torch.from_numpy(example["action_map_pixel_values"]).permute(0, 3, 1, 2).contiguous()
                        action_map_pixel_values = action_map_pixel_values / 255.
                    else:
                        action_map_pixel_values = torch.zeros_like(pixel_values)

                if args.fix_sample_size is not None:
                    # Get adapt hw for resize
                    fix_sample_size = list(map(lambda x: int(x), fix_sample_size))
                    pose_resize_size = fix_sample_size
                    transform = transforms.Compose([
                        transforms.Resize(fix_sample_size, interpolation=transforms.InterpolationMode.BILINEAR),  # Image.BICUBIC
                        transforms.CenterCrop(fix_sample_size),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                    ])

                    transform_no_normalize = transforms.Compose([
                        transforms.Resize(fix_sample_size, interpolation=transforms.InterpolationMode.BILINEAR),  # Image.BICUBIC
                        transforms.CenterCrop(fix_sample_size),
                    ])
                elif args.random_ratio_crop:
                    # Get adapt hw for resize
                    b, c, h, w = pixel_values.size()
                    th, tw = random_sample_size
                    if th / tw > h / w:
                        nh = int(th)
                        nw = int(w / h * nh)
                    else:
                        nw = int(tw)
                        nh = int(h / w * nw)
                    pose_resize_size = [nh, nw]
                    
                    transform = transforms.Compose([
                        transforms.Resize([nh, nw]),
                        transforms.CenterCrop([int(x) for x in random_sample_size]),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                    ])
    
                    transform_no_normalize = transforms.Compose([
                        transforms.Resize([nh, nw]),
                        transforms.CenterCrop([int(x) for x in random_sample_size]),
                    ])
                else:
                    # Get adapt hw for resize
                    closest_size = list(map(lambda x: int(x), closest_size))
                    if closest_size[0] / h > closest_size[1] / w:
                        resize_size = closest_size[0], int(w * closest_size[0] / h)
                    else:
                        resize_size = int(h * closest_size[1] / w), closest_size[1]
                    pose_resize_size = resize_size
                    
                    transform = transforms.Compose([
                        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),  # Image.BICUBIC
                        transforms.CenterCrop(closest_size),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                    ])
    
                    transform_no_normalize = transforms.Compose([
                        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),  # Image.BICUBIC
                        transforms.CenterCrop(closest_size),
                    ])

                new_examples["pixel_values"].append(transform(pixel_values)[:batch_video_length])
                if not skip_unused_control_pixel_values:
                    new_examples["control_pixel_values"].append(transform(control_pixel_values))
                control_type = _get_control_type(example)
                new_examples["control_type"].append(control_type)
                if args.enable_action_map_info:
                    new_examples["action_map_pixel_values"].append(transform(action_map_pixel_values)[:batch_video_length])
                    new_examples["action_map_mask"].append(float(control_type == "action_map" and example.get("action_map_pixel_values", None) is not None))
            
                if args.train_mode == "control_camera_ref":
                    use_camera = use_camera_flags[example_idx]
                    new_examples["control_camera_mask"].append(float(use_camera))
                    if batch_has_camera:
                        if not use_camera:
                            example_pixel_values = new_examples["pixel_values"][-1]
                            control_camera_values_size = (
                                example_pixel_values.size()[0], 
                                6, 
                                example_pixel_values.size()[2], 
                                example_pixel_values.size()[3]
                            )
                            local_control_camera_values = torch.zeros(control_camera_values_size, dtype=example_pixel_values.dtype)
                            new_examples["control_camera_values"].append(local_control_camera_values)
                        else:
                            local_control_camera_values = process_pose_params(example["control_camera_values"], height=pose_resize_size[0], width=pose_resize_size[1]).permute(0, 3, 1, 2).contiguous()
                            new_examples["control_camera_values"].append(transform_no_normalize(local_control_camera_values))

                if args.enable_arm_info:
                    local_arm_action, local_arm_mask = _prepare_arm_action(example)
                    new_examples["arm_action_values"].append(local_arm_action)
                    new_examples["arm_action_mask"].append(local_arm_mask)
                
                new_examples["text"].append(example["text"])
                if args.enable_object_temporal_attention_bias:
                    new_examples["object_temporal_attention"].append(example.get("object_temporal_attention", []))
                    new_examples["object_temporal_attention_config"].append(example.get("object_temporal_attention_config", {}))

                if args.train_mode != "control":
                    clip_index = _sample_clip_index(len(new_examples["pixel_values"][-1]))
                    new_examples["clip_idx"].append(clip_index)

                    ref_pixel_values = new_examples["pixel_values"][-1][clip_index].unsqueeze(0)
                    new_examples["ref_pixel_values"].append(ref_pixel_values)

                    clip_pixel_values = new_examples["pixel_values"][-1][clip_index].permute(1, 2, 0).contiguous()
                    clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
                    new_examples["clip_pixel_values"].append(clip_pixel_values)

                    if args.add_inpaint_info:
                        mask = get_random_mask(new_examples["pixel_values"][-1].size())
                        mask_pixel_values = new_examples["pixel_values"][-1] * (1 - mask) 
                        # Wan 2.1 use 0 for masked pixels
                        # + torch.ones_like(new_examples["pixel_values"][-1]) * -1 * mask
                        new_examples["mask_pixel_values"].append(mask_pixel_values)
                        new_examples["mask"].append(mask)

            # Limit the number of frames to the same
            new_examples["pixel_values"] = torch.stack([example for example in new_examples["pixel_values"]])
            if not skip_unused_control_pixel_values:
                new_examples["control_pixel_values"] = torch.stack([example[:batch_video_length] for example in new_examples["control_pixel_values"]])
            if args.train_mode != "control":
                new_examples["ref_pixel_values"] = torch.stack([example for example in new_examples["ref_pixel_values"]])
                new_examples["clip_pixel_values"] = torch.stack([example for example in new_examples["clip_pixel_values"]])
                new_examples["clip_idx"] = torch.tensor(new_examples["clip_idx"])
            if args.train_mode == "control_camera_ref":
                if batch_has_camera:
                    new_examples["control_camera_values"] = torch.stack([example[:batch_video_length] for example in new_examples["control_camera_values"]])
                else:
                    new_examples["control_camera_values"] = None
                new_examples["control_camera_mask"] = torch.tensor(new_examples["control_camera_mask"], dtype=torch.float32)
            if args.enable_arm_info:
                new_examples["arm_action_values"] = torch.stack([example for example in new_examples["arm_action_values"]])
                new_examples["arm_action_mask"] = torch.tensor(new_examples["arm_action_mask"], dtype=torch.float32)
            if args.enable_action_map_info:
                new_examples["action_map_pixel_values"] = torch.stack([example for example in new_examples["action_map_pixel_values"]])
                new_examples["action_map_mask"] = torch.tensor(new_examples["action_map_mask"], dtype=torch.float32)
            if args.add_inpaint_info:
                new_examples["mask_pixel_values"] = torch.stack([example for example in new_examples["mask_pixel_values"]])
                new_examples["mask"] = torch.stack([example for example in new_examples["mask"]])

            # Encode prompts when enable_text_encoder_in_dataloader=True
            if args.enable_text_encoder_in_dataloader:
                prompt_ids = tokenizer(
                    new_examples['text'], 
                    max_length=args.tokenizer_max_length, 
                    padding="max_length", 
                    add_special_tokens=True, 
                    truncation=True, 
                    return_tensors="pt"
                )
                encoder_hidden_states = text_encoder(
                    prompt_ids.input_ids
                )[0]
                new_examples['encoder_attention_mask'] = prompt_ids.attention_mask
                new_examples['encoder_hidden_states'] = encoder_hidden_states

            return new_examples
        
        # DataLoaders creation:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            persistent_workers=True if args.dataloader_num_workers != 0 else False,
            num_workers=args.dataloader_num_workers,
            worker_init_fn=worker_init_fn(args.seed + accelerator.process_index)
        )
    else:
        # DataLoaders creation:
        batch_sampler_generator = torch.Generator().manual_seed(args.seed)
        batch_sampler = ImageVideoSampler(RandomSampler(train_dataset, generator=batch_sampler_generator), train_dataset, args.train_batch_size)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=batch_sampler, 
            collate_fn=collate_fn_no_bucket,
            persistent_workers=True if args.dataloader_num_workers != 0 else False,
            num_workers=args.dataloader_num_workers,
            worker_init_fn=worker_init_fn(args.seed + accelerator.process_index)
        )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    transformer3d, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer3d, optimizer, train_dataloader, lr_scheduler
    )

    if fsdp_stage != 0 or zero_stage != 0:
        from functools import partial

        from videox_fun.dist import set_multi_gpus_devices, shard_model
        shard_fn = partial(shard_model, device_id=accelerator.device, param_dtype=weight_dtype)
        text_encoder = shard_fn(text_encoder)

    if args.use_ema:
        ema_transformer3d.to(accelerator.device)

    # Move text_encode and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
    if not args.enable_text_encoder_in_dataloader:
        text_encoder.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        keys_to_pop = [k for k, v in tracker_config.items() if isinstance(v, list)]
        for k in keys_to_pop:
            tracker_config.pop(k)
            print(f"Removed tracker_config['{k}']")
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    object_temporal_attention_bias_controller = None
    if args.enable_object_temporal_attention_bias:
        object_temporal_attention_bias_controller = ObjectTemporalAttentionBiasController(
            unwrap_model(transformer3d),
            tokenizer,
            layers=args.object_temporal_attention_layers,
            source_num_frames=args.object_temporal_attention_source_frames,
            require_token_match=args.object_temporal_attention_require_token_match,
        ).attach()
        logger.info(
            "Enabled object-temporal attention logit bias: layers=%s source_frames=%s",
            args.object_temporal_attention_layers,
            args.object_temporal_attention_source_frames,
        )

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            valid_dirs = []
            for checkpoint_name in dirs:
                checkpoint_dir = os.path.join(args.output_dir, checkpoint_name)
                has_model_state = (
                    os.path.isfile(os.path.join(checkpoint_dir, "diffusion_pytorch_model.safetensors"))
                    or os.path.isfile(os.path.join(checkpoint_dir, "pytorch_model.bin"))
                    or os.path.isdir(os.path.join(checkpoint_dir, "transformer"))
                    or os.path.isfile(os.path.join(checkpoint_dir, "pytorch_model_fsdp_0", ".metadata"))
                )
                has_training_state = (
                    os.path.isfile(os.path.join(checkpoint_dir, "scheduler.bin"))
                    and (
                        os.path.isfile(os.path.join(checkpoint_dir, "optimizer_0", ".metadata"))
                        or os.path.isdir(os.path.join(checkpoint_dir, "optimizer"))
                    )
                )
                if has_model_state and has_training_state:
                    valid_dirs.append(checkpoint_name)
                elif accelerator.is_main_process:
                    logger.warning(f"Skipping incomplete checkpoint {checkpoint_dir}")
            path = valid_dirs[-1] if len(valid_dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            global_step = int(path.split("-")[1])

            initial_global_step = global_step

            pkl_path = os.path.join(os.path.join(args.output_dir, path), "sampler_pos_start.pkl")
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as file:
                    _, first_epoch = pickle.load(file)
            else:
                first_epoch = global_step // num_update_steps_per_epoch
            print(f"Load pkl from {pkl_path}. Get first_epoch = {first_epoch}.")

            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    if args.multi_stream:
        # create extra cuda streams to speedup inpaint vae computation
        vae_stream_1 = torch.cuda.Stream()
        vae_stream_2 = torch.cuda.Stream()
    else:
        vae_stream_1 = None
        vae_stream_2 = None

    # Calculate the index we need
    boundary        = config['transformer_additional_kwargs'].get('boundary', 0.900)
    split_timesteps = args.train_sampling_steps * boundary
    differences     = torch.abs(noise_scheduler.timesteps - split_timesteps)
    closest_index   = torch.argmin(differences).item()
    print(f"The boundary is {boundary} and the boundary_type is {args.boundary_type}. The closest_index we calculate is {closest_index}")
    if args.boundary_type == "high":
        start_num_idx = 0
        train_sampling_steps = closest_index
    elif args.boundary_type == "low":
        start_num_idx = closest_index
        train_sampling_steps = args.train_sampling_steps - closest_index
    else:
        start_num_idx = 0
        train_sampling_steps = args.train_sampling_steps
    idx_sampling = DiscreteSampling(train_sampling_steps, start_num_idx=start_num_idx, uniform_sampling=args.uniform_sampling)

    timing_file = None
    if args.benchmark_timing_path and accelerator.is_main_process:
        os.makedirs(os.path.dirname(args.benchmark_timing_path), exist_ok=True)
        timing_file = open(args.benchmark_timing_path, "w", buffering=1)

    def synchronize_for_timing():
        if args.benchmark_timing_path and torch.cuda.is_available():
            torch.cuda.synchronize(accelerator.device)

    debug_heartbeat_enabled = os.environ.get("WAN22_DEBUG_HEARTBEAT", "0") == "1"
    debug_heartbeat_steps = int(os.environ.get("WAN22_DEBUG_HEARTBEAT_STEPS", "20"))
    debug_heartbeat_every = max(1, int(os.environ.get("WAN22_DEBUG_HEARTBEAT_EVERY", "1")))
    debug_progress_every_steps = int(os.environ.get("WAN22_DEBUG_PROGRESS_EVERY_STEPS", "10"))
    debug_heartbeat_sync = os.environ.get("WAN22_DEBUG_HEARTBEAT_SYNC", "0") == "1"
    debug_heartbeat_stop_step = initial_global_step + debug_heartbeat_steps
    debug_heartbeat_file = None
    debug_heartbeat_dir = os.environ.get("WAN22_DEBUG_HEARTBEAT_DIR", "")
    if debug_heartbeat_enabled and debug_heartbeat_dir:
        os.makedirs(debug_heartbeat_dir, exist_ok=True)
        debug_heartbeat_file = os.path.join(
            debug_heartbeat_dir,
            f"heartbeat_rank{accelerator.process_index}_local{accelerator.local_process_index}.log",
        )

    def _debug_value_summary(value):
        if value is None:
            return "None"
        if torch.is_tensor(value):
            shape = tuple(value.shape)
            return f"Tensor(shape={shape}, dtype={value.dtype}, device={value.device})"
        if isinstance(value, np.ndarray):
            return f"ndarray(shape={value.shape}, dtype={value.dtype})"
        if isinstance(value, (list, tuple)):
            preview = ", ".join(_debug_value_summary(item) for item in list(value)[:3])
            suffix = ", ..." if len(value) > 3 else ""
            return f"{type(value).__name__}(len={len(value)}, items=[{preview}{suffix}])"
        if isinstance(value, dict):
            return f"dict(keys={list(value.keys())[:8]})"
        return repr(value)

    def debug_heartbeat(phase, *, epoch=None, dataloader_step=None, force=False, **items):
        if not debug_heartbeat_enabled:
            return
        if not force:
            if global_step >= debug_heartbeat_stop_step:
                return
            if dataloader_step is not None and dataloader_step % debug_heartbeat_every != 0:
                return
        if debug_heartbeat_sync and torch.cuda.is_available():
            torch.cuda.synchronize(accelerator.device)
        memory_text = ""
        if torch.cuda.is_available():
            try:
                allocated_gb = torch.cuda.memory_allocated(accelerator.device) / (1024 ** 3)
                reserved_gb = torch.cuda.memory_reserved(accelerator.device) / (1024 ** 3)
                memory_text = f" cuda_allocated_gb={allocated_gb:.2f} cuda_reserved_gb={reserved_gb:.2f}"
            except Exception as exc:
                memory_text = f" cuda_memory_error={type(exc).__name__}:{exc}"
        item_text = " ".join(f"{key}={_debug_value_summary(value)}" for key, value in items.items())
        message = (
            f"[HEARTBEAT {time.strftime('%Y-%m-%d %H:%M:%S')}] "
            f"rank={accelerator.process_index}/{accelerator.num_processes} "
            f"local_rank={accelerator.local_process_index} "
            f"device={accelerator.device} "
            f"epoch={epoch} dataloader_step={dataloader_step} global_step={global_step} "
            f"sync_gradients={accelerator.sync_gradients} phase={phase}"
            f"{memory_text} {item_text}"
        )
        print(message, flush=True)
        if debug_heartbeat_file is not None:
            with open(debug_heartbeat_file, "a", buffering=1) as f:
                f.write(message + "\n")

    last_iter_end = time.perf_counter()
    benchmark_update_data_time = 0.0
    benchmark_update_compute_time = 0.0
    benchmark_update_vae_time = 0.0
    benchmark_update_micro_steps = 0

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        batch_sampler.sampler.generator = torch.Generator().manual_seed(args.seed + epoch)
        for step, batch in enumerate(train_dataloader):
            control_latents = None
            control_camera_latents = None
            control_camera_mask = None
            action_map_mask = None
            arm_action_values = None
            arm_action_mask = None
            mask_conditions = None
            full_ref = None
            debug_heartbeat(
                "batch_received",
                epoch=epoch,
                dataloader_step=step,
                batch_idx=batch.get("idx"),
                batch_data_type=batch.get("data_type"),
                batch_control_type=batch.get("control_type"),
                pixel_values=batch.get("pixel_values"),
            )
            synchronize_for_timing()
            iter_start = time.perf_counter()
            data_time = iter_start - last_iter_end
            compute_start = iter_start
            vae_encode_time = 0.0
            # Data batch sanity check
            if not args.skip_sanity_check and epoch == first_epoch and step == 0 and accelerator.is_local_main_process:
                pixel_values, texts = batch['pixel_values'].cpu(), batch['text']
                control_pixel_values = batch.get("control_pixel_values", None)
                pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                if control_pixel_values is not None:
                    control_pixel_values = control_pixel_values.cpu()
                    control_pixel_values = rearrange(control_pixel_values, "b f c h w -> b c f h w")
                os.makedirs(os.path.join(args.logging_dir, "sanity_check"), exist_ok=True)
                for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                    pixel_value = pixel_value[None, ...]
                    gif_name = '-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_step}-{idx}'
                    save_videos_grid(pixel_value, f"{args.logging_dir}/sanity_check/{gif_name[:10]}.gif", rescale=True)
                    if control_pixel_values is not None:
                        control_pixel_value = control_pixel_values[idx][None, ...]
                        save_videos_grid(control_pixel_value, f"{args.logging_dir}/sanity_check/{gif_name[:10]}_control.gif", rescale=True)
                
                if args.train_mode != "control":
                    ref_pixel_values = batch["ref_pixel_values"].cpu()
                    ref_pixel_values = rearrange(ref_pixel_values, "b f c h w -> b c f h w")
                    for idx, (ref_pixel_value, text) in enumerate(zip(ref_pixel_values, texts)):
                        ref_pixel_value = ref_pixel_value[None, ...]
                        gif_name = '-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_step}-{idx}'
                        save_videos_grid(ref_pixel_value, f"{args.logging_dir}/sanity_check/{gif_name[:10]}_ref.gif", rescale=True)

                if args.add_inpaint_info:
                    clip_pixel_values, mask_pixel_values, texts = batch['clip_pixel_values'].cpu(), batch['mask_pixel_values'].cpu(), batch['text']
                    mask_pixel_values = rearrange(mask_pixel_values, "b f c h w -> b c f h w")
                    for idx, (clip_pixel_value, pixel_value, text) in enumerate(zip(clip_pixel_values, mask_pixel_values, texts)):
                        pixel_value = pixel_value[None, ...]
                        Image.fromarray(np.uint8(clip_pixel_value)).save(f"{args.logging_dir}/sanity_check/clip_{gif_name[:10] if not text == '' else f'{global_step}-{idx}'}.png")
                        save_videos_grid(pixel_value, f"{args.logging_dir}/sanity_check/mask_{gif_name[:10] if not text == '' else f'{global_step}-{idx}'}.gif", rescale=True)

            with accelerator.accumulate(transformer3d):
                # Convert images to latent space
                pixel_values = batch["pixel_values"].to(weight_dtype)
                control_pixel_values = None
                if batch.get("control_pixel_values", None) is not None:
                    control_pixel_values = batch["control_pixel_values"].to(weight_dtype)
                control_camera_values = None
                control_camera_mask = None
                arm_action_values = None
                arm_action_mask = None
                action_map_pixel_values = None
                action_map_mask = None
                if args.train_mode == "control_camera_ref":
                    control_camera_mask = batch.get("control_camera_mask", None)
                    if control_camera_mask is not None:
                        control_camera_mask = control_camera_mask.to(device=pixel_values.device, dtype=torch.float32)
                    batch_control_camera_values = batch.get("control_camera_values", None)
                    if batch_control_camera_values is not None and (
                        control_camera_mask is None or bool(torch.any(control_camera_mask > 0).item())
                    ):
                        control_camera_values = batch_control_camera_values.to(weight_dtype)
                if args.enable_arm_info and "arm_action_values" in batch:
                    arm_action_values = batch["arm_action_values"].to(weight_dtype)
                    arm_action_mask = batch.get("arm_action_mask", None)
                    if arm_action_mask is not None:
                        arm_action_mask = arm_action_mask.to(device=arm_action_values.device, dtype=torch.float32)
                if args.enable_action_map_info and batch.get("action_map_pixel_values", None) is not None:
                    action_map_pixel_values = batch["action_map_pixel_values"].to(weight_dtype)
                    action_map_mask = batch.get("action_map_mask", None)
                    if action_map_mask is not None:
                        action_map_mask = action_map_mask.to(device=action_map_pixel_values.device, dtype=torch.float32)

                # Increase the batch size when the length of the latent sequence of the current sample is small
                if args.auto_tile_batch_size and args.training_with_video_token_length and zero_stage != 3:
                    if args.video_sample_n_frames * args.token_sample_size * args.token_sample_size // 16 >= pixel_values.size()[1] * pixel_values.size()[3] * pixel_values.size()[4]:
                        pixel_values = torch.tile(pixel_values, (4, 1, 1, 1, 1))
                        if control_pixel_values is not None:
                            control_pixel_values = torch.tile(control_pixel_values, (4, 1, 1, 1, 1))
                        if action_map_pixel_values is not None:
                            action_map_pixel_values = torch.tile(action_map_pixel_values, (4, 1, 1, 1, 1))
                            if action_map_mask is not None:
                                action_map_mask = torch.tile(action_map_mask, (4,))
                        if args.train_mode == "control_camera_ref":
                            if control_camera_values is not None:
                                control_camera_values = torch.tile(control_camera_values, (4, 1, 1, 1, 1))
                            if control_camera_mask is not None:
                                control_camera_mask = torch.tile(control_camera_mask, (4,))
                        if arm_action_values is not None:
                            arm_action_values = torch.tile(arm_action_values, (4, 1, 1))
                            if arm_action_mask is not None:
                                arm_action_mask = torch.tile(arm_action_mask, (4,))
                        if args.enable_text_encoder_in_dataloader:
                            batch['encoder_hidden_states'] = torch.tile(batch['encoder_hidden_states'], (4, 1, 1))
                            batch['encoder_attention_mask'] = torch.tile(batch['encoder_attention_mask'], (4, 1))
                        else:
                            batch['text'] = batch['text'] * 4
                    elif args.video_sample_n_frames * args.token_sample_size * args.token_sample_size // 4 >= pixel_values.size()[1] * pixel_values.size()[3] * pixel_values.size()[4]:
                        pixel_values = torch.tile(pixel_values, (2, 1, 1, 1, 1))
                        if control_pixel_values is not None:
                            control_pixel_values = torch.tile(control_pixel_values, (2, 1, 1, 1, 1))
                        if action_map_pixel_values is not None:
                            action_map_pixel_values = torch.tile(action_map_pixel_values, (2, 1, 1, 1, 1))
                            if action_map_mask is not None:
                                action_map_mask = torch.tile(action_map_mask, (2,))
                        if args.train_mode == "control_camera_ref":
                            if control_camera_values is not None:
                                control_camera_values = torch.tile(control_camera_values, (2, 1, 1, 1, 1))
                            if control_camera_mask is not None:
                                control_camera_mask = torch.tile(control_camera_mask, (2,))
                        if arm_action_values is not None:
                            arm_action_values = torch.tile(arm_action_values, (2, 1, 1))
                            if arm_action_mask is not None:
                                arm_action_mask = torch.tile(arm_action_mask, (2,))
                        if args.enable_text_encoder_in_dataloader:
                            batch['encoder_hidden_states'] = torch.tile(batch['encoder_hidden_states'], (2, 1, 1))
                            batch['encoder_attention_mask'] = torch.tile(batch['encoder_attention_mask'], (2, 1))
                        else:
                            batch['text'] = batch['text'] * 2
                
                if args.train_mode != "control":
                    ref_pixel_values = batch["ref_pixel_values"].to(weight_dtype)
                    clip_idx = batch["clip_idx"]
                    # Increase the batch size when the length of the latent sequence of the current sample is small
                    if args.auto_tile_batch_size and args.training_with_video_token_length and zero_stage != 3:
                        if args.video_sample_n_frames * args.token_sample_size * args.token_sample_size // 16 >= pixel_values.size()[1] * pixel_values.size()[3] * pixel_values.size()[4]:
                            ref_pixel_values = torch.tile(ref_pixel_values, (4, 1, 1, 1, 1))
                            clip_idx = torch.tile(clip_idx, (4,))
                        elif args.video_sample_n_frames * args.token_sample_size * args.token_sample_size // 4 >= pixel_values.size()[1] * pixel_values.size()[3] * pixel_values.size()[4]:
                            ref_pixel_values = torch.tile(ref_pixel_values, (2, 1, 1, 1, 1))
                            clip_idx = torch.tile(clip_idx, (2,))

                if args.add_inpaint_info:
                    mask_pixel_values = batch["mask_pixel_values"].to(weight_dtype)
                    mask = batch["mask"].to(weight_dtype)
                    # Increase the batch size when the length of the latent sequence of the current sample is small
                    if args.auto_tile_batch_size and args.training_with_video_token_length and not zero_stage == 3:
                        if args.video_sample_n_frames * args.token_sample_size * args.token_sample_size // 16 >= pixel_values.size()[1] * pixel_values.size()[3] * pixel_values.size()[4]:
                            mask_pixel_values = torch.tile(mask_pixel_values, (4, 1, 1, 1, 1))
                            mask = torch.tile(mask, (4, 1, 1, 1, 1))
                        elif args.video_sample_n_frames * args.token_sample_size * args.token_sample_size // 4 >= pixel_values.size()[1] * pixel_values.size()[3] * pixel_values.size()[4]:
                            mask_pixel_values = torch.tile(mask_pixel_values, (2, 1, 1, 1, 1))
                            mask = torch.tile(mask, (2, 1, 1, 1, 1))

                if args.random_frame_crop:
                    def _create_special_list(length):
                        if length == 1:
                            return [1.0]
                        if length >= 2:
                            last_element = 0.90
                            remaining_sum = 1.0 - last_element
                            other_elements_value = remaining_sum / (length - 1)
                            special_list = [other_elements_value] * (length - 1) + [last_element]
                            return special_list
                    select_frames = [_tmp for _tmp in list(range(sample_n_frames_bucket_interval + 1, args.video_sample_n_frames + sample_n_frames_bucket_interval, sample_n_frames_bucket_interval))]
                    select_frames_prob = np.array(_create_special_list(len(select_frames)))
                    
                    if len(select_frames) != 0:
                        if rng is None:
                            temp_n_frames = np.random.choice(select_frames, p = select_frames_prob)
                        else:
                            temp_n_frames = rng.choice(select_frames, p = select_frames_prob)
                    else:
                        temp_n_frames = 1

                    # Magvae needs the number of frames to be 4n + 1.
                    temp_n_frames = (temp_n_frames - 1) // sample_n_frames_bucket_interval + 1

                    pixel_values = pixel_values[:, :temp_n_frames, :, :]
                    if control_pixel_values is not None:
                        control_pixel_values = control_pixel_values[:, :temp_n_frames, :, :]
                    if action_map_pixel_values is not None:
                        action_map_pixel_values = action_map_pixel_values[:, :temp_n_frames, :, :]
                    if args.train_mode == "control_camera_ref" and control_camera_values is not None:
                        control_camera_values = control_camera_values[:, :temp_n_frames, :, :, :]
                    
                # Keep all node same token length to accelerate the traning when resolution grows.
                if args.keep_all_node_same_token_length:
                    if args.token_sample_size > 256:
                        numbers_list = list(range(256, args.token_sample_size + 1, 128))

                        if numbers_list[-1] != args.token_sample_size:
                            numbers_list.append(args.token_sample_size)
                    else:
                        numbers_list = [256]
                    numbers_list = [_number * _number * args.video_sample_n_frames for _number in  numbers_list]
            
                    actual_token_length = index_rng.choice(numbers_list)
                    actual_video_length = (min(
                            actual_token_length / pixel_values.size()[-1] / pixel_values.size()[-2], args.video_sample_n_frames
                    ) - 1) // sample_n_frames_bucket_interval * sample_n_frames_bucket_interval + 1
                    actual_video_length = int(max(actual_video_length, 1))

                    # Magvae needs the number of frames to be 4n + 1.
                    actual_video_length = (actual_video_length - 1) // sample_n_frames_bucket_interval + 1

                    pixel_values = pixel_values[:, :actual_video_length, :, :]
                    if control_pixel_values is not None:
                        control_pixel_values = control_pixel_values[:, :actual_video_length, :, :]
                    if action_map_pixel_values is not None:
                        action_map_pixel_values = action_map_pixel_values[:, :actual_video_length, :, :]
                    if args.train_mode == "control_camera_ref" and control_camera_values is not None:
                        control_camera_values = control_camera_values[:, :actual_video_length, :, :, :]

                if args.low_vram:
                    torch.cuda.empty_cache()
                    vae.to(accelerator.device)
                    if not args.enable_text_encoder_in_dataloader:
                        text_encoder.to("cpu")

                synchronize_for_timing()
                vae_encode_start = time.perf_counter()
                with torch.no_grad():
                    # This way is quicker when batch grows up
                    def _batch_encode_vae(pixel_values):
                        pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                        bs = args.vae_mini_batch
                        new_pixel_values = []
                        for i in range(0, pixel_values.shape[0], bs):
                            pixel_values_bs = pixel_values[i : i + bs]
                            pixel_values_bs = vae.encode(pixel_values_bs)[0]
                            pixel_values_bs = pixel_values_bs.sample()
                            new_pixel_values.append(pixel_values_bs)
                        return torch.cat(new_pixel_values, dim = 0)
                    if vae_stream_1 is not None:
                        vae_stream_1.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(vae_stream_1):
                            latents = _batch_encode_vae(pixel_values)
                    else:
                        latents = _batch_encode_vae(pixel_values)

                    action_map_latents = None
                    if action_map_pixel_values is not None and action_map_mask is not None and bool(torch.any(action_map_mask > 0).item()):
                        action_map_latents = _batch_encode_vae(action_map_pixel_values)

                    if args.train_mode != "control_camera_ref":
                        control_latents = _batch_encode_vae(control_pixel_values)
                        # Make control latents to zero
                        for bs_index in range(control_latents.size()[0]):
                            if rng is None:
                                zero_init_control_latents_conv_in = np.random.choice([0, 1], p = [0.90, 0.10])
                            else:
                                zero_init_control_latents_conv_in = rng.choice([0, 1], p = [0.90, 0.10])

                            if zero_init_control_latents_conv_in:
                                control_latents[bs_index] = control_latents[bs_index] * 0
                        control_camera_latents = None
                    else:
                        control_latents = None
                        if control_camera_values is None:
                            control_camera_latents = None
                        else:
                            control_camera_latents = rearrange(control_camera_values, "b f c h w -> b c f h w")
                            control_camera_latents = torch.concat(
                                [
                                    torch.repeat_interleave(control_camera_latents[:, :, 0:1], repeats=4, dim=2), 
                                    control_camera_latents[:, :, 1:]
                                ], dim=2
                            ).transpose(1, 2).contiguous()
                            control_camera_latents = control_camera_latents.view(control_camera_latents.shape[0], control_camera_latents.shape[1] // 4, 4, control_camera_latents.shape[2], control_camera_latents.shape[3], control_camera_latents.shape[4])
                            control_camera_latents = control_camera_latents.transpose(2, 3).contiguous()
                            control_camera_latents = control_camera_latents.view(control_camera_latents.shape[0], control_camera_latents.shape[1], control_camera_latents.shape[2] * 4, control_camera_latents.shape[4], control_camera_latents.shape[5])
                            control_camera_latents = control_camera_latents.transpose(1, 2)
                            
                    if args.train_mode != "control":
                        ref_latents = _batch_encode_vae(ref_pixel_values)
                        if args.add_full_ref_image_in_self_attention:
                            full_ref = ref_latents[:, :, 0].clone()

                        ref_latents_conv_in = torch.zeros_like(latents).to(ref_latents.device, ref_latents.dtype)
                        ref_latents_conv_in[:, :, :1] = ref_latents
                        for bs_index in range(ref_latents.size()[0]):
                            if rng is None:
                                zero_init_ref_latents_conv_in = np.random.choice([0, 1], p = [0.90, 0.10])
                            else:
                                zero_init_ref_latents_conv_in = rng.choice([0, 1], p = [0.90, 0.10])

                            if clip_idx[bs_index] != 0 or (zero_init_ref_latents_conv_in and latents.size()[1] != 1):
                                ref_latents_conv_in[bs_index, :, :1] = ref_latents_conv_in[bs_index, :, :1] * 0

                            if args.add_full_ref_image_in_self_attention:
                                if rng is None:
                                    zero_init_full_ref_conv_in = np.random.choice([0, 1], p = [0.90, 0.10])
                                else:
                                    zero_init_full_ref_conv_in = rng.choice([0, 1], p = [0.90, 0.10])
                                if clip_idx[bs_index] == 0 or zero_init_full_ref_conv_in:
                                    full_ref[bs_index] = full_ref[bs_index] * 0

                    if args.add_inpaint_info:
                        t2v_flag = [(_mask == 1).all() for _mask in mask]
                        new_t2v_flag = []
                        for _mask in t2v_flag:
                            if _mask and np.random.rand() < 0.90:
                                new_t2v_flag.append(0)
                            else:
                                new_t2v_flag.append(1)
                        t2v_flag = torch.from_numpy(np.array(new_t2v_flag)).to(accelerator.device, dtype=weight_dtype)
                        
                        mask = rearrange(mask, "b f c h w -> b c f h w")
                        mask = torch.concat(
                            [
                                torch.repeat_interleave(mask[:, :, 0:1], repeats=4, dim=2), 
                                mask[:, :, 1:]
                            ], dim=2
                        )
                        mask = mask.view(mask.shape[0], mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4])
                        mask = mask.transpose(1, 2)
                        mask_conditions = F.interpolate(mask[:, :1], size=latents.size()[-3:], mode='trilinear', align_corners=True).to(accelerator.device, weight_dtype)
                        mask = resize_mask(1 - mask, latents)

                        # Encode inpaint latents.
                        mask_latents = _batch_encode_vae(mask_pixel_values)

                        inpaint_latents = torch.concat([mask, mask_latents], dim=1)
                        inpaint_latents = t2v_flag[:, None, None, None, None] * inpaint_latents
                        if action_map_latents is not None and action_map_mask is not None:
                            action_gate = action_map_mask.to(device=inpaint_latents.device, dtype=inpaint_latents.dtype).view(-1, 1, 1, 1, 1)
                            latent_start = inpaint_latents.size(1) - vae.latent_channels
                            base_latent_part = inpaint_latents[:, latent_start:].clone()
                            action_latent_part = base_latent_part.clone()
                            frames_to_use = min(action_latent_part.size(2), action_map_latents.size(2))
                            if frames_to_use > 1:
                                action_latent_part[:, :, 1:frames_to_use] = action_map_latents[:, :, 1:frames_to_use]
                            inpaint_latents[:, latent_start:] = action_gate * action_latent_part + (1.0 - action_gate) * base_latent_part
                    else:
                        inpaint_latents = None

                    if control_latents is None:
                        if inpaint_latents is None:
                            control_latents = ref_latents_conv_in
                        else:
                            control_latents = inpaint_latents
                    else:
                        if inpaint_latents is None:
                            control_latents = torch.cat([control_latents, ref_latents_conv_in], dim = 1)
                        else:
                            control_latents = torch.cat([control_latents, inpaint_latents], dim = 1)
                                
                # wait for latents = vae.encode(pixel_values) to complete
                if vae_stream_1 is not None:
                    torch.cuda.current_stream().wait_stream(vae_stream_1)
                synchronize_for_timing()
                vae_encode_time = time.perf_counter() - vae_encode_start

                if args.low_vram:
                    vae.to('cpu')
                    torch.cuda.empty_cache()
                    if not args.enable_text_encoder_in_dataloader:
                        text_encoder.to(accelerator.device)

                if args.enable_text_encoder_in_dataloader:
                    prompt_embeds = batch['encoder_hidden_states'].to(device=latents.device)
                else:
                    with torch.no_grad():
                        prompt_ids = tokenizer(
                            batch['text'], 
                            padding="max_length", 
                            max_length=args.tokenizer_max_length, 
                            truncation=True, 
                            add_special_tokens=True, 
                            return_tensors="pt"
                        )
                        text_input_ids = prompt_ids.input_ids
                        prompt_attention_mask = prompt_ids.attention_mask
                        if object_temporal_attention_bias_controller is not None:
                            object_temporal_attention_bias_controller.set_batch(
                                batch.get("object_temporal_attention", []),
                                text_input_ids.to(latents.device),
                                source_num_frames=args.object_temporal_attention_source_frames,
                            )

                        seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
                        prompt_embeds = text_encoder(text_input_ids.to(latents.device), attention_mask=prompt_attention_mask.to(latents.device))[0]
                        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]

                if args.low_vram and not args.enable_text_encoder_in_dataloader:
                    text_encoder.to('cpu')
                    torch.cuda.empty_cache()

                bsz, channel, num_frames, height, width = latents.size()
                debug_heartbeat(
                    "latents_and_text_ready",
                    epoch=epoch,
                    dataloader_step=step,
                    latents=latents,
                    control_latents=control_latents,
                    control_camera_latents=control_camera_latents,
                    mask_conditions=mask_conditions,
                    prompt_embeds=prompt_embeds,
                    arm_action_values=arm_action_values,
                    arm_action_mask=arm_action_mask,
                )
                noise = torch.randn(latents.size(), device=latents.device, generator=torch_rng, dtype=weight_dtype)

                if not args.uniform_sampling:
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme=args.weighting_scheme,
                        batch_size=bsz,
                        logit_mean=args.logit_mean,
                        logit_std=args.logit_std,
                        mode_scale=args.mode_scale,
                    )
                    indices = (u * noise_scheduler.config.num_train_timesteps).long()
                else:
                    # Sample a random timestep for each image
                    # timesteps = generate_timestep_with_lognorm(0, args.train_sampling_steps, (bsz,), device=latents.device, generator=torch_rng)
                    # timesteps = torch.randint(0, args.train_sampling_steps, (bsz,), device=latents.device, generator=torch_rng)
                    indices = idx_sampling(bsz, generator=torch_rng, device=latents.device)
                    indices = indices.long().cpu()
                timesteps = noise_scheduler.timesteps[indices].to(device=latents.device)

                def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
                    sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
                    schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
                    timesteps = timesteps.to(accelerator.device)
                    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

                    sigma = sigmas[step_indices].flatten()
                    while len(sigma.shape) < n_dim:
                        sigma = sigma.unsqueeze(-1)
                    return sigma

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
                noisy_latents = (1.0 - sigmas) * latents + sigmas * noise

                # Add noise
                target = noise - latents
                
                target_shape = (vae.latent_channels, num_frames, width, height)
                seq_len = math.ceil(
                    (target_shape[2] * target_shape[3]) /
                    (accelerator.unwrap_model(transformer3d).config.patch_size[1] * accelerator.unwrap_model(transformer3d).config.patch_size[2]) *
                    target_shape[1]
                )

                if spatial_compression_ratio >= 16:
                    mask_conditions_bs = mask_conditions.size()[0]
                    mask_conditions[:, :, 1:, :, :] = 1
                    if not mask_conditions[:, :, 0, :, :].any():
                        noisy_latents = (1 - mask_conditions) * control_latents[:, -vae.latent_channels:] + mask_conditions * noisy_latents
                        
                        temp_ts = (mask_conditions[:, 0, :, ::2, ::2] * timesteps[:, None, None, None]).flatten(1)
                        timesteps = torch.cat([temp_ts, temp_ts.new_ones(mask_conditions_bs, seq_len - temp_ts.size(1)) * timesteps[:, None,]], dim = 1)
                    else:
                        timesteps = mask_conditions.new_ones(mask_conditions_bs, seq_len) * timesteps[:, None,]

                # Predict the noise residual. Method1 diagnoses only the arm
                # condition; text, reference, camera, latent input, timestep,
                # noise, and model RNG are identical across the on/null pair.
                transformer_for_action_mask = accelerator.unwrap_model(transformer3d)
                if action_map_mask is None:
                    transformer_for_action_mask._current_action_map_mask = torch.zeros(bsz, device=latents.device, dtype=torch.float32)
                else:
                    transformer_for_action_mask._current_action_map_mask = action_map_mask.to(device=latents.device, dtype=torch.float32).view(-1)

                method1_effect_map = None
                method1_active_mask = None
                method1_main_arm_action_mask = arm_action_mask
                if args.enable_method1_focused_loss:
                    method1_requires_effect_map = args.method1_loss_variant == "CAER"
                    if arm_action_values is None:
                        raise RuntimeError(
                            "Method1 focused loss requires arm_action_values in every LIBERO batch"
                        )
                    if arm_action_mask is None:
                        method1_base_arm_mask = (
                            arm_action_values.float().abs().sum(dim=(1, 2)) > 1e-6
                        ).to(device=latents.device, dtype=torch.float32)
                    else:
                        method1_base_arm_mask = arm_action_mask.to(
                            device=latents.device, dtype=torch.float32
                        ).view(-1)
                    if method1_base_arm_mask.numel() != bsz:
                        raise RuntimeError(
                            "arm_action_mask batch size does not match latent batch size: "
                            f"{method1_base_arm_mask.numel()} != {bsz}"
                        )

                    action_dropout = torch.rand(
                        (bsz,), device=latents.device, generator=torch_rng
                    ) < float(args.method1_action_dropout_prob)
                    method1_keep_mask = (~action_dropout).to(torch.float32)
                    method1_main_arm_action_mask = (
                        method1_base_arm_mask * method1_keep_mask
                    )
                    method1_active_mask = method1_main_arm_action_mask.view(
                        bsz, 1, 1, 1, 1
                    )

                    scheduler_sigmas = noise_scheduler.sigmas.to(
                        device=latents.device, dtype=torch.float32
                    )
                    method1_sigma_index = int(
                        torch.argmin(
                            (scheduler_sigmas - float(args.method1_tau_s)).abs()
                        ).item()
                    )
                    method1_sigma_index = max(
                        0,
                        min(method1_sigma_index, len(noise_scheduler.timesteps) - 1),
                    )
                    method1_base_timestep = noise_scheduler.timesteps[
                        method1_sigma_index
                    ].to(device=latents.device)
                    method1_base_timesteps = method1_base_timestep.repeat(bsz)
                    method1_sigmas = get_sigmas(
                        method1_base_timesteps,
                        n_dim=latents.ndim,
                        dtype=latents.dtype,
                    )
                    if (
                        method1_requires_effect_map
                        and global_step == 0
                        and step == 0
                        and accelerator.is_main_process
                    ):
                        print(
                            "method1_diagnostic_noise "
                            f"target_sigma={args.method1_tau_s:.6f} "
                            f"actual_sigma={scheduler_sigmas[method1_sigma_index].item():.6f} "
                            f"scheduler_index={method1_sigma_index} "
                            f"timestep={method1_base_timestep.item():.6f}",
                            flush=True,
                        )

                    method1_noise = torch.randn(
                        latents.size(),
                        device=latents.device,
                        generator=method1_torch_rng,
                        dtype=weight_dtype,
                    )
                    method1_noisy_latents = (
                        (1.0 - method1_sigmas) * latents
                        + method1_sigmas * method1_noise
                    )
                    method1_timesteps = method1_base_timesteps
                    if spatial_compression_ratio >= 16:
                        mask_conditions_bs = mask_conditions.size(0)
                        if not mask_conditions[:, :, 0, :, :].any():
                            method1_noisy_latents = (
                                (1 - mask_conditions)
                                * control_latents[:, -vae.latent_channels:]
                                + mask_conditions * method1_noisy_latents
                            )
                            method1_temp_ts = (
                                mask_conditions[:, 0, :, ::2, ::2]
                                * method1_base_timesteps[:, None, None, None]
                            ).flatten(1)
                            method1_timesteps = torch.cat(
                                [
                                    method1_temp_ts,
                                    method1_temp_ts.new_ones(
                                        mask_conditions_bs,
                                        seq_len - method1_temp_ts.size(1),
                                    )
                                    * method1_base_timesteps[:, None],
                                ],
                                dim=1,
                            )
                        else:
                            method1_timesteps = (
                                mask_conditions.new_ones(mask_conditions_bs, seq_len)
                                * method1_base_timesteps[:, None]
                            )

                    diagnostic_device_index = (
                        accelerator.device.index
                        if accelerator.device.index is not None
                        else torch.cuda.current_device()
                    )
                    with torch.no_grad(), torch.random.fork_rng(
                        devices=[diagnostic_device_index]
                    ):
                        pair_cpu_rng_state = torch.get_rng_state()
                        pair_cuda_rng_state = torch.cuda.get_rng_state(
                            diagnostic_device_index
                        )
                        with torch.cuda.amp.autocast(
                            dtype=weight_dtype
                        ), torch.cuda.device(device=accelerator.device):
                            method1_pred_on = None
                            if method1_requires_effect_map:
                                method1_pred_on = transformer3d(
                                    x=method1_noisy_latents,
                                    context=prompt_embeds,
                                    t=method1_timesteps,
                                    seq_len=seq_len,
                                    y=control_latents,
                                    y_camera=control_camera_latents if args.train_mode == "control_camera_ref" else None,
                                    y_camera_mask=control_camera_mask if args.train_mode == "control_camera_ref" and control_camera_latents is not None else None,
                                    arm_action=arm_action_values,
                                    arm_action_mask=method1_base_arm_mask,
                                    full_ref=full_ref if args.add_full_ref_image_in_self_attention else None,
                                )
                            torch.set_rng_state(pair_cpu_rng_state)
                            torch.cuda.set_rng_state(
                                pair_cuda_rng_state,
                                device=diagnostic_device_index,
                            )
                            method1_pred_null = None
                            if method1_requires_effect_map:
                                method1_pred_null = transformer3d(
                                    x=method1_noisy_latents,
                                    context=prompt_embeds,
                                    t=method1_timesteps,
                                    seq_len=seq_len,
                                    y=control_latents,
                                    y_camera=control_camera_latents if args.train_mode == "control_camera_ref" else None,
                                    y_camera_mask=control_camera_mask if args.train_mode == "control_camera_ref" and control_camera_latents is not None else None,
                                    arm_action=arm_action_values,
                                    arm_action_mask=torch.zeros_like(method1_base_arm_mask),
                                    full_ref=full_ref if args.add_full_ref_image_in_self_attention else None,
                                )
                        if method1_requires_effect_map:
                            method1_effect_map = torch.linalg.vector_norm(
                                method1_pred_on.float() - method1_pred_null.float(),
                                ord=2,
                                dim=1,
                                keepdim=True,
                            ).detach()
                            del method1_pred_on, method1_pred_null

                debug_heartbeat(
                    "before_transformer_forward",
                    epoch=epoch,
                    dataloader_step=step,
                    noisy_latents=noisy_latents,
                    timesteps=timesteps,
                    seq_len=seq_len,
                    target=target,
                    control_latents=control_latents,
                    control_camera_latents=control_camera_latents if args.train_mode == "control_camera_ref" else None,
                    control_camera_mask=control_camera_mask if args.train_mode == "control_camera_ref" else None,
                    action_map_mask=action_map_mask,
                    indices=indices,
                )
                with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=accelerator.device):
                    noise_pred = transformer3d(
                        x=noisy_latents,
                        context=prompt_embeds,
                        t=timesteps,
                        seq_len=seq_len,
                        y=control_latents,
                        y_camera=control_camera_latents if args.train_mode == "control_camera_ref" else None,
                        y_camera_mask=control_camera_mask if args.train_mode == "control_camera_ref" and control_camera_latents is not None else None,
                        arm_action=arm_action_values,
                        arm_action_mask=method1_main_arm_action_mask,
                        full_ref=full_ref if args.add_full_ref_image_in_self_attention else None,
                    )
                debug_heartbeat(
                    "after_transformer_forward",
                    epoch=epoch,
                    dataloader_step=step,
                    noise_pred=noise_pred,
                )
                
                def custom_mse_loss(noise_pred, target, weighting=None, threshold=50):
                    noise_pred = noise_pred.float()
                    target = target.float()
                    diff = noise_pred - target
                    mse_loss = F.mse_loss(noise_pred, target, reduction='none')
                    mask = (diff.abs() <= threshold).float()
                    masked_loss = mse_loss * mask
                    if weighting is not None:
                        masked_loss = masked_loss * weighting
                    final_loss = masked_loss.mean()
                    return final_loss
                
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                method1_rho = None
                method1_uniform_loss = None
                if args.enable_method1_focused_loss:
                    (
                        loss,
                        method1_rho,
                        method1_uniform_loss,
                        _,
                        _,
                    ) = method1_focused_flow_loss(
                        noise_pred,
                        target,
                        method1_effect_map,
                        active_mask=method1_active_mask,
                        eps=args.method1_eps,
                        mse_threshold=args.method1_mse_threshold,
                        exclude_first_frame=True,
                        loss_variant=args.method1_loss_variant,
                    )
                    if args.method1_log_stats and accelerator.is_main_process and accelerator.sync_gradients:
                        print(
                            f"method1_stats step={global_step} "
                            f"variant={args.method1_loss_variant} "
                            f"rho_mean={method1_rho.float().mean().item():.6f} "
                            f"rho_max={method1_rho.float().max().item():.6f} "
                            f"active_ratio={method1_active_mask.float().mean().item():.6f} "
                            f"focused_loss={loss.detach().item():.6f} "
                            f"uniform_loss={method1_uniform_loss.item():.6f}",
                            flush=True,
                        )
                else:
                    loss = custom_mse_loss(noise_pred.float(), target.float(), weighting.float())
                    loss = loss.mean()
                debug_heartbeat(
                    "after_loss",
                    epoch=epoch,
                    dataloader_step=step,
                    loss=loss.detach(),
                    weighting=weighting,
                )

                if args.motion_sub_loss and noise_pred.size()[2] > 2:
                    gt_sub_noise = noise_pred[:, :, 1:].float() - noise_pred[:, :, :-1].float()
                    pre_sub_noise = target[:, :, 1:].float() - target[:, :, :-1].float()
                    sub_loss = F.mse_loss(gt_sub_noise, pre_sub_noise, reduction="mean")
                    loss = loss * (1 - args.motion_sub_loss_ratio) + sub_loss * args.motion_sub_loss_ratio

                # Gather the losses across all processes for logging (if we use distributed training).
                debug_heartbeat("before_loss_gather", epoch=epoch, dataloader_step=step, loss=loss.detach())
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                debug_heartbeat("after_loss_gather", epoch=epoch, dataloader_step=step, avg_loss=avg_loss.detach())
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                debug_heartbeat("before_backward", epoch=epoch, dataloader_step=step, loss=loss.detach())
                accelerator.backward(loss)
                debug_heartbeat("after_backward", epoch=epoch, dataloader_step=step)
                if accelerator.sync_gradients:
                    if not args.use_deepspeed and not args.use_fsdp:
                        trainable_params_grads = [p.grad for p in trainable_params if p.grad is not None]
                        trainable_params_total_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in trainable_params_grads]), 2)
                        max_grad_norm = linear_decay(args.max_grad_norm * args.initial_grad_norm_ratio, args.max_grad_norm, args.abnormal_norm_clip_start, global_step)
                        if trainable_params_total_norm / max_grad_norm > 5 and global_step > args.abnormal_norm_clip_start:
                            actual_max_grad_norm = max_grad_norm / min((trainable_params_total_norm / max_grad_norm), 10)
                        else:
                            actual_max_grad_norm = max_grad_norm
                    else:
                        actual_max_grad_norm = args.max_grad_norm

                    if not args.use_deepspeed and not args.use_fsdp and args.report_model_info and accelerator.is_main_process:
                        if trainable_params_total_norm > 1 and global_step > args.abnormal_norm_clip_start:
                            for name, param in transformer3d.named_parameters():
                                if param.requires_grad:
                                    writer.add_scalar(f'gradients/before_clip_norm/{name}', param.grad.norm(), global_step=global_step)

                    debug_heartbeat("before_clip_grad_norm", epoch=epoch, dataloader_step=step)
                    norm_sum = accelerator.clip_grad_norm_(trainable_params, actual_max_grad_norm)
                    debug_heartbeat("after_clip_grad_norm", epoch=epoch, dataloader_step=step, norm_sum=norm_sum)
                    if not args.use_deepspeed and not args.use_fsdp and args.report_model_info and accelerator.is_main_process:
                        writer.add_scalar(f'gradients/norm_sum', norm_sum, global_step=global_step)
                        writer.add_scalar(f'gradients/actual_max_grad_norm', actual_max_grad_norm, global_step=global_step)
                debug_heartbeat("before_optimizer_step", epoch=epoch, dataloader_step=step)
                optimizer.step()
                debug_heartbeat("after_optimizer_step", epoch=epoch, dataloader_step=step)
                lr_scheduler.step()
                optimizer.zero_grad()
                debug_heartbeat("after_scheduler_zero_grad", epoch=epoch, dataloader_step=step)

            synchronize_for_timing()
            iter_end = time.perf_counter()
            if args.benchmark_timing_path:
                benchmark_update_data_time += data_time
                benchmark_update_compute_time += iter_end - compute_start
                benchmark_update_vae_time += vae_encode_time
                benchmark_update_micro_steps += 1

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:

                if args.use_ema:
                    ema_transformer3d.step(transformer3d.parameters())
                progress_bar.update(1)
                global_step += 1
                logged_train_loss = train_loss
                accelerator.log({"train_loss": logged_train_loss}, step=global_step)
                debug_heartbeat(
                    "after_progress_update",
                    epoch=epoch,
                    dataloader_step=step,
                    force=(
                        global_step <= debug_heartbeat_stop_step
                        or (
                            debug_progress_every_steps > 0
                            and global_step % debug_progress_every_steps == 0
                        )
                    ),
                    logged_train_loss=logged_train_loss,
                    lr=lr_scheduler.get_last_lr()[0],
                )
                if timing_file is not None:
                    timing_file.write(
                        json.dumps(
                            {
                                "global_step": global_step,
                                "epoch": epoch,
                                "micro_steps": benchmark_update_micro_steps,
                                "data_time_s": benchmark_update_data_time,
                                "compute_time_s": benchmark_update_compute_time,
                                "vae_encode_time_s": benchmark_update_vae_time,
                                "iter_time_s": benchmark_update_data_time + benchmark_update_compute_time,
                                "loss": logged_train_loss,
                                "lr": lr_scheduler.get_last_lr()[0],
                            }
                        )
                        + "\n"
                    )
                benchmark_update_data_time = 0.0
                benchmark_update_compute_time = 0.0
                benchmark_update_vae_time = 0.0
                benchmark_update_micro_steps = 0
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if args.use_deepspeed or args.use_fsdp or accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            if accelerator.is_main_process:
                                checkpoints = os.listdir(args.output_dir)
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                                if len(checkpoints) >= args.checkpoints_total_limit:
                                    num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    logger.info(
                                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                    )
                                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                        shutil.rmtree(removing_checkpoint, ignore_errors=True)
                            if args.use_deepspeed or args.use_fsdp:
                                accelerator.wait_for_everyone()

                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        debug_heartbeat("before_save_state", epoch=epoch, dataloader_step=step, force=True, save_path=save_path)
                        accelerator.save_state(save_path)
                        debug_heartbeat("after_save_state", epoch=epoch, dataloader_step=step, force=True, save_path=save_path)
                        logger.info(f"Saved state to {save_path}")

                if args.validation_prompts is not None and global_step % args.validation_steps == 0:
                    if args.use_ema:
                        # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                        ema_transformer3d.store(transformer3d.parameters())
                        ema_transformer3d.copy_to(transformer3d.parameters())
                    log_validation(
                        vae,
                        text_encoder,
                        tokenizer,
                        transformer3d,
                        args,
                        config,
                        accelerator,
                        weight_dtype,
                        global_step,
                    )
                    if args.use_ema:
                        # Switch back to the original transformer3d parameters.
                        ema_transformer3d.restore(transformer3d.parameters())

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            last_iter_end = iter_end

            if global_step >= args.max_train_steps:
                break

        if args.validation_prompts is not None and epoch % args.validation_epochs == 0:
            if args.use_ema:
                # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                ema_transformer3d.store(transformer3d.parameters())
                ema_transformer3d.copy_to(transformer3d.parameters())
            log_validation(
                vae,
                text_encoder,
                tokenizer,
                transformer3d,
                args,
                config,
                accelerator,
                weight_dtype,
                global_step,
            )
            if args.use_ema:
                # Switch back to the original transformer3d parameters.
                ema_transformer3d.restore(transformer3d.parameters())

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer3d = unwrap_model(transformer3d)
        if args.use_ema:
            ema_transformer3d.copy_to(transformer3d.parameters())

    if not args.skip_final_checkpoint and (args.use_deepspeed or args.use_fsdp or accelerator.is_main_process):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        accelerator.save_state(save_path)
        logger.info(f"Saved state to {save_path}")

    if timing_file is not None:
        timing_file.close()
    accelerator.end_training()


if __name__ == "__main__":
    main()
