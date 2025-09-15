from typing import Optional, List
import numpy as np
import torch
from cv2 import dilate
from diffusers import DDIMScheduler, StableDiffusionPipeline
from tqdm import tqdm


def get_stable_diffusion_model(args):
    """Initialize Stable Diffusion model with appropriate configuration"""
    device = torch.device(f'cuda:{args.gpu_id}') if torch.cuda.is_available() else torch.device('cpu')

    if args.real_image_path != "":
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False
        )
        ldm_stable = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base",
            use_auth_token=args.auth_token,
            scheduler=scheduler
        ).to(device)
    else:
        ldm_stable = StableDiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1-base",
            use_auth_token=args.auth_token
        ).to(device)

    return ldm_stable


def get_stable_diffusion_config(args):
    """Create configuration dictionary for diffusion model"""
    return {
        "low_resource": args.low_resource if hasattr(args, 'low_resource') else False,
        "num_diffusion_steps": args.num_diffusion_steps,
        "guidance_scale": args.guidance_scale,
        "max_num_words": args.max_num_words if hasattr(args, 'max_num_words') else 77
    }
