import argparse
import random
import os
import numpy as np
import torch
from torch.backends import cudnn
from text_driven_attack import tdattack
from load_diffusion import get_stable_diffusion_model, get_stable_diffusion_config
from attention_Control import AttentionControlEdit

def parse_args():
    parser = argparse.ArgumentParser(description="Text-Driven DM-based Unrestricted Adversarial Attack")

    parser.add_argument('--dataset_name', default='imagenet_compatible', type=str,
                        help='Dataset name')
    parser.add_argument('--res', default=224, type=int, help='Input resolution')
    parser.add_argument('--iterations', default=20, type=int,
                        help='Iterations for optimizing adversarial image')
    parser.add_argument('--guidance_scale', default=3, type=float,
                        help='Guidance scale for diffusion model')
    parser.add_argument('--model_name', default='resnet', type=str,
                        help='Classifier model name')
    parser.add_argument('--images_root', default="data/images", type=str,
                        help='The clean images root directory')
    parser.add_argument('--label_path', default="data/labels.txt", type=str,
                        help='The clean images labels.txt')
    parser.add_argument('--save_path', default=r'C:\Users\PC\Desktop\output', type=str,
                        help='Path to save results')
    parser.add_argument('--error_prompt', default='adversarial_prompt.txt', type=str,
                        help='File with error prompts')

    # Attack parameters
    parser.add_argument('--attack_loss_weight', default=10, type=float,
                        help='Attack loss weight')
    parser.add_argument('--corr_loss_weight', default=1, type=int,
                        help='Corr loss weight factor')
    parser.add_argument('--similarity_loss_weight', default=100, type=int,
                        help='Similarity loss weight factor')

    # Timestep parameters
    parser.add_argument('--num_diffusion_steps', default=20, type=int,
                        help='Total DDIM sampling steps')
    parser.add_argument('--T1', default=15, type=int, help='Early Stage End Steps')
    parser.add_argument('--ddim_inversion_steps', default=5, type=int,
                        help='Steps for DDIM inversion')

    # Prompt injection parameters
    parser.add_argument('--use_prompt_switching', action='store_true',
                        help='Enable prompt switching')
    parser.add_argument('--prompt_injection_steps', default=15, type=int, nargs='+',
                        help='Steps to inject error prompts')

    return parser.parse_args()

def main():
    args = parse_args()

    # Set random seed for reproducibility
    seed_torch(42)

    # Initialize diffusion model with specified configuration
    diffusion_model = get_stable_diffusion_model(args)
    model_config = get_stable_diffusion_config(args)

    # Load error prompts from file if exists
    error_prompts = []
    if args.error_prompt and os.path.exists(args.error_prompt):
        with open(args.error_prompt, 'r') as f:
            error_prompts = [line.strip() for line in f.readlines()]

    # Initialize attention controller with specified parameters
    controller = AttentionControlEdit(
        num_steps=args.num_diffusion_steps,
        self_replace_steps=(args.num_diffusion_steps, args.T1),
        res=args.res
    )

    # Execute adversarial attack
    adv_image, clean_acc, adv_acc = tdattack(
        model=diffusion_model,
        label=args.label,
        controller=controller,
        num_inference_steps=args.num_diffusion_steps,
        guidance_scale=args.guidance_scale,
        image=args.image,
        save_path=args.save_path,
        res=args.res,
        model_name=args.model_name,
        prompt_injection_steps=args.prompt_injection_steps,
        iterations=args.iterations,
        args=args
    )

    print(f"Adversarial attack completed. Clean accuracy: {clean_acc * 100}%, Adversarial accuracy: {adv_acc * 100}%")

if __name__ == "__main__":
    main()
