from typing import Optional
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm
from torch import optim
import other_attacks
from attention_utils import view_images, aggregate_attention
from prompt_switching import PromptSwitching
from prompt_seq_aligner import get_replacement_mapper, get_word_inds
import prompt_utils

def preprocess(image, res=512):
    """Preprocess image to tensor format with normalization"""
    image = image.resize((res, res), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)[:, :3, :, :].cuda()
    return 2.0 * image - 1.0


def encoder(image, model, res=512):
    """Encode image to latent space"""
    generator = torch.Generator().manual_seed(8888)
    image = preprocess(image, res)
    gpu_generator = torch.Generator(device=image.device)
    gpu_generator.manual_seed(generator.initial_seed())
    return 0.18215 * model.vae.encode(image).latent_dist.sample(generator=gpu_generator)


@torch.no_grad()
def ddim_reverse_sample(image, prompt, model, num_inference_steps: int = 20, guidance_scale: float = 2.5, res=512):
    """DDIM inversion to get latent code"""
    batch_size = 1
    max_length = 77
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    text_input = model.tokenizer(
        prompt[0],
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    context = [uncond_embeddings, text_embeddings]
    context = torch.cat(context)
    model.scheduler.set_timesteps(num_inference_steps)
    latents = encoder(image, model, res=res)
    timesteps = model.scheduler.timesteps.flip(0)
    all_latents = [latents]

    for t in tqdm(timesteps[:-1], desc="DDIM_inverse"):
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        next_timestep = t + model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
        alpha_bar_next = model.scheduler.alphas_cumprod[
            next_timestep] if next_timestep <= model.scheduler.config.num_train_timesteps else torch.tensor(0.0)
        reverse_x0 = (1 / torch.sqrt(model.scheduler.alphas_cumprod[t]) *
                      (latents - noise_pred * torch.sqrt(1 - model.scheduler.alphas_cumprod[t])))
        latents = reverse_x0 * torch.sqrt(alpha_bar_next) + torch.sqrt(1 - alpha_bar_next) * noise_pred
        all_latents.append(latents)
    return latents, all_latents


def register_attention_control(model, controller):
    """Register attention control hooks"""

    def ca_forward(self, place_in_unet):
        def forward(
                hidden_states: torch.FloatTensor,
                encoder_hidden_states: Optional[torch.FloatTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                temb: Optional[torch.FloatTensor] = None,
        ):
            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)
            batch_size, sequence_length, _ = (
                hidden_states.shape
                if encoder_hidden_states is None
                else encoder_hidden_states.shape
            )
            if attention_mask is not None:
                attention_mask = self.prepare_attention_mask(
                    attention_mask, sequence_length, batch_size
                )
                attention_mask = attention_mask.view(
                    batch_size, self.heads, -1, attention_mask.shape[-1]
                )

            if self.group_norm is not None:
                hidden_states = self.group_norm(
                    hidden_states.transpose(1, 2)
                ).transpose(1, 2)

            query = self.to_q(hidden_states)
            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = hidden_states if encoder_hidden_states is None else encoder_hidden_states
            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            def reshape_heads_to_batch_dim(tensor):
                batch_size, seq_len, dim = tensor.shape
                head_size = self.heads
                tensor = tensor.reshape(
                    batch_size, seq_len, head_size, dim // head_size
                ).permute(0, 2, 1, 3).reshape(
                    batch_size * head_size, seq_len, dim // head_size
                )
                return tensor

            query, key, value = map(reshape_heads_to_batch_dim, [query, key, value])
            sim = torch.einsum("b i d, b j d -> b i j", query, key) * self.scale
            attn = sim.softmax(dim=-1)
            attn = controller(attn, is_cross, place_in_unet)

            def reshape_batch_dim_to_heads(tensor):
                batch_size, seq_len, dim = tensor.shape
                head_size = self.heads
                tensor = tensor.reshape(
                    batch_size // head_size, head_size, seq_len, dim
                ).permute(0, 2, 1, 3).reshape(
                    batch_size // head_size, seq_len, dim * head_size
                )
                return tensor

            out = torch.einsum("b i j, b j d -> b i d", attn, value)
            out = reshape_batch_dim_to_heads(out)
            out = self.to_out[0](out)
            out = self.to_out[1](out)
            out = out / self.rescale_output_factor
            return out

        return forward

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == "Attention":
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, "children"):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")
    controller.num_att_layers = cross_att_count


def reset_attention_control(model):
    def ca_forward(self):
        def forward(
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            temb: Optional[torch.FloatTensor] = None,
            # scale: float = 1.0,
        ):
            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)

            batch_size, sequence_length, _ = (
                hidden_states.shape
                if encoder_hidden_states is None
                else encoder_hidden_states.shape
            )

            if attention_mask is not None:
                attention_mask = self.prepare_attention_mask(
                    attention_mask, sequence_length, batch_size
                )
                # scaled_dot_product_attention expects attention_mask shape to be
                # (batch, heads, source_length, target_length)
                attention_mask = attention_mask.view(
                    batch_size, self.heads, -1, attention_mask.shape[-1]
                )  # type: ignore

            if self.group_norm is not None:
                hidden_states = self.group_norm(
                    hidden_states.transpose(1, 2)
                ).transpose(1, 2)

            query = self.to_q(hidden_states)
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(
                    encoder_hidden_states
                )

            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            def reshape_heads_to_batch_dim(tensor):
                batch_size, seq_len, dim = tensor.shape
                head_size = self.heads
                tensor = tensor.reshape(
                    batch_size, seq_len, head_size, dim // head_size
                )
                tensor = tensor.permute(0, 2, 1, 3).reshape(
                    batch_size * head_size, seq_len, dim // head_size
                )
                return tensor

            query = reshape_heads_to_batch_dim(query)
            key = reshape_heads_to_batch_dim(key)
            value = reshape_heads_to_batch_dim(value)

            sim = torch.einsum("b i d, b j d -> b i j", query, key) * self.scale

            attn = sim.softmax(dim=-1)
            out = torch.einsum("b i j, b j d -> b i d", attn, value)

            def reshape_batch_dim_to_heads(tensor):
                batch_size, seq_len, dim = tensor.shape
                head_size = self.heads
                tensor = tensor.reshape(
                    batch_size // head_size, head_size, seq_len, dim
                )
                tensor = tensor.permute(0, 2, 1, 3).reshape(
                    batch_size // head_size, seq_len, dim * head_size
                )
                return tensor

            out = reshape_batch_dim_to_heads(out)
            out = self.to_out[0](out)
            out = self.to_out[1](out)

            out = out / self.rescale_output_factor

            return out

        return forward

    def register_recr(net_):
        if net_.__class__.__name__ == "Attention":
            net_.forward = ca_forward(net_)
        elif hasattr(net_, "children"):
            for net__ in net_.children():
                register_recr(net__)

    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            register_recr(net[1])
        elif "up" in net[0]:
            register_recr(net[1])
        elif "mid" in net[0]:
            register_recr(net[1])

def init_latent(latent, model, height, width, batch_size):
    """Initialize latent space with proper dimensions"""
    latents = latent.expand(batch_size, model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents


def diffusion_step(model, latents, context, t, guidance_scale):
    """Perform a single diffusion step with guidance"""
    latents_input = torch.cat([latents] * 2)
    noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
    noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    return latents


def latent2image(vae, latents):
    """Convert latent representation to image"""
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


def get_prompt_injection_context(model, prompt):
    """Generate prompt injection context for specific timesteps"""
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(model.device)
    return model.text_encoder(text_input)[0]


@torch.enable_grad()
def tdattack(
        model,
        label,
        controller,
        num_inference_steps: int = 50,
        guidance_scale: float = 3,
        image=None,
        model_name="resnet",
        save_path=r"C:\Users\PC\Desktop\output",
        res=224,
        prompt_injection_steps=45,
        iterations=20,
        args=None
):
    """
    Args:
        model: Diffusion model
        label: True label for attack
        controller: Attention controller
        num_inference_steps: Number of diffusion steps
        guidance_scale: Guidance scale for diffusion
        image: Input image for attack
        model_name: Target classifier name
        save_path: Path to save results
        res: Input resolution
        prompt_injection_steps: Timesteps to inject error prompts
        iterations: Optimization iterations
        args: Additional command-line arguments
    """
    # Dataset label processing
    if args.dataset_name == "imagenet_compatible":
        from dataset_caption import imagenet_label
    else:
        raise NotImplementedError(f"Dataset {args.dataset_name} not implemented")

    label = torch.from_numpy(label).long().cuda()
    model.vae.requires_grad_(False)
    model.text_encoder.requires_grad_(False)
    model.unet.requires_grad_(False)
    classifier = other_attacks.model_selection(model_name).eval()
    classifier.requires_grad_(False)

    # Image preprocessing and initial classification
    height = width = res
    test_image = image.resize((height, height), resample=Image.LANCZOS)
    test_image = np.float32(test_image) / 255.0
    test_image = test_image[:, :, :3]
    test_image -= (0.485, 0.456, 0.406)
    test_image /= (0.229, 0.224, 0.225)
    test_image = test_image.transpose((2, 0, 1))
    test_image = torch.from_numpy(test_image).unsqueeze(0)
    pred = classifier(test_image.cuda())
    pred_accuracy_clean = (torch.argmax(pred, 1).detach() == label).sum().item() / len(label)
    print(f"Accuracy on benign examples: {pred_accuracy_clean * 100}%")

    # Generate proxy words and prompts
    proxy_words, prompts, proxy_prompts = prompt_utils.get_proxy_prompts(args, model)
    print("Proxy words:", proxy_words)
    print("Proxy prompts:", proxy_prompts)

    # Get token replacement mapping
    prompt_tokens = model.tokenizer(prompts, return_tensors="pt").input_ids[0]
    proxy_tokens = model.tokenizer([pp["prompt"] for pp in proxy_prompts], return_tensors="pt").input_ids
    replacement_mapper = get_replacement_mapper(prompt_tokens, proxy_tokens[0])
    word_inds = get_word_inds(model.tokenizer, prompts[0])
    print("Replacement mapper:", replacement_mapper)
    print("Word indices:", word_inds)

    # Prompt switching setup
    prompt_switcher = PromptSwitching(
        args=args,
        object_of_interest_index=45,
        avg_cross_attn=None
    )

    # DDIM inversion sampling
    latent, inversion_latents = ddim_reverse_sample(image, prompts, model, num_inference_steps, 0, res=height)
    inversion_latents = inversion_latents[::-1]
    batch_size = len(prompts)
    latent = inversion_latents[args.ddim_inversion_steps - 1]

    # Optimize unconditional embeddings
    max_length = 77
    uncond_input = model.tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    text_input = model.tokenizer(prompts, padding="max_length", max_length=model.tokenizer.model_max_length,
                                 truncation=True, return_tensors="pt")
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    all_uncond_emb = []
    latent, latents = init_latent(latent, model, height, width, batch_size)
    uncond_embeddings.requires_grad_(True)
    optimizer = optim.AdamW([uncond_embeddings], lr=1e-1)
    loss_func = torch.nn.MSELoss()
    context = torch.cat([uncond_embeddings, text_embeddings])

    # Optimization loop for unconditional embeddings
    for ind, t in enumerate(tqdm(model.scheduler.timesteps[1 + args.ddim_inversion_steps - 1:],
                                 desc="Optimize_uncond_embed")):
        for _ in range(10 + 2 * ind):
            out_latents = diffusion_step(model, latents, context, t, guidance_scale)
            optimizer.zero_grad()
            loss = loss_func(out_latents, inversion_latents[args.ddim_inversion_steps - 1 + ind + 1])
            loss.backward()
            optimizer.step()
            context = torch.cat([uncond_embeddings, text_embeddings])
        with torch.no_grad():
            latents = diffusion_step(model, latents, context, t, guidance_scale).detach()
            all_uncond_emb.append(uncond_embeddings.detach().clone())

    # Register attention control
    register_attention_control(model, controller)
    text_input = model.tokenizer(prompts, padding="max_length", max_length=model.tokenizer.model_max_length,
                                 truncation=True, return_tensors="pt")
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    context = [[torch.cat([all_uncond_emb[i]] * batch_size), text_embeddings] for i in range(len(all_uncond_emb))]
    context = [torch.cat(i) for i in context]
    original_latent = latent.clone()
    latent.requires_grad_(True)
    optimizer = optim.AdamW([latent], lr=1e-2)
    cross_entro = torch.nn.CrossEntropyLoss()
    init_image = preprocess(image, res)

    # Attack loop
    pbar = tqdm(range(iterations), desc="Iterations")
    for iteration in pbar:
        controller.reset()
        latents = torch.cat([original_latent, latent])

        # Dynamic prompt injection
        current_prompt = prompts
        if args.use_prompt_switching and iteration in args.prompt_injection_steps:
            current_prompt = [args.error_prompt]

        # Diffusion process with attention control
        for ind, t in enumerate(model.scheduler.timesteps[1 + args.ddim_inversion_steps - 1:]):
            if ind in args.prompt_injection_steps:
                injection_context = get_prompt_injection_context(model, current_prompt)
                injection_context = prompt_switcher.get_context_for_v(t, context[ind], injection_context)
            else:
                injection_context = context[ind]
            latents = diffusion_step(model, latents, injection_context, t, guidance_scale)

        # Decode latents to image directly
        init_out_image = model.vae.decode(1 / 0.18215 * latents)['sample'][1:]

        # Classifier prediction and loss calculation
        out_image = (init_out_image / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1)
        mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=out_image.dtype, device=out_image.device)
        std = torch.as_tensor([0.229, 0.224, 0.225], dtype=out_image.dtype, device=out_image.device)
        out_image = out_image.sub(mean).div(std).permute(0, 3, 1, 2)
        pred = classifier(out_image) if args.dataset_name == "imagenet_compatible" else classifier(out_image) / 10

        attack_loss = -cross_entro(pred, label) * args.attack_loss_weight

        corr_loss = (torch.tensor(0.0, device=latents.device) * args.corr_loss_weight)

        struct_loss = controller.loss * args.structure_loss_weight

        total_loss = attack_loss + corr_loss + struct_loss

        # Optimization step + logging
        pbar.set_postfix_str(
            f"attack_loss: {attack_loss.item():.5f}, "
            f"corr_loss: {corr_loss.item():.5f}, "
            f"struct_loss: {struct_loss.item():.5f}"
        )
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    #  Post-processing and visualization
    with torch.no_grad():
        controller.reset()
        latents = torch.cat([original_latent, latent])
        for ind, t in enumerate(model.scheduler.timesteps[1 + args.ddim_inversion_steps - 1:]):
            latents = diffusion_step(model, latents, context[ind], t, guidance_scale)

    # Image generation and saving
    out_image = model.vae.decode(1 / 0.18215 * latents.detach())['sample'][1:]
    out_image = (out_image / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1)
    mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=out_image.dtype, device=out_image.device)
    std = torch.as_tensor([0.229, 0.224, 0.225], dtype=out_image.dtype, device=out_image.device)
    out_image = out_image.sub(mean).div(std).permute(0, 3, 1, 2)
    pred = classifier(out_image)
    pred_label = torch.argmax(pred, 1).detach()
    pred_accuracy = (pred_label == label).sum().item() / len(label)
    print(f"Accuracy on adversarial examples: {pred_accuracy * 100}%")

    # Visualization and saving
    image = latent2image(model.vae, latents.detach())
    perturbed = image[1:]  # use generated images directly
    view_images(perturbed, show=False, save_path=save_path + "_adv_image.png")

    reset_attention_control(model)
    return perturbed[0], pred_accuracy_clean, pred_accuracy

