import torch
from scipy.signal import medfilt2d

class PromptSwitching:
    def __init__(self, args, object_of_interest_index, avg_cross_attn=None):
        """
        Class for implementing prompt switching during diffusion process

        Args:
            args: Command line arguments
            object_of_interest_index: Index of the target object in the prompt
            avg_cross_attn: Average cross-attention maps from previous steps
        """
        self.args = args
        self.object_of_interest_index = object_of_interest_index

        # Get parameters from args with consistent naming
        self.start_prompt_range = args.start_prompt_range if hasattr(args, 'start_prompt_range') else 0
        self.end_prompt_range = args.end_prompt_range if hasattr(args, 'end_prompt_range') else 0
        self.avg_cross_attn = avg_cross_attn
        self.low_resource = args.low_resource if hasattr(args, 'low_resource') else False

        # Time step parameters
        self.prompt_injection_steps = args.prompt_injection_steps if hasattr(args, 'prompt_injection_steps') else []

        # Object preservation parameters
        self.objects_to_preserve = []
        if hasattr(args, 'preserve_objects'):
            self.objects_to_preserve = [args.prompt.split().index(o) + 1 for o in args.preserve_objects]
        self.obj_pixels_injection_threshold = args.obj_pixels_injection_threshold if hasattr(args,
                                                                                             'obj_pixels_injection_threshold') else 0.5

    def get_context_for_v(self, t, context, other_context):
        """
        Control context injection for value vectors

        Args:
            t: Current diffusion step
            context: Base context
            other_context: Alternative context

        Returns:
            Modified context vector
        """
        if other_context is not None and self.start_prompt_range <= t < self.end_prompt_range:
            if self.low_resource:
                return other_context
            else:
                v_context = context.clone()
                v_context[v_context.shape[0] // 2:] = other_context
                return v_context
        return context

    def get_cross_attn(self, diffusion_model_wrapper, t, attn, place_in_unet, batch_size):
        """
        Inject modifications into cross-attention maps

        Args:
            diffusion_model_wrapper: Diffusion model wrapper instance
            t: Current diffusion step
            attn: Attention maps
            place_in_unet: UNet location ("up", "mid", "down")
            batch_size: Batch size

        Returns:
            Modified attention maps
        """
        if t in self.prompt_injection_steps:
            if self.low_resource:
                # Apply median filter to attention maps
                filtered = torch.from_numpy(
                    medfilt2d(attn[:, :, self.object_of_interest_index].cpu().numpy(), kernel_size=3)
                ).to(attn.device)
                attn[:, :, self.object_of_interest_index] = 0.2 * filtered + 0.8 * attn[:, :,
                                                                                   self.object_of_interest_index]
            else:
                # Process second half of batch with filtered attention
                min_h = attn.shape[0] // 2
                filtered = torch.from_numpy(
                    medfilt2d(attn[min_h:, :, self.object_of_interest_index].cpu().numpy(), kernel_size=3)
                ).to(attn.device)
                attn[min_h:, :, self.object_of_interest_index] = 0.2 * filtered + 0.8 * attn[min_h:, :,
                                                                                        self.object_of_interest_index]
        return attn

    def get_self_attn(self, diffusion_model_wrapper, t, attn, place_in_unet, batch_size):
        """
        Inject modifications into self-attention maps

        Args:
            diffusion_model_wrapper: Diffusion model wrapper instance
            t: Current diffusion step
            attn: Attention maps
            place_in_unet: UNet location ("up", "mid", "down")
            batch_size: Batch size

        Returns:
            Modified attention maps
        """
        if attn.shape[1] <= 32 ** 2 and self.avg_cross_attn is not None:
            key = f"{place_in_unet}_cross"
            if hasattr(diffusion_model_wrapper, f'{key}_index'):
                cr = self.avg_cross_attn[key][getattr(diffusion_model_wrapper, f'{key}_index')]
                setattr(diffusion_model_wrapper, f'{key}_index', getattr(diffusion_model_wrapper, f'{key}_index') + 1)

                # Apply attention mask
                h = attn.shape[0] // batch_size
                tokens = self.objects_to_preserve
                normalized_cross_attn = (cr - cr.min()) / (cr.max() - cr.min() + 1e-8)
                mask = torch.zeros_like(attn[0])

                # Create mask for preserved objects
                for tk in tokens:
                    mask_tk_in = torch.unique(
                        (normalized_cross_attn[:, :, tk] > self.obj_pixels_injection_threshold).nonzero(as_tuple=True)[
                            1])
                    mask[mask_tk_in, :] = 1
                    mask[:, mask_tk_in] = 1

                # Remove object from self mask if enabled
                if hasattr(self.args, 'remove_obj_from_self_mask') and self.args.remove_obj_from_self_mask:
                    obj_patches = torch.unique(
                        (normalized_cross_attn[:, :,
                         self.object_of_interest_index] > self.obj_pixels_injection_threshold).nonzero(as_tuple=True)[1]
                    )
                    mask[obj_patches, :] = 0
                    mask[:, obj_patches] = 0

                # Apply mask to attention maps
                attn[h:] = attn[h:] * (1 - mask) + attn[:h].repeat(batch_size - 1, 1, 1) * mask

        return attn