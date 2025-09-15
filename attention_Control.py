from typing import Union, Tuple
import torch
import abc


class AttentionControl(abc.ABC):
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def between_steps(self):
        """Operations to perform between diffusion steps"""
        return

    def reset(self):
        """Reset the controller state"""
        self.cur_step = 0
        self.cur_att_layer = 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        # Process attention for the second half of the batch (e.g., perturbed image)
        if self.cur_att_layer >= 0:
            h = attn.shape[0]
            self.forward(attn[h // 2:], is_cross, place_in_unet)

        # Advance layer counter
        self.cur_att_layer += 1
        # When all layers are processed, advance the step and run hooks
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn


class AttentionStore(AttentionControl):
    def __init__(self, res):
        """
        Store attention maps during the diffusion process.

        Args:
            res: Resolution for attention map storage.
        """
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.res = res

    @staticmethod
    def get_empty_store():
        """Initialize empty storage for attention maps"""
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        """Store attention maps during forward pass"""
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        # Only store attention maps for appropriate resolutions to avoid memory issues
        if attn.shape[1] <= (self.res // 16) ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        """Combine attention maps across steps"""
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] = self.step_store[key][i] + self.attention_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        """Get average attention maps across all steps"""
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in
                             self.attention_store}
        return average_attention

    def reset(self):
        """Reset the attention storage"""
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


class AttentionControlEdit(AttentionStore, abc.ABC):
    def __init__(self, num_steps: int,
                 self_replace_steps: Union[float, Tuple[float, float]], res):
        """
        Controller for attention-based adversarial editing.

        Args:
            num_steps: Total diffusion steps.
            self_replace_steps: Range for self-attention replacement (start, end).
                If float, interpreted as (0, value) in fraction of steps.
            res: Image resolution.
        """
        super(AttentionControlEdit, self).__init__(res)
        self.batch_size = 2
        # Convert replacement steps from fraction to absolute steps
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.loss = 0
        self.criterion = torch.nn.MSELoss()

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        """
        Process attention maps during diffusion.

        Args:
            attn: Attention map tensor.
            is_cross: Whether this is cross-attention.
            place_in_unet: UNet location ("up", "mid", "down").
        """
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        # Only modify attention during specific steps
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            # Split attention maps into base and modified
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if not is_cross:
                # Structural consistency loss for self-attention replacement
                self.loss += self.criterion(attn[1:], self.replace_self_attention(attn_base, attn_repalce))
            # Reshape attention maps back to original format
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn

    def replace_self_attention(self, attn_base, att_replace):
        """Replace self-attention maps with base attention"""
        return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
