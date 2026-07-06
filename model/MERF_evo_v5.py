import torch
import torch.nn as nn
import torch.nn.functional as F

from model.MERF_v6 import MERF as BaseMERF


class MERF(BaseMERF):
    """MERF v6 with a site/action evolution adapter and no candidate memory."""

    def __init__(self, args):
        super().__init__(args)
        max_evo_sites = getattr(args, "max_evo_sites", 64)
        self.evo_bias_scale = getattr(args, "evo_bias_scale", 1.0)
        self.use_site_bias = getattr(args, "use_site_bias", True)
        self.site_action_bias = nn.Parameter(torch.zeros(max_evo_sites, 2))

    def get_binary_log_probs(self, policy_batch, dataset, device, temperature=1.0):
        qs = super().choose_best_action(policy_batch, device)
        site_indices, stay_actions, forward_actions = dataset.get_site_action_tensors(
            policy_batch, device
        )
        site_qs = qs[0, site_indices]
        stay_qs = site_qs.gather(1, stay_actions.unsqueeze(-1)).squeeze(-1)
        forward_qs = site_qs.gather(1, forward_actions.unsqueeze(-1)).squeeze(-1)
        binary_logits = -torch.stack([stay_qs, forward_qs], dim=-1)

        n_sites = binary_logits.size(0)
        if n_sites > self.site_action_bias.size(0):
            raise ValueError(
                f"Need {n_sites} evo sites, but max_evo_sites is {self.site_action_bias.size(0)}"
            )
        if self.use_site_bias:
            binary_logits = binary_logits + self.evo_bias_scale * self.site_action_bias[:n_sites]

        if temperature != 1.0:
            binary_logits = binary_logits / temperature
        return F.log_softmax(binary_logits, dim=-1)
