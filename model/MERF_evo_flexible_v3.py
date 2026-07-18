"""Flexible CR evolution policy v3: full MERF backbone without an action-bias head."""

import torch
import torch.nn.functional as F

from model.MERF_v6 import MERF as BaseMERF


class MERF(BaseMERF):
    """Expose the binary CR action policy while keeping only v1 backbone parameters."""

    def get_binary_log_probs(self, policy_batch, dataset, device, temperature=1.0):
        qs = super().choose_best_action(policy_batch, device)
        site_indices, stay_actions, forward_actions = dataset.get_site_action_tensors(
            policy_batch, device
        )
        site_qs = qs[0, site_indices]
        stay_qs = site_qs.gather(1, stay_actions.unsqueeze(-1)).squeeze(-1)
        forward_qs = site_qs.gather(1, forward_actions.unsqueeze(-1)).squeeze(-1)
        binary_logits = -torch.stack([stay_qs, forward_qs], dim=-1)
        if temperature != 1.0:
            binary_logits = binary_logits / temperature
        return F.log_softmax(binary_logits, dim=-1)
