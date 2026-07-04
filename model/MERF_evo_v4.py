import torch
import torch.nn as nn
import torch.nn.functional as F

from model.MERF_v6 import MERF as BaseMERF


class MERF(BaseMERF):
    """MERF v6 with optional predictor-supervised evolution adapters."""

    def __init__(self, args):
        super().__init__(args)
        max_evo_sites = getattr(args, "max_evo_sites", 64)
        max_evo_candidates = getattr(args, "max_evo_candidates", 70000)
        self.use_site_bias = getattr(args, "use_site_bias", False)
        self.use_candidate_bias = getattr(args, "use_candidate_bias", False)
        self.evo_bias_scale = getattr(args, "evo_bias_scale", 1.0)
        self.site_action_bias = nn.Parameter(torch.zeros(max_evo_sites, 2))
        self.candidate_score_bias = nn.Parameter(torch.zeros(max_evo_candidates))

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

    def get_candidate_scores(self, binary_log_probs, dataset):
        candidate_scores = dataset.candidate_log_probs(binary_log_probs, reduce="mean")
        n_candidates = candidate_scores.size(0)
        if n_candidates > self.candidate_score_bias.size(0):
            raise ValueError(
                f"Need {n_candidates} candidates, but max_evo_candidates is "
                f"{self.candidate_score_bias.size(0)}"
            )
        if self.use_candidate_bias:
            candidate_scores = candidate_scores + self.candidate_score_bias[:n_candidates].to(
                candidate_scores.device
            )
        return candidate_scores
