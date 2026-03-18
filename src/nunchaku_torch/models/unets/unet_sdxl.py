import torch
from diffusers.models.attention import FeedForward

from ..attention import _patch_linear
from ..linear import SVDQW4A4Linear


class NunchakuSDXLFeedForward(FeedForward):
    def __init__(self, ff: FeedForward, **kwargs):
        super(FeedForward, self).__init__()
        self.net = _patch_linear(ff.net, SVDQW4A4Linear, **kwargs)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states
