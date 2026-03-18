import diffusers
import torch
from packaging.version import Version
from torch import nn


def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    assert dim % 2 == 0, "The dimension must be even."

    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)

    batch_size, seq_length = pos.shape
    out = torch.einsum("...n,d->...nd", pos, omega)

    cos_out = torch.cos(out)
    sin_out = torch.sin(out)
    stacked_out = torch.stack([sin_out, cos_out], dim=-1)
    out = stacked_out.view(batch_size, -1, dim // 2, 1, 2)

    return out.float()


class NunchakuFluxPosEmbed(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super(NunchakuFluxPosEmbed, self).__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if Version(diffusers.__version__) >= Version("0.31.0"):
            ids = ids[None, ...]
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)


def pack_rotemb(rotemb: torch.Tensor) -> torch.Tensor:
    assert rotemb.dtype == torch.float32
    B = rotemb.shape[0]
    M = rotemb.shape[1]
    D = rotemb.shape[2] * 2
    assert rotemb.shape == (B, M, D // 2, 1, 2)
    assert M % 16 == 0
    assert D % 8 == 0
    rotemb = rotemb.reshape(B, M // 16, 16, D // 8, 8)
    rotemb = rotemb.permute(0, 1, 3, 2, 4)
    rotemb = rotemb.reshape(*rotemb.shape[0:3], 2, 8, 4, 2)
    rotemb = rotemb.permute(0, 1, 2, 4, 5, 3, 6)
    rotemb = rotemb.contiguous()
    rotemb = rotemb.view(B, M, D)
    return rotemb
