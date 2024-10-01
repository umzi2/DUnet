import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.utils import spectral_norm


class DySample(nn.Module):
    """Adapted from 'Learning to Upsample by Learning to Sample':
    https://arxiv.org/abs/2308.15085
    https://github.com/tiny-smart/dysample
    """

    def __init__(
            self,
            in_channels: int,
            out_ch: int,
            scale: int = 2,
            groups: int = 4,
            end_convolution: bool = True,
    ) -> None:
        super().__init__()

        try:
            assert in_channels >= groups
            assert in_channels % groups == 0
        except:
            msg = "Incorrect in_channels and groups values."
            raise ValueError(msg)

        out_channels = 2 * groups * scale ** 2
        self.scale = scale
        self.groups = groups
        self.end_convolution = end_convolution
        if end_convolution:
            self.end_conv = nn.Conv2d(in_channels, out_ch, kernel_size=1)

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        if self.training:
            nn.init.trunc_normal_(self.offset.weight, std=0.02)
            nn.init.constant_(self.scope.weight, val=0)

        self.register_buffer("init_pos", self._init_pos())

    def _init_pos(self) -> Tensor:
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return (
            torch.stack(torch.meshgrid([h, h], indexing="ij"))
            .transpose(1, 2)
            .repeat(1, self.groups, 1)
            .reshape(1, -1, 1, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5

        coords = (
            torch.stack(torch.meshgrid([coords_w, coords_h], indexing="ij"))
            .transpose(1, 2)
            .unsqueeze(1)
            .unsqueeze(0)
            .type(x.dtype)
            .to(x.device, non_blocking=True)
        )
        normalizer = torch.tensor(
            [W, H], dtype=x.dtype, device=x.device, pin_memory=True
        ).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1

        coords = (
            F.pixel_shuffle(coords.reshape(B, -1, H, W), self.scale)
            .view(B, 2, -1, self.scale * H, self.scale * W)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
        )
        output = F.grid_sample(
            x.reshape(B * self.groups, -1, H, W),
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        ).view(B, -1, self.scale * H, self.scale * W)

        if self.end_convolution:
            output = self.end_conv(output)

        return output

class Down(nn.Sequential):
    def __init__(self,dim):
        super().__init__(
            spectral_norm(nn.Conv2d(dim,dim*2,3,2,1)),
            nn.Mish(True)
        )
class Up(nn.Sequential):
    def __init__(self,dim):
        super().__init__(
            DySample(dim,dim,2,4,False),
            spectral_norm(nn.Conv2d(dim,dim//2,3,1,1))
        )


class DUnet(nn.Module):
    def __init__(self,in_ch:int = 3, dim:int = 64):
        super().__init__()
        self.in_to_dim = nn.Conv2d(in_ch,dim,3,1,1)
        #encode x
        self.e_x1 = Down(dim)
        self.e_x2 = Down(dim * 2)
        self.e_x3 = Down(dim * 4)
        #up
        self.up1 = Up(dim * 8)
        self.up2 = Up(dim * 4)
        self.up3 = Up(dim * 2)
        #end conv
        self.end_conv = nn.Sequential(
            spectral_norm(nn.Conv2d(dim, dim, 3, 1, 1, bias=False)),
            nn.Mish(True),
            spectral_norm(nn.Conv2d(dim, dim, 3, 1, 1, bias=False)),
            nn.Mish(True),
            nn.Conv2d(dim, 1, 3, 1, 1)
        )
    def forward(self,x: Tensor) -> Tensor:
        x0 = self.in_to_dim(x)
        x1 = self.e_x1(x0)
        x2 = self.e_x2(x1)
        x3 = self.e_x3(x2)
        x = self.up1(x3)+x2
        x = self.up2(x)+x1
        x = self.up3(x)+x0
        return self.end_conv(x)
