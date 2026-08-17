"""Core Teacher and Student networks for PTID."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _validate_channels(channels: Sequence[int]) -> tuple[int, int, int, int, int]:
    values = tuple(int(value) for value in channels)
    if len(values) != 5 or any(value <= 0 for value in values):
        raise ValueError("channels must contain five positive integers")
    return values


def _validate_spatial_shape(x: Tensor) -> None:
    height, width = x.shape[-2:]
    if height % 16 or width % 16:
        raise ValueError("input height and width must be divisible by 16")


class ResidualBlock(nn.Module):
    """Conv-BN-ReLU-Conv-BN with a residual shortcut."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.shortcut(x)
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.activation(x + residual)


class ChannelAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        reduction: int,
        minimum_hidden_channels: int,
    ) -> None:
        super().__init__()
        if reduction <= 0 or minimum_hidden_channels <= 0:
            raise ValueError("CBAM channel settings must be positive")
        hidden = max(channels // reduction, minimum_hidden_channels)
        self.shared_projection = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        average = self.shared_projection(F.adaptive_avg_pool2d(x, 1))
        maximum = self.shared_projection(F.adaptive_max_pool2d(x, 1))
        return x * torch.sigmoid(average + maximum)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("spatial attention kernel_size must be a positive odd integer")
        self.projection = nn.Conv2d(
            2,
            1,
            kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        average = torch.mean(x, dim=1, keepdim=True)
        maximum = torch.amax(x, dim=1, keepdim=True)
        weights = torch.sigmoid(self.projection(torch.cat((average, maximum), dim=1)))
        return x * weights


class CBAM(nn.Module):
    def __init__(
        self,
        channels: int,
        reduction: int,
        minimum_hidden_channels: int,
        spatial_kernel_size: int,
    ) -> None:
        super().__init__()
        self.channel = ChannelAttention(channels, reduction, minimum_hidden_channels)
        self.spatial = SpatialAttention(spatial_kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.spatial(self.channel(x))


class AuxiliaryAttentionBranch(nn.Module):
    """Training-only Conv-BN-ReLU-CBAM transformation."""

    def __init__(
        self,
        channels: int,
        cbam_reduction: int,
        cbam_minimum_hidden_channels: int,
        cbam_spatial_kernel_size: int,
    ) -> None:
        super().__init__()
        self.transform = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            CBAM(
                channels,
                cbam_reduction,
                cbam_minimum_hidden_channels,
                cbam_spatial_kernel_size,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.transform(x)


class EncoderTail(nn.Module):
    """Encoder stages E2-E4 and the bottleneck."""

    def __init__(self, channels: Sequence[int]) -> None:
        super().__init__()
        c1, c2, c3, c4, c5 = _validate_channels(channels)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = ResidualBlock(c1, c2)
        self.enc3 = ResidualBlock(c2, c3)
        self.enc4 = ResidualBlock(c3, c4)
        self.bottleneck = ResidualBlock(c4, c5)

    def forward(self, enc1: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        bottleneck = self.bottleneck(self.pool(enc4))
        return enc2, enc3, enc4, bottleneck


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.fusion = ResidualBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.upsample(x)
        return self.fusion(torch.cat((x, skip), dim=1))


class UNetDecoder(nn.Module):
    def __init__(self, channels: Sequence[int]) -> None:
        super().__init__()
        c1, c2, c3, c4, c5 = _validate_channels(channels)
        self.dec4 = DecoderBlock(c5, c4, c4)
        self.dec3 = DecoderBlock(c4, c3, c3)
        self.dec2 = DecoderBlock(c3, c2, c2)
        self.dec1 = DecoderBlock(c2, c1, c1)
        self.bm_head = nn.Conv2d(c1, 1, 1)
        self.ubm_head = nn.Conv2d(c1, 1, 1)

    def forward(
        self,
        bottleneck: Tensor,
        enc4: Tensor,
        enc3: Tensor,
        enc2: Tensor,
        enc1: Tensor,
    ) -> tuple[Tensor, Tensor]:
        x = self.dec4(bottleneck, enc4)
        x = self.dec3(x, enc3)
        x = self.dec2(x, enc2)
        x = self.dec1(x, enc1)
        return self.bm_head(x), self.ubm_head(x)


class TemporalTeacher(nn.Module):
    """Multi-frame Teacher with a weight-shared E1 and temporal fusion."""

    def __init__(self, n_frames: int, channels: Sequence[int]) -> None:
        super().__init__()
        if n_frames <= 0:
            raise ValueError("n_frames must be positive")
        self.n_frames = int(n_frames)
        self.channels = _validate_channels(channels)
        c1 = self.channels[0]
        self.shared_enc1 = ResidualBlock(1, c1)
        self.temporal_fusion = nn.Sequential(
            nn.Conv2d(self.n_frames * c1, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.encoder_tail = EncoderTail(self.channels)
        self.decoder = UNetDecoder(self.channels)

    def forward(
        self,
        ssh_sequence: Tensor,
        return_features: bool = False,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, dict[str, Tensor]]:
        if ssh_sequence.ndim != 4:
            raise ValueError("Teacher input must have shape [B, T, H, W]")
        batch, frames, height, width = ssh_sequence.shape
        if frames != self.n_frames:
            raise ValueError(f"expected {self.n_frames} frames, received {frames}")
        _validate_spatial_shape(ssh_sequence)

        per_frame = self.shared_enc1(ssh_sequence.reshape(batch * frames, 1, height, width))
        per_frame = per_frame.reshape(batch, frames, self.channels[0], height, width)
        enc1 = self.temporal_fusion(per_frame.flatten(1, 2))
        enc2, enc3, enc4, bottleneck = self.encoder_tail(enc1)
        bm, ubm = self.decoder(bottleneck, enc4, enc3, enc2, enc1)
        if return_features:
            return bm, ubm, {"enc2": enc2, "enc3": enc3, "enc4": enc4}
        return bm, ubm


class StudentBackbone(nn.Module):
    """Single-frame U-Net retained at inference."""

    def __init__(self, channels: Sequence[int]) -> None:
        super().__init__()
        self.channels = _validate_channels(channels)
        self.enc1 = ResidualBlock(1, self.channels[0])
        self.encoder_tail = EncoderTail(self.channels)
        self.decoder = UNetDecoder(self.channels)

    def forward(
        self,
        center_ssh: Tensor,
        return_features: bool = False,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, dict[str, Tensor]]:
        if center_ssh.ndim != 4 or center_ssh.shape[1] != 1:
            raise ValueError("Student input must have shape [B, 1, H, W]")
        _validate_spatial_shape(center_ssh)
        enc1 = self.enc1(center_ssh)
        enc2, enc3, enc4, bottleneck = self.encoder_tail(enc1)
        bm, ubm = self.decoder(bottleneck, enc4, enc3, enc2, enc1)
        if return_features:
            return bm, ubm, {"enc2": enc2, "enc3": enc3, "enc4": enc4}
        return bm, ubm


class PTIDStudent(nn.Module):
    """Single-frame backbone with removable auxiliary attention branches."""

    available_distillation_layers = ("enc2", "enc3", "enc4")

    def __init__(
        self,
        channels: Sequence[int],
        distilled_layers: Sequence[str],
        cbam_reduction: int,
        cbam_minimum_hidden_channels: int,
        cbam_spatial_kernel_size: int,
    ) -> None:
        super().__init__()
        self.backbone = StudentBackbone(channels)
        requested = tuple(distilled_layers)
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("distilled_layers must be non-empty and unique")
        if any(name not in self.available_distillation_layers for name in requested):
            raise ValueError(f"distilled_layers must be selected from {self.available_distillation_layers}")
        self.distilled_layers = requested
        channel_by_layer = dict(zip(self.available_distillation_layers, self.backbone.channels[1:4]))
        self.auxiliary = nn.ModuleDict(
            {
                name: AuxiliaryAttentionBranch(
                    channel_by_layer[name],
                    cbam_reduction,
                    cbam_minimum_hidden_channels,
                    cbam_spatial_kernel_size,
                )
                for name in self.distilled_layers
            }
        )

    def forward(self, center_ssh: Tensor) -> tuple[Tensor, Tensor]:
        """Inference path; auxiliary branches are not executed."""

        return self.backbone(center_ssh)

    def forward_with_auxiliary(
        self,
        center_ssh: Tensor,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor], dict[str, Tensor]]:
        """Distillation path returning raw and auxiliary encoder features."""

        bm, ubm, features = self.backbone(center_ssh, return_features=True)
        auxiliary = {name: self.auxiliary[name](features[name]) for name in self.distilled_layers}
        return bm, ubm, features, auxiliary

