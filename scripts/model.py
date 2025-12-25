from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class UpBlock(nn.Module):
    """Upsampling block."""
    def __init__(self, in_channels: int, out_channels: int, use_attention: bool = False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.res = ResBlock(out_channels, out_channels, stride=1)
        self.attention = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn(self.up(x)))
        x = self.res(x)
        return x


class ImprovedEncoder(nn.Module):
    """Encoder network."""
    def __init__(self, in_channels: int = 3, latent_dim: int = 512, feature_size: int = 16):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.feature_size = feature_size
        self.flatten_dim = 512 * feature_size * feature_size
        
        self.fc_mu = nn.Sequential(
            nn.Linear(self.flatten_dim, latent_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(latent_dim * 2, latent_dim)
        )
        self.fc_logvar = nn.Sequential(
            nn.Linear(self.flatten_dim, latent_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(latent_dim * 2, latent_dim)
        )

    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int):
        layers = []
        layers.append(ResBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.initial(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        h = F.adaptive_avg_pool2d(h, (self.feature_size, self.feature_size))
        h = torch.flatten(h, start_dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = ImprovedVAE.reparameterize(mu, logvar)
        return z, mu, logvar


class ImprovedDecoder(nn.Module):
    """Decoder network."""
    def __init__(self, out_channels: int = 3, latent_dim: int = 512, feature_size: int = 16):
        super().__init__()
        self.feature_size = feature_size
        
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim * 2, 512 * feature_size * feature_size)
        )
        
        self.up1 = UpBlock(512, 512, use_attention=False)
        self.up2 = UpBlock(512, 256, use_attention=False)
        self.up3 = UpBlock(256, 128, use_attention=False)
        self.up4 = UpBlock(128, 64, use_attention=False)
        self.up5 = UpBlock(64, 32, use_attention=False)
        
        self.refine = nn.Sequential(
            ResBlock(32, 32),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        
        self.out = nn.Conv2d(16, out_channels, 3, stride=1, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(-1, 512, self.feature_size, self.feature_size)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        h = self.up4(h)
        h = self.up5(h)
        h = self.refine(h)
        return torch.sigmoid(self.out(h))


class ImprovedVAE(nn.Module):
    def __init__(self, in_channels: int = 3, latent_dim: int = 512, feature_size: int = 16):
        super().__init__()
        self.encoder = ImprovedEncoder(in_channels=in_channels, latent_dim=latent_dim, feature_size=feature_size)
        self.decoder = ImprovedDecoder(out_channels=in_channels, latent_dim=latent_dim, feature_size=feature_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mu, logvar = self.encoder(x)
        recon = self.decoder(z)
        return recon, mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def loss_function(
        recon_x: torch.Tensor, 
        x: torch.Tensor, 
        mu: torch.Tensor, 
        logvar: torch.Tensor, 
        beta: float = 0.1,
        perceptual_weight: float = 0.0,
        vgg_features = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        perceptual_loss = torch.tensor(0.0, device=x.device)
        if perceptual_weight > 0 and vgg_features is not None:
            with torch.no_grad():
                target_features = vgg_features(x)
            recon_features = vgg_features(recon_x)
            perceptual_loss = F.mse_loss(recon_features, target_features)
        
        loss = recon_loss + beta * kld + perceptual_weight * perceptual_loss
        return loss, recon_loss, kld, perceptual_loss


class VGGPerceptualLoss(nn.Module):
    """VGG perceptual loss."""
    def __init__(self):
        super().__init__()
        try:
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        except:
            vgg = models.vgg16(pretrained=True).features
        
        self.features = nn.Sequential(*list(vgg.children())[:16])
        
        for param in self.features.parameters():
            param.requires_grad = False
        
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        return self.features(x)


# Backwards compatibility
ResNetEncoder = ImprovedEncoder
ResNetDecoder = ImprovedDecoder
ResNetVAE = ImprovedVAE
