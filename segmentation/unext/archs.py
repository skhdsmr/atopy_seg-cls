"""
UNeXt (MLP-based Rapid Medical Image Segmentation, MICCAI 2022) 모델 정의.

원 논문/구현(jeya-maria-jose/UNeXt-pytorch)의 UNext 구조를 그대로 재현하되,
timm 의존성(DropPath / trunc_normal_ / to_2tuple)을 제거하기 위해 필요한 유틸을
inline 으로 구현했다(이 환경엔 timm 미설치).

구조 요약:
  - Encoder: Conv 3단(16→32→128, 각 stage 뒤 maxpool /2) + Tokenized-MLP 2단
  - Bottleneck 을 지나 Decoder 에서 대칭 skip-connection 으로 복원
  - 입력 H,W 는 32의 배수여야 함(총 다운샘플 /32).
  - num_classes=1 + BCEDiceLoss(sigmoid) 이 기본(단일 클래스 병변 분할).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["UNext"]


# --- timm 대체 유틸 (self-contained) ----------------------------------------
def to_2tuple(x):
    if isinstance(x, (tuple, list)):
        return tuple(x)
    return (x, x)


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)


class DropPath(nn.Module):
    """Stochastic depth. drop_prob=0 이면 no-op."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        rand.floor_()
        return x.div(keep_prob) * rand


# --- 공통 가중치 초기화 ------------------------------------------------------
def _init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()


class DWConv(nn.Module):
    """Depthwise conv (MLP 내부 위치 인코딩 역할)."""

    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class shiftmlp(nn.Module):
    """Shifted-MLP: 축(H, W)별로 채널을 shift 해 지역성을 주입한 토큰 MLP."""

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0, shift_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.shift_size = shift_size
        self.pad = shift_size // 2
        self.apply(_init_weights)

    def forward(self, x, H, W):
        B, N, C = x.shape

        # --- axis-1(H) shift ---
        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, s, 2)
                   for x_c, s in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)
        x_s = x_s.reshape(B, C, H * W).contiguous()
        x_shift_r = x_s.transpose(1, 2)

        x = self.fc1(x_shift_r)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)

        # --- axis-2(W) shift ---
        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, s, 3)
                   for x_c, s in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)
        x_s = x_s.reshape(B, C, H * W).contiguous()
        x_shift_c = x_s.transpose(1, 2)

        x = self.fc2(x_shift_c)
        x = self.drop(x)
        return x


class shiftedBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0, drop_path=0.0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = shiftmlp(in_features=dim, hidden_features=mlp_hidden_dim,
                            act_layer=act_layer, drop=drop)
        self.apply(_init_weights)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(_init_weights)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class UNext(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, img_size=512,
                 embed_dims=(128, 160, 256), drop_rate=0.0, drop_path_rate=0.0,
                 norm_layer=nn.LayerNorm, depths=(1, 1, 1), **kwargs):
        super().__init__()

        # --- Conv encoder ---
        self.encoder1 = nn.Conv2d(input_channels, 16, 3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(32, 128, 3, stride=1, padding=1)
        self.ebn1 = nn.BatchNorm2d(16)
        self.ebn2 = nn.BatchNorm2d(32)
        self.ebn3 = nn.BatchNorm2d(128)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])
        self.dnorm3 = norm_layer(160)
        self.dnorm4 = norm_layer(128)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # --- Tokenized-MLP encoder ---
        self.block1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], mlp_ratio=1, drop=drop_rate, drop_path=dpr[0],
            norm_layer=norm_layer)])
        self.block2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[2], mlp_ratio=1, drop=drop_rate, drop_path=dpr[1],
            norm_layer=norm_layer)])
        # --- Tokenized-MLP decoder ---
        self.dblock1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], mlp_ratio=1, drop=drop_rate, drop_path=dpr[0],
            norm_layer=norm_layer)])
        self.dblock2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[0], mlp_ratio=1, drop=drop_rate, drop_path=dpr[1],
            norm_layer=norm_layer)])

        self.patch_embed3 = OverlapPatchEmbed(
            img_size=img_size // 4, patch_size=3, stride=2,
            in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=img_size // 8, patch_size=3, stride=2,
            in_chans=embed_dims[1], embed_dim=embed_dims[2])

        # --- Conv decoder ---
        self.decoder1 = nn.Conv2d(256, 160, 3, stride=1, padding=1)
        self.decoder2 = nn.Conv2d(160, 128, 3, stride=1, padding=1)
        self.decoder3 = nn.Conv2d(128, 32, 3, stride=1, padding=1)
        self.decoder4 = nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.dbn1 = nn.BatchNorm2d(160)
        self.dbn2 = nn.BatchNorm2d(128)
        self.dbn3 = nn.BatchNorm2d(32)
        self.dbn4 = nn.BatchNorm2d(16)

        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        B = x.shape[0]

        # --- Conv stage ---
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out

        # --- Tokenized MLP stage ---
        out, H, W = self.patch_embed3(out)
        for blk in self.block1:
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        out, H, W = self.patch_embed4(out)
        for blk in self.block2:
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # --- Decoder ---
        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)),
                                   scale_factor=2, mode="bilinear", align_corners=False))
        out = torch.add(out, t4)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock1:
            out = blk(out, H, W)

        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)),
                                   scale_factor=2, mode="bilinear", align_corners=False))
        out = torch.add(out, t3)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock2:
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)),
                                   scale_factor=2, mode="bilinear", align_corners=False))
        out = torch.add(out, t2)
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)),
                                   scale_factor=2, mode="bilinear", align_corners=False))
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out),
                                   scale_factor=2, mode="bilinear", align_corners=False))

        return self.final(out)
