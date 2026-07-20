import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# =============================================================================
# 1.  U-NET BACKGROUND REMOVAL
# =============================================================================

class UNetConvBlock(nn.Module):
    """
    One double-conv block used in both encoder and decoder of U-Net.
    Pattern: Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    BatchNorm is fine here (unlike FeatEnHancer) because we are producing
    a per-pixel mask, not preserving subtle pixel-level brightness gradients.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetBackgroundRemover(nn.Module):
    """
    Lightweight U-Net that predicts a soft foreground mask (0=background,
    1=foreground) from an RGB image.

    Encoder path  (contracting):  3 -> 32 -> 64 -> 128 -> 256
    Bottleneck:                  256 -> 512
    Decoder path  (expansive):   512 -> 256 -> 128 -> 64 -> 32
    Output:                       32 ->  1  sigmoid -> mask in [0,1]

    The mask composites the cow onto a neutral grey background:
        output = image * mask + neutral_bg * (1 - mask)

    We use smaller channels (32/64/128/256/512) vs the original U-Net
    (64/128/256/512/1024) to keep CPU memory manageable.

    No separate segmentation labels are needed — trained jointly with the
    classification loss so the mask removes whatever confuses the classifier.
    """

    def __init__(self, base_ch: int = 32):
        super().__init__()

        # -- Encoder (contracting path) ---------------------------------------
        self.enc1 = UNetConvBlock(3,         base_ch)       # 224->224, 3->32
        self.enc2 = UNetConvBlock(base_ch,   base_ch * 2)   # 112->112, 32->64
        self.enc3 = UNetConvBlock(base_ch*2, base_ch * 4)   # 56->56,  64->128
        self.enc4 = UNetConvBlock(base_ch*4, base_ch * 8)   # 28->28, 128->256
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)   # halves spatial dims

        # -- Bottleneck -------------------------------------------------------
        self.bottleneck = UNetConvBlock(base_ch*8, base_ch*16)  # 14->14, 256->512

        # -- Decoder (expansive path) -----------------------------------------
        # Each step: transposed conv (doubles spatial) + concat skip + double conv
        self.up4  = nn.ConvTranspose2d(base_ch*16, base_ch*8,  kernel_size=2, stride=2)
        self.dec4 = UNetConvBlock(base_ch*16, base_ch*8)    # cat enc4: 512->256

        self.up3  = nn.ConvTranspose2d(base_ch*8,  base_ch*4,  kernel_size=2, stride=2)
        self.dec3 = UNetConvBlock(base_ch*8,  base_ch*4)    # cat enc3: 256->128

        self.up2  = nn.ConvTranspose2d(base_ch*4,  base_ch*2,  kernel_size=2, stride=2)
        self.dec2 = UNetConvBlock(base_ch*4,  base_ch*2)    # cat enc2: 128->64

        self.up1  = nn.ConvTranspose2d(base_ch*2,  base_ch,    kernel_size=2, stride=2)
        self.dec1 = UNetConvBlock(base_ch*2,  base_ch)      # cat enc1: 64->32

        # -- Output head ------------------------------------------------------
        self.out_conv = nn.Conv2d(base_ch, 1, kernel_size=1)  # 32->1 channel

        # Neutral background: grey (0.0 in Normalize([0.5],[0.5]) space).
        # register_buffer moves with .to(device) but is NOT a learned parameter.
        self.register_buffer('neutral_bg', torch.zeros(1, 3, 1, 1))

    def forward(self, x):
        """
        x      : (B, 3, H, W)  normalised input image
        returns: masked_img (B, 3, H, W)  cow on neutral grey
                 mask       (B, 1, H, W)  soft foreground probability
        """
        # Encoder
        e1 = self.enc1(x)               # (B, 32,  224, 224)
        e2 = self.enc2(self.pool(e1))   # (B, 64,  112, 112)
        e3 = self.enc3(self.pool(e2))   # (B, 128,  56,  56)
        e4 = self.enc4(self.pool(e3))   # (B, 256,  28,  28)

        # Bottleneck
        bn = self.bottleneck(self.pool(e4))   # (B, 512, 14, 14)

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(bn), e4], dim=1))  # (B, 256, 28, 28)
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))  # (B, 128, 56, 56)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))  # (B,  64,112,112)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # (B,  32,224,224)

        # Mask: sigmoid gives probability per pixel in [0, 1]
        mask = torch.sigmoid(self.out_conv(d1))   # (B, 1, 224, 224)

        # Composite: foreground pixels kept, background replaced with grey
        bg         = self.neutral_bg.expand(x.size(0), 3, x.size(2), x.size(3))
        masked_img = x * mask + bg * (1.0 - mask)   # (B, 3, H, W)

        return masked_img, mask


# =============================================================================
# 2.  FEATENHANCER COMPONENTS
# =============================================================================

class FeatureEnhancementNetwork(nn.Module):
    """
    Intra-scale feature extraction network (FEN).
    6 conv layers with symmetrical skip concatenation. NO BatchNorm — preserves
    pixel-level spatial relationships needed for texture-based breed features.
    """
    def __init__(self, in_channels=3, mid_channels=32):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.e1 = nn.Sequential(nn.Conv2d(mid_channels,     mid_channels, 3, padding=1), nn.ReLU(inplace=True))
        self.e2 = nn.Sequential(nn.Conv2d(mid_channels,     mid_channels, 3, padding=1), nn.ReLU(inplace=True))
        self.e3 = nn.Sequential(nn.Conv2d(mid_channels,     mid_channels, 3, padding=1), nn.ReLU(inplace=True))
        self.d3 = nn.Sequential(nn.Conv2d(mid_channels * 2, mid_channels, 3, padding=1), nn.ReLU(inplace=True))
        self.d2 = nn.Sequential(nn.Conv2d(mid_channels * 2, mid_channels, 3, padding=1), nn.ReLU(inplace=True))
        self.d1 = nn.Sequential(nn.Conv2d(mid_channels * 2, mid_channels, 3, padding=1), nn.ReLU(inplace=True))

    def forward(self, x):
        x0 = self.entry(x)
        e1 = self.e1(x0)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        d3 = self.d3(torch.cat([e3, e2], dim=1))
        d2 = self.d2(torch.cat([d3, e1], dim=1))
        d1 = self.d1(torch.cat([d2, x0], dim=1))
        return d1


class SAFA(nn.Module):
    """
    Scale-Aware Attentional Feature Aggregation.
    Fuses full-res F and quarter-scale Fq via N-block element-wise attention,
    producing Fh at 1/8 resolution.
    """
    def __init__(self, channels=32, n_blocks=8):
        super().__init__()
        self.n_blocks = n_blocks
        self.block_ch = channels // n_blocks
        self.proj_F = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=7, stride=4, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.proj_Fq = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, Ff, Fq):
        Q = self.proj_F(Ff)
        K = self.proj_Fq(Fq)
        B, _, Hd, Wd = Q.shape
        blocks_q  = Q.view(B, self.n_blocks, self.block_ch, Hd, Wd)
        blocks_k  = K.view(B, self.n_blocks, self.block_ch, Hd, Wd)
        W_norm    = torch.softmax(blocks_q * blocks_k, dim=2)
        Fqk       = torch.cat([Q, K], dim=1)
        Fqk_b     = Fqk[:, :self.n_blocks * self.block_ch].view(B, self.n_blocks, self.block_ch, Hd, Wd)
        return (W_norm * Fqk_b).view(B, -1, Hd, Wd)


class FeatEnHancer(nn.Module):
    """
    Full FeatEnHancer module. Receives the background-removed image and
    produces an enhanced RGB image optimised for classification.
    """
    def __init__(self, mid_channels=32, n_blocks=8):
        super().__init__()
        self.down_q   = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=7, stride=4, padding=3, groups=3, bias=False),
            nn.ReLU(inplace=True)
        )
        self.down_o   = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1, groups=3, bias=False),
            nn.ReLU(inplace=True)
        )
        self.fen_full = FeatureEnhancementNetwork(3, mid_channels)
        self.fen_q    = FeatureEnhancementNetwork(3, mid_channels)
        self.fen_o    = FeatureEnhancementNetwork(3, mid_channels)
        self.safa     = SAFA(mid_channels, n_blocks)
        self.merge    = nn.Conv2d(mid_channels, mid_channels, 3, padding=1)
        self.out_proj = nn.Sequential(
            nn.Conv2d(mid_channels, 3, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):
        H, W  = x.shape[2], x.shape[3]
        Iq    = self.down_q(x)
        Io    = self.down_o(Iq)
        Ff    = self.fen_full(x)
        Fq    = self.fen_q(Iq)
        Fo    = self.fen_o(Io)
        Fh    = self.safa(Ff, Fq)
        Fh_up = F.interpolate(Fh, size=(H, W), mode='bilinear', align_corners=False)
        Fo_up = F.interpolate(Fo, size=(H, W), mode='bilinear', align_corners=False)
        out   = self.out_proj(self.merge(Fh_up + Fo_up))
        return torch.clamp(out + x, -1.0, 1.0)


# =============================================================================
# 3.  COMBINED MODEL: U-Net -> FeatEnHancer -> Swin Transformer
# =============================================================================

class CowBreedClassifier(nn.Module):
    """
    Full end-to-end pipeline:

      Raw image
        |
        v
      UNetBackgroundRemover  -- removes distracting background
        |
        v
      FeatEnHancer           -- enhances low-light hierarchical features
        |
        v
      Swin Transformer Tiny  -- classifies breed
        |
        v
      Breed logits (+ mask for auxiliary loss)
    """

    def __init__(self, num_classes: int, pretrained: bool = True,
                 unet_base_ch: int = 32,
                 feat_mid_channels: int = 32, feat_n_blocks: int = 8):
        super().__init__()
        self.bg_remover = UNetBackgroundRemover(base_ch=unet_base_ch)
        self.enhancer   = FeatEnHancer(mid_channels=feat_mid_channels,
                                       n_blocks=feat_n_blocks)
        self.backbone   = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=pretrained,
            num_classes=num_classes
        )

    def forward(self, x):
        x_masked, mask = self.bg_remover(x)    # Stage 1: remove background
        x_enhanced     = self.enhancer(x_masked)  # Stage 2: enhance features
        logits         = self.backbone(x_enhanced) # Stage 3: classify breed
        return logits, mask
