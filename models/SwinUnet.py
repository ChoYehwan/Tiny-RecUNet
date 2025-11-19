import torch
import torch.nn as nn
from .swin_sys import SwinTransformerSys


class SwinUnet(nn.Module):
    """Wrapper around SwinTransformerSys wired to current config.model_args['SwinUnet'].

    Expected kwargs (from config.py):
    - patch_size, in_chans, embed_dim, depths, decoder_depths, num_heads,
      window_size, mlp_ratio, qkv_bias, qk_scale, ape, patch_norm,
      final_upsample, out_channels
    Optional kwargs (fallback to sensible defaults or global config.image_size):
    - img_size, drop_rate, attn_drop_rate, drop_path_rate, use_checkpoint
    """

    def __init__(self, patch_size: int = 4, in_channels: int = 3, embed_dim: int = 96, depths=(2, 2, 6, 2), 
                 decoder_depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), window_size: int = 7, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, qk_scale=None, ape: bool = False, patch_norm: bool = True,
                 final_upsample: str = "expand_first", out_channels: int = 1,
        # optional/advanced
        img_size: int = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        use_checkpoint: bool = False
                ):
        super().__init__()

        # Resolve image size: prefer explicit img_size; else fall back to global config.image_size; else 224.
        if img_size is None:
            try:
                from config import image_size as _global_img_size  # avoid hard dependency if unused
                img_size = int(_global_img_size)
            except Exception:
                img_size = 224

        # Normalize possible list inputs from config to tuples (Swin code works with lists/tuples interchangeably)
        if isinstance(depths, list):
            depths = tuple(depths)
        if isinstance(decoder_depths, list):
            decoder_depths = tuple(decoder_depths)
        if isinstance(num_heads, list):
            num_heads = tuple(num_heads)

        # Build the underlying Swin U-Net system
        self.backbone = SwinTransformerSys(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            num_classes=out_channels,
            embed_dim=embed_dim,
            depths=depths,
            depths_decoder=decoder_depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            ape=ape,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            final_upsample=final_upsample,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If single-channel input but backbone expects 3, replicate to 3 channels
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.backbone(x)
        return torch.sigmoid(logits)