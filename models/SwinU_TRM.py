import torch
import torch.nn as nn
from models.swin_sys import SwinTransformerSys, PatchExpand, FinalPatchExpand_X4

class SwinEncoder(nn.Module):
    """Swin Transformer encoder that outputs bottleneck and skip connection features."""
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, depths=(2,2,6,2),
        num_heads=(3,6,12,24), window_size=7, out_channels=1):
        super().__init__()
        self.backbone = SwinTransformerSys(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            num_classes=out_channels, embed_dim=embed_dim, depths=list(depths), num_heads=list(num_heads),
            window_size=window_size, mlp_ratio=4.0, qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0,
            drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=False, patch_norm=True, use_checkpoint=False
        )
        self.img_size = img_size
        self.patch_size = patch_size

    def forward(self, x):
        """
        Returns:
          bottleneck_X: (B, L_bot, C_bot)
          skip_X_list: list of length num_layers, each (B, L_i, C_i) BEFORE its BasicLayer executes.
        """
        bottleneck_X, skip_X_list = self.backbone.forward_features(x)
        return bottleneck_X, skip_X_list
    
# ---- Norm & MLP ----
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        var = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (x * self.scale).to(x.dtype)

class SwiGLU(nn.Module):
    def __init__(self, dim, expansion=2.0):
        super().__init__()
        inner = int(dim * expansion)
        self.proj = nn.Linear(dim, inner * 2, bias=False)
        self.out = nn.Linear(inner, dim, bias=False)
    def forward(self, x):
        a,b = self.proj(x).chunk(2, dim=-1)
        return self.out(torch.silu(a) * b)
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
    def forward(self, X):
        B,L,C = X.shape
        qkv = self.qkv(X).view(B,L,3,self.num_heads,self.head_dim).permute(2,0,3,1,4)
        q,k,v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1,2).reshape(B,L,C)
        return self.proj(out)

class TRMBlock(nn.Module):
    """Original TRM style block: Attention residual + RMSNorm then SwiGLU residual + RMSNorm.
    No memory gating; matches TinyRecursiveReasoningModel_ACTV1Block structure (sans rotary, puzzle embeddings).
    """
    def __init__(self, dim, num_heads=8, mlp_expansion=2.0, drop_path=0.0):
        super().__init__()
        self.attn = Attention(dim, num_heads)
        self.norm1 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, mlp_expansion)
        self.norm2 = RMSNorm(dim)
        self.drop_path = nn.Identity() if drop_path <= 0 else nn.Dropout(drop_path)
    def forward(self, X):
        X = self.norm1(X + self.drop_path(self.attn(X)))
        X = self.norm2(X + self.drop_path(self.mlp(X)))
        return X


class TRMReasoningModule(nn.Module):
    """Reasoning module: adds injection once then applies a stack of TRMBlocks.
    Mirrors TinyRecursiveReasoningModel_ACTV1ReasoningModule semantics.
    """
    def __init__(self, dim: int, depth: int, num_heads: int, mlp_expansion: float = 2.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            TRMBlock(dim=dim, num_heads=num_heads, mlp_expansion=mlp_expansion)
            for _ in range(depth)
        ])
    def forward(self, x: torch.Tensor, injection: torch.Tensor):
        x = x + injection
        for blk in self.blocks:
            x = blk(x)
        return x

class TRMDecoder(nn.Module):
    """U-Net style decoder that preserves (B, L, C) token format, reusing Swin's PatchExpand
    and skip concatenation strategy, but replacing Swin blocks with TRMBlocks.

    Args:
        embed_dim: base embedding dim from Swin encoder.
        depths: encoder stage depths (to infer number of stages & channel progression).
        trm_depths: list of TRM block counts per decoder stage (len == num_layers).
        num_heads: list of attention heads per corresponding encoder stage (used reversed in decoder except first stage which uses deepest heads).
        mlp_expansion: expansion ratio for SwiGLU inside TRM blocks.
        use_memory: enable gating between recurrent memory and current features.
        norm_layer: normalization layer applied after decoder stages (LayerNorm on tokens).
        out_channels: segmentation output channels.
        img_size, patch_size: for final spatial reshape after FinalPatchExpand_X4.
    """
    def __init__(self,
                 embed_dim: int,
                 depths: tuple,
                 trm_depths: tuple,
                 num_heads: tuple,
                 mlp_expansion: float = 2.0,
                 norm_layer=nn.LayerNorm,
                 out_channels: int = 1,
                 img_size: int = 224,
                 patch_size: int = 4):
        super().__init__()
        self.num_layers = len(depths)
        assert len(trm_depths) == self.num_layers, "trm_depths length must equal encoder num_layers"
        self.img_size = img_size
        self.patch_size = patch_size

        # Channel progression deepest -> shallow (decoder processing order)
        self.stage_dims = [embed_dim * (2 ** i) for i in range(self.num_layers)]  # [e,2e,4e,8e]
        self.stage_dims_rev = list(reversed(self.stage_dims))  # [8e,4e,2e,e]

        # First decoder module: PatchExpand on bottleneck tokens (no TRM before upsample)
        self.first_up = PatchExpand(
            input_resolution=(img_size // patch_size // (2 ** (self.num_layers - 1)),
                               img_size // patch_size // (2 ** (self.num_layers - 1))),
            dim=self.stage_dims_rev[0],
            dim_scale=2,
            norm_layer=norm_layer
        )

        # For subsequent stages create: concat projection + TRM blocks + optional PatchExpand (except final)
        self.skip_proj = nn.ModuleList()      # learnable projection for skip tokens
        self.concat_proj = nn.ModuleList()    # fuse and compress
        self.trm_stages = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for idx in range(1, self.num_layers):
            enc_dim = self.stage_dims_rev[idx]
            skip_original_dim = self.stage_dims[idx-1]
            self.skip_proj.append(nn.Linear(skip_original_dim, enc_dim, bias=False))
            self.concat_proj.append(nn.Linear(enc_dim + enc_dim, enc_dim, bias=False))
            blocks = nn.ModuleList([
                TRMBlock(dim=enc_dim,
                         num_heads=num_heads[self.num_layers - 1 - idx],
                         mlp_expansion=mlp_expansion)
                for _ in range(trm_depths[idx])
            ])
            self.trm_stages.append(blocks)
            up_required = (idx < self.num_layers - 1)
            self.upsamples.append(
                PatchExpand(
                    input_resolution=(img_size // patch_size // (2 ** (self.num_layers - 1 - idx)),
                                       img_size // patch_size // (2 ** (self.num_layers - 1 - idx))),
                    dim=enc_dim,
                    dim_scale=2,
                    norm_layer=norm_layer
                ) if up_required else None
            )

        # Normalization after decoder token sequence assembly
        self.norm = norm_layer(self.stage_dims[0])  # final shallow dim = embed_dim
        # Final upsample to pixel space
        self.final_up = FinalPatchExpand_X4(
            input_resolution=(img_size // patch_size, img_size // patch_size),
            dim=self.stage_dims[0],
            dim_scale=4,
            norm_layer=norm_layer
        )
        self.seg_head = nn.Conv2d(self.stage_dims[0], out_channels, kernel_size=1)

    def forward(self, bottleneck_X: torch.Tensor, skip_X_list: list):
        # skip_X_list: list of pre-layer encoder tokens [stage0_input, stage1_input, ...]
        # Start from bottleneck tokens x
        x = bottleneck_X  # (B, L_bot, C_bot)

        # First upsample (PatchExpand)
        x = self.first_up(x)  # (B, L_up, C_up)

        # Iterate remaining decoder stages
        for stage_idx, (blocks, upsample, proj, skip_linear) in enumerate(zip(self.trm_stages, self.upsamples, self.concat_proj, self.skip_proj)):
            # Determine skip index (mirror of SwinUnet logic): skip token taken from encoder input at stage = num_layers - 1 - (stage_idx + 1)
            enc_skip_index = self.num_layers - 1 - (stage_idx + 1)
            skip_tokens = skip_X_list[enc_skip_index]  # (B, L_skip, C_skip)
            skip_tokens = skip_linear(skip_tokens)

            # Concatenate along channel dim (token last dim)
            x = torch.cat([x, skip_tokens], dim=-1)
            x = proj(x)  # reduce back to current_dim

            # Apply TRM blocks
            for blk in blocks:
                x = blk(x)

            # Optional upsample to next resolution
            if upsample is not None:
                x = upsample(x)

        # Final norm
        x = self.norm(x)

        # FinalPatchExpand_X4 -> spatial map
        x = self.final_up(x)  # (B, H*W, C_final)
        B, HW, C = x.shape
        H = W = int((HW) ** 0.5)
        feat = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        logits = self.seg_head(feat)
        return torch.sigmoid(logits), logits

    def forward_full(self, bottleneck_X: torch.Tensor, skip_X_list: list):
        x = bottleneck_X
        x = self.first_up(x)
        for stage_idx, (blocks, upsample, proj, skip_linear) in enumerate(zip(self.trm_stages, self.upsamples, self.concat_proj, self.skip_proj)):
            enc_skip_index = self.num_layers - 1 - (stage_idx + 1)
            skip_tokens = skip_linear(skip_X_list[enc_skip_index])
            x = torch.cat([x, skip_tokens], dim=-1)
            x = proj(x)
            for blk in blocks:
                x = blk(x)
            if upsample is not None:
                x = upsample(x)
        x = self.norm(x)
        shallow_tokens = x
        x = self.final_up(x)
        B, HW, C = x.shape
        H = W = int(HW ** 0.5)
        feat = x.view(B, H, W, C).permute(0,3,1,2).contiguous()
        logits = self.seg_head(feat)
        return torch.sigmoid(logits), logits, feat, shallow_tokens


class SwinU_TRM(nn.Module):
    """Hybrid model: original SwinUnet encoder & bottleneck + TRM token-based decoder.

    Keeps original upsampling philosophy (PatchExpand + skip concat) but swaps decoder blocks
    for TRMBlocks with optional memory gating.
    """
    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_chans=3,
                 out_channels=1,
                 embed_dim=96,
                 depths=(2,2,6,2),
                 num_heads=(3,6,12,24),
                 window_size=7,
                 trm_depths=(0,1,1,2),
                 trm_mlp_expansion=2.0,
                 enable_recursion=False,
                 H_cycles=3,
                 L_cycles=6,
                 reasoning_depth=2,
                 early_stop_threshold=0.0):
        super().__init__()
        self.enable_recursion = enable_recursion
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        self.reasoning_depth = reasoning_depth
        self.early_stop_threshold = early_stop_threshold
        deepest_dim = embed_dim * (2 ** (len(depths) - 1))
        self.encoder = SwinEncoder(img_size=img_size,
                                    patch_size=patch_size,
                                    in_chans=in_chans,
                                    embed_dim=embed_dim,
                                    depths=depths,
                                    num_heads=num_heads,
                                    window_size=window_size,
                                    out_channels=out_channels)
        self.decoder = TRMDecoder(embed_dim=embed_dim,
                                   depths=depths,
                                   trm_depths=trm_depths,
                                   num_heads=num_heads,
                                   mlp_expansion=trm_mlp_expansion,
                                   out_channels=out_channels,
                                   img_size=img_size,
                                   patch_size=patch_size)
        # Reasoning module (shared for z_H and z_L updates)
        deepest_heads = num_heads[-1]
        self.reasoning = TRMReasoningModule(dim=deepest_dim, depth=reasoning_depth, num_heads=deepest_heads,
                                            mlp_expansion=trm_mlp_expansion)
        # Initial states H_init / L_init vectors
        self.H_init = nn.Parameter(torch.randn(deepest_dim) * 0.02)
        self.L_init = nn.Parameter(torch.randn(deepest_dim) * 0.02)
        # Q-head (halt / continue logits) using first token
        self.q_head = nn.Linear(deepest_dim, 2)

    def forward(self, x):
        if not self.enable_recursion:
            bottleneck_X, skip_X_list = self.encoder(x)
            seg_prob, _ = self.decoder(bottleneck_X, skip_X_list)
            return seg_prob
        seg_probs, _, _ = self.forward_recursive(x, return_all=False)
        return seg_probs[-1]

    def forward_with_logits(self, x):
        if not self.enable_recursion:
            bottleneck_X, skip_X_list = self.encoder(x)
            seg_prob, seg_logits = self.decoder(bottleneck_X, skip_X_list)
            return seg_prob, seg_logits
        seg_probs, seg_logits_list, _ = self.forward_recursive(x, return_all=True)
        return seg_probs[-1], seg_logits_list[-1]

    def forward_recursive(self, x, return_all=True):
        bottleneck_X, skip_X_list = self.encoder(x)  # (B, L_bot, C_bot)
        z_H = bottleneck_X + self.H_init.view(1,1,-1)
        z_L = bottleneck_X + self.L_init.view(1,1,-1)
        seg_probs = []
        seg_logits_list = []
        q_hats = []
        # H_cycles - 1 no_grad
        with torch.no_grad():
            for _ in range(self.H_cycles - 1):
                for _ in range(self.L_cycles):
                    z_L = self.reasoning(z_L, injection=z_H + bottleneck_X)
                z_H = self.reasoning(z_H, injection=z_L)
                # optional early stop
                q_logits = self.q_head(z_H[:,0])  # (B,2)
                halt_prob = torch.sigmoid(q_logits[:,0])
                if self.early_stop_threshold > 0 and halt_prob.mean().item() > self.early_stop_threshold:
                    break
        # Final cycle with grad
        for _ in range(self.L_cycles):
            z_L = self.reasoning(z_L, injection=z_H + bottleneck_X)
        z_H = self.reasoning(z_H, injection=z_L)
        # Decode once
        seg_prob, seg_logits, feat, shallow_tokens = self.decoder.forward_full(z_H, skip_X_list)
        seg_probs.append(seg_prob)
        seg_logits_list.append(seg_logits)
        q_logits = self.q_head(z_H[:,0])
        q_hats.append(torch.sigmoid(q_logits[:,0:1]))  # return halt prob
        return (seg_probs, seg_logits_list, q_hats) 