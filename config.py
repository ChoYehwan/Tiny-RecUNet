import datetime

hyperparameter_tuning = True
# Random search configuration 
tuning_trials = 10

model_name = "SwinU_TRM"
exp_name = model_name + "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Weights & Biases
wandb_project = "AICONV_FINAL"  
wandb_entity = "TRMUNet"   
wandb_tags = [model_name]          

batch_size = 16
epochs = 50
lr = 0.000
workers = 4
weights = "./"
image_size = 224
aug_scale = 0.05
aug_angle = 15

# dataset split controls (single global seed for deterministic splits & randomness)
val_count = 10
test_count = 10
global_seed = 42  # replaces separate val_seed/test_seed

# (Test set always evaluated after training; flags removed for simplicity)

model_args = {
    "UNet": {
        "in_channels": 3,
        "out_channels": 1,
    },
    "TransUnet": {
        "in_channels": 3,
        "out_channels": 1,
        "img_size": 224,
        "backbone": "resnet34",
        "pretrained_backbone": False,
        "embed_dim": 256,
        "depth": 6,
        "num_heads": 8,
        "mlp_ratio": 4.0,
        "patch_grid": None,          # e.g., (14,14) for 224x224; None = full grid (H/16, W/16)
        "embed_dropout": 0.1,        # e.g., 0.1 for embedding dropout like the paper
        "decoder_channels": (256, 128, 64, 16),
        "head_channels": 512,
    },
    "SwinUnet": {
        "in_channels": 3,
        "out_channels": 1,
        "patch_size": 4,
        "embed_dim": 96,
        "depths": [2, 2, 6, 2],              # encoder stage depths
        "decoder_depths": [2, 2, 6, 2],      # coupled to depths; tuning only 'depths'
        "num_heads": [3, 6, 12, 24],
        "window_size": 7,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "qk_scale": None,
        "ape": False,
        "patch_norm": True,
        "final_upsample": "expand_first",
        # explicit defaults for advanced regularization (tunable)
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
        "drop_path_rate": 0.1,
    }
    ,
    "SwinU_TRM": {
        "in_channels": 3,
        "out_channels": 1,
        "patch_size": 4,
        "embed_dim": 96,
        "depths": [2, 2, 6, 2],
        "num_heads": [3, 6, 12, 24],
        "window_size": 7,
        # TRM-specific decoder configuration
        "trm_depths": [0, 1, 1, 2],
        "trm_mlp_expansion": 2.0,
        # Always recursive now; enable_recursion removed
        "H_cycles": 3,
        "L_cycles": 6,
        "reasoning_depth": 2,  # reasoning module depth per z_H/z_L update
        "early_stop_threshold": 0.0,
        # Regularization defaults (added for overfitting control)
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
        "drop_path_rate": 0.1,
    }
}


sweep_space = {
    # Global training hyperparameters can repeat in each model section if desired
    "TransUnet": {
        "lr": {"log_uniform": [1e-5, 1e-3]},
        "batch_size": {"values": [8, 16]},  # discrete
        "embed_dim": {"values": [128, 256]},
        "depth": {"values": [4, 6]},
        "num_heads": {"values": [4, 8]},
        "embed_dropout": {"uniform": [0.0, 0.3]},
        "decoder_channels": {"values": ["256,128,64,16", "512,256,128,32"]},
        "head_channels": {"values": [512, 256]},
        "patch_grid": {"values": ["None", "14,14"]},
    },
    "SwinUnet": {
        "lr": {"log_uniform": [1e-5, 1e-3]},
        "batch_size": {"values": [8, 16]},
        "embed_dim": {"values": [96, 128],},
        "patch_size": {"values": [4, 7]},
        "depths": {"values": ["2,2,6,2", "2,2,4,2"]},  # decoder_depths mirrors this
        # dropout-related tunables
        "drop_rate": {"uniform": [0.0, 0.2]},
        "attn_drop_rate": {"uniform": [0.0, 0.2]},
        "drop_path_rate": {"uniform": [0.0, 0.3]},
    },
    # UNet
    "UNet": {
        "lr": {"log_uniform": [1e-5, 1e-3]},
        "batch_size": {"values": [8, 16]},
    },
    "SwinU_TRM": {
        "lr": {"log_uniform": [1e-5, 1e-3]},
        "batch_size": {"values": [8, 16]},
        # Capacity & structure (narrowed for overfitting mitigation)
        "embed_dim": {"values": [96]},
        "patch_size": {"values": [4, 7]},
        "depths": {"values": ["2,2,6,2", "2,2,4,2"]},
        "trm_depths": {"values": ["0,1,1,2", "0,1,1,1"]},
        "trm_mlp_expansion": {"values": [1.25, 1.5, 2.0]},
        # Recursion breadth
        "H_cycles": {"values": [2, 3]},
        "L_cycles": {"values": [4]},
        "reasoning_depth": {"values": [1, 2]},
        # Early stop threshold encourages shorter effective depth
        "early_stop_threshold": {"uniform": [0.2, 0.5]},
        # Regularization sweeps
        "drop_rate": {"uniform": [0.0, 0.3]},
        "attn_drop_rate": {"uniform": [0.0, 0.3]},
        "drop_path_rate": {"uniform": [0.05, 0.35]},
        }
}

# SwinUnet: valid coupling between embed_dim and num_heads for attention divisibility per stage
# The sampler uses this mapping to pick num_heads given embed_dim; we don't tune num_heads independently.
swin_valid_heads = {
    96:  "3,6,12,24",
    128: "2,4,8,16",
}

# SwinUnet: valid window sizes conditioned on patch_size to reduce padding and keep windows aligned
# Keys are patch_size, values are lists of permissible window_size values
swin_valid_window = {
    4: [7],       # 56x56 initial grid -> 7x7 windows tile cleanly (8 windows per side)
    7: [4, 8],    # 32x32 grid -> 4 or 8 both divide; 4 gives more local focus, 8 fewer windows
}