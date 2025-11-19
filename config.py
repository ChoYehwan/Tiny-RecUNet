import datetime

hyperparameter_tuning = True
# Random search configuration 
tuning_trials = 2

# Weights & Biases
wandb_project = "TRMUnet"  # your W&B project name
wandb_entity = "choyh0909-handong-global-university"          # set to your W&B entity/org or keep None
wandb_tags = []             # e.g., [model_name]


model_name = "SwinUnet"
exp_name = model_name + "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

batch_size = 16
epochs = 50
lr = 0.0001
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
        "depths": [2, 2, 6, 2],
        "decoder_depths": [2, 2, 6, 2],
        "num_heads": [3, 6, 12, 24],
        "window_size": 7,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "qk_scale": None,
        "ape": False,
        "patch_norm": True,
        "final_upsample": "expand_first",
    }
}

# Define tunable parameter names for generic overriding from wandb.config
# Central sweep space (ranges/choices). If left empty {} only a single run happens.
sweep_space = {
    # Global training hyperparameters can repeat in each model section if desired
    "TransUnet": {
        # Continuous ranges (random strategy only): lr log-uniform, aug_scale uniform, aug_angle int uniform, embed_dropout uniform
        "lr": {"log_uniform": [1e-5, 1e-3]},
        "batch_size": {"values": [8, 16]},  # discrete
        "aug_scale": {"uniform": [0.0, 0.10]},
        "aug_angle": {"int_uniform": [0, 25]},
        "embed_dim": {"values": [128, 256]},
        "depth": {"values": [4, 6]},
        "num_heads": {"values": [4, 8]},
        "embed_dropout": {"uniform": [0.0, 0.3]},
        "decoder_channels": {"values": ["256,128,64,16", "512,256,128,32"]},
        "head_channels": {"values": [512, 256]},
        "patch_grid": {"values": ["None", "14,14"]},
    },
    "SwinUnet": {
        "lr": {"log_uniform": [5e-5, 5e-4]},
        "batch_size": {"values": [8, 16]},
        "aug_scale": {"uniform": [0.0, 0.10]},
        "aug_angle": {"int_uniform": [0, 25]},
        "embed_dim": {"values": [96, 128]},
        "patch_size": {"values": [4]},
        "depths": {"values": ["2,2,6,2", "2,2,4,2"]},
        "decoder_depths": {"values": ["2,2,6,2", "2,2,4,2"]},
        "num_heads": {"values": ["3,6,12,24", "2,4,8,16"]},
        "window_size": {"values": [7, 5]},
        "final_upsample": {"values": ["expand_first", "pixelshuffle"]},
    },
    # UNet (example minimal tuning)
    "UNet": {
        "lr": {"values": [1e-4, 2e-4]},
        "batch_size": {"values": [8, 16]},
        "aug_scale": {"values": [0.0, 0.05]},
        "aug_angle": {"values": [0, 15]},
    }
}

