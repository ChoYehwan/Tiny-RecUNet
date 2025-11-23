import os
import sys
import random
import datetime
import matplotlib.pyplot as plt
from skimage.io import imsave
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import time
from dataset import data_loaders
from utils import (
    Tee,
    DiceLoss,
    dsc_per_volume, dsc_distribution,
    postprocess_per_volume,
    log_scalar_summary, log_loss_summary,
    plot_dsc,
    gray2rgb,
    outline,
)
from config import *
from config import (
    model_name, batch_size, epochs, lr, global_seed,
    wandb_project, wandb_entity, wandb_tags,
)
from models import model_dict

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # 중복 라이브러리 로드 방지

# Optional Weights & Biases
try:
    import wandb
    HAS_WANDB = True
except Exception:
    HAS_WANDB = False

def set_global_seed(seed: int):
    """Set seeds for reproducibility across Python, NumPy, Torch (CPU/CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_validate(run_overrides=None, run_tag=""):
    start_time = time.time()
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")
    print("using device:", device) 

    # Re-seed at the beginning of each run (keeps grid runs identical splits & augment randomness)
    set_global_seed(global_seed)

    # Apply overrides dictionary (grid search combo) else use base config
    local_batch_size = run_overrides.get("batch_size", batch_size) if run_overrides else batch_size
    local_epochs = int(run_overrides.get("epochs", epochs)) if run_overrides else epochs
    local_lr = float(run_overrides.get("lr", lr)) if run_overrides else lr
    local_aug_scale = run_overrides.get("aug_scale", aug_scale) if run_overrides else aug_scale
    local_aug_angle = run_overrides.get("aug_angle", aug_angle) if run_overrides else aug_angle
    local_model_name = model_name  # fixed per experiment

    # Initialize optional W&B run
    wb_run = None
    if HAS_WANDB:
        run_name = f"{exp_name}{run_tag}"
        wb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity if wandb_entity not in (None, "", "None") else None,
            name=run_name,
            group=exp_name,
            tags=wandb_tags if isinstance(wandb_tags, (list, tuple)) else None,
        )

    loader_train, loader_valid, loader_test = data_loaders(local_batch_size, workers, image_size, local_aug_scale, local_aug_angle)
    loaders = {"train": loader_train, "valid": loader_valid}

    # Generic override: use config.tunable_model_args list; parse automatically based on original types.
    orig_args = model_args[local_model_name]
    base_args = dict(orig_args)

    def parse_value(val, orig):
        if val is None:
            return orig
        # Keep None textual
        if isinstance(val, str) and val.strip().lower() in ["none", "null"]:
            return None
        if isinstance(orig, (int, float)):
            try:
                return type(orig)(val)
            except Exception:
                return orig
        if isinstance(orig, (list, tuple)):
            # Accept list/tuple directly
            if isinstance(val, (list, tuple)):
                return list(val) if isinstance(orig, list) else tuple(val)
            if isinstance(val, str):
                txt = val.strip().replace('[','').replace(']','').replace('(','').replace(')','')
                parts = [p.strip() for p in txt.split(',') if p.strip()]
                # Try int cast else keep string
                parsed = []
                for p in parts:
                    try:
                        parsed.append(int(p))
                    except Exception:
                        try:
                            parsed.append(float(p))
                        except Exception:
                            parsed.append(p)
                return list(parsed) if isinstance(orig, list) else tuple(parsed)
        # If orig is None or other type, still try to parse comma-separated lists into a tuple
        if isinstance(val, str) and ("," in val or any(c in val for c in "[]()")):
            txt = val.strip().replace('[','').replace(']','').replace('(','').replace(')','')
            parts = [p.strip() for p in txt.split(',') if p.strip()]
            parsed = []
            for p in parts:
                try:
                    parsed.append(int(p))
                except Exception:
                    try:
                        parsed.append(float(p))
                    except Exception:
                        parsed.append(p)
            if len(parsed) > 0:
                return tuple(parsed)
        # Fallback string
        return val

    # Override model-specific hyperparameters if present in run_overrides
    if run_overrides:
        for key, val in run_overrides.items():
            if key in base_args:
                base_args[key] = parse_value(val, base_args[key])
    # Coupling: For SwinUnet, keep decoder_depths identical to depths (tune only depths)
    if local_model_name == "SwinUnet":
        if "depths" in base_args:
            base_args["decoder_depths"] = list(base_args["depths"]) if isinstance(base_args["depths"], (list, tuple)) else base_args["depths"]

    # Swin family sanity: ensure channels divisible by heads per stage
    def _assert_swin_heads_compat(args_dict):
        if local_model_name not in ("SwinUnet", "SwinU_TRM"):
            return
        try:
            embed = int(args_dict.get("embed_dim", 96))
            depths_local = args_dict.get("depths", (2, 2, 6, 2))
            heads = args_dict.get("num_heads", (3, 6, 12, 24))
            # normalize types
            if isinstance(depths_local, list):
                depths_local = tuple(depths_local)
            if isinstance(heads, list):
                heads = tuple(heads)
            # length check (use min length to be tolerant)
            L = min(len(depths_local), len(heads))
            for i in range(L):
                C_i = embed * (2 ** i)
                h_i = int(heads[i])
                if C_i % h_i != 0:
                    raise ValueError(
                        f"Invalid {local_model_name} config: stage {i} channels {C_i} not divisible by num_heads {h_i}. "
                        f"Try compatible heads (e.g., for embed_dim={embed}: [2,4,8,16] or ensure each C_i divisible by heads_i)."
                    )
        except Exception as e:
            # Re-raise with context
            raise

    # Instantiate and log parameter count
    ModelClass = model_dict[local_model_name]
    # Validate Swin heads compatibility before building model
    _assert_swin_heads_compat(base_args)
    model = ModelClass(**base_args)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[Run {run_tag}] Param count: {param_count}")
    model.to(device)

    dsc_loss = DiceLoss()
    best_validation_dsc = 0.0

    optimizer = optim.Adam(model.parameters(), lr=local_lr)

    loss_train = []
    loss_valid = []
    
    step = 0
    
    train_loss_history = []
    valid_loss_history = []

    # Log config to W&B
    if HAS_WANDB and wb_run is not None:
        cfg_log = {
            "model_name": local_model_name,
            "batch_size": local_batch_size,
            "epochs": local_epochs,
            "lr": local_lr,
            "aug_scale": local_aug_scale,
            "aug_angle": local_aug_angle,
            "image_size": image_size,
            "workers": workers,
            "param_count": int(param_count),
            "overrides": run_overrides,
            "model_args": base_args,
        }
        try:
            wandb.config.update(cfg_log, allow_val_change=True)
            wandb.define_metric("epoch")
            wandb.define_metric("train_loss", step_metric="epoch")
            wandb.define_metric("val_loss", step_metric="epoch")
            wandb.define_metric("val_dsc", step_metric="epoch")
        except Exception:
            pass

    for epoch in range(1, local_epochs + 1):
        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            validation_pred = []
            validation_true = []

            for i, data in enumerate(loaders[phase]):
                if phase == "train":
                    step += 1

                x, y_true = data
                x, y_true = x.to(device), y_true.to(device)
                # if device.type == "cuda":
                #     print("GPU memory allocated:", torch.cuda.memory_allocated() // (1024*1024), "MB")

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    y_pred = model(x)

                    loss = dsc_loss(y_pred, y_true)

                    if phase == "valid":
                        loss_valid.append(loss.item())
                        y_pred_np = y_pred.detach().cpu().numpy()
                        validation_pred.extend(
                            [y_pred_np[s] for s in range(y_pred_np.shape[0])]
                        )
                        y_true_np = y_true.detach().cpu().numpy()
                        validation_true.extend(
                            [y_true_np[s] for s in range(y_true_np.shape[0])]
                        )
                    if phase == "train":
                        loss_train.append(loss.item())
                        loss.backward()
                        optimizer.step()

            if phase == "train":
                mean_train_loss = np.mean(loss_train)
                log_loss_summary(loss_train, epoch)
                train_loss_history.append(mean_train_loss)
                loss_train = []
                # wandb logging (train loss only for curve tracking)
                if HAS_WANDB and wb_run is not None:
                    try:
                        wandb.log({"epoch": epoch, "train_loss": float(mean_train_loss)}, step=epoch)
                    except Exception:
                        pass

            if phase == "valid":
                mean_valid_loss = np.mean(loss_valid)
                log_loss_summary(loss_valid, epoch, prefix="val_")
                valid_loss_history.append(mean_valid_loss)
                mean_dsc = np.mean(
                    dsc_per_volume(
                        validation_pred,
                        validation_true,
                        loader_valid.dataset.patient_slice_index,
                    )
                )
                log_scalar_summary("val_dsc", mean_dsc, epoch)
                if mean_dsc > best_validation_dsc:
                    best_validation_dsc = mean_dsc
                    torch.save(model.state_dict(), os.path.join(weights, f"{local_model_name}.pt"))
                loss_valid = []
                # wandb logging: log val_loss and val_dsc (numeric only)
                if HAS_WANDB and wb_run is not None:
                    try:
                        wandb.log({
                            "epoch": epoch,
                            "val_loss": float(mean_valid_loss),
                            "val_dsc": float(mean_dsc),
                            "best_val_dsc": float(best_validation_dsc),
                        }, step=epoch)
                    except Exception:
                        pass

    os.makedirs(f"./result/{exp_name}", exist_ok=True)
    # Save loss curves after training
    plt.figure()
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(valid_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_curve_filename = f"loss_curve{run_tag}.png" if run_tag else "loss_curve.png"
    loss_curve_path = f'./result/{exp_name}/{loss_curve_filename}'
    plt.savefig(loss_curve_path)
    plt.close()
    
    print(f"\n[Run {run_tag}] Best validation mean DSC: {best_validation_dsc:4f}\n")
    
    # Robust load across torch versions (weights_only introduced in newer torch)
    _ckpt_path = os.path.join(weights, f"{local_model_name}.pt")
    try:
        state_dict = torch.load(_ckpt_path, weights_only=True)
    except TypeError:
        state_dict = torch.load(_ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()

    def evaluate(loader, split_name="val", save_overlays=True):
        input_list = []
        pred_list = []
        true_list = []
        for i, data in enumerate(loader):
            x, y_true = data
            x, y_true = x.to(device), y_true.to(device)
            with torch.set_grad_enabled(False):
                y_pred = model(x)
                y_pred_np = y_pred.detach().cpu().numpy()
                pred_list.extend([y_pred_np[s] for s in range(y_pred_np.shape[0])])
                y_true_np = y_true.detach().cpu().numpy()
                true_list.extend([y_true_np[s] for s in range(y_true_np.shape[0])])
                x_np = x.detach().cpu().numpy()
                input_list.extend([x_np[s] for s in range(x_np.shape[0])])
        volumes_local = postprocess_per_volume(
            input_list,
            pred_list,
            true_list,
            loader.dataset.patient_slice_index,
            loader.dataset.patients,
        )
        dsc_dist_local = dsc_distribution(volumes_local)
        dsc_plot_local = plot_dsc(dsc_dist_local)
        imsave(f"./result/{exp_name}/dsc_{split_name}.png", dsc_plot_local)
        mean_dsc_local = float(np.mean(list(dsc_dist_local.values()))) if len(dsc_dist_local) > 0 else 0.0
        if save_overlays:
            for p in volumes_local:
                x = volumes_local[p][0]
                y_pred = volumes_local[p][1]
                y_true = volumes_local[p][2]
                for s in range(x.shape[0]):
                    image = gray2rgb(x[s, 1])
                    image = outline(image, y_pred[s, 0], color=[255, 0, 0])
                    image = outline(image, y_true[s, 0], color=[0, 255, 0])
                    p_id = p.split("/")[-1].split("\\")[-1]
                    filename = f"{p_id}-{str(s).zfill(2)}.png"
                    out_dir = os.path.join(f"./result/{exp_name}/result_img/{split_name}")
                    os.makedirs(out_dir, exist_ok=True)
                    imsave(os.path.join(out_dir, filename), image)
        return mean_dsc_local

    # Validation evaluation (no overlays); Test evaluation (with overlays)
    mean_val_after = evaluate(loader_valid, split_name=f"val{run_tag}", save_overlays=False)
    mean_test_after = evaluate(loader_test, split_name=f"test{run_tag}", save_overlays=True)

    # Log loss curve image to wandb (only curve image, not overlays)
    if HAS_WANDB and wb_run is not None:
        try:
            if os.path.exists(loss_curve_path):
                wandb.log({"loss_curve_image": wandb.Image(loss_curve_path)})
        except Exception:
            pass

    # wandb: final metrics
    if HAS_WANDB and wb_run is not None:
        try:
            wandb.log({
                "final_val_dsc": float(mean_val_after),
                "final_test_dsc": float(mean_test_after) if mean_test_after is not None else None,
                "best_validation_dsc": float(best_validation_dsc),
            })
        except Exception:
            pass

    # Save summary JSON
    metrics_filename = f"metrics_summary{run_tag}.json" if run_tag else "metrics_summary.json"
    summary_path = f"./result/{exp_name}/{metrics_filename}"
    import json
    with open(summary_path, "w") as f_json:
        json.dump({
            "best_validation_dsc": float(best_validation_dsc),
            "final_validation_dsc": float(mean_val_after),
            "final_test_dsc": float(mean_test_after) if mean_test_after is not None else None,
            "epochs": local_epochs,
            "model_name": local_model_name,
            "run_tag": run_tag,
            "overrides": run_overrides,
        }, f_json, indent=2)
    print(f"Saved metrics summary to {summary_path}")
    # 시간 측정
    total_time = time.time() - start_time
    print(f"Total elapsed time: {total_time:.2f} seconds")

    # wandb finish
    if HAS_WANDB and wb_run is not None:
        try:
            wandb.log({"elapsed_seconds": float(total_time)})
            wandb.finish()
        except Exception:
            pass

if __name__ == "__main__":
    log_filename = f"{exp_name}.log"
    log_file = open(log_filename, 'w')
    sys.stdout = Tee(sys.__stdout__, log_file)
    # Initial global seed (before any dataset construction)
    set_global_seed(global_seed)
    # Simple internal grid search if sweep_space has any 'values' lists
    from config import sweep_space, hyperparameter_tuning, tuning_trials, global_seed
    if not hyperparameter_tuning:
        print("Hyperparameter tuning disabled -> single run with base config.")
        train_validate(run_overrides={}, run_tag="")
    else:
        model_space = sweep_space.get(model_name, {})
        if not model_space:
            print(f"hyperparameter_tuning=True but no sweep_space defined for model '{model_name}' -> single run.")
            train_validate(run_overrides={}, run_tag="")
        else:
            # Random search only (grid removed)
            def continuous_present(space):
                for spec in space.values():
                    if isinstance(spec, dict) and any(k in spec for k in ["uniform", "log_uniform", "int_uniform"]):
                        return True
                return False
            has_continuous = continuous_present(model_space)

            from math import prod, log10
            # For discrete-only params, cap trials by cartesian size; for any continuous param, use trials as-is
            if not has_continuous:
                discrete_lengths = []
                for spec in model_space.values():
                    if isinstance(spec, dict) and 'values' in spec:
                        discrete_lengths.append(len(spec['values']))
                total_cartesian = prod(discrete_lengths) if len(discrete_lengths) > 0 else 0
                trials = min(tuning_trials, total_cartesian) if total_cartesian > 0 else tuning_trials
            else:
                total_cartesian = float('inf')
                trials = tuning_trials

            rng = random.Random(global_seed)

            def sample_param(spec):
                if 'values' in spec:
                    return rng.choice(spec['values'])
                if 'uniform' in spec:
                    lo, hi = spec['uniform']
                    return lo + (hi - lo) * rng.random()
                if 'log_uniform' in spec:
                    lo, hi = spec['log_uniform']
                    log_lo, log_hi = log10(lo), log10(hi)
                    x = log_lo + (log_hi - log_lo) * rng.random()
                    return 10 ** x
                if 'int_uniform' in spec:
                    lo, hi = spec['int_uniform']
                    return rng.randint(int(lo), int(hi))
                return None

            runs = []
            seen = set()
            attempts = 0
            max_attempts = trials * 10
            keys_order = list(model_space.keys())
            # For SwinUnet, restrict sampling to valid (embed_dim, num_heads) pairs
            try:
                from config import swin_valid_heads as _swin_valid_heads_map
                # ensure integer keys
                swin_valid_heads = {int(k): v for k, v in _swin_valid_heads_map.items()}
            except Exception:
                swin_valid_heads = {96: "3,6,12,24", 128: "2,4,8,16"}
            # And restrict window_size choices based on patch_size when provided
            try:
                from config import swin_valid_window as _swin_valid_window_map
                swin_valid_window = {int(k): list(v) for k, v in _swin_valid_window_map.items()}
            except Exception:
                swin_valid_window = {2: [7, 14], 4: [7], 7: [4, 8]}
            swin_embed_choices = None
            if model_name in ("SwinUnet", "SwinU_TRM"):
                # if embed_dim has a provided discrete list, respect it, otherwise default to keys of map
                embed_spec = model_space.get("embed_dim")
                if isinstance(embed_spec, dict) and 'values' in embed_spec:
                    swin_embed_choices = [int(v) for v in embed_spec['values']]
                else:
                    swin_embed_choices = list(swin_valid_heads.keys())
                # ensure choices are valid keys in the map
                swin_embed_choices = [e for e in swin_embed_choices if e in swin_valid_heads]
                # we will sample other keys excluding embed_dim/num_heads/window_size/patch_size directly (handled specially)
                keys_order_wo_embed_heads = [k for k in keys_order if k not in ("embed_dim", "num_heads", "window_size", "patch_size")]
            # helper: build merged args and validate Swin heads compatibility
            def _merge_and_valid(attempt_overrides):
                try:
                    base = dict(model_args[model_name])
                    # lightweight parser for common types
                    def _parse_like_base(key, val):
                        if key not in base:
                            return val
                        orig = base[key]
                        if isinstance(orig, (int, float)):
                            try:
                                return type(orig)(val)
                            except Exception:
                                return orig
                        if isinstance(orig, (list, tuple)):
                            if isinstance(val, (list, tuple)):
                                return list(val) if isinstance(orig, list) else tuple(val)
                            if isinstance(val, str):
                                txt = val.strip().replace('[','').replace(']','').replace('(','').replace(')','')
                                parts = [p.strip() for p in txt.split(',') if p.strip()]
                                parsed = []
                                for p in parts:
                                    try:
                                        parsed.append(int(p))
                                    except Exception:
                                        try:
                                            parsed.append(float(p))
                                        except Exception:
                                            parsed.append(p)
                                return list(parsed) if isinstance(orig, list) else tuple(parsed)
                        return val
                    for k, v in attempt_overrides.items():
                        base[k] = _parse_like_base(k, v)
                    # Swin heads divisibility check
                    if model_name in ("SwinUnet", "SwinU_TRM"):
                        embed = int(base.get("embed_dim", 96))
                        heads = base.get("num_heads", (3,6,12,24))
                        depths_local = base.get("depths", (2,2,6,2))
                        if isinstance(heads, list): heads = tuple(heads)
                        if isinstance(depths_local, list): depths_local = tuple(depths_local)
                        L = min(len(depths_local), len(heads))
                        for i in range(L):
                            C_i = embed * (2 ** i)
                            h_i = int(heads[i])
                            if C_i % h_i != 0:
                                return None  # invalid combo
                    return base
                except Exception:
                    return None
            while len(runs) < trials and attempts < max_attempts:
                attempt = {}
                if model_name in ("SwinUnet", "SwinU_TRM"):
                    # choose a valid embed_dim and its matching heads
                    if not swin_embed_choices:
                        # fallback: if none available, default to 96
                        chosen_embed = 96
                    else:
                        chosen_embed = rng.choice(swin_embed_choices)
                    attempt["embed_dim"] = chosen_embed
                    attempt["num_heads"] = swin_valid_heads[chosen_embed]
                    # sample patch_size first (if specified), then window_size based on mapping
                    patch_spec = model_space.get("patch_size")
                    if isinstance(patch_spec, dict):
                        chosen_patch = sample_param(patch_spec)
                    else:
                        chosen_patch = model_args[model_name].get("patch_size", 4)
                    # coerce to int
                    try:
                        chosen_patch = int(chosen_patch)
                    except Exception:
                        chosen_patch = 4
                    attempt["patch_size"] = chosen_patch
                    # decide window_size by mapping; fallback to default 7
                    if chosen_patch in swin_valid_window and len(swin_valid_window[chosen_patch]) > 0:
                        attempt["window_size"] = rng.choice(swin_valid_window[chosen_patch])
                    else:
                        # If sweep specified window_size independently, respect it; else default
                        win_spec = model_space.get("window_size")
                        attempt["window_size"] = sample_param(win_spec) if isinstance(win_spec, dict) else model_args[model_name].get("window_size", 7)

                    # sample remaining keys normally
                    for k in keys_order_wo_embed_heads:
                        spec = model_space[k]
                        if isinstance(spec, dict):
                            attempt[k] = sample_param(spec)
                else:
                    for k in keys_order:
                        spec = model_space[k]
                        if isinstance(spec, dict):
                            attempt[k] = sample_param(spec)
                # validate attempt for SwinUnet (skip incompatible head/channel combos)
                merged = _merge_and_valid(attempt)
                if merged is None:
                    attempts += 1
                    continue
                if not has_continuous:
                    sig = tuple(attempt[k] for k in keys_order)
                    if sig in seen:
                        attempts += 1
                        continue
                    seen.add(sig)
                runs.append(attempt)
                attempts += 1

            print(f"Running RANDOM search for model '{model_name}' -> trials={len(runs)} (requested={tuning_trials}, cartesian={'∞' if total_cartesian==float('inf') else total_cartesian}, continuous={has_continuous})")
            for idx, overrides in enumerate(runs):
                tag = f"_r{idx}" if len(runs) > 1 else ""
                print(f"\n=== Random Run {idx+1}/{len(runs)} overrides={overrides} ===")
                train_validate(run_overrides=overrides, run_tag=tag)