"""Training entry point for GeoSR super-resolution (Phase 6).

Usage
-----
  python -m model.training.train                 # defaults from TrainConfig
  python -m model.training.train --config cfg.yaml
  python -m model.training.train --resume
  python -m model.training.train --data path/to/image.tif --epochs 20 --loss geosr

The loop reads ``loss_dict["total"]`` for the backward pass (stable across the
l1-only and multi-component GeoSRLoss variants) and logs per-component values.
Validation metrics (PSNR/SSIM/SAM/ERGAS) pick the best checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.architectures import BaselineSR, AdvancedSR, build_advanced, build_baseline, build_geosr_v2
from model.datasets.dataset import DatasetConfig, SatelliteSRDataset
from model.evaluation.evaluate import evaluate_model
from model.losses import GeoSRLoss, ReconstructionLoss
from model.training.checkpoint import CheckpointManager
from model.training.config import TrainConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gpus_per_worker() -> int:
    n = 1 if torch.cuda.is_available() else 0
    return 0 if n == 0 else n


def build_dataloaders(cfg: TrainConfig) -> Dict[str, DataLoader]:
    train_cfg = DatasetConfig(
        tiff_path=cfg.tiff_path,
        split="train",
        patch_size=cfg.patch_size,
        scale_factor=cfg.scale_factor,
        ratios=cfg.ratios,
        seed=cfg.seed,
        use_augmentation=True,
    )
    val_cfg = DatasetConfig(
        tiff_path=cfg.tiff_path,
        split="validation",
        patch_size=cfg.patch_size,
        scale_factor=cfg.scale_factor,
        ratios=cfg.ratios,
        seed=cfg.seed,
        use_augmentation=False,
    )
    g = torch.Generator()
    g.manual_seed(cfg.seed)
    train_ds = SatelliteSRDataset(train_cfg)
    val_ds = SatelliteSRDataset(val_cfg)
    nw = gpus_per_worker()
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cw(cfg.num_workers, nw), worker_init_fn=_worker_init(cfg.seed),
        generator=g, drop_last=True, pin_memory=nw > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(1, cfg.batch_size), shuffle=False,
        num_workers=cw(cfg.num_workers, nw), worker_init_fn=_worker_init(cfg.seed),
        pin_memory=nw > 0,
    )
    print(f"[data] train patches={len(train_ds)} val patches={len(val_ds)} "
          f"scale_factor={cfg.scale_factor} bands={cfg.num_channels}")
    return {"train": train_loader, "val": val_loader}


def cw(cfg_workers: int, ngpu: int) -> int:
    return 0 if ngpu == 0 else cfg_workers


def _worker_init(seed: int):
    def init(worker_id):
        s = seed + 1000 * worker_id
        np.random.seed(s)
        random.seed(s)
    return init


def build_model(cfg: TrainConfig) -> nn.Module:
    if cfg.model_name == "baseline":
        return build_baseline(
            num_channels=cfg.num_channels,
            base_channels=cfg.base_channels,
            num_resblocks=cfg.num_resblocks,
            scale_factor=cfg.scale_factor,
            use_global_residual=cfg.use_global_residual,
        )
    if cfg.model_name == "advanced":
        return build_advanced(
            num_channels=cfg.num_channels,
            base_channels=cfg.base_channels,
            num_groups=cfg.num_groups,
            blocks_per_group=cfg.blocks_per_group,
            reduction=cfg.reduction,
            scale_factor=cfg.scale_factor,
        )
    if cfg.model_name == "geosr_v2":
        # GeoSRv2 is a fixed 10 m -> 5 m (2x) architecture (see architectures/geosr_v2.py).
        if cfg.scale_factor != 2:
            raise ValueError(
                f"GeoSRv2 is trained for scale_factor=2 (10m -> 5m); got {cfg.scale_factor}."
            )
        return build_geosr_v2(num_channels=cfg.num_channels)
    raise ValueError(
        f"Unknown model_name '{cfg.model_name}'. Use 'baseline', 'advanced', or 'geosr_v2'."
    )


def build_loss(cfg: TrainConfig) -> nn.Module:
    lw = cfg.loss_weights
    if cfg.loss == "l1":
        return ReconstructionLoss("l1")
    if cfg.loss in ("geosr", "charbonnier"):
        recon_name = "charbonnier" if cfg.loss == "charbonnier" else "l1"
    else:
        raise ValueError(f"Unknown loss '{cfg.loss}'. Use 'l1', 'geosr' or 'charbonnier'.")
    # ssim+edge are folded into StructuralLoss internal weights; structural_weight
    # is a binary active flag (avoids double-scaling).
    struct_active = 1.0 if (lw.ssim > 0 or lw.edge > 0) else 0.0
    return GeoSRLoss(
        reconstruction=recon_name,
        reconstruction_weight=lw.l1,
        spectral_weight=lw.spectral,
        structural_weight=struct_active,
        ssim_weight=lw.ssim,
        edge_weight=lw.edge,
        data_range=1.5,
    )


def build_optimizer_scheduler(cfg: TrainConfig, model: nn.Module):
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    n_epochs = max(1, cfg.epochs)
    if cfg.lr_scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    elif cfg.lr_scheduler == "multistep":
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=list(cfg.lr_milestones))
    elif cfg.lr_scheduler == "none":
        sched = None
    else:
        raise ValueError(f"Unknown scheduler '{cfg.lr_scheduler}'.")
    return opt, sched


def _log_line(log_path: Optional[Path], line: str) -> None:
    print(line)
    if log_path is not None:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def _validate(model, loader, device, amp):
    m = evaluate_model(model, loader, device=device, data_range=1.5,
                       scale_factor=4.0, use_amp=amp)
    return m


def main(arg_list: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="GeoSR training")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML/JSON config file (overrides defaults).")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint.")
    parser.add_argument("--data", type=str, default=None, help="Override tiff_path.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--loss", type=str, default=None, help="l1 | geosr | charbonnier")
    parser.add_argument("--model", type=str, default=None, help="baseline | advanced")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(arg_list)

    cfg = TrainConfig()
    if args.config:
        cfg = _load_config(args.config, cfg)
    # CLI overrides
    if args.data:
        cfg.tiff_path = args.data
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.loss is not None:
        cfg.loss = args.loss
    if args.model is not None:
        cfg.model_name = args.model
    if args.seed is not None:
        cfg.seed = args.seed

    set_seed(cfg.seed)
    device = cfg.device
    run_name = f"{cfg.model_name}_{cfg.loss}_{cfg.experiment_name}"
    run_dir = cfg.checkpoint_dir_for_run(run_name)
    cm = CheckpointManager(run_dir)
    log_path = cm.run_dir / "train.log"
    log_path.write_text("", encoding="utf-8")
    cm.save_config(cfg)

    loaders = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    criterion = build_loss(cfg).to(device)
    optimizer, scheduler = build_optimizer_scheduler(cfg, model)
    scaler = torch.cuda.amp.GradScaler() if (cfg.amp_enabled) else None
    amp = cfg.amp_enabled

    start_epoch = 0
    best_val = -math.inf
    if args.resume:
        ckpt = cm.load(model, optimizer, scheduler, scaler, device=device)
        start_epoch = int(ckpt.get("epoch", 0))
        best_val = float(ckpt.get("best_metric", -math.inf))
        _log_line(log_path, f"[resume] from epoch {start_epoch} best={best_val:.4f}")

    metric_name = "psnr"
    from dataclasses import asdict as _asdict
    lw_cfg = _asdict(cfg.loss_weights)
    print(f"[train] model={cfg.model_name} loss={cfg.loss} epochs={cfg.epochs} "
          f"device={device} amp={amp} loss_weights={lw_cfg}")

    for epoch in range(start_epoch, cfg.epochs):
        loaders["train"].dataset.set_epoch(epoch)
        model.train()
        running: Dict[str, float] = {}
        running["total"] = 0.0
        seen = 0
        t0 = time.time()
        for it, batch in enumerate(loaders["train"]):
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            optimizer.zero_grad(set_to_none=True)
            if amp:
                with torch.autocast("cuda"):
                    pred = model(lr)
                    loss_dict = criterion(pred, hr)
            else:
                pred = model(lr)
                loss_dict = criterion(pred, hr)
            loss = loss_dict["total"]
            if scaler is not None:
                scaler.scale(loss).backward()
                if cfg.clip_grad > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
                optimizer.step()

            bs = lr.shape[0]
            running["total"] += loss.item() * bs
            seen += bs
            for k, v in loss_dict.items():
                if k != "total" and isinstance(v, torch.Tensor):
                    running[k] = running.get(k, 0.0) + float(v.item()) * bs
            if (it + 1) % 25 == 0:
                avg = running["total"] / max(1, seen)
                _log_line(log_path, f"  epoch {epoch} iter {it+1} loss_total={avg:.4f} "
                                    f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if scheduler is not None:
            scheduler.step()

        avg_train = running["total"] / max(1, seen)
        val_metrics = _validate(model, loaders["val"], device, amp)
        val_psnr = val_metrics["psnr"]
        is_best = val_psnr > best_val
        best_val = max(best_val, val_psnr)
        ckpt_path = cm.save(
            model, optimizer, scheduler, scaler,
            config=cfg, epoch=epoch, iteration=0,
            best_metric=best_val, metric_name=metric_name,
            model_name=cfg.model_name, is_best=is_best,
            extra={"train_loss": avg_train, "val_metrics": val_metrics},
        )
        comp = {k: running.get(k, 0.0) / max(1, seen) for k in ("reconstruction", "spectral", "structural")}
        _log_line(
            log_path,
            f"[epoch {epoch:03d}] train_loss={avg_train:.4f} "
            f"({comp['reconstruction']:.3f}/{comp['spectral']:.3f}/{comp['structural']:.3f}) "
            f"val_psnr={val_psnr:.3f} ssim={val_metrics['ssim']:.3f} "
            f"sam={val_metrics['sam']:.3f} ergas={val_metrics['ergas']:.3f} "
            f"{'BEST' if is_best else ''} -> {ckpt_path.name} "
            f"({time.time()-t0:.1f}s)"
        )

    torch.save({"run_dir": str(cm.run_dir)}, cm.run_dir / "run_meta.json")
    _log_line(log_path, f"[done] best_val_{metric_name}={best_val:.4f}")
    return 0


def _load_config(path: str, base: TrainConfig) -> TrainConfig:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(text) or {}
    except Exception:
        data = json.loads(text)
    merged = base.as_dict()
    # as_dict adds device/amp_enabled; from_dict strips unknown keys.
    merged.update(data)
    return TrainConfig.from_dict(merged)


if __name__ == "__main__":
    raise SystemExit(main())
