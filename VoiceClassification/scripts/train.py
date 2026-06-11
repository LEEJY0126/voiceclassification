"""
train.py

Usage:
    python3 train.py
    python3 train.py --config config.yaml
"""

import sys
import argparse
import yaml
from pathlib import Path

# VoiceClassification/ 을 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from voicebuffer import VoiceBuffer, BufferConfig   # voicebuffer/__init__.py
from utils.dataset import build_dataloaders         # utils/dataset.py
from model.model import VoiceClassifier             # model/model.py


# ------------------------------------------------------------------
# Config 로드
# ------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------
# 1 epoch 학습
# ------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(dim=1) == y).sum().item()
        total      += len(y)

    return total_loss / total, correct / total


# ------------------------------------------------------------------
# 1 epoch 검증
# ------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss   = criterion(logits, y)

        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(dim=1) == y).sum().item()
        total      += len(y)

    return total_loss / total, correct / total


# ------------------------------------------------------------------
# 학습 루프
# ------------------------------------------------------------------

def train(config_path: str = "config.yaml"):

    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent.parent / config_path
        
    cfg = load_config(config_path)

    # 시드 고정
    seed = cfg["train"]["seed"]
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # DataLoader
    buf_cfg = BufferConfig(
        sample_rate=cfg["buffer"]["sample_rate"],
        window_sec =cfg["buffer"]["window_sec"],
        hop_sec    =cfg["buffer"]["hop_sec"],
    )
    train_loader, val_loader = build_dataloaders(
        data_dir   =cfg["data"]["data_dir"],
        val_ratio  =cfg["data"]["val_ratio"],
        max_windows_per_file= cfg["data"]["max_windows_per_file"],
        call_pattern = cfg["data"]["call_pattern"],
        noise_call_ratio= cfg["data"]["noise_call_ratio"],
        batch_size =cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
        cfg        =buf_cfg,
        seed       =seed,
    )

    # 모델
    model = VoiceClassifier().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"파라미터: {total_params:,} ({total_params/1e6:.2f}M)\n")

    # 손실함수 / 옵티마이저 / 스케줄러
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           =cfg["optimizer"]["lr"],
        weight_decay =cfg["optimizer"]["weight_decay"],
    )
    scheduler = None
    if cfg["scheduler"]["use"]:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode    ="min",
            patience=cfg["scheduler"]["patience"],
            factor  =cfg["scheduler"]["factor"],
            min_lr  =cfg["scheduler"]["min_lr"],
        )

    # 체크포인트 폴더
    save_dir = Path(cfg["checkpoint"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # 학습 루프
    best_val_loss = float("inf")
    epochs        = cfg["train"]["epochs"]

    print(f"{'Epoch':>6} │ {'Train Loss':>10} {'Train Acc':>10} │ {'Val Loss':>10} {'Val Acc':>10} │ {'LR':>10}")
    print("─" * 70)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_acc   = validate(model, val_loader, criterion, device)

        if scheduler:
            scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"{epoch:>6} │ {train_loss:>10.4f} {train_acc:>9.1%} │"
            f" {val_loss:>10.4f} {val_acc:>9.1%} │ {current_lr:>10.6f}"
        )

        # best 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = save_dir / "best.pt"
            torch.save({
                "epoch"     : epoch,
                "model_state": model.state_dict(),
                "val_loss"  : val_loss,
                "val_acc"   : val_acc,
                "config"    : cfg,
            }, ckpt_path)
            print(f"         └─ ✅ best 저장 (val_loss={val_loss:.4f})")

    print("\n학습 완료!")
    print(f"best val_loss : {best_val_loss:.4f}")
    print(f"체크포인트    : {save_dir / 'best.pt'}")


# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    train(args.config)