"""Task 2 (CFP + registered FFA cross-modal AV segmentation) training entrypoint.

Requires registered FFA images -- CFP/FFA pairs are NOT pre-aligned (verified
empirically: raw ORB matching gives ~256px displacement std, essentially
noise). Run src/registration/minima_wrapper.py on both splits first.

Usage:
    python src/train_task2.py --fold 0 --epochs 60 \
        --data-root data/raw/GAVE2_preliminary --ffa-root data/registered \
        --warm-start-task1 runs/task1/fold0/latest.pth
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets.gave_dataset import GaveAVDataset  # noqa: E402
from datasets.splits import kfold_case_ids  # noqa: E402
from losses.rrloss import BCE3Loss, RRLoss, RRClDiceLoss  # noqa: E402
from losses.cldice import ArteryVeinClDiceLoss  # noqa: E402
from models.rrwnet import build_model, transfer_task1_to_task2  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="data/raw/GAVE2_preliminary")
    p.add_argument("--ffa-root", type=str, default="data/registered", help="Output of minima_wrapper.py")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--patch-size", type=int, default=384, help="See train_task1.py -- 384 verified stable on the T550")
    p.add_argument("--base-ch", type=int, default=64)
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--steps-per-epoch", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--amp-dtype", type=str, default="bf16", choices=["none", "fp16", "bf16"])
    p.add_argument("--warm-start-task1", type=str, default=None, help="Path to a trained Task1 checkpoint (latest.pth or final.pth) to initialize the RGB encoder + decoder from")
    p.add_argument("--pos-weight", type=float, default=5.0, help="See train_task1.py -- upweights vessel-positive pixels, 0 disables")
    p.add_argument(
        "--vein-pos-weight", type=float, default=8.0,
        help="Separate (higher) pos_weight for the vein channel only. Real leaderboard data (2026-07-18) showed "
             "Task2's vein topology trails artery badly (COR 0.35 vs 0.61, INF 0.65 vs 0.38) under a uniform "
             "pos_weight -- this pushes vein harder independently. Set equal to --pos-weight to disable the asymmetry.",
    )
    p.add_argument("--cldice-weight", type=float, default=0.3, help="See train_task1.py -- soft-clDice topology loss weight, 0 disables")
    p.add_argument(
        "--vein-topology-ratio", type=float, default=2.0,
        help="Relative weight of vein vs artery inside the clDice term (artery fixed at 1.0). "
             "Same motivation as --vein-pos-weight -- vein topology needs more pressure than artery here. 1.0 = symmetric.",
    )
    p.add_argument("--out-dir", type=str, default="runs/task2")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--max-seconds", type=float, default=None)
    p.add_argument("--checkpoint-every-epochs", type=int, default=2)
    p.add_argument("--val-every-epochs", type=int, default=5, help="See train_task1.py")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=77)
    p.add_argument(
        "--fusion", type=str, default="additive", choices=["additive", "xattn"],
        help="'additive' = baseline SE-gated add fusion (model.py's NewUNetModule). "
             "'xattn' = cross-attention fusion between RGB and FFA branches at each "
             "encoder scale (models/cmrrwnet_xattn.py) -- no spatial interaction "
             "between modalities exists in the additive version.",
    )
    return p.parse_args()


@torch.no_grad()
def run_validation(model, val_loader, criterion, device, amp_enabled, amp_torch_dtype):
    model.eval()
    total_loss = 0.0
    n = 0
    for batch in val_loader:
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        roi = batch["roi"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=amp_torch_dtype, enabled=amp_enabled):
            predictions = model(image)
            loss = criterion(predictions, label, roi)
        total_loss += loss.item()
        n += 1
    model.train()
    return total_loss / max(n, 1)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    splits = kfold_case_ids(n_cases=50, n_folds=args.n_folds, seed=args.seed)
    train_ids, val_ids = splits[args.fold]
    print(f"Fold {args.fold}: {len(train_ids)} train / {len(val_ids)} val cases")

    train_ds = GaveAVDataset(
        args.data_root, split="training", case_ids=train_ids, patch_size=args.patch_size,
        use_ffa=True, train=True, seed=args.seed, ffa_root=args.ffa_root,
    )
    val_ds = GaveAVDataset(
        args.data_root, split="training", case_ids=val_ids, patch_size=args.patch_size,
        use_ffa=True, train=False, seed=args.seed + 1, ffa_root=args.ffa_root,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_task = "task2_xattn" if args.fusion == "xattn" else "task2"
    model = build_model(model_task, base_ch=args.base_ch, iterations=args.iterations, pretrained=False).to(device)

    if args.warm_start_task1:
        ckpt = torch.load(args.warm_start_task1, map_location=device)
        task1_sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model = transfer_task1_to_task2(model, task1_sd)

    base_criterion = BCE3Loss(
        pos_weight=args.pos_weight if args.pos_weight > 0 else None,
        vein_pos_weight=args.vein_pos_weight if args.pos_weight > 0 else None,
    )
    if args.cldice_weight > 0:
        cldice_loss = ArteryVeinClDiceLoss(artery_weight=1.0, vein_weight=args.vein_topology_ratio)
        criterion = RRClDiceLoss(base_criterion, cldice_loss, cldice_weight=args.cldice_weight)
    else:
        criterion = RRLoss(base_criterion)
    criterion = criterion.to(device)  # pos_weight is a buffer, must move with the module
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    amp_enabled = args.amp_dtype != "none" and device.type == "cuda"
    amp_torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "none": None}[args.amp_dtype]
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and args.amp_dtype == "fp16")

    out_dir = Path(args.out_dir) / f"fold{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    global_step = 0
    start_epoch = 0
    best_val_loss = float("inf")
    latest_path = out_dir / "latest.pth"
    best_path = out_dir / "best.pth"
    log_path = out_dir / "train_log.csv"

    if args.resume and latest_path.exists():
        ckpt = torch.load(latest_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"]
        global_step = ckpt["global_step"]
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from {latest_path}: epoch={start_epoch} global_step={global_step}")
    else:
        with open(log_path, "w") as f:
            f.write("epoch,step,loss,val_loss,elapsed_s\n")

    def save_checkpoint(path: Path, epoch: int):
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "scaler": scaler.state_dict(), "epoch": epoch, "global_step": global_step,
             "best_val_loss": best_val_loss},
            path,
        )

    t_start = time.time()
    last_epoch_duration = 0.0
    for epoch in range(start_epoch, args.epochs):
        if args.max_seconds is not None:
            elapsed_so_far = time.time() - t_start
            if elapsed_so_far + last_epoch_duration * 1.2 > args.max_seconds:
                print(f"[max_seconds budget] stopping before epoch {epoch+1} (elapsed={elapsed_so_far:.0f}s, budget={args.max_seconds:.0f}s)")
                save_checkpoint(latest_path, epoch)
                return
        epoch_t0 = time.time()
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        train_iter = iter(train_loader)
        for step in range(args.steps_per_epoch):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            image = batch["image"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            roi = batch["roi"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_torch_dtype, enabled=amp_enabled):
                predictions = model(image)
                loss = criterion(predictions, label, roi)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            if torch.isnan(loss):
                raise RuntimeError(
                    f"NaN loss at step {global_step} (epoch {epoch+1}) -- amp_dtype={args.amp_dtype}. "
                    "See train_task1.py's note on fp16 instability on some GPUs; try --amp-dtype bf16 or none."
                )

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            if args.max_steps is not None and global_step >= args.max_steps:
                print(f"[smoke test] reached max_steps={args.max_steps}, stopping")
                torch.save(model.state_dict(), out_dir / "smoke_test_ckpt.pth")
                return

        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t_start
        last_epoch_duration = time.time() - epoch_t0

        val_loss_str = ""
        val_loss = None
        if args.val_every_epochs > 0 and ((epoch + 1) % args.val_every_epochs == 0 or epoch == args.epochs - 1):
            val_loss = run_validation(model, val_loader, criterion, device, amp_enabled, amp_torch_dtype)
            val_loss_str = f"  val_loss={val_loss:.4f}"
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(best_path, epoch + 1)
                val_loss_str += " (best, saved)"

        print(f"epoch {epoch+1}/{args.epochs}  loss={avg_loss:.4f}{val_loss_str}  elapsed={elapsed:.0f}s  epoch_dur={last_epoch_duration:.0f}s")
        with open(log_path, "a") as f:
            f.write(f"{epoch+1},{global_step},{avg_loss:.6f},{val_loss if val_loss is not None else ''},{elapsed:.1f}\n")

        if (epoch + 1) % args.checkpoint_every_epochs == 0 or epoch == args.epochs - 1:
            save_checkpoint(latest_path, epoch + 1)
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            torch.save(model.state_dict(), out_dir / f"epoch{epoch+1}.pth")

    torch.save(model.state_dict(), out_dir / "final.pth")


if __name__ == "__main__":
    main()
