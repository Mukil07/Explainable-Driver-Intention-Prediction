"""
Evaluate the released VCBM i3d checkpoint (i3d_nosim.pth) on the DAAD-X val split.

Mirrors models/eval.py + tools/engine.py:val() for the -ego_cbm -multitask -bottleneck
configuration, with two deliberate deviations (both reported at runtime):

  1. load_state_dict(strict=False) -- the released checkpoint has no
     first_model.tm.center_coord (it was trained with the composite-similarity
     block disabled, which is also the default in model/TM.py::forward).
  2. The final partial batch is padded up to batch=8 and the padded rows are
     masked out of the metrics, so all N val samples are scored rather than
     eval.py's drop_last=True (which would silently discard the tail).
     Batch must stay 8: first_model.tm.centers is shaped (8, K, 2048).

Run from the models/ directory.
"""
import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm.auto import tqdm

from model import build_model
from utils.loader import CustomDataset

BATCH = 8  # fixed by the shape of first_model.tm.centers


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pad_collate(batch):
    """Pad the final short batch up to BATCH by repeating its last element.

    Returns the usual 5-tuple plus a bool mask marking the real samples.
    """
    n = len(batch)
    padded = list(batch) + [batch[-1]] * (BATCH - n)
    img1 = torch.stack([b[0] for b in padded])
    img2 = torch.stack([b[1] for b in padded])
    cls = torch.tensor([b[2] for b in padded])
    gaze = torch.tensor([b[3] for b in padded])
    ego = torch.tensor([b[4] for b in padded], dtype=torch.float)
    mask = torch.zeros(BATCH, dtype=torch.bool)
    mask[:n] = True
    return img1, img2, cls, gaze, ego, mask


def score_model(model, loader, device, ego_threshold=0.5, dump=None):
    """Score a model over `loader` and print the metric block.

    Shared by eval_daadx.py and eval.py so both report identical numbers.
    `loader` must use pad_collate (6-tuple batches with a validity mask).
    Returns (y, yh, yg, ygh, ye, yeh).
    """
    crit_man = nn.CrossEntropyLoss()
    crit_ego = nn.BCEWithLogitsLoss()

    Lg_raw, Le_raw, Lm_raw = [], [], []   # raw logits for diagnostics
    P, L = [], []            # maneuver
    Pg, Lg = [], []          # gaze
    Pe, Le = [], []          # ego (multilabel)
    running_loss, nb = 0.0, 0

    with torch.no_grad():
        for img1, img2, cls, gaze, ego, mask in tqdm(loader):
            img1 = img1.to(device).float()
            img2 = img2.to(device).float()
            label = cls.to(device)
            ego = ego.to(device)

            outputs = model(img1, img2)

            loss = crit_man(outputs[0], label) + 0.5 * crit_ego(
                torch.hstack(outputs[2:]), ego
            )
            running_loss += loss.item()
            nb += 1

            m = mask.cpu()
            P.append(torch.argmax(outputs[0], dim=1).cpu()[m])
            L.append(cls[m])
            Pg.append(torch.argmax(outputs[1], dim=1).cpu()[m])
            Lg.append(gaze[m])
            Pe.append((torch.sigmoid(torch.hstack(outputs[2:])) > ego_threshold).float().cpu()[m])
            Le.append(ego.cpu()[m])
            Lm_raw.append(outputs[0].cpu()[m])
            Lg_raw.append(outputs[1].cpu()[m])
            Le_raw.append(torch.hstack(outputs[2:]).cpu()[m])

    y, yh = np.hstack(L), np.hstack(P)
    yg, ygh = np.hstack(Lg), np.hstack(Pg)
    ye, yeh = torch.cat(Le).numpy(), torch.cat(Pe).numpy()

    print("\n" + "=" * 66)
    print(f"scored samples: {len(y)}   batches: {nb}   mean loss: {running_loss/nb:.4f}")
    print("=" * 66)
    print("MANEUVER (7-way)")
    print(f"  accuracy        {accuracy_score(y, yh):.4f}")
    print(f"  f1 (weighted)   {f1_score(y, yh, average='weighted', zero_division=0):.4f}")
    print(f"  f1 (macro)      {f1_score(y, yh, average='macro', zero_division=0):.4f}")
    print("\nGAZE EXPLANATION (15-way)")
    print(f"  accuracy        {accuracy_score(yg, ygh):.4f}")
    print(f"  f1 (weighted)   {f1_score(yg, ygh, average='weighted', zero_division=0):.4f}")
    print(f"\nEGO EXPLANATION (17 multilabel)  [threshold={ego_threshold}]")
    print(f"  subset accuracy {accuracy_score(ye, yeh):.4f}")
    print(f"  f1 (weighted)   {f1_score(ye, yeh, average='weighted', zero_division=0):.4f}")
    print(f"  f1 (micro)      {f1_score(ye, yeh, average='micro', zero_division=0):.4f}")
    print(f"  f1 (macro)      {f1_score(ye, yeh, average='macro', zero_division=0):.4f}")
    if dump:
        np.savez(dump, y=y, yh=yh, yg=yg, ygh=ygh, ye=ye, yeh=yeh,
                 man_logits=torch.cat(Lm_raw).numpy(),
                 gaze_logits=torch.cat(Lg_raw).numpy(),
                 ego_logits=torch.cat(Le_raw).numpy())
        print(f"\n[dump] wrote {dump}")
    print("\nper-maneuver breakdown")
    print(classification_report(y, yh, zero_division=0, digits=3))
    return y, yh, yg, ygh, ye, yeh


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--csv", type=str, default="DATA/val.csv")
    p.add_argument("--model", type=str, default="i3d_proposed")
    p.add_argument("--dataset", type=str, default="dipx")
    p.add_argument("--technique", type=str, default="i3d_nosim_eval")
    p.add_argument("--num_classes", type=int, default=7)
    p.add_argument("--n_attributes", type=int, default=17)
    p.add_argument("--multitask_classes", type=int, default=15)
    p.add_argument("--clusters", type=int, default=10)
    p.add_argument("--dropout", type=float, default=0.45)
    p.add_argument("--expand_dim", type=int, default=0)
    p.add_argument("--connect_CY", type=bool, default=False)
    p.add_argument("--use_relu", type=bool, default=False)
    p.add_argument("--use_sigmoid", type=bool, default=False)
    p.add_argument("--batch", type=int, default=BATCH)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=37)
    p.add_argument("--debug", type=str, default=None)
    p.add_argument("-bottleneck", action="store_true")
    p.add_argument("-gaze_cbm", action="store_true")
    p.add_argument("-ego_cbm", action="store_true")
    p.add_argument("-multitask", action="store_true")
    p.add_argument("-combined_bottleneck", action="store_true")
    p.add_argument("--dump", type=str, default=None)
    p.add_argument("-no_composite_sim", action="store_true",
                   help="DISABLE the full LTM composite spatio-temporal similarity "
                        "(model/TM.py::use_composite_sim). Composite sim is ON by default "
                        "here, matching train_sim.py. Pass this for the released 'nosim' "
                        "checkpoints (i3d_nosim.pth, mvitv2_nosim.pth), which were trained "
                        "with feature-cosine clustering only.")
    p.add_argument("-composite_sim", action="store_true",
                   help="deprecated no-op; composite sim is now the default.")
    p.add_argument("--sampling", type=str, default="repo",
                   choices=["repo", "repo_det", "paper", "paper_det"],
                   help="repo = decode_video as shipped: [start-4s, end+1s] window from "
                        "time.csv with random jitter. paper = the scheme described in the "
                        "paper (severity s=1): split the FULL clip into 16 equal segments and "
                        "sample one frame uniformly at random from each. paper_det = same "
                        "segments but take each segment's centre frame (deterministic). "
                        "repo_det = the repo window with the random jitter REMOVED, which is "
                        "the correct eval protocol: the supplement lists that jitter as a "
                        "training augmentation.")
    p.add_argument("--input_norm", type=str, default="raw",
                   choices=["raw", "pm1", "unit", "rgb_pm1"],
                   help="raw = repo default (0-255 BGR, decode_video applies nothing); "
                        "pm1 = (x/255)*2-1 as load_rgb_frames does; unit = x/255; "
                        "rgb_pm1 = BGR->RGB then (x/255)*2-1, i.e. full pytorch-i3d preprocessing")
    p.add_argument("--ego_threshold", type=float, default=0.5,
                   help="sigmoid cutoff for the 17 ego attributes; the repo hardcodes 0.5")
    args = p.parse_args()

    assert args.batch == BATCH, "batch must be 8 (tm.centers is (8, K, 2048))"

    if args.sampling != "repo":
        import cv2
        import utils.loader as _loader

        def paper_decode(video_path, start, end, num_frames=16, _mode=args.sampling):
            """16 frames, one per equal segment of the WHOLE clip (paper, severity s=1).

            start/end from time.csv are deliberately ignored -- the paper describes
            segmenting the full video, not a window around the maneuver.
            """
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video {video_path}")
            T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if T < 1:
                cap.release()
                return np.array([]), True
            edges = np.linspace(0, T, num_frames + 1)
            frames = []
            for i in range(num_frames):
                lo, hi = int(edges[i]), max(int(edges[i + 1]), int(edges[i]) + 1)
                idx = (lo + hi - 1) // 2 if _mode == "paper_det" else random.randint(lo, hi - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(idx, T - 1))
                ok, fr = cap.read()
                if not ok:
                    if frames:
                        frames.append(frames[-1])
                        continue
                    cap.release()
                    return np.array([]), True
                frames.append(fr)
            cap.release()
            return np.array(frames), False

        def repo_det_decode(video_path, start, end, num_frames=16):
            """decode_video() as shipped, minus the random temporal jitter."""
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video {video_path}")
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            if start <= 4:
                start_time, end_time = 0, end * frame_rate + 1 * frame_rate
            else:
                start_time = start * frame_rate - 4 * frame_rate
                end_time = end * frame_rate + 1 * frame_rate
            interval = max((end_time - start_time) // num_frames, 1)
            frames = []
            for i in range(int(start_time), int(end_time), int(interval)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)      # <-- no jitter added to i
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                if len(frames) == num_frames:
                    break
            cap.release()
            flag = False
            try:
                while len(frames) < num_frames:
                    frames.append(frames[-1])
            except Exception:
                flag = True
            return np.array(frames), flag

        _loader.decode_video = (repo_det_decode if args.sampling == "repo_det"
                                else paper_decode)
        print(f"[data] sampling = {args.sampling} (decode_video monkeypatched)")
    seed_all(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(args).to(device)
    model.first_model.tm.use_composite_sim = not args.no_composite_sim
    print(f"[LTM] composite similarity = {model.first_model.tm.use_composite_sim}"
          + ("  (nosim mode: -no_composite_sim)" if args.no_composite_sim else ""))

    ckp = torch.load(args.weights, map_location=device)
    missing, unexpected = model.load_state_dict(ckp, strict=False)
    print(f"[ckpt] loaded {args.weights}")
    print(f"[ckpt] missing keys   ({len(missing)}): {list(missing)}")
    print(f"[ckpt] unexpected keys({len(unexpected)}): {list(unexpected)}")
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    class NormWrapper(torch.utils.data.Dataset):
        """Apply the preprocessing that decode_video() omits (load_rgb_frames does it)."""
        def __init__(self, base, mode):
            self.base, self.mode = base, mode
        def __len__(self):
            return len(self.base)
        def __getitem__(self, i):
            v1, v2, t, g, e = self.base[i]
            def f(v):
                v = v.float()
                if self.mode == "raw":
                    return v
                if self.mode == "unit":
                    return v / 255.0
                if self.mode == "pm1":
                    return (v / 255.0) * 2 - 1
                if self.mode == "rgb_pm1":
                    return (v[[2, 1, 0]] / 255.0) * 2 - 1
            return f(v1), f(v2), t, g, e

    dataset = CustomDataset(args.csv, debug=args.debug)
    if args.input_norm != "raw":
        base = dataset
        dataset = NormWrapper(base, args.input_norm)
        dataset.face_path, dataset.road_path = base.face_path, base.road_path
    print(f"[data] input_norm = {args.input_norm}")
    print(f"[data] csv={args.csv}  samples with a front_view video present: {len(dataset)}")
    print(f"[data] gaze branch dir: {dataset.face_path}")
    print(f"[data] road branch dir: {dataset.road_path}")

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH, shuffle=False, drop_last=False,
        num_workers=args.workers, pin_memory=True, collate_fn=pad_collate,
    )

    score_model(model, loader, device, ego_threshold=args.ego_threshold, dump=args.dump)


if __name__ == "__main__":
    main()
