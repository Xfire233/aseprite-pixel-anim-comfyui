#!/usr/bin/env python
"""Frame-to-frame consistency validation for a pixel loop (run BEFORE reporting a loop done).

Usage: uv run --with pillow --with numpy python scripts/validate_loop.py --frames work/idle/frame*.png
Checks (PASS/FAIL each, exit 1 on any FAIL):
  bbox spread | silhouette IoU (adjacent + closure) | centroid_y spread |
  face sharpness outlier | adj-diff coherence
"""
import argparse, glob, sys

import numpy as np
from PIL import Image


def load(p):
    return np.array(Image.open(p).convert("RGBA")).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", nargs="+", required=True, help="frame PNGs (unique frames; globs ok in shell)")
    ap.add_argument("--iou-min", type=float, default=0.78)
    ap.add_argument("--belt-spread", type=float, default=12.0, help="vertical motion excursion tolerance")
    ap.add_argument("--face-ratio", type=float, default=0.65)
    ap.add_argument("--band-spread", type=int, default=3)
    ap.add_argument("--adj-max", type=int, default=10500, help="max mean adjacent-frame diff px (motion/lighting coherence budget)")
    ap.add_argument("--w-spread", type=int, default=12, help="max bbox width spread across loop")
    args = ap.parse_args()
    paths = []
    for p in args.frames:
        paths += sorted(glob.glob(p)) if "*" in p else [p]
    frames = [load(p) for p in paths]
    n = len(frames)
    H, W = frames[0].shape[:2]
    ok = True

    def check(name, passed, detail):
        nonlocal ok
        ok = ok and passed
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")

    alphas = [f[:, :, 3] > 0 for f in frames]
    boxes = []
    for a in alphas:
        ys, xs = np.where(a)
        boxes.append((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    ws = [b[2] - b[0] for b in boxes]; hs = [b[3] - b[1] for b in boxes]
    tops = [b[1] for b in boxes]; lefts = [b[0] for b in boxes]
    check("bbox spread", max(ws) - min(ws) <= args.w_spread and max(hs) - min(hs) <= 18 and max(tops) - min(tops) <= 18 and max(lefts) - min(lefts) <= max(6, (args.w_spread + 1) // 2),
          f"w{ws} h{hs} top{tops} left{lefts}")

    ious = []
    for i in range(n):
        a, b = alphas[i], alphas[(i + 1) % n]
        ious.append(float((a & b).sum() / max(1, (a | b).sum())))
    check("silhouette IoU", min(ious) >= args.iou_min, f"{[round(v,3) for v in ious]} min>={args.iou_min}")

    centroids_y = []
    for a in alphas:
        ys = np.where(a)[0]
        centroids_y.append(float(ys.mean()) if len(ys) > 20 else 0.0)
    check("centroid_y spread", max(centroids_y) - min(centroids_y) <= max(2.5, args.belt_spread),
          f"{[round(c, 2) for c in centroids_y]} spread<={max(2.5, args.belt_spread)}")
    sharps = []
    for f, b in zip(frames, boxes):
        bw, bh = b[2] - b[0], b[3] - b[1]
        g = f[b[1]:b[1] + int(bh * 0.22), b[0] + int(bw * 0.25):b[0] + int(bw * 0.75), :3].mean(axis=2)
        lap = -4 * g + np.roll(g, 1, 0) + np.roll(g, -1, 0) + np.roll(g, 1, 1) + np.roll(g, -1, 1)
        sharps.append(float(lap.var()))
    med = float(np.median(sharps))
    check("face sharpness", min(sharps) >= args.face_ratio * med,
          f"{[round(s) for s in sharps]} min>={args.face_ratio}x median {med:.0f}")

    adj = []
    for i in range(n):
        a, b = frames[i], frames[(i + 1) % n]
        adj.append(int((np.abs(a - b).max(axis=2) > 0).sum()))
    check("adj-diff coherence", float(np.mean(adj)) <= args.adj_max, f"{adj} mean {np.mean(adj):.0f} <= {args.adj_max}")
    print("OVERALL:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
