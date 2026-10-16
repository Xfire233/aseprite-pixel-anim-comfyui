#!/usr/bin/env python
"""通用像素角色动画组装管线 (configurable canvas):
自适应背景抠图 (RGBA alpha / 边框漫水填充 + 内部色彩扩边消色溢 + Color-Line Alpha Unmixing)
+ 全序列包围盒对齐裁剪 (BOX 面积平均下采样) + 前景专属共享调色板 (MEDIANCUT, 无抖动)
+ 时序 Alpha 共识 + Aseprite CLI 自动组装 (.aseprite + 4x GIF + Sprite Sheet + 2x 全宽检视表).

Usage:
  uv run --with pillow --with numpy python scripts/pixelize.py --frames f1.png f2.png f3.png --pingpong \
      --canvas 128x224 --out work/idle --assemble output/idle.aseprite --tag idle_loop --zoom work/zoom.png

Canvas is per-project choice (portrait full-body default 128x224; legacy landscape 160x112).
GIF is exported at --gif-scale (default 4) nearest-neighbor; sheet stays 1x.
"""
import argparse, glob, os, subprocess, sys
from collections import deque

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ASEPRITE as ASE, PROJECT_ROOT as ROOT  # noqa: E402


def load_rgba(path, bg_tol=18):
    im = Image.open(path) if isinstance(path, str) else path
    has_alpha_mask = False
    alpha_mask_in = None
    if im.mode in ("RGBA", "LA") or ("transparency" in im.info):
        rgba_in = np.array(im.convert("RGBA"))
        if (rgba_in[:, :, 3] < 16).sum() > (rgba_in.shape[0] * rgba_in.shape[1] * 0.08):
            has_alpha_mask = True
            alpha_mask_in = rgba_in[:, :, 3]

    a = np.array(im.convert("RGB")).astype(np.float32)
    H, Wd, _ = a.shape
    border_px = np.concatenate([a[0, :, :], a[H - 1, :, :], a[:, 0, :], a[:, Wd - 1, :]], axis=0)
    bg_color = np.median(border_px, axis=0)
    border_std = float(np.std(border_px, axis=0).max())
    tol = float(max(bg_tol, min(42.0, border_std * 2.0 + 14.0)))

    dist_bg = np.abs(a - bg_color).max(axis=2)
    l2_bg = np.sqrt(((a - bg_color[None, None, :]) ** 2).sum(axis=2))
    bg_is_white = (bg_color.min() >= 235) and ((bg_color.max() - bg_color.min()) <= 15)
    neighbors_8 = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))

    if has_alpha_mask:
        fg_coarse = alpha_mask_in >= 112
    else:
        m_seed = (dist_bg <= 14.0) if bg_is_white else ((dist_bg <= tol) | (l2_bg <= tol * 1.4))
        seen = np.zeros((H, Wd), bool)
        dq = deque()
        for x in range(Wd):
            for y in (0, H - 1):
                if m_seed[y, x] and not seen[y, x]:
                    seen[y, x] = True; dq.append((y, x))
        for y in range(H):
            for x in (0, Wd - 1):
                if m_seed[y, x] and not seen[y, x]:
                    seen[y, x] = True; dq.append((y, x))
        while dq:
            y, x = dq.popleft()
            for dy, dx in neighbors_8:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < Wd and m_seed[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True; dq.append((ny, nx))

        # Open enclosed background pockets inside hair/limbs (keep white clothing interior highlights intact)
        unreached_bg = m_seed & (~seen)
        seen_hole = np.zeros((H, Wd), bool)
        for y in range(H):
            for x in range(Wd):
                if unreached_bg[y, x] and not seen_hole[y, x]:
                    comp = [(y, x)]
                    seen_hole[y, x] = True
                    dq = deque([(y, x)])
                    border_vals = []
                    while dq:
                        cy, cx = dq.popleft()
                        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < H and 0 <= nx < Wd:
                                if unreached_bg[ny, nx] and not seen_hole[ny, nx]:
                                    seen_hole[ny, nx] = True; dq.append((ny, nx)); comp.append((ny, nx))
                                elif not m_seed[ny, nx]:
                                    border_vals.append(a[ny, nx])
                    if not bg_is_white and len(comp) >= 4:
                        for cy, cx in comp:
                            seen[cy, cx] = True
                    elif bg_is_white and len(comp) >= 14 and border_vals:
                        b_arr = np.array(border_vals)
                        if (b_arr.min(axis=1) > 210).mean() < 0.25:
                            for cy, cx in comp:
                                seen[cy, cx] = True

        fg_coarse = ~seen
    # Erode foreground mask to isolate pure interior colors unpolluted by background anti-aliasing
    er3 = fg_coarse.copy()
    for _ in range(3):
        er3 = er3 & np.roll(er3, 1, 0) & np.roll(er3, -1, 0) & np.roll(er3, 1, 1) & np.roll(er3, -1, 1)
        er3[0, :] = False; er3[-1, :] = False; er3[:, 0] = False; er3[:, -1] = False
    er6 = er3.copy()
    for _ in range(3):
        er6 = er6 & np.roll(er6, 1, 0) & np.roll(er6, -1, 0) & np.roll(er6, 1, 1) & np.roll(er6, -1, 1)
        er6[0, :] = False; er6[-1, :] = False; er6[:, 0] = False; er6[:, -1] = False

    chroma = a.max(axis=2) - a.min(axis=2)
    is_pale_halo = (a.min(axis=2) > 198) & (chroma < 24)
    is_bg_tint = l2_bg < max(55.0, tol * 2.1)
    if has_alpha_mask:
        interior = er6 & (alpha_mask_in >= 242)
    else:
        interior = er6 | (er3 & (~is_pale_halo) & (~is_bg_tint))
    ref_fg = a.copy()
    valid = interior.copy()
    if valid.sum() > 20:
        for _ in range(18):
            unfilled = ~valid
            if not unfilled.any():
                break
            acc = np.zeros((H, Wd, 3), dtype=np.float32)
            cnt = np.zeros((H, Wd), dtype=np.float32)
            for dy, dx in neighbors_8:
                v_s = np.roll(np.roll(valid, dy, axis=0), dx, axis=1)
                c_s = np.roll(np.roll(ref_fg, dy, axis=0), dx, axis=1)
                m_t = unfilled & v_s
                acc[m_t] += c_s[m_t]
                cnt[m_t] += 1.0
            new_m = unfilled & (cnt > 0)
            if not new_m.any():
                break
            ref_fg[new_m] = acc[new_m] / cnt[new_m, None]
            valid |= new_m

    # Color-Line Alpha Unmixing in the boundary band (fg_coarse & ~interior)
    band = fg_coarse & (~interior)
    diff_cb = a - bg_color[None, None, :]
    diff_fb = ref_fg - bg_color[None, None, :]
    denom = (diff_fb ** 2).sum(axis=2) + 1e-4
    est_alpha = np.clip((diff_cb * diff_fb).sum(axis=2) / denom, 0.0, 1.0)
    fb_norm = np.sqrt(denom)

    white_vec = np.array([255.0, 255.0, 255.0], dtype=np.float32)
    diff_cw = a - white_vec[None, None, :]
    diff_fw = ref_fg - white_vec[None, None, :]
    denom_w = (diff_fw ** 2).sum(axis=2) + 1e-4
    est_alpha_w = np.clip((diff_cw * diff_fw).sum(axis=2) / denom_w, 0.0, 1.0)
    fw_norm = np.sqrt(denom_w)

    if has_alpha_mask:
        final_fg = alpha_mask_in >= 185
    else:
        alpha_thresh = 0.54 if bg_is_white else 0.62
        keep_band = np.where(fb_norm > 25.0, est_alpha >= alpha_thresh, dist_bg > 12.0)
        keep_band = keep_band & np.where(fw_norm > 48.0, est_alpha_w >= 0.52, True)
        final_fg = interior | (band & keep_band)
    clean_rgb = a.copy()
    clean_rgb[~interior] = ref_fg[~interior]
    alpha = np.where(final_fg, 255, 0).astype(np.uint8)

    # Drop tiny disconnected border/corner ringing islands (e.g. 2-5 px Wan 3D-VAE corner artifacts)
    fg = alpha > 0
    min_island = max(16, (H * Wd) // 3000)
    seen_fg = np.zeros((H, Wd), bool)
    for y in range(H):
        for x in range(Wd):
            if fg[y, x] and not seen_fg[y, x]:
                comp = [(y, x)]
                seen_fg[y, x] = True
                dq = deque([(y, x)])
                while dq:
                    cy, cx = dq.popleft()
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < H and 0 <= nx < Wd and fg[ny, nx] and not seen_fg[ny, nx]:
                            seen_fg[ny, nx] = True
                            dq.append((ny, nx))
                            comp.append((ny, nx))
                if len(comp) < min_island:
                    for cy, cx in comp:
                        alpha[cy, cx] = 0
    rgba = np.dstack([clean_rgb.round().astype(np.uint8), alpha])
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0:
        return rgba, (0, 0, Wd, H)
    return rgba, (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", nargs="+", required=True)
    ap.add_argument("--pingpong", action="store_true")
    ap.add_argument("--canvas", default="128x224")
    ap.add_argument("--out", required=True)
    ap.add_argument("--assemble")
    ap.add_argument("--tag", default="loop")
    ap.add_argument("--durations", default=None)
    ap.add_argument("--zoom")
    ap.add_argument("--gif-scale", type=int, default=4)
    ap.add_argument("--consensus", choices=["alpha", "full", "none"], default="alpha")
    ap.add_argument("--colors", type=int, default=32)
    ap.add_argument("--subsample", type=int, default=0,
                    help="evenly subsample N keyframes from input frames before pingpong/pixelize (0 = use all)")
    ap.add_argument("--no-register", action="store_true", default=True,
                    help="skip silhouette IoU registration (default True for fixed-camera Wan I2V clips)")
    # Legacy no-op flags retained only for CLI backwards compatibility
    ap.add_argument("--band", default="off", help=argparse.SUPPRESS)
    ap.add_argument("--no-sandal", action="store_true", default=True, help=argparse.SUPPRESS)
    ap.add_argument("--stocking-tint", action="store_true", default=False, help=argparse.SUPPRESS)
    args = ap.parse_args()
    W, H = map(int, args.canvas.split("x"))

    os.makedirs(args.out, exist_ok=True)
    frame_paths = []
    for p in args.frames:
        frame_paths += sorted(glob.glob(p)) if "*" in p else [p]
    if args.subsample and 0 < args.subsample < len(frame_paths):
        idxs_sel = np.linspace(0, len(frame_paths) - 1, args.subsample).round().astype(int).tolist()
        frame_paths = [frame_paths[i] for i in idxs_sel]
        print(f"subsampled {len(frame_paths)} frames (indices {idxs_sel}): {[os.path.basename(p) for p in frame_paths]}")
    uniq, boxes = [], []
    for p in frame_paths:
        r, b = load_rgba(p)
        uniq.append(r); boxes.append(b)

    # --- silhouette registration vs reference frame (IoU search at canvas size) ---
    ref_i = len(uniq) // 2

    if not args.no_register:
        def small_alpha(r):
            return np.array(Image.fromarray(r[:, :, 3]).resize((W, H), Image.BILINEAR)) > 64

        ref_small = small_alpha(uniq[ref_i])
        warped = []
        for i, r in enumerate(uniq):
            if i == ref_i:
                warped.append(r)
                print(f"frame{i+1}: reference")
                continue
            src = Image.fromarray(r[:, :, 3])
            best = (-1, 1.0, 0, 0)
            for s in [1 + d / 100 for d in range(-4, 5)]:
                sm = np.array(src.resize((round(W * s), round(H * s)), Image.BILINEAR)) > 64
                base_x, base_y = (W - sm.shape[1]) // 2, (H - sm.shape[0]) // 2
                for dy in range(-4, 5):
                    for dx in range(-4, 5):
                        canvas = np.zeros((H, W), bool)
                        x0s, y0s = base_x + dx, base_y + dy
                        xs0, xs1 = max(0, x0s), min(W, x0s + sm.shape[1])
                        ys0, ys1 = max(0, y0s), min(H, y0s + sm.shape[0])
                        if xs1 <= xs0 or ys1 <= ys0:
                            continue
                        canvas[ys0:ys1, xs0:xs1] = sm[ys0 - y0s:ys1 - y0s, xs0 - x0s:xs1 - x0s]
                        iou = (canvas & ref_small).sum() / max(1, (canvas | ref_small).sum())
                        if iou > best[0]:
                            best = (iou, s, dx, dy)
            _, s, dx, dy = best
            H0, W0 = r.shape[:2]
            rim = Image.fromarray(r, "RGBA").resize((round(W0 * s), round(H0 * s)), Image.BILINEAR)
            canvas = Image.new("RGBA", (W0, H0), (0, 0, 0, 0))
            canvas.alpha_composite(rim, (round((W0 - rim.width) / 2 + dx * W0 / W),
                                          round((H0 - rim.height) / 2 + dy * H0 / H)))
            warped.append(np.array(canvas))
            print(f"frame{i+1}: register scale={s:.2f} dx={dx} dy={dy} IoU={best[0]:.3f}")
        uniq = warped

    # --- union crop -> canvas (BOX downscale = area average, least speckle) ---
    boxes = []
    for r in uniq:
        ys, xs = np.where(r[:, :, 3] > 0)
        boxes.append((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes); y1 = max(b[3] for b in boxes)
    cw, ch = x1 - x0, y1 - y0
    scale = min(W / cw, (H - 1) / ch)
    nw, nh = round(cw * scale), round(ch * scale)
    ox, oy = (W - nw) // 2, (H - 1) - nh
    print(f"union crop {cw}x{ch} -> {nw}x{nh} on {W}x{H} boxes={boxes}")
    canv_rgb, canv_a = [], []
    for r in uniq:
        rgb_s = Image.fromarray(r[y0:y1, x0:x1, :3], "RGB").resize((nw, nh), Image.BOX)
        a_s = Image.fromarray(r[y0:y1, x0:x1, 3], "L").resize((nw, nh), Image.BOX)
        c_rgb = Image.new("RGB", (W, H), (0, 0, 0))
        c_rgb.paste(rgb_s, (ox, oy))
        c_a = Image.new("L", (W, H), 0)
        c_a.paste(a_s, (ox, oy))
        canv_rgb.append(c_rgb)
        canv_a.append(c_a)
    seq_rgb = canv_rgb + canv_rgb[-2:0:-1] if args.pingpong and len(canv_rgb) >= 3 else canv_rgb
    seq_a = canv_a + canv_a[-2:0:-1] if args.pingpong and len(canv_a) >= 3 else canv_a
    n = len(seq_rgb)

    # --- shared palette (sampled from opaque foreground pixels only so background (0,0,0) never skews MEDIANCUT) ---
    alphas = np.stack([np.array(ca) >= 128 for ca in seq_a])
    fg_samples = np.concatenate([np.array(cr)[al] for cr, al in zip(seq_rgb, alphas) if al.any()], axis=0)
    if len(fg_samples) > args.colors:
        side = int(np.ceil(np.sqrt(len(fg_samples))))
        pad_px = np.resize(fg_samples, (side * side, 3)).reshape(side, side, 3)
        pal = Image.fromarray(pad_px, "RGB").quantize(colors=args.colors, method=Image.MEDIANCUT)
    else:
        mosaic = Image.new("RGB", (W, H * n))
        for i, cr in enumerate(seq_rgb):
            mosaic.paste(cr, (0, H * i))
        pal = mosaic.quantize(colors=args.colors, method=Image.MEDIANCUT)
    palette = np.array(pal.getpalette()[:args.colors * 3]).reshape(args.colors, 3).astype(np.int32)
    idxs = np.stack([np.array(cr.quantize(palette=pal, dither=Image.NONE)) for cr in seq_rgb])
    idx2, al2 = idxs.copy(), alphas.copy()

    # --- motion-safe temporal consensus ---
    oc = alphas.sum(axis=0)
    need = 99 if args.consensus == "none" else max(2, n - 1)
    for y in range(H):
        for x in range(W):
            o = oc[y, x]
            if args.consensus != "none" and o == 1 and n > 2:
                # Only drop single-frame pixels if they are spatially isolated speckle (< 2 4-neighbors)
                for t_idx in range(n):
                    if al2[t_idx, y, x]:
                        nb = 0
                        if y > 0 and alphas[t_idx, y - 1, x]: nb += 1
                        if y < H - 1 and alphas[t_idx, y + 1, x]: nb += 1
                        if x > 0 and alphas[t_idx, y, x - 1]: nb += 1
                        if x < W - 1 and alphas[t_idx, y, x + 1]: nb += 1
                        if nb < 2:
                            al2[t_idx, y, x] = False
            elif o >= need:
                vals = idxs[alphas[:, y, x], y, x]
                v, cnt = np.unique(vals, return_counts=True)
                k = cnt.argmax()
                if cnt[k] >= need:
                    missing = ~alphas[:, y, x]
                    idx2[missing, y, x] = v[k]
                    if args.consensus == "full":
                        idx2[:, y, x] = v[k]
                    al2[:, y, x] = True

    # Remove tiny disconnected foreground islands (< 16 pixels) per frame so silhouette edges are 100% speck-free
    for t_idx in range(n):
        al_cur = al2[t_idx]
        visited = np.zeros_like(al_cur, dtype=bool)
        ys_a, xs_a = np.where(al_cur)
        for sy, sx in zip(ys_a, xs_a):
            if visited[sy, sx]:
                continue
            comp_pts = [(sy, sx)]
            visited[sy, sx] = True
            head = 0
            while head < len(comp_pts):
                cy, cx = comp_pts[head]
                head += 1
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < H and 0 <= nx < W and al_cur[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        comp_pts.append((ny, nx))
            if len(comp_pts) < 16:
                for py_i, px_i in comp_pts:
                    al2[t_idx, py_i, px_i] = False
    def to_rgba(idx, al):
        o = np.zeros((H, W, 4), np.uint8)
        o[:, :, :3] = palette[idx].astype(np.uint8)
        o[:, :, 3] = np.where(al, 255, 0)
        o[~al] = 0
        return o

    frames = [to_rgba(idx2[i], al2[i]) for i in range(n)]
    d = [int((np.abs(frames[i].astype(int) - frames[(i + 1) % n].astype(int)).max(axis=2) > 0).sum()) for i in range(n)]
    print(f"adj-diff px {d} mean {np.mean(d):.0f}")
    paths = []
    for i, a in enumerate(frames, 1):
        p = os.path.join(args.out, f"frame{i}.png")
        Image.fromarray(a, "RGBA").save(p)
        paths.append(p)
    if args.zoom:
        pad = 2
        ims = [Image.fromarray(a, "RGBA") for a in frames]
        s = Image.new("RGBA", (W * n + pad * (n - 1), H), (40, 40, 40, 255))
        for i, im in enumerate(ims):
            x_off = (W + pad) * i
            s.alpha_composite(im, (x_off, 0))
            if i > 0:
                for px in range(x_off - pad, x_off):
                    for py in range(H):
                        s.putpixel((px, py), (72, 72, 78, 255))
        s.resize((s.width * 2, H * 2), Image.NEAREST).save(args.zoom)
        print(f"zoom -> {args.zoom}")
    if args.assemble:
        durs = args.durations or ",".join(["200"] * n)
        base = os.path.abspath(args.assemble)
        has_ase = bool(ASE and (os.path.isfile(ASE) or shutil.which(ASE)))
        if has_ase:
            cmd = [ASE, "-b",
                   "--script-param", f"width={W}",
                   "--script-param", f"height={H}",
                   "--script-param", f"frames={','.join(os.path.abspath(p) for p in paths)}",
                   "--script-param", f"durations={durs}",
                   "--script-param", f"tag={args.tag}",
                   "--script-param", f"out={base}",
                   "--script", os.path.join(ROOT, "scripts/assemble.lua")]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            print("assemble", r.returncode, r.stdout.strip())
            stem = base[:-9] if base.endswith(".aseprite") else base
            print(f"gif{args.gif_scale}x", subprocess.run(
                [ASE, "-b", base, "--scale", str(args.gif_scale), "--save-as", stem + ".gif"],
                capture_output=True, text=True, timeout=180).returncode)
            print("sheet", subprocess.run(
                [ASE, "-b", base, "--sheet", stem + "_sheet.png"],
                capture_output=True, text=True, timeout=180).returncode)
        else:
            stem = base[:-9] if base.endswith(".aseprite") else base
            dur_list = [int(d) for d in durs.split(",") if d.isdigit()]
            dur_val = dur_list[0] if dur_list else 180
            scale = max(1, int(args.gif_scale))
            gif_frames = [im.resize((W * scale, H * scale), Image.NEAREST) for im in ims]
            if gif_frames:
                gif_frames[0].save(stem + ".gif", save_all=True, append_images=gif_frames[1:], duration=dur_val, loop=0, disposal=2)
            sheet_img = Image.new("RGBA", (W * len(ims), H), (0, 0, 0, 0))
            for idx_im, im in enumerate(ims):
                sheet_img.paste(im, (idx_im * W, 0))
            sheet_img.save(stem + "_sheet.png")
            print("[pixelize] Aseprite CLI not found; assembled .gif and _sheet.png using Pillow fallback (.aseprite file skipped).")

if __name__ == "__main__":
    main()
