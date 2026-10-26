"""
Assignment 2 training pipeline: varroa mites on honeybees.

    uv run python main.py train

Start to finish, no manual steps:
  1. download the Roboflow dataset (skipped if already on disk)
  2. describe it: images and instances per class and split, object sizes
  3. finetune yolo11n from the COCO checkpoint, copy best weights to weights/
  4. evaluate on the TEST split: finetuned model vs COCO zero-shot vs predict-nothing
  5. score each test image (max varroa confidence) for the business decision
  6. sweep the threshold, pick the cheapest one, pick the review band
  7. save everything app.py needs into artifacts/

Environment overrides (handy for a quick dry run, not needed normally):
  A2_EPOCHS, A2_DATA_DIR (use a local YOLO dataset, skip download), A2_DEVICE,
  A2_SKIP_TRAIN=1 (re-run evaluation and costing on the existing weights/best.pt)
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import config as cfg
import decision as dec

ART = cfg.ARTIFACTS
SPLITS = {"train": "train", "val": "valid", "test": "test"}
IOUV = np.linspace(0.5, 0.95, 10)


# --------------------------------------------------------------------------- utils
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pick_device() -> str:
    if os.environ.get("A2_DEVICE"):
        return os.environ["A2_DEVICE"]
    import torch
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def save_json(obj, name: str) -> None:
    (ART / name).write_text(json.dumps(obj, indent=2, default=float))


# --------------------------------------------------------------------------- 1. data
def get_dataset() -> Path:
    data_dir = Path(os.environ.get("A2_DATA_DIR", cfg.DATA_DIR)).resolve()
    if not (data_dir / "data.yaml").exists():
        from roboflow import Roboflow
        from api_key import load_api_key
        log(f"Downloading {cfg.RF_WORKSPACE}/{cfg.RF_PROJECT} v{cfg.RF_VERSION} ...")
        rf = Roboflow(api_key=load_api_key())
        (rf.workspace(cfg.RF_WORKSPACE).project(cfg.RF_PROJECT)
           .version(cfg.RF_VERSION).download(cfg.RF_FORMAT, location=str(data_dir)))
    else:
        log(f"Dataset already at {data_dir}")

    names = yaml.safe_load((data_dir / "data.yaml").read_text())["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    # Write our own yaml with absolute paths; Roboflow's relative ones are fragile.
    local = {"path": str(data_dir), "train": "train/images", "val": "valid/images",
             "test": "test/images", "names": {i: n for i, n in enumerate(names)}}
    local_yaml = data_dir / "data_local.yaml"
    local_yaml.write_text(yaml.safe_dump(local, sort_keys=False))
    return local_yaml


def image_files(folder: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)


def read_labels(img: Path) -> np.ndarray:
    """YOLO labels as (n, 5) [cls, cx, cy, w, h], normalised."""
    lbl = img.parent.parent / "labels" / (img.stem + ".txt")
    if not lbl.exists() or lbl.stat().st_size == 0:
        return np.zeros((0, 5))
    rows = [list(map(float, l.split()[:5])) for l in lbl.read_text().splitlines() if l.strip()]
    return np.array(rows).reshape(-1, 5)


# --------------------------------------------------------------------------- 2. describe
def describe(data_yaml: Path, names: list[str]) -> dict:
    root = Path(yaml.safe_load(data_yaml.read_text())["path"])
    pos_id = names.index(cfg.POSITIVE_CLASS)
    stats, size_rows = {"classes": names, "splits": {}}, []
    for split, folder in SPLITS.items():
        imgs = image_files(root / folder / "images")
        inst = {n: 0 for n in names}
        imgs_with = {n: 0 for n in names}
        empty = 0
        for img in imgs:
            lab = read_labels(img)
            if len(lab) == 0:
                empty += 1
            for c in range(len(names)):
                k = int((lab[:, 0] == c).sum())
                inst[names[c]] += k
                imgs_with[names[c]] += int(k > 0)
            for row in lab:
                size_rows.append({"split": split, "class": names[int(row[0])],
                                  "rel_area": row[3] * row[4], "rel_w": row[3], "rel_h": row[4]})
        stats["splits"][split] = {
            "images": len(imgs), "instances": inst, "images_with_class": imgs_with,
            "empty_images": empty,
            "positive_image_share": imgs_with[names[pos_id]] / max(len(imgs), 1),
        }
    sizes = pd.DataFrame(size_rows)
    size_summary = {}
    for n in names:
        s = sizes[sizes["class"] == n]
        if len(s):
            size_summary[n] = {
                "median_rel_area_pct": 100 * float(s.rel_area.median()),
                "p10_rel_area_pct": 100 * float(s.rel_area.quantile(0.1)),
                "p90_rel_area_pct": 100 * float(s.rel_area.quantile(0.9)),
                "median_side_px_at_640": 640 * float(np.sqrt(s.rel_area.median())),
                "share_small_lt_32px_at_640": float(((s.rel_w * 640 < 32) & (s.rel_h * 640 < 32)).mean()),
            }
    stats["object_size"] = size_summary
    save_json(stats, "dataset_stats.json")
    _plot_sizes(sizes, names)
    log(f"Dataset: {json.dumps({k: v['images'] for k, v in stats['splits'].items()})}")
    return stats


def _plot_sizes(sizes: pd.DataFrame, names: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for n in names:
        s = sizes[sizes["class"] == n].rel_area * 100
        if len(s):
            ax.hist(np.log10(s.clip(lower=1e-4)), bins=50, alpha=0.6, label=n)
    ax.set_xlabel("log10(box area as % of image)")
    ax.set_ylabel("instances")
    ax.set_title("Object size relative to image")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ART / "object_sizes.png", dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- 3. train
def finetune(data_yaml: Path, device: str) -> Path:
    from ultralytics import YOLO
    epochs = int(os.environ.get("A2_EPOCHS", cfg.EPOCHS))
    log(f"Finetuning {cfg.BASE_MODEL} for up to {epochs} epochs on {device} ...")
    model = YOLO(cfg.BASE_MODEL)
    res = model.train(data=str(data_yaml), epochs=epochs, imgsz=cfg.IMGSZ, batch=cfg.BATCH,
                      patience=cfg.PATIENCE, seed=cfg.SEED, device=device,
                      project=str(cfg.ROOT / "runs"), name="finetune", exist_ok=True,
                      plots=True, verbose=False)
    best = Path(res.save_dir) / "weights" / "best.pt"
    cfg.WEIGHTS_PATH.parent.mkdir(exist_ok=True)
    shutil.copy(best, cfg.WEIGHTS_PATH)
    for f in ["results.csv", "results.png"]:
        if (Path(res.save_dir) / f).exists():
            shutil.copy(Path(res.save_dir) / f, ART / f"train_{f}")
    save_json({"base_model": cfg.BASE_MODEL, "epochs_max": epochs, "imgsz": cfg.IMGSZ,
               "batch": cfg.BATCH, "patience": cfg.PATIENCE, "seed": cfg.SEED,
               "device": device, "dataset": f"{cfg.RF_WORKSPACE}/{cfg.RF_PROJECT}/{cfg.RF_VERSION}"},
              "train_settings.json")
    log(f"Best weights copied to {cfg.WEIGHTS_PATH}")
    return cfg.WEIGHTS_PATH


# --------------------------------------------------------------------------- 4. evaluate
def xywhn_to_xyxy(b: np.ndarray) -> np.ndarray:
    cx, cy, w, h = b.T
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def match(pred_xyxy, pred_cls, pred_conf, gt_xyxy, gt_cls) -> np.ndarray:
    """(n_pred, 10) bool: is each prediction a TP at each IoU threshold (greedy by conf)."""
    tp = np.zeros((len(pred_cls), len(IOUV)), bool)
    if len(pred_cls) == 0 or len(gt_cls) == 0:
        return tp
    iou = box_iou(pred_xyxy, gt_xyxy)
    iou[pred_cls[:, None] != gt_cls[None, :]] = 0
    order = np.argsort(-pred_conf)
    for k, thr in enumerate(IOUV):
        used = np.zeros(len(gt_cls), bool)
        for i in order:
            cand = np.where((iou[i] >= thr) & ~used)[0]
            if len(cand):
                j = cand[np.argmax(iou[i, cand])]
                used[j] = True
                tp[i, k] = True
    return tp


def predict_test(model, imgs: list[Path], device: str, class_map=None):
    """Run a model over the test images, one image at a time.

    One at a time on purpose: Ultralytics gives NMS a time budget per batch, and on
    a laptop a 32-image batch at conf=0.001 blows it, silently dropping the boxes of
    the remaining images. That would show up as missed mites that the model did see.
    class_map maps model class id -> our class id.
    """
    out = []
    for p in imgs:
        r = model.predict(str(p), conf=0.001, iou=0.7, imgsz=cfg.IMGSZ, max_det=300,
                          device=device, verbose=False)[0]
        b = r.boxes
        xyxy = b.xyxyn.cpu().numpy() if len(b) else np.zeros((0, 4))
        cls = b.cls.cpu().numpy().astype(int) if len(b) else np.zeros(0, int)
        conf = b.conf.cpu().numpy() if len(b) else np.zeros(0)
        if class_map is not None:
            mapped = np.array([class_map.get(int(c), -1) for c in cls], int)
            keep = mapped >= 0
            xyxy, cls, conf = xyxy[keep], mapped[keep], conf[keep]
        out.append((xyxy, cls, conf))
    return out


def confusion(preds, gts, names, op_conf: float = 0.25) -> list[list[int]]:
    """Box-level confusion matrix at the operating confidence, IoU >= 0.5.

    Rows = true class (+ background), columns = predicted class (+ background).
    A prediction overlapping a box of another class counts as a class confusion;
    an unmatched prediction is a false alarm on background; an unmatched true box
    is a miss (predicted background).
    """
    nc = len(names)
    m = np.zeros((nc + 1, nc + 1), int)
    for (xyxy, cls, conf), (gxyxy, gcls) in zip(preds, gts):
        keep = conf >= op_conf
        xyxy, cls, conf = xyxy[keep], cls[keep], conf[keep]
        iou = box_iou(xyxy, gxyxy)
        used_p, used_g = np.zeros(len(cls), bool), np.zeros(len(gcls), bool)
        if iou.size:
            pairs = np.argwhere(iou >= 0.5)
            pairs = pairs[np.argsort(-iou[pairs[:, 0], pairs[:, 1]])]
            for i, j in pairs:
                if not used_p[i] and not used_g[j]:
                    used_p[i] = used_g[j] = True
                    m[gcls[j], cls[i]] += 1
        for i in np.where(~used_p)[0]:
            m[nc, cls[i]] += 1
        for j in np.where(~used_g)[0]:
            m[gcls[j], nc] += 1
    return m.tolist()


def detection_metrics(preds, gts, names, op_conf: float = 0.25) -> dict:
    """Per-class AP50 / AP50-95, and precision/recall at a fixed operating confidence."""
    from ultralytics.utils.metrics import ap_per_class
    tps, confs, pcls, tcls = [], [], [], []
    for (xyxy, cls, conf), (gxyxy, gcls) in zip(preds, gts):
        tps.append(match(xyxy, cls, conf, gxyxy, gcls))
        confs.append(conf); pcls.append(cls); tcls.append(gcls)
    tp = np.concatenate(tps)
    conf = np.concatenate(confs); pc = np.concatenate(pcls); tc = np.concatenate(tcls)

    per = {n: {"instances": int((tc == i).sum()), "AP50": 0.0, "AP50_95": 0.0}
           for i, n in enumerate(names)}
    if len(conf):
        res = ap_per_class(tp, conf, pc, tc, names={i: n for i, n in enumerate(names)})
        ap, classes = res[5], res[6]
        for k, c in enumerate(classes):
            per[names[c]]["AP50"] = float(ap[k, 0])
            per[names[c]]["AP50_95"] = float(ap[k].mean())
    # P / R at the operating confidence (matched at IoU 0.5)
    for i, n in enumerate(names):
        sel = (pc == i) & (conf >= op_conf)
        tp_i = int(tp[sel, 0].sum())
        n_pred, n_gt = int(sel.sum()), int((tc == i).sum())
        per[n].update({"precision": tp_i / n_pred if n_pred else 0.0,
                       "recall": tp_i / n_gt if n_gt else 0.0,
                       "TP": tp_i, "FP": n_pred - tp_i, "FN": n_gt - tp_i})
    overall = {"mAP50": float(np.mean([per[n]["AP50"] for n in names])),
               "mAP50_95": float(np.mean([per[n]["AP50_95"] for n in names]))}
    return {"overall": overall, "per_class": per, "op_conf": op_conf}


def evaluate(data_yaml: Path, names: list[str], device: str) -> pd.DataFrame:
    from ultralytics import YOLO
    root = Path(yaml.safe_load(data_yaml.read_text())["path"])
    imgs = image_files(root / "test" / "images")
    gts = []
    for p in imgs:
        lab = read_labels(p)
        gts.append((xywhn_to_xyxy(lab[:, 1:5]) if len(lab) else np.zeros((0, 4)),
                    lab[:, 0].astype(int)))

    results = {}
    log("Evaluating finetuned model on test ...")
    ft = YOLO(str(cfg.WEIGHTS_PATH))
    ft_preds = predict_test(ft, imgs, device)
    results["Finetuned yolo11n"] = detection_metrics(ft_preds, gts, names)

    # Ultralytics' own test-split validation: its confusion matrix, and a cross-check.
    val = ft.val(data=str(data_yaml), split="test", imgsz=cfg.IMGSZ, batch=cfg.BATCH,
                 device=device, project=str(cfg.ROOT / "runs"), name="test_eval",
                 exist_ok=True, plots=True, verbose=False)
    save_json({"labels": names + ["background"],
               "matrix": confusion(ft_preds, gts, names, 0.25),
               "note": "rows = true, cols = predicted; boxes at conf >= 0.25, IoU >= 0.5",
               "ultralytics_test_mAP50": float(val.box.map50),
               "ultralytics_test_mAP50_95": float(val.box.map)},
              "confusion_matrix.json")

    # COCO zero-shot. COCO has no bee or mite class, so we are generous: every COCO
    # detection counts as "bee" (class-agnostic), and it can never say "varroa".
    log("Evaluating COCO zero-shot baseline ...")
    coco = YOLO(cfg.BASE_MODEL)
    bee_id = names.index("bee") if "bee" in names else 0
    coco_preds = predict_test(coco, imgs, device, class_map={c: bee_id for c in coco.names})
    results["COCO yolo11n zero-shot"] = detection_metrics(coco_preds, gts, names)

    # Trivial: predict nothing.
    empty = [(np.zeros((0, 4)), np.zeros(0, int), np.zeros(0)) for _ in imgs]
    results["Trivial (predict nothing)"] = detection_metrics(empty, gts, names)
    save_json(results, "detection_results.json")

    rows = [{"model": m, "class": c, **v}
            for m, r in results.items() for c, v in r["per_class"].items()]
    pd.DataFrame(rows).to_csv(ART / "per_class.csv", index=False)

    # Image-level scores: the unit of the business decision.
    pos_id = names.index(cfg.POSITIVE_CLASS)
    img_rows = []
    for p, (xyxy, cls, conf), (_, gcls) in zip(imgs, ft_preds, gts):
        v = conf[cls == pos_id]
        img_rows.append({
            "image": p.name,
            "has_varroa": int((gcls == pos_id).sum() > 0),
            "n_varroa_true": int((gcls == pos_id).sum()),
            "n_bee_true": int((gcls != pos_id).sum()),
            "score": float(v.max()) if len(v) else 0.0,
            "n_varroa_pred_025": int(((cls == pos_id) & (conf >= 0.25)).sum()),
        })
    scores = pd.DataFrame(img_rows)
    scores.to_csv(ART / "test_image_scores.csv", index=False)
    log(f"Test images: {len(scores)}, with varroa: {int(scores.has_varroa.sum())}")
    return scores


# --------------------------------------------------------------------------- 6. costing
def costing(scores: pd.DataFrame) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = dec.Costs.from_config(cfg)
    y, s = scores.has_varroa.values, scores.score.values

    sweep = pd.DataFrame(dec.sweep_single(s, y, c, cfg.THRESHOLD_STEP))
    sweep.to_csv(ART / "cost_sweep.csv", index=False)
    i = int(sweep.cost_per_1000.values.argmin())
    t_star = float(sweep.threshold[i])

    lo, hi, band = dec.best_band(s, y, c, cfg.THRESHOLD_STEP)

    # Sensitivity: how the best single threshold moves with the two shakiest assumptions.
    sens = []
    for fn in [40, 75, 150, 300, 600]:
        for pi in [0.02, 0.05, 0.10, 0.20, 0.40]:
            cc = dec.Costs(fn, c.fp, c.tp, c.tn, c.review, pi)
            sw = pd.DataFrame(dec.sweep_single(s, y, cc, cfg.THRESHOLD_STEP))
            k = int(sw.cost_per_1000.values.argmin())
            alt_cc = dec.alternatives(cc)
            best_alt = min(alt_cc, key=alt_cc.get)
            sens.append({"cost_fn": fn, "base_rate": pi, "best_threshold": float(sw.threshold[k]),
                         "model_cost_per_1000": float(sw.cost_per_1000[k]),
                         "best_no_model_policy": best_alt,
                         "best_no_model_cost_per_1000": float(alt_cc[best_alt])})
    pd.DataFrame(sens).to_csv(ART / "sensitivity.csv", index=False)

    alt = dec.alternatives(c)
    policies = {**alt,
                f"Model alone (t={t_star:.2f})": float(sweep.cost_per_1000[i]),
                f"Model + review ({lo:.2f}-{hi:.2f})": band["cost_per_1000"]}
    op = {"t_star": t_star, "single": sweep.iloc[i].to_dict(),
          "band_low": lo, "band_high": hi, "band": band,
          "staffing": dec.staffing(band["review_share"], cfg),
          "policies_cost_per_1000": policies,
          "assumptions": {k: getattr(cfg, k) for k in
                          ["COST_FN", "COST_FP", "COST_TP", "COST_TN", "COST_REVIEW", "BASE_RATE",
                           "N_HIVES", "CHECKS_PER_HIVE_PER_WEEK", "WORKDAYS_PER_WEEK",
                           "REVIEW_MINUTES", "PRODUCTIVE_HOURS_PER_DAY"]}}
    save_json(op, "operating_point.json")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sweep.threshold, sweep.cost_per_1000, label="model alone")
    ax.axvline(t_star, ls="--", c="k")
    ax.scatter([t_star], [sweep.cost_per_1000[i]], c="k", zorder=3,
               label=f"chosen t={t_star:.2f}  (€{sweep.cost_per_1000[i]:,.0f} per 1,000)")
    for name, v in alt.items():
        ax.axhline(v, ls=":", lw=1, c="grey")
        ax.text(1.0, v, name, ha="right", va="bottom", fontsize=7)
    ax.set_xlabel("varroa confidence threshold")
    ax.set_ylabel("expected cost per 1,000 hive checks (€)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ART / "cost_curve.png", dpi=120)
    plt.close(fig)
    log(f"Cheapest single threshold {t_star:.2f}; review band {lo:.2f}-{hi:.2f} "
        f"sends {100 * band['review_share']:.1f}% to a person")
    return op


# --------------------------------------------------------------------------- main
def main() -> None:
    ART.mkdir(exist_ok=True)
    device = pick_device()
    data_yaml = get_dataset()
    names = list(yaml.safe_load(data_yaml.read_text())["names"].values())
    if cfg.POSITIVE_CLASS not in names:
        raise SystemExit(f"Positive class {cfg.POSITIVE_CLASS!r} not in dataset classes {names}")
    describe(data_yaml, names)
    if os.environ.get("A2_SKIP_TRAIN") and cfg.WEIGHTS_PATH.exists():
        log(f"A2_SKIP_TRAIN set: reusing {cfg.WEIGHTS_PATH} (no retraining)")
    else:
        finetune(data_yaml, device)
    scores = evaluate(data_yaml, names, device)
    costing(scores)
    log("Done. Commit weights/best.pt and artifacts/.")


if __name__ == "__main__":
    main()
