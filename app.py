"""
Assignment 2 dashboard: varroa screening for beehives.

    uv run python main.py app

Loads the committed weights in weights/ and the artifacts/ written by train.py.
It never trains anything. Every number here is recomputed from the saved
test-set scores, so moving a slider is a real what-if on the real model.
"""

from __future__ import annotations

import json

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import config as cfg
import decision as dec

ART = cfg.ARTIFACTS
if not cfg.WEIGHTS_PATH.exists() or not (ART / "operating_point.json").exists():
    raise SystemExit("Missing weights/best.pt or artifacts/. Run `uv run python main.py train` first.")

from ultralytics import YOLO  # noqa: E402

MODEL = YOLO(str(cfg.WEIGHTS_PATH))
NAMES = MODEL.names
POS_ID = [k for k, v in NAMES.items() if v == cfg.POSITIVE_CLASS][0]

OP = json.loads((ART / "operating_point.json").read_text())
SCORES = pd.read_csv(ART / "test_image_scores.csv")
Y, S = SCORES.has_varroa.values, SCORES.score.values
PER_CLASS = pd.read_csv(ART / "per_class.csv")
DET = json.loads((ART / "detection_results.json").read_text())
CM = json.loads((ART / "confusion_matrix.json").read_text())
STATS = json.loads((ART / "dataset_stats.json").read_text())

DEFAULT_COSTS = dec.Costs.from_config(cfg)


# --------------------------------------------------------------------------- helpers
def decide(score: float, low: float, high: float) -> str:
    low, high = min(low, high), max(low, high)
    if score >= high:
        return "TREAT (model is confident mites are present)"
    if score < low:
        return "PASS (leave hive alone)"
    return "HUMAN REVIEW (beekeeper does a manual mite check)"


def costs_from(fn, fp, tp, tn, review, base_rate) -> dec.Costs:
    return dec.Costs(float(fn), float(fp), float(tp), float(tn), float(review), float(base_rate))


# --------------------------------------------------------------------------- tab 1: predict
def predict(image, conf, low, high):
    if image is None:
        return None, "Upload a photo of bees at a hive entrance."
    r = MODEL.predict(image, conf=float(conf), imgsz=cfg.IMGSZ, verbose=False)[0]
    plotted = r.plot()[:, :, ::-1]  # BGR -> RGB
    cls = r.boxes.cls.cpu().numpy().astype(int) if len(r.boxes) else np.zeros(0, int)
    confs = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.zeros(0)
    # The decision score is the top varroa confidence, as in training (not filtered by conf).
    full = MODEL.predict(image, conf=0.001, imgsz=cfg.IMGSZ, verbose=False)[0]
    fcls = full.boxes.cls.cpu().numpy().astype(int) if len(full.boxes) else np.zeros(0, int)
    fconf = full.boxes.conf.cpu().numpy() if len(full.boxes) else np.zeros(0)
    score = float(fconf[fcls == POS_ID].max()) if (fcls == POS_ID).any() else 0.0
    counts = ", ".join(f"{NAMES[k]}: {int((cls == k).sum())}" for k in NAMES)
    md = (f"**Detections at confidence ≥ {conf:.2f}:** {counts}\n\n"
          f"**Varroa score** (highest mite confidence): **{score:.3f}**\n\n"
          f"**Decision with band {min(low, high):.2f}–{max(low, high):.2f}:** {decide(score, low, high)}")
    return plotted, md


# --------------------------------------------------------------------------- tab 2: decision
def decision_view(low, high, fn, fp, tp, tn, review, base_rate):
    c = costs_from(fn, fp, tp, tn, review, base_rate)
    low, high = min(low, high), max(low, high)
    r = dec.three_way(S, Y, low, high)
    out = dec.cost_per_1000(r, c)
    staff = dec.staffing(out["review_share"], cfg)

    cm = pd.DataFrame(
        {"Pass": [r["pos_pass"], r["neg_pass"]],
         "Human review": [r["pos_review"], r["neg_review"]],
         "Treat": [r["pos_fail"], r["neg_fail"]]},
        index=["Actually infested", "Actually healthy"]).reset_index(names="Test set")

    pi = c.base_rate
    per1000 = pd.DataFrame(
        {"Pass": [1000 * pi * r["p_pass"], 1000 * (1 - pi) * r["n_pass"]],
         "Human review": [1000 * pi * r["p_review"], 1000 * (1 - pi) * r["n_review"]],
         "Treat": [1000 * pi * r["p_fail"], 1000 * (1 - pi) * r["n_fail"]]},
        index=["Infested", "Healthy"]).round(1).reset_index(names=f"Per 1,000 at base rate {pi:.0%}")

    single = pd.DataFrame(dec.sweep_single(S, Y, c, cfg.THRESHOLD_STEP))
    k = int(single.cost_per_1000.values.argmin())
    policies = {**dec.alternatives(c),
                f"Model alone (best t={single.threshold[k]:.2f})": single.cost_per_1000[k],
                f"Model + review (your band {low:.2f}–{high:.2f})": out["cost_per_1000"]}
    pol = pd.DataFrame({"Policy": list(policies), "€ per 1,000 checks": [round(v, 0) for v in policies.values()]})

    md = (f"### Expected cost: **€{out['cost_per_1000']:,.0f} per 1,000 hive checks**\n"
          f"- Sent to a person: **{100 * out['review_share']:.1f}%** of checks\n"
          f"- Infested hives missed: **{out['missed_per_1000']:.1f}** per 1,000; "
          f"healthy hives treated for nothing: **{out['false_treat_per_1000']:.1f}** per 1,000\n"
          f"- Workload ({cfg.N_HIVES} hives, {cfg.CHECKS_PER_HIVE_PER_WEEK} check/hive/week): "
          f"**{staff['reviews_per_day']:.1f} reviews/day** × {cfg.REVIEW_MINUTES} min = "
          f"**{staff['review_hours_per_day']:.1f} h/day ≈ {staff['people_needed']:.2f} people**")
    return md, cm, per1000, pol, cost_curve(c, low, high)


def cost_curve(c: dec.Costs, low=None, high=None):
    sw = pd.DataFrame(dec.sweep_single(S, Y, c, cfg.THRESHOLD_STEP))
    k = int(sw.cost_per_1000.values.argmin())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sw.threshold, y=sw.cost_per_1000, mode="lines", name="Model alone"))
    fig.add_trace(go.Scatter(x=[sw.threshold[k]], y=[sw.cost_per_1000[k]], mode="markers+text",
                             marker=dict(size=12, color="black"), name="Cheapest threshold",
                             text=[f"t={sw.threshold[k]:.2f}"], textposition="top center"))
    fig.add_vline(x=OP["t_star"], line_dash="dash", line_color="red",
                  annotation_text=f"chosen in REPORT t={OP['t_star']:.2f}")
    for name, v in dec.alternatives(c).items():
        fig.add_hline(y=v, line_dash="dot", line_color="grey", annotation_text=name,
                      annotation_position="top left")
    if low is not None:
        fig.add_vrect(x0=min(low, high), x1=max(low, high), fillcolor="orange", opacity=0.15,
                      line_width=0, annotation_text="review band")
    fig.update_layout(xaxis_title="Varroa confidence threshold",
                      yaxis_title="Expected € per 1,000 hive checks", height=420,
                      margin=dict(l=40, r=20, t=30, b=40), legend=dict(orientation="h"))
    return fig


# --------------------------------------------------------------------------- tab 3: performance
def per_class_table():
    t = PER_CLASS[["model", "class", "instances", "AP50", "AP50_95", "precision", "recall", "TP", "FP", "FN"]].copy()
    for col in ["AP50", "AP50_95", "precision", "recall"]:
        t[col] = t[col].round(3)
    return t


def overall_table():
    return pd.DataFrame([{"Model": m, "mAP50": round(r["overall"]["mAP50"], 3),
                          "mAP50-95": round(r["overall"]["mAP50_95"], 3)} for m, r in DET.items()])


def confusion_fig():
    m = np.array(CM["matrix"])
    labels = CM["labels"]
    fig = go.Figure(go.Heatmap(z=m, x=[f"pred {l}" for l in labels], y=[f"true {l}" for l in labels],
                               text=m.astype(int), texttemplate="%{text}", colorscale="Blues"))
    fig.update_layout(title="Box confusion matrix, test split (conf ≥ 0.25, IoU ≥ 0.5)", height=380,
                      yaxis=dict(autorange="reversed"), margin=dict(l=40, r=20, t=50, b=40))
    return fig


def dataset_table():
    rows = []
    for split, s in STATS["splits"].items():
        row = {"split": split, "images": s["images"], "empty images": s["empty_images"],
               f"share of images with {cfg.POSITIVE_CLASS}": round(s["positive_image_share"], 3)}
        for n, v in s["instances"].items():
            row[f"{n} instances"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def size_table():
    return pd.DataFrame([{"class": k, **{a: round(b, 3) for a, b in v.items()}}
                         for k, v in STATS["object_size"].items()])


# --------------------------------------------------------------------------- layout
with gr.Blocks(title="Varroa screening") as demo:
    gr.Markdown("# Varroa mite screening for beehives\n"
                "One case = one entrance photo of one hive. Positive class = **varroa**. "
                "Below the low threshold the hive is left alone, above the high one it is treated, "
                "in between a beekeeper checks by hand.")
    with gr.Row():
        low = gr.Slider(0, 1, value=OP["band_low"], step=0.01, label="Low threshold (below = pass)")
        high = gr.Slider(0, 1, value=OP["band_high"], step=0.01, label="High threshold (at or above = treat)")

    with gr.Tab("Decision & cost"):
        with gr.Accordion("Cost assumptions (€ per hive check)", open=False):
            with gr.Row():
                fn = gr.Number(cfg.COST_FN, label="Missed infestation (FN)")
                fp = gr.Number(cfg.COST_FP, label="Healthy hive treated (FP)")
                tp = gr.Number(cfg.COST_TP, label="Infested hive treated (TP)")
                tn = gr.Number(cfg.COST_TN, label="Healthy hive left alone (TN)")
                rv = gr.Number(cfg.COST_REVIEW, label="Manual review")
                br = gr.Slider(0.01, 0.6, value=cfg.BASE_RATE, step=0.01, label="Base rate (share infested)")
        summary = gr.Markdown()
        with gr.Row():
            cm_tbl = gr.Dataframe(label="Confusion matrix on the test set (counts)")
            per1000_tbl = gr.Dataframe(label="Same, per 1,000 checks at the assumed base rate")
        pol_tbl = gr.Dataframe(label="Against the alternatives")
        curve = gr.Plot(label="Cost against threshold")

    with gr.Tab("Try an image"):
        with gr.Row():
            img_in = gr.Image(type="numpy", label="Hive entrance photo")
            img_out = gr.Image(label="Predictions")
        conf = gr.Slider(0.01, 1, value=OP["t_star"], step=0.01, label="Box display confidence")
        pred_md = gr.Markdown()
        gr.Button("Run model").click(predict, [img_in, conf, low, high], [img_out, pred_md])

    with gr.Tab("Model performance"):
        gr.Markdown("Test split. Finetuned model vs COCO zero-shot (every COCO box counted as 'bee'; "
                    "it has no mite class) vs predicting nothing. P/R at confidence 0.25, IoU 0.5.")
        gr.Dataframe(overall_table(), label="Overall")
        gr.Dataframe(per_class_table(), label="Per class")
        gr.Plot(confusion_fig())

    with gr.Tab("Dataset"):
        gr.Dataframe(dataset_table(), label="Images and instances per split")
        gr.Dataframe(size_table(), label="Object size relative to the 640×640 image")
        if (ART / "object_sizes.png").exists():
            gr.Image(str(ART / "object_sizes.png"), label="Object sizes")

    inputs = [low, high, fn, fp, tp, tn, rv, br]
    outputs = [summary, cm_tbl, per1000_tbl, pol_tbl, curve]
    for comp in inputs:
        comp.change(decision_view, inputs, outputs)
    demo.load(decision_view, inputs, outputs)


if __name__ == "__main__":
    demo.launch()
