"""
Shared settings for train.py and app.py.

Everything a business decision depends on lives here, in one place, so the
training run and the dashboard can never disagree about it. Every euro figure
and rate below is an ASSUMPTION that REPORT.md has to defend. Change them here,
re-run `uv run python main.py train`, and the sweep, the chosen thresholds and
the dashboard all follow.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --- Data (Roboflow Universe, used as-is, not forked) -----------------------
RF_WORKSPACE = "bee-9phqe"
RF_PROJECT = "beehive-qyqml"
RF_VERSION = 1           # v1: 6,602 images, 4621/1321/660 split, no augmentation
RF_FORMAT = "yolov11"
DATA_DIR = ROOT / "datasets" / f"{RF_PROJECT}-v{RF_VERSION}"

POSITIVE_CLASS = "varroa"  # the thing we must not miss

# --- Model / training -------------------------------------------------------
BASE_MODEL = "yolo11n.pt"   # COCO-pretrained checkpoint, also the zero-shot baseline
EPOCHS = 15          # what the submitted weights were trained with (M1 laptop, ~3.3 h)
IMGSZ = 640
BATCH = 16
PATIENCE = 10
SEED = 0

WEIGHTS_PATH = ROOT / "weights" / "best.pt"
ARTIFACTS = ROOT / "artifacts"   # everything the app needs, committed

# --- Business assumptions (one "case" = one entrance photo of one hive) -----
# Costs in euros per case, by outcome.
COST_FN = 150.0     # infested hive left untreated until the next check
COST_FP = 20.0      # healthy hive treated: product + labour (+ resistance, stress)
COST_TP = 20.0      # infested hive treated: same treatment, but money well spent
COST_TN = 0.0       # healthy hive left alone
COST_REVIEW = 6.0   # beekeeper does a manual mite check (minutes x hourly rate)

BASE_RATE = 0.10    # share of hive checks where the hive is really infested

# Operation size, used to turn "share sent to review" into people.
N_HIVES = 400
CHECKS_PER_HIVE_PER_WEEK = 1
WORKDAYS_PER_WEEK = 5
REVIEW_MINUTES = 12
PRODUCTIVE_HOURS_PER_DAY = 6

# Threshold grid for the sweeps.
THRESHOLD_STEP = 0.01
