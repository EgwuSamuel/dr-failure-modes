import os
import glob

QUICK = os.environ.get("DRFM_QUICK", "0") == "1"
WORK = os.environ.get("DRFM_WORK", "/kaggle/working/quick" if QUICK else "/kaggle/working")
INPUT = os.environ.get("DRFM_INPUT", "/kaggle/input")
OUT = os.path.join(WORK, "outputs")
CACHE = os.environ.get("DRFM_CACHE", os.path.join(WORK, "cache"))
CKPT = os.path.join(WORK, "checkpoints")
RUNS = os.path.join(OUT, "runs")
TABLES = os.path.join(OUT, "tables")
FIGS = os.path.join(OUT, "figures")
CLAIMS = os.path.join(WORK, "claims")
LOGS = os.path.join(WORK, "logs")

SHORT = 256
IMG = 224
NUM_CLASSES = 5
REFERABLE = 2
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

ARCHS = os.environ.get("DRFM_ARCHS", "efficientnet_b0,resnet50,vit_small_patch16_224").split(",")
ARCH_LABEL = {"efficientnet_b0": "EfficientNet-B0", "resnet50": "ResNet-50", "vit_small_patch16_224": "ViT-Small"}


def label(a):
    return ARCH_LABEL.get(a, a)


SEEDS = [0, 1, 2, 3, 4]
TERTILE_SEEDS = [0, 1, 2]
SPLIT_SEED = 0
LR_GRID = [3e-5, 1e-4, 3e-4]

PROTOCOLS = {
    "main": {"epochs": 30, "warmup": 2, "patience": 8, "weight_decay": 1e-4, "batch": 64},
    "v1": {"epochs": 10, "warmup": 0, "patience": 0, "weight_decay": 1e-4, "batch": 64, "lr": 3e-4},
}

CORRUPTIONS = ["gaussian_blur", "brightness", "darkening", "contrast", "gaussian_noise", "jpeg"]
COMPOUNDS = ["compound_blur_dark_jpeg", "compound_noise_contrast_jpeg", "compound_field"]
NATIVE_CORRUPTIONS = ["gaussian_blur", "brightness", "contrast", "gaussian_noise", "jpeg"]
SEVERITIES = [1, 2, 3, 4, 5]
XAI_SEVERITIES = [3, 5]
EXTERNAL_SEVERITIES = [3, 5]

PARAMS = {
    "gaussian_blur": [0.5, 1.0, 1.5, 2.0, 3.0],
    "brightness": [1.2, 1.4, 1.6, 1.8, 2.1],
    "darkening": [0.8, 0.65, 0.5, 0.4, 0.3],
    "contrast": [0.85, 0.7, 0.55, 0.4, 0.3],
    "gaussian_noise": [3, 6, 10, 15, 22],
    "jpeg": [60, 45, 32, 22, 14],
}
NATIVE_PARAMS = {
    "gaussian_blur": [0.5, 1.0, 2.0, 3.0, 4.0],
    "brightness": [1.2, 1.4, 1.6, 1.8, 2.1],
    "contrast": [0.85, 0.7, 0.55, 0.4, 0.3],
    "gaussian_noise": [8, 16, 28, 42, 60],
    "jpeg": [60, 45, 32, 22, 14],
}

N_BOOT = 2000
ALPHA = 0.05
AUC_MARGIN = 0.02
SHIFT_MARGIN = 0.05
LABEL_NOISE_RATES = [0.1, 0.2, 0.3]
LABEL_NOISE_REPS = 200
GRADE_MATCH_REPS = 200
PREVALENCE_REF = 406
PREVALENCE_NONREF = 594
WORKERS = int(os.environ.get("DRFM_WORKERS", "2"))
EVAL_BATCH = 128

if QUICK:
    SEEDS = [0]
    TERTILE_SEEDS = [0]
    LR_GRID = [1e-4]
    SEVERITIES = [1, 5]
    XAI_SEVERITIES = [5]
    EXTERNAL_SEVERITIES = [5]
    PROTOCOLS["main"].update({"epochs": 2, "warmup": 1, "patience": 0})
    PROTOCOLS["v1"].update({"epochs": 2})
    N_BOOT = 200
    LABEL_NOISE_REPS = 20
    GRADE_MATCH_REPS = 20

QUICK_N = int(os.environ.get("DRFM_QUICK_N", "400"))


def find_aptos():
    env = os.environ.get("DRFM_APTOS")
    if env:
        return env
    for c in sorted(glob.glob(os.path.join(INPUT, "**", "train.csv"), recursive=True)):
        d = os.path.dirname(c)
        if os.path.isdir(os.path.join(d, "train_images")):
            return d
    raise FileNotFoundError("APTOS 2019 not found under " + INPUT)


def ensure_dirs():
    for d in [OUT, CACHE, CKPT, RUNS, TABLES, FIGS, CLAIMS, LOGS]:
        os.makedirs(d, exist_ok=True)
