from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = PROJECT_ROOT / "assets"
DATA_DIR = PROJECT_ROOT / "data" / "landmarks"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
RUNTIME_DIR = PROJECT_ROOT / ".rps"

HAND_LANDMARKER_PATH = ASSETS_DIR / "hand_landmarker.task"
HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_LANDMARKER_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"

DEFAULT_CHECKPOINT_PATH = MODELS_DIR / "gesture_mlp.pt"
DEFAULT_TEMPORAL_POLICY_PATH = MODELS_DIR / "temporal_policy.json"
TEMPORAL_POLICY_CANDIDATE_PATH = MODELS_DIR / "temporal_policy-candidate.json"
DEVICE_CONFIG_PATH = RUNTIME_DIR / "device.json"
