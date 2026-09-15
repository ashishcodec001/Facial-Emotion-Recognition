from pathlib import Path
from huggingface_hub import hf_hub_download

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

path = hf_hub_download(
    repo_id="phamquiluan/ResidualMaskingNetwork",
    filename="onnx/resmasking_int8.onnx",
    local_dir=MODEL_DIR,
)

print(f"Model downloaded to: {path}")
