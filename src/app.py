"""
FastAPI server for MNIST digit classification.
Serve the trained model via REST API.
"""
import io
import os
import sys
import yaml
import torch
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException

# Ensure src is on path when running as module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import get_model

# Project root (parent of src)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(
    title="MNIST Digit Classification API",
    description="Predict handwritten digits (0-9) from image uploads",
    version="1.0.0",
)

# Loaded at startup
model = None
device = None
config = None

# Same preprocessing as training/inference
TRANSFORM = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load and resolve config paths."""
    if not os.path.isabs(config_path):
        config_path = os.path.join(PROJECT_ROOT, config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    for key in ["model_dir", "checkpoint_dir"]:
        if key in cfg.get("paths", {}):
            p = cfg["paths"][key]
            if not os.path.isabs(p):
                cfg["paths"][key] = os.path.normpath(os.path.join(PROJECT_ROOT, p))
    return cfg


def get_model_path(cfg: dict) -> str:
    """Resolve path to best or final model."""
    best = os.path.join(cfg["paths"]["checkpoint_dir"], "best.pth")
    if os.path.isfile(best):
        return best
    return os.path.join(cfg["paths"]["model_dir"], "final_model.pth")


def preprocess_image_bytes(image_bytes: bytes) -> torch.Tensor:
    """Preprocess uploaded image bytes to model input tensor."""
    img = Image.open(io.BytesIO(image_bytes))
    tensor = TRANSFORM(img).unsqueeze(0)
    return tensor


def predict_digit(tensor: torch.Tensor) -> tuple[int, float, list[float]]:
    """Run model and return (digit, confidence, list of 10 probs)."""
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)
        pred = logits.argmax(dim=1).item()
        conf = probs[0][pred].item()
    return pred, conf, probs[0].cpu().tolist()


@app.on_event("startup")
def startup():
    global model, device, config
    config = load_config()
    device = torch.device(
        config["training"]["device"] if torch.cuda.is_available() else "cpu"
    )
    model_path = get_model_path(config)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train with: python src/train.py"
        )
    model_obj = get_model(
        num_classes=config["model"]["num_classes"],
        device=device,
    )
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model_obj.load_state_dict(ckpt["model_state_dict"])
    else:
        model_obj.load_state_dict(ckpt)
    model_obj.eval()
    model = model_obj


@app.get("/")
def root():
    return {
        "service": "MNIST Digit Classification API",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict with image file",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Upload an image; returns predicted digit (0-9), confidence, and per-class probabilities."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image (e.g. image/png, image/jpeg)")
    try:
        contents = await file.read()
        tensor = preprocess_image_bytes(contents)
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e!s}")
    digit, confidence, probabilities = predict_digit(tensor)
    return {
        "predicted_digit": digit,
        "confidence": round(confidence, 6),
        "probabilities": {str(i): round(p, 6) for i, p in enumerate(probabilities)},
    }
