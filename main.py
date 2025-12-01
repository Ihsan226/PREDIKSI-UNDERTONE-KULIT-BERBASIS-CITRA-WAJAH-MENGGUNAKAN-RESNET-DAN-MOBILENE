from pathlib import Path
from typing import Optional, List, Dict

import io
import json

import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from torchvision import transforms

# Reuse model & loader from existing code
from src.deployment import load_model

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
PREPARED_DIR = ROOT / "prepared"
REPORTS_DIR = ROOT / "reports"

app = FastAPI(title="CRISP-DM Image Classifier")
app.mount("/static", StaticFiles(directory=str(ROOT / "app" / "static")), name="static")
if PREPARED_DIR.exists():
    app.mount("/prepared", StaticFiles(directory=str(PREPARED_DIR)), name="prepared")
if REPORTS_DIR.exists():
    app.mount("/reports", StaticFiles(directory=str(REPORTS_DIR)), name="reports")

templates = Jinja2Templates(directory=str(ROOT / "app" / "templates"))

# Load model on startup (if available)
try:
    MODEL, CLASSES = load_model(MODELS_DIR)
    AVAILABLE = True
except Exception:
    MODEL, CLASSES, AVAILABLE = None, [], False

TFM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict_image(img: Image.Image) -> Dict:
    if not AVAILABLE:
        return {"error": "Model belum tersedia. Latih model terlebih dahulu."}
    x = TFM(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits = MODEL(x)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy().tolist()
    best_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
    return {
        "predicted": CLASSES[best_idx],
        "probs": [{"label": CLASSES[i], "prob": float(p)} for i, p in enumerate(probs)],
    }


def list_sample_images(max_per_class: int = 4) -> List[Dict[str, str]]:
    samples: List[Dict[str, str]] = []
    if not PREPARED_DIR.exists():
        return samples
    val_root = PREPARED_DIR / "val"
    if not val_root.exists():
        return samples
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for cls_dir in sorted([p for p in val_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        imgs = [p for p in sorted(cls_dir.iterdir(), key=lambda p: p.name) if p.suffix.lower() in exts]
        for p in imgs[:max_per_class]:
            # URL relative to /prepared mount
            rel = p.relative_to(PREPARED_DIR).as_posix()
            samples.append({"url": f"/prepared/{rel}", "label": cls_dir.name})
    return samples


@app.get("/health")
def health():
    return {"status": "ok", "model": AVAILABLE, "classes": CLASSES}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Load metrics if available
    history = {}
    best_val = None
    if (REPORTS_DIR / "training_history.json").exists():
        try:
            history = json.loads((REPORTS_DIR / "training_history.json").read_text(encoding="utf-8"))
            if "val_acc" in history and history["val_acc"]:
                best_val = max(history["val_acc"])  # best validation accuracy
        except Exception:
            pass
    cm_img = "/reports/confusion_matrix.png" if (REPORTS_DIR / "confusion_matrix.png").exists() else None
    cls_report = None
    if (REPORTS_DIR / "classification_report.json").exists():
        try:
            cls_report = json.loads((REPORTS_DIR / "classification_report.json").read_text(encoding="utf-8"))
        except Exception:
            pass
    samples = list_sample_images()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "model_available": AVAILABLE,
            "classes": CLASSES,
            "best_val": best_val,
            "cm_img": cm_img,
            "history": history,
            "cls_report": cls_report,
            "samples": samples,
        },
    )


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content))
        result = predict_image(img)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.post("/predict-by-path")
async def predict_by_path(path: str = Form(...)):
    # Only allow reading within PREPARED_DIR for safety
    full = (ROOT / path).resolve() if not path.startswith("prepared/") else (ROOT / path).resolve()
    try:
        # Enforce that file lies under PREPARED_DIR
        if PREPARED_DIR.exists():
            _ = full.relative_to(PREPARED_DIR)
        img = Image.open(full)
        result = predict_image(img)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
