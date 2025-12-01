# CRISP-DM Image Classification Pipeline

This project applies the CRISP-DM methodology to a simple image classification task using the dataset in `archive/train/` with classes `Black`, `Brown`, and `White`.

## Structure
- `src/`: Python modules implementing each CRISP-DM phase
- `reports/`: Generated metrics and visuals
- `models/`: Saved models and artifacts

## Quick Start

1) Create and activate a Python environment (optional but recommended).

2) Install dependencies:

```
pip install -r requirements.txt
```

3) Run CRISP-DM steps:

```
python -m src.business_understanding
python -m src.data_understanding --data "archive/train"
python -m src.data_preparation --data "archive/train" --out "prepared"
python -m src.modeling --data "prepared" --reports "reports" --models "models"
python -m src.evaluation --reports "reports" --models "models"
python -m src.deployment --models "models" --image "path/to/image.jpg"
```

Notes:
- Adjust paths if running from a different working directory.
- `prepared/` is generated with train/val split and transformations.

## Run the Web App

After training/evaluation, start the web UI (FastAPI + Tailwind + Chart.js):

```
uvicorn app.main:app --reload
```

Then open http://127.0.0.1:8000 in your browser. You can:
- Upload gambar untuk prediksi interaktif.
- Klik contoh dari validation set.
- Lihat grafik training history dan confusion matrix.

### Django Alternative

You can also run a Django version of the web app:

```
pip install -r requirements.txt
cd django_app
python manage.py migrate
python manage.py runserver
```

Open http://127.0.0.1:8000 — features mirror the FastAPI UI (upload, samples, charts). The Django app reuses the trained model saved in `models/` and the artifacts in `prepared/` and `reports/`.

## CRISP-DM Phases Covered
- Business Understanding: Goal and success criteria.
- Data Understanding: Class counts, image sizes, sample grid.
- Data Preparation: Train/val split, normalization, resizing.
- Modeling: Baseline CNN (PyTorch) with accuracy.
- Evaluation: Confusion matrix and classification report.
- Deployment: Simple `predict.py` to classify a new image.
