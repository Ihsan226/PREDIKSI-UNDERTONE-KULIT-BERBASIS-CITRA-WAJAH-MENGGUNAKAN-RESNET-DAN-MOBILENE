import typer
from pathlib import Path
import json
import torch
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
from src.modeling import SimpleCNN

app = typer.Typer(add_completion=False)


@app.command()
def main(
    reports: str = typer.Option("reports", help="Reports folder"),
    models: str = typer.Option("models", help="Models folder"),
    data: str = typer.Option("prepared", help="Prepared data folder"),
):
    reports_path = Path(reports)
    models_path = Path(models)
    prepared = Path(data)

    classes = json.loads((models_path / "classes.json").read_text(encoding="utf-8"))["classes"]
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_ds = datasets.ImageFolder(prepared / "val", transform=normalize)
    loader = torch.utils.data.DataLoader(val_ds, batch_size=32)

    model = SimpleCNN(num_classes=len(classes))
    model.load_state_dict(torch.load(models_path / "model.pt", map_location="cpu"))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x)
            pred = out.argmax(dim=1).numpy()
            y_pred.extend(pred.tolist())
            y_true.extend(y.numpy().tolist())

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)

    # Save confusion matrix figure
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    plt.colorbar(im, ax=ax)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    (reports_path).mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(reports_path / "confusion_matrix.png")

    # Save metrics
    Path(reports_path / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(f"Saved evaluation to {reports_path}")


if __name__ == "__main__":
    app()