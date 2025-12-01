import typer
from pathlib import Path
from typing import Tuple
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.optim as optim
from src.utils import ensure_dir, save_json

app = typer.Typer(add_completion=False)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def loaders(prepared: Path, batch_size: int = 32) -> Tuple[DataLoader, DataLoader, list]:
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_ds = datasets.ImageFolder(prepared / "train", transform=normalize)
    val_ds = datasets.ImageFolder(prepared / "val", transform=normalize)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader, train_ds.classes


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total


@app.command()
def main(
    data: str = typer.Option("prepared", help="Prepared data folder"),
    reports: str = typer.Option("reports", help="Reports output folder"),
    models: str = typer.Option("models", help="Models output folder"),
    epochs: int = typer.Option(5, min=1, max=50),
    batch_size: int = typer.Option(32, min=8, max=128),
    lr: float = typer.Option(1e-3),
):
    prepared = Path(data)
    reports_path = Path(reports)
    models_path = Path(models)
    ensure_dir(reports_path)
    ensure_dir(models_path)

    train_loader, val_loader, classes = loaders(prepared, batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch {ep}: train_acc={tr_acc:.3f} val_acc={val_acc:.3f}")

    save_json(models_path / "classes.json", {"classes": classes})
    torch.save(model.state_dict(), models_path / "model.pt")
    save_json(reports_path / "training_history.json", history)
    print(f"Saved model to {models_path / 'model.pt'} and history to {reports_path}")


if __name__ == "__main__":
    app()