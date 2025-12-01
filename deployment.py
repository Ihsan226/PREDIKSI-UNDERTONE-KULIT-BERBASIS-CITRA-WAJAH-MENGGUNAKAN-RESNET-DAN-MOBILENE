import typer
from pathlib import Path
import json
import torch
from PIL import Image
from torchvision import transforms
from src.modeling import SimpleCNN

app = typer.Typer(add_completion=False)


def load_model(models: Path):
    classes = json.loads((models / "classes.json").read_text(encoding="utf-8"))["classes"]
    model = SimpleCNN(num_classes=len(classes))
    model.load_state_dict(torch.load(models / "model.pt", map_location="cpu"))
    model.eval()
    return model, classes


@app.command()
def main(models: str = typer.Option("models", help="Models folder"), image: str = typer.Option(..., help="Image file path")):
    models_path = Path(models)
    model, classes = load_model(models_path)
    tfm = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    im = Image.open(image).convert("RGB")
    x = tfm(im).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        pred = out.argmax(dim=1).item()
    typer.echo(f"Prediction: {classes[pred]}")


if __name__ == "__main__":
    app()