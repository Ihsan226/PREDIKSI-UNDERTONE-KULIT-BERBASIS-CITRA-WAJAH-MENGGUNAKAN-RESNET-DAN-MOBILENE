import typer
from pathlib import Path
from collections import Counter
from typing import List
from PIL import Image
import matplotlib.pyplot as plt

from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer(add_completion=False)


def list_images(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return [p for p in folder.rglob("*") if p.suffix.lower() in exts]


@app.command()
def main(data: str = typer.Option(..., help="Path to archive/train")):
    data_path = Path(data)
    assert data_path.exists(), f"Data path not found: {data_path}"

    classes = [p for p in data_path.iterdir() if p.is_dir()]
    table = Table(title="Class Counts")
    table.add_column("Class")
    table.add_column("Images", justify="right")

    counts = {}
    for cls in classes:
        n = len(list_images(cls))
        counts[cls.name] = n
        table.add_row(cls.name, str(n))

    console.print(table)

    # Sample image sizes
    sizes = []
    sample_paths = []
    for cls in classes:
        imgs = list_images(cls)
        sample_paths.extend(imgs[:5])
        for p in imgs[:50]:
            try:
                with Image.open(p) as im:
                    sizes.append(im.size)
            except Exception:
                pass

    if sizes:
        w = [s[0] for s in sizes]
        h = [s[1] for s in sizes]
        console.print(f"Sampled {len(sizes)} images, avg size: {sum(w)//len(w)}x{sum(h)//len(h)}")

    # Create a sample grid figure
    if sample_paths:
        n = min(len(sample_paths), 12)
        cols = 4
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes = axes.flatten()
        for i in range(len(axes)):
            axes[i].axis("off")
        for i, p in enumerate(sample_paths[:n]):
            try:
                im = Image.open(p)
                axes[i].imshow(im)
                axes[i].set_title(p.parent.name)
            except Exception:
                pass
        out = Path("reports") / "data_understanding_samples.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out)
        console.print(f"Saved sample grid to {out}")


if __name__ == "__main__":
    app()