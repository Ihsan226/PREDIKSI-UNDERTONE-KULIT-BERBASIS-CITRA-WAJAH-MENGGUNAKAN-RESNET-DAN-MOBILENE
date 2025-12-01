import typer
from pathlib import Path
from typing import Tuple
from PIL import Image
import random
from src.utils import ensure_dir, save_json

app = typer.Typer(add_completion=False)


def split_train_val(src: Path, dst: Path, val_ratio: float = 0.2) -> Tuple[int, int]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    total, val_total = 0, 0
    for cls in [p for p in src.iterdir() if p.is_dir()]:
        imgs = [p for p in cls.iterdir() if p.suffix.lower() in exts]
        random.shuffle(imgs)
        n_val = max(1, int(len(imgs) * val_ratio))
        train_imgs = imgs[n_val:]
        val_imgs = imgs[:n_val]
        for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs)]:
            out_dir = dst / split_name / cls.name
            ensure_dir(out_dir)
            for p in split_imgs:
                try:
                    im = Image.open(p).convert("RGB")
                    im = im.resize((128, 128))
                    im.save(out_dir / p.name)
                except Exception:
                    pass
        total += len(imgs)
        val_total += len(val_imgs)
    return total, val_total


@app.command()
def main(
    data: str = typer.Option(..., help="Path to archive/train"),
    out: str = typer.Option("prepared", help="Output folder for prepared data"),
    val_ratio: float = typer.Option(0.2, min=0.05, max=0.5, help="Validation split ratio"),
):
    src = Path(data)
    dst = Path(out)
    ensure_dir(dst)
    total, val_total = split_train_val(src, dst, val_ratio)
    meta = {"source": str(src), "prepared": str(dst), "total": total, "val_total": val_total, "image_size": [128, 128]}
    save_json(dst / "metadata.json", meta)
    typer.echo(f"Prepared data at {dst}, total images: {total}, val: {val_total}")


if __name__ == "__main__":
    app()