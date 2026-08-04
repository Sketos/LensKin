#!/usr/bin/env python
"""
Simple image / PDF viewer for hosts without dedicated PNG/PDF apps.

Uses matplotlib for display. PDF pages are rasterized when possible
(pymupdf, pdf2image, or poppler ``pdftoppm``).

Examples
--------
  python scripts/view_media.py plot.png
  python scripts/view_media.py cornerplot.pdf
  python scripts/view_media.py plots/*.png
  python scripts/view_media.py output/run/image --recursive
  python scripts/view_media.py file.pdf --page 0
  python scripts/view_media.py plots/ --html gallery.html   # no GUI needed
"""

from __future__ import annotations

import argparse
import html
import sys
import tempfile
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gif", ".bmp", ".webp"}
PDF_SUFFIXES = {".pdf"}


def _collect_paths(inputs, recursive=False):
    paths = []
    for raw in inputs:
        path = Path(raw).expanduser().resolve()
        if path.is_dir():
            pattern = "**/*" if recursive else "*"
            for child in sorted(path.glob(pattern)):
                if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES | PDF_SUFFIXES:
                    paths.append(child)
        elif path.is_file():
            paths.append(path)
        else:
            print(f"WARNING: not found: {raw}", file=sys.stderr)
    # Stable unique order
    seen = set()
    unique = []
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _load_image(path):
    import matplotlib.pyplot as plt
    import numpy as np

    array = plt.imread(path)
    if array.ndim == 2:
        return array
    if array.shape[-1] == 4:
        # Drop alpha for display consistency
        return array[..., :3]
    return np.asarray(array)


def _pdf_pages_pymupdf(path, dpi=120):
    import fitz
    import numpy as np

    doc = fitz.open(path)
    pages = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    for page in doc:
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        pages.append(image)
    doc.close()
    return pages


def _pdf_pages_pdf2image(path, dpi=120):
    from pdf2image import convert_from_path
    import numpy as np

    images = convert_from_path(str(path), dpi=dpi)
    return [np.asarray(img.convert("RGB")) for img in images]


def _pdf_pages_pdftoppm(path, dpi=120):
    import subprocess
    import numpy as np
    import matplotlib.pyplot as plt

    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp) / "page"
        result = subprocess.run(
            [
                "pdftoppm",
                "-png",
                "-r",
                str(dpi),
                str(path),
                str(prefix),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "pdftoppm failed")
        pages = []
        for png in sorted(Path(tmp).glob("page*.png")):
            pages.append(_load_image(png))
        if not pages:
            raise RuntimeError("pdftoppm produced no pages")
        return pages


def load_pdf_pages(path, dpi=120, page=None):
    errors = []
    loaders = (
        ("pymupdf", _pdf_pages_pymupdf),
        ("pdf2image", _pdf_pages_pdf2image),
        ("pdftoppm", _pdf_pages_pdftoppm),
    )
    pages = None
    for name, loader in loaders:
        try:
            pages = loader(path, dpi=dpi)
            break
        except Exception as exc:  # noqa: BLE001 - try next backend
            errors.append(f"{name}: {exc}")
    if pages is None:
        raise RuntimeError(
            "Could not render PDF. Install one of: pymupdf (`pip install pymupdf`), "
            "pdf2image (+ poppler), or poppler's pdftoppm.\n"
            + "\n".join(f"  - {e}" for e in errors)
        )
    if page is not None:
        if page < 0 or page >= len(pages):
            raise IndexError(f"page {page} out of range [0, {len(pages) - 1}]")
        return [pages[page]]
    return pages


def load_frames(path, dpi=120, page=None):
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return [(_load_image(path), path.name)]
    if suffix in PDF_SUFFIXES:
        pages = load_pdf_pages(path, dpi=dpi, page=page)
        if len(pages) == 1:
            labels = [path.name]
        else:
            labels = [f"{path.name} [p{i}]" for i in range(len(pages))]
        return list(zip(pages, labels))
    raise ValueError(f"Unsupported file type: {path}")


def show_interactive(paths, dpi=120, page=None, cmap=None):
    import matplotlib.pyplot as plt

    frames = []
    for path in paths:
        frames.extend(load_frames(path, dpi=dpi, page=page))
    if not frames:
        print("No images to display.", file=sys.stderr)
        return 1

    index = {"i": 0}

    figure, axis = plt.subplots(figsize=(10, 8))
    figure.canvas.manager.set_window_title("view_media")

    def draw():
        axis.clear()
        image, label = frames[index["i"]]
        if image.ndim == 2:
            axis.imshow(image, cmap=cmap or "gray", origin="upper")
        else:
            axis.imshow(image, origin="upper")
        axis.set_title(f"{label}  ({index['i'] + 1}/{len(frames)})")
        axis.axis("off")
        figure.tight_layout()
        figure.canvas.draw_idle()

    def on_key(event):
        if event.key in {"right", "n", " ", "pagedown"}:
            index["i"] = (index["i"] + 1) % len(frames)
            draw()
        elif event.key in {"left", "p", "pageup"}:
            index["i"] = (index["i"] - 1) % len(frames)
            draw()
        elif event.key in {"q", "escape"}:
            plt.close(figure)
        elif event.key == "home":
            index["i"] = 0
            draw()
        elif event.key == "end":
            index["i"] = len(frames) - 1
            draw()

    figure.canvas.mpl_connect("key_press_event", on_key)
    draw()
    print(
        f"Showing {len(frames)} frame(s). "
        "Keys: ←/→ or n/p = navigate, q = quit."
    )
    plt.show()
    return 0


def write_html_gallery(paths, output_html, dpi=120, page=None):
    """Write a self-contained HTML gallery (images embedded as files alongside)."""
    output_html = Path(output_html).expanduser().resolve()
    output_html.parent.mkdir(parents=True, exist_ok=True)
    asset_dir = output_html.parent / f"{output_html.stem}_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    entries = []
    frame_i = 0
    for path in paths:
        for image, label in load_frames(path, dpi=dpi, page=page):
            frame_i += 1
            out_name = f"frame_{frame_i:04d}.png"
            out_path = asset_dir / out_name
            plt.imsave(out_path, image)
            rel = out_path.relative_to(output_html.parent).as_posix()
            entries.append((label, rel))

    lines = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'>",
        f"<title>{html.escape(output_html.name)}</title>",
        "<style>",
        "body{font-family:sans-serif;margin:1.5rem;background:#111;color:#eee;}",
        "img{max-width:100%;height:auto;border:1px solid #444;margin:0.5rem 0 1.5rem;}",
        "h2{font-size:1rem;font-weight:600;margin:1.2rem 0 0.3rem;}",
        "</style></head><body>",
        f"<h1>{html.escape(str(output_html))}</h1>",
        f"<p>{len(entries)} frame(s)</p>",
    ]
    for label, rel in entries:
        lines.append(f"<h2>{html.escape(label)}</h2>")
        lines.append(f"<img src='{html.escape(rel)}' alt='{html.escape(label)}'>")
    lines.append("</body></html>")
    output_html.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output_html} ({len(entries)} frames, assets in {asset_dir})")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Display PNG/JPEG/TIFF/PDF files with Python (matplotlib)."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Image/PDF files or directories containing them.",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recurse into directories.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="Rasterization DPI for PDF pages (default: 120).",
    )
    parser.add_argument(
        "--page",
        type=int,
        default=None,
        help="PDF page index to show (0-based). Default: all pages.",
    )
    parser.add_argument(
        "--cmap",
        default=None,
        help="Matplotlib colormap for single-channel images (e.g. gray, viridis).",
    )
    parser.add_argument(
        "--html",
        metavar="FILE",
        default=None,
        help="Write an HTML gallery instead of opening a GUI (useful with no display).",
    )
    args = parser.parse_args(argv)

    paths = _collect_paths(args.paths, recursive=args.recursive)
    if not paths:
        print("No supported media files found.", file=sys.stderr)
        return 1

    print(f"Found {len(paths)} file(s):")
    for path in paths:
        print(f"  {path}")

    if args.html:
        return write_html_gallery(paths, args.html, dpi=args.dpi, page=args.page)

    try:
        return show_interactive(paths, dpi=args.dpi, page=args.page, cmap=args.cmap)
    except Exception as exc:  # noqa: BLE001
        # Common on headless clusters with no DISPLAY / backend.
        print(f"Interactive display failed: {exc}", file=sys.stderr)
        fallback = Path("view_media_gallery.html")
        print(
            f"Falling back to HTML gallery: {fallback}\n"
            "Open it in a browser, or re-run with --html path.html",
            file=sys.stderr,
        )
        return write_html_gallery(paths, fallback, dpi=args.dpi, page=args.page)


if __name__ == "__main__":
    raise SystemExit(main())
