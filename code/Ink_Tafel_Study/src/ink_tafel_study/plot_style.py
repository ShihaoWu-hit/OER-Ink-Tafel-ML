from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    "red_1": "#F6CFCB",
    "red_2": "#E9A6A1",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "neutral_dark": "#4D4D4D",
    "teal": "#42949E",
    "violet": "#9A4D8E",
}

_SUPPORTED_FIGURE_FORMATS = frozenset(
    {"pdf", "svg", "eps", "png", "jpg", "jpeg", "tif", "tiff"}
)


def apply_publication_style(font_size: int = 12) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": font_size,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 1.5,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            # Matplotlib otherwise creates random SVG element identifiers,
            # which makes byte-identical reruns impossible to verify.
            "svg.hashsalt": "ink-tafel-study-v1",
        }
    )


def save_figure(
    fig: plt.Figure,
    output_base: str | Path,
    *,
    formats: Iterable[str] = ("png", "pdf"),
    dpi: int = 300,
) -> list[Path]:
    base = Path(output_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.5)
    written: list[Path] = []
    normalized_formats = tuple(
        str(extension).strip().lower().lstrip(".") for extension in formats
    )
    unsupported = sorted(set(normalized_formats) - _SUPPORTED_FIGURE_FORMATS)
    if unsupported:
        raise ValueError(f"Unsupported figure formats: {unsupported}")
    for extension in normalized_formats:
        path = base.with_suffix(f".{extension}")
        metadata = (
            {"CreationDate": None, "ModDate": None}
            if extension == "pdf"
            else {"Date": None}
            if extension == "svg"
            else None
        )
        fig.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
            metadata=metadata,
        )
        written.append(path)
    plt.close(fig)
    return written
