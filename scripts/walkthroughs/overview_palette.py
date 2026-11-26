"""Shared model-to-colour mapping and chart style for the walkthrough figures (notes/walkthroughs/).

Every walkthrough figure colours a model the same way, so a reader who learns "Qwen3-8B is blue" in one
figure can rely on it in the next. Import this module rather than copying hex values:

    import sys; sys.path.insert(0, "scripts/walkthroughs")
    from overview_palette import model_color, model_label, apply_style

Where the values come from. The eight categorical slots are the documented default palette of the
data-visualization method used for these figures (light mode, chart surface #fcfcfb), in its fixed
slot order: blue, orange, aqua, yellow, magenta, green, violet, red. The Qwen3 size series takes steps of
the blue ramp (one hue, lighter = smaller), so the size order is visible in the colour. Models that are
never interpreted (they fail a competence rule) are drawn in the muted grey.

Validation (run 2026-11-26 with the method's validator, `validate_palette.py`, the Python twin of
`validate_palette.js`; Node was not available on the machine):
  * the eight slots, adjacent pairs, light mode: every hard check passes (worst adjacent CVD
    distance 9.1, worst normal-vision distance 19.6); aqua, yellow and magenta sit below 3:1 contrast
    on the surface, so every chart carries the model name as visible text beside each mark;
  * the Qwen3 blue ramp (#6da7ec, #2a78d6, #1c5cab, #104281) as an ordinal ramp: passes (monotone
    lightness, adjacent steps at least 0.06 apart, light end 2.44:1, hue spread 3 degrees);
  * under all-pairs checking only the first three slots pass, as the method documents. The dot plots
    show more than three models at once, so colour is never the only identity channel there: every
    row is labelled with the model's name on the axis, and the legend repeats the mapping.
"""
from __future__ import annotations

# --- the documented categorical slots (light mode) -------------------------------------------------
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")

# Blue ramp steps (one hue), used for the Qwen3 size series.
BLUE_300, BLUE_450, BLUE_550, BLUE_650 = "#6da7ec", "#2a78d6", "#1c5cab", "#104281"
# Yellow, a darker documented step, for the Gemma 4 models (same family hue as Gemma 2).
YELLOW_DARK = "#c98500"

# --- chart chrome and ink (light mode) ---------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# --- model -> colour ---------------------------------------------------------------------------------
# Keys are the model tags used in the metrics files; several spellings map to one model.
MODEL_COLORS: dict[str, str] = {
    # reference models
    "qwen3-8b": BLUE,
    "llama3.1-8b": ORANGE,
    # families wave A and later studies
    "mistral-7b-v0.3": AQUA,
    "gemma2-9b-it": YELLOW,
    "llama3.2-3b": MAGENTA,
    # families wave B and TextCraft round 3 (current-generation models)
    "ministral3-8b-it": GREEN,
    "qwen3.5-9b": VIOLET,
    "qwen3.8-27b": RED,
    "gemma4-e4b-it": YELLOW_DARK,
    "gemma4-12b-it": YELLOW_DARK,
    "gemma4-31b-it": YELLOW_DARK,
    # Qwen3 size series (ordinal blue ramp; 8B is the slot-1 blue above)
    "qwen3-4b": BLUE_300,
    "qwen3-14b": BLUE_550,
    "qwen3-32b": BLUE_650,
    # models that fail a competence rule and are never interpreted
    "llama3.2-1b": MUTED,
    "qwen3-0.6b": MUTED,
    "qwen3-1.7b": MUTED,
    "olmo3-7b-it": MUTED,
}

_ALIASES = {
    "qwen3-8b-fewshot": "qwen3-8b",
    "llama3.1-8b-fewshot": "llama3.1-8b",
    "gemma2-9b": "gemma2-9b-it",
    "mistral-7b": "mistral-7b-v0.3",
    "qwen3.5-9b-plain": "qwen3.5-9b",
    "qwen3.5-9b-base-plain": "qwen3.5-9b",   # same hue; the base model is drawn with a hollow marker
    "qwen3.5-9b-base": "qwen3.5-9b",
}

MODEL_LABELS: dict[str, str] = {
    "qwen3-8b": "Qwen3-8B",
    "llama3.1-8b": "Llama-3.1-8B-Instruct",
    "mistral-7b-v0.3": "Mistral-7B-Instruct-v0.3",
    "gemma2-9b-it": "Gemma-2-9B-it",
    "llama3.2-3b": "Llama-3.2-3B-Instruct",
    "llama3.2-1b": "Llama-3.2-1B-Instruct",
    "ministral3-8b-it": "Ministral-3-8B-Instruct",
    "qwen3.5-9b": "Qwen3.5-9B",
    "qwen3.8-27b": "Qwen3.8-27B",
    "gemma4-e4b-it": "Gemma-4-E4B-it",
    "gemma4-12b-it": "Gemma-4-12B-it",
    "gemma4-31b-it": "Gemma-4-31B-it",
    "qwen3-4b": "Qwen3-4B",
    "qwen3-14b": "Qwen3-14B",
    "qwen3-32b": "Qwen3-32B",
    "qwen3-0.6b": "Qwen3-0.6B",
    "qwen3-1.7b": "Qwen3-1.7B",
    "olmo3-7b-it": "OLMo-3-7B-Instruct",
}


def canonical(tag: str) -> str:
    """The mapping key for any spelling of a model tag used in the metrics files."""
    t = tag.strip().lower().replace(" (size run)", "").replace(" (reference)", "")
    return _ALIASES.get(t, t)


def model_color(tag: str) -> str:
    """The model's colour; an unknown model raises rather than getting a generated hue."""
    key = canonical(tag)
    if key not in MODEL_COLORS:
        raise KeyError(f"no colour assigned to model {tag!r}; add it to MODEL_COLORS (fold, never generate)")
    return MODEL_COLORS[key]


def model_label(tag: str) -> str:
    key = canonical(tag)
    return MODEL_LABELS.get(key, tag)


def apply_style() -> None:
    """Recessive chrome: hairline solid grid and axes, system sans, text in ink tokens."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9.5,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "text.color": INK,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    })


if __name__ == "__main__":
    for key, colour in MODEL_COLORS.items():
        print(f"{MODEL_LABELS.get(key, key):28s} {colour}")
