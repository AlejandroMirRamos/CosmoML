"""Shared paper figure style: large serif (Palatino/mathpazo) typography.

Centralises the matplotlib styling used across every notebook figure so the
plots are legible at paper size. Uses real LaTeX (``text.usetex``) with the
``mathpazo`` (Palatino) font when the toolchain is available, falling back to
matplotlib's built-in mathtext serif otherwise (so the notebooks never crash on
a machine without LaTeX).

``texify`` rewrites the Unicode that the notebooks use in titles/labels
(``ΛCDM``, ``w₀wₐCDM``, ``·``, ``χ²`` …) into LaTeX, because ``usetex`` cannot
render those code points directly.
"""
from __future__ import annotations
import shutil
import matplotlib
import matplotlib.pyplot as plt

# LaTeX is usable only if both latex and dvipng are on PATH (dvipng is needed
# for the Agg/PNG and inline backends).
USETEX = bool(shutil.which("latex") and shutil.which("dvipng"))

# Palatino text + matching math.
LATEX_PREAMBLE = r"\usepackage{mathpazo}"

# Paper-size font sizes (clearly larger than the matplotlib defaults).
FONT_SIZES = {
    "font.size":        15,
    "axes.titlesize":   18,
    "axes.labelsize":   17,
    "xtick.labelsize":  14,
    "ytick.labelsize":  14,
    "legend.fontsize":  14,
    "figure.titlesize": 19,
}

# Canonical LaTeX labels per parameter, shared by SHAP feature names etc.
PARAM_LABELS = {
    "Om": r"$\Omega_m$", "H0": r"$H_0$", "w0": r"$w_0$", "wa": r"$w_a$",
    "C1": r"$C_1$", "C3": r"$C_3$", "bq": r"$\beta^q$",
}

# Unicode -> LaTeX. Longer/compound keys first so e.g. ``χ²`` wins over ``χ``.
_TEX_SUBS = [
    ("w₀wₐ", r"$w_0 w_a$"),
    ("ΛCDM", r"$\Lambda$CDM"),
    ("Ωm",   r"$\Omega_m$"),
    ("χ²",   r"$\chi^2$"),
    ("Λ", r"$\Lambda$"), ("Ω", r"$\Omega$"), ("Σ", r"$\Sigma$"),
    ("Δ", r"$\Delta$"),  ("χ", r"$\chi$"),  ("β", r"$\beta$"),
    ("σ", r"$\sigma$"),  ("τ", r"$\tau$"),
    ("·", r"$\cdot$"),   ("→", r"$\to$"),   ("×", r"$\times$"),
    ("±", r"$\pm$"),     ("≈", r"$\approx$"),
    ("≤", r"$\leq$"),    ("≥", r"$\geq$"),
    ("₀", "0"), ("ₐ", "a"),   # stray subscripts
]


def texify(s: str | None) -> str | None:
    """Rewrite the Unicode used in titles/labels into usetex-safe LaTeX."""
    if not s:
        return s
    for uni, tex in _TEX_SUBS:
        s = s.replace(uni, tex)
    if USETEX:
        # '&' is the LaTeX alignment-tab character; escape it for text mode so
        # dataset labels like "Pantheon+ & BAO" don't crash the LaTeX run.
        s = s.replace("&", r"\&")
    return s


def feature_labels(columns) -> list[str]:
    """Map raw feature/column names to their canonical LaTeX labels."""
    return [PARAM_LABELS.get(c, c) for c in columns]


def apply_paper_style(usetex: bool | None = None) -> None:
    """Apply the shared large serif style to all matplotlib plots.

    Call once per notebook (and re-call after importing libraries such as
    getdist or SMEFT19 that overwrite the rc params).
    """
    use = USETEX if usetex is None else (usetex and USETEX)
    plt.rcParams["text.usetex"] = use
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Palatino", "Palatino Linotype",
                                  "URW Palladio L", "DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "cm"   # used only when usetex is off
    if use:
        plt.rcParams["text.latex.preamble"] = LATEX_PREAMBLE
    plt.rcParams.update(FONT_SIZES)


def reassert_usetex() -> None:
    """Restore ``text.usetex`` to the desired state.

    getdist's plot styles (and SMEFT19 on import) flip ``text.usetex``; call this
    right before rendering so figures keep the chosen typography.
    """
    matplotlib.rcParams["text.usetex"] = USETEX


def style_getdist(g) -> None:
    """Enlarge fonts on a getdist subplot plotter for paper readability."""
    s = g.settings
    s.axes_fontsize = 14        # tick numbers
    s.axes_labelsize = 20       # axis labels
    s.legend_fontsize = 16
    s.title_limit_fontsize = 15
    s.colorbar_axes_fontsize = 13
    s.scaling = False           # don't auto-shrink fonts on many-panel grids
