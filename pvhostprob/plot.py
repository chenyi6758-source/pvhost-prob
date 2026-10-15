"""Figures: exceedance curves and locational hosting-capacity bars."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_exceedance(curve_df, risk=0.05, markers=None, title=None, path=None):
    """P(violation) vs installed PV per bus, with Wilson 95% band.

    Parameters
    ----------
    curve_df : pandas.DataFrame
        Output of :func:`exceedance_curve` (columns kw_per_bus, p_viol,
        ci95_lo, ci95_hi).
    risk : float
        Accepted risk level (horizontal dashed line).
    markers : dict, optional
        {label: kw} vertical lines, e.g. {"deterministic HC": 12.3}.
    path : str, optional
        Save figure here.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = curve_df["kw_per_bus"].to_numpy()
    y = curve_df["p_viol"].to_numpy()
    ax.fill_between(
        x,
        curve_df["ci95_lo"].to_numpy(),
        curve_df["ci95_hi"].to_numpy(),
        alpha=0.25,
        label="95% Wilson CI",
    )
    ax.plot(x, y, "o-", ms=4, label="P(violation)")
    ax.axhline(risk, color="red", ls="--", lw=1.2, label=f"accepted risk {risk:.0%}")
    ax.set_xlabel("Installed PV per candidate bus (kW)")
    ax.set_ylabel("P(any limit violated)")
    ax.set_title(title or "Exceedance curve: violation probability vs PV size")
    ax.set_ylim(-0.02, 1.02)
    from matplotlib.transforms import blended_transform_factory

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for i, (label, kw) in enumerate((markers or {}).items()):
        ax.axvline(kw, color="k", ls=":", lw=1.2)
        ax.text(
            kw,
            0.96 - 0.10 * i,
            f" {label}\n {kw:.2f} kW",
            fontsize=8,
            va="top",
            ha="left",
            transform=trans,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
        )
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def plot_locational(loc_df, risk=0.05, title=None, path=None):
    """Bar chart of per-bus probabilistic hosting capacity, colored by binding limit."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = {"voltage": "#d62728", "thermal": "#1f77b4"}
    bars = ax.bar(
        loc_df["bus_name"],
        loc_df["hc_kw"],
        color=[colors.get(b, "gray") for b in loc_df["binding"]],
    )
    ax.bar_label(bars, fmt="%.1f", fontsize=8)
    ax.set_xlabel("Bus (PV connected at this bus only)")
    ax.set_ylabel(f"Hosting capacity at {risk:.0%} risk (kW)")
    ax.set_title(title or "Locational probabilistic hosting capacity")
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(color=colors["voltage"], label="voltage binds"),
            Patch(color=colors["thermal"], label="thermal binds"),
        ],
        fontsize=8,
    )
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig
