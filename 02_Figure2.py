import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# PARAMETRES POLICE FIGURES
# =============================================================================
# Augmenter ces valeurs si vous souhaitez encore agrandir les textes.
FONT_SIZE = 13
LABEL_SIZE = 20
TICK_SIZE = 16
LEGEND_SIZE = 16
STAR_SIZE = 20

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": LEGEND_SIZE,
})


# =============================================================================
# PARAMETRES AXE BRISÉ
# =============================================================================
y_low_max = 0.10
y_high_min = 0.35
y_high_max = 0.5

# =============================================================================
# PARAMETRES SIGNIFICATIVITE
# =============================================================================
p_col = "p_perm_empirical"   # colonne de p-valeur dans rsa_results.csv
alpha_1 = 0.05
alpha_2 = 0.01

# décalage vertical des étoiles
star_offset_low = 0.008
star_offset_high = 0.015


# =============================================================================
# FICHIERS
# =============================================================================
rsa_csv = "./exports_rsa/rsa_results.csv"
noise_ceiling_csv = "./exports_noise_ceiling/noise_ceiling_results.csv"


# =============================================================================
# HELPERS
# =============================================================================
def fmt_hhmm(h):
    h24 = h % 24
    hh = int(np.floor(h24))
    mm = int(np.round((h24 - hh) * 60)) % 60
    return f"{hh:02d}:{mm:02d}"


def get_star_string(p):
    if not np.isfinite(p):
        return None
    if p < alpha_2:
        return "**"
    elif p < alpha_1:
        return "*"
    else:
        return None


# =============================================================================
# CHARGEMENT
# =============================================================================
df_rsa = pd.read_csv(rsa_csv)
df_nc = pd.read_csv(noise_ceiling_csv)

df = pd.merge(
    df_rsa,
    df_nc[["start_hour", "R2_noise_ceiling"]],
    on="start_hour",
    how="inner"
)


# =============================================================================
# ORDRE TEMPOREL
# =============================================================================
x_raw = df["center_hour_mod"].to_numpy()
x = ((x_raw + 12) % 24) - 12
order = np.argsort(x)

df = df.iloc[order].reset_index(drop=True)
x = x[order]


# =============================================================================
# DONNEES
# =============================================================================
y_r2 = df["R2_obs"].to_numpy()

ci_low = df["R2_ci_low"].to_numpy()
ci_high = df["R2_ci_high"].to_numpy()

yerr_low = np.maximum(0, y_r2 - ci_low)
yerr_high = np.maximum(0, ci_high - y_r2)

y_nc = df["R2_noise_ceiling"].to_numpy()
pvals = df[p_col].to_numpy()


# =============================================================================
# FIGURE AXE BRISE
# =============================================================================
fig, (ax_top, ax_bottom) = plt.subplots(
    2,
    1,
    sharex=True,
    figsize=(10, 6),
    gridspec_kw={"height_ratios": [1, 2]}
)

# plots
for ax in [ax_top, ax_bottom]:
    ax.plot(x, y_r2, marker="o", linewidth=1.8, label=r"$\rho^2$")

    ax.errorbar(
        x,
        y_r2,
        yerr=[yerr_low, yerr_high],
        fmt="none",
        capsize=4
    )

    ax.plot(x, y_nc, linestyle="--", linewidth=2, label="Noise ceiling")


# =============================================================================
# AJOUT DES ETOILES DE SIGNIFICATIVITE
# =============================================================================
for xi, yi, yeh, p in zip(x, y_r2, yerr_high, pvals):
    stars = get_star_string(p)
    if stars is None:
        continue

    y_star = yi + yeh

    # choisir l'axe où placer l'étoile
    if 0 <= y_star <= y_low_max:
        ax = ax_bottom
        y_text = y_star + star_offset_low
    elif y_high_min <= y_star <= y_high_max:
        ax = ax_top
        y_text = y_star + star_offset_high
    else:
        # si le point tombe dans la cassure, on le place sur l'axe du haut
        ax = ax_top
        y_text = max(y_high_min + 0.01, y_high_min + star_offset_high)

    ax.text(
        xi,
        y_text,
        stars,
        ha="center",
        va="bottom",
        fontsize=STAR_SIZE,
        fontweight="bold"
    )


# =============================================================================
# LIMITES AXE
# =============================================================================
ax_bottom.set_ylim(0, y_low_max)
ax_top.set_ylim(y_high_min, y_high_max)

# cacher les spines
ax_top.spines["bottom"].set_visible(False)
ax_bottom.spines["top"].set_visible(False)

ax_top.tick_params(labeltop=False, labelsize=TICK_SIZE)
ax_bottom.tick_params(axis="both", labelsize=TICK_SIZE)
ax_top.tick_params(axis="y", labelsize=TICK_SIZE)
ax_bottom.xaxis.tick_bottom()

# marques de cassure
d = 0.5
kwargs = dict(
    marker=[(-1, -d), (1, d)],
    markersize=12,
    linestyle="none",
    color="k",
    mec="k",
    mew=1
)

ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kwargs)
ax_bottom.plot([0, 1], [1, 1], transform=ax_bottom.transAxes, **kwargs)


# =============================================================================
# AXE X
# =============================================================================
ticks = np.arange(-12, 13, 3)
ax_bottom.set_xticks(ticks)
ax_bottom.set_xticklabels([fmt_hhmm(t) for t in ticks])
ax_bottom.set_xlim(-12, 12)
ax_top.legend(
    fontsize=LEGEND_SIZE,
    loc="center right",
    frameon=True
)

# =============================================================================
# LABELS
# =============================================================================
ax_bottom.set_xlabel("Time (hour of day)")
ax_bottom.set_ylabel(r"$\rho^2$")
ax_top.set_ylabel(r"$\rho^2$")
ax_top.legend(fontsize=LEGEND_SIZE)
ax_top.legend(
    fontsize=LEGEND_SIZE,
    loc="center right",
    frameon=True
)
plt.tight_layout()

output_dir = './'
base_name = 'Figure_2'

out_eps = os.path.join(output_dir, f"{base_name}.eps")
plt.savefig(out_eps, format="eps", dpi=300, bbox_inches="tight")
print(f"Export figure : {out_eps}")

out_png = os.path.join(output_dir, f"{base_name}.png")
plt.savefig(out_png, format="png", dpi=600, bbox_inches="tight")
print(f"Export figure : {out_png}")

plt.show()