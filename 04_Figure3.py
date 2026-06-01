import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import random
from scipy.stats import pearsonr, spearmanr, ttest_rel
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests
from itertools import combinations
from pathlib import Path

np.random.seed(44)
random.seed(44)


# ============================================================
# OUTPUT LOCATION
# ============================================================
# All figure exports are written next to this script, even if the
# script is launched from another working directory.
SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

# EPS export: keep fonts editable/vector-friendly when possible.
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["pdf.fonttype"] = 42

# Larger fonts for publication-quality figures.
FONT_SIZE_BASE = 18
FONT_SIZE_LABEL = 20
FONT_SIZE_TICK = 16
FONT_SIZE_LEGEND = 16
FONT_SIZE_HEATMAP_ANNOT = 16
FONT_SIZE_STAR = 22
FONT_SIZE_COLORBAR = 18

plt.rcParams.update({
    "font.size": FONT_SIZE_BASE,
    "axes.labelsize": FONT_SIZE_LABEL,
    "axes.titlesize": FONT_SIZE_LABEL,
    "xtick.labelsize": FONT_SIZE_TICK,
    "ytick.labelsize": FONT_SIZE_TICK,
    "legend.fontsize": FONT_SIZE_LEGEND,
    "legend.title_fontsize": FONT_SIZE_LEGEND,
})

# ============================================================
# PARAMETERS
# ============================================================
CSV_PATH = "./data.csv"

N_REPETITIONS = 100
MIN_VALID_DAYS = 4
MAX_DAYS_PER_ZONE = 6

# Choices:
# "pearson"
# "spearman"
# "cosine"
# "euclidean_similarity"
SIMILARITY_METRIC = "spearman"

NEAR_DISTANCES = [10, 20, 40]
FAR_DISTANCES = [100, 200, 300]

TIME_WINDOWS = [
    (4, 22, "4-22h"),
    (22, 4, "22-4h")
]

features_cols = [
    "BGNf", "ACTtMean", "ECU", "EAS", "MFC", "KURTf"
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def build_time_mask(hours, start, end):
    """
    Select hours in the interval [start, end[
    Handles windows crossing midnight, e.g. 22 -> 4
    """
    if start < end:
        return (hours >= start) & (hours < end)
    elif start > end:
        return (hours >= start) | (hours < end)
    else:
        return pd.Series(True, index=hours.index)


def build_analysis_date(date_series, hour_series, start, end):
    dates = pd.to_datetime(date_series, errors="raise").dt.normalize()

    if start > end:
        mask_early_morning = hour_series < end
        dates.loc[mask_early_morning] = dates.loc[mask_early_morning] - pd.Timedelta(days=1)

    return dates


def safe_pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.nan
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan

    r, _ = pearsonr(x, y)
    return r if np.isfinite(r) else np.nan


def safe_spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.nan
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan

    r, _ = spearmanr(x, y)
    return r if np.isfinite(r) else np.nan


def safe_cosine_similarity(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 1:
        return np.nan

    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)

    if norm_x == 0 or norm_y == 0:
        return np.nan

    sim = np.dot(x, y) / (norm_x * norm_y)
    return sim if np.isfinite(sim) else np.nan


def safe_euclidean_similarity(x, y):
    """
    Converts Euclidean distance into similarity in ]0, 1]
    similarity = 1 / (1 + distance)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 1:
        return np.nan

    d = np.linalg.norm(x - y)
    sim = 1.0 / (1.0 + d)
    return sim if np.isfinite(sim) else np.nan


def compute_similarity(x, y, metric="pearson"):
    metric = metric.lower()

    if metric == "pearson":
        return safe_pearson(x, y)
    elif metric == "spearman":
        return safe_spearman(x, y)
    elif metric == "cosine":
        return safe_cosine_similarity(x, y)
    elif metric == "euclidean_similarity":
        return safe_euclidean_similarity(x, y)
    else:
        raise ValueError(
            "SIMILARITY_METRIC must be one of: "
            "'pearson', 'spearman', 'cosine', 'euclidean_similarity'"
        )


def compute_pairwise_mean_similarity(df_rep_scaled, features_cols, metric="pearson"):
    """
    For each unique pair of distances to edge,
    compute the mean similarity between soundscapes:
    mean of all similarities between all profile pairs
    belonging to the two distances.
    """
    distances = sorted(df_rep_scaled["Distance_lisiere"].dropna().unique())
    rows = []

    for i, d1 in enumerate(distances):
        X1 = df_rep_scaled.loc[
            df_rep_scaled["Distance_lisiere"] == d1, features_cols
        ].to_numpy()

        for d2 in distances[i + 1:]:
            X2 = df_rep_scaled.loc[
                df_rep_scaled["Distance_lisiere"] == d2, features_cols
            ].to_numpy()

            if len(X1) == 0 or len(X2) == 0:
                rows.append((d1, d2, np.nan, np.nan, 0))
                continue

            sims = []
            for x in X1:
                for y in X2:
                    sim = compute_similarity(x, y, metric=metric)
                    if np.isfinite(sim):
                        sims.append(sim)

            sims = np.asarray(sims, dtype=float)

            if len(sims) == 0:
                rows.append((d1, d2, np.nan, np.nan, 0))
            else:
                rows.append((d1, d2, np.mean(sims), np.std(sims), len(sims)))

    return pd.DataFrame(
        rows,
        columns=[
            "distance_1",
            "distance_2",
            "Mean_similarity",
            "SD_similarity_within_rep",
            "N_profile_pairs"
        ]
    )


def summarize_distance_groups(rep_pairwise, near_distances, far_distances):
    """
    For one repetition, summarize:
    - mean similarity within the near group
    - mean similarity within the far group
    - mean similarity between the two groups
    """
    rows = []

    group_defs = {
        "10-20-40 m": "near",
        "100-200-300 m": "far",
        "10-20-40 m vs 100-200-300 m": "between"
    }

    for group_label, group_type in group_defs.items():

        if group_type == "near":
            mask = (
                rep_pairwise["distance_1"].isin(near_distances)
                & rep_pairwise["distance_2"].isin(near_distances)
            )

        elif group_type == "far":
            mask = (
                rep_pairwise["distance_1"].isin(far_distances)
                & rep_pairwise["distance_2"].isin(far_distances)
            )

        elif group_type == "between":
            mask = (
                (
                    rep_pairwise["distance_1"].isin(near_distances)
                    & rep_pairwise["distance_2"].isin(far_distances)
                ) |
                (
                    rep_pairwise["distance_1"].isin(far_distances)
                    & rep_pairwise["distance_2"].isin(near_distances)
                )
            )

        sub = rep_pairwise.loc[mask, "Mean_similarity"].dropna()

        if len(sub) == 0:
            mean_sim = np.nan
            n_pairs = 0
        else:
            mean_sim = sub.mean()
            n_pairs = len(sub)

        rows.append({
            "Group": group_label,
            "Mean_similarity": mean_sim,
            "N_pairs_used": n_pairs
        })

    return pd.DataFrame(rows)


def build_similarity_matrix(df_summary, distances, diag_value=np.nan):
    """
    Build a symmetric similarity matrix between distances.
    """
    sim_matrix = pd.DataFrame(
        np.nan,
        index=distances,
        columns=distances,
        dtype=float
    )

    for d in distances:
        sim_matrix.loc[d, d] = diag_value

    for _, row in df_summary.iterrows():
        d1 = row["distance_1"]
        d2 = row["distance_2"]
        sim = row["Mean_similarity"]

        sim_matrix.loc[d1, d2] = sim
        sim_matrix.loc[d2, d1] = sim

    return sim_matrix


def build_dissimilarity_matrix(sim_matrix, metric="pearson"):
    """
    Convert the final similarity matrix into a dissimilarity matrix.
    """
    metric = metric.lower()
    sim = sim_matrix.copy().astype(float)
    sim_array = sim.to_numpy(copy=True)

    if metric in ["pearson", "spearman"]:
        sim_array = np.clip(sim_array, -1.0, 1.0)
        diss_array = 1.0 - sim_array

    elif metric == "cosine":
        sim_array = np.clip(sim_array, -1.0, 1.0)
        diss_array = np.arccos(sim_array) / np.pi

    elif metric == "euclidean_similarity":
        valid = np.isfinite(sim_array) & (sim_array > 0)
        diss_array = np.full_like(sim_array, np.nan, dtype=float)
        diss_array[valid] = (1.0 / sim_array[valid]) - 1.0

    else:
        raise ValueError(
            "SIMILARITY_METRIC must be one of: "
            "'pearson', 'spearman', 'cosine', 'euclidean_similarity'"
        )

    diss_array = (diss_array + diss_array.T) / 2.0
    np.fill_diagonal(diss_array, 0.0)

    return pd.DataFrame(diss_array, index=sim.index, columns=sim.columns)


# ============================================================
# STATS
# ============================================================
def pvalue_to_stars(p):
    if pd.isna(p):
        return "n.s."
    elif p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "n.s."


def run_two_way_rm_anova(df_boxplot):
    """
    Two-way repeated-measures ANOVA:
    within-subject factors = Window x Group
    subject = Repetition
    """
    df_anova = df_boxplot[["Repetition", "Window", "Group", "Mean_similarity"]].dropna().copy()

    aov = AnovaRM(
        data=df_anova,
        depvar="Mean_similarity",
        subject="Repetition",
        within=["Window", "Group"]
    ).fit()

    anova_table = aov.anova_table.copy()
    anova_table["Effect"] = anova_table.index
    anova_table = anova_table.reset_index(drop=True)

    cols = ["Effect"] + [c for c in anova_table.columns if c != "Effect"]
    anova_table = anova_table[cols]

    return aov, anova_table


def posthoc_time_within_group(df_boxplot, group_order, window_order, correction="holm"):
    """
    Post hoc paired comparisons of time period within each group.
    These are the comparisons displayed as stars on the figure.
    """
    results = []

    for group in group_order:
        sub = df_boxplot[df_boxplot["Group"] == group].copy()

        pivot = sub.pivot_table(
            index="Repetition",
            columns="Window",
            values="Mean_similarity",
            aggfunc="mean"
        )

        pivot = pivot.dropna(subset=window_order)

        if len(pivot) < 2:
            t_stat = np.nan
            p_unc = np.nan
            n_pairs = len(pivot)
            mean_1 = np.nan
            mean_2 = np.nan
        else:
            t_stat, p_unc = ttest_rel(
                pivot[window_order[0]],
                pivot[window_order[1]],
                nan_policy="omit"
            )
            n_pairs = len(pivot)
            mean_1 = pivot[window_order[0]].mean()
            mean_2 = pivot[window_order[1]].mean()

        results.append({
            "Comparison_type": "Time period within group",
            "Group": group,
            "Comparison": f"{window_order[0]} vs {window_order[1]}",
            "N_pairs": n_pairs,
            f"Mean_{window_order[0]}": mean_1,
            f"Mean_{window_order[1]}": mean_2,
            "t_stat": t_stat,
            "p_uncorrected": p_unc
        })

    results = pd.DataFrame(results)

    valid_mask = results["p_uncorrected"].notna()
    results["p_adjusted"] = np.nan
    results["Reject_H0"] = False
    results["Stars"] = "n.s."

    if valid_mask.sum() > 0:
        reject, p_adj, _, _ = multipletests(
            results.loc[valid_mask, "p_uncorrected"],
            method=correction
        )
        results.loc[valid_mask, "p_adjusted"] = p_adj
        results.loc[valid_mask, "Reject_H0"] = reject
        results.loc[valid_mask, "Stars"] = [pvalue_to_stars(p) for p in p_adj]

    return results


def posthoc_group_within_time(df_boxplot, group_order, window_order, correction="holm"):
    """
    Post hoc paired comparisons of distance groups within each time period.
    """
    results = []

    for window in window_order:
        sub = df_boxplot[df_boxplot["Window"] == window].copy()

        pivot = sub.pivot_table(
            index="Repetition",
            columns="Group",
            values="Mean_similarity",
            aggfunc="mean"
        )

        for g1, g2 in combinations(group_order, 2):
            if g1 not in pivot.columns or g2 not in pivot.columns:
                results.append({
                    "Comparison_type": "Group within time period",
                    "Window": window,
                    "Group_1": g1,
                    "Group_2": g2,
                    "N_pairs": 0,
                    f"Mean_{g1}": np.nan,
                    f"Mean_{g2}": np.nan,
                    "t_stat": np.nan,
                    "p_uncorrected": np.nan
                })
                continue

            pair = pivot[[g1, g2]].dropna()

            if len(pair) < 2:
                t_stat = np.nan
                p_unc = np.nan
                n_pairs = len(pair)
                mean_1 = np.nan
                mean_2 = np.nan
            else:
                t_stat, p_unc = ttest_rel(pair[g1], pair[g2], nan_policy="omit")
                n_pairs = len(pair)
                mean_1 = pair[g1].mean()
                mean_2 = pair[g2].mean()

            results.append({
                "Comparison_type": "Group within time period",
                "Window": window,
                "Group_1": g1,
                "Group_2": g2,
                "N_pairs": n_pairs,
                f"Mean_{g1}": mean_1,
                f"Mean_{g2}": mean_2,
                "t_stat": t_stat,
                "p_uncorrected": p_unc
            })

    results = pd.DataFrame(results)

    results["p_adjusted"] = np.nan
    results["Reject_H0"] = False
    results["Stars"] = "n.s."

    for window in window_order:
        mask = (results["Window"] == window) & results["p_uncorrected"].notna()
        if mask.sum() > 0:
            reject, p_adj, _, _ = multipletests(
                results.loc[mask, "p_uncorrected"],
                method=correction
            )
            results.loc[mask, "p_adjusted"] = p_adj
            results.loc[mask, "Reject_H0"] = reject
            results.loc[mask, "Stars"] = [pvalue_to_stars(p) for p in p_adj]

    return results


def add_significance_annotations(ax, df_boxplot, posthoc_time_results, group_order):
    """
    Add brackets + stars for the time-period comparison within each group.
    Stars correspond to Holm-adjusted p-values.
    """
    y_min = df_boxplot["Mean_similarity"].min()
    y_max = df_boxplot["Mean_similarity"].max()
    y_range = y_max - y_min

    if not np.isfinite(y_range) or y_range == 0:
        y_range = 1.0

    offset = 0.2

    for i, group in enumerate(group_order):
        row = posthoc_time_results[posthoc_time_results["Group"] == group]
        if row.empty:
            continue

        stars = row["Stars"].iloc[0]

        sub = df_boxplot[df_boxplot["Group"] == group]
        if sub.empty:
            continue

        local_max = sub["Mean_similarity"].max()
        y = local_max + 0.08 * y_range
        h = 0.03 * y_range

        x1 = i - offset
        x2 = i + offset

        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c="black")
        ax.text(
            (x1 + x2) / 2,
            y + h + 0.01 * y_range,
            stars,
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE_STAR
        )

    ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.22 * y_range)


# ============================================================
# FINAL COMBINED FIGURE
# ============================================================
def plot_combined_figure(sim_matrices_dict, df_boxplot, metric, group_order, window_order, posthoc_time_results):
    """
    Final figure:
    - first row: 2 similarity matrices
    - second row: boxplots
    - no titles
    - stars based on post hoc comparisons of time period within each group
    """
    all_vals = []
    for mat in sim_matrices_dict.values():
        vals = mat.to_numpy().astype(float).ravel()
        vals = vals[np.isfinite(vals)]
        all_vals.extend(vals)

    if len(all_vals) == 0:
        raise ValueError("No valid values found for the heatmaps.")

    common_vmin = np.min(all_vals)
    common_vmax = np.max(all_vals)

    fig = plt.figure(figsize=(19, 13))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.1],
        hspace=0.35,
        wspace=0.25
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    # Heatmap 1
    sns.heatmap(
        sim_matrices_dict[window_order[0]],
        annot=True,
        cmap="viridis",
        fmt=".2f",
        linewidths=0.5,
        vmin=common_vmin,
        vmax=common_vmax,
        cbar=False,
        annot_kws={"size": FONT_SIZE_HEATMAP_ANNOT},
        ax=ax1
    )
    ax1.set_title("")
    ax1.set_xlabel("Distance to edge (m)", fontsize=FONT_SIZE_LABEL)
    ax1.set_ylabel("Distance to edge (m)", fontsize=FONT_SIZE_LABEL)
    ax1.tick_params(axis="both", labelsize=FONT_SIZE_TICK)

    # Heatmap 2
    hm2 = sns.heatmap(
        sim_matrices_dict[window_order[1]],
        annot=True,
        cmap="viridis",
        fmt=".2f",
        linewidths=0.5,
        vmin=common_vmin,
        vmax=common_vmax,
        cbar=False,
        annot_kws={"size": FONT_SIZE_HEATMAP_ANNOT},
        ax=ax2
    )
    ax2.set_title("")
    ax2.set_xlabel("Distance to edge (m)", fontsize=FONT_SIZE_LABEL)
    ax2.set_ylabel("Distance to edge (m)", fontsize=FONT_SIZE_LABEL)
    ax2.tick_params(axis="both", labelsize=FONT_SIZE_TICK)

    # Shared colorbar
    mappable = hm2.collections[0]
    cbar = fig.colorbar(
        mappable,
        ax=[ax1, ax2],
        orientation="vertical",
        fraction=0.025,
        pad=0.02
    )
    cbar.set_label(f"Mean similarity ({metric})", fontsize=FONT_SIZE_COLORBAR)
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICK)

    # Boxplots
    sns.boxplot(
        data=df_boxplot,
        x="Group",
        y="Mean_similarity",
        hue="Window",
        order=group_order,
        hue_order=window_order,
        ax=ax3
    )

    sns.stripplot(
        data=df_boxplot,
        x="Group",
        y="Mean_similarity",
        hue="Window",
        dodge=True,
        order=group_order,
        hue_order=window_order,
        alpha=0.55,
        size=4,
        ax=ax3
    )

    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend(
        handles[:len(window_order)],
        labels[:len(window_order)],
        title="Time period",
        fontsize=FONT_SIZE_LEGEND,
        title_fontsize=FONT_SIZE_LEGEND
    )

    ax3.set_xlabel("Distance-to-edge group", fontsize=FONT_SIZE_LABEL)
    ax3.set_ylabel(f"Mean similarity ({metric})", fontsize=FONT_SIZE_LABEL)
    ax3.set_title("")
    ax3.tick_params(axis="x", rotation=0, labelsize=FONT_SIZE_TICK)
    ax3.tick_params(axis="y", labelsize=FONT_SIZE_TICK)

    # Stars from adjusted post hoc tests
    add_significance_annotations(ax3, df_boxplot, posthoc_time_results, group_order)

    plt.tight_layout()

    out_fig = SCRIPT_DIR / "Figure_3.eps"
    plt.savefig(out_fig, format="eps", dpi=300, bbox_inches="tight")
    print(f"Export figure finale EPS : {out_fig}")

    out_fig = SCRIPT_DIR / "Figure_3.png"
    plt.savefig(out_fig, format="png", dpi=600, bbox_inches="tight")
    print(f"Export figure finale PNG : {out_fig}")

    plt.show()


# ============================================================
# PREPARE DATA FOR EACH TIME WINDOW
# ============================================================
def prepare_window_dataframe(df_raw, start_hour, end_hour, window_label):
    df = df_raw.copy()

    time_mask = build_time_mask(df["Heure_num"], start_hour, end_hour)
    df = df[time_mask].copy()

    if df.empty:
        raise ValueError(f"No data found in time window {window_label}.")

    df["Date_analyse"] = build_analysis_date(
        df["Date"],
        df["Heure_num"],
        start_hour,
        end_hour
    )

    df["Intervalle"] = window_label
    return df


def run_window_analysis(df_raw, start_hour, end_hour, window_label,
                        features_cols, n_repetitions,
                        min_valid_days, max_days_per_zone,
                        metric, near_distances, far_distances):
    """
    Run the full analysis for one time window and return:
    - aggregated soundscapes
    - pairwise similarities by repetition
    - group summary by repetition
    - global pairwise summary
    """
    df = prepare_window_dataframe(df_raw, start_hour, end_hour, window_label)
    zone_to_distance = df.groupby("Zone")["Distance_lisiere"].agg(lambda x: x.mode()[0]).to_dict()

    all_rep_pairwise = []
    all_rep_soundscapes = []
    all_group_summaries = []

    for rep in range(n_repetitions):
        seed_value = 44 + rep
        np.random.seed(seed_value)
        random.seed(seed_value)

        results = []
        selected_zones = df["Zone"].dropna().unique()

        for zone in selected_zones:
            df_zone = df[df["Zone"] == zone].copy()
            valid_days = sorted(df_zone["Date_analyse"].dropna().unique())

            if len(valid_days) < min_valid_days:
                print(
                    f"[{window_label}] Zone {zone} ignored at repetition {rep + 1} "
                    f"(valid days = {len(valid_days)})"
                )
                continue

            n_days = max_days_per_zone if len(valid_days) >= max_days_per_zone else min_valid_days
            selected_days = np.random.choice(valid_days, size=n_days, replace=False)

            for date in selected_days:
                df_day = df_zone[df_zone["Date_analyse"] == date].copy()

                if df_day.empty:
                    continue

                mean_vals = df_day[features_cols].mean()

                result = {
                    "Zone": zone,
                    "Date_analyse": pd.to_datetime(date),
                    "Intervalle": window_label,
                    "Distance_lisiere": zone_to_distance[zone],
                    "Repetition": rep + 1
                }
                result.update(mean_vals.to_dict())
                results.append(result)

        df_rep = pd.DataFrame(results)

        if df_rep.empty:
            continue

        scaler = StandardScaler()
        df_rep_scaled = df_rep.copy()
        df_rep_scaled[features_cols] = scaler.fit_transform(df_rep[features_cols])

        all_rep_soundscapes.append(df_rep_scaled)

        rep_pairwise = compute_pairwise_mean_similarity(
            df_rep_scaled,
            features_cols,
            metric=metric
        )
        rep_pairwise["Repetition"] = rep + 1
        rep_pairwise["Window"] = window_label
        all_rep_pairwise.append(rep_pairwise)

        rep_group_summary = summarize_distance_groups(
            rep_pairwise,
            near_distances=near_distances,
            far_distances=far_distances
        )
        rep_group_summary["Repetition"] = rep + 1
        rep_group_summary["Window"] = window_label
        all_group_summaries.append(rep_group_summary)

    if len(all_group_summaries) == 0:
        raise ValueError(f"No valid results for time window {window_label}.")

    df_soundscapes_all = (
        pd.concat(all_rep_soundscapes, ignore_index=True)
        if len(all_rep_soundscapes) > 0
        else pd.DataFrame()
    )
    df_pairwise_all = (
        pd.concat(all_rep_pairwise, ignore_index=True)
        if len(all_rep_pairwise) > 0
        else pd.DataFrame()
    )
    df_group_summary = pd.concat(all_group_summaries, ignore_index=True)

    df_pairwise_summary = (
        df_pairwise_all
        .groupby(["distance_1", "distance_2"], as_index=False)
        .agg(
            Mean_similarity=("Mean_similarity", "mean"),
            SD_similarity_across_reps=("Mean_similarity", "std"),
            Mean_n_profile_pairs=("N_profile_pairs", "mean"),
            N_valid_reps=("Repetition", "nunique")
        )
    )

    return df_soundscapes_all, df_pairwise_all, df_group_summary, df_pairwise_summary


# ============================================================
# LOAD AND PREPARE RAW DATA
# ============================================================
df_raw = pd.read_csv(CSV_PATH, sep=",")

df_raw["Distance_lisiere"] = pd.to_numeric(
    df_raw["Distance_lisiere"].astype(str).str.extract(r"(\d+)")[0],
    errors="coerce"
)

df_raw = df_raw[df_raw["rainy (1 = rainy, 0 = not rainy)"] == 0].copy()

df_raw["Heure_num"] = (
    df_raw["Heure"]
    .astype(str)
    .str.zfill(6)
    .str[:2]
    .astype(int)
    % 24
)

df_raw["Date_parsed"] = pd.to_datetime(df_raw["Date"], errors="coerce", dayfirst=True)
bad_dates = df_raw["Date_parsed"].isna()

if bad_dates.any():
    print("Removed invalid dates:")
    print(df_raw.loc[bad_dates, "Date"].astype(str).unique()[:20])

df_raw = df_raw.loc[~bad_dates].copy()
df_raw["Date"] = df_raw["Date_parsed"]
df_raw.drop(columns=["Date_parsed"], inplace=True)

all_distances_raw = sorted(df_raw["Distance_lisiere"].dropna().unique())
print(f"Available distances in CSV: {all_distances_raw}")
print(f"Similarity metric: {SIMILARITY_METRIC}")


# ============================================================
# ANALYSIS FOR BOTH TIME WINDOWS
# ============================================================
all_soundscapes = []
all_pairwise = []
all_group_boxes = []
all_pairwise_summaries = []

sim_matrices_by_window = {}
dissim_matrices_by_window = {}

for start_hour, end_hour, window_label in TIME_WINDOWS:
    print(f"\n===== Analysis for time window {window_label} =====")

    (
        df_soundscapes_window,
        df_pairwise_window,
        df_group_window,
        df_pairwise_summary_window
    ) = run_window_analysis(
        df_raw=df_raw,
        start_hour=start_hour,
        end_hour=end_hour,
        window_label=window_label,
        features_cols=features_cols,
        n_repetitions=N_REPETITIONS,
        min_valid_days=MIN_VALID_DAYS,
        max_days_per_zone=MAX_DAYS_PER_ZONE,
        metric=SIMILARITY_METRIC,
        near_distances=NEAR_DISTANCES,
        far_distances=FAR_DISTANCES
    )

    all_soundscapes.append(df_soundscapes_window)
    all_pairwise.append(df_pairwise_window)
    all_group_boxes.append(df_group_window)

    df_pairwise_summary_window["Window"] = window_label
    all_pairwise_summaries.append(df_pairwise_summary_window)

    distances_window = sorted(df_soundscapes_window["Distance_lisiere"].dropna().unique())
    sim_matrix_window = build_similarity_matrix(
        df_pairwise_summary_window,
        distances_window,
        diag_value=np.nan
    )
    dissim_matrix_window = build_dissimilarity_matrix(
        sim_matrix_window,
        metric=SIMILARITY_METRIC
    )

    sim_matrices_by_window[window_label] = sim_matrix_window
    dissim_matrices_by_window[window_label] = dissim_matrix_window

    print(f"\nPairwise summary for {window_label}:")
    print(df_pairwise_summary_window)



# ============================================================
# CONCATENATE RESULTS
# ============================================================
df_soundscapes_all = pd.concat(all_soundscapes, ignore_index=True)
df_pairwise_all = pd.concat(all_pairwise, ignore_index=True)
df_boxplot = pd.concat(all_group_boxes, ignore_index=True)
df_pairwise_summaries_all = pd.concat(all_pairwise_summaries, ignore_index=True)

group_order = [
    "10-20-40 m",
    "100-200-300 m",
    "10-20-40 m vs 100-200-300 m"
]
window_order = ["4-22h", "22-4h"]

df_boxplot["Group"] = pd.Categorical(
    df_boxplot["Group"],
    categories=group_order,
    ordered=True
)
df_boxplot["Window"] = pd.Categorical(
    df_boxplot["Window"],
    categories=window_order,
    ordered=True
)

print("\nSummary of values used for boxplots:")
print(
    df_boxplot.groupby(["Group", "Window"], observed=False)["Mean_similarity"]
    .agg(["count", "mean", "std", "min", "max"])
    .reset_index()
)


# ============================================================
# TWO-WAY REPEATED-MEASURES ANOVA
# ============================================================
anova_model, anova_table = run_two_way_rm_anova(df_boxplot)

print("\n===== TWO-WAY REPEATED-MEASURES ANOVA =====")
print("Within-subject factors: Time period x Distance-to-edge group")
print(anova_table)



# ============================================================
# POST HOC TESTS
# ============================================================
posthoc_time_results = posthoc_time_within_group(
    df_boxplot=df_boxplot,
    group_order=group_order,
    window_order=window_order,
    correction="holm"
)

print("\n===== POST HOC: TIME PERIOD WITHIN EACH DISTANCE GROUP (Holm-corrected) =====")
print(posthoc_time_results)



posthoc_group_results = posthoc_group_within_time(
    df_boxplot=df_boxplot,
    group_order=group_order,
    window_order=window_order,
    correction="holm"
)

print("\n===== POST HOC: DISTANCE GROUP COMPARISONS WITHIN EACH TIME PERIOD (Holm-corrected) =====")
print(posthoc_group_results)



# ============================================================
# FINAL COMBINED FIGURE
# ============================================================
plot_combined_figure(
    sim_matrices_dict=sim_matrices_by_window,
    df_boxplot=df_boxplot,
    metric=SIMILARITY_METRIC,
    group_order=group_order,
    window_order=window_order,
    posthoc_time_results=posthoc_time_results
)



print("\nDone.")
