import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

from joblib import Parallel, delayed


# =============================================================================
# 0) PARAMÈTRES
# =============================================================================
csv_path = "./data.csv"

representation_mode = "all"   # "all" ou "pca"

# PCA (si representation_mode == "pca")
pca_n_components = 3
pca_whiten = True

# z-score global sur toutes les observations conservées
zscore_features_globally = True

# métrique pour les RDMs observées
obs_rdm_metric = "correlation"   # "correlation", "euclidean", "seuclidean", ...
seuclidean_variance_eps = 1e-12

# fenêtres temporelles
window_hours = 3
step_hours = 0.5

# seuils qualité
min_zones = 5
min_days = 2
min_common_zones_noise = 3

# noise ceiling leave-zone-out
N_drop = 3
n_trials = 100   # nombre de tirages bootstrap avec remise par paire de jours

# parallélisation
n_jobs = -1   # -1 = tous les coeurs

# seed
seed = 0

# exports génériques
output_dir = "./exports_noise_ceiling"
base_name = "noise_ceiling_results"


# =============================================================================
# 1) HELPERS
# =============================================================================
def hhmmss_to_hour_decimal(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.zfill(6)
    hh = s.str[:2].astype(int)
    mm = s.str[2:4].astype(int)
    ss = s.str[4:6].astype(int)
    return hh + mm / 60 + ss / 3600


def upper_tri_vec(M: np.ndarray) -> np.ndarray:
    tri = np.triu_indices_from(M, k=1)
    return M[tri]


def _safe_squareform_pdist(X: np.ndarray, metric: str, V=None):
    if X.ndim != 2:
        raise ValueError("X doit être de dimension 2.")
    if X.shape[0] < 2:
        raise ValueError("Il faut au moins 2 observations pour pdist.")

    if metric == "seuclidean":
        if V is None:
            raise ValueError("metric='seuclidean' exige V.")
        return squareform(pdist(X, metric=metric, V=V))
    return squareform(pdist(X, metric=metric))


def fmt_hhmm(h):
    h24 = h % 24
    hh = int(np.floor(h24))
    mm = int(np.round((h24 - hh) * 60)) % 60
    return f"{hh:02d}:{mm:02d}"


def get_df_bin_for_window(df_time: pd.DataFrame, start: float, window_hours: float) -> pd.DataFrame:
    end = start + window_hours
    if end <= 24:
        mask = (df_time["hour_decimal"] >= start) & (df_time["hour_decimal"] < end)
    else:
        mask = (df_time["hour_decimal"] >= start) | (df_time["hour_decimal"] < (end - 24))
    return df_time.loc[mask].copy()


def compute_zone_features(df_sub: pd.DataFrame, obs_cols: list) -> pd.DataFrame:
    return df_sub.groupby("Zone")[obs_cols].median().dropna()


def build_rdm_from_zone_features(Z_feat: pd.DataFrame, obs_metric: str, V_obs=None):
    if Z_feat is None or len(Z_feat) < 3:
        return None

    X = Z_feat.values
    if obs_metric == "seuclidean":
        if V_obs is None:
            raise ValueError("V_obs est requis pour seuclidean.")
        return _safe_squareform_pdist(X, metric=obs_metric, V=V_obs)
    return _safe_squareform_pdist(X, metric=obs_metric)


def safe_spearman_rho(v1, v2):
    if len(v1) == 0 or len(v2) == 0:
        return np.nan
    if np.all(v1 == v1[0]) or np.all(v2 == v2[0]):
        return np.nan
    rho, _ = spearmanr(v1, v2, nan_policy="omit")
    return rho


# =============================================================================
# 2) NOISE CEILING PAR PAIRES DE JOURS + LEAVE-N-ZONES-OUT + TIRAGES AVEC REMISE
# =============================================================================
def noise_ceiling_by_day_pairs_leavezones(
    df_bin: pd.DataFrame,
    obs_cols: list,
    obs_metric: str,
    N_drop: int,
    n_trials: int,
    rng: np.random.Generator,
    min_common_zones: int = 3,
    min_days: int = 2,
    V_obs=None,
):
    """
    Pour une tranche horaire donnée:
      - construit les représentations par jour (Zone x features),
      - pour chaque paire de jours,
      - prend les zones communes,
      - effectue n_trials tirages leave-(N_drop)-zones-out AVEC REMISE,
      - calcule la corrélation Spearman entre les deux RDMs journalières,
      - retourne la moyenne globale des corrélations.

    Retour:
      rho_mean, rho_std, n_days_used, n_day_pairs_used, n_total_valid_trials, corrs_all
    """

    days = sorted(df_bin["Date"].dropna().astype(str).unique().tolist())
    if len(days) < min_days:
        return np.nan, np.nan, 0, 0, 0, []

    day_zone_feat = {}
    for d in days:
        df_d = df_bin[df_bin["Date"].astype(str) == d]
        Z_feat = compute_zone_features(df_d, obs_cols)
        if len(Z_feat) >= min_common_zones:
            day_zone_feat[d] = Z_feat

    used_days = sorted(day_zone_feat.keys())
    if len(used_days) < min_days:
        return np.nan, np.nan, len(used_days), 0, 0, []

    corrs_all = []
    n_pairs_used = 0
    n_total_valid_trials = 0

    for i in range(len(used_days)):
        for j in range(i + 1, len(used_days)):
            d1, d2 = used_days[i], used_days[j]
            Z1 = day_zone_feat[d1]
            Z2 = day_zone_feat[d2]

            common = Z1.index.intersection(Z2.index)
            n_common = len(common)

            if n_common < min_common_zones:
                continue

            keep_size = n_common - N_drop
            if keep_size < min_common_zones:
                continue

            common = np.array(common)
            pair_corrs = []

            for _ in range(n_trials):
                keep_zones = rng.choice(common, size=keep_size, replace=True)

                Z1_keep = Z1.loc[keep_zones]
                Z2_keep = Z2.loc[keep_zones]

                if len(Z1_keep) < min_common_zones or len(Z2_keep) < min_common_zones:
                    continue

                try:
                    if obs_metric == "seuclidean":
                        RDM1 = build_rdm_from_zone_features(Z1_keep, obs_metric, V_obs=V_obs)
                        RDM2 = build_rdm_from_zone_features(Z2_keep, obs_metric, V_obs=V_obs)
                    else:
                        RDM1 = build_rdm_from_zone_features(Z1_keep, obs_metric, V_obs=None)
                        RDM2 = build_rdm_from_zone_features(Z2_keep, obs_metric, V_obs=None)
                except Exception:
                    continue

                if RDM1 is None or RDM2 is None:
                    continue

                v1 = upper_tri_vec(RDM1)
                v2 = upper_tri_vec(RDM2)

                rho = safe_spearman_rho(v1, v2)

                if np.isfinite(rho):
                    pair_corrs.append(float(rho))

            if len(pair_corrs) > 0:
                corrs_all.extend(pair_corrs)
                n_pairs_used += 1
                n_total_valid_trials += len(pair_corrs)

    if len(corrs_all) == 0:
        return np.nan, np.nan, len(used_days), n_pairs_used, n_total_valid_trials, []

    corrs_all = np.asarray(corrs_all, dtype=float)
    rho_mean = float(np.nanmean(corrs_all))
    rho_std = float(np.nanstd(corrs_all, ddof=1)) if len(corrs_all) > 1 else 0.0

    return rho_mean, rho_std, len(used_days), n_pairs_used, n_total_valid_trials, corrs_all.tolist()


# =============================================================================
# 3) CHARGEMENT + PRÉTRAITEMENT
# =============================================================================
df_all = pd.read_csv(csv_path)

df_all = df_all[df_all["rainy (1 = rainy, 0 = not rainy)"] == 0].copy()
print("Nombre d'observations après filtrage pluie :", len(df_all))

df_all["Distance_lisiere_num"] = (
    df_all["Distance_lisiere"]
    .astype(str)
    .str.extract(r"(\d+)", expand=False)
    .astype(float)
)
df_all = df_all.dropna(subset=["Distance_lisiere_num"]).copy()

exclude_cols = {
    "Distance_lisiere_num",
    "Distance_lisiere",
    "Fichier",
    "Zone",
    "Heure",
    "jour/nuit",
    "Date",
    "Identifiant",
    "rainy (1 = rainy, 0 = not rainy)"
}

num_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in num_cols if c not in exclude_cols]
feature_cols = [c for c in feature_cols if df_all[c].nunique(dropna=True) > 1]

if len(feature_cols) < 1:
    raise ValueError("Aucune colonne numérique exploitable trouvée.")

X_global = df_all[feature_cols].copy()
X_global = X_global.fillna(X_global.median(numeric_only=True))

if zscore_features_globally:
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_global_z = scaler.fit_transform(X_global)

    for i, c in enumerate(feature_cols):
        df_all[c + "_z"] = X_global_z[:, i]

    feature_cols_used = [c + "_z" for c in feature_cols]
else:
    feature_cols_used = feature_cols

if representation_mode.lower() == "pca":
    X_for_pca = df_all[feature_cols_used].copy()
    X_for_pca = X_for_pca.fillna(X_for_pca.median(numeric_only=True))

    pca = PCA(n_components=pca_n_components, whiten=pca_whiten, random_state=0)
    PC = pca.fit_transform(X_for_pca.values)

    for k in range(pca_n_components):
        df_all[f"PC{k+1}"] = PC[:, k]

    obs_cols = [f"PC{k+1}" for k in range(pca_n_components)]
    print("Variance expliquée PCA :", pca.explained_variance_ratio_)
    print("Variance expliquée cumulée :", pca.explained_variance_ratio_.sum())

elif representation_mode.lower() == "all":
    obs_cols = feature_cols_used
    print(f"Mode ALL: {len(obs_cols)} features utilisées.")

else:
    raise ValueError("representation_mode doit être 'all' ou 'pca'.")

if obs_rdm_metric == "seuclidean":
    X_obs_global = df_all[obs_cols].copy()
    X_obs_global = X_obs_global.fillna(X_obs_global.median(numeric_only=True))
    V_obs = np.var(X_obs_global.values, axis=0, ddof=1)
    V_obs = np.maximum(V_obs, seuclidean_variance_eps)
else:
    V_obs = None

df_time = df_all.copy()
df_time["hour_decimal"] = hhmmss_to_hour_decimal(df_time["Heure"])


# =============================================================================
# 4) CALCUL PARALLÉLISÉ DU NOISE CEILING PAR TRANCHE HORAIRE
# =============================================================================
starts = np.arange(0, 24, step_hours).astype(float)


def process_window(start):
    end = start + window_hours
    label = f"{start:05.1f}-{(end % 24):05.1f}"

    df_bin = get_df_bin_for_window(df_time, start, window_hours)

    n_obs = len(df_bin)
    n_zones = df_bin["Zone"].nunique()

    if n_obs == 0 or n_zones < min_zones:
        result_row = {
            "window_label": label,
            "start_hour": float(start),
            "end_hour": float(end),
            "center_hour": float(start + window_hours / 2),
            "center_hour_mod": float((start + window_hours / 2) % 24),
            "n_obs": int(n_obs),
            "n_zones": int(n_zones),
            "rho_noise_ceiling": np.nan,
            "rho_noise_ceiling_std": np.nan,
            "R2_noise_ceiling": np.nan,
            "R2_noise_ceiling_std": np.nan,
            "n_days_used": 0,
            "n_day_pairs_used": 0,
            "n_total_valid_trials": 0,
        }
        return float(start), result_row, []

    rng_w = np.random.default_rng(seed + int(round(start * 1000)))

    rho_nc, rho_nc_std, n_days_used, n_pairs_used, n_total_valid_trials, corrs_all = (
        noise_ceiling_by_day_pairs_leavezones(
            df_bin=df_bin,
            obs_cols=obs_cols,
            obs_metric=obs_rdm_metric,
            N_drop=N_drop,
            n_trials=n_trials,
            rng=rng_w,
            min_common_zones=min_common_zones_noise,
            min_days=min_days,
            V_obs=V_obs,
        )
    )

    corrs_arr = np.asarray(corrs_all, dtype=float) if len(corrs_all) > 0 else np.array([], dtype=float)

    if len(corrs_arr) > 0:
        r2_vals = corrs_arr ** 2
        r2_mean_direct = float(np.mean(r2_vals))
        r2_std_direct = float(np.std(r2_vals, ddof=1)) if len(r2_vals) > 1 else 0.0
    else:
        r2_mean_direct = np.nan
        r2_std_direct = np.nan

    result_row = {
        "window_label": label,
        "start_hour": float(start),
        "end_hour": float(end),
        "center_hour": float(start + window_hours / 2),
        "center_hour_mod": float((start + window_hours / 2) % 24),
        "n_obs": int(n_obs),
        "n_zones": int(n_zones),
        "rho_noise_ceiling": float(rho_nc) if np.isfinite(rho_nc) else np.nan,
        "rho_noise_ceiling_std": float(rho_nc_std) if np.isfinite(rho_nc_std) else np.nan,
        "R2_noise_ceiling": float(r2_mean_direct) if np.isfinite(r2_mean_direct) else np.nan,
        "R2_noise_ceiling_std": float(r2_std_direct) if np.isfinite(r2_std_direct) else np.nan,
        "n_days_used": int(n_days_used),
        "n_day_pairs_used": int(n_pairs_used),
        "n_total_valid_trials": int(n_total_valid_trials),
    }

    detail_rows = []
    for k, rho_val in enumerate(corrs_all):
        detail_rows.append({
            "window_label": label,
            "start_hour": float(start),
            "end_hour": float(end),
            "center_hour": float(start + window_hours / 2),
            "center_hour_mod": float((start + window_hours / 2) % 24),
            "trial_index": int(k),
            "rho_noise_ceiling_trial": float(rho_val),
            "R2_noise_ceiling_trial": float(rho_val ** 2) if np.isfinite(rho_val) else np.nan,
        })

    return float(start), result_row, detail_rows


print(f"\nCalcul parallèle par fenêtre : n_jobs={n_jobs}")

out = Parallel(n_jobs=n_jobs, prefer="processes")(
    delayed(process_window)(start)
    for start in tqdm(starts, desc="Fenêtres")
)

out.sort(key=lambda x: x[0])

results = [o[1] for o in out]
details_nested = [o[2] for o in out]

df_results = pd.DataFrame(results).sort_values("center_hour").reset_index(drop=True)

detail_rows = []
for rows in details_nested:
    detail_rows.extend(rows)

df_noise_trials = pd.DataFrame(detail_rows)

print("\n=== Résultats noise ceiling ===")
print(df_results[
    [
        "window_label",
        "n_obs",
        "n_zones",
        "n_days_used",
        "n_day_pairs_used",
        "n_total_valid_trials",
        "rho_noise_ceiling",
        "rho_noise_ceiling_std",
        "R2_noise_ceiling",
        "R2_noise_ceiling_std",
    ]
])


# =============================================================================
# 5) EXPORT
# =============================================================================
os.makedirs(output_dir, exist_ok=True)

out_csv = os.path.join(output_dir, f"{base_name}.csv")
df_results.to_csv(out_csv, index=False)
print(f"\nExport CSV résumé : {out_csv}")

out_csv_trials = os.path.join(output_dir, f"{base_name}_TRIALS_LONG.csv")
print(f"Export CSV détails : {out_csv_trials}")


# =============================================================================
# 6) PLOT
# =============================================================================
x_raw = df_results["center_hour_mod"].values
y_nc = df_results["R2_noise_ceiling"].values
y_nc_std = df_results["R2_noise_ceiling_std"].values

x = ((x_raw + 12) % 24) - 12
order = np.argsort(x)
x = x[order]
y_nc = y_nc[order]
y_nc_std = y_nc_std[order]

plt.figure(figsize=(10, 5))

has_nc = np.isfinite(y_nc)
if np.any(has_nc):
    x_plot = x[has_nc]
    y_plot = y_nc[has_nc]
    y_std_plot = y_nc_std[has_nc]

    plt.plot(x_plot, y_plot, marker="o", linewidth=2, label="Noise ceiling")

    has_std = np.isfinite(y_std_plot)
    if np.any(has_std):
        plt.fill_between(
            x_plot[has_std],
            y_plot[has_std] - y_std_plot[has_std],
            y_plot[has_std] + y_std_plot[has_std],
            alpha=0.25,
            label="± 1 ET"
        )

plt.axvline(0)
plt.title(
    f"Noise ceiling par tranche horaire\n"
    f"fenêtre={window_hours}h | pas={step_hours}h | leave-{N_drop} zones | trials={n_trials}"
)
plt.xlabel("Heure (centrée sur minuit)")
plt.ylabel("R² noise ceiling")

ticks = np.arange(-12, 13, 3)
plt.xticks(ticks, [fmt_hhmm(t) for t in ticks])
plt.xlim(-12, 12)
plt.legend()
plt.tight_layout()

out_png = os.path.join(output_dir, f"{base_name}.png")
plt.savefig(out_png, dpi=300, bbox_inches="tight")
print(f"Export figure : {out_png}")

plt.show()