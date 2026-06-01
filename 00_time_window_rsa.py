import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr, t as t_dist

from joblib import Parallel, delayed


# =============================================================================
# 0) PARAMÈTRES
# =============================================================================
representation_mode = "all"   # "pca" ou "all"

# PCA
pca_n_components = 3
pca_whiten = True

# z-score global
zscore_features_globally = True

# métriques
obs_rdm_metric = "correlation"
model_rdm_metric = "euclidean"

# pour seuclidean
seuclidean_variance_eps = 1e-12

# fenêtres temporelles
window_hours = 3
step_hours = 0.5

# bootstrap leave-N zones
N_drop = 3
n_boot = 30 #500

# permutations RSA
n_perm = 500

# seuil qualité
min_zones = 5

# BUBBLES
n_bubble_trials = 200 
p_bubble = 0.20
bubble_score_mode = "rho"      # "rho" ou "r2"
bubble_revcorr_metric = "pearson"   # "spearman" ou "pearson"

# NULL DISTRIBUTION BUBBLES
n_bubble_null = 200
min_valid_null = 20

# filtrage simple
use_abs_weight_threshold = False
abs_weight_thresh = 0.05

# significativité tranches horaires
alpha_time = 0.05

# significativité feature-wise contre la nulle bubbles
alpha_feat = 0.05/60

# parallélisation
n_jobs = -1

# seed
seed = 0

# exports génériques
output_dir = "./exports_rsa"
base_name = "rsa_results"


# =============================================================================
# 1) CHARGER + FILTRER
# =============================================================================
df_all = pd.read_csv("./data.csv")
df_all = df_all[df_all["rainy (1 = rainy, 0 = not rainy)"] == 0].copy()
print("Nombre d'observations après filtrage pluie :", len(df_all))


# =============================================================================
# 2) DISTANCE LISIÈRE -> NUMÉRIQUE
# =============================================================================
df_all["Distance_lisiere_num"] = (
    df_all["Distance_lisiere"]
    .astype(str)
    .str.extract(r"(\d+)", expand=False)
    .astype(float)
)
df_all = df_all.dropna(subset=["Distance_lisiere_num"]).copy()


# =============================================================================
# 3) FEATURES NUMÉRIQUES
# =============================================================================
exclude_cols = {
    "Distance_lisiere_num",
    "Distance_lisiere",
    "Fichier", "Zone", "Heure", "jour/nuit", "Date",
    "Identifiant",
    "rainy (1 = rainy, 0 = not rainy)"
}

num_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in num_cols if c not in exclude_cols]
feature_cols = [c for c in feature_cols if df_all[c].nunique(dropna=True) > 1]

if len(feature_cols) < 1:
    raise ValueError("Aucune colonne numérique exploitable trouvée.")


# =============================================================================
# 4) PRÉTRAITEMENT GLOBAL
# =============================================================================
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


# =============================================================================
# 5) REPRÉSENTATION OBSERVÉE
# =============================================================================
if representation_mode.lower() == "pca":
    X_for_pca = df_all[feature_cols_used].copy()
    X_for_pca = X_for_pca.fillna(X_for_pca.median(numeric_only=True))

    pca = PCA(n_components=pca_n_components, random_state=0, whiten=pca_whiten)
    PC = pca.fit_transform(X_for_pca.values)

    for k in range(pca_n_components):
        df_all[f"PC{k+1}"] = PC[:, k]

    print(
        "Variance expliquée (PCA globale) :",
        pca.explained_variance_ratio_,
        "| cumul :",
        pca.explained_variance_ratio_.sum()
    )
    obs_cols = [f"PC{k+1}" for k in range(pca_n_components)]

elif representation_mode.lower() == "all":
    obs_cols = feature_cols_used
    print(f"Mode ALL: {len(obs_cols)} indices utilisés.")

else:
    raise ValueError("representation_mode doit être 'pca' ou 'all'.")


# =============================================================================
# 6) V POUR SEUCLIDEAN
# =============================================================================
if obs_rdm_metric == "seuclidean":
    X_obs_global = df_all[obs_cols].copy()
    X_obs_global = X_obs_global.fillna(X_obs_global.median(numeric_only=True))
    V_obs = np.var(X_obs_global.values, axis=0, ddof=1)
    V_obs = np.maximum(V_obs, seuclidean_variance_eps)
else:
    V_obs = None


# =============================================================================
# 7) HELPERS
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
    try:
        if metric == "seuclidean":
            if V is None:
                raise ValueError("metric='seuclidean' exige V.")
            return squareform(pdist(X, metric=metric, V=V))
        else:
            return squareform(pdist(X, metric=metric))
    except Exception as e:
        raise ValueError(f"Erreur pdist avec metric='{metric}'. Détail: {e}")


def rdm_obs_from_zone_features(Z_feat: pd.DataFrame, obs_metric: str, V_obs=None):
    if Z_feat is None or len(Z_feat) < 3:
        return None
    return _safe_squareform_pdist(Z_feat.values, metric=obs_metric, V=V_obs)


def rdm_obs_from_df(df_bin: pd.DataFrame, obs_metric: str, obs_cols: list, V_obs=None):
    Z_feat = df_bin.groupby("Zone")[obs_cols].median().dropna()
    if len(Z_feat) < 3:
        return None, None
    RDM_obs = rdm_obs_from_zone_features(Z_feat, obs_metric, V_obs=V_obs)
    return Z_feat, RDM_obs


def rdm_model_from_df(df_bin: pd.DataFrame, zone_index: pd.Index, model_metric: str):
    Z_dist = df_bin.groupby("Zone")["Distance_lisiere_num"].median().reindex(zone_index)
    if Z_dist.isna().any():
        return None, None
    RDM_model = _safe_squareform_pdist(Z_dist.values.reshape(-1, 1), metric=model_metric)
    return Z_dist, RDM_model


def rsa_spearman_rho(RDM_obs: np.ndarray, RDM_model: np.ndarray):
    rho, p = spearmanr(upper_tri_vec(RDM_obs), upper_tri_vec(RDM_model), nan_policy="omit")
    return rho, p


def bubble_mask_score_corr(mask_vec: np.ndarray, score_vec: np.ndarray, metric: str):
    if metric.lower() == "spearman":
        r, _ = spearmanr(mask_vec, score_vec, nan_policy="omit")
        return 0.0 if not np.isfinite(r) else float(r)
    elif metric.lower() == "pearson":
        if np.all(mask_vec == mask_vec[0]):
            return 0.0
        r, _ = pearsonr(mask_vec.astype(float), score_vec.astype(float))
        return 0.0 if not np.isfinite(r) else float(r)
    else:
        raise ValueError("bubble_revcorr_metric doit être 'spearman' ou 'pearson'.")


def bubble_probe_features_from_zone_matrix(
    Z_feat_full: pd.DataFrame,
    RDM_model: np.ndarray,
    obs_metric: str,
    V_obs,
    rng: np.random.Generator,
    n_trials: int,
    p_on: float,
    score_mode: str = "rho",
    revcorr_metric: str = "spearman",
    min_valid_trials: int = 10,
):
    n_feat = Z_feat_full.shape[1]
    if n_feat < 1:
        return None

    masks = np.zeros((n_trials, n_feat), dtype=np.int8)
    scores = np.full(n_trials, np.nan, dtype=float)

    V_full = None
    if obs_metric == "seuclidean":
        V_full = np.asarray(V_obs)

    for t in range(n_trials):
        m = rng.random(n_feat) < p_on
        if not np.any(m):
            m[rng.integers(0, n_feat)] = True
        masks[t, :] = m.astype(np.int8)

        Zm = Z_feat_full.iloc[:, m]
        if Zm.shape[1] < 1 or len(Zm) < 3:
            continue

        if obs_metric == "seuclidean":
            Vm = V_full[m]
            RDM_obs_m = _safe_squareform_pdist(Zm.values, metric=obs_metric, V=Vm)
        else:
            RDM_obs_m = _safe_squareform_pdist(Zm.values, metric=obs_metric)

        rho, _ = rsa_spearman_rho(RDM_obs_m, RDM_model)
        if not np.isfinite(rho):
            continue

        scores[t] = rho if score_mode == "rho" else (rho ** 2)

    ok = np.isfinite(scores)
    if np.sum(ok) < min_valid_trials:
        return None

    masks_ok = masks[ok]
    scores_ok = scores[ok]

    w_obs = np.zeros(n_feat, dtype=float)
    for f in range(n_feat):
        w_obs[f] = bubble_mask_score_corr(masks_ok[:, f], scores_ok, metric=revcorr_metric)

    return w_obs


def bubble_null_distribution_from_zone_matrix(
    Z_feat_full: pd.DataFrame,
    Z_dist: pd.Series,
    obs_metric: str,
    model_metric: str,
    V_obs,
    rng: np.random.Generator,
    n_null: int,
    n_trials: int,
    p_on: float,
    score_mode: str = "rho",
    revcorr_metric: str = "spearman",
):
    """
    Génère une distribution nulle des poids BUBBLES pour chaque feature
    en shufflant la structure de la matrice modèle à chaque répétition.
    """
    n_feat = Z_feat_full.shape[1]
    W_null = []

    zvals = Z_dist.values.copy()

    for _ in range(n_null):
        perm = rng.permutation(len(zvals))
        zperm = zvals[perm]

        try:
            RDM_model_perm = _safe_squareform_pdist(
                zperm.reshape(-1, 1),
                metric=model_metric
            )
        except Exception:
            continue

        w_null = bubble_probe_features_from_zone_matrix(
            Z_feat_full=Z_feat_full,
            RDM_model=RDM_model_perm,
            obs_metric=obs_metric,
            V_obs=V_obs,
            rng=rng,
            n_trials=n_trials,
            p_on=p_on,
            score_mode=score_mode,
            revcorr_metric=revcorr_metric,
        )

        if w_null is None:
            continue

        W_null.append(w_null)

    if len(W_null) == 0:
        return np.zeros((0, n_feat))

    return np.vstack(W_null)


def feature_ttests_from_null(
    w_obs: np.ndarray,
    W_null: np.ndarray,
    min_valid_null_local: int,
):
    """
    Compare chaque poids observé w_obs[f] à la distribution nulle W_null[:, f]
    avec un t-test bilatéral de type:
        t = (w_obs[f] - mean(null_f)) / (sd(null_f) / sqrt(n))
    Retourne moyenne nulle, sd nulle, t-stat, p-valeur t, nb de nulls valides.
    """
    n_feat = len(w_obs)

    null_mean = np.full(n_feat, np.nan)
    null_std = np.full(n_feat, np.nan)
    t_stat = np.full(n_feat, np.nan)
    p_t = np.full(n_feat, np.nan)
    n_null_valid = np.zeros(n_feat, dtype=int)

    if W_null is None or W_null.size == 0:
        return null_mean, null_std, t_stat, p_t, n_null_valid

    for f in range(n_feat):
        x0 = W_null[:, f]
        x0 = x0[np.isfinite(x0)]

        n_null_valid[f] = len(x0)

        if len(x0) < min_valid_null_local or not np.isfinite(w_obs[f]):
            continue

        mu = np.mean(x0)
        sd = np.std(x0, ddof=1) if len(x0) > 1 else np.nan

        null_mean[f] = mu
        null_std[f] = sd

        n = len(x0)
        if np.isfinite(sd) and sd > 0 and n > 1:
            t_val = (w_obs[f] - mu) / (sd / np.sqrt(n))
            p_val = 2 * t_dist.sf(np.abs(t_val), df=n - 1)

            t_stat[f] = t_val
            p_t[f] = p_val

    return null_mean, null_std, t_stat, p_t, n_null_valid


def bootstrap_leave_N_zones_with_bubbles(
    df_bin: pd.DataFrame,
    zones_all: np.ndarray,
    N_drop: int,
    n_boot: int,
    rng: np.random.Generator,
    obs_metric: str,
    model_metric: str,
    obs_cols: list,
    V_obs=None,
    n_bubble_trials: int = 200,
    p_bubble: float = 0.2,
    bubble_score_mode: str = "rho",
    bubble_revcorr_metric: str = "spearman",
):
    Z = len(zones_all)
    keep_size = Z - N_drop
    if keep_size < 3:
        return None, None

    r2s = []
    W_list = []

    for _ in range(n_boot):
        keep = rng.choice(zones_all, size=keep_size, replace=False)
        df_sub = df_bin[df_bin["Zone"].isin(keep)]

        Z_feat, RDM_obs = rdm_obs_from_df(df_sub, obs_metric, obs_cols, V_obs=V_obs)
        if RDM_obs is None:
            continue

        _, RDM_model = rdm_model_from_df(df_sub, Z_feat.index, model_metric)
        if RDM_model is None:
            continue

        rho, _ = rsa_spearman_rho(RDM_obs, RDM_model)
        if not np.isfinite(rho):
            continue

        r2s.append(rho ** 2)

        w = bubble_probe_features_from_zone_matrix(
            Z_feat_full=Z_feat,
            RDM_model=RDM_model,
            obs_metric=obs_metric,
            V_obs=V_obs,
            rng=rng,
            n_trials=n_bubble_trials,
            p_on=p_bubble,
            score_mode=bubble_score_mode,
            revcorr_metric=bubble_revcorr_metric,
        )

        if w is None:
            W_list.append(np.full(len(obs_cols), np.nan))
        else:
            W_list.append(w)

    if len(r2s) == 0:
        return None, None

    return np.array(r2s), np.vstack(W_list)


def permutation_test_model(
    df_bin: pd.DataFrame,
    zone_index: pd.Index,
    RDM_obs: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
    model_metric: str,
):
    Z_dist = df_bin.groupby("Zone")["Distance_lisiere_num"].median().reindex(zone_index)
    if Z_dist.isna().any():
        return None

    rhos = []
    for _ in range(n_perm):
        perm = rng.permutation(len(Z_dist))
        Zp = Z_dist.values[perm]
        RDM_model_perm = _safe_squareform_pdist(Zp.reshape(-1, 1), metric=model_metric)

        rho_perm, _ = rsa_spearman_rho(RDM_obs, RDM_model_perm)
        if np.isfinite(rho_perm):
            rhos.append(rho_perm)

    if len(rhos) == 0:
        return None
    return np.array(rhos)


def fmt_hhmm(h):
    h24 = h % 24
    hh = int(np.floor(h24))
    mm = int(np.round((h24 - hh) * 60)) % 60
    return f"{hh:02d}:{mm:02d}"


# =============================================================================
# 8) FENÊTRES TEMPORELLES
# =============================================================================
df_time = df_all.copy()
df_time["hour_decimal"] = hhmmss_to_hour_decimal(df_time["Heure"])

starts = np.arange(0, 24, step_hours).astype(float)


def process_window(start: float):
    rng_w = np.random.default_rng(seed + int(round(start * 1000)))

    end = start + window_hours
    label = f"{start:05.1f}-{(end % 24):05.1f}"

    if end <= 24:
        mask = (df_time["hour_decimal"] >= start) & (df_time["hour_decimal"] < end)
    else:
        mask = (df_time["hour_decimal"] >= start) | (df_time["hour_decimal"] < (end - 24))

    df_bin = df_time[mask].copy()
    n_obs = len(df_bin)
    n_z = df_bin["Zone"].nunique()
    if n_z < min_zones:
        return None

    Z_feat, RDM_obs = rdm_obs_from_df(df_bin, obs_rdm_metric, obs_cols, V_obs=V_obs)
    if RDM_obs is None:
        return None

    Z_dist, RDM_model = rdm_model_from_df(df_bin, Z_feat.index, model_rdm_metric)
    if RDM_model is None:
        return None

    rho_obs, _ = rsa_spearman_rho(RDM_obs, RDM_model)
    if not np.isfinite(rho_obs):
        return None
    r2_obs = rho_obs ** 2

    zones_all = Z_feat.index.to_numpy()

    boot_r2, boot_W = bootstrap_leave_N_zones_with_bubbles(
        df_bin, zones_all, N_drop, n_boot, rng_w,
        obs_metric=obs_rdm_metric,
        model_metric=model_rdm_metric,
        obs_cols=obs_cols,
        V_obs=V_obs,
        n_bubble_trials=n_bubble_trials,
        p_bubble=p_bubble,
        bubble_score_mode=bubble_score_mode,
        bubble_revcorr_metric=bubble_revcorr_metric,
    )

    if boot_r2 is not None and len(boot_r2) > 0:
        r2_ci_low, r2_ci_high = np.quantile(boot_r2, [0.025, 0.975])
        r2_boot_mean = float(np.mean(boot_r2))
        r2_boot_std = float(np.std(boot_r2, ddof=1)) if len(boot_r2) > 1 else 0.0
    else:
        r2_ci_low, r2_ci_high = np.nan, np.nan
        r2_boot_mean, r2_boot_std = np.nan, np.nan

    if boot_W is not None and boot_W.size > 0:
        W_boot_mean = np.nanmean(boot_W, axis=0)
        W_boot_std = np.nanstd(boot_W, axis=0, ddof=1) if boot_W.shape[0] > 1 else np.full(len(obs_cols), np.nan)
    else:
        W_boot_mean = np.full(len(obs_cols), np.nan)
        W_boot_std = np.full(len(obs_cols), np.nan)

    # BUBBLES observés sur la fenêtre complète
    W_obs_full = bubble_probe_features_from_zone_matrix(
        Z_feat_full=Z_feat,
        RDM_model=RDM_model,
        obs_metric=obs_rdm_metric,
        V_obs=V_obs,
        rng=rng_w,
        n_trials=n_bubble_trials,
        p_on=p_bubble,
        score_mode=bubble_score_mode,
        revcorr_metric=bubble_revcorr_metric,
    )
    if W_obs_full is None:
        W_obs_full = np.full(len(obs_cols), np.nan)

    # NULL DISTRIBUTION BUBBLES : shuffle de la matrice modèle
    W_null = bubble_null_distribution_from_zone_matrix(
        Z_feat_full=Z_feat,
        Z_dist=Z_dist,
        obs_metric=obs_rdm_metric,
        model_metric=model_rdm_metric,
        V_obs=V_obs,
        rng=rng_w,
        n_null=n_bubble_null,
        n_trials=n_bubble_trials,
        p_on=p_bubble,
        score_mode=bubble_score_mode,
        revcorr_metric=bubble_revcorr_metric,
    )

    # T-TEST FEATURE-WISE : valeur observée vs distribution nulle
    W_null_mean, W_null_std, W_t, W_p, W_n_null_valid = feature_ttests_from_null(
        W_obs_full,
        W_null,
        min_valid_null_local=min_valid_null,
    )

    perm_rhos = permutation_test_model(
        df_bin, Z_feat.index, RDM_obs, n_perm, rng_w,
        model_metric=model_rdm_metric
    )

    if perm_rhos is not None:
        p_perm_emp = (np.sum(np.abs(perm_rhos) >= np.abs(rho_obs)) + 1) / (len(perm_rhos) + 1)

        mu0 = float(np.mean(perm_rhos))
        sd0 = float(np.std(perm_rhos, ddof=1)) if len(perm_rhos) > 1 else np.nan
        if np.isfinite(sd0) and sd0 > 0:
            t_stat = (rho_obs - mu0) / (sd0 / np.sqrt(len(perm_rhos)))
            p_t = 2 * t_dist.sf(np.abs(t_stat), df=len(perm_rhos) - 1)
        else:
            t_stat, p_t = np.nan, np.nan
    else:
        p_perm_emp, t_stat, p_t = np.nan, np.nan, np.nan

    center_hour = float(start + window_hours / 2)
    center_hour_mod = float(center_hour % 24)

    result_row = {
        "window_label": label,
        "start_hour": float(start),
        "end_hour": float(end),
        "center_hour": center_hour,
        "center_hour_mod": center_hour_mod,

        "n_obs": int(n_obs),
        "n_zones": int(n_z),

        "representation_mode": representation_mode,
        "zscore_features_globally": bool(zscore_features_globally),
        "n_obs_features": int(len(obs_cols)),
        "obs_rdm_metric": obs_rdm_metric,
        "model_rdm_metric": model_rdm_metric,

        "rho_obs": float(rho_obs),
        "R2_obs": float(r2_obs),

        "R2_boot_mean": float(r2_boot_mean) if np.isfinite(r2_boot_mean) else np.nan,
        "R2_boot_std": float(r2_boot_std) if np.isfinite(r2_boot_std) else np.nan,
        "R2_ci_low": float(r2_ci_low) if np.isfinite(r2_ci_low) else np.nan,
        "R2_ci_high": float(r2_ci_high) if np.isfinite(r2_ci_high) else np.nan,

        "p_perm_empirical": float(p_perm_emp) if np.isfinite(p_perm_emp) else np.nan,
        "t_stat_perm": float(t_stat) if np.isfinite(t_stat) else np.nan,
        "p_t_perm": float(p_t) if np.isfinite(p_t) else np.nan,

        "n_bubble_trials": int(n_bubble_trials),
        "p_bubble": float(p_bubble),
        "bubble_score_mode": str(bubble_score_mode),
        "bubble_revcorr_metric": str(bubble_revcorr_metric),

        "n_bubble_null": int(n_bubble_null),
        "min_valid_null": int(min_valid_null),

        "abs_weight_thresh": float(abs_weight_thresh),
        "use_abs_weight_threshold": bool(use_abs_weight_threshold),
        "alpha_time": float(alpha_time),
        "alpha_feat": float(alpha_feat),
    }

    feature_stats = pd.DataFrame({
        "feature": obs_cols,
        "start_hour": float(start),
        "window_label": label,
        "center_hour": center_hour,
        "center_hour_mod": center_hour_mod,
        "W_obs_full": W_obs_full,
        "W_boot_mean": W_boot_mean,
        "W_boot_std": W_boot_std,
        "W_null_mean": W_null_mean,
        "W_null_std": W_null_std,
        "W_t_stat": W_t,
        "W_p_ttest": W_p,
        "W_n_null_valid": W_n_null_valid,
    })

    return (float(start), label, result_row, W_boot_mean, feature_stats)


print(f"\nParallélisation par fenêtre : n_jobs={n_jobs}")
out = Parallel(n_jobs=n_jobs, prefer="processes")(
    delayed(process_window)(s) for s in tqdm(starts, desc="Fenêtres")
)

out = [o for o in out if o is not None]
out.sort(key=lambda x: x[0])

results = [o[2] for o in out]
bubble_starts = [o[0] for o in out]
bubble_labels = [o[1] for o in out]
bubble_W_rows = [o[3] for o in out]
feature_stats_list = [o[4] for o in out]


# =============================================================================
# 9) DATAFRAME + EXPORTS
# =============================================================================
df_results = pd.DataFrame(results).sort_values("center_hour").reset_index(drop=True)

print("\n=== Résultats fenêtres glissantes ===")
cols_show = [
    "window_label", "start_hour", "n_obs", "n_zones",
    "rho_obs", "R2_obs", "R2_boot_std",
    "R2_ci_low", "R2_ci_high", "p_perm_empirical"
]
print(df_results[cols_show])

os.makedirs(output_dir, exist_ok=True)

out_csv = os.path.join(output_dir, f"{base_name}.csv")
df_results.to_csv(out_csv, index=False)
print(f"\nExport complet : {out_csv}")

rsa_reload_cols = [
    "window_label",
    "start_hour",
    "end_hour",
    "center_hour",
    "center_hour_mod",
    "n_obs",
    "n_zones",
    "rho_obs",
    "R2_obs",
    "R2_boot_mean",
    "R2_boot_std",
    "R2_ci_low",
    "R2_ci_high",
    "p_perm_empirical",
    "t_stat_perm",
    "p_t_perm",
]
df_rsa_reload = df_results[rsa_reload_cols].copy()

out_rsa_reload = os.path.join(output_dir, f"{base_name}_RSA_R2_only_reloadable.csv")
df_rsa_reload.to_csv(out_rsa_reload, index=False)
print(f"Export RSA léger : {out_rsa_reload}")


# =============================================================================
# 10) EXPORT MATRICE BUBBLES
# =============================================================================
W_mat_raw = np.vstack(bubble_W_rows) if len(bubble_W_rows) else np.zeros((0, len(obs_cols)))

bubble_W_df_raw = pd.DataFrame(W_mat_raw, columns=obs_cols)
bubble_W_df_raw["start_hour"] = bubble_starts
bubble_W_df_raw["window_label"] = bubble_labels

bubble_W_df = df_results[["start_hour", "center_hour", "center_hour_mod", "window_label"]].merge(
    bubble_W_df_raw, on="start_hour", how="left"
)
bubble_W_mat = bubble_W_df[obs_cols].to_numpy()

out_bubbles_W_csv = os.path.join(output_dir, f"{base_name}_BUBBLES_weights_matrix.csv")
bubble_W_df.to_csv(out_bubbles_W_csv, index=False)
print(f"Export bubbles weights matrix : {out_bubbles_W_csv}")


# =============================================================================
# 10B) EXPORT STATS FEATURE-WISE CONTRE LA NULLE
# =============================================================================
if len(feature_stats_list) > 0:
    df_bubble_feature_stats = pd.concat(feature_stats_list, axis=0, ignore_index=True)

    out_bubble_feature_stats = os.path.join(
        output_dir,
        f"{base_name}_BUBBLES_feature_null_stats.csv"
    )
    df_bubble_feature_stats.to_csv(out_bubble_feature_stats, index=False)
    print(f"Export bubbles null stats : {out_bubble_feature_stats}")
else:
    df_bubble_feature_stats = pd.DataFrame()

if not df_bubble_feature_stats.empty:
    bubble_p_df = df_bubble_feature_stats.pivot(
        index="start_hour",
        columns="feature",
        values="W_p_ttest"
    ).reset_index()

    bubble_p_df = df_results[["start_hour", "center_hour", "center_hour_mod", "window_label"]].merge(
        bubble_p_df, on="start_hour", how="left"
    )

    P_mat = bubble_p_df[obs_cols].to_numpy()

    out_bubble_p_csv = os.path.join(output_dir, f"{base_name}_BUBBLES_feature_pvalues_matrix.csv")
    bubble_p_df.to_csv(out_bubble_p_csv, index=False)
    print(f"Export bubbles p-values matrix : {out_bubble_p_csv}")
else:
    P_mat = np.full_like(bubble_W_mat, np.nan, dtype=float)


# =============================================================================
# 11) PLOT RSA SEULE
# =============================================================================
sig_code = np.zeros(len(df_results), dtype=int)
pvals_time = df_results["p_perm_empirical"].values
sig_code[(pvals_time < 0.05)] = 1
sig_code[(pvals_time < 0.01)] = 2

x_raw = df_results["center_hour_mod"].values
y = df_results["R2_obs"].values

x = ((x_raw + 12) % 24) - 12
order = np.argsort(x)
x, y, sig_code = x[order], y[order], sig_code[order]

ci_low = df_results.loc[order, "R2_ci_low"].values
ci_high = df_results.loc[order, "R2_ci_high"].values
has_ci = np.isfinite(ci_low) & np.isfinite(ci_high) & np.isfinite(y)

plt.figure()

plt.scatter(x, y, c=sig_code, label="R² observé")

if np.any(has_ci):
    y_ci = y[has_ci]
    x_ci = x[has_ci]
    lo = ci_low[has_ci]
    hi = ci_high[has_ci]
    yerr_low = np.maximum(0, y_ci - lo)
    yerr_high = np.maximum(0, hi - y_ci)

    plt.errorbar(
        x_ci,
        y_ci,
        yerr=[yerr_low, yerr_high],
        fmt="none"
    )

plt.axvline(0)
plt.title(
    f"R² (Spearman²) vs heure\n"
    f"fenêtre={window_hours}h | pas={step_hours}h | drop={N_drop} zones\n"
    f"repr={representation_mode} | zscore={int(zscore_features_globally)} | obs_metric={obs_rdm_metric}"
)
plt.xlabel("Heure (centrée sur minuit)")
plt.ylabel("R²")
ticks = np.arange(-12, 13, 3)
plt.xticks(ticks, [fmt_hhmm(t) for t in ticks])
plt.xlim(-12, 12)
plt.legend()
plt.tight_layout()

out_rsa_fig = os.path.join(output_dir, f"{base_name}_FIG_RSA_R2_only.png")
plt.savefig(out_rsa_fig, dpi=300, bbox_inches="tight")
print(f"Export figure RSA : {out_rsa_fig}")
plt.show()

