import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr, ttest_ind


# =============================================================================
# 0) PARAMETRES
# =============================================================================
representation_mode = "all"   # "pca" ou "all"

# PCA
pca_n_components = 3
pca_whiten = True

# z-score global
zscore_features_globally = True

# metriques
obs_rdm_metric = "correlation"   # "euclidean", "correlation", "seuclidean", etc.
model_rdm_metric = "euclidean"

# pour seuclidean
seuclidean_variance_eps = 1e-12

# plage horaire unique
analysis_start_hour = 22.0
analysis_end_hour = 4.0

# bootstrap leave-N zones
N_drop = 3
n_boot = 30 #500

# permutations RSA
n_perm = 30 #300

# seuil qualite
min_zones = 5

# BUBBLES
n_bubble_trials = 30 #300
p_bubble = 0.75
bubble_score_mode = "rho"           # "rho" ou "r2"
bubble_revcorr_metric = "pearson"   # "spearman" ou "pearson"

# distribution nulle BUBBLES
n_bubble_null = 300
min_valid_null = 1
min_valid_boot = 1

# seuil feature-wise historique (Bonferroni)
alpha_feat = 0.05

# seed
seed = 0

# exports
output_dir = "./output"
base_name = "rsa_results_single_window_bootstrap_vs_null"

# selection post hoc
select_fdr_alpha = 0.05
select_min_boot_valid = 50
select_min_null_valid = 50
select_min_sign_consistency = 0.80
select_min_abs_effect_size = 0.50
select_corr_threshold = 0.85
select_max_features = 8
select_sort_by = "abs_effect_size"   # "abs_effect_size" ou "abs_W_obs_full"


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
    if metric == "seuclidean":
        if V is None:
            raise ValueError("metric='seuclidean' exige V.")
        return squareform(pdist(X, metric=metric, V=V))
    return squareform(pdist(X, metric=metric))


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

    if metric.lower() == "pearson":
        if np.all(mask_vec == mask_vec[0]):
            return 0.0
        r, _ = pearsonr(mask_vec.astype(float), score_vec.astype(float))
        return 0.0 if not np.isfinite(r) else float(r)

    raise ValueError("bubble_revcorr_metric doit etre 'spearman' ou 'pearson'.")


def validate_hour(h: float, name: str):
    if not np.isfinite(h) or h < 0 or h >= 24:
        raise ValueError(f"{name} doit etre dans [0, 24). Recu: {h}")


def window_span_hours(start: float, end: float) -> float:
    span = (end - start) % 24
    if span <= 0:
        raise ValueError("La plage horaire a une duree nulle. Donne deux horaires differents.")
    return span


def select_time_window_bounds(df_in: pd.DataFrame, start: float, end: float):
    if start < end:
        mask = (df_in["hour_decimal"] >= start) & (df_in["hour_decimal"] < end)
    else:
        mask = (df_in["hour_decimal"] >= start) | (df_in["hour_decimal"] < end)
    return df_in[mask].copy()


def centered_hour_for_window(start: float, end: float) -> float:
    span = window_span_hours(start, end)
    return (start + span / 2) % 24


def fmt_hhmm(h):
    h24 = h % 24
    hh = int(np.floor(h24))
    mm = int(np.round((h24 - hh) * 60)) % 60
    return f"{hh:02d}:{mm:02d}"


def p_to_stars(p):
    if not np.isfinite(p):
        return "ns"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def benjamini_hochberg(pvals, alpha=0.05):
    pvals = np.asarray(pvals, dtype=float)
    qvals = np.full_like(pvals, np.nan, dtype=float)
    reject = np.zeros(len(pvals), dtype=bool)

    mask = np.isfinite(pvals)
    p = pvals[mask]
    m = len(p)
    if m == 0:
        return reject, qvals

    order = np.argsort(p)
    ranked = p[order]

    q_ranked = ranked * m / np.arange(1, m + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0, 1)

    thresh = alpha * np.arange(1, m + 1) / m
    passed = ranked <= thresh
    reject_ranked = np.zeros(m, dtype=bool)
    if np.any(passed):
        kmax = np.max(np.where(passed)[0])
        reject_ranked[:kmax + 1] = True

    inv_order = np.empty(m, dtype=int)
    inv_order[order] = np.arange(m)

    qvals_masked = q_ranked[inv_order]
    reject_masked = reject_ranked[inv_order]

    qvals[mask] = qvals_masked
    reject[mask] = reject_masked
    return reject, qvals


def compute_sign_consistency_from_boot(boot_W, obs_cols):
    if boot_W is None or boot_W.size == 0:
        return pd.Series(np.nan, index=obs_cols, dtype=float)

    out = {}
    for j, feat in enumerate(obs_cols):
        vals = boot_W[:, j]
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            out[feat] = np.nan
            continue
        prop_pos = np.mean(vals > 0)
        prop_neg = np.mean(vals < 0)
        out[feat] = max(prop_pos, prop_neg)
    return pd.Series(out, dtype=float)


def greedy_nonredundant_feature_selection(
    df_candidates,
    corr_abs,
    corr_threshold=0.85,
    max_features=8,
    score_col="abs_effect_size",
):
    if df_candidates is None or df_candidates.empty:
        return []

    if score_col not in df_candidates.columns:
        score_col = "abs_effect_size"

    ranked = df_candidates.sort_values(
        by=[score_col, "p_fdr", "abs_W_obs_full"],
        ascending=[False, True, False],
    )

    selected = []
    for feat in ranked["feature"]:
        if feat not in corr_abs.index:
            continue
        keep = True
        for s in selected:
            if s in corr_abs.columns and np.isfinite(corr_abs.loc[feat, s]):
                if corr_abs.loc[feat, s] >= corr_threshold:
                    keep = False
                    break
        if keep:
            selected.append(feat)
        if len(selected) >= max_features:
            break

    return selected


def compute_rsa_for_feature_subset(
    df_bin: pd.DataFrame,
    feature_subset: list,
    subset_name: str,
    obs_metric: str,
    model_metric: str,
    obs_cols_full: list,
    V_obs_full,
    n_perm: int,
    perm_seed: int,
):
    feature_subset = [f for f in feature_subset if f in obs_cols_full]

    result = {
        "subset_name": subset_name,
        "n_features": int(len(feature_subset)),
        "n_zones": np.nan,
        "rho_obs": np.nan,
        "R2_obs": np.nan,
        "p_perm_empirical": np.nan,
        "note": "",
    }

    if len(feature_subset) == 0:
        result["note"] = "Aucune feature dans ce sous-ensemble."
        return result

    if obs_metric == "correlation" and len(feature_subset) < 2:
        result["note"] = "obs_rdm_metric='correlation' exige au moins 2 features."
        return result

    V_subset = None
    if obs_metric == "seuclidean":
        feat_to_idx_full = {f: i for i, f in enumerate(obs_cols_full)}
        V_subset = np.array([V_obs_full[feat_to_idx_full[f]] for f in feature_subset], dtype=float)

    try:
        Z_feat_sub, RDM_obs_sub = rdm_obs_from_df(
            df_bin=df_bin,
            obs_metric=obs_metric,
            obs_cols=feature_subset,
            V_obs=V_subset,
        )
    except Exception as e:
        result["note"] = f"Echec construction RDM observee: {e}"
        return result

    if RDM_obs_sub is None or Z_feat_sub is None:
        result["note"] = "RDM observee impossible a construire."
        return result

    _, RDM_model_sub = rdm_model_from_df(
        df_bin=df_bin,
        zone_index=Z_feat_sub.index,
        model_metric=model_metric,
    )

    if RDM_model_sub is None:
        result["note"] = "RDM modele impossible a construire."
        return result

    rho_sub, _ = rsa_spearman_rho(RDM_obs_sub, RDM_model_sub)
    if not np.isfinite(rho_sub):
        result["note"] = "rho non fini."
        return result

    perm_rhos_sub = permutation_test_model(
        df_bin=df_bin,
        zone_index=Z_feat_sub.index,
        RDM_obs=RDM_obs_sub,
        n_perm=n_perm,
        rng=np.random.default_rng(perm_seed),
        model_metric=model_metric,
    )

    if perm_rhos_sub is not None and len(perm_rhos_sub) > 0:
        p_perm_sub = (np.sum(np.abs(perm_rhos_sub) >= np.abs(rho_sub)) + 1) / (len(perm_rhos_sub) + 1)
    else:
        p_perm_sub = np.nan

    result.update(
        {
            "n_zones": int(len(Z_feat_sub)),
            "rho_obs": float(rho_sub),
            "R2_obs": float(rho_sub ** 2),
            "p_perm_empirical": float(p_perm_sub) if np.isfinite(p_perm_sub) else np.nan,
        }
    )
    return result


def make_feature_subset_definitions(
    df_bubbles: pd.DataFrame,
    obs_cols: list,
    selected_feature_list: list,
):
    mask_sig_pos = df_bubbles["significant_fdr"] & (df_bubbles["W_boot_mean_test"] > 0)
    mask_sig_neg = df_bubbles["significant_fdr"] & (df_bubbles["W_boot_mean_test"] < 0)

    df_sig_pos = (
        df_bubbles.loc[mask_sig_pos]
        .sort_values(by="effect_size", ascending=False)
        .copy()
    )

    df_sig_neg = (
        df_bubbles.loc[mask_sig_neg]
        .sort_values(by="effect_size", ascending=True)
        .copy()
    )

    df_neither = df_bubbles.loc[~(mask_sig_pos | mask_sig_neg)].copy()

    subset_dict = {
        "selected_positive_greedy": [f for f in selected_feature_list if f in obs_cols],
        "all_sig_positive": df_sig_pos["feature"].tolist(),
        "top5_sig_positive_effect": df_sig_pos["feature"].head(5).tolist(),
        "top10_sig_positive_effect": df_sig_pos["feature"].head(10).tolist(),
        "all_sig_negative": df_sig_neg["feature"].tolist(),
        "top5_sig_negative_effect": df_sig_neg["feature"].head(5).tolist(),
        "top10_sig_negative_effect": df_sig_neg["feature"].head(10).tolist(),
        "neither_sig_positive_nor_negative": [
            f for f in df_neither["feature"].tolist() if f in obs_cols
        ],
    }

    return subset_dict


def subset_dict_to_long_df(subset_dict: dict):
    rows = []
    for subset_name, feats in subset_dict.items():
        if len(feats) == 0:
            rows.append(
                {
                    "subset_name": subset_name,
                    "rank_in_subset": np.nan,
                    "feature": np.nan,
                    "n_features_in_subset": 0,
                }
            )
        else:
            for rank, feat in enumerate(feats, start=1):
                rows.append(
                    {
                        "subset_name": subset_name,
                        "rank_in_subset": rank,
                        "feature": feat,
                        "n_features_in_subset": len(feats),
                    }
                )
    return pd.DataFrame(rows)


def build_rsa_subset_comparison_table(
    df_bin: pd.DataFrame,
    subset_dict: dict,
    obs_metric: str,
    model_metric: str,
    obs_cols_full: list,
    V_obs_full,
    n_perm: int,
    seed: int,
):
    rows = []

    res_all = compute_rsa_for_feature_subset(
        df_bin=df_bin,
        feature_subset=obs_cols_full,
        subset_name="all_features",
        obs_metric=obs_metric,
        model_metric=model_metric,
        obs_cols_full=obs_cols_full,
        V_obs_full=V_obs_full,
        n_perm=n_perm,
        perm_seed=seed + 1000,
    )
    res_all["subset_base_name"] = "all_features"
    res_all["subset_mode"] = "all"
    rows.append(res_all)

    for k, (subset_name, feature_subset) in enumerate(subset_dict.items(), start=1):
        feature_subset = [f for f in feature_subset if f in obs_cols_full]
        feature_subset_set = set(feature_subset)
        complement_subset = [f for f in obs_cols_full if f not in feature_subset_set]

        res_only = compute_rsa_for_feature_subset(
            df_bin=df_bin,
            feature_subset=feature_subset,
            subset_name=f"{subset_name}__only",
            obs_metric=obs_metric,
            model_metric=model_metric,
            obs_cols_full=obs_cols_full,
            V_obs_full=V_obs_full,
            n_perm=n_perm,
            perm_seed=seed + 1000 + 2 * k,
        )
        res_only["subset_base_name"] = subset_name
        res_only["subset_mode"] = "only"
        rows.append(res_only)

        res_without = compute_rsa_for_feature_subset(
            df_bin=df_bin,
            feature_subset=complement_subset,
            subset_name=f"{subset_name}__without",
            obs_metric=obs_metric,
            model_metric=model_metric,
            obs_cols_full=obs_cols_full,
            V_obs_full=V_obs_full,
            n_perm=n_perm,
            perm_seed=seed + 1000 + 2 * k + 1,
        )
        res_without["subset_base_name"] = subset_name
        res_without["subset_mode"] = "without"
        rows.append(res_without)

    return pd.DataFrame(rows)


# =============================================================================
# 2) BUBBLES
# =============================================================================
def bubble_probe_features_from_zone_matrix(
    Z_feat_full: pd.DataFrame,
    RDM_model: np.ndarray,
    obs_metric: str,
    V_obs,
    rng: np.random.Generator,
    n_trials: int,
    p_on: float,
    score_mode: str = "rho",
    revcorr_metric: str = "pearson",
    min_valid_trials: int = 10,
):
    n_feat = Z_feat_full.shape[1]
    if n_feat < 1:
        return None

    masks = np.zeros((n_trials, n_feat), dtype=np.int8)
    scores = np.full(n_trials, np.nan, dtype=float)

    V_full = np.asarray(V_obs) if obs_metric == "seuclidean" else None

    for t in range(n_trials):
        m = rng.random(n_feat) < p_on
        if not np.any(m):
            m[rng.integers(0, n_feat)] = True
        masks[t, :] = m.astype(np.int8)

        Zm = Z_feat_full.iloc[:, m]
        if Zm.shape[1] < 1 or len(Zm) < 3:
            continue

        if obs_metric == "seuclidean":
            RDM_obs_m = _safe_squareform_pdist(Zm.values, metric=obs_metric, V=V_full[m])
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
    n_feat = Z_feat_full.shape[1]
    W_null = []

    zvals = Z_dist.values.copy()

    for _ in range(n_null):
        perm = rng.permutation(len(zvals))
        zperm = zvals[perm]

        try:
            RDM_model_perm = _safe_squareform_pdist(zperm.reshape(-1, 1), metric=model_metric)
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

        if w_null is not None:
            W_null.append(w_null)

    if len(W_null) == 0:
        return np.zeros((0, n_feat))

    return np.vstack(W_null)


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


def feature_ttests_boot_vs_null(
    boot_W: np.ndarray,
    W_null: np.ndarray,
    min_valid_boot_local: int = 10,
    min_valid_null_local: int = 20,
):
    if boot_W is None or boot_W.size == 0:
        n_feat = W_null.shape[1] if (W_null is not None and W_null.ndim == 2 and W_null.size > 0) else 0
        return (
            np.full(n_feat, np.nan), np.full(n_feat, np.nan),
            np.full(n_feat, np.nan), np.full(n_feat, np.nan),
            np.full(n_feat, np.nan), np.full(n_feat, np.nan),
            np.zeros(n_feat, dtype=int), np.zeros(n_feat, dtype=int),
        )

    n_feat = boot_W.shape[1]

    boot_mean = np.full(n_feat, np.nan)
    boot_std = np.full(n_feat, np.nan)
    null_mean = np.full(n_feat, np.nan)
    null_std = np.full(n_feat, np.nan)
    t_stat = np.full(n_feat, np.nan)
    p_t = np.full(n_feat, np.nan)
    n_boot_valid = np.zeros(n_feat, dtype=int)
    n_null_valid = np.zeros(n_feat, dtype=int)

    if W_null is None or W_null.size == 0:
        return boot_mean, boot_std, null_mean, null_std, t_stat, p_t, n_boot_valid, n_null_valid

    for f in range(n_feat):
        xb = boot_W[:, f]
        xb = xb[np.isfinite(xb)]

        xn = W_null[:, f]
        xn = xn[np.isfinite(xn)]

        n_boot_valid[f] = len(xb)
        n_null_valid[f] = len(xn)

        if len(xb) < min_valid_boot_local or len(xn) < min_valid_null_local:
            continue

        boot_mean[f] = np.mean(xb)
        boot_std[f] = np.std(xb, ddof=1) if len(xb) > 1 else np.nan

        null_mean[f] = np.mean(xn)
        null_std[f] = np.std(xn, ddof=1) if len(xn) > 1 else np.nan

        if len(xb) > 1 and len(xn) > 1:
            t_res = ttest_ind(xb, xn, equal_var=False, nan_policy="omit")
            t_stat[f] = t_res.statistic
            p_t[f] = t_res.pvalue

    return boot_mean, boot_std, null_mean, null_std, t_stat, p_t, n_boot_valid, n_null_valid


# =============================================================================
# 3) TEST PERMUTATION RSA
# =============================================================================
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


# =============================================================================
# 4) CHARGER + FILTRER
# =============================================================================
df_all = pd.read_csv("./data.csv")
df_all = df_all[df_all["rainy (1 = rainy, 0 = not rainy)"] == 0].copy()
print("Nombre d'observations apres filtrage pluie :", len(df_all))


# =============================================================================
# 5) DISTANCE LISIERE -> NUMERIQUE
# =============================================================================
df_all["Distance_lisiere_num"] = (
    df_all["Distance_lisiere"]
    .astype(str)
    .str.extract(r"(\d+)", expand=False)
    .astype(float)
)
df_all = df_all.dropna(subset=["Distance_lisiere_num"]).copy()


# =============================================================================
# 6) FEATURES NUMERIQUES
# =============================================================================
exclude_cols = {
    "Distance_lisiere_num",
    "Distance_lisiere",
    "Fichier", "Zone", "Heure", "jour/nuit", "Date",
    "Identifiant",
    "rainy (1 = rainy, 0 = not rainy)",
}

num_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in num_cols if c not in exclude_cols]
feature_cols = [c for c in feature_cols if df_all[c].nunique(dropna=True) > 1]

if len(feature_cols) < 1:
    raise ValueError("Aucune colonne numerique exploitable trouvee.")


# =============================================================================
# 7) PRETRAITEMENT GLOBAL
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
# 8) REPRESENTATION OBSERVEE
# =============================================================================
if representation_mode.lower() == "pca":
    X_for_pca = df_all[feature_cols_used].copy()
    X_for_pca = X_for_pca.fillna(X_for_pca.median(numeric_only=True))

    pca = PCA(n_components=pca_n_components, random_state=0, whiten=pca_whiten)
    PC = pca.fit_transform(X_for_pca.values)

    for k in range(pca_n_components):
        df_all[f"PC{k+1}"] = PC[:, k]

    print(
        "Variance expliquee (PCA globale) :",
        pca.explained_variance_ratio_,
        "| cumul :",
        pca.explained_variance_ratio_.sum(),
    )
    obs_cols = [f"PC{k+1}" for k in range(pca_n_components)]

elif representation_mode.lower() == "all":
    obs_cols = feature_cols_used
    print(f"Mode ALL: {len(obs_cols)} indices utilises.")

else:
    raise ValueError("representation_mode doit etre 'pca' ou 'all'.")


# =============================================================================
# 9) V POUR SEUCLIDEAN
# =============================================================================
if obs_rdm_metric == "seuclidean":
    X_obs_global = df_all[obs_cols].copy()
    X_obs_global = X_obs_global.fillna(X_obs_global.median(numeric_only=True))
    V_obs = np.var(X_obs_global.values, axis=0, ddof=1)
    V_obs = np.maximum(V_obs, seuclidean_variance_eps)
else:
    V_obs = None


# =============================================================================
# 10) SELECTION DE LA PLAGE HORAIRE UNIQUE
# =============================================================================
validate_hour(analysis_start_hour, "analysis_start_hour")
validate_hour(analysis_end_hour, "analysis_end_hour")

analysis_span_hours = window_span_hours(analysis_start_hour, analysis_end_hour)
analysis_center_hour = centered_hour_for_window(analysis_start_hour, analysis_end_hour)
analysis_label = f"{analysis_start_hour:05.1f}-{analysis_end_hour:05.1f}"
analysis_label_hhmm = f"{fmt_hhmm(analysis_start_hour)}-{fmt_hhmm(analysis_end_hour)}"

df_time = df_all.copy()
df_time["hour_decimal"] = hhmmss_to_hour_decimal(df_time["Heure"])

df_bin = select_time_window_bounds(df_time, analysis_start_hour, analysis_end_hour)

print("\n=== PLAGE HORAIRE UNIQUE ===")
print("Label :", analysis_label_hhmm)
print("Duree (h) :", analysis_span_hours)
print("n_obs :", len(df_bin))
print("n_zones :", df_bin["Zone"].nunique())

if len(df_bin) == 0:
    raise ValueError("Aucune observation dans la plage horaire selectionnee.")

if df_bin["Zone"].nunique() < min_zones:
    raise ValueError(
        f"Nombre de zones insuffisant dans la plage {analysis_label_hhmm}: "
        f"{df_bin['Zone'].nunique()} < min_zones ({min_zones})."
    )


# =============================================================================
# 11) RSA + BUBBLES
# =============================================================================
rng = np.random.default_rng(seed)

Z_feat, RDM_obs = rdm_obs_from_df(df_bin, obs_rdm_metric, obs_cols, V_obs=V_obs)
if RDM_obs is None:
    raise ValueError("Impossible de construire la RDM observee sur cette plage horaire. Il faut au moins 3 zones valides.")

Z_dist, RDM_model = rdm_model_from_df(df_bin, Z_feat.index, model_rdm_metric)
if RDM_model is None:
    raise ValueError("Impossible de construire la RDM modele sur cette plage horaire.")

rho_obs, _ = rsa_spearman_rho(RDM_obs, RDM_model)
if not np.isfinite(rho_obs):
    raise ValueError("rho_obs non fini sur cette plage horaire.")
r2_obs = rho_obs ** 2

zones_all = Z_feat.index.to_numpy()

boot_r2, boot_W = bootstrap_leave_N_zones_with_bubbles(
    df_bin=df_bin,
    zones_all=zones_all,
    N_drop=N_drop,
    n_boot=n_boot,
    rng=rng,
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
    boot_W = np.zeros((0, len(obs_cols)))

W_obs_full = bubble_probe_features_from_zone_matrix(
    Z_feat_full=Z_feat,
    RDM_model=RDM_model,
    obs_metric=obs_rdm_metric,
    V_obs=V_obs,
    rng=rng,
    n_trials=n_bubble_trials,
    p_on=p_bubble,
    score_mode=bubble_score_mode,
    revcorr_metric=bubble_revcorr_metric,
)
if W_obs_full is None:
    W_obs_full = np.full(len(obs_cols), np.nan)

W_null = bubble_null_distribution_from_zone_matrix(
    Z_feat_full=Z_feat,
    Z_dist=Z_dist,
    obs_metric=obs_rdm_metric,
    model_metric=model_rdm_metric,
    V_obs=V_obs,
    rng=rng,
    n_null=n_bubble_null,
    n_trials=n_bubble_trials,
    p_on=p_bubble,
    score_mode=bubble_score_mode,
    revcorr_metric=bubble_revcorr_metric,
)

(
    W_boot_mean_test,
    W_boot_std_test,
    W_null_mean,
    W_null_std,
    W_t,
    W_p,
    W_n_boot_valid,
    W_n_null_valid,
) = feature_ttests_boot_vs_null(
    boot_W=boot_W,
    W_null=W_null,
    min_valid_boot_local=min_valid_boot,
    min_valid_null_local=min_valid_null,
)

perm_rhos = permutation_test_model(
    df_bin=df_bin,
    zone_index=Z_feat.index,
    RDM_obs=RDM_obs,
    n_perm=n_perm,
    rng=rng,
    model_metric=model_rdm_metric,
)

if perm_rhos is not None:
    p_perm_emp = (np.sum(np.abs(perm_rhos) >= np.abs(rho_obs)) + 1) / (len(perm_rhos) + 1)
else:
    p_perm_emp = np.nan


# =============================================================================
# 12) TABLEAUX RESULTATS + SELECTION POST HOC
# =============================================================================
df_summary = pd.DataFrame([
    {
        "window_label": analysis_label,
        "window_label_hhmm": analysis_label_hhmm,
        "start_hour": float(analysis_start_hour),
        "end_hour": float(analysis_end_hour),
        "window_span_hours": float(analysis_span_hours),
        "center_hour": float(analysis_center_hour),
        "n_obs": int(len(df_bin)),
        "n_zones": int(df_bin["Zone"].nunique()),
        "rho_obs": float(rho_obs),
        "R2_obs": float(r2_obs),
        "R2_boot_mean": float(r2_boot_mean) if np.isfinite(r2_boot_mean) else np.nan,
        "R2_boot_std": float(r2_boot_std) if np.isfinite(r2_boot_std) else np.nan,
        "R2_ci_low": float(r2_ci_low) if np.isfinite(r2_ci_low) else np.nan,
        "R2_ci_high": float(r2_ci_high) if np.isfinite(r2_ci_high) else np.nan,
        "p_perm_empirical": float(p_perm_emp) if np.isfinite(p_perm_emp) else np.nan,
    }
])

df_bubbles = pd.DataFrame(
    {
        "feature": obs_cols,
        "W_obs_full": W_obs_full,
        "W_boot_mean": W_boot_mean,
        "W_boot_std": W_boot_std,
        "W_boot_mean_test": W_boot_mean_test,
        "W_boot_std_test": W_boot_std_test,
        "W_n_boot_valid": W_n_boot_valid,
        "W_null_mean": W_null_mean,
        "W_null_std": W_null_std,
        "W_n_null_valid": W_n_null_valid,
        "W_t_stat": W_t,
        "W_p_ttest": W_p,
    }
)

sign_consistency = compute_sign_consistency_from_boot(boot_W, obs_cols)
df_bubbles["sign_consistency"] = df_bubbles["feature"].map(sign_consistency)

df_bubbles["delta_mean"] = df_bubbles["W_boot_mean_test"] - df_bubbles["W_null_mean"]

pooled_sd = np.sqrt((df_bubbles["W_boot_std_test"] ** 2 + df_bubbles["W_null_std"] ** 2) / 2)
pooled_sd = pooled_sd.replace(0, np.nan)
df_bubbles["effect_size"] = df_bubbles["delta_mean"] / pooled_sd
df_bubbles["abs_effect_size"] = np.abs(df_bubbles["effect_size"])

reject_fdr, qvals = benjamini_hochberg(df_bubbles["W_p_ttest"].values, alpha=select_fdr_alpha)
df_bubbles["p_fdr"] = qvals
df_bubbles["significant_fdr"] = reject_fdr

df_bubbles["significant_vs_null"] = df_bubbles["W_p_ttest"] < alpha_feat
df_bubbles["stars"] = df_bubbles["W_p_ttest"].apply(p_to_stars)
df_bubbles["abs_W_obs_full"] = np.abs(df_bubbles["W_obs_full"])

df_bubbles = df_bubbles.sort_values(by=["abs_W_obs_full", "abs_effect_size"], ascending=[False, False]).reset_index(drop=True)

# Candidats positifs uniquement pour la selection greedy "principale"
df_candidates = df_bubbles[
    (df_bubbles["W_n_boot_valid"] >= select_min_boot_valid)
    & (df_bubbles["W_n_null_valid"] >= select_min_null_valid)
    & (df_bubbles["significant_fdr"])
    & (df_bubbles["sign_consistency"] >= select_min_sign_consistency)
    & (df_bubbles["abs_effect_size"] >= select_min_abs_effect_size)
    & (df_bubbles["W_boot_mean_test"] > 0)
].copy()

X_zone_for_selection = df_bin.groupby("Zone")[obs_cols].median().dropna()
if X_zone_for_selection.shape[1] > 0:
    corr_abs = X_zone_for_selection.corr().abs()
else:
    corr_abs = pd.DataFrame(index=obs_cols, columns=obs_cols, dtype=float)

selected_feature_list = greedy_nonredundant_feature_selection(
    df_candidates=df_candidates,
    corr_abs=corr_abs,
    corr_threshold=select_corr_threshold,
    max_features=select_max_features,
    score_col=select_sort_by,
)

df_selected_features = df_candidates[df_candidates["feature"].isin(selected_feature_list)].copy()
selection_rank = {f: i + 1 for i, f in enumerate(selected_feature_list)}
df_selected_features["selection_rank"] = df_selected_features["feature"].map(selection_rank)
df_selected_features = df_selected_features.sort_values("selection_rank").reset_index(drop=True)


# =============================================================================
# 12bis) DEFINITIONS DE SOUS-ENSEMBLES + RSA FINALE COMPARATIVE
# =============================================================================
feature_subset_dict = make_feature_subset_definitions(
    df_bubbles=df_bubbles,
    obs_cols=obs_cols,
    selected_feature_list=selected_feature_list,
)

df_feature_subsets_long = subset_dict_to_long_df(feature_subset_dict)

df_feature_subset_counts = pd.DataFrame(
    [
        {"subset_name": name, "n_features": len(feats)}
        for name, feats in feature_subset_dict.items()
    ]
).sort_values(by="subset_name").reset_index(drop=True)

df_rsa_feature_sets = build_rsa_subset_comparison_table(
    df_bin=df_bin,
    subset_dict=feature_subset_dict,
    obs_metric=obs_rdm_metric,
    model_metric=model_rdm_metric,
    obs_cols_full=obs_cols,
    V_obs_full=V_obs,
    n_perm=n_perm,
    seed=seed,
)

print("\n=== RESUME RSA ===")
print(df_summary)

print("\n=== TOP 15 FEATURES BUBBLES ===")
print(
    df_bubbles.head(15)[
        [
            "feature",
            "W_obs_full",
            "W_p_ttest",
            "p_fdr",
            "stars",
            "significant_fdr",
            "sign_consistency",
            "effect_size",
        ]
    ]
)

print("\n=== CANDIDATS POST HOC POSITIFS ===")
print(
    df_candidates[
        [
            "feature",
            "W_p_ttest",
            "p_fdr",
            "sign_consistency",
            "effect_size",
            "abs_effect_size",
            "abs_W_obs_full",
        ]
    ].sort_values(by=["abs_effect_size", "p_fdr"], ascending=[False, True])
)

print("\n=== SUBSET FINAL SELECTIONNE (GREEDY POSITIF) ===")
print(
    df_selected_features[
        [
            "selection_rank",
            "feature",
            "W_p_ttest",
            "p_fdr",
            "sign_consistency",
            "effect_size",
            "abs_effect_size",
            "abs_W_obs_full",
        ]
    ]
)

print("\n=== COMPTAGE DES SOUS-ENSEMBLES D'INDICES ===")
print(df_feature_subset_counts)

print("\n=== RSA FINALE PAR SOUS-ENSEMBLE DE FEATURES ===")
print(
    df_rsa_feature_sets[
        [
            "subset_name",
            "subset_base_name",
            "subset_mode",
            "n_features",
            "n_zones",
            "rho_obs",
            "R2_obs",
            "p_perm_empirical",
            "note",
        ]
    ]
)


# =============================================================================
# 13) EXPORTS
# =============================================================================
os.makedirs(output_dir, exist_ok=True)

out_summary = os.path.join(output_dir, f"{base_name}_summary.csv")
df_summary.to_csv(out_summary, index=False)
print(f"\nExport summary : {out_summary}")

out_bubbles = os.path.join(output_dir, f"{base_name}_bubbles_features.csv")
df_bubbles.to_csv(out_bubbles, index=False)
print(f"Export bubbles : {out_bubbles}")

out_candidates = os.path.join(output_dir, f"{base_name}_bubbles_candidates_posthoc_positive.csv")
df_candidates.to_csv(out_candidates, index=False)
print(f"Export candidates : {out_candidates}")

out_selected = os.path.join(output_dir, f"{base_name}_bubbles_selected_features.csv")
df_selected_features.to_csv(out_selected, index=False)
print(f"Export selected features : {out_selected}")

out_selected_txt = os.path.join(output_dir, f"{base_name}_bubbles_selected_features.txt")
with open(out_selected_txt, "w", encoding="utf-8") as f:
    for feat in selected_feature_list:
        f.write(f"{feat}\n")
print(f"Export selected feature list : {out_selected_txt}")

out_feature_subset_counts = os.path.join(output_dir, f"{base_name}_feature_subset_counts.csv")
df_feature_subset_counts.to_csv(out_feature_subset_counts, index=False)
print(f"Export feature subset counts : {out_feature_subset_counts}")

out_feature_subsets_long = os.path.join(output_dir, f"{base_name}_feature_subsets_long.csv")
df_feature_subsets_long.to_csv(out_feature_subsets_long, index=False)
print(f"Export feature subsets long : {out_feature_subsets_long}")

out_rsa_feature_sets = os.path.join(output_dir, f"{base_name}_rsa_feature_sets.csv")
df_rsa_feature_sets.to_csv(out_rsa_feature_sets, index=False)
print(f"Export RSA feature sets : {out_rsa_feature_sets}")

feature_order = df_bubbles["feature"].tolist()
feat_to_idx = {f: i for i, f in enumerate(obs_cols)}

long_rows = []

if W_null is not None and W_null.size > 0:
    for f in feature_order:
        j = feat_to_idx[f]
        vals = W_null[:, j]
        vals = vals[np.isfinite(vals)]
        for v in vals:
            long_rows.append({"feature": f, "distribution": "null", "value": float(v)})

if boot_W is not None and boot_W.size > 0:
    for f in feature_order:
        j = feat_to_idx[f]
        vals = boot_W[:, j]
        vals = vals[np.isfinite(vals)]
        for v in vals:
            long_rows.append({"feature": f, "distribution": "bootstrap", "value": float(v)})

df_bubbles_long = pd.DataFrame(long_rows)
out_bubbles_long = os.path.join(output_dir, f"{base_name}_bubbles_distributions_long.csv")
df_bubbles_long.to_csv(out_bubbles_long, index=False)
print(f"Export bubbles long : {out_bubbles_long}")

if len(selected_feature_list) >= 2:
    selected_corr = corr_abs.loc[selected_feature_list, selected_feature_list]
    out_selected_corr = os.path.join(output_dir, f"{base_name}_selected_features_corr_abs.csv")
    selected_corr.to_csv(out_selected_corr, index=True)
    print(f"Export selected feature corr abs : {out_selected_corr}")


# =============================================================================
# 14) PLOT RSA
# =============================================================================
y = df_summary.loc[0, "R2_obs"]
ci_low = df_summary.loc[0, "R2_ci_low"]
ci_high = df_summary.loc[0, "R2_ci_high"]

plt.figure(figsize=(6, 5))
plt.scatter([0], [y], s=80)

if np.isfinite(ci_low) and np.isfinite(ci_high) and np.isfinite(y):
    yerr_low = max(0, y - ci_low)
    yerr_high = max(0, ci_high - y)
    plt.errorbar([0], [y], yerr=[[yerr_low], [yerr_high]], fmt="none")

plt.xticks([0], [analysis_label_hhmm])
plt.ylabel("R²")
plt.title(f"RSA sur plage horaire unique\n{analysis_label_hhmm}")
plt.tight_layout()

out_rsa_fig = os.path.join(output_dir, f"{base_name}_RSA.png")
plt.savefig(out_rsa_fig, dpi=300, bbox_inches="tight")
print(f"Export figure RSA : {out_rsa_fig}")
plt.show()


# =============================================================================
# 15) VIOLIN PLOTS BUBBLES : nulle vs bootstrap + observe + etoiles
#    CLASSES PAR W_boot_mean DECROISSANT
# =============================================================================
plot_df = (
    df_bubbles
    .sort_values(by="W_boot_mean", ascending=False, na_position="last")
    .reset_index(drop=True)
)

n_feat_plot = len(plot_df)
selected_set = set(selected_feature_list)

fig_w = max(12, min(36, 0.6 * n_feat_plot))
fig_h = 8
fig, ax = plt.subplots(figsize=(fig_w, fig_h))

ymin_candidates = []
ymax_candidates = []

for i, feat in enumerate(plot_df["feature"]):
    j = feat_to_idx[feat]

    null_vals = np.array([])
    if W_null is not None and W_null.size > 0:
        null_vals = W_null[:, j]
        null_vals = null_vals[np.isfinite(null_vals)]

    boot_vals = np.array([])
    if boot_W is not None and boot_W.size > 0:
        boot_vals = boot_W[:, j]
        boot_vals = boot_vals[np.isfinite(boot_vals)]

    obs_val = plot_df.loc[i, "W_obs_full"]

    x_null = i - 0.18
    x_boot = i + 0.18

    if len(null_vals) > 1:
        vp_null = ax.violinplot(
            [null_vals],
            positions=[x_null],
            widths=0.28,
            showmeans=False,
            showmedians=True,
            showextrema=False
        )
        for body in vp_null["bodies"]:
            body.set_alpha(0.5)
        vp_null["cmedians"].set_linewidth(1.2)
        ymin_candidates.append(np.nanmin(null_vals))
        ymax_candidates.append(np.nanmax(null_vals))

    if len(boot_vals) > 1:
        vp_boot = ax.violinplot(
            [boot_vals],
            positions=[x_boot],
            widths=0.28,
            showmeans=False,
            showmedians=True,
            showextrema=False
        )
        for body in vp_boot["bodies"]:
            body.set_alpha(0.5)
        vp_boot["cmedians"].set_linewidth(1.2)
        ymin_candidates.append(np.nanmin(boot_vals))
        ymax_candidates.append(np.nanmax(boot_vals))

    if np.isfinite(obs_val):
        ax.scatter(x_boot, obs_val, s=18, zorder=3)
        ymin_candidates.append(obs_val)
        ymax_candidates.append(obs_val)

if len(ymin_candidates) == 0 or len(ymax_candidates) == 0:
    ymin, ymax = -1, 1
else:
    ymin = float(np.nanmin(ymin_candidates))
    ymax = float(np.nanmax(ymax_candidates))
    if not np.isfinite(ymin) or not np.isfinite(ymax):
        ymin, ymax = -1, 1
    elif ymin == ymax:
        ymin -= 1
        ymax += 1

yrange = ymax - ymin
pad = 0.15 * yrange if yrange > 0 else 1.0
ax.set_ylim(ymin - 0.05 * pad, ymax + 1.6 * pad)

for i, feat in enumerate(plot_df["feature"]):
    j = feat_to_idx[feat]

    null_vals = np.array([])
    if W_null is not None and W_null.size > 0:
        null_vals = W_null[:, j]
        null_vals = null_vals[np.isfinite(null_vals)]

    boot_vals = np.array([])
    if boot_W is not None and boot_W.size > 0:
        boot_vals = boot_W[:, j]
        boot_vals = boot_vals[np.isfinite(boot_vals)]

    obs_val = plot_df.loc[i, "W_obs_full"]
    stars = plot_df.loc[i, "stars"]

    local_max = []
    if len(null_vals) > 0:
        local_max.append(np.nanmax(null_vals))
    if len(boot_vals) > 0:
        local_max.append(np.nanmax(boot_vals))
    if np.isfinite(obs_val):
        local_max.append(obs_val)

    if len(local_max) == 0:
        star_y = ymax + 0.2 * pad
    else:
        star_y = max(local_max) + 0.18 * pad

    ax.text(i, star_y, stars, ha="center", va="bottom", fontsize=10)

ax.set_xticks(np.arange(n_feat_plot))
ax.set_xticklabels(plot_df["feature"], rotation=90)
ax.set_ylabel("Poids BUBBLES")
ax.set_title(
    f"BUBBLES par feature (tri = W_boot_mean décroissant)\n"
    f"Plage {analysis_label_hhmm} | gauche = nulle | droite = bootstrap | point = observe | etoiles = t-test bootstrap vs nulle"
)

for tick in ax.get_xticklabels():
    if tick.get_text() in selected_set:
        tick.set_color("darkred")
        tick.set_fontweight("bold")

ax.text(
    0.01,
    0.99,
    "Gauche: nulle   Droite: bootstrap   Point: observe   Rouge: subset post hoc retenu",
    transform=ax.transAxes,
    ha="left",
    va="top",
)

plt.tight_layout()

out_bubbles_fig = os.path.join(output_dir, f"{base_name}_BUBBLES_violinplot_sorted_by_bootstrap.png")
plt.savefig(out_bubbles_fig, dpi=300, bbox_inches="tight")
print(f"Export figure BUBBLES : {out_bubbles_fig}")
plt.show()