"""Sensitivity analysis of a DOE study (5.11.0).

Which inputs move a response, how strongly, in which direction, along which shape and together with which other
inputs - asked of the same finished cases by every family of method in common use, so that a finding rests on
their agreement rather than on one criterion.

    statistics on the runs     Pearson, Spearman and Kendall correlation; partial (PCC) and partial rank (PRCC)
                               correlation; standardised (rank) regression coefficients (SRC, SRRC); the analysis
                               of variance of a quadratic response surface (main, quadratic and two-factor terms,
                               the Pareto chart of standardised effects); the correlation ratio (a first-order
                               index from the data alone); mutual information, distance correlation and HSIC
                               (dependence of any shape, with permutation p-values); the LASSO path.
    machine learning           Gaussian process (ARD length scales), random forest (impurity and permutation
                               importance), gradient boosting (permutation importance), each scored by
                               cross-validation; Shapley values (SHAP) and Shapley interaction values of the best
                               of them; partial dependence and ICE curves; accumulated local effects (ALE);
                               Friedman's H-statistic.
    variance-based, screening  Sobol first, total and second-order indices on the surrogate (Saltelli 2010,
                               bootstrap intervals); a sparse polynomial chaos expansion and its Sobol indices in
                               closed form (Sudret 2008); eFAST (Saltelli 1999); Morris elementary effects
                               (Campolongo 2007); derivative-based measures (DGSM, Sobol & Kucherenko 2009); the
                               moment-independent PAWN index (Pianosi & Wagener 2015).

Every model-based measure carries the cross-validated Q2 of the model behind it, and the consensus weighs it by
that score: a measure taken from a model that does not predict the runs it never saw says nothing about the
material. Written on numpy, scipy and scikit-learn only, because the offline bundle installs from the wheels it
ships with.
"""
from __future__ import annotations

import math
import time
import warnings

import numpy as np

# =========================================================================
# the data of a study
# =========================================================================


def _bounds_of(p):
    if p.get("values"):
        v = [float(x) for x in p["values"]]
        return min(v), max(v)
    lo, hi = float(p.get("min", 0.0)), float(p.get("max", 0.0))
    return min(lo, hi), max(lo, hi)


def dataset(study, rows, response, inputs=None, realised=True):
    """The finished cases of a study as X (one column per input) and y (the response).

    inputs: the parameter paths to use (None: all of the study's). realised: where a case recorded what its
    structure really holds for a parameter (a content the packing fell short of, a size realised), that value
    is the input rather than the one asked for - the DOE page reads a study against these too."""
    params = [p for p in (study.get("parameters") or []) if p.get("path")]
    if inputs:
        want = set(inputs)
        params = [p for p in params if p["path"] in want]
    X, y, labels, used_real = [], [], [], [False] * len(params)
    for r in rows:
        if r.get("state") != "done":
            continue
        met = r.get("metrics") or {}
        v = met.get(response)
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(v):
            continue
        xs = []
        for k, p in enumerate(params):
            x = (r.get("values") or {}).get(p["path"])
            rv = met.get("real_" + p["path"]) if realised else None
            if rv is not None:
                try:
                    if np.isfinite(float(rv)):
                        x = float(rv)
                        used_real[k] = True
                except (TypeError, ValueError):
                    pass
            xs.append(x)
        if any(x is None for x in xs):
            continue
        try:
            X.append([float(x) for x in xs])
        except (TypeError, ValueError):
            continue
        y.append(v)
        labels.append(r.get("label") or r.get("job_id") or "")
    X = np.asarray(X, float).reshape(len(X), len(params))
    info = []
    for k, p in enumerate(params):
        lo, hi = _bounds_of(p)
        if len(X):
            lo, hi = min(lo, float(X[:, k].min())), max(hi, float(X[:, k].max()))
        info.append({"path": p["path"], "name": p.get("label") or p["path"], "unit": p.get("unit", ""),
                     "lo": lo, "hi": hi, "realised": bool(used_real[k])})
    return X, np.asarray(y, float), info, labels


# =========================================================================
# small statistics helpers
# =========================================================================
def _rank(a, axis=0):
    from scipy.stats import rankdata
    return rankdata(a, axis=axis)


def _z(a):
    a = np.asarray(a, float)
    s = a.std(axis=0)
    return (a - a.mean(axis=0)) / np.where(s > 0, s, 1.0)


def _pearson_cols(X, y):
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    den = np.sqrt((Xc ** 2).sum(axis=0) * (yc ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (Xc * yc[:, None]).sum(axis=0) / den
    return np.where(np.isfinite(r), r, 0.0)


def _partial_cols(X, y):
    """Partial correlation of every column with y, the other columns held fixed (inverse correlation matrix)."""
    d = X.shape[1]
    if d == 1:
        return _pearson_cols(X, y)
    M = np.column_stack([X, y])
    s = M.std(axis=0)
    if np.any(s <= 0):
        return np.zeros(d)
    R = np.corrcoef(M, rowvar=False)
    P = np.linalg.pinv(R)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = -P[:d, d] / np.sqrt(np.abs(P[:d, :d].diagonal() * P[d, d]))
    return np.clip(np.where(np.isfinite(r), r, 0.0), -1.0, 1.0)


def _t_p(r, n, k):
    """Two-sided p of a correlation r of n points with k variables partialled out."""
    from scipy import stats
    df = n - 2 - k
    if df <= 0:
        return [None] * len(r)
    out = []
    for v in r:
        v = float(np.clip(v, -0.999999, 0.999999))
        t = v * math.sqrt(df / (1.0 - v * v))
        out.append(float(2.0 * stats.t.sf(abs(t), df)))
    return out


def _ci(samples, alpha=0.05):
    a = np.asarray(samples, float)
    lo = np.nanpercentile(a, 100 * alpha / 2, axis=0)
    hi = np.nanpercentile(a, 100 * (1 - alpha / 2), axis=0)
    return np.stack([lo, hi], axis=-1)


def _clean(v):
    """JSON-safe: numpy to Python, NaN/inf to None."""
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, np.ndarray):
        return _clean(v.tolist())
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


# =========================================================================
# 1. statistics on the runs
# =========================================================================
def correlations(X, y, boot=600, seed=0):
    """Pearson, Spearman, Kendall, PCC and PRCC with bootstrap 95 % intervals and p-values."""
    from scipy import stats
    n, d = X.shape
    Xr, yr = _rank(X), _rank(y)
    out = {}
    pear = _pearson_cols(X, y)
    spear = _pearson_cols(Xr, yr)
    kend, kp = [], []
    for i in range(d):
        try:
            t, p = stats.kendalltau(X[:, i], y)
        except Exception:                                           # noqa: BLE001
            t, p = 0.0, 1.0
        kend.append(0.0 if not np.isfinite(t) else float(t))
        kp.append(1.0 if not np.isfinite(p) else float(p))
    can_partial = n > d + 1
    pcc = _partial_cols(X, y) if can_partial else np.full(d, np.nan)
    prcc = _partial_cols(Xr, yr) if can_partial else np.full(d, np.nan)
    rng = np.random.default_rng(seed)
    B = {k: [] for k in ("pearson", "spearman", "kendall", "pcc", "prcc")}
    for _ in range(int(boot)):
        idx = rng.integers(0, n, n)
        Xb, yb = X[idx], y[idx]
        if np.std(yb) <= 0:
            continue
        Xbr, ybr = _rank(Xb), _rank(yb)
        B["pearson"].append(_pearson_cols(Xb, yb))
        B["spearman"].append(_pearson_cols(Xbr, ybr))
        # Kendall's tau from the sign agreement of all pairs (vectorised; ties count zero)
        dx = np.sign(Xb[:, None, :] - Xb[None, :, :])
        dy = np.sign(yb[:, None] - yb[None, :])
        num = (dx * dy[:, :, None]).sum(axis=(0, 1))
        den = np.sqrt((np.abs(dx)).sum(axis=(0, 1)) * np.abs(dy).sum())
        with np.errstate(invalid="ignore", divide="ignore"):
            B["kendall"].append(np.where(den > 0, num / den, 0.0))
        if can_partial:
            B["pcc"].append(_partial_cols(Xb, yb))
            B["prcc"].append(_partial_cols(Xbr, ybr))
    vals = {"pearson": pear, "spearman": spear, "kendall": np.array(kend), "pcc": pcc, "prcc": prcc}
    pv = {"pearson": _t_p(pear, n, 0), "spearman": _t_p(spear, n, 0), "kendall": kp,
          "pcc": _t_p(pcc, n, d - 1) if can_partial else [None] * d,
          "prcc": _t_p(prcc, n, d - 1) if can_partial else [None] * d}
    for k, v in vals.items():
        if k in ("pcc", "prcc") and not can_partial:
            out[k] = {"value": [None] * d, "ci": [[None, None]] * d, "p": [None] * d,
                      "skipped": f"needs more runs than inputs + 1 ({n} runs, {d} inputs)"}
            continue
        ci = _ci(B[k]) if B[k] else np.full((d, 2), np.nan)
        out[k] = {"value": v, "ci": ci, "p": pv[k]}
    return out


def regression(X, y, boot=600, seed=0):
    """Standardised regression coefficients (SRC) and their rank form (SRRC), with R2 and 95 % intervals."""
    from scipy import stats
    n, d = X.shape
    out = {}
    for name, (A, b) in {"src": (X, y), "srrc": (_rank(X), _rank(y))}.items():
        if n <= d + 1:
            out[name] = {"value": [None] * d, "ci": [[None, None]] * d, "r2": None,
                         "skipped": f"needs more runs than inputs + 1 ({n} runs, {d} inputs)"}
            continue
        Z, w = _z(A), _z(b)
        M = np.column_stack([np.ones(n), Z])
        coef, *_ = np.linalg.lstsq(M, w, rcond=None)
        res = w - M @ coef
        dfr = n - d - 1
        s2 = float(res @ res) / dfr
        cov = s2 * np.linalg.pinv(M.T @ M)
        se = np.sqrt(np.abs(np.diag(cov)))[1:]
        tq = stats.t.ppf(0.975, dfr)
        beta = coef[1:]
        ss = float((w ** 2).sum())
        r2 = 1.0 - float(res @ res) / ss if ss > 0 else None
        tv = beta / np.where(se > 0, se, np.inf)
        pv = [float(2 * stats.t.sf(abs(t), dfr)) for t in tv]
        out[name] = {"value": beta, "ci": np.stack([beta - tq * se, beta + tq * se], axis=-1), "r2": r2, "p": pv}
    return out


def _coded(X, lo, hi):
    span = np.where(hi > lo, hi - lo, 1.0)
    return 2.0 * (X - lo) / span - 1.0


def anova(X, y, names, lo, hi):
    """The analysis of variance of a response surface on coded inputs (-1 .. 1).

    The largest model the runs support: the full quadratic (main, quadratic and two-factor interaction terms),
    else main and interaction terms, else main and quadratic, else main only - the residual degrees of freedom
    kept at two or more. A term that the design cannot tell apart from the others (x^2 of a two-level factor) is
    dropped. Each term: coefficient, standard error, t, Type III sum of squares, F, p and its share of the total
    variation (eta^2). Without residual degrees of freedom, Lenth's pseudo standard error stands in."""
    from scipy import stats
    n, d = X.shape
    C = _coded(X, lo, hi)
    cand = {
        "full": [("main", (i,)) for i in range(d)] + [("quad", (i,)) for i in range(d)]
                + [("inter", (i, j)) for i in range(d) for j in range(i + 1, d)],
        "inter": [("main", (i,)) for i in range(d)] + [("inter", (i, j)) for i in range(d) for j in range(i + 1, d)],
        "quad": [("main", (i,)) for i in range(d)] + [("quad", (i,)) for i in range(d)],
        "main": [("main", (i,)) for i in range(d)],
    }

    def col(kind, ix):
        if kind == "main":
            return C[:, ix[0]]
        if kind == "quad":
            q = C[:, ix[0]] ** 2
            return q - q.mean()
        return C[:, ix[0]] * C[:, ix[1]]

    chosen = None
    for level in ("full", "inter", "quad", "main"):
        terms = []
        cols = [np.ones(n)]
        for kind, ix in cand[level]:
            c = col(kind, ix)
            if np.std(c) <= 1e-12:
                continue
            trial = np.column_stack(cols + [c])
            if np.linalg.matrix_rank(trial, tol=1e-9) < trial.shape[1]:
                continue
            cols.append(c)
            terms.append((kind, ix))
        if n - len(cols) >= 2 or level == "main":
            chosen = (level, terms, np.column_stack(cols))
            break
    level, terms, M = chosen
    p = M.shape[1]
    coef, *_ = np.linalg.lstsq(M, y, rcond=None)
    res = y - M @ coef
    sse = float(res @ res)
    sst = float(((y - y.mean()) ** 2).sum())
    dfr = n - p
    out_terms = []
    lenth = None
    if dfr > 0:
        mse = sse / dfr
        cov = mse * np.linalg.pinv(M.T @ M)
        se = np.sqrt(np.abs(np.diag(cov)))
        tcrit = float(stats.t.ppf(0.975, dfr))
    else:
        # Lenth (1989): the pseudo standard error of the effects of a saturated design
        c = np.abs(coef[1:])
        s0 = 1.5 * np.median(c) if len(c) else 0.0
        pse = 1.5 * np.median(c[c < 2.5 * s0]) if np.any(c < 2.5 * s0) else s0
        se = np.concatenate([[np.nan], np.full(len(c), pse if pse > 0 else np.nan)])
        dfl = max(1.0, len(c) / 3.0)
        tcrit = float(stats.t.ppf(0.975, dfl))
        lenth = {"pse": float(pse), "df": dfl}
    for k, (kind, ix) in enumerate(terms, start=1):
        Mr = np.delete(M, k, axis=1)
        cr, *_ = np.linalg.lstsq(Mr, y, rcond=None)
        rr = y - Mr @ cr
        ss = max(0.0, float(rr @ rr) - sse)
        t = coef[k] / se[k] if se[k] and np.isfinite(se[k]) and se[k] > 0 else None
        if dfr > 0:
            F = ss / (sse / dfr) if sse > 0 else None
            pval = float(stats.f.sf(F, 1, dfr)) if F is not None else 0.0
        else:
            F = None
            pval = float(2 * stats.t.sf(abs(t), lenth["df"])) if t is not None else None
        nm = (names[ix[0]] if kind == "main" else f"{names[ix[0]]}²" if kind == "quad"
              else f"{names[ix[0]]} × {names[ix[1]]}")
        sym = (f"x{ix[0] + 1}" if kind == "main" else f"x{ix[0] + 1}²" if kind == "quad"
               else f"x{ix[0] + 1}·x{ix[1] + 1}")
        out_terms.append({"kind": kind, "inputs": list(ix), "name": nm, "symbol": sym, "coef": float(coef[k]),
                          "se": None if not np.isfinite(se[k]) else float(se[k]), "t": t, "ss": ss, "F": F,
                          "p": pval, "eta2": ss / sst if sst > 0 else None})
    r2 = 1.0 - sse / sst if sst > 0 else None
    adj = 1.0 - (1.0 - r2) * (n - 1) / dfr if (r2 is not None and dfr > 0) else None
    press = None
    if dfr > 0:
        H = M @ np.linalg.pinv(M.T @ M) @ M.T
        h = np.clip(np.diag(H), 0, 0.999999)
        press = float(((res / (1.0 - h)) ** 2).sum())
    q2 = 1.0 - press / sst if (press is not None and sst > 0) else None
    # each input's share: its main and quadratic terms and half of every interaction it takes part in
    share = np.zeros(d)
    for t in out_terms:
        e = t["eta2"] or 0.0
        if t["kind"] == "inter":
            share[t["inputs"][0]] += e / 2
            share[t["inputs"][1]] += e / 2
        else:
            share[t["inputs"][0]] += e
    return {"model": level, "terms": out_terms, "r2": r2, "adj_r2": adj, "q2": q2, "df_resid": dfr,
            "t_crit": tcrit, "lenth": lenth, "intercept": float(coef[0]), "share": share,
            "residual_share": sse / sst if sst > 0 else None}


def correlation_ratio(X, y, max_bins=None):
    """The first-order index from the runs alone: Var(E[y | x_i]) / Var(y), x_i grouped by its levels (a few
    distinct values) or by quantile bins. A small-sample estimator; read it beside the surrogate's S1."""
    n, d = X.shape
    vy = float(np.var(y))
    out = np.zeros(d)
    nb = max_bins or max(2, int(round(math.sqrt(n))))
    for i in range(d):
        x = X[:, i]
        u = np.unique(x)
        if len(u) <= max(6, nb):
            groups = [y[x == v] for v in u]
        else:
            q = np.quantile(x, np.linspace(0, 1, nb + 1))
            b = np.clip(np.searchsorted(q, x, side="right") - 1, 0, nb - 1)
            groups = [y[b == k] for k in range(nb)]
        groups = [g for g in groups if len(g)]
        m = y.mean()
        between = sum(len(g) * (g.mean() - m) ** 2 for g in groups) / n
        # the bias of a group mean from few points (E[between] = true + (k-1) var/n): subtracted
        k = len(groups)
        within = sum(((g - g.mean()) ** 2).sum() for g in groups) / max(1, n - k)
        corr = max(0.0, between - (k - 1) * within / n)
        out[i] = corr / vy if vy > 0 else 0.0
    return np.clip(out, 0, 1)


def _dist_dcor(x, y):
    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    A = a - a.mean(0) - a.mean(1)[:, None] + a.mean()
    B = b - b.mean(0) - b.mean(1)[:, None] + b.mean()
    return A, B


def dependence(X, y, perms=299, seed=0):
    """Mutual information (k-nearest neighbours), distance correlation and the normalised HSIC of every input
    with the response - dependence of any shape, not only a trend - each with a permutation p-value."""
    from sklearn.feature_selection import mutual_info_regression
    n, d = X.shape
    rng = np.random.default_rng(seed)
    k = max(1, min(3, n - 2))
    Zx = _z(X)
    zy = _z(y.reshape(-1, 1))[:, 0]

    def mi(yv):
        return np.mean([mutual_info_regression(Zx, yv, n_neighbors=k, random_state=s) for s in range(3)], axis=0)
    mi0 = mi(zy)
    mi_null = np.array([mutual_info_regression(Zx, rng.permutation(zy), n_neighbors=k, random_state=0)
                        for _ in range(min(perms, 99))])
    mi_p = ((mi_null >= mi0[None, :]).sum(axis=0) + 1) / (len(mi_null) + 1)
    # distance correlation (Szekely 2007)
    dc, dcp = np.zeros(d), np.ones(d)
    hs, hsp = np.zeros(d), np.ones(d)
    _, By = _dist_dcor(zy, zy)
    vy = float((By * By).mean())

    def gram(v):
        dd = (v[:, None] - v[None, :]) ** 2
        med = np.median(dd[dd > 0]) if np.any(dd > 0) else 1.0
        K = np.exp(-dd / med)
        H = np.eye(len(v)) - 1.0 / len(v)
        return H @ K @ H
    Ly = gram(zy)
    hy = float((Ly * Ly).sum())
    for i in range(d):
        Ax, _ = _dist_dcor(Zx[:, i], Zx[:, i])
        vx = float((Ax * Ax).mean())
        dcov = float((Ax * By).mean())
        den = math.sqrt(max(vx * vy, 0.0))
        dc[i] = math.sqrt(max(dcov, 0.0) / den) if den > 0 else 0.0
        Kx = gram(Zx[:, i])
        hx = float((Kx * Kx).sum())
        h0 = float((Kx * Ly).sum())
        hs[i] = h0 / math.sqrt(hx * hy) if hx > 0 and hy > 0 else 0.0
        cnt_d, cnt_h = 1, 1
        for _ in range(perms):
            o = rng.permutation(n)
            Bp = By[np.ix_(o, o)]
            if float((Ax * Bp).mean()) >= dcov:
                cnt_d += 1
            Lp = Ly[np.ix_(o, o)]
            if float((Kx * Lp).sum()) >= h0:
                cnt_h += 1
        dcp[i] = cnt_d / (perms + 1)
        hsp[i] = cnt_h / (perms + 1)
    return {"mi": {"value": mi0, "p": mi_p}, "dcor": {"value": dc, "p": dcp}, "hsic": {"value": hs, "p": hsp}}


def lasso(X, y, seed=0):
    """The LASSO path on standardised inputs: the order in which inputs enter as the penalty falls is a ranking
    of its own; the penalty chosen by cross-validation gives the inputs a sparse linear model keeps."""
    from sklearn.linear_model import LassoCV, lasso_path
    n, d = X.shape
    Z, w = _z(X), _z(y.reshape(-1, 1))[:, 0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        alphas, coefs, _ = lasso_path(Z, w, n_alphas=60, eps=1e-3)
    enter = []
    for i in range(d):
        nz = np.flatnonzero(np.abs(coefs[i]) > 1e-10)
        enter.append(float(alphas[nz[0]]) if len(nz) else 0.0)
    best = None
    if n >= 8:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cv = LassoCV(cv=min(5, n), n_alphas=60, eps=1e-3, random_state=seed).fit(Z, w)
            best = {"alpha": float(cv.alpha_), "coef": cv.coef_}
        except Exception:                                            # noqa: BLE001
            best = None
    return {"alphas": alphas, "coefs": coefs, "enter": enter, "cv": best}


def main_effects(X, y, names, max_levels=7, bins=5):
    """The mean response at each level of an input (bins of equal count when it has many values), +- one
    standard error: the classic main-effects plot of a design of experiments."""
    n, d = X.shape
    out = []
    for i in range(d):
        x = X[:, i]
        u = np.unique(x)
        if len(u) <= max_levels:
            keys = [(float(v), x == v) for v in u]
        else:
            q = np.quantile(x, np.linspace(0, 1, bins + 1))
            b = np.clip(np.searchsorted(q, x, side="right") - 1, 0, bins - 1)
            keys = [(float(x[b == k].mean()), b == k) for k in range(bins) if np.any(b == k)]
        lv = []
        for xv, m in keys:
            g = y[m]
            lv.append({"x": xv, "mean": float(g.mean()), "se": float(g.std(ddof=1) / math.sqrt(len(g))) if len(g) > 1 else None,
                       "n": int(len(g))})
        out.append(lv)
    return out


# =========================================================================
# 2. models, scored by cross-validation
# =========================================================================
class _Unit:
    def __init__(self, lo, hi):
        self.lo = np.asarray(lo, float)
        self.span = np.where(np.asarray(hi, float) > self.lo, np.asarray(hi, float) - self.lo, 1.0)

    def __call__(self, X):
        return (np.atleast_2d(np.asarray(X, float)) - self.lo) / self.span


def _fit_gp(U, y, seed, restarts=3):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    d = U.shape[1]
    k = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.full(d, 0.5), length_scale_bounds=(1e-2, 1e2),
                                                  nu=2.5) + WhiteKernel(1e-4, (1e-10, 1e-1))
    gp = GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=restarts, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        gp.fit(U, y)
    return gp


def _gp_scales(gp, d):
    try:
        for name, val in gp.kernel_.get_params().items():
            if name.endswith("length_scale") and np.ndim(val) == 1 and len(val) == d:
                return np.asarray(val, float)
    except Exception:                                                # noqa: BLE001
        pass
    return None


def _pce_basis(U, alphas):
    """Orthonormal Legendre polynomials on [0, 1] for the multi-indices `alphas`."""
    from numpy.polynomial import legendre as LG
    Xs = 2.0 * np.atleast_2d(U) - 1.0
    deg = int(max((max(a) for a in alphas), default=0))
    P = np.empty((Xs.shape[0], Xs.shape[1], deg + 1))
    for k in range(deg + 1):
        c = np.zeros(k + 1)
        c[k] = 1.0
        P[:, :, k] = LG.legval(Xs, c) * math.sqrt(2 * k + 1)
    out = np.ones((Xs.shape[0], len(alphas)))
    for j, a in enumerate(alphas):
        for i, k in enumerate(a):
            if k:
                out[:, j] *= P[:, i, k]
    return out


def _multi_indices(d, p, q=1.0):
    """Total degree <= p (hyperbolic q-norm truncation for q < 1), constant first."""
    out = [tuple([0] * d)]

    def rec(prefix, left):
        if len(prefix) == d:
            a = tuple(prefix)
            if sum(a) > 0 and (sum(v ** q for v in a)) ** (1.0 / q) <= p + 1e-9:
                out.append(a)
            return
        for k in range(left + 1):
            rec(prefix + [k], left - k)
    rec([], p)
    seen, uniq = set(), []
    for a in out:
        if a not in seen:
            seen.add(a)
            uniq.append(a)
    uniq.sort(key=lambda a: (sum(a), a[::-1]))
    return uniq


def _fit_pce(U, y, seed=0, max_degree=4):
    """A sparse polynomial chaos expansion (least-angle regression on Legendre polynomials, then least squares
    on the terms it selects: hybrid LAR, Blatman & Sudret 2011); the degree with the best leave-one-out Q2."""
    from sklearn.linear_model import LarsCV
    n, d = U.shape
    best = None
    for p in range(1, max_degree + 1):
        al = _multi_indices(d, p, q=1.0 if d <= 4 else 0.75)
        Psi = _pce_basis(U, al)
        if Psi.shape[1] <= n - 1:
            keep = list(range(Psi.shape[1]))
        else:
            if n < 6:
                break
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    lars = LarsCV(cv=min(5, n), fit_intercept=True, max_n_alphas=200).fit(Psi[:, 1:], y)
                keep = [0] + [j + 1 for j in np.flatnonzero(np.abs(lars.coef_) > 1e-12)]
            except Exception:                                        # noqa: BLE001
                continue
            if len(keep) > n - 1:
                keep = keep[:n - 1]
        A = Psi[:, keep]
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        res = y - A @ c
        G = np.linalg.pinv(A.T @ A)
        h = np.clip(np.einsum("ij,jk,ik->i", A, G, A), 0, 0.999)
        sst = float(((y - y.mean()) ** 2).sum())
        q2 = 1.0 - float(((res / (1 - h)) ** 2).sum()) / sst if sst > 0 else -np.inf
        if best is None or q2 > best["loo_q2"] + 1e-9:
            best = {"alphas": [al[j] for j in keep], "coef": c, "degree": p, "loo_q2": q2}
    if best is None:
        al = _multi_indices(d, 1)
        A = _pce_basis(U, al)
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        best = {"alphas": al, "coef": c, "degree": 1, "loo_q2": None}
    return best


def pce_sobol(pce, d):
    """Sobol indices of a polynomial chaos expansion in closed form: each squared coefficient is the variance
    of its term, which belongs to exactly the inputs whose degree in it is non-zero."""
    al, c = pce["alphas"], np.asarray(pce["coef"], float)
    V = sum(c[j] ** 2 for j, a in enumerate(al) if sum(a) > 0)
    S1, ST, S2 = np.zeros(d), np.zeros(d), np.zeros((d, d))
    if V <= 0:
        return {"S1": S1, "ST": ST, "S2": S2, "variance": 0.0}
    for j, a in enumerate(al):
        sup = [i for i, k in enumerate(a) if k]
        if not sup:
            continue
        v = c[j] ** 2 / V
        for i in sup:
            ST[i] += v
        if len(sup) == 1:
            S1[sup[0]] += v
        elif len(sup) == 2:
            S2[sup[0], sup[1]] += v
            S2[sup[1], sup[0]] += v
    return {"S1": S1, "ST": ST, "S2": S2, "variance": float(V), "mean": float(c[0]),
            "n_terms": int(len(al)), "degree": int(pce["degree"])}


class Models:
    """The surrogates of one response, each fitted to every run and scored by cross-validation on the same
    folds (leave-one-out up to 12 runs, five folds above)."""

    KINDS = ("linear", "quadratic", "gp", "rf", "gbm", "pce")
    LABELS = {"linear": "Linear regression", "quadratic": "Quadratic response surface",
              "gp": "Gaussian process (Matérn 5/2, ARD)", "rf": "Random forest", "gbm": "Gradient boosting",
              "pce": "Polynomial chaos expansion"}

    def __init__(self, X, y, lo, hi, seed=0):
        self.X, self.y = X, y
        self.n, self.d = X.shape
        self.unit = _Unit(lo, hi)
        self.seed = int(seed)
        self.fitted = {}
        self.cv = {}

    # ---------------------------------------------------------- builders
    def _build(self, kind, X, y):
        U = self.unit(X)
        n, d = U.shape
        if kind == "linear":
            A = np.column_stack([np.ones(n), U])
            c, *_ = np.linalg.lstsq(A, y, rcond=None)
            return lambda Q: np.column_stack([np.ones(len(Q)), self.unit(Q)]) @ c
        if kind == "quadratic":
            def design(V):
                cols = [np.ones(len(V))] + [V[:, i] for i in range(d)]
                for i in range(d):
                    for j in range(i, d):
                        cols.append(V[:, i] * V[:, j])
                return np.column_stack(cols)
            A = design(U)
            if A.shape[1] < n:
                c, *_ = np.linalg.lstsq(A, y, rcond=None)
            else:
                # more terms than runs: ridge, its penalty by generalised cross-validation
                from sklearn.linear_model import RidgeCV
                r = RidgeCV(alphas=np.logspace(-6, 2, 30), fit_intercept=False).fit(A, y)
                c = r.coef_
            return lambda Q: design(self.unit(Q)) @ c
        if kind == "gp":
            gp = _fit_gp(U, y, self.seed, restarts=3 if len(y) == self.n else 1)
            f = lambda Q: gp.predict(self.unit(Q))                  # noqa: E731
            f.gp = gp
            return f
        if kind == "rf":
            from sklearn.ensemble import RandomForestRegressor
            rf = RandomForestRegressor(n_estimators=300, max_features=0.7, min_samples_leaf=1, random_state=self.seed,
                                       n_jobs=-1).fit(U, y)

            def f(Q):
                # threads pay off on a large batch only; a few rows are faster on one
                Q = self.unit(Q)
                rf.n_jobs = -1 if len(Q) > 20000 else 1
                return rf.predict(Q)
            f.rf = rf
            return f
        if kind == "gbm":
            from sklearn.ensemble import GradientBoostingRegressor
            g = GradientBoostingRegressor(n_estimators=400, learning_rate=0.04, max_depth=3,
                                          subsample=0.8 if n >= 20 else 1.0,
                                          min_samples_leaf=max(1, n // 25), random_state=self.seed).fit(U, y)
            return lambda Q: g.predict(self.unit(Q))
        if kind == "pce":
            pce = _fit_pce(U, y, self.seed)
            f = lambda Q: _pce_basis(self.unit(Q), pce["alphas"]) @ pce["coef"]   # noqa: E731
            f.pce = pce
            return f
        raise KeyError(kind)

    def folds(self):
        n = self.n
        k = n if n <= 12 else 5
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(n)
        return [(np.setdiff1d(idx, part), part) for part in np.array_split(idx, k)]

    def fit_all(self, kinds=None, log=None):
        kinds = kinds or self.KINDS
        folds = self.folds()
        sst = float(((self.y - self.y.mean()) ** 2).sum())
        for kind in kinds:
            if kind in ("quadratic",) and self.n < 4:
                continue
            t0 = time.time()
            try:
                self.fitted[kind] = self._build(kind, self.X, self.y)
            except Exception as e:                                   # noqa: BLE001
                self.cv[kind] = {"error": f"{type(e).__name__}: {e}"}
                continue
            pred = np.full(self.n, np.nan)
            try:
                for tr, te in folds:
                    if len(tr) < 3:
                        raise ValueError("too few runs for cross-validation")
                    f = self._build(kind, self.X[tr], self.y[tr])
                    pred[te] = f(self.X[te])
                q2 = 1.0 - float(((self.y - pred) ** 2).sum()) / sst if sst > 0 else None
                rmse = float(np.sqrt(np.mean((self.y - pred) ** 2)))
                fit = self.fitted[kind](self.X)
                r2 = 1.0 - float(((self.y - fit) ** 2).sum()) / sst if sst > 0 else None
                self.cv[kind] = {"q2": q2, "rmse": rmse, "r2_train": r2, "pred": pred, "folds": len(folds),
                                 "seconds": time.time() - t0}
            except Exception as e:                                   # noqa: BLE001
                self.cv[kind] = {"error": f"{type(e).__name__}: {e}", "q2": None}
            if log:
                log(f"   {kind}: Q2 {self.cv[kind].get('q2')}")

    def q2(self, kind):
        v = (self.cv.get(kind) or {}).get("q2")
        return None if v is None or not np.isfinite(v) else float(v)

    def best(self, among):
        cands = [(self.q2(k), k) for k in among if k in self.fitted and self.q2(k) is not None]
        if not cands:
            return next((k for k in among if k in self.fitted), None)
        top = max(q for q, _ in cands)
        # the Gaussian process wins a near tie: smooth, and the usual surrogate of physics responses
        for pref in ("gp", "pce", "quadratic", "gbm", "rf", "linear"):
            if pref in among and self.q2(pref) is not None and self.q2(pref) >= top - 0.02:
                return pref
        return max(cands)[1]


def permutation_importance_cv(models, kind, repeats=5, outer=2):
    """The rise of the cross-validated error when one input is shuffled in the held-out runs, as a share of
    the response's variance (Breiman 2001, on held-out data so that an overfitted model cannot score well)."""
    rng = np.random.default_rng(models.seed + 7)
    X, y, d = models.X, models.y, models.d
    vy = float(np.var(y)) or 1.0
    draws = []
    for o in range(outer):
        folds = models.folds() if o == 0 else _reshuffle(models, o)
        for tr, te in folds:
            if len(tr) < 3 or len(te) < 1:
                continue
            f = models._build(kind, X[tr], y[tr])
            base = float(np.mean((f(X[te]) - y[te]) ** 2))
            # one shuffle of a single held-out run changes nothing: the held-out runs are shuffled with the
            # training runs' values of that input (the reference distribution of the input)
            for _ in range(repeats):
                row = np.zeros(d)
                for i in range(d):
                    Xp = X[te].copy()
                    Xp[:, i] = rng.choice(X[:, i], size=len(te), replace=True)
                    row[i] = (float(np.mean((f(Xp) - y[te]) ** 2)) - base) / vy
                draws.append(row)
    return np.asarray(draws)


def _reshuffle(models, o):
    n = models.n
    k = n if n <= 12 else 5
    rng = np.random.default_rng(models.seed + 101 * o)
    idx = rng.permutation(n)
    return [(np.setdiff1d(idx, part), part) for part in np.array_split(idx, k)]


# =========================================================================
# 3. explaining a model: Shapley values, partial dependence, ALE, H
# =========================================================================
def shapley(predict, X, background, budget=1.6e6, n_perm=96, seed=0):
    """Interventional Shapley values of every run (SHAP, Lundberg & Lee 2017) and, when the coalitions can be
    enumerated (up to 10 inputs), the Shapley interaction values (Lundberg et al. 2020) from the same table.

    v(S)(x) = mean over the background runs b of f(x_S, b_rest); exact over all 2^d coalitions. The interaction
    value of i and j splits their joint effect evenly between the two; the main effect of i is what is left of
    its Shapley value."""
    n, d = X.shape
    m = len(background)
    if d <= 10:
        ns = 2 ** d
        m = int(np.clip(budget / max(1, ns * n), 6, m))
        rng = np.random.default_rng(seed)
        B = background[rng.choice(len(background), m, replace=False)] if m < len(background) else background
        m = len(B)
        V = np.empty((ns, n))
        base = np.repeat(B[None, :, :], n, axis=0)
        for s in range(ns):
            on = np.array([(s >> k) & 1 for k in range(d)], bool)
            Z = base.copy()
            if on.any():
                Z[:, :, on] = X[:, None, on]
            V[s] = np.asarray(predict(Z.reshape(-1, d)), float).reshape(n, m).mean(axis=1)
        pop = np.array([bin(s).count("1") for s in range(ns)])
        fa = math.factorial
        w = np.array([fa(k) * fa(d - k - 1) / fa(d) for k in range(d)])
        phi = np.zeros((n, d))
        for i in range(d):
            bit = 1 << i
            S = np.array([s for s in range(ns) if not s & bit])
            phi[:, i] = (w[pop[S]][:, None] * (V[S | bit] - V[S])).sum(axis=0)
        inter = None
        if d >= 2:
            wij = np.array([fa(k) * fa(d - k - 2) / (2.0 * fa(d - 1)) for k in range(d - 1)])
            inter = np.zeros((n, d, d))
            for i in range(d):
                for j in range(i + 1, d):
                    bi, bj = 1 << i, 1 << j
                    S = np.array([s for s in range(ns) if not (s & bi) and not (s & bj)])
                    delta = V[S | bi | bj] - V[S | bi] - V[S | bj] + V[S]
                    v = (wij[pop[S]][:, None] * delta).sum(axis=0)
                    inter[:, i, j] = v
                    inter[:, j, i] = v
            for i in range(d):
                inter[:, i, i] = phi[:, i] - inter[:, i, :].sum(axis=1) + inter[:, i, i]
        full = (1 << d) - 1
        # the partial dependences at the runs come from the same table: v({i}), v({i, j}), v(all but i)
        pd1 = np.stack([V[1 << i] for i in range(d)], axis=1)
        pd2 = {(i, j): V[(1 << i) | (1 << j)] for i in range(d) for j in range(i + 1, d)}
        pdm = np.stack([V[full ^ (1 << i)] for i in range(d)], axis=1)
        return {"phi": phi, "base": float(V[0].mean()), "inter": inter, "exact": True, "background": int(m),
                "f": V[full], "pd1": pd1, "pd2": pd2, "pdm": pdm}
    # more inputs: Monte Carlo over orderings (Strumbelj & Kononenko 2014)
    rng = np.random.default_rng(seed)
    m = int(min(m, max(6, budget / max(1, n_perm * (d + 1) * n))))
    B = background[rng.choice(len(background), m, replace=False)] if m < len(background) else background
    m = len(B)
    phi = np.zeros((n, d))
    for _ in range(n_perm):
        order = rng.permutation(d)
        Z = np.repeat(B[None, :, :], n, axis=0)
        prev = np.asarray(predict(Z.reshape(-1, d)), float).reshape(n, m).mean(axis=1)
        for i in order:
            Z[:, :, i] = X[:, None, i]
            cur = np.asarray(predict(Z.reshape(-1, d)), float).reshape(n, m).mean(axis=1)
            phi[:, i] += cur - prev
            prev = cur
    phi /= n_perm
    base = float(np.asarray(predict(B), float).mean())
    return {"phi": phi, "base": base, "inter": None, "exact": False, "background": int(m), "permutations": n_perm}


def h_statistic(sh, X, predict, background):
    """Friedman & Popescu's H^2 of every pair (the share of the joint partial dependence that is not the sum of
    the two single ones) and of every input with all the others, from the partial dependences at the runs."""
    n, d = X.shape
    if sh.get("pd1") is None:
        # build the partial dependences directly (more than 10 inputs)
        m = min(len(background), 32)
        B = background[:m]

        def pd(cols):
            Z = np.repeat(B[None, :, :], n, axis=0)
            Z[:, :, cols] = X[:, None, cols]
            return np.asarray(predict(Z.reshape(-1, d)), float).reshape(n, m).mean(axis=1)
        pd1 = np.stack([pd([i]) for i in range(d)], axis=1)
        pd2 = {(i, j): pd([i, j]) for i in range(d) for j in range(i + 1, d)}
        pdm = np.stack([pd([k for k in range(d) if k != i]) for i in range(d)], axis=1)
        f = np.asarray(predict(X), float)
    else:
        pd1, pd2, pdm, f = sh["pd1"], sh["pd2"], sh["pdm"], sh["f"]
    c = lambda v: v - v.mean()                                       # noqa: E731
    H2 = np.zeros((d, d))
    for (i, j), v in pd2.items():
        num = float((c(v) - c(pd1[:, i]) - c(pd1[:, j])) ** 2 @ np.ones(n))
        den = float(c(v) @ c(v))
        H2[i, j] = H2[j, i] = num / den if den > 1e-30 else 0.0
    Ht = np.zeros(d)
    fc = c(f)
    den = float(fc @ fc)
    for i in range(d):
        num = float(((fc - c(pd1[:, i]) - c(pdm[:, i])) ** 2).sum())
        Ht[i] = num / den if den > 1e-30 else 0.0
    return {"pair": np.clip(H2, 0, 1), "total": np.clip(Ht, 0, 1)}


def pdp_ice(predict, X, lo, hi, grid=30, max_ice=60, seed=0):
    """Partial dependence (the mean of the ICE curves) of every input over its range, the ICE curves of the
    runs (Goldstein et al. 2015) and the 10-90 % band they span."""
    n, d = X.shape
    rng = np.random.default_rng(seed)
    pick = np.arange(n) if n <= max_ice else rng.choice(n, max_ice, replace=False)
    out = []
    for i in range(d):
        g = np.linspace(lo[i], hi[i], grid)
        Z = np.repeat(X[None, :, :], grid, axis=0)
        Z[:, :, i] = g[:, None]
        F = np.asarray(predict(Z.reshape(-1, d)), float).reshape(grid, n).T     # n x grid
        out.append({"x": g, "pd": F.mean(axis=0), "lo": np.percentile(F, 10, axis=0),
                    "hi": np.percentile(F, 90, axis=0), "ice": F[pick]})
    return out


def pdp2(predict, X, i, j, lo, hi, grid=26, max_bg=48):
    n, d = X.shape
    B = X if n <= max_bg else X[np.linspace(0, n - 1, max_bg).astype(int)]
    gi, gj = np.linspace(lo[i], hi[i], grid), np.linspace(lo[j], hi[j], grid)
    Gi, Gj = np.meshgrid(gi, gj, indexing="ij")
    Z = np.repeat(B[None, :, :], grid * grid, axis=0)
    Z[:, :, i] = Gi.ravel()[:, None]
    Z[:, :, j] = Gj.ravel()[:, None]
    F = np.asarray(predict(Z.reshape(-1, d)), float).reshape(grid * grid, len(B)).mean(axis=1).reshape(grid, grid)
    return {"i": int(i), "j": int(j), "xi": gi, "xj": gj, "z": F}


def ale(predict, X, lo, hi, bins=8):
    """Accumulated local effects (Apley & Zhu 2020): the effect of an input from small moves inside the runs'
    own neighbourhoods, which stays honest when the inputs are correlated (the partial dependence does not)."""
    n, d = X.shape
    out = []
    for i in range(d):
        x = X[:, i]
        u = np.unique(x)
        if len(u) <= bins + 1:
            edges = u
        else:
            edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
        if len(edges) < 2:
            out.append({"x": [float(lo[i]), float(hi[i])], "ale": [0.0, 0.0]})
            continue
        k = np.clip(np.searchsorted(edges, x, side="left"), 1, len(edges) - 1)
        Zl, Zh = X.copy(), X.copy()
        Zl[:, i] = edges[k - 1]
        Zh[:, i] = edges[k]
        diff = np.asarray(predict(Zh), float) - np.asarray(predict(Zl), float)
        eff = np.zeros(len(edges) - 1)
        cnt = np.zeros(len(edges) - 1)
        for b in range(len(edges) - 1):
            m = k == b + 1
            cnt[b] = m.sum()
            eff[b] = diff[m].mean() if m.any() else 0.0
        acc = np.concatenate([[0.0], np.cumsum(eff)])
        # centred on the runs' own distribution
        mid = 0.5 * (acc[1:] + acc[:-1])
        acc -= float((mid * cnt).sum() / max(cnt.sum(), 1))
        out.append({"x": edges, "ale": acc})
    return out


# =========================================================================
# 4. variance-based and screening measures on a surrogate (inputs uniform over their ranges)
# =========================================================================
def _qmc(n, d, seed):
    try:
        from scipy.stats import qmc
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return qmc.Sobol(d, scramble=True, seed=seed).random(n)
    except Exception:                                                # noqa: BLE001
        return np.random.default_rng(seed).random((n, d))


def sobol(f_unit, d, N=2048, boot=200, seed=0, second=True):
    """First, total (Jansen) and second-order Sobol indices (Saltelli 2002, 2010) of f on the unit cube, with
    bootstrap 95 % intervals. f_unit takes points of the unit cube."""
    AB = _qmc(N, 2 * d, seed)
    A, B = AB[:, :d], AB[:, d:]
    fA = np.asarray(f_unit(A), float)
    fB = np.asarray(f_unit(B), float)
    fABi, fBAi = [], []
    for i in range(d):
        Ci = A.copy()
        Ci[:, i] = B[:, i]
        fABi.append(np.asarray(f_unit(Ci), float))
        if second:
            Di = B.copy()
            Di[:, i] = A[:, i]
            fBAi.append(np.asarray(f_unit(Di), float))
    mu = float(np.mean(np.concatenate([fA, fB])))
    fA, fB = fA - mu, fB - mu
    fABi = [v - mu for v in fABi]
    fBAi = [v - mu for v in fBAi]

    def est(idx):
        a, b = fA[idx], fB[idx]
        V = float(np.var(np.concatenate([a, b])))
        if V <= 0:
            return None
        s1 = np.array([np.mean(b * (fABi[i][idx] - a)) / V for i in range(d)])
        st = np.array([0.5 * np.mean((a - fABi[i][idx]) ** 2) / V for i in range(d)])
        s2 = np.zeros((d, d))
        if second:
            for i in range(d):
                for j in range(i + 1, d):
                    vij = np.mean(fBAi[i][idx] * fABi[j][idx] - a * b) / V
                    s2[i, j] = s2[j, i] = vij - s1[i] - s1[j]
        return s1, st, s2, V
    full = est(np.arange(N))
    if full is None:
        return None
    rng = np.random.default_rng(seed + 1)
    bs = [est(rng.integers(0, N, N)) for _ in range(int(boot))]
    bs = [b for b in bs if b is not None]
    out = {"S1": full[0], "ST": full[1], "variance": full[3], "N": int(N), "evaluations": int(N * (2 * d + 2)),
           "S1_ci": _ci([b[0] for b in bs]) if bs else None, "ST_ci": _ci([b[1] for b in bs]) if bs else None}
    if second:
        out["S2"] = full[2]
        out["S2_ci"] = _ci([b[2] for b in bs]) if bs else None
    return out


def efast(f_unit, d, N=1025, M=4, seed=0):
    """The extended Fourier amplitude sensitivity test (Saltelli, Tarantola & Chan 1999): first-order and total
    indices from the spectrum of f along a search curve that runs each input at its own frequency."""
    if N < 4 * M * M + 1:
        N = 4 * M * M + 1
    om0 = math.floor((N - 1) / (2 * M))
    m = math.floor(om0 / (2 * M))
    omega = np.zeros(d)
    omega[0] = om0
    if d > 1:
        omega[1:] = np.floor(np.linspace(1, m, d - 1)) if m >= d - 1 else (np.arange(d - 1) % max(m, 1)) + 1
    s = (2 * math.pi / N) * np.arange(N)
    rng = np.random.default_rng(seed)
    S1, ST = np.zeros(d), np.zeros(d)
    for i in range(d):
        om = np.zeros(d)
        om[i] = omega[0]
        rest = [k for k in range(d) if k != i]
        om[rest] = omega[1:]
        phi = 2 * math.pi * rng.random(d)
        U = 0.5 + np.arcsin(np.sin(om[None, :] * s[:, None] + phi[None, :])) / math.pi
        Y = np.asarray(f_unit(U), float)
        f = np.fft.fft(Y)
        Sp = (np.abs(f[1:(N + 1) // 2]) / N) ** 2
        V = 2 * Sp.sum()
        if V <= 0:
            continue
        D1 = 2 * Sp[np.arange(1, M + 1) * int(om0) - 1].sum()
        Dt = 2 * Sp[:int(om0 // 2)].sum()
        S1[i] = D1 / V
        ST[i] = 1 - Dt / V
    return {"S1": np.clip(S1, 0, 1), "ST": np.clip(ST, 0, 1), "N": int(N), "M": int(M), "evaluations": int(N * d)}


def morris(f_unit, d, r=60, levels=8, boot=300, seed=0):
    """Morris elementary effects (Morris 1991; mu* of Campolongo et al. 2007): r one-at-a-time trajectories
    across the unit cube. mu* is the mean absolute effect of moving an input over its whole range, sigma their
    spread - large where the effect depends on the input's own value (curvature) or on the others (interaction)."""
    rng = np.random.default_rng(seed)
    delta = levels / (2.0 * (levels - 1))
    grid = np.arange(levels) / (levels - 1)
    EE = np.zeros((r, d))
    pts = []
    plan = []
    for t in range(r):
        x0 = rng.choice(grid[grid <= 1 - delta + 1e-12], d)
        sgn = np.where(rng.random(d) < 0.5, 1.0, -1.0)
        x0 = np.where(sgn < 0, x0 + delta, x0)
        order = rng.permutation(d)
        x = x0.copy()
        traj = [x.copy()]
        for i in order:
            x = x.copy()
            x[i] += sgn[i] * delta
            traj.append(x.copy())
        pts.append(np.array(traj))
        plan.append((order, sgn))
    P = np.concatenate(pts)
    F = np.asarray(f_unit(np.clip(P, 0, 1)), float).reshape(r, d + 1)
    for t, (order, sgn) in enumerate(plan):
        for k, i in enumerate(order):
            EE[t, i] = (F[t, k + 1] - F[t, k]) / (sgn[i] * delta)
    mu_star = np.abs(EE).mean(axis=0)
    bs = []
    for _ in range(int(boot)):
        idx = rng.integers(0, r, r)
        bs.append(np.abs(EE[idx]).mean(axis=0))
    return {"mu_star": mu_star, "mu": EE.mean(axis=0), "sigma": EE.std(axis=0, ddof=1),
            "mu_star_ci": _ci(bs), "r": int(r), "levels": int(levels), "evaluations": int(r * (d + 1))}


def dgsm(f_unit, d, N=1024, h=1e-3, seed=0, variance=None):
    """Derivative-based global sensitivity measures nu_i = E[(df/du_i)^2] on the unit cube (Sobol & Kucherenko
    2009), with the upper bound they put on the total Sobol index: ST_i <= nu_i / (pi^2 V)."""
    U = _qmc(N, d, seed + 3)
    f0 = np.asarray(f_unit(U), float)
    V = float(variance) if variance else float(np.var(f0))
    nu = np.zeros(d)
    for i in range(d):
        up, dn = U.copy(), U.copy()
        up[:, i] = np.clip(U[:, i] + h, 0, 1)
        dn[:, i] = np.clip(U[:, i] - h, 0, 1)
        g = (np.asarray(f_unit(up), float) - np.asarray(f_unit(dn), float)) / (up[:, i] - dn[:, i])
        nu[i] = float(np.mean(g ** 2))
    bound = nu / (math.pi ** 2 * V) if V > 0 else np.zeros(d)
    return {"nu": nu, "bound": bound, "N": int(N), "evaluations": int(N * (2 * d + 1))}


def pawn(f_unit, d, N=8192, slices=10, boot=100, seed=0):
    """The moment-independent PAWN index (Pianosi & Wagener 2015): the Kolmogorov-Smirnov distance between the
    distribution of the response and its distributions with one input held within a slice of its range,
    summarised by the median (and the maximum) over the slices. It reads the whole distribution, not the
    variance alone. ks_crit: the distance two samples of these sizes reach by chance (5 %)."""
    U = _qmc(N, d, seed + 5)
    Y = np.asarray(f_unit(U), float)
    grid = np.quantile(Y, np.linspace(0, 1, 201))

    def cdf(v):
        return np.searchsorted(np.sort(v), grid, side="right") / len(v)

    def index(idx):
        Ys = Y[idx]
        Fu = cdf(Ys)
        med, mx = np.zeros(d), np.zeros(d)
        for i in range(d):
            x = U[idx, i]
            ks = []
            for k in range(slices):
                m = (x >= k / slices) & (x < (k + 1) / slices)
                if m.sum() < 5:
                    continue
                ks.append(np.abs(cdf(Ys[m]) - Fu).max())
            med[i] = np.median(ks) if ks else 0.0
            mx[i] = np.max(ks) if ks else 0.0
        return med, mx
    med, mx = index(np.arange(N))
    rng = np.random.default_rng(seed + 9)
    bs = [index(rng.integers(0, N, N))[0] for _ in range(int(boot))]
    nc = N / slices
    crit = 1.358 * math.sqrt((nc + N) / (nc * N))
    return {"median": med, "max": mx, "median_ci": _ci(bs), "ks_crit": crit, "N": int(N), "slices": int(slices)}


# =========================================================================
# 5. what the methods agree on
# =========================================================================
def _kendall_w(R):
    """Kendall's coefficient of concordance of the rankings in the rows of R (methods x inputs)."""
    m, k = R.shape
    if m < 2 or k < 2:
        return None
    Rs = R.sum(axis=0)
    S = float(((Rs - Rs.mean()) ** 2).sum())
    return 12.0 * S / (m * m * (k ** 3 - k))


def _shape(pd_curve):
    """increasing, decreasing, peaked, valley or flat, from a partial dependence curve."""
    y = np.asarray(pd_curve, float)
    rng = float(y.max() - y.min())
    if rng <= 1e-12 * (abs(float(y.mean())) + 1e-30):
        return "flat"
    dy = np.diff(y)
    up, dn = float(dy[dy > 0].sum()), float(-dy[dy < 0].sum())
    tot = up + dn
    if up >= 0.9 * tot:
        return "increasing"
    if dn >= 0.9 * tot:
        return "decreasing"
    k = int(np.argmax(y))
    if 0 < k < len(y) - 1 and (y[k] - y[0]) > 0.15 * rng and (y[k] - y[-1]) > 0.15 * rng:
        return "peaked"
    k = int(np.argmin(y))
    if 0 < k < len(y) - 1 and (y[0] - y[k]) > 0.15 * rng and (y[-1] - y[k]) > 0.15 * rng:
        return "valley"
    return "increasing" if up > dn else "decreasing"


# the methods of the consensus: (id, label, family, what it reads, the model behind it or None)
METHODS = [
    ("pearson", "Pearson |r|", "Correlation", "linear trend", None),
    ("spearman", "Spearman |ρ|", "Correlation", "monotonic trend", None),
    ("kendall", "Kendall |τ|", "Correlation", "monotonic trend (pairs)", None),
    ("pcc", "PCC |r|", "Correlation", "linear, others held fixed", None),
    ("prcc", "PRCC |r|", "Correlation", "monotonic, others held fixed", None),
    ("src", "SRC |β|", "Regression", "standardised linear effect", None),
    ("srrc", "SRRC |β|", "Regression", "standardised rank effect", None),
    ("anova", "ANOVA η²", "Regression", "variance of the response surface", None),
    ("corr_ratio", "Correlation ratio η²", "Dependence", "first-order index from the runs", None),
    ("mi", "Mutual information", "Dependence", "dependence of any shape", None),
    ("dcor", "Distance correlation", "Dependence", "dependence of any shape", None),
    ("hsic", "HSIC", "Dependence", "dependence of any shape (kernel)", None),
    ("lasso", "LASSO entry", "Regression", "order of entry along the path", None),
    ("gp_ard", "GP 1/ℓ (ARD)", "Machine learning", "how fast the response changes", "gp"),
    ("rf_mdi", "RF impurity (MDI)", "Machine learning", "split gain in the forest", "rf"),
    ("rf_perm", "RF permutation", "Machine learning", "error rise on held-out runs", "rf"),
    ("gbm_perm", "GBM permutation", "Machine learning", "error rise on held-out runs", "gbm"),
    ("shap", "mean |SHAP|", "Machine learning", "average contribution per run", "explainer"),
    ("sobol_s1", "Sobol S₁", "Variance-based", "variance it explains alone", "surrogate"),
    ("sobol_st", "Sobol S_T", "Variance-based", "variance it takes part in", "surrogate"),
    ("pce_st", "PCE S_T", "Variance-based", "total index, closed form", "pce"),
    ("efast_st", "eFAST S_T", "Variance-based", "total index, Fourier", "surrogate"),
    ("morris", "Morris μ*", "Screening", "mean absolute elementary effect", "surrogate"),
    ("dgsm", "DGSM ν", "Screening", "mean squared derivative", "surrogate"),
    ("pawn", "PAWN", "Moment-independent", "shift of the whole distribution", "surrogate"),
    ("h_total", "H-statistic (total)", "Interaction", "share in interactions", "explainer"),
]


def consensus(scores, weights, d, family=None):
    """Each method's importances scaled to 1 at its largest; the consensus is their weighted mean, and the ranks
    a Borda count. A method's weight is the cross-validated Q2 of the model behind it (1 for a statistic of the
    runs), shared within its family: five correlation coefficients read the same trend, and counting them five
    times would let a trend outvote the variance-based indices on an effect with an optimum inside the range."""
    ids = [m for m in scores if scores[m] is not None]
    M = np.zeros((len(ids), d))
    for k, m in enumerate(ids):
        v = np.abs(np.nan_to_num(np.asarray(scores[m], float)))
        M[k] = v / v.max() if v.max() > 0 else 0.0
    fam = family or {}
    size = {}
    for m in ids:
        size[fam.get(m, m)] = size.get(fam.get(m, m), 0) + 1
    w = np.array([weights.get(m, 1.0) / size[fam.get(m, m)] for m in ids])
    score = (w[:, None] * M).sum(axis=0) / max(w.sum(), 1e-12)
    from scipy.stats import rankdata
    R = np.vstack([rankdata(-row, method="average") for row in M]) if len(ids) else np.zeros((0, d))
    mean_rank = (w[:, None] * R).sum(axis=0) / max(w.sum(), 1e-12) if len(ids) else np.zeros(d)
    top = (R == R.min(axis=1, keepdims=True)).sum(axis=0) if len(ids) else np.zeros(d)
    return {"methods": ids, "matrix": M, "weights": w, "score": score, "ranks": R, "mean_rank": mean_rank,
            "order": np.argsort(-score), "kendall_w": _kendall_w(R) if len(ids) > 1 else None, "top_count": top}


# =========================================================================
# the analysis of one response
# =========================================================================
def analyse(study, rows, response, metrics=None, inputs=None, realised=True, options=None, log=None):
    """Every method on one response of one study. -> a JSON-ready dict (see the module docstring)."""
    t_all = time.time()
    opt = dict(options or {})
    seed = int(opt.get("seed", 0) or 0)
    boot = int(opt.get("bootstrap", 600) or 600)
    sobol_n = int(opt.get("sobol_n", 2048) or 2048)
    sobol_n = 2 ** int(round(math.log2(max(256, min(sobol_n, 16384)))))
    perms = int(opt.get("permutations", 299) or 299)
    X, y, info, labels = dataset(study, rows, response, inputs, realised)
    meta = next((m for m in (metrics or []) if m.get("key") == response), None) or {"key": response,
                                                                                   "label": response, "unit": ""}
    out = {"response": {"key": response, "label": meta.get("label", response), "unit": meta.get("unit", "")},
           "n": int(len(y)), "skipped": {}, "notes": [], "timing": {}, "options": opt}
    # an input that does not vary over the finished runs cannot be assessed
    keep = [k for k in range(X.shape[1]) if len(y) and np.ptp(X[:, k]) > 0]
    dropped = [info[k]["name"] for k in range(X.shape[1]) if k not in keep]
    if dropped:
        out["notes"].append(f"Not varied over the finished runs, so left out: {', '.join(dropped)}.")
    X = X[:, keep]
    info = [info[k] for k in keep]
    n, d = X.shape
    for k, it in enumerate(info):
        it["symbol"] = f"x{k + 1}"
    out["inputs"] = info
    out["d"] = int(d)
    out["data"] = {"X": X, "y": y, "labels": labels}
    if d == 0:
        out["error"] = "This study varies no parameter over its finished runs."
        return _clean(out)
    if n < 4:
        out["error"] = f"At least four finished runs are needed; this response has {n}."
        return _clean(out)
    if float(np.std(y)) <= 0:
        out["error"] = "Every finished run gave the same value, so nothing moves it."
        return _clean(out)
    names = [it["name"] for it in info]
    lo = np.array([it["lo"] for it in info])
    hi = np.array([it["hi"] for it in info])
    lo = np.minimum(lo, X.min(axis=0))
    hi = np.maximum(hi, X.max(axis=0))
    out["design"] = study.get("design")
    if any(it["realised"] for it in info):
        out["notes"].append("Inputs marked realised take the value each case's structure really holds "
                            "(content or size reached), not the value it asked for.")
    if d > 1:
        C = np.corrcoef(X, rowvar=False)
        off = np.abs(C[np.triu_indices(d, 1)])
        out["input_corr"] = C
        if off.size and float(off.max()) > 0.3:
            i, j = [int(v) for v in np.argwhere(np.abs(np.triu(C, 1)) == off.max())[0]]
            out["notes"].append(f"{names[i]} and {names[j]} are correlated over the runs (r = {C[i, j]:.2f}): "
                                "the measures of a model sampled with independent inputs (partial dependence, "
                                "Sobol, Morris) extrapolate there; PCC/PRCC, ALE and SHAP stay closer to the runs.")
    if study.get("design") == "oat":
        out["notes"].append("A one-factor-at-a-time design never moves two inputs together, so no method can "
                            "see an interaction in it; a full factorial or a Latin hypercube can.")
    if n < 3 * d + 3:
        out["notes"].append(f"{n} runs for {d} inputs: enough for trends; the model-based measures carry wide "
                            "intervals and a Q2 that should be read before them.")

    def timed(name, fn):
        t0 = time.time()
        try:
            return fn()
        except Exception as e:                                       # noqa: BLE001
            out["skipped"][name] = f"{type(e).__name__}: {e}"
            return None
        finally:
            out["timing"][name] = round(time.time() - t0, 3)

    M = out.setdefault("methods", {})
    # ---------------------------------------------------------- statistics
    cor = timed("correlations", lambda: correlations(X, y, boot=boot, seed=seed))
    if cor:
        M.update(cor)
    reg = timed("regression", lambda: regression(X, y, boot=boot, seed=seed))
    if reg:
        M.update(reg)
    an = timed("anova", lambda: anova(X, y, names, X.min(axis=0), X.max(axis=0)))
    if an:
        M["anova"] = an
    cr = timed("corr_ratio", lambda: correlation_ratio(X, y))
    if cr is not None:
        M["corr_ratio"] = {"value": cr}
    dep = timed("dependence", lambda: dependence(X, y, perms=perms, seed=seed))
    if dep:
        M.update(dep)
    la = timed("lasso", lambda: lasso(X, y, seed=seed))
    if la:
        M["lasso"] = la
    out["main_effects"] = timed("main_effects", lambda: main_effects(X, y, names))
    # ---------------------------------------------------------- models
    mods = Models(X, y, lo, hi, seed=seed)
    timed("models", lambda: mods.fit_all(log=log))
    out["models"] = {k: {"label": Models.LABELS[k], **{kk: vv for kk, vv in (mods.cv.get(k) or {}).items()}}
                     for k in Models.KINDS if k in mods.cv}
    smooth = [k for k in ("gp", "pce", "quadratic") if k in mods.fitted]
    sur = opt.get("surrogate") if opt.get("surrogate") in mods.fitted else mods.best(smooth)
    expl = opt.get("explainer") if opt.get("explainer") in mods.fitted else mods.best(
        [k for k in ("gp", "rf", "gbm", "pce", "quadratic") if k in mods.fitted])
    out["surrogate"] = {"global": sur, "explainer": expl, "global_q2": mods.q2(sur) if sur else None,
                        "explainer_q2": mods.q2(expl) if expl else None}
    for k, v in (mods.cv or {}).items():
        q = v.get("q2")
        if q is not None and q < 0.5 and k in (sur, expl):
            out["notes"].append(f"The {Models.LABELS[k].lower()} predicts the held-out runs with Q2 = {q:.2f}: "
                                "the measures taken from it are weighted down in the consensus.")
    # GP length scales
    if "gp" in mods.fitted:
        sc = _gp_scales(mods.fitted["gp"].gp, d)
        if sc is not None:
            M["gp_ard"] = {"length_scale": sc, "value": 1.0 / sc}
    if "rf" in mods.fitted:
        rf = mods.fitted["rf"].rf
        std = np.std([t.feature_importances_ for t in rf.estimators_], axis=0)
        M["rf_mdi"] = {"value": rf.feature_importances_, "std": std}
        dr = timed("rf_perm", lambda: permutation_importance_cv(mods, "rf", repeats=3, outer=1))
        if dr is not None and len(dr):
            M["rf_perm"] = {"value": dr.mean(axis=0), "draws": dr, "ci": _ci(dr)}
    if "gbm" in mods.fitted:
        dg = timed("gbm_perm", lambda: permutation_importance_cv(mods, "gbm", repeats=3, outer=2))
        if dg is not None and len(dg):
            M["gbm_perm"] = {"value": dg.mean(axis=0), "draws": dg, "ci": _ci(dg)}
    if "pce" in mods.fitted:
        ps = timed("pce", lambda: pce_sobol(mods.fitted["pce"].pce, d))
        if ps:
            M["pce"] = ps
    # ---------------------------------------------------------- explaining the best model
    if expl:
        fx = mods.fitted[expl]
        sh = timed("shap", lambda: shapley(fx, X, X, seed=seed))
        if sh:
            M["shap"] = {"phi": sh["phi"], "base": sh["base"], "value": np.abs(sh["phi"]).mean(axis=0),
                         "exact": sh["exact"], "background": sh["background"], "model": expl,
                         "inter_mean_abs": (np.abs(sh["inter"]).mean(axis=0) if sh.get("inter") is not None else None)}
            hs = timed("h_statistic", lambda: h_statistic(sh, X, fx, X))
            if hs:
                M["h_stat"] = hs
        pdi = timed("pdp", lambda: pdp_ice(fx, X, lo, hi, seed=seed))
        if pdi:
            out["pdp"] = pdi
            # a curve that moves the response by less than a tenth of the strongest one is flat, whatever its form
            rngs = [float(np.ptp(p["pd"])) for p in pdi]
            top = max(rngs) if rngs else 0.0
            out["shapes"] = [("flat" if top > 0 and r < 0.1 * top else _shape(p["pd"])) for p, r in zip(pdi, rngs)]
        al = timed("ale", lambda: ale(fx, X, lo, hi))
        if al:
            out["ale"] = al
    # ---------------------------------------------------------- variance-based on the surrogate
    if sur:
        fs = mods.fitted[sur]
        unit = mods.unit
        span = unit.span

        def f_unit(U):
            return fs(unit.lo + np.atleast_2d(U) * span)
        sb = timed("sobol", lambda: sobol(f_unit, d, N=sobol_n, boot=200, seed=seed, second=d > 1))
        if sb:
            M["sobol"] = {**sb, "model": sur}
        ef = timed("efast", lambda: efast(f_unit, d, seed=seed))
        if ef:
            M["efast"] = {**ef, "model": sur}
        mo = timed("morris", lambda: morris(f_unit, d, seed=seed))
        if mo:
            M["morris"] = {**mo, "model": sur}
        dg = timed("dgsm", lambda: dgsm(f_unit, d, seed=seed, variance=(sb or {}).get("variance")))
        if dg:
            M["dgsm"] = {**dg, "model": sur}
        pw = timed("pawn", lambda: pawn(f_unit, d, seed=seed))
        if pw:
            M["pawn"] = {**pw, "model": sur}
    # ---------------------------------------------------------- consensus
    def val(k, sub="value"):
        e = M.get(k)
        if not e or e.get(sub) is None or e.get("skipped"):
            return None
        v = np.asarray([np.nan if x is None else x for x in np.atleast_1d(e[sub])], float)
        return None if np.all(~np.isfinite(v)) else v
    scores = {
        "pearson": val("pearson"), "spearman": val("spearman"), "kendall": val("kendall"),
        "pcc": val("pcc"), "prcc": val("prcc"), "src": val("src"), "srrc": val("srrc"),
        "anova": np.asarray(M["anova"]["share"]) if "anova" in M else None,
        "corr_ratio": val("corr_ratio"), "mi": val("mi"), "dcor": val("dcor"), "hsic": val("hsic"),
        "lasso": np.asarray(M["lasso"]["enter"]) if "lasso" in M else None,
        "gp_ard": val("gp_ard"), "rf_mdi": val("rf_mdi"), "rf_perm": None, "gbm_perm": None,
        "shap": val("shap"),
        "sobol_s1": np.clip(val("sobol", "S1"), 0, None) if val("sobol", "S1") is not None else None,
        "sobol_st": np.clip(val("sobol", "ST"), 0, None) if val("sobol", "ST") is not None else None,
        "pce_st": val("pce", "ST"), "efast_st": val("efast", "ST"), "morris": val("morris", "mu_star"),
        "dgsm": val("dgsm", "nu"), "pawn": val("pawn", "median"),
        "h_total": val("h_stat", "total") if "h_stat" in M else None,
    }
    for k in ("rf_perm", "gbm_perm"):
        v = val(k)
        scores[k] = np.clip(v, 0, None) if v is not None else None
    # the H-statistic measures interaction, not importance: shown beside the consensus, not in it
    h_total = scores.pop("h_total")
    q2w = {}
    for mid, _lab, _fam, _what, model in METHODS:
        if model is None:
            q2w[mid] = 1.0
            continue
        kind = {"surrogate": sur, "explainer": expl}.get(model, model)
        q = mods.q2(kind) if kind else None
        q2w[mid] = float(np.clip(q if q is not None else 0.0, 0.0, 1.0))
    fam = {m: f for m, _lab, f, _what, _model in METHODS}
    cons = consensus({k: v for k, v in scores.items() if v is not None}, q2w, d, family=fam)
    out["consensus"] = cons
    out["method_info"] = [{"id": m, "label": lab, "family": fam, "reads": what,
                           "model": ({"surrogate": sur, "explainer": expl}.get(model, model) if model else None),
                           "weight": q2w.get(m)} for m, lab, fam, what, model in METHODS]
    out["h_total"] = h_total
    # the direction of every input: its partial rank correlation (the others held fixed), so that an input whose
    # trend is masked by a stronger one varied beside it still reads right; Spearman where PRCC cannot be had
    pr = (M.get("prcc") or {}).get("value")
    sp = (M.get("spearman") or {}).get("value")
    sgn = pr if pr is not None and all(v is not None for v in np.atleast_1d(pr)) else sp
    out["direction"] = [None if sgn is None or sgn[i] is None else (1 if sgn[i] > 0.1 else -1 if sgn[i] < -0.1 else 0)
                        for i in range(d)]
    # ---------------------------------------------------------- interactions
    pairs = []
    s2 = M.get("sobol", {}).get("S2")
    s2ci = M.get("sobol", {}).get("S2_ci")
    p2 = M.get("pce", {}).get("S2")
    hs2 = M.get("h_stat", {}).get("pair")
    sint = M.get("shap", {}).get("inter_mean_abs")
    shap_tot = float(np.abs(M["shap"]["phi"]).mean(axis=0).sum()) if "shap" in M else None
    an_terms = {tuple(t["inputs"]): t for t in (M.get("anova", {}).get("terms") or []) if t["kind"] == "inter"}
    # a model that does not predict the runs it never saw cannot vouch for an interaction
    TRUST = 0.5
    ok_sur = (mods.q2(sur) or -1) >= TRUST if sur else False
    ok_expl = (mods.q2(expl) or -1) >= TRUST if expl else False
    ok_pce = (mods.q2("pce") or -1) >= TRUST
    out["trust"] = {"threshold": TRUST, "surrogate": ok_sur, "explainer": ok_expl, "pce": ok_pce}
    if not (ok_sur or ok_expl or ok_pce):
        out["notes"].insert(0, "No model predicts the held-out runs well (Q2 < 0.5): only the statistics of the "
                               "runs themselves can be read; the model-based panels are shown for orientation. "
                               "More runs (or a narrower range) are needed before trusting them.")
    for i in range(d):
        for j in range(i + 1, d):
            e = {"i": i, "j": j, "name": f"{names[i]} × {names[j]}", "symbol": f"x{i + 1}·x{j + 1}"}
            if s2 is not None:
                e["sobol_s2"] = float(s2[i][j])
                if s2ci is not None:
                    e["sobol_s2_ci"] = [float(s2ci[i][j][0]), float(s2ci[i][j][1])]
            if p2 is not None:
                e["pce_s2"] = float(p2[i][j])
            if hs2 is not None:
                e["h2"] = float(hs2[i][j])
            if sint is not None and shap_tot:
                e["shap_int"] = float(2 * sint[i][j] / shap_tot)
            t = an_terms.get((i, j))
            if t:
                e["anova_eta2"], e["anova_p"], e["anova_t"] = t["eta2"], t["p"], t["t"]
            flags = []
            if ok_sur and e.get("sobol_s2") is not None and e["sobol_s2"] > 0.01 and (e.get("sobol_s2_ci") or [0])[0] > 0:
                flags.append("Sobol S₂")
            if ok_pce and e.get("pce_s2") is not None and e["pce_s2"] > 0.01:
                flags.append("PCE S₂")
            if ok_expl and e.get("h2") is not None and e["h2"] > 0.05:
                flags.append("H²")
            if ok_expl and e.get("shap_int") is not None and e["shap_int"] > 0.05:
                flags.append("SHAP")
            if e.get("anova_p") is not None and e["anova_p"] < 0.05:
                flags.append("ANOVA")
            e["detected_by"] = flags
            pairs.append(e)
    measures = ["sobol_s2", "pce_s2", "h2", "shap_int", "anova_eta2"]
    for mkey in measures:
        vals = [max(0.0, p.get(mkey) or 0.0) for p in pairs]
        top = max(vals) if vals else 0.0
        for p, v in zip(pairs, vals):
            p[mkey + "_n"] = v / top if top > 0 else 0.0
    for p in pairs:
        have = [p[m + "_n"] for m in measures if p.get(m) is not None]
        p["score"] = float(np.mean(have)) if have else 0.0
    pairs.sort(key=lambda p: (-len(p["detected_by"]), -p["score"]))
    out["interactions"] = pairs
    if "sobol" in M:
        out["interaction_share"] = float(max(0.0, 1.0 - float(np.clip(M["sobol"]["S1"], 0, None).sum())))
    # 2-D partial dependence of the strongest pairs (or of the two leading inputs)
    if expl and d >= 2:
        fx = mods.fitted[expl]
        top_pairs = [(p["i"], p["j"]) for p in pairs[:2] if p["detected_by"]] or \
                    [tuple(sorted([int(cons["order"][0]), int(cons["order"][1])]))]
        out["pdp2"] = [timed(f"pdp2_{i}_{j}", lambda i=i, j=j: pdp2(fx, X, i, j, lo, hi)) for i, j in top_pairs]
        out["pdp2"] = [p for p in out["pdp2"] if p]
    out["findings"] = _findings(out, names)
    out["timing"]["total"] = round(time.time() - t_all, 2)
    return _clean(out)


def _findings(out, names):
    """The results in sentences, each with the evidence behind it."""
    M, cons = out["methods"], out["consensus"]
    d = len(names)
    order = list(cons["order"])
    R = cons["ranks"]
    nm = len(cons["methods"])
    lines = []
    sc = cons["score"]
    lead = order[0]
    n_top = int(cons["top_count"][lead]) if nm else 0
    s = f"{names[lead]} matters most: first in {n_top} of {nm} measures"
    trust = out.get("trust") or {}
    if "sobol" in M and trust.get("surrogate"):
        st = M["sobol"]["ST"][lead]
        s += f"; it takes part in {100 * st:.0f} % of the response's variance (Sobol S_T)"
    lines.append({"kind": "lead", "text": s + "."})
    for i in order[1:]:
        if sc[i] >= 0.5 * sc[lead]:
            lines.append({"kind": "important", "text": f"{names[i]} follows (consensus {sc[i]:.2f} of 1; "
                                                       f"mean rank {cons['mean_rank'][i]:.1f})."})
    weak = [names[i] for i in order if sc[i] < 0.15 * sc[lead]]
    if weak:
        lines.append({"kind": "weak", "text": "Little or no effect: " + ", ".join(weak)
                      + (" - candidates to fix in further studies." if len(weak) < d else ".")})
    # direction and shape of the inputs that matter
    for i in order[:min(d, 4)]:
        if sc[i] < 0.3 * sc[lead]:
            continue
        shp = (out.get("shapes") or [None] * d)[i]
        dr = out["direction"][i]
        if shp == "flat":
            continue
        if shp in ("peaked", "valley"):
            lines.append({"kind": "shape", "text": f"{names[i]} has an optimum inside its range "
                                                   f"({'a maximum' if shp == 'peaked' else 'a minimum'} of the "
                                                   "partial dependence): a trend measure under-rates it."})
        elif dr:
            lines.append({"kind": "shape", "text": f"Raising {names[i]} {'raises' if dr > 0 else 'lowers'} the "
                                                   f"response ({shp or 'monotonic'})."})
    an = M.get("anova") or {}
    for t in an.get("terms") or []:
        if t["kind"] == "quad" and t.get("p") is not None and t["p"] < 0.05:
            lines.append({"kind": "shape", "text": f"Curvature in {names[t['inputs'][0]]}: its quadratic term is "
                                                   f"significant (p = {t['p']:.3g})."})
    for p in (out.get("interactions") or [])[:3]:
        if len(p["detected_by"]) >= 2:
            bits = []
            if p.get("sobol_s2") is not None:
                bits.append(f"S₂ = {p['sobol_s2']:.3f}")
            if p.get("h2") is not None:
                bits.append(f"H² = {p['h2']:.2f}")
            if p.get("anova_p") is not None:
                bits.append(f"ANOVA p = {p['anova_p']:.3g}")
            lines.append({"kind": "interaction", "text": f"{p['name']} act together, found by "
                                                         f"{len(p['detected_by'])} of 5 measures ({', '.join(bits)}):"
                                                         " the effect of one depends on the level of the other."})
    if out.get("interaction_share") is not None and trust.get("surrogate"):
        sh = out["interaction_share"]
        lines.append({"kind": "share", "text": (f"Interactions carry {100 * sh:.0f} % of the variance (1 - ΣS₁); "
                                                "the response is close to additive." if sh < 0.1 else
                                                f"Interactions carry {100 * sh:.0f} % of the variance (1 - ΣS₁): "
                                                "no input can be optimised alone.")})
    w = cons.get("kendall_w")
    if w is not None:
        lines.append({"kind": "agreement", "text": f"The {nm} measures agree on the ranking with Kendall's W = "
                                                   f"{w:.2f} (1 = identical rankings)" + (
                                                       "." if w >= 0.7 else
                                                       "; where they differ, the model-free statistics and the "
                                                       "variance-based indices answer different questions - "
                                                       "see the panels.")})
    return lines


# =========================================================================
# every response at once
# =========================================================================
def overview(study, rows, metrics, inputs=None, realised=True, seed=0):
    """A light pass over every response of the study: Spearman's rho (signed) and the total Sobol index of a
    Gaussian process fitted to each, with its Q2 - which input drives which property."""
    out = {"responses": [], "inputs": None}
    for m in metrics or []:
        X, y, info, _ = dataset(study, rows, m["key"], inputs, realised)
        if len(y) < 4 or float(np.std(y)) <= 0:
            continue
        keep = [k for k in range(X.shape[1]) if np.ptp(X[:, k]) > 0]
        if not keep:
            continue
        if out["inputs"] is None:
            out["inputs"] = [info[k] for k in keep]
        if [it["path"] for it in out["inputs"]] != [info[k]["path"] for k in keep]:
            continue
        X = X[:, keep]
        d = X.shape[1]
        rho = _pearson_cols(_rank(X), _rank(y))
        e = {"key": m["key"], "label": m.get("label", m["key"]), "unit": m.get("unit", ""), "n": int(len(y)),
             "spearman": rho, "realised_metric": bool(m.get("realised"))}
        try:
            lo, hi = X.min(axis=0), X.max(axis=0)
            mods = Models(X, y, lo, hi, seed=seed)
            mods.fit_all(kinds=("gp",))
            fs = mods.fitted.get("gp")
            if fs is not None:
                unit = mods.unit
                sb = sobol(lambda U: fs(unit.lo + np.atleast_2d(U) * unit.span), d, N=1024, boot=0, seed=seed,
                           second=False)
                if sb:
                    e["sobol_st"] = np.clip(sb["ST"], 0, 1)
                    e["sobol_s1"] = np.clip(sb["S1"], 0, 1)
                e["q2"] = mods.q2("gp")
        except Exception as ex:                                      # noqa: BLE001
            e["error"] = f"{type(ex).__name__}: {ex}"
        out["responses"].append(e)
    return _clean(out)
