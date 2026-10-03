"""Optimisation on top of a DOE study.

Three steps, in the order an engineer uses them:

    sensitivity   which input actually moves the response. Pearson correlation
                  and standardised regression coefficients come from the runs
                  themselves; Sobol indices come from the surrogate and also
                  catch interactions and curvature that a linear measure misses.
    surrogate     a Gaussian process fitted to the finished runs, with its
                  accuracy measured by cross-validation, so the model is trusted
                  only as far as it has been checked.
    optimum       the best point of the surrogate, found by differential
                  evolution, with the standard deviation of the prediction; and
                  the point most worth running next (expected improvement),
                  which is where the next real run buys the most.

The surrogate replaces a run of minutes with a prediction of microseconds, so
the whole range can be searched - but it is only as good as the runs behind it,
which is why every result carries the cross-validated accuracy and the
predicted uncertainty. The optimum is a *prediction*: the honest last step is
to queue a real run there and compare, which the Optimization tab offers.

Goals are unified as "maximise u(y)":

    maximise   u = y
    minimise   u = -y
    target     u = -|y - target|
"""
from __future__ import annotations

import math

import numpy as np


# =========================================================================
# data
# =========================================================================
def _bounds(parameters):
    out = []
    for p in parameters:
        if p.get("values"):
            v = [float(x) for x in p["values"]]
            out.append((min(v), max(v)))
        else:
            lo, hi = float(p.get("min", 0.0)), float(p.get("max", 0.0))
            out.append((min(lo, hi), max(lo, hi)))
    return out


def dataset(study, rows, response):
    """Finished cases of a study as X (inputs) and y (one response)."""
    params = [p for p in (study.get("parameters") or []) if p.get("path")]
    paths = [p["path"] for p in params]
    names = [p.get("label") or p["path"] for p in params]
    units = [p.get("unit", "") for p in params]
    X, y, labels = [], [], []
    for r in rows:
        if r.get("state") != "done":
            continue
        v = (r.get("metrics") or {}).get(response)
        if v is None:
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(v):
            continue
        xs = [(r.get("values") or {}).get(p) for p in paths]
        if any(x is None for x in xs):
            continue
        X.append([float(x) for x in xs])
        y.append(v)
        labels.append(r.get("label") or r.get("job_id"))
    return (np.asarray(X, float).reshape(len(X), len(paths)), np.asarray(y, float),
            names, paths, units, _bounds(params), labels)


def _utility(y, goal, target):
    y = np.asarray(y, float)
    if goal == "min":
        return -y
    if goal == "target" and target is not None:
        return -np.abs(y - float(target))
    return y


# =========================================================================
# sensitivity from the runs themselves
# =========================================================================
def linear_effects(X, y, names):
    """Pearson correlation and standardised regression coefficients.

    The coefficients are the change of the response in standard deviations per
    standard deviation of the input, so they are directly comparable between
    inputs of different units. They describe a linear model only; the R² says
    how much of the response that model explains.
    """
    n, d = X.shape
    sx = X.std(axis=0)
    sy = y.std()
    Z = (X - X.mean(axis=0)) / np.where(sx > 0, sx, 1.0)
    w = (y - y.mean()) / (sy if sy > 0 else 1.0)
    beta = np.zeros(d)
    r2 = None
    if n > d:
        A = np.column_stack([np.ones(n), Z])
        sol, *_ = np.linalg.lstsq(A, w, rcond=None)
        beta = sol[1:]
        pred = A @ sol
        ss = float((w ** 2).sum())
        r2 = float(1.0 - ((w - pred) ** 2).sum() / ss) if ss > 0 else None
    out = []
    for i, nm in enumerate(names):
        pear = None
        if sx[i] > 0 and sy > 0:
            pear = float(np.corrcoef(X[:, i], y)[0, 1])
        lo, hi = float(X[:, i].min()), float(X[:, i].max())
        out.append({"name": nm, "src": float(beta[i]), "pearson": pear,
                    "sampled_min": lo, "sampled_max": hi})
    return out, r2


# =========================================================================
# surrogate
# =========================================================================
KERNELS = {
    "matern52": ("Matérn 5/2", 2.5),
    "matern32": ("Matérn 3/2", 1.5),
    "matern12": ("Matérn 1/2 (exponential)", 0.5),
    "rbf": ("Squared exponential (RBF)", None),
    "quadratic": ("Quadratic response surface", None),
}


def _build_kernel(name, d, opt):
    """The kernel the user asked for.

    Which kernel fits best is a property of the response, not something that can
    be decided once for every study: a smooth response suits the RBF, a rough
    one Matérn 1/2. So it is a setting, and cross-validation says which choice
    was right.
    """
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel
    lsb = tuple(opt.get("length_scale_bounds") or (1e-2, 1e2))
    ls = np.full(d, float(opt.get("length_scale") or 0.5))
    if name == "rbf":
        base = RBF(length_scale=ls, length_scale_bounds=lsb)
    else:
        base = Matern(length_scale=ls, length_scale_bounds=lsb,
                      nu=KERNELS.get(name, KERNELS["matern52"])[1] or 2.5)
    noise = opt.get("noise")
    # a fixed noise level is how a response that repeats imperfectly (a new
    # random structure for every case) stops the model from interpolating scatter
    white = (WhiteKernel(max(float(noise), 1e-12), "fixed")
             if noise not in (None, "") else WhiteKernel(1e-6, (1e-12, 1e-1)))
    return ConstantKernel(1.0, (1e-3, 1e3)) * base + white


def fit_surrogate(X, y, bounds, seed=0, options=None):
    """Gaussian process on the unit cube, with a quadratic fallback.

    `options` tunes the model: kernel, length-scale bounds, fixed noise and the
    number of restarts of the likelihood optimiser.
    """
    opt = dict(options or {})
    kind = opt.get("kernel") or "matern52"
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    span = np.where(hi > lo, hi - lo, 1.0)

    def to_unit(Xq):
        return (np.atleast_2d(np.asarray(Xq, float)) - lo) / span

    U = to_unit(X)
    d = U.shape[1]
    try:
        if kind == "quadratic":
            raise RuntimeError("the quadratic surface was asked for")
        import warnings

        from sklearn.exceptions import ConvergenceWarning
        from sklearn.gaussian_process import GaussianProcessRegressor
        kern = _build_kernel(kind, d, opt)
        gp = GaussianProcessRegressor(kernel=kern, normalize_y=True,
                                      n_restarts_optimizer=int(opt.get("restarts", 4)),
                                      random_state=int(seed))
        # the optimiser not reaching its tolerance on a handful of points is
        # normal here and says nothing about the fit, which is reported by
        # cross-validation instead
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp.fit(U, y)

        def predict(Xq, std=False):
            m, s = gp.predict(to_unit(Xq), return_std=True)
            return (m, s) if std else m

        # the fitted length scales rank the inputs: a short scale means the
        # response changes quickly along that input
        scales = None
        try:
            for name, val in gp.kernel_.get_params().items():
                if name.endswith("length_scale") and np.ndim(val) == 1 and len(val) == d:
                    scales = [float(v) for v in val]
        except Exception:                                            # noqa: BLE001
            scales = None
        return {"predict": predict,
                "kind": f"Gaussian process ({KERNELS.get(kind, KERNELS['matern52'])[0]}, "
                        f"one length scale per input)",
                "kernel": kind, "length_scales": scales}
    except Exception:                                                # noqa: BLE001
        # quadratic response surface: enough for a handful of runs, and it
        # never fails to fit
        def design(Uq):
            Uq = np.atleast_2d(Uq)
            cols = [np.ones(len(Uq))] + [Uq[:, i] for i in range(d)]
            for i in range(d):
                for j in range(i, d):
                    cols.append(Uq[:, i] * Uq[:, j])
            return np.column_stack(cols)

        A = design(U)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = float(np.sqrt(np.mean((A @ coef - y) ** 2)))

        def predict(Xq, std=False):
            m = design(to_unit(Xq)) @ coef
            return (m, np.full(len(m), resid)) if std else m

        return {"predict": predict, "kind": "quadratic response surface", "length_scales": None}


def cross_validate(X, y, bounds, seed=0, options=None):
    """Leave-one-out for a small set, 5-fold above it.

    Cross-validation is the only honest statement about a surrogate: the fit
    quality on the training points themselves would be near perfect for a
    Gaussian process, which interpolates.
    """
    n = len(y)
    if n < 5:
        return None
    folds = n if n <= 24 else 5
    idx = np.arange(n)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(idx)
    pred = np.zeros(n)
    for part in np.array_split(idx, folds):
        train = np.setdiff1d(idx, part)
        if len(train) < 3:
            return None
        # the same settings as the real fit, or the score would describe a
        # different model from the one the optimum is taken from
        m = fit_surrogate(X[train], y[train], bounds, seed, options)
        pred[part] = m["predict"](X[part])
    ss = float(((y - y.mean()) ** 2).sum())
    r2 = float(1.0 - ((y - pred) ** 2).sum() / ss) if ss > 0 else None
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    return {"r2": r2, "rmse": rmse, "predicted": [float(v) for v in pred],
            "actual": [float(v) for v in y], "folds": int(folds)}


# =========================================================================
# Sobol indices on the surrogate
# =========================================================================
def sobol(model, bounds, n=512, seed=0):
    """First-order and total Sobol indices (Saltelli estimators).

    S1 is the share of the variance explained by that input alone, ST includes
    everything it takes part in, so ST - S1 is the share that only appears
    through interactions with other inputs.
    """
    d = len(bounds)
    if d == 0:
        return None
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    rng = np.random.default_rng(int(seed))
    A = lo + (hi - lo) * rng.random((n, d))
    B = lo + (hi - lo) * rng.random((n, d))
    fA = np.asarray(model["predict"](A), float)
    fB = np.asarray(model["predict"](B), float)
    var = float(np.var(np.concatenate([fA, fB])))
    if not np.isfinite(var) or var <= 0:
        return None
    # centre the outputs: the first-order estimator multiplies by f(B), so a
    # response with a large mean (a density of 2.2, say) and a small variance
    # drowns the estimate in noise and reports zero for an input that in fact
    # explains nearly everything
    mean = float(np.mean(np.concatenate([fA, fB])))
    fA = fA - mean
    fB = fB - mean
    s1, st = [], []
    for i in range(d):
        AB = A.copy()
        AB[:, i] = B[:, i]
        fAB = np.asarray(model["predict"](AB), float) - mean
        s1.append(float(np.clip(np.mean(fB * (fAB - fA)) / var, 0.0, 1.0)))
        st.append(float(np.clip(np.mean((fA - fAB) ** 2) / (2.0 * var), 0.0, 1.0)))
    return {"s1": s1, "total": st, "samples": int(n)}


# =========================================================================
# optimum and the next run to make
# =========================================================================
def _de(fun, bounds, seed, maxiter=200):
    from scipy.optimize import differential_evolution
    spread = [(a, b if b > a else a + 1e-9) for a, b in bounds]
    return differential_evolution(fun, spread, seed=int(seed), maxiter=maxiter,
                                  tol=1e-7, polish=True)


def optimum(model, bounds, goal, target=None, seed=0):
    """Best point of the surrogate, with the uncertainty of the prediction."""
    def neg_utility(x):
        m = float(np.asarray(model["predict"](np.atleast_2d(x)), float)[0])
        return -float(_utility(np.array([m]), goal, target)[0])

    res = _de(neg_utility, bounds, seed)
    x = np.atleast_2d(res.x)
    mean, std = model["predict"](x, std=True)
    return {"x": [float(v) for v in res.x], "predicted": float(mean[0]),
            "std": float(std[0]), "converged": bool(res.success)}


def suggest_next(model, bounds, y, goal, target=None, seed=0):
    """Expected improvement: where the next real run is worth most.

    A point is worth running when the surrogate predicts a good value *or* when
    it is uncertain there; expected improvement balances the two, so the
    suggestion is not simply the predicted optimum again.
    """
    u_obs = _utility(y, goal, target)
    best = float(np.max(u_obs))

    def neg_ei(x):
        m, s = model["predict"](np.atleast_2d(x), std=True)
        mu = float(_utility(np.asarray(m, float), goal, target)[0])
        sd = float(max(np.asarray(s, float)[0], 1e-12))
        imp = mu - best
        z = imp / sd
        cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        pdf = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        return -(imp * cdf + sd * pdf)

    res = _de(neg_ei, bounds, seed, maxiter=120)
    x = np.atleast_2d(res.x)
    mean, std = model["predict"](x, std=True)
    return {"x": [float(v) for v in res.x], "predicted": float(mean[0]),
            "std": float(std[0]), "expected_improvement": float(-res.fun)}


# =========================================================================
# the whole study
# =========================================================================
# =========================================================================
# NSGA-II: more than one objective at a time
# =========================================================================
def _as_min(Y, goals, targets=None):
    """Every objective rewritten as something to minimise."""
    Y = np.atleast_2d(np.asarray(Y, float))
    out = np.empty_like(Y)
    for k, g in enumerate(goals):
        if g == "min":
            out[:, k] = Y[:, k]
        elif g == "target":
            t = (targets or {}).get(k, 0.0)
            out[:, k] = np.abs(Y[:, k] - float(t))
        else:
            out[:, k] = -Y[:, k]
    return out


def _fronts(F):
    """Fast non-dominated sort of rows of F (all to be minimised)."""
    n = len(F)
    dominates = [None] * n
    count = np.zeros(n, int)
    for p in range(n):
        le = np.all(F[p] <= F, axis=1)
        lt = np.any(F[p] < F, axis=1)
        dominates[p] = np.flatnonzero(le & lt)
        ge = np.all(F <= F[p], axis=1)
        gt = np.any(F < F[p], axis=1)
        count[p] = int(np.count_nonzero(ge & gt))
    fronts, cur = [], list(np.flatnonzero(count == 0))
    while cur:
        fronts.append(cur)
        nxt = []
        for p in cur:
            for q in dominates[p]:
                count[q] -= 1
                if count[q] == 0:
                    nxt.append(int(q))
        cur = nxt
    return fronts


def _crowding(F):
    """Crowding distance: keeps the front evenly spread instead of clumped."""
    n, m = F.shape
    d = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for k in range(m):
        o = np.argsort(F[:, k])
        f = F[o, k]
        d[o[0]] = d[o[-1]] = np.inf
        span = f[-1] - f[0]
        if span > 0:
            d[o[1:-1]] += (f[2:] - f[:-2]) / span
    return d


def nsga2(evaluate, bounds, goals, targets=None, pop=64, generations=60, seed=0):
    """Multi-objective optimisation of the surrogates (NSGA-II).

    Several properties are usually wanted at once and they usually fight - more
    filler raises the conductivity and the stiffness but also the weight and the
    stress. There is then no single best design, only a set of designs where
    nothing can be improved without giving something else up. NSGA-II returns
    that set (the Pareto front) so the trade-off can be seen and chosen from.

    Written directly on numpy rather than pulled in from a library, because the
    offline bundle installs only from the wheels it ships with.

    `evaluate(X) -> (n, m)` raw objective values; `goals` is one of
    "min"/"max"/"target" per objective.
    """
    rng = np.random.default_rng(int(seed))
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    span = np.where(hi > lo, hi - lo, 1e-12)
    d = len(bounds)
    pop = max(8, int(pop))

    # a Latin hypercube start covers the box more evenly than uniform noise
    grid = (rng.permuted(np.tile(np.arange(pop), (d, 1)), axis=1).T + rng.random((pop, d))) / pop
    X = lo + grid * span

    def clipped(Z):
        return np.clip(Z, lo, hi)

    def step(Xp, Fp):
        # binary tournament on (front, crowding)
        order = np.empty(len(Xp), int)
        rank = np.empty(len(Xp), int)
        crowd = np.empty(len(Xp))
        for r, fr in enumerate(_fronts(Fp)):
            rank[fr] = r
            crowd[fr] = _crowding(Fp[fr])
        a, b = rng.integers(0, len(Xp), (2, len(Xp)))
        better = np.where((rank[a] < rank[b]) | ((rank[a] == rank[b]) & (crowd[a] > crowd[b])), a, b)
        order[:] = better
        P = Xp[order]
        # simulated binary crossover
        Q = P.copy()
        for i in range(0, len(P) - 1, 2):
            if rng.random() < 0.9:
                u = rng.random(d)
                beta = np.where(u <= 0.5, (2 * u) ** (1 / 16), (1 / (2 * (1 - u))) ** (1 / 16))
                c1 = 0.5 * ((1 + beta) * P[i] + (1 - beta) * P[i + 1])
                c2 = 0.5 * ((1 - beta) * P[i] + (1 + beta) * P[i + 1])
                Q[i], Q[i + 1] = c1, c2
        # polynomial mutation
        mut = rng.random((len(Q), d)) < (1.0 / d)
        u = rng.random((len(Q), d))
        delta = np.where(u < 0.5, (2 * u) ** (1 / 21) - 1, 1 - (2 * (1 - u)) ** (1 / 21))
        Q = np.where(mut, Q + delta * span, Q)
        return clipped(Q)

    Y = np.atleast_2d(np.asarray(evaluate(X), float))
    F = _as_min(Y, goals, targets)
    for _ in range(max(1, int(generations))):
        C = step(X, F)
        Yc = np.atleast_2d(np.asarray(evaluate(C), float))
        Xa = np.vstack([X, C])
        Ya = np.vstack([Y, Yc])
        Fa = _as_min(Ya, goals, targets)
        keep = []
        for fr in _fronts(Fa):
            if len(keep) + len(fr) <= pop:
                keep.extend(fr)
                continue
            room = pop - len(keep)
            if room > 0:
                cd = _crowding(Fa[fr])
                keep.extend([fr[i] for i in np.argsort(-cd)[:room]])
            break
        keep = np.array(keep, int)
        X, Y, F = Xa[keep], Ya[keep], Fa[keep]

    front = _fronts(F)[0]
    o = np.argsort(F[front][:, 0])
    front = np.array(front, int)[o]
    # Objectives that do not conflict have a front of exactly one design, and
    # the whole population converges onto it. Reporting the population size then
    # claims dozens of trade-offs where there is really one choice, so identical
    # designs are counted once.
    if len(front) > 1:
        seen, keep = set(), []
        for i in front:
            sig = tuple(np.round(Y[i], 9))
            if sig in seen:
                continue
            seen.add(sig)
            keep.append(int(i))
        front = np.array(keep, int)
    # the knee: the front point closest to the ideal corner once each objective
    # is scaled to 0..1, i.e. the most balanced compromise on offer
    Ff = F[front]
    rng_f = np.ptp(Ff, axis=0)
    norm = (Ff - Ff.min(axis=0)) / np.where(rng_f > 0, rng_f, 1.0)
    knee = int(front[int(np.argmin(np.linalg.norm(norm, axis=1)))])
    return {"x": X[front].tolist(), "y": Y[front].tolist(),
            "knee_x": X[knee].tolist(), "knee_y": Y[knee].tolist(),
            "n_front": int(len(front)), "generations": int(generations), "pop": int(pop)}


def pareto(study, rows, responses, goals, targets=None, seed=0, options=None,
           pop=64, generations=60):
    """Trade-off front between several responses of one study.

    One surrogate is fitted per response from the same finished runs, and
    NSGA-II is then run over all of them at once. Each surrogate carries its own
    cross-validated score, because a front is only worth as much as the weakest
    model underneath it.
    """
    out = {"responses": list(responses), "goals": list(goals), "targets": targets or {}}
    models, bounds, names, units, scores = [], None, None, [], []
    for r in responses:
        X, y, nm, paths, un, bd, labels = dataset(study, rows, r)
        if X.shape[1] == 0:
            out["error"] = "This study varies no parameter."
            return out
        if len(y) < 3:
            out["error"] = f"Three finished runs are needed; this study has {len(y)}."
            return out
        if float(np.std(y)) <= 0:
            out["error"] = f"Every finished run gave the same value for '{r}'."
            return out
        bounds, names = bd, nm
        m = fit_surrogate(X, y, bd, seed, options)
        models.append(m)
        units.append(un)
        cv = cross_validate(X, y, bd, seed, options)
        scores.append({"response": r, "kind": m["kind"],
                       "r2": (cv or {}).get("r2"), "rmse": (cv or {}).get("rmse")})
        out["n_points"] = int(len(y))

    tmap = {i: (targets or {}).get(r) for i, r in enumerate(responses)
            if (targets or {}).get(r) is not None}

    def evaluate(Xq):
        return np.column_stack([np.asarray(m["predict"](Xq), float).ravel() for m in models])

    res = nsga2(evaluate, bounds, goals, tmap, pop=pop, generations=generations, seed=seed)
    out.update(res)
    out["parameters"] = names
    out["units"] = units
    out["surrogates"] = scores
    out["bounds"] = [[float(a), float(b)] for a, b in bounds]
    return out


def analyse(study, rows, response, goal="max", target=None, seed=0, sobol_samples=512,
            options=None):
    """Sensitivity, surrogate and optimum for one response of one study."""
    X, y, names, paths, units, bounds, labels = dataset(study, rows, response)
    out = {"response": response, "goal": goal, "target": target,
           "n_points": int(len(y)), "parameters": names, "paths": paths, "units": units,
           "bounds": [[float(a), float(b)] for a, b in bounds]}
    if X.shape[1] == 0:
        out["error"] = "This study varies no parameter."
        return out
    if len(y) < 3:
        out["error"] = f"Three finished runs are needed; this study has {len(y)}."
        return out
    if float(np.std(y)) <= 0:
        out["error"] = "Every finished run gave the same value, so nothing can be fitted."
        return out

    effects, r2lin = linear_effects(X, y, names)
    out["effects"] = effects
    out["linear_r2"] = r2lin

    u = _utility(y, goal, target)
    b = int(np.argmax(u))
    out["observed_best"] = {"label": labels[b] if b < len(labels) else "", "value": float(y[b]),
                            "x": [float(v) for v in X[b]]}
    out["samples"] = {"x": X.tolist(), "y": [float(v) for v in y], "labels": labels}

    model = fit_surrogate(X, y, bounds, seed, options)
    out["surrogate"] = {"kind": model["kind"], "length_scales": model["length_scales"],
                        "kernel": model.get("kernel"),
                        # echoed back so the tuning step can show which settings
                        # produced the score below
                        "options": dict(options or {})}
    # the same options as the fit, or the score would describe a different model
    # from the one the optimum is taken from
    cv = cross_validate(X, y, bounds, seed, options)
    if cv:
        out["surrogate"]["cv"] = cv
    try:
        sb = sobol(model, bounds, n=int(sobol_samples), seed=seed)
        if sb:
            for i, e in enumerate(out["effects"]):
                e["sobol_first"] = sb["s1"][i]
                e["sobol_total"] = sb["total"][i]
            out["sobol_samples"] = sb["samples"]
    except Exception as e:                                           # noqa: BLE001
        out["sobol_error"] = f"{type(e).__name__}: {e}"
    try:
        out["optimum"] = optimum(model, bounds, goal, target, seed)
        out["optimum"]["values"] = {p: v for p, v in zip(paths, out["optimum"]["x"])}
    except Exception as e:                                           # noqa: BLE001
        out["optimum_error"] = f"{type(e).__name__}: {e}"
    try:
        out["next_run"] = suggest_next(model, bounds, y, goal, target, seed)
        out["next_run"]["values"] = {p: v for p, v in zip(paths, out["next_run"]["x"])}
    except Exception as e:                                           # noqa: BLE001
        out["next_run_error"] = f"{type(e).__name__}: {e}"
    return out
