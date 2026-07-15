import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
import numpy as np
import ROOT
import uproot
import yaml

PIMASS = 0.13957018


def load_sigma_eta_phi(path, pt_bins):
    """Return [(sigma_eta, sigma_phi), ...], one per pT bin, from get_sigma_eta_phi.py output."""
    with uproot.open(path) as f:
        eta_hist, phi_hist = f["hist_eta"], f["hist_phi"]
        eta_values, eta_edges = eta_hist.to_numpy()
        phi_values, phi_edges = phi_hist.to_numpy()

    # the sigmas are indexed by pT bin, so the two binnings have to be the same
    for name, edges in (("hist_eta", eta_edges), ("hist_phi", phi_edges)):
        if len(edges) != len(pt_bins) or not np.allclose(edges, pt_bins):
            raise RuntimeError(
                f"pT binning of {name} in {path} does not match the config pt_bins:\n"
                f"  file:   {list(edges)}\n  config: {list(pt_bins)}"
            )

    return list(zip(eta_values, phi_values))


def read_histograms(path, pt_bins):
    """Return list of (low, high, counts) for each pT bin histogram in mc.root."""
    histos = []
    with uproot.open(path) as f:
        for lo, hi in zip(pt_bins[:-1], pt_bins[1:]):
            key = f"mass_pt_{lo}_{hi}"
            counts, edges = f[key].to_numpy()
            histos.append((lo, hi, counts.astype(np.float64), edges))
    return histos


# The DSCB model lives in dscb.cxx and is compiled to run faster
_DSCB_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dscb.cxx")
if ROOT.gSystem.CompileMacro(_DSCB_SRC, "kO") != 1:
    raise RuntimeError(f"ACLiC failed to compile {_DSCB_SRC}")


def configure_func(func, is_mc, lo, hi, init, norm0, mu=None, width=None, fix_core=False):
    """
    Initialise DSCB parameters
    """
    if fix_core:
        func.FixParameter(0, norm0)
        func.FixParameter(1, mu)
        func.FixParameter(2, width)
        func.FixParameter(3, init["a1"])
        func.FixParameter(4, init["p1"])
        func.FixParameter(5, init["a2"])
        func.FixParameter(6, init["p2"])
        if not is_mc:
            func.FixParameter(7, init["norm_bkg"])
            func.FixParameter(8, init["slope"])
        return

    func.SetParameter(0, norm0);         func.SetParLimits(0, 0.0, 1.0e9)
    func.SetParameter(1, init["mu"]);    func.SetParLimits(1, lo, hi)
    func.SetParameter(2, init["width"]); func.SetParLimits(2, 1.0e-4, 0.05)
    func.SetParameter(3, init["a1"]);    func.SetParLimits(3, 1.0, 30.0)
    func.SetParameter(4, init["p1"]);    func.SetParLimits(4, 1.0, 10.0)
    func.SetParameter(5, init["a2"]);    func.SetParLimits(5, 1.0, 30.0)
    func.SetParameter(6, init["p2"]);    func.SetParLimits(6, 1.0, 10.0)
    if not is_mc:
        func.SetParameter(7, init["norm_bkg"]); func.SetParLimits(7, 0.0, 1.0e9)
        func.SetParameter(8, init["slope"]);    func.SetParLimits(8, -5.0, 0.0)


def prefit_dscb(hist, func, is_mc, lo, hi, max_retries=10):
    init = {"mu": hist.GetMean(), "width": max(hist.GetRMS(), 1.0e-3),
            "a1": 1.5, "p1": 2.0, "a2": 1.5, "p2": 2.0}
    if not is_mc:
        init["norm_bkg"] = max(hist.GetBinContent(1), 1.0)
        init["slope"] = 0.0
    norm0 = max(hist.GetMaximum(), 1.0)
    configure_func(func, is_mc, lo, hi, init, norm0, fix_core=False)
    res = hist.Fit(func, "LSRQN0")
    tries = 0
    while tries < max_retries and (not res.Get() or res.CovMatrixStatus() < 3):
        res = hist.Fit(func, "LSRQN0")
        tries += 1

    out = {"mu": func.GetParameter(1), "width": func.GetParameter(2),
           "norm": func.GetParameter(0),
           "a1": func.GetParameter(3), "p1": func.GetParameter(4),
           "a2": func.GetParameter(5), "p2": func.GetParameter(6)}
    if not is_mc:
        out["norm_bkg"] = func.GetParameter(7)
        out["slope"] = func.GetParameter(8)
    return out


def load_decays(path, tree_name, pt_bins, max_per_bin, seed,
                eta_cut, alpha_cut, qt_cut, chunk=20_000_000):
    """
        For every K0s we store: pos_pt, neg_pt, eta_pos, eta_neg, phi_pos, phi_neg.

        Decays are kept only if the K0s passes the same selection used for the
        data/MC signal (see signal_producer.py): |eta| < eta_cut,
        |alpha| < alpha_cut and qt > qt_cut (Armenteros-Podolanski variables).
    """
    nbins = len(pt_bins) - 1
    edges = np.asarray(pt_bins, dtype=np.float64)
    cols = {k: [] for k in ("pos_pt", "pos_eta", "pos_phi", "neg_pt", "neg_eta", "neg_phi")}
    store = [{k: [] for k in cols} for _ in range(nbins)]
    filled = np.zeros(nbins, dtype=np.int64)
    max_entries = np.full(nbins, max_per_bin, dtype=np.int64)

    branches = ["pos_pt", "pos_eta", "pos_phi", "neg_pt", "neg_eta", "neg_phi"]
    scanned = 0
    for arrays in uproot.iterate(
        f"{path}:{tree_name}", branches, library="np", step_size=chunk
    ):
        pos_pt = arrays["pos_pt"].astype(np.float64)
        neg_pt = arrays["neg_pt"].astype(np.float64)
        pos_eta = arrays["pos_eta"].astype(np.float64)
        neg_eta = arrays["neg_eta"].astype(np.float64)
        pos_phi = arrays["pos_phi"].astype(np.float64)
        neg_phi = arrays["neg_phi"].astype(np.float64)
        cos_dphi = np.cos(pos_phi - neg_phi)
        scanned += pos_pt.size

        # true K0s pT = |pt_pos + pt_neg| (vector sum of the two daughters)
        k0s_pt = np.sqrt(pos_pt**2 + neg_pt**2 + 2.0 * pos_pt * neg_pt * cos_dphi)
        ibin = np.digitize(pos_pt, edges) - 1  # -1 / nbins => out of range

        sinh_p = np.sinh(pos_eta)
        sinh_n = np.sinh(neg_eta)
        cosh_p = np.sqrt(1.0 + sinh_p**2)
        cosh_n = np.sqrt(1.0 + sinh_n**2)

        # c12 is the angular part of the momentum dot product:
        #   p1.p2 = pt1 pt2 (cos phi1 cos phi2 + sin phi1 sin phi2) + pz1 pz2
        #         = pt1 pt2 [cos(phi1 - phi2) + sinh eta1 sinh eta2]
        # so c12 = cos(dphi) + sinh(eta1) sinh(eta2).
        c12 = cos_dphi + sinh_p * sinh_n

        # --- K0s eta + Armenteros-Podolanski selection ---
        p_p2 = (pos_pt * cosh_p) ** 2
        p_n2 = (neg_pt * cosh_n) ** 2
        dot_pn = pos_pt * neg_pt * c12
        ptot2 = p_p2 + p_n2 + 2.0 * dot_pn

        pz = pos_pt * sinh_p + neg_pt * sinh_n
        eta_k0s = np.arcsinh(pz / k0s_pt)

        alpha = (p_p2 - p_n2) / ptot2
        qt = np.sqrt(np.maximum(p_p2 * ptot2 - (p_p2 + dot_pn) ** 2, 0.0)) / np.sqrt(ptot2)

        keep = (np.abs(eta_k0s) < eta_cut) & (np.abs(alpha) < alpha_cut) & (qt > qt_cut)
        ibin[~keep] = -1  # decays failing the selection get no valid bin

        chunk_cols = {"pos_pt": pos_pt, "neg_pt": neg_pt, "pos_eta": pos_eta, "neg_eta": neg_eta, "pos_phi": pos_phi, "neg_phi": neg_phi}

        for b in range(nbins):
            if filled[b] >= max_entries[b]:
                continue
            # If number of events in this bin is larger than the remaining capacity,
            # we take only a subset of them.
            sel = ibin == b
            n = int(sel.sum())
            if n == 0:
                continue
            to_take = min(n, max_entries[b] - filled[b])
            idx = np.flatnonzero(sel)[:to_take]
            for k in cols:
                store[b][k].append(chunk_cols[k][idx])
            filled[b] += to_take

        if np.all(filled >= max_entries):
            break

    rng = np.random.default_rng(seed)
    decays = []
    for b in range(nbins):
        if filled[b] == 0:
            decays.append(None)
            continue
        data = {k: np.concatenate(store[b][k]) for k in cols}
        n = data["pos_pt"].size
        data["z_pt_p"] = rng.standard_normal(n)
        data["z_pt_n"] = rng.standard_normal(n)
        data["z_eta_p"] = rng.standard_normal(n)
        data["z_eta_n"] = rng.standard_normal(n)
        data["z_phi_p"] = rng.standard_normal(n)
        data["z_phi_n"] = rng.standard_normal(n)
        decays.append(data)
    return decays

def smeared_mass(data, delta, sigma, sigma_eta, sigma_phi):
    pt_p = data["pos_pt"] + delta + sigma * data["z_pt_p"]
    pt_n = data["neg_pt"] + delta + sigma * data["z_pt_n"]
    eta_p = data["pos_eta"] + sigma_eta * data["z_eta_p"]
    eta_n = data["neg_eta"] + sigma_eta * data["z_eta_n"]
    phi_p = data["pos_phi"] + sigma_phi * data["z_phi_p"]
    phi_n = data["neg_phi"] + sigma_phi * data["z_phi_n"]
    cos_dphi = np.cos(phi_p - phi_n)
    sinh_p = np.sinh(eta_p)
    sinh_n = np.sinh(eta_n)
    cosh_p = np.sqrt(1.0 + sinh_p**2)
    cosh_n = np.sqrt(1.0 + sinh_n**2)
    c12 = cos_dphi + sinh_p * sinh_n
    ok = (pt_p > 0.0) & (pt_n > 0.0)
    pt_p, pt_n = pt_p[ok], pt_n[ok]
    e1 = np.sqrt(PIMASS**2 + (pt_p * cosh_p[ok]) ** 2)
    e2 = np.sqrt(PIMASS**2 + (pt_n * cosh_n[ok]) ** 2)
    m2 = 2.0 * PIMASS**2 + 2.0 * (e1 * e2 - pt_p * pt_n * c12[ok])
    return np.sqrt(np.clip(m2, 0.0, None))


def make_chi2(data, hist, func, mass_min, mass_max, is_mc, tail_init, lo, hi):
    norm0 = max(tail_init["norm"], 1.0)
    last = {}

    def chi2(par):
        delta, sigma, sigma_eta, sigma_phi = par[0], par[1], par[2], par[3]
        mass = smeared_mass(data, delta, sigma, sigma_eta, sigma_phi)
        sel = (mass >= mass_min) & (mass <= mass_max)
        ms = mass[sel]
        if ms.size < 100:
            return 1.0e12
        mu = float(ms.mean())
        width = float(ms.std())
        if not (np.isfinite(mu) and np.isfinite(width)) or width <= 0.0:
            return 1.0e12
        configure_func(func, is_mc, lo, hi, tail_init, norm0, mu, width, fix_core=True)
        hist.Fit(func, "LRQN0")
        c2 = func.GetChisquare()
        last.update(mu=mu, width=width, chi2=c2, ndf=func.GetNDF())
        return c2

    return chi2, last


def fit_bin(data, counts, edges, mass_min, mass_max, is_mc, tag, fixed_sigma, max_retries=5):
    lo, hi = float(edges[0]), float(edges[-1])
    hist = to_th1(f"fit_h_{tag}", counts, edges, err_floor=1.0)  # modified chi2 errors
    hist.SetDirectory(0)

    func_def = ROOT.double_sided_cb if is_mc else ROOT.double_sided_cb_plus_bkg
    npar = 7 if is_mc else 9
    func = ROOT.TF1(f"fit_f_{tag}", func_def, lo, hi, npar)

    tail_init = prefit_dscb(hist, func, is_mc, lo, hi)

    chi2, last = make_chi2(data, hist, func, mass_min, mass_max, is_mc, tail_init, lo, hi)
    functor = ROOT.Math.Functor(chi2, 4)

    # Starting points for (deltapt, sigmapt, sigma_eta, sigma_phi; the first is the nominal seed,
    # the rest are fallbacks, tried only if Migrad fails to find a valid minimum
    seeds = [(0.0, 0.01, 0.0, 0.0), (0.0, 0.005, 0.0, 0.0), (0.0, 0.02, 0.0, 0.0), (0.0, 0.05, 0.0, 0.0),
             (-0.01, 0.01, 0.0, 0.0), (0.01, 0.01, 0.0, 0.0), (-0.03, 0.03, 0.0, 0.0), (0.03, 0.03, 0.0, 0.0)]

    best = None
    for s_delta_pt, s_sigma_pt, s_sigma_eta, s_sigma_phi in seeds[: 1 + max_retries]:
        m = ROOT.Math.Factory.CreateMinimizer("Minuit2", "Migrad")
        m.SetFunction(functor)
        m.SetMaxFunctionCalls(100000)
        m.SetTolerance(0.001)
        m.SetStrategy(1)
        m.SetPrintLevel(0)
        m.SetLimitedVariable(0, "deltapt", s_delta_pt, 1e-4, -0.2, 0.2)
        m.SetLimitedVariable(1, "sigmapt", s_sigma_pt, 1e-4, 1e-5, 0.2)
        if fixed_sigma is not None:
            m.SetFixedVariable(2, "sigmaeta", fixed_sigma[0])
            m.SetFixedVariable(3, "sigmaphi", fixed_sigma[1])
        else:
            m.SetLimitedVariable(2, "sigmaeta", s_sigma_eta, 1e-4, -0.2, 0.2)
            m.SetLimitedVariable(3, "sigmaphi", s_sigma_phi, 1e-4, -0.2, 0.2)

        ok = bool(m.Minimize())
        cand = {"ok": ok, "status": int(m.Status()), "chi2": m.MinValue(),
                "delta_pt": m.X()[0], "delta_pt_err": m.Errors()[0],
                "sigma_pt": m.X()[1], "sigma_pt_err": m.Errors()[1],
                "sigma_eta": m.X()[2], "sigma_eta_err": m.Errors()[2],
                "sigma_phi": m.X()[3], "sigma_phi_err": m.Errors()[3],
                "seed": (s_delta_pt, s_sigma_pt, s_sigma_eta, s_sigma_phi)}
        # prefer a valid minimum (ok), then the lowest chi2
        if best is None or (cand["ok"], cand["status"] == 0, -cand["chi2"]) > (best["ok"], best["status"] == 0, -best["chi2"]):
            best = cand
            npars = 2 if fixed_sigma is not None else 4
            corr = [[m.Correlation(i, j) for j in range(npars)] for i in range(npars)]
        if ok and best["status"] == 0:
            break


    chi2([best["delta_pt"], best["sigma_pt"], best["sigma_eta"], best["sigma_phi"]])

    n_free = 2 if fixed_sigma is not None else 4  # free smearing params (eta/phi fixed => 2)
    ndf = max(last["ndf"] - n_free, 1)
    return {
        "ok": best["ok"],
        "status": best["status"],  # Minuit2 status: 0 = converged OK
        "seed": best["seed"],
        "delta_pt": best["delta_pt"], "delta_pt_err": best["delta_pt_err"],
        "sigma_pt": best["sigma_pt"], "sigma_pt_err": best["sigma_pt_err"],
        "sigma_eta": best["sigma_eta"], "sigma_eta_err": best["sigma_eta_err"],
        "sigma_phi": best["sigma_phi"], "sigma_phi_err": best["sigma_phi_err"],
        "chi2": best["chi2"], "ndf": ndf,
        "mu": last["mu"], "width": last["width"],
        "func": func,
        "corr": corr
    }

def to_th1(name, counts, edges, err_floor=0.0):
    h = ROOT.TH1D(name, name, len(counts), edges[0], edges[-1])
    for i, c in enumerate(counts):
        h.SetBinContent(i + 1, c)
        h.SetBinError(i + 1, np.sqrt(max(c, err_floor)))
    return h

def get_signal_func(func):
    signal_func = ROOT.TF1(func.GetName() + "_signal", ROOT.double_sided_cb, func.GetXmin(), func.GetXmax(), 7)
    for i in range(7):
        signal_func.SetParameter(i, func.GetParameter(i))
    return signal_func

def get_background_func(func):
    bkg_func = ROOT.TF1(func.GetName() + "_background", "[0]*exp([1]*x)", func.GetXmin(), func.GetXmax(), 2)
    for i in range(2):
        bkg_func.SetParameter(i, func.GetParameter(i+7))
    return bkg_func

def _profile_err(vals, chi2_profile, chi2_min):
    """Estimate 1-sigma uncertainty from a chi2 profile via the delta-chi2=1 crossing."""
    best_idx = int(np.argmin(chi2_profile))
    target = chi2_min + 1.0
    err_lo = err_hi = 0.0
    for k in range(best_idx - 1, -1, -1):
        if chi2_profile[k] >= target:
            frac = (target - chi2_profile[k + 1]) / (chi2_profile[k] - chi2_profile[k + 1])
            err_lo = vals[best_idx] - (vals[k + 1] + frac * (vals[k] - vals[k + 1]))
            break
    for k in range(best_idx + 1, len(vals)):
        if chi2_profile[k] >= target:
            frac = (target - chi2_profile[k - 1]) / (chi2_profile[k] - chi2_profile[k - 1])
            err_hi = (vals[k - 1] + frac * (vals[k] - vals[k - 1])) - vals[best_idx]
            break
    step = float(np.diff(vals).mean())
    return 0.5 * (err_lo + err_hi) if (err_lo > 0 or err_hi > 0) else step


def grid_search_bin(data, counts, edges, mass_min, mass_max, is_mc, tag,
                    n_coarse=21, n_fine=161, fine_half_steps=4):
    lo, hi = float(edges[0]), float(edges[-1])
    hist = to_th1(f"fit_h_{tag}", counts, edges, err_floor=1.0)
    hist.SetDirectory(0)

    func_def = ROOT.double_sided_cb if is_mc else ROOT.double_sided_cb_plus_bkg
    npar = 7 if is_mc else 9
    func = ROOT.TF1(f"fit_f_{tag}", func_def, lo, hi, npar)

    tail_init = prefit_dscb(hist, func, is_mc, lo, hi)
    chi2_fn, last = make_chi2(data, hist, func, mass_min, mass_max, is_mc, tail_init, lo, hi)

    # --- coarse scan over the full search range ---
    c_delta = np.linspace(-0.05, 0.05, n_coarse)
    c_sigma = np.linspace(5e-4, 0.06, n_coarse)
    c_grid = np.full((n_coarse, n_coarse), np.inf)
    for i, d in enumerate(c_delta):
        for j, s in enumerate(c_sigma):
            c_grid[i, j] = chi2_fn([d, s])

    ci, cj = np.unravel_index(int(np.argmin(c_grid)), c_grid.shape)
    c_dd = c_delta[1] - c_delta[0]
    c_ds = c_sigma[1] - c_sigma[0]

    # --- fine scan: ±fine_half_steps coarse steps around the coarse minimum ---
    # fine step = (2 * fine_half_steps * coarse_step) / (n_fine - 1)
    f_delta = np.linspace(
        np.clip(c_delta[ci] - fine_half_steps * c_dd, c_delta[0], c_delta[-1]),
        np.clip(c_delta[ci] + fine_half_steps * c_dd, c_delta[0], c_delta[-1]),
        n_fine,
    )
    f_sigma = np.linspace(
        np.clip(c_sigma[cj] - fine_half_steps * c_ds, c_sigma[0], c_sigma[-1]),
        np.clip(c_sigma[cj] + fine_half_steps * c_ds, c_sigma[0], c_sigma[-1]),
        n_fine,
    )
    f_grid = np.full((n_fine, n_fine), np.inf)
    for i, d in enumerate(f_delta):
        for j, s in enumerate(f_sigma):
            f_grid[i, j] = chi2_fn([d, s])

    flat_idx = int(np.argmin(f_grid))
    best_i, best_j = np.unravel_index(flat_idx, f_grid.shape)
    best_delta = float(f_delta[best_i])
    best_sigma = float(f_sigma[best_j])
    best_chi2 = float(f_grid[best_i, best_j])

    # Errors and correlation from the inverse Hessian (central finite differences).
    # This is more reliable than the Δchi2=1 profile when the chi2 landscape is steep
    # (i.e. total chi2 >> 1), because the Hessian uses the local curvature rather than
    # an absolute threshold that would otherwise fall within a fraction of one grid step.
    delta_err = sigma_err = corr = 0.0
    if 0 < best_i < n_fine - 1 and 0 < best_j < n_fine - 1:
        dd = f_delta[1] - f_delta[0]
        ds = f_sigma[1] - f_sigma[0]
        h00 = (f_grid[best_i + 1, best_j] - 2 * best_chi2 + f_grid[best_i - 1, best_j]) / dd**2
        h11 = (f_grid[best_i, best_j + 1] - 2 * best_chi2 + f_grid[best_i, best_j - 1]) / ds**2
        h01 = (f_grid[best_i + 1, best_j + 1] - f_grid[best_i + 1, best_j - 1]
               - f_grid[best_i - 1, best_j + 1] + f_grid[best_i - 1, best_j - 1]) / (4 * dd * ds)
        det = h00 * h11 - h01**2
        if det > 0 and h00 > 0 and h11 > 0:
            c00 = h11 / det   # Cov[delta, delta]
            c11 = h00 / det   # Cov[sigma, sigma]
            c01 = -h01 / det  # Cov[delta, sigma]
            if c00 > 0 and c11 > 0:
                delta_err = float(np.sqrt(c00))
                sigma_err = float(np.sqrt(c11))
                corr = float(c01 / np.sqrt(c00 * c11))
    # Fall back to profile-based errors if Hessian is degenerate
    if delta_err == 0.0:
        delta_err = _profile_err(f_delta, f_grid[:, best_j], best_chi2)
    if sigma_err == 0.0:
        sigma_err = _profile_err(f_sigma, f_grid[best_i, :], best_chi2)

    chi2_fn([best_delta, best_sigma])  # repopulate last{}

    ndf = max(last["ndf"] - 2, 1)
    return {
        "ok": True,
        "status": 0,
        "seed": None,
        "delta": best_delta, "delta_err": delta_err,
        "sigma": best_sigma, "sigma_err": sigma_err,
        "chi2": best_chi2, "ndf": ndf,
        "mu": last["mu"], "width": last["width"],
        "func": func,
        "corr": corr,
    }


def _fit_bin_worker(args):
    """Worker: run one pT bin fit and return a picklable result (no ROOT objects)."""
    low, high, counts, edges, data, mass_min, mass_max, is_mc, grid_search, fixed_sigma = args
    tag = f"{low}_{high}"
    if grid_search:
        r = grid_search_bin(data, counts, edges, mass_min, mass_max, is_mc, tag)
    else:
        r = fit_bin(data, counts, edges, mass_min, mass_max, is_mc, tag, fixed_sigma)
    func = r.pop("func")
    r["func_params"] = [func.GetParameter(i) for i in range(func.GetNpar())]
    r["func_xmin"] = func.GetXmin()
    r["func_xmax"] = func.GetXmax()
    r["func_npar"] = func.GetNpar()
    return r


def _rebuild_func(r, is_mc, tag):
    func_def = ROOT.double_sided_cb if is_mc else ROOT.double_sided_cb_plus_bkg
    func = ROOT.TF1(f"fit_f_{tag}", func_def, r["func_xmin"], r["func_xmax"], r["func_npar"])
    for i, p in enumerate(r["func_params"]):
        func.SetParameter(i, p)
    return func


def fit_all_bins(decays, histos, mass_min, mass_max, is_mc, outdir, sigma_eta_phi, grid_search=False):
    header = f"{'pt_lo':>6} {'pt_hi':>6} {'delta[MeV]':>14} {'sigma pT[MeV]':>14} {'sigma eta':>14} {'sigma phi':>14} {'chi2/ndf':>10} {'status':>7}"
    print("\n" + header)
    print("-" * len(header))

    # Split bins into those with data and those without
    valid, skipped_set = [], set()
    for ib, (low, high, counts, edges) in enumerate(histos):
        data = decays[ib]
        if data is None or counts.sum() <= 0:
            skipped_set.add(ib)
        else:
            valid.append((ib, low, high, counts, edges, data))

    # Fit all valid bins in parallel; one worker per bin, pool created once
    bin_args = [
        (low, high, counts, edges, data, mass_min, mass_max, is_mc, grid_search,
         sigma_eta_phi[i_pt] if sigma_eta_phi is not None else None)
        for i_pt, low, high, counts, edges, data in valid
    ]
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=len(bin_args), mp_context=ctx) as pool:
        raw = list(pool.map(_fit_bin_worker, bin_args))

    raw_by_idx = {ib: r for (ib, *_), r in zip(valid, raw)}

    # ROOT I/O is sequential; print in original bin order
    results = []
    for ib, (low, high, counts, edges) in enumerate(histos):
        if ib in skipped_set:
            print(f"{low:6.2f} {high:6.2f}   (skipped: no stats)")
            continue

        r = raw_by_idx[ib]
        tag = f"{low}_{high}"
        func = _rebuild_func(r, is_mc, tag)
        r["func"] = func
        r.update(lo=low, hi=high)
        results.append(r)

        centers = 0.5 * (edges[:-1] + edges[1:])
        model = np.array([func.Eval(c) for c in centers])

        suffix = f"pt_{low}_{high}"
        outdir.cd()
        to_th1(f"data_{suffix}", counts, edges).Write()
        to_th1(f"template_{suffix}", model, edges).Write()
        func.Write(f"fit_func_{suffix}")

        if not is_mc:
            signal_func = get_signal_func(func)
            signal_model = np.array([signal_func.Eval(c) for c in centers])
            bkg_func = get_background_func(func)
            bkg_model = np.array([bkg_func.Eval(c) for c in centers])
            to_th1(f"signal_{suffix}", signal_model, edges).Write()
            to_th1(f"background_{suffix}", bkg_model, edges).Write()
            signal_func.Write(f"fit_signal_func_{suffix}")
            bkg_func.Write(f"fit_bkg_func_{suffix}")

        print(f"{low:6.2f} {high:6.2f} "
              f"{1e3*r['delta_pt']:7.3f}±{1e3*r['delta_pt_err']:<6.3f} "
              f"{1e3*r['sigma_pt']:7.3f}±{1e3*r['sigma_pt_err']:<6.3f} "
              f"{r['sigma_eta']:7.3f}±{r['sigma_eta_err']:<6.3f} "
              f"{r['sigma_phi']:7.3f}±{r['sigma_phi_err']:<6.3f} "
              f"{r['chi2']/r['ndf']:10.2f} {r['status']:7d}")

    return results


def write_graphs(results, label, outdir):
    if not results:
        return
    outdir.cd()
    x = np.array([0.5 * (r["lo"] + r["hi"]) for r in results])
    ex = np.array([0.5 * (r["hi"] - r["lo"]) for r in results])
    for key in ("delta_pt", "sigma_pt", "sigma_eta", "sigma_phi"):
        y = np.array([r[key] for r in results])
        ey = np.array([r[f"{key}_err"] for r in results])
        g = ROOT.TGraphErrors(len(x), x, y, ex, ey)
        g.SetName(f"{key}_vs_pt_{label}")
        g.SetTitle(f"{key} vs p_{{T}} ({label});K^{{0}}_{{S}} p_{{T}} (GeV/c);{key} (GeV/c)")
        g.SetMarkerStyle(20)
        g.Write()
        # Normalized quantities
        g = ROOT.TGraphErrors(len(x), x, y / x, ex, ey / x)
        g.SetName(f"{key}pt_over_pt_vs_pt_{label}")
        g.SetTitle(f"{key}pt/pT vs p_{{T}} ({label});K^{{0}}_{{S}} p_{{T}} (GeV/c);{key}pt/p_{{T}}")
        g.SetMarkerStyle(20)
        g.Write()


    # Draw correlation for each pt bin as a 2D histogram (x=pt, y=corr)
    c_corr = ROOT.TCanvas(f"corr_vs_pt_{label}", f"Correlation vs pT ({label})", 800, 600)
    c_corr.Divide(len(x) // 4 + 1, 4)
    histos_corr = []
    for i, r in enumerate(results):
        c_corr.cd(i + 1)
        corr = r["corr"]
        histos_corr.append(ROOT.TH2D(f"corr_{r['lo']}_{r['hi']}_{label}", f"Correlation (pT {r['lo']:.2f}-{r['hi']:.2f} GeV/c);;",
                       len(corr), 0, len(corr), len(corr), 0, len(corr)))
        histos_corr[-1].GetXaxis().SetBinLabel(1, "delta_pt")
        histos_corr[-1].GetXaxis().SetBinLabel(2, "sigma_pt")
        if len(corr) > 2:
            histos_corr[-1].GetXaxis().SetBinLabel(3, "sigma_eta")
            histos_corr[-1].GetXaxis().SetBinLabel(4, "sigma_phi")
            histos_corr[-1].GetYaxis().SetBinLabel(4, "delta_pt")
            histos_corr[-1].GetYaxis().SetBinLabel(3, "sigma_pt")
        histos_corr[-1].GetYaxis().SetBinLabel(2, "sigma_eta")
        histos_corr[-1].GetYaxis().SetBinLabel(1, "sigma_phi")

        for j in range(len(corr)):
            for k in range(len(corr)):
                histos_corr[-1].SetBinContent(j + 1, len(corr) - k, corr[j][k])
        histos_corr[-1].GetZaxis().SetRangeUser(-1.0, 1.0)
        histos_corr[-1].Draw("COLZ TEXT")
    c_corr.Write()

    if len(corr) == 2:
        # We draw the correlation vs pT as a 1D graph only for the 2x2 case (delta_pt, sigma_pt)
        y = np.array([r["corr"][0][1] for r in results])
        ey = np.array([0.] * len(y))
        g = ROOT.TGraphErrors(len(x), x, y, ex, ey)
        g.SetName(f"g_corr_vs_pt_{label}")
        g.SetTitle(f"Correlation vs p_{{T}} ({label});K^{{0}}_{{S}} p_{{T}} (GeV/c);Correlation")
        g.SetMarkerStyle(20)
        g.Write()

def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    pt_bins, mass_min, mass_max = cfg["pt_bins"], float(cfg["mass_min"]), float(cfg["mass_max"])

    if not cfg["fit"]["do"]:
        print("Fit is disabled in cfg. Exiting.")
        return

    histos = {
        "data": read_histograms(cfg["fit"]["inputs"]["data"], pt_bins),
        "mc": read_histograms(cfg["fit"]["inputs"]["mc"], pt_bins),
    }

    sigma_eta_phi = None
    if cfg["fit"]["fix_sigma_phi_eta_to_mc"]:
        sigma_file = cfg["fit"]["file_for_sigma_phi_eta"]
        sigma_eta_phi = load_sigma_eta_phi(sigma_file, pt_bins)

    print("Loading generated decays (this scans the big tree) ...")
    decays = load_decays(cfg["fit"]["decays"], "decays", pt_bins,
                         cfg["fit"]["max_per_bin"], cfg["seed"],
                         float(cfg["eta_cut"]), float(cfg["alpha_cut"]),
                         float(cfg["qt_cut"]))

    outfile = ROOT.TFile(cfg["fit"]["output"], "RECREATE")
    for label in ("data", "mc"):
        print(f"\n=== Fitting {label} ===")
        subdir = outfile.mkdir(label)
        results = fit_all_bins(decays, histos[label], mass_min, mass_max,
                               label == "mc", subdir,
                               sigma_eta_phi=sigma_eta_phi,
                               grid_search=cfg["fit"].get("grid_search", False))
        write_graphs(results, label, outfile)  # graphs stay in the main folder

    outfile.Close()
    print(f"\nWrote results to {cfg['fit']['output']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit the K0s track-pT smearing (deltapt, sigmapt) per pT bin.")
    parser.add_argument("--config", default="config/config.yaml", help="path to config.yaml")
    args = parser.parse_args()

    main(args.config)
