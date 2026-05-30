import argparse
import os
import numpy as np
import ROOT
import uproot
import yaml

PIMASS = 0.13957018

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


def load_decays(path, tree_name, pt_bins, max_per_bin, seed, chunk=20_000_000):
    """
        For every K0s we store: pos_pt, neg_pt, cosh(eta_pos), cosh(eta_neg)
        and c12 = cos(dphi) + sinh(eta_pos)*sinh(eta_neg)
    """
    nbins = len(pt_bins) - 1
    edges = np.asarray(pt_bins, dtype=np.float64)
    cols = {k: [] for k in ("pos_pt", "neg_pt", "cosh_p", "cosh_n", "c12")}
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
        cos_dphi = np.cos(arrays["pos_phi"].astype(np.float64)
                          - arrays["neg_phi"].astype(np.float64))
        scanned += pos_pt.size

        # true K0s pT = |pt_pos + pt_neg| (vector sum of the two daughters)
        k0s_pt = np.sqrt(pos_pt**2 + neg_pt**2 + 2.0 * pos_pt * neg_pt * cos_dphi)
        ibin = np.digitize(k0s_pt, edges) - 1  # -1 / nbins => out of range

        sinh_p = np.sinh(arrays["pos_eta"].astype(np.float64))
        sinh_n = np.sinh(arrays["neg_eta"].astype(np.float64))
        cosh_p = np.sqrt(1.0 + sinh_p**2)
        cosh_n = np.sqrt(1.0 + sinh_n**2)

        # c12 is the angular part of the momentum dot product:
        #   p1.p2 = pt1 pt2 (cos phi1 cos phi2 + sin phi1 sin phi2) + pz1 pz2
        #         = pt1 pt2 [cos(phi1 - phi2) + sinh eta1 sinh eta2]
        # so c12 = cos(dphi) + sinh(eta1) sinh(eta2).
        c12 = cos_dphi + sinh_p * sinh_n

        chunk_cols = {"pos_pt": pos_pt, "neg_pt": neg_pt,
                      "cosh_p": cosh_p, "cosh_n": cosh_n, "c12": c12}

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
        data["z_p"] = rng.standard_normal(n)
        data["z_n"] = rng.standard_normal(n)
        decays.append(data)
    return decays

def smeared_mass(data, delta, sigma):
    pt_p = data["pos_pt"] + delta + sigma * data["z_p"]
    pt_n = data["neg_pt"] + delta + sigma * data["z_n"]
    ok = (pt_p > 0.0) & (pt_n > 0.0)
    pt_p, pt_n = pt_p[ok], pt_n[ok]
    e1 = np.sqrt(PIMASS**2 + (pt_p * data["cosh_p"][ok]) ** 2)
    e2 = np.sqrt(PIMASS**2 + (pt_n * data["cosh_n"][ok]) ** 2)
    m2 = 2.0 * PIMASS**2 + 2.0 * (e1 * e2 - pt_p * pt_n * data["c12"][ok])
    return np.sqrt(np.clip(m2, 0.0, None))


def make_chi2(data, hist, func, mass_min, mass_max, is_mc, tail_init, lo, hi):
    norm0 = max(tail_init["norm"], 1.0)
    last = {}

    def chi2(par):
        delta, sigma = par[0], par[1]
        mass = smeared_mass(data, delta, sigma)
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


def fit_bin(data, counts, edges, mass_min, mass_max, is_mc, tag, max_retries=5):
    lo, hi = float(edges[0]), float(edges[-1])
    hist = to_th1(f"fit_h_{tag}", counts, edges, err_floor=1.0)  # modified chi2 errors
    hist.SetDirectory(0)

    func_def = ROOT.double_sided_cb if is_mc else ROOT.double_sided_cb_plus_bkg
    npar = 7 if is_mc else 9
    func = ROOT.TF1(f"fit_f_{tag}", func_def, lo, hi, npar)

    tail_init = prefit_dscb(hist, func, is_mc, lo, hi)

    chi2, last = make_chi2(data, hist, func, mass_min, mass_max, is_mc, tail_init, lo, hi)
    functor = ROOT.Math.Functor(chi2, 2)

    # Starting points for (deltapt, sigmapt; the first is the nominal seed,
    # the rest are fallbacks, tried only if Migrad fails to find a valid minimum
    seeds = [(0.0, 0.01), (0.0, 0.005), (0.0, 0.02), (0.0, 0.05),
             (-0.01, 0.01), (0.01, 0.01), (-0.03, 0.03), (0.03, 0.03)]

    best = None
    for s_delta, s_sigma in seeds[: 1 + max_retries]:
        m = ROOT.Math.Factory.CreateMinimizer("Minuit2", "Migrad")
        m.SetFunction(functor)
        m.SetMaxFunctionCalls(100000)
        m.SetTolerance(0.001)
        m.SetStrategy(1)
        m.SetPrintLevel(0)
        m.SetLimitedVariable(0, "deltapt", s_delta, 1e-4, -0.2, 0.2)
        m.SetLimitedVariable(1, "sigmapt", s_sigma, 1e-4, 1e-5, 0.2)

        ok = bool(m.Minimize())
        cand = {"ok": ok, "status": int(m.Status()), "chi2": m.MinValue(),
                "delta": m.X()[0], "delta_err": m.Errors()[0],
                "sigma": m.X()[1], "sigma_err": m.Errors()[1],
                "seed": (s_delta, s_sigma)}
        # prefer a valid minimum (ok), then the lowest chi2
        if best is None or (cand["ok"], -cand["chi2"]) > (best["ok"], -best["chi2"]):
            best = cand
        if ok:
            break

    corr = m.Correlation(0, 1)

    chi2([best["delta"], best["sigma"]])

    ndf = max(last["ndf"] - 2, 1) 
    return {
        "ok": best["ok"],
        "status": best["status"],  # Minuit2 status: 0 = converged OK
        "seed": best["seed"],
        "delta": best["delta"], "delta_err": best["delta_err"],
        "sigma": best["sigma"], "sigma_err": best["sigma_err"],
        "chi2": best["chi2"], "ndf": ndf,
        "mu": last["mu"], "width": last["width"],
        "func": func,
        "corr": corr,
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

def fit_all_bins(decays, histos, mass_min, mass_max, is_mc, outdir):
    results = []
    header = f"{'pt_lo':>6} {'pt_hi':>6} {'delta[MeV]':>14} {'sigma[MeV]':>14} {'chi2/ndf':>10} {'status':>7}"
    print("\n" + header)
    print("-" * len(header))

    for ib, (low, high, counts, edges) in enumerate(histos):
        data = decays[ib]
        if data is None or counts.sum() <= 0:
            print(f"{low:6.2f} {high:6.2f}   (skipped: no stats)")
            continue

        r = fit_bin(data, counts, edges, mass_min, mass_max, is_mc, f"{low}_{high}")
        r.update(lo=low, hi=high)
        results.append(r)

        centers = 0.5 * (edges[:-1] + edges[1:])
        model = np.array([r["func"].Eval(c) for c in centers])

        if not is_mc:
            signal_func = get_signal_func(r["func"])
            signal_model = np.array([signal_func.Eval(c) for c in centers])
            bkg_func = get_background_func(r["func"])
            bkg_model = np.array([bkg_func.Eval(c) for c in centers])

        suffix = f"pt_{low}_{high}"
        outdir.cd()
        to_th1(f"data_{suffix}", counts, edges).Write()
        to_th1(f"template_{suffix}", model, edges).Write()
        r["func"].Write(f"fit_func_{suffix}")
        if not is_mc:
            to_th1(f"signal_{suffix}", signal_model, edges).Write()
            to_th1(f"background_{suffix}", bkg_model, edges).Write()
            signal_func.Write(f"fit_signal_func_{suffix}")
            bkg_func.Write(f"fit_bkg_func_{suffix}")

        print(f"{low:6.2f} {high:6.2f} "
              f"{1e3*r['delta']:7.3f}±{1e3*r['delta_err']:<6.3f} "
              f"{1e3*r['sigma']:7.3f}±{1e3*r['sigma_err']:<6.3f} "
              f"{r['chi2']/r['ndf']:10.2f} {r['status']:7d}")

    return results


def write_graphs(results, label, outdir):
    if not results:
        return
    outdir.cd()
    x = np.array([0.5 * (r["lo"] + r["hi"]) for r in results])
    ex = np.array([0.5 * (r["hi"] - r["lo"]) for r in results])
    for key in ("delta", "sigma"):
        y = np.array([r[key] for r in results])
        ey = np.array([r[f"{key}_err"] for r in results])
        g = ROOT.TGraphErrors(len(x), x, y, ex, ey)
        g.SetName(f"{key}pt_vs_pt_{label}")
        g.SetTitle(f"{key}pt vs p_{{T}} ({label});K^{{0}}_{{S}} p_{{T}} (GeV/c);{key}pt (GeV/c)")
        g.SetMarkerStyle(20)
        g.Write()
        # Normalized quantities
        g = ROOT.TGraphErrors(len(x), x, y / x, ex, ey / x)
        g.SetName(f"{key}pt_over_pt_vs_pt_{label}")
        g.SetTitle(f"{key}pt/pT vs p_{{T}} ({label});K^{{0}}_{{S}} p_{{T}} (GeV/c);{key}pt/p_{{T}}")
        g.SetMarkerStyle(20)
        g.Write()

    y = np.array([r["corr"] for r in results])
    ey = np.array([0.] * len(y))
    g = ROOT.TGraphErrors(len(x), x, y, ex, ey)
    g.SetName(f"corr_vs_pt_{label}")
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

    print("Loading generated decays (this scans the big tree) ...")
    decays = load_decays(cfg["fit"]["decays"], "decays", pt_bins,
                         cfg["fit"]["max_per_bin"], cfg["seed"])

    outfile = ROOT.TFile(cfg["fit"]["output"], "RECREATE")
    for label in ("data", "mc"):
        print(f"\n=== Fitting {label} ===")
        subdir = outfile.mkdir(label)
        results = fit_all_bins(decays, histos[label], mass_min, mass_max,
                               label == "mc", subdir)
        write_graphs(results, label, outfile)  # graphs stay in the main folder

    outfile.Close()
    print(f"\nWrote results to {cfg['fit']['output']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit the K0s track-pT smearing (deltapt, sigmapt) per pT bin.")
    parser.add_argument("--config", default="config/config.yaml", help="path to config.yaml")
    args = parser.parse_args()

    main(args.config)
