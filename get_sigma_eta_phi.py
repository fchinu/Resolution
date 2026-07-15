import yaml
import uproot
import pandas as pd
import numpy as np
import argparse
import ROOT


def add_kine_vars(df):
    """Add more kinematic variables to the dataframe."""
    _pimass = 0.13957018

    num = (df['fPxPos']**2 - df['fPxNeg']**2) + (df['fPyPos']**2 - df['fPyNeg']**2) + (df['fPzPos']**2 - df['fPzNeg']**2)
    den = (df['fPxPos'] + df['fPxNeg'])**2 + (df['fPyPos'] + df['fPyNeg'])**2 + (df['fPzPos'] + df['fPzNeg'])**2
    df['fAlpha'] = num / den
    df.loc[den == 0, 'fAlpha'] = pd.NA
    df['fPx'] = df['fPxPos'] + df['fPxNeg']
    df['fPy'] = df['fPyPos'] + df['fPyNeg']
    df['fPz'] = df['fPzPos'] + df['fPzNeg']
    df['fP2'] = df['fPx']**2 + df['fPy']**2 + df['fPz']**2
    df['fPhiPos'] = np.arctan2(df['fPyPos'], df['fPxPos'])
    df['fPhiNeg'] = np.arctan2(df['fPyNeg'], df['fPxNeg'])
    df['fPhiPosMC'] = np.arctan2(df['fPyPosMC'], df['fPxPosMC'])
    df['fPhiNegMC'] = np.arctan2(df['fPyNegMC'], df['fPxNegMC'])
    df['fEtaPos'] = -np.log(np.tan(0.5 * np.arctan2(np.sqrt(df['fPxPos']**2 + df['fPyPos']**2), df['fPzPos'])))
    df['fEtaNeg'] = -np.log(np.tan(0.5 * np.arctan2(np.sqrt(df['fPxNeg']**2 + df['fPyNeg']**2), df['fPzNeg'])))
    df['fEtaPosMC'] = -np.log(np.tan(0.5 * np.arctan2(np.sqrt(df['fPxPosMC']**2 + df['fPyPosMC']**2), df['fPzPosMC'])))
    df['fEtaNegMC'] = -np.log(np.tan(0.5 * np.arctan2(np.sqrt(df['fPxNegMC']**2 + df['fPyNegMC']**2), df['fPzNegMC'])))
    df['fPtPos'] = np.hypot(df['fPxPos'], df['fPyPos'])
    df['fPtNeg'] = np.hypot(df['fPxNeg'], df['fPyNeg'])
    df['fP2Pos'] = df['fPxPos']**2 + df['fPyPos']**2 + df['fPzPos']**2
    df['fP2Neg'] = df['fPxNeg']**2 + df['fPyNeg']**2 + df['fPzNeg']**2
    df['fEPos'] = np.sqrt(_pimass**2 + df['fP2Pos'])
    df['fENeg'] = np.sqrt(_pimass**2 + df['fP2Neg'])
    df['fK0sPt'] = np.hypot(df['fPx'], df['fPy'])
    df['fMass'] = np.sqrt((df['fEPos'] + df['fENeg'])**2 - df['fP2'])
    df['fQt'] = np.sqrt((df['fPyPos'] * df['fPz'] - df['fPzPos'] * df['fPy'])**2 + (df['fPzPos'] * df['fPx'] - df['fPxPos'] * df['fPz'])**2 + (df['fPxPos'] * df['fPy'] - df['fPyPos'] * df['fPx'])**2) / np.sqrt(df['fP2'])
    return df

def core_std(res, n_sigma=3, n_iter=5):
    sigma = res.std()
    for _ in range(n_iter):
        sigma = res[res.abs() < n_sigma * sigma].std()
    return sigma

def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    with uproot.open(cfg["sigma_eta_phi"]["input"]) as f:
        df = []
        for k in f.keys():
            if "O2mcv0tableap" in k:
                df.append(f[k].arrays(library="pd"))
    df = pd.concat(df)

    df_reco = df[df["fIsReco"] == True]
    df_reco = add_kine_vars(df_reco)
    df_sel = df_reco.query(f"fQt > {cfg['qt_cut']} and abs(fAlpha) < {cfg['alpha_cut']}")

    pt_bins = cfg["pt_bins"]

    std_phi = []
    std_eta = []
    std_phi_with_tails = []
    std_eta_with_tails = []

    for pt_min, pt_max in zip(pt_bins[:-1], pt_bins[1:]):
        df_pt = df_sel.query(f"{pt_min} < fPtPos < {pt_max}")
        res_eta = (df_pt['fEtaPos'] - df_pt['fEtaPosMC'])
        res_phi = (df_pt['fPhiPos'] - df_pt['fPhiPosMC'] + np.pi) % (2*np.pi) - np.pi
        df_pt = df_sel.query(f"{pt_min} < fPtNeg < {pt_max}")
        res_eta = pd.concat([res_eta, df_pt['fEtaNeg'] - df_pt['fEtaNegMC']])
        res_phi = pd.concat([res_phi, (df_pt['fPhiNeg'] - df_pt['fPhiNegMC'] + np.pi) % (2*np.pi) - np.pi])
        std_phi.append(core_std(res_phi))
        std_eta.append(core_std(res_eta))
        std_phi_with_tails.append(res_phi.std())
        std_eta_with_tails.append(res_eta.std())

    pt_centers = [(pt_min + pt_max) / 2 for pt_min, pt_max in zip(pt_bins[:-1], pt_bins[1:])]
    hist_phi = ROOT.TH1F("hist_phi", "hist_phi", len(pt_centers), np.array(pt_bins, dtype=np.float64))
    hist_eta = ROOT.TH1F("hist_eta", "hist_eta", len(pt_centers), np.array(pt_bins, dtype=np.float64))
    hist_phi_with_tails = ROOT.TH1F("hist_phi_with_tails", "hist_phi_with_tails", len(pt_centers), np.array(pt_bins, dtype=np.float64))
    hist_eta_with_tails = ROOT.TH1F("hist_eta_with_tails", "hist_eta_with_tails", len(pt_centers), np.array(pt_bins, dtype=np.float64))
    for i, (pt, phi, eta, phi_with_tails, eta_with_tails) in enumerate(zip(pt_centers, std_phi, std_eta, std_phi_with_tails, std_eta_with_tails)):
        hist_phi.SetBinContent(i + 1, phi)
        hist_eta.SetBinContent(i + 1, eta)
        hist_phi_with_tails.SetBinContent(i + 1, phi_with_tails)
        hist_eta_with_tails.SetBinContent(i + 1, eta_with_tails)

        hist_phi.SetBinError(i + 1, 1.e-12)
        hist_eta.SetBinError(i + 1, 1.e-12)
        hist_phi_with_tails.SetBinError(i + 1, 1.e-12)
        hist_eta_with_tails.SetBinError(i + 1, 1.e-12)

    hist_phi.SetLineColor(ROOT.kRed)
    hist_phi.SetMarkerColor(ROOT.kRed)
    hist_phi.SetMarkerStyle(ROOT.kFullCircle)
    hist_phi.SetMarkerSize(1.2)

    hist_eta.SetLineColor(ROOT.kAzure-3)
    hist_eta.SetMarkerColor(ROOT.kAzure-3)
    hist_eta.SetMarkerStyle(ROOT.kFullCircle)
    hist_eta.SetMarkerSize(1.2)

    hist_phi_with_tails.SetLineColor(ROOT.kRed)
    hist_phi_with_tails.SetMarkerColor(ROOT.kRed)
    hist_phi_with_tails.SetMarkerStyle(ROOT.kOpenCircle)
    hist_phi_with_tails.SetMarkerSize(1.2)

    hist_eta_with_tails.SetLineColor(ROOT.kAzure-3)
    hist_eta_with_tails.SetMarkerColor(ROOT.kAzure-3)
    hist_eta_with_tails.SetMarkerStyle(ROOT.kOpenCircle)
    hist_eta_with_tails.SetMarkerSize(1.2)

    c = ROOT.TCanvas("c", "c", 800, 600)
    c.DrawFrame(0, 0, 6, 0.045, ";#it{p}_{T} (GeV/c);#sigma_{#eta,#phi}")
    hist_phi.Draw("pe, same")
    hist_eta.Draw("pe, same")
    hist_phi_with_tails.Draw("pe, same")
    hist_eta_with_tails.Draw("pe, same")

    leg = ROOT.TLegend(0.55, 0.65, 0.85, 0.85)
    leg.SetBorderSize(0)
    leg.AddEntry(hist_phi, "#sigma_{#phi} (core)", "pe")
    leg.AddEntry(hist_eta, "#sigma_{#eta} (core)", "pe")
    leg.AddEntry(hist_phi_with_tails, "#sigma_{#phi} (with tails)", "pe")
    leg.AddEntry(hist_eta_with_tails, "#sigma_{#eta} (with tails)", "pe")
    leg.Draw("same")

    c.SaveAs(cfg["sigma_eta_phi"]["output"].replace(".root", ".pdf"))

    with ROOT.TFile.Open(cfg["sigma_eta_phi"]["output"], "RECREATE") as f:
        hist_phi.Write()
        hist_eta.Write()
        hist_phi_with_tails.Write()
        hist_eta_with_tails.Write()
        c.Write()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get the standard deviation of eta and phi resolutions as a function of pt")
    parser.add_argument("--config", default="config/config.yaml", help="path to config.yaml")
    args = parser.parse_args()

    main(args.config)