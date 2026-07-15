import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml

PIMASS = 0.13957018


def run_signal(cfg: dict, is_mc: bool):
    cfg_signal = cfg["signal"]
    if is_mc:
        input_file = cfg_signal["mc"]["input"]
        output_file = cfg_signal["mc"]["output"]
    else:
        input_file = cfg_signal["data"]["input"]
        output_file = cfg_signal["data"]["output"]
    
    table = "O2mcv0tableap" if is_mc else "O2v0tableap"
    with uproot.open(input_file) as f:
        df = []
        for k in f.keys():
            if table in k:
                df.append(f[k].arrays(library="pd"))
    df = pd.concat(df)

    if is_mc:
        df = df[df["fIsReco"] == 1]

    df['fPx'] = df['fPxPos'] + df['fPxNeg']
    df['fPy'] = df['fPyPos'] + df['fPyNeg']
    df['fPz'] = df['fPzPos'] + df['fPzNeg']
    df['fP2'] = df['fPx']**2 + df['fPy']**2 + df['fPz']**2
    df['fPtPos'] = np.hypot(df['fPxPos'], df['fPyPos'])
    df['fPtNeg'] = np.hypot(df['fPxNeg'], df['fPyNeg'])
    df['fP2Pos'] = df['fPxPos']**2 + df['fPyPos']**2 + df['fPzPos']**2
    df['fP2Neg'] = df['fPxNeg']**2 + df['fPyNeg']**2 + df['fPzNeg']**2
    df['fEPos'] = np.sqrt(PIMASS**2 + df['fP2Pos'])
    df['fENeg'] = np.sqrt(PIMASS**2 + df['fP2Neg'])
    df['fK0sPt'] = np.hypot(df['fPx'], df['fPy'])
    df['fMass'] = np.sqrt((df['fEPos'] + df['fENeg'])**2 - df['fP2'])
    num = (df['fPxPos']**2 - df['fPxNeg']**2) + (df['fPyPos']**2 - df['fPyNeg']**2) + (df['fPzPos']**2 - df['fPzNeg']**2)
    den = (df['fPxPos'] + df['fPxNeg'])**2 + (df['fPyPos'] + df['fPyNeg'])**2 + (df['fPzPos'] + df['fPzNeg'])**2
    df['fAlpha'] = num / den
    df['fQt'] = np.sqrt((df['fPyPos'] * df['fPz'] - df['fPzPos'] * df['fPy'])**2 + (df['fPzPos'] * df['fPx'] - df['fPxPos'] * df['fPz'])**2 + (df['fPxPos'] * df['fPy'] - df['fPyPos'] * df['fPx'])**2) / np.sqrt(df['fP2'])

    df = df.query(f"fMass > {cfg['mass_min']} and fMass < {cfg['mass_max']} and abs(fEta) < {cfg['eta_cut']} and abs(fAlpha) < {cfg['alpha_cut']} and fQt > {cfg['qt_cut']}")

    with uproot.recreate(output_file) as f:
        for pt_min, pt_max in zip(cfg["pt_bins"][:-1], cfg["pt_bins"][1:]):
            df_pt = df.query(f"fPtPos > {pt_min} and fPtPos < {pt_max}")
            bins = (cfg["mass_max"] - cfg["mass_min"]) * 2000
            hist_mass = np.histogram(df_pt["fMass"], bins=100, range=(cfg["mass_min"], cfg["mass_max"]))
            f[f"mass_pt_{pt_min}_{pt_max}"] = hist_mass

def signal(cfg: dict):
    cfg_signal = cfg["signal"]
    
    if cfg_signal["data"]["do"]:
        run_signal(cfg, is_mc=False)
    if cfg_signal["mc"]["do"]:
        run_signal(cfg, is_mc=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot K0s mass distribution")
    parser.add_argument("config", type=str, help="config yaml file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    signal(config)

