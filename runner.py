import argparse
import os
import subprocess
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))

def run(command):
    """Run a command, printing it first. Exit on failure."""
    display = " ".join(command)
    print(f"\n>>> {display}")
    result = subprocess.run(command)
    if result.returncode != 0:
        sys.exit(f"Step failed (exit {result.returncode}): {display}")


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)


    gen = cfg["generation"]
    if gen["do"]:
        output = gen["output"]
        nevents = gen["nEvents"]
        seed = cfg["seed"]
        nthreads = gen["nthreads"]
        macro = os.path.join(HERE, "simulate_decays.cxx")
        cmd = f'{macro}+("{output}", {nevents}, {seed}, {nthreads})'
        run(["root", "-l", "-b", "-q", cmd])

    sig = cfg["signal"]
    if sig["data"]["do"] or sig["mc"]["do"]:
        run([sys.executable, os.path.join(HERE, "signal_producer.py"), config_path])

    if cfg["sigma_eta_phi"]["do"]:
        run([sys.executable, os.path.join(HERE, "get_sigma_eta_phi.py"),
             "--config", config_path])

    if cfg["fit"]["do"]:
        run([sys.executable, os.path.join(HERE, "fit_resolution.py"),
             "--config", config_path])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Handle the workflow of the analysis."
    )
    parser.add_argument(
        "config",
        help="path to config YAML (default: config/config.yaml)",
    )
    args = parser.parse_args()
    
    main(args.config)
