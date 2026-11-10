"""Check that a list of colours stays distinguishable under colour-vision deficiency.
Usage: python cvd_check.py '#0072B2' '#D55E00' ...   (exit code 1 if any pair is too close)"""
import sys, itertools
import numpy as np
from colorspacious import cspace_convert

THRESH = 10.0  # minimum CAM02-UCS distance between any two colours

def rgb(h):
    h = h.lstrip("#"); return np.array([int(h[i:i+2], 16) / 255 for i in (0, 2, 4)])

def sim(c, kind):
    if kind == "normal": return c
    if kind == "grayscale":
        g = cspace_convert(c, "sRGB1", "JCh"); g[1] = 0
        return np.clip(cspace_convert(g, "JCh", "sRGB1"), 0, 1)
    cvd = {"name": "sRGB1+CVD", "cvd_type": kind, "severity": 100}
    return np.clip(cspace_convert(c, cvd, "sRGB1"), 0, 1)

def main(cols):
    ok = True
    for kind in ["normal", "deuteranomaly", "protanomaly", "tritanomaly", "grayscale"]:
        lab = [cspace_convert(sim(rgb(c), kind), "sRGB1", "CAM02-UCS") for c in cols]
        worst = min(((np.linalg.norm(lab[i] - lab[j]), cols[i], cols[j])
                     for i, j in itertools.combinations(range(len(cols)), 2)))
        flag = "OK" if worst[0] >= THRESH or kind == "grayscale" else "TOO CLOSE"
        if flag != "OK": ok = False
        print(f"{kind:14s} min distance {worst[0]:5.1f} ({worst[1]} vs {worst[2]}) {flag}")
    return ok

if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1:]) else 1)
