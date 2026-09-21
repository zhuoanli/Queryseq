"""Colour-blind safety check for the figure palette.

The dataviz guidance says to compute this rather than eyeball it, and ships a
Node validator; this cluster has no Node runtime, so the same checks are
implemented here.  Distances are Euclidean in OKLab x100; CVD is simulated with
the Machado et al. (2009) severity-1.0 matrices on linear RGB.

Checks, on the adjacent pairlist (the relevant one for lines and grouped bars):
  normal-vision dE >= 15   hard floor
  CVD dE        >= 8       target, 6-8 legal only with a second encoding
  contrast vs surface >= 3:1, else the series needs a visible direct label
"""
import itertools
import sys

import numpy as np

M_LIN2LMS = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                      [0.2119034982, 0.6806995451, 0.1073969566],
                      [0.0883024619, 0.2817188376, 0.6299787005]])
M_LMS2LAB = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                      [1.9779984951, -2.4285922050, 0.4505937099],
                      [0.0259040371, 0.7827717662, -0.8086757660]])
CVD = {
    "protan": np.array([[0.152286, 1.052583, -0.204868],
                        [0.114503, 0.786281, 0.099216],
                        [-0.003882, -0.048116, 1.051998]]),
    "deutan": np.array([[0.367322, 0.860646, -0.227968],
                        [0.280085, 0.672501, 0.047413],
                        [-0.011820, 0.042940, 0.968881]]),
    "tritan": np.array([[1.255528, -0.076749, -0.178779],
                        [-0.078411, 0.930809, 0.147602],
                        [0.004733, 0.691367, 0.303900]]),
}


def hex2rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def oklab(rgb):
    lms = M_LIN2LMS @ to_linear(rgb)
    return M_LMS2LAB @ np.cbrt(np.clip(lms, 0, None))


def de(a, b):
    return float(np.linalg.norm(oklab(a) - oklab(b)) * 100)


def simulate(rgb, kind):
    return np.clip(CVD[kind] @ to_linear(rgb), 0, 1) ** (1 / 2.4)


def rel_lum(rgb):
    return float(np.dot(to_linear(rgb), [0.2126, 0.7152, 0.0722]))


def contrast(a, b):
    l1, l2 = sorted([rel_lum(a), rel_lum(b)], reverse=True)
    return (l1 + 0.05) / (l2 + 0.05)


def main(hexes, surface="#ffffff", pairs="adjacent"):
    cols = [hex2rgb(h) for h in hexes]
    idx = (list(zip(range(len(cols) - 1), range(1, len(cols))))
           if pairs == "adjacent"
           else list(itertools.combinations(range(len(cols)), 2)))
    ok = True
    print(f"{'pair':>9s} {'normal':>8s} {'protan':>8s} {'deutan':>8s} "
          f"{'tritan':>8s}   verdict")
    for i, j in idx:
        n = de(cols[i], cols[j])
        cv = {k: de(simulate(cols[i], k), simulate(cols[j], k)) for k in CVD}
        worst = min(cv.values())
        bad = n < 15 or worst < 6
        warn = 6 <= worst < 8
        ok &= not bad
        v = "FAIL" if bad else ("warn(needs 2nd encoding)" if warn else "pass")
        print(f"{i + 1}-{j + 1:>7d} {n:8.1f} {cv['protan']:8.1f} "
              f"{cv['deutan']:8.1f} {cv['tritan']:8.1f}   {v}")
    print()
    s = hex2rgb(surface)
    for i, c in enumerate(cols):
        r = contrast(c, s)
        print(f"  slot {i + 1} {hexes[i]} contrast vs surface {r:.2f}:1"
              f"{'  -> needs a visible direct label' if r < 3 else ''}")
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    hx = sys.argv[1].split(",")
    sf = sys.argv[2] if len(sys.argv) > 2 else "#ffffff"
    pr = sys.argv[3] if len(sys.argv) > 3 else "adjacent"
    sys.exit(main(hx, sf, pr))
