#!/usr/bin/env python3
"""Regenerate docs/benchmarks.{png,svg} from the measured speed probes.

Run from the repo root:
    python3 docs/make_benchmarks.py

Inputs are the raw speed_probe.py outputs, so the chart can never drift from the
data it claims to show:
    speed_published.json  -- this build
    speed_nospec.json     -- same build, --speculative-algorithm off

Aggregate throughput is total completion tokens / longest wall time across the
three concurrent streams. It is NOT the sum of the per-stream decode rates: the
streams overlap, so summing their rates overstates what the box actually
delivers (60.9 vs 57.7 tok/s on this run).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

KINDS = ["prose", "code", "math"]
BAR = {"prose": "#2b7cd3", "code": "#f0662a", "math": "#1fa97a"}
PURPLE = "#8b5cf6"
GSM8K = {"plain": "GSM8K 96/100", "build": "GSM8K 188/200"}


def load(path):
    with open(path) as fh:
        d = json.load(fh)
    single = {k: d["single"][k]["dec"] for k in KINDS}
    multi = d["multi"]
    agg = sum(v["ct"] for v in multi.values()) / max(v["wall"] for v in multi.values())
    return single, agg


def main():
    plain, plain_agg = load(os.path.join(ROOT, "speed_nospec.json"))
    build, build_agg = load(os.path.join(ROOT, "speed_published.json"))

    fig = plt.figure(figsize=(15.2, 9.6), dpi=100, facecolor="#fafafa")
    gs = fig.add_gridspec(1, 2, width_ratios=[2.25, 1], left=0.055, right=0.985,
                          top=0.90, bottom=0.165, wspace=0.16)
    ax = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1])
    for a in (ax, ax2):
        a.set_facecolor("#fafafa")
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.spines["left"].set_color("#999")
        a.spines["bottom"].set_color("#999")
        a.tick_params(colors="#333", labelsize=13)
        a.set_axisbelow(True)
        a.grid(axis="y", color="#d8d8d8", lw=0.9)

    # ---- left: single-stream decode --------------------------------------
    w, gap = 0.26, 0.02
    centres = [0.45, 1.55]
    for gi, (src, centre) in enumerate(zip((plain, build), centres)):
        for ki, k in enumerate(KINDS):
            x = centre + (ki - 1) * (w + gap)
            v = src[k]
            ax.bar(x, v, w, color=BAR[k], label=k if gi == 0 else None, zorder=3)
            ax.text(x, v + 0.55, f"{v:.1f}", ha="center", va="bottom",
                    fontsize=14, color="#222")

    top = max(max(build.values()), max(plain.values()))
    ax.set_ylim(0, top * 1.24)
    for centre, key in zip(centres, ("plain", "build")):
        ax.text(centre, top * 1.155, GSM8K[key], ha="center", fontsize=14, color="#555")

    ax.set_xticks(centres)
    ax.set_xticklabels(["plain decoding\n(no speculation)", "this build"], fontsize=15)
    ax.set_xlim(-0.15, 2.15)
    ax.set_ylabel("decode tok/s, single stream (400-token greedy, after first token)",
                  fontsize=14, color="#333")
    ax.set_title("MiMo-V2.6-Flash on one DGX Spark: decode speed",
                 fontsize=19, color="#222", pad=26, loc="left")
    ax.legend(frameon=False, ncol=3, fontsize=15, loc="upper left",
              bbox_to_anchor=(0.0, 1.035))

    # ---- right: 3-stream aggregate ---------------------------------------
    xs = [0.3, 1.0]
    for x, v in zip(xs, (plain_agg, build_agg)):
        ax2.bar(x, v, 0.42, color=PURPLE, zorder=3)
        ax2.text(x, v + 0.7, f"{v:.1f}", ha="center", va="bottom", fontsize=14, color="#222")
    ax2.set_ylim(0, build_agg * 1.20)
    ax2.set_xlim(-0.05, 1.35)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(["plain decoding", "this build"], fontsize=14,
                        rotation=28, ha="right")
    ax2.set_ylabel("aggregate tok/s, 3 concurrent streams", fontsize=14, color="#333")
    ax2.set_title("3 streams (prose + code + math)", fontsize=17, color="#222",
                  pad=26, loc="left")

    fig.text(0.013, 0.050,
             "All experts resident, 256k context. 4 running requests. "
             "Aggregate = 1200 completion tokens / longest stream wall time (not the sum of per-stream rates).",
             fontsize=12.5, color="#444")
    fig.text(0.013, 0.017,
             "GSM8K = greedy, 100 problems (200 where two slices were run); "
             "repeat runs of one build differ by up to 3 per 100 and up to 5 % in speed.",
             fontsize=12.5, color="#444")

    png = os.path.join(HERE, "benchmarks.png")
    fig.savefig(png, facecolor="#fafafa")
    fig.savefig(os.path.join(HERE, "benchmarks.svg"), facecolor="#fafafa")
    print(f"wrote {png}")
    print(f"  single  prose {build['prose']:.1f}  code {build['code']:.1f}  math {build['math']:.1f}")
    print(f"  aggregate {build_agg:.1f} (plain {plain_agg:.1f})")


if __name__ == "__main__":
    main()
