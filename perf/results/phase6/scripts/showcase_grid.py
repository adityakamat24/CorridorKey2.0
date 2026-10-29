import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

SURFACE, INK, INK2, MUTED = "#1a1a19", "#f0efec", "#c3c2b7", "#898781"
plt.rcParams.update({"font.family": "Segoe UI", "text.color": INK})
src, out, footer_claim = sys.argv[1], sys.argv[2], sys.argv[3]
times = {"hair_detail": (86, 52), "smoke": (93, 63), "motion_blur": (100, 73), "tos": (333, 130)}
stats = json.load(open(os.path.join(src, "stats.json")))
rows = [(k, label) for k, label in (("hair_detail", "Fine hair"), ("smoke", "Smoke"),
                                    ("motion_blur", "Motion blur"), ("tos", "Tears of Steel, 4K"))
        if k in stats]
cols = (("plate", "Green screen plate"), ("alpha", "Alpha matte"), ("comp", "Composite"))


def grid(name, title, subtitle, tag, aspect_of):
    heights = [aspect_of(k) for k, _ in rows]
    fig_w = 12.0
    cell_w = 0.27 * fig_w
    fig_h = 1.5 + sum(cell_w * h for h in heights) + 0.64 * len(rows) + 0.5 + 0.2 * footer_claim.count("\n")
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150, facecolor=SURFACE)
    fig.text(0.035, 1 - 0.45 / fig_h, title, fontsize=19, fontweight="bold", color=INK, va="top")
    fig.text(0.035, 1 - 0.95 / fig_h, subtitle, fontsize=11.5, color=INK2, va="top")
    y = fig_h - 1.5
    for k, label in rows:
        h = cell_w * aspect_of(k)
        s = stats[k]
        t_o, t_n = times[k]
        fig.text(0.035, (y - 0.08) / fig_h, f"{label}", fontsize=12.5, fontweight="bold", color=INK, va="top")
        fig.text(0.245, (y - 0.1) / fig_h,
                 f"{s['frames']} frames on an RTX 4090: original {t_o} s, optimised {t_n} s", fontsize=10, color=MUTED, va="top")
        y -= 0.42
        for i, (c, head) in enumerate(cols):
            ax = fig.add_axes([0.035 + i * 0.322, (y - h) / fig_h, 0.30, h / fig_h])
            ax.imshow(mpimg.imread(os.path.join(src, f"{k}_{tag}{c}.png")), cmap="gray", vmin=0, vmax=1)
            ax.set_axis_off()
            ax.text(0.0, 1.0, f" {head} ", transform=ax.transAxes, fontsize=9, color=INK,
                    va="top", ha="left", bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.75, pad=2))
        y -= h + 0.22
    fig.text(0.035, 0.12 / fig_h, "Plates: Hollywood Camera Work (hollywoodcamerawork.com). Tears of Steel: (CC) Blender "
             "Foundation, mango.blender.org.\nBoth runs at the same 8-frame windows, model load included. " + footer_claim,
             fontsize=8.5, color=MUTED, va="bottom", linespacing=1.5)
    fig.savefig(os.path.join(out, name), facecolor=SURFACE)
    plt.close(fig)


grid("corridorkey_greenscreen.png", "Same key, faster",
     "Optimised CorridorKey 2.0 on Hollywood Camera Work and Tears of Steel green screens.", "",
     lambda k: stats[k]["size"][1] / stats[k]["size"][0])
grid("corridorkey_greenscreen_zoom.png", "Edges up close",
     "Crops of the busiest matte edges, enlarged without smoothing.", "zoom_",
     lambda k: 1.0)
print("saved")
