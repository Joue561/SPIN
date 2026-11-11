# -*- coding: utf-8 -*-
"""
more_figures.py
把全部实验结果画成图（与 make_figures.py 同一套样式），并生成一张总览缩略图 contact_sheet.png，方便挑选放进论文的图。
数字全部取自已核对的服务器输出（collect_paper_numbers / verify_analysis / dataset_stats / 实验总结文档）。

用法：python more_figures.py [--out_dir figs]
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import make_figures as mf
from make_figures import BLUE, ORANGE, AQUA, VIOLET, GRAY, INK, INK2, GRID, COL_W, grid, save

YELLOW = "#eda100"


def hbar_labels(ax, bars, vals, fmt="{:.1f}", dx=3):
    for b, v in zip(bars, vals):
        x = b.get_width()
        ax.annotate(fmt.format(v), (x, b.get_y() + b.get_height() / 2), xytext=(dx if x >= 0 else -dx, 0),
                    textcoords="offset points", ha="left" if x >= 0 else "right", va="center",
                    fontsize=6.1, color=INK2)


def clean_y(ax):
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


# ------------------------------------------------------------------ 1. 显著性：ΔMF1 与 ΔF_avg 的 95% 置信区间
CI = [  # 设定, ΔMF1, lo, hi, ΔFavg, lo, hi
    ("C · GPT-4o-mini", 4.25, 3.34, 5.19, 2.90, 2.13, 3.67), ("C · GPT-4o", 3.15, 2.42, 3.87, 2.01, 1.41, 2.63),
    ("C · Llama-8B", 12.49, 10.87, 14.13, 7.09, 5.65, 8.53), ("C · Llama-70B", 1.37, 0.48, 2.25, -0.11, -0.82, 0.58),
    ("Z · GPT-4o-mini", 3.21, 1.04, 5.30, 4.17, 1.59, 6.82), ("Z · GPT-4o", 1.60, 0.47, 2.74, 2.44, 1.04, 3.87),
    ("Z · Llama-8B", 14.48, 11.95, 17.02, 3.00, 0.11, 5.88), ("Z · Llama-70B", 3.15, 0.82, 5.52, -3.08, -5.71, -0.47),
]


def fig_ci_forest(out):
    fig, axes = plt.subplots(1, 2, figsize=(COL_W * 2, 2.1), sharey=True, gridspec_kw={"wspace": 0.06})
    n = len(CI)
    for k, (ax, title) in enumerate(zip(axes, ("Δ Macro-F1 (95% CI)", "Δ F_avg (95% CI)"))):
        grid(ax, "x")
        for i, row in enumerate(CI):
            y = n - 1 - i
            d, lo, hi = row[1 + 3 * k: 4 + 3 * k]
            col = BLUE if row[0].startswith("C") else ORANGE
            sig = lo > 0 or hi < 0
            ax.plot([lo, hi], [y, y], color=col, linewidth=1.6, solid_capstyle="round", zorder=2)
            ax.scatter([d], [y], s=26, zorder=3, facecolor=col if sig else "white", edgecolor=col, linewidth=1.2)
            ax.annotate(f"{d:+.2f}".replace("-", "−"), (hi, y), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=6.1, color=INK2)
        ax.axvline(0, color=INK2, linewidth=0.7)
        ax.axhline(3.5, color=GRID, linewidth=0.8)
        ax.set_title(title, loc="left", fontsize=7.5, color=INK)
        ax.set_xlim(-7, 20)
        clean_y(ax)
    axes[0].set_yticks(range(n), [r[0] for r in CI][::-1])
    axes[1].annotate("hollow = CI includes 0", (1, -0.18), xycoords="axes fraction", ha="right", fontsize=6.1,
                     color=INK2)
    save(fig, out, "fig_ci_forest")


# ------------------------------------------------------------------ 2. 与已发表结果对比
PUB_C = [("GLAN, reported", 66.10, GRAY), ("GLAN, reproduced", 64.04, GRAY), ("GLAN + SPIN", 65.43, BLUE),
         ("GPT-4o-mini kNN", 68.95, GRAY), ("GPT-4o-mini kNN + SPIN", 71.30, BLUE),
         ("GPT-4o kNN", 76.63, GRAY), ("GPT-4o kNN + SPIN", 77.94, BLUE)]
PUB_Z = [("GPT-4o-mini zero-shot, reported", 38.16, GRAY), ("SITPCL, reported", 43.81, GRAY),
         ("SITPCL, reproduced", 43.40, GRAY), ("SITPCL + SPIN", 44.47, ORANGE),
         ("GPT-4o-mini kNN", 43.71, GRAY), ("GPT-4o-mini kNN + SPIN", 46.91, ORANGE),
         ("GPT-4, reported", 48.62, GRAY), ("GPT-4o kNN", 56.26, GRAY), ("GPT-4o kNN + SPIN", 57.86, ORANGE)]


def fig_published(out):
    fig, axes = plt.subplots(1, 2, figsize=(COL_W * 2, 2.3), gridspec_kw={"wspace": 0.95})
    for ax, data, title, lim in ((axes[0], PUB_C, "C-MTCSD · topic-averaged F_avg", (55, 82)),
                                 (axes[1], PUB_Z, "ZS-CSD · Macro-F1", (30, 62))):
        grid(ax, "x")
        ys = range(len(data))[::-1]
        bars = ax.barh(list(ys), [d[1] - lim[0] for d in data], left=lim[0], height=0.62,
                       color=[d[2] for d in data], zorder=3)
        for b, d in zip(bars, data):
            ax.annotate(f"{d[1]:.2f}", (d[1], b.get_y() + b.get_height() / 2), xytext=(3, 0),
                        textcoords="offset points", va="center", fontsize=6.1, color=INK2)
        ax.set_yticks(list(ys), [d[0] for d in data])
        ax.set_xlim(*lim)
        ax.set_title(title, loc="left", fontsize=7.5, color=INK)
        clean_y(ax)
    save(fig, out, "fig_published")


# ------------------------------------------------------------------ 3. 插件：其他提示策略与监督模型
PLUG = [("C · 4o-mini CoT", 63.71, 70.74, True), ("C · 4o-mini full path", 62.78, 69.67, True),
        ("C · GLAN (3 seeds)", 68.02, 70.05, True),
        ("Z · 4o-mini CoT", 46.11, 51.49, True), ("Z · 4o-mini full path", 44.08, 46.34, True),
        ("Z · GPT-4o CoT", 65.94, 66.38, False), ("Z · SITPCL (3 seeds)", 43.40, 44.47, False)]


def fig_plugin(out):
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    grid(ax, "x")
    n = len(PLUG)
    for i, (name, b, s, sig) in enumerate(PLUG):
        y = n - 1 - i
        col = BLUE if name.startswith("C") else ORANGE
        ax.plot([b, s], [y, y], color=col, linewidth=2.2, zorder=2)
        ax.scatter([b], [y], s=26, color=GRAY, zorder=3, edgecolor="white", linewidth=1)
        ax.scatter([s], [y], s=26, zorder=4, facecolor=col if sig else "white", edgecolor=col, linewidth=1.2)
        ax.annotate(f"+{s - b:.2f}", (max(b, s), y), xytext=(5, 0), textcoords="offset points", va="center",
                    fontsize=6.1, color=INK2)
    ax.axhline(3.5 - 0.0, color=GRID, linewidth=0.8)
    ax.set_yticks(range(n), [p[0] for p in PLUG][::-1])
    ax.set_xlabel("Macro-F1 (%)"); ax.set_xlim(40, 76)
    clean_y(ax)
    ax.annotate("grey = base · coloured = + SPIN · hollow = not significant", (0, 1.02),
                xycoords="axes fraction", fontsize=6.1, color=INK2)
    save(fig, out, "fig_plugin")


# ------------------------------------------------------------------ 4. 监督模型各种子的增益
SEEDS = {"GLAN (C-MTCSD)": [("seed 0", 1.94, True), ("seed 1", 2.35, True), ("seed 2", 1.78, True)],
         "SITPCL (ZS-CSD)": [("seed 1234", -0.32, False), ("seed 42", 1.55, False), ("seed 2026", 1.97, False)]}


def fig_supervised_seeds(out):
    fig, ax = plt.subplots(figsize=(COL_W, 1.7))
    grid(ax, "y")
    x, ticks, labels = 0, [], []
    for (model, rows), col in zip(SEEDS.items(), (BLUE, ORANGE)):
        for seed, d, sig in rows:
            ax.bar(x, d, width=0.7, color=col if sig else "white", edgecolor=col, linewidth=1.2, zorder=3)
            ax.annotate(f"{d:+.2f}".replace("-", "−"), (x, d), xytext=(0, 3 if d >= 0 else -9),
                        textcoords="offset points", ha="center", fontsize=6.1, color=INK2)
            ticks.append(x); labels.append(seed); x += 1
        ax.annotate(model, (x - 2, -1.25), ha="center", fontsize=6.8, color=INK, annotation_clip=False)
        x += 0.8
    ax.axhline(0, color=INK2, linewidth=0.7)
    ax.set_xticks(ticks, labels, fontsize=6.3)
    ax.set_ylabel("Δ Macro-F1"); ax.set_ylim(-0.8, 3)
    ax.annotate("filled = p < 0.001 · hollow = not significant", (1, 1.02), xycoords="axes fraction",
                ha="right", fontsize=6.1, color=INK2)
    save(fig, out, "fig_supervised_seeds")


# ------------------------------------------------------------------ 5. SITPCL 的迁移诊断
TRANSFER = [("seed 1234", 6.23, -0.32, 2.88), ("seed 42", 4.57, 1.55, 4.65), ("seed 2026", 6.73, 1.97, 4.11)]


def fig_sitpcl_transfer(out):
    fig, ax = plt.subplots(figsize=(COL_W, 1.8))
    grid(ax, "y")
    w = 0.26
    for i, (seed, v, p, u) in enumerate(TRANSFER):
        for j, (val, col) in enumerate(((v, GRAY), (p, ORANGE), (u, AQUA))):
            x = i + (j - 1) * (w + 0.02)
            ax.bar(x, val, width=w, color=col, zorder=3)
            ax.annotate(f"{val:+.1f}".replace("-", "−"), (x, val), xytext=(0, 2 if val >= 0 else -8),
                        textcoords="offset points", ha="center", fontsize=5.9, color=INK2)
    ax.axhline(0, color=INK2, linewidth=0.7)
    ax.set_xticks(range(3), [t[0] for t in TRANSFER])
    ax.set_ylabel("Δ Macro-F1"); ax.set_ylim(-1, 8)
    h = [plt.Rectangle((0, 0), 1, 1, color=GRAY, label="validation (λ chosen here)"),
         plt.Rectangle((0, 0), 1, 1, color=ORANGE, label="test, protocol"),
         plt.Rectangle((0, 0), 1, 1, color=AQUA, label="test, E and λ fit on test")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, frameon=False,
              handlelength=1, columnspacing=0.8, fontsize=6.3)
    save(fig, out, "fig_sitpcl_transfer")


# ------------------------------------------------------------------ 6. 检索键消融
KEYS = ["full key", "type + depth", "type", "depth only", "random"]
ABL = {"C-MTCSD · kNN": (68.25, [72.51, 71.46, 70.74, 70.00, 68.25]),
       "ZS-CSD · kNN": (43.71, [46.91, 45.17, 44.46, 43.39, 43.71]),
       "ZS-CSD · CoT": (46.11, [51.49, 48.66, 48.00, 49.49, 46.11])}


def fig_ablation(out):
    fig, axes = plt.subplots(1, 3, figsize=(COL_W * 2, 1.75), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, (title, (base, vals)), col in zip(axes, ABL.items(), (BLUE, ORANGE, ORANGE)):
        grid(ax, "x")
        ys = list(range(len(KEYS)))[::-1]
        deltas = [v - base for v in vals]
        bars = ax.barh(ys, deltas, height=0.62, color=[col] + [GRAY] * (len(KEYS) - 1), zorder=3)
        hbar_labels(ax, bars, deltas, "{:+.2f}")
        ax.axvline(0, color=INK2, linewidth=0.7)
        ax.set_title(title, loc="left", fontsize=7.5, color=INK)
        ax.set_xlabel("Δ Macro-F1 over base"); ax.set_xlim(-1, 7)
        clean_y(ax)
    axes[0].set_yticks(list(range(len(KEYS)))[::-1], KEYS)
    save(fig, out, "fig_ablation")


# ------------------------------------------------------------------ 7. 相对 / 绝对 / 组合知识
KT = [  # 设定, 基础, 只用相对, 只用绝对, 组合
    ("C · GPT-4o-mini", 68.25, 70.65, 70.78, 72.51), ("C · GPT-4o", 74.62, 76.99, 77.74, 77.78),
    ("C · Llama-8B", 53.81, 62.11, 66.09, 66.30), ("C · Llama-70B", 73.41, 74.38, 74.64, 74.78),
    ("Z · GPT-4o-mini", 43.71, 44.66, 45.33, 46.91), ("Z · GPT-4o", 56.26, 57.86, 57.06, 57.86),
    ("Z · Llama-8B", 29.05, 41.19, 43.11, 43.52), ("Z · Llama-70B", 43.97, 49.51, 47.98, 47.13),
    ("Z · 4o-mini CoT", 46.11, 49.56, 49.75, 51.49),
]


def fig_knowledge_type(out):
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    grid(ax, "x")
    n = len(KT)
    for i, (name, b, r, a_, c) in enumerate(KT):
        y = n - 1 - i
        ax.plot([0, max(r, a_, c) - b], [y, y], color=GRID, linewidth=2.5, zorder=1)
        ax.scatter([r - b], [y], s=24, marker="s", color=AQUA, zorder=3, edgecolor="white", linewidth=0.8)
        ax.scatter([a_ - b], [y], s=26, marker="D", color=VIOLET, zorder=3, edgecolor="white", linewidth=0.8)
        ax.scatter([c - b], [y], s=30, marker="o", color=BLUE if name.startswith("C") else ORANGE, zorder=4,
                   edgecolor="white", linewidth=0.8)
    ax.axvline(0, color=INK2, linewidth=0.7)
    ax.axhline(4.5, color=GRID, linewidth=0.8)
    ax.set_yticks(range(n), [k[0] for k in KT][::-1])
    ax.set_xlabel("Δ Macro-F1 over base"); ax.set_xlim(-0.5, 15.5)
    clean_y(ax)
    h = [plt.Line2D([], [], marker="s", ls="", color=AQUA, label="relative only", ms=4),
         plt.Line2D([], [], marker="D", ls="", color=VIOLET, label="absolute only", ms=4),
         plt.Line2D([], [], marker="o", ls="", color=INK2, label="both (SPIN)", ms=4.5)]
    ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=3, frameon=False,
              handletextpad=0.2, columnspacing=1.0)
    save(fig, out, "fig_knowledge_type")


# ------------------------------------------------------------------ 8. 检索质量：关系准确率
RET = {"C-MTCSD": [("SPIN full key", 58.7), ("LR, surface cues", 57.0), ("depth only", 52.7),
                   ("majority per topic", 51.6), ("LLM annotation", 49.3), ("random key", 44.7)],
       "ZS-CSD": [("SPIN full key", 55.0), ("LR, surface cues", 52.3), ("depth only", 49.4),
                  ("majority", 49.4), ("random key", 46.3)]}


def fig_retrieval(out):
    fig, axes = plt.subplots(1, 2, figsize=(COL_W * 2, 1.7), gridspec_kw={"wspace": 0.55})
    for ax, (title, rows), col in zip(axes, RET.items(), (BLUE, ORANGE)):
        grid(ax, "x")
        ys = list(range(len(rows)))[::-1]
        bars = ax.barh(ys, [r[1] - 40 for r in rows], left=40, height=0.62,
                       color=[col] + [GRAY] * (len(rows) - 1), zorder=3)
        for b, r in zip(bars, rows):
            ax.annotate(f"{r[1]:.1f}", (r[1], b.get_y() + b.get_height() / 2), xytext=(3, 0),
                        textcoords="offset points", va="center", fontsize=6.1, color=INK2)
        ax.set_yticks(ys, [r[0] for r in rows])
        ax.set_xlim(40, 62); ax.set_xlabel("Relation accuracy (%)")
        ax.set_title(title, loc="left", fontsize=7.5, color=INK)
        clean_y(ax)
    save(fig, out, "fig_retrieval")


# ------------------------------------------------------------------ 9. 父子误差相关性
RR = [("C · GPT-4o-mini", 1.16), ("C · GPT-4o", 1.13), ("C · Llama-8B", 1.12), ("C · Llama-70B", 1.39),
      ("C · GLAN s0", 1.21), ("C · GLAN s1", 1.23), ("C · GLAN s2", 1.27),
      ("Z · GPT-4o-mini", 1.12), ("Z · GPT-4o", 1.00), ("Z · Llama-8B", 0.92), ("Z · Llama-70B", 1.19),
      ("Z · GPT-4o CoT", 1.33), ("Z · SITPCL s1234", 0.94), ("Z · SITPCL s42", 0.83), ("Z · SITPCL s2026", 0.97)]


def fig_error_corr(out):
    fig, ax = plt.subplots(figsize=(COL_W, 2.9))
    grid(ax, "x")
    n = len(RR)
    for i, (name, v) in enumerate(RR):
        y = n - 1 - i
        col = BLUE if name.startswith("C") else ORANGE
        ax.plot([1, v], [y, y], color=col, linewidth=1.6, zorder=2)
        ax.scatter([v], [y], s=24, color=col, zorder=3, edgecolor="white", linewidth=0.8)
        ax.annotate(f"{v:.2f}", (v, y), xytext=(4 if v >= 1 else -4, 0), textcoords="offset points",
                    ha="left" if v >= 1 else "right", va="center", fontsize=6, color=INK2)
    ax.axvline(1, color=INK2, linewidth=0.7)
    ax.axhline(7.5, color=GRID, linewidth=0.8)
    ax.set_yticks(range(n), [r[0] for r in RR][::-1])
    ax.set_xlabel("Risk ratio  P(child wrong | parent wrong) / P(child wrong | parent right)", fontsize=6.6)
    ax.set_xlim(0.7, 1.5)
    clean_y(ax)
    save(fig, out, "fig_error_corr")


# ------------------------------------------------------------------ 10. 动机：提示策略与「知识放进 prompt vs 放在模型外」
PROMPTS = [("current reply only", 55.12, GRAY), ("+ parent", 61.32, GRAY), ("full path", 62.78, GRAY),
           ("context window", 62.07, GRAY), ("chain-of-thought", 63.71, GRAY), ("kNN few-shot", 68.25, GRAY),
           ("window + annotation in prompt", 61.33, YELLOW), ("window + patterns in prompt", 59.46, YELLOW),
           ("annotation used outside (tree inference)", 66.00, BLUE), ("kNN + SPIN", 72.51, BLUE)]


def fig_prompting(out):
    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    grid(ax, "x")
    ys = list(range(len(PROMPTS)))[::-1]
    bars = ax.barh(ys, [p[1] - 50 for p in PROMPTS], left=50, height=0.62, color=[p[2] for p in PROMPTS], zorder=3)
    for b, p in zip(bars, PROMPTS):
        ax.annotate(f"{p[1]:.1f}", (p[1], b.get_y() + b.get_height() / 2), xytext=(3, 0),
                    textcoords="offset points", va="center", fontsize=6.1, color=INK2)
    ax.set_yticks(ys, [p[0] for p in PROMPTS])
    ax.set_xlim(50, 76); ax.set_xlabel("Macro-F1 (%), GPT-4o-mini, C-MTCSD")
    ax.axhline(3.5, color=GRID, linewidth=0.8); ax.axhline(1.5, color=GRID, linewidth=0.8)
    clean_y(ax)
    h = [plt.Rectangle((0, 0), 1, 1, color=GRAY, label="prompting only"),
         plt.Rectangle((0, 0), 1, 1, color=YELLOW, label="knowledge in prompt"),
         plt.Rectangle((0, 0), 1, 1, color=BLUE, label="knowledge outside model")]
    ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.3, 1.0), ncol=3, frameon=False,
              handlelength=1, columnspacing=0.8, fontsize=6.3)
    save(fig, out, "fig_prompting")


# ------------------------------------------------------------------ 11. 数据：深度分布
DEPTH_C = {0: 452, 1: 5800, 2: 11291, 3: 4405, 4: 1578, 5: 737}
DEPTH_Z = {1: 299, 2: 2739, 3: 5422, 4: 3648, 5: 2330, 6: 1255, 7: 646, 8: 297, 9: 164, 10: 96,
           11: 52, 12: 33, 13: 14, 14: 11, 15: 2}


def fig_data_depth(out):
    fig, ax = plt.subplots(figsize=(COL_W, 1.7))
    grid(ax, "y")
    # 以「根 = 第 0 层」对齐：ZS-CSD 的深度 1 是根
    tc, tz = sum(DEPTH_C.values()), sum(DEPTH_Z.values())
    xs = range(0, 15)
    w = 0.4
    ax.bar([x - w / 2 - 0.01 for x in xs], [100 * DEPTH_C.get(x, 0) / tc for x in xs], width=w, color=BLUE,
           zorder=3, label="C-MTCSD (mean 2.13)")
    ax.bar([x + w / 2 + 0.01 for x in xs], [100 * DEPTH_Z.get(x + 1, 0) / tz for x in xs], width=w, color=ORANGE,
           zorder=3, label="ZS-CSD (mean 3.91)")
    ax.set_xticks(list(xs)); ax.set_xlabel("Distance from root (root = 0)")
    ax.set_ylabel("% of nodes")
    ax.legend(frameon=False, loc="upper right")
    save(fig, out, "fig_data_depth")


# ------------------------------------------------------------------ 总览缩略图
ALL = [("fig_main", "主结果"), ("fig_ci_forest", "显著性 / 置信区间"), ("fig_gain_vs_base", "增益 vs 基础强度"),
       ("fig_published", "与已发表结果对比"), ("fig_plugin", "插件：其他基础模型"),
       ("fig_supervised_seeds", "监督模型各种子"), ("fig_sitpcl_transfer", "SITPCL 迁移诊断"),
       ("fig_ablation", "检索键消融"), ("fig_knowledge_type", "相对 / 绝对知识"), ("fig_retrieval", "检索质量"),
       ("fig_correction_room", "修正空间上界"), ("fig_parent_correct", "父节点对错与净收益"),
       ("fig_error_corr", "父子误差相关"), ("fig_prompting", "动机：知识放哪里"), ("fig_data_depth", "数据：深度分布")]


def contact_sheet(out):
    plt.rcParams["font.family"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei", "DejaVu Sans"]
    items = [(n, t) for n, t in ALL if (out / f"{n}.png").exists()]
    cols = 3
    rows = (len(items) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.2, rows * 3.3))
    for ax in axes.flat:
        ax.axis("off")
    for k, (ax, (n, t)) in enumerate(zip(axes.flat, items), 1):
        ax.imshow(mpimg.imread(out / f"{n}.png"))
        ax.set_title(f"{k}. {n}  {t}", fontsize=10, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(out / "contact_sheet.png", dpi=110)
    plt.close(fig)
    print("  ✅ contact_sheet.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="figs")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    mf.fig_main(out); mf.fig_gain_vs_base(out); mf.fig_correction_room(out); mf.fig_parent_correct(out)
    for f in (fig_ci_forest, fig_published, fig_plugin, fig_supervised_seeds, fig_sitpcl_transfer, fig_ablation,
              fig_knowledge_type, fig_retrieval, fig_error_corr, fig_prompting, fig_data_depth):
        f(out)
    contact_sheet(out)


if __name__ == "__main__":
    main()
