"""
Figure 3: Adaptive Class-Balanced Focal Loss (Stage 5)
Vẽ sơ đồ loss function dùng cho Figure 1 hoặc figure riêng.

Run: python draw_stage5_loss.py
Output: figure3_adaptive_cb_loss.pdf / .png
"""

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

matplotlib.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'mathtext.fontset': 'cm',
})

# ─────────────────────────────────────────────────────────────────────────────
# COLOURS
# ─────────────────────────────────────────────────────────────────────────────
C_FOCAL = '#E67E22'
C_STATIC = '#2980B9'
C_ADAPTIVE = '#8E44AD'
C_LOSS = '#C0392B'
C_CALLBACK = '#27AE60'
C_BG = '#FDF6EC'
C_BORDER = '#D35400'


def darken(hex_color, factor=0.2):
    h = hex_color.lstrip('#')
    rgb = [int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
    rgb = [max(0, c - factor) for c in rgb]
    return '#{:02x}{:02x}{:02x}'.format(*(int(c * 255) for c in rgb))


def rbox(ax, x, y, w, h, text='', sub='', fc='#ccc', tc='white',
         fs=9, lw=1.2, ls='-', zorder=5):
    ec = darken(fc, 0.15)
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle='round,pad=0.04,rounding_size=0.12',
        fc=fc, ec=ec, lw=lw, linestyle=ls, zorder=zorder))
    if text:
        ty = y + h / 2 + (0.12 if sub else 0)
        ax.text(x + w / 2, ty, text,
                ha='center', va='center', fontsize=fs, color=tc,
                fontweight='bold', linespacing=1.3, zorder=zorder + 1)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.18, sub,
                ha='center', va='center', fontsize=fs - 1.5,
                color=tc, alpha=0.9, style='italic', zorder=zorder + 1)


def arrow(ax, x1, y1, x2, y2, color='#333', lw=1.4, zorder=10):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                shrinkA=3, shrinkB=3),
                zorder=zorder)


def dashed_arrow(ax, x1, y1, x2, y2, color='#555', lw=1.2, rad=0.0, zorder=10):
    style = f'arc3,rad={rad}' if rad else 'arc3,rad=0.0'
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                linestyle='dashed', shrinkA=3, shrinkB=3,
                                connectionstyle=style),
                zorder=zorder)


def draw_stage5():
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-1.0, 11.0)
    ax.set_aspect('equal')
    ax.axis('off')

    # ══════════════════════════════════════════════════════════════════════
    # OUTER DASHED BOX — Stage 5
    # ══════════════════════════════════════════════════════════════════════
    ax.add_patch(FancyBboxPatch(
        (0.0, 0.0), 12.0, 10.5,
        boxstyle='round,pad=0.1,rounding_size=0.25',
        fc=C_BG, ec=C_BORDER, lw=2.0, linestyle='dashed',
        alpha=0.6, zorder=0))
    ax.text(6.0, 10.7, 'STAGE 5 — Adaptive Class-Balanced Focal Loss\n(Training Only)',
            ha='center', va='bottom', fontsize=12, fontweight='bold',
            color='#7B341E')

    # ══════════════════════════════════════════════════════════════════════
    # INPUT BOXES (top)
    # ══════════════════════════════════════════════════════════════════════
    # y_pred
    rbox(ax, 1.5, 9.2, 2.8, 0.9,
         text=r'$\hat{y}$ (Softmax Probs)', sub='shape: (B, 3)',
         fc='#4CAF50', tc='white', fs=9)

    # y_true
    rbox(ax, 7.5, 9.2, 2.8, 0.9,
         text=r'$y$ (One-Hot Label)', sub='shape: (B, 3)',
         fc='#78909C', tc='white', fs=9)

    # ══════════════════════════════════════════════════════════════════════
    # THREE COMPONENT BOXES
    # ══════════════════════════════════════════════════════════════════════
    # --- Focal Term (left) ---
    rbox(ax, 0.3, 6.2, 3.4, 2.4, fc=C_FOCAL, tc='white', fs=9,
         text='Focal Term')
    ax.text(2.0, 7.6, r'$p_t = \sum_c y_c \cdot \hat{y}_c$',
            ha='center', va='center', fontsize=10, color='white', zorder=6)
    ax.text(2.0, 7.0, r'$\mathrm{focal} = (1 - p_t)^{\gamma}$',
            ha='center', va='center', fontsize=10, color='white', zorder=6)
    ax.text(2.0, 6.5, r'$\gamma = 2.0$',
            ha='center', va='center', fontsize=9, color='#FDEBD0', zorder=6)

    # --- Static CB Weight (center) ---
    rbox(ax, 4.2, 6.2, 3.6, 2.4, fc=C_STATIC, tc='white', fs=9,
         text='Static CB Weight')
    ax.text(6.0, 7.6,
            r'$w_c = \frac{1-\beta}{1-\beta^{n_c}}$',
            ha='center', va='center', fontsize=11, color='white', zorder=6)
    ax.text(6.0, 6.9, r'$\beta = 0.9999$',
            ha='center', va='center', fontsize=9, color='#D6EAF8', zorder=6)
    # Dataset numbers
    ax.text(6.0, 6.45,
            'Fully: 1050→0.828\n'
            'Partial: 306→1.930\n'
            'Spoiled: 704→1.241',
            ha='center', va='center', fontsize=7.0, color='#EBF5FB',
            linespacing=1.4, zorder=6, family='monospace')

    # --- Adaptive Factor (right) ---
    rbox(ax, 8.3, 6.2, 3.4, 2.4, fc=C_ADAPTIVE, tc='white', fs=9,
         text='Adaptive Factor')
    ax.text(10.0, 7.55,
            r'$\mathrm{target}_c = (1 - \mathrm{recall}_c) + \varepsilon$',
            ha='center', va='center', fontsize=9, color='white', zorder=6)
    ax.text(10.0, 7.0,
            r'$f_c^{(t)} = (1-\tau) f_c^{(t-1)} + \tau \cdot \mathrm{target}_c$',
            ha='center', va='center', fontsize=9, color='white', zorder=6)
    ax.text(10.0, 6.5,
            r'$\tau=0.3,\ \varepsilon=0.1,\ f_c^{(0)}=1$',
            ha='center', va='center', fontsize=8, color='#E8DAEF', zorder=6)

    # ══════════════════════════════════════════════════════════════════════
    # ARROWS: inputs → components
    # ══════════════════════════════════════════════════════════════════════
    # y_pred → Focal
    arrow(ax, 2.9, 9.2, 2.0, 8.6, color=C_FOCAL, lw=1.4)
    # y_true → Focal
    arrow(ax, 8.9, 9.2, 2.0, 8.6, color='#555', lw=1.0)
    # y_true → Static CB (chỉ cần y_true để chọn weight)
    arrow(ax, 8.9, 9.2, 6.0, 8.6, color=C_STATIC, lw=1.0)

    # ══════════════════════════════════════════════════════════════════════
    # COMBINED WEIGHT BOX
    # ══════════════════════════════════════════════════════════════════════
    rbox(ax, 3.5, 4.4, 5.0, 1.0, fc='#5D6D7E', tc='white', fs=9,
         text=r'$w_{\mathrm{combined}} = w_c^{\mathrm{static}} \times f_c^{(t)} \,/\, \mathrm{mean}$')

    # Arrows from Static CB and Adaptive → combined_w
    arrow(ax, 6.0, 6.2, 6.0, 5.4, color=C_STATIC, lw=1.4)
    arrow(ax, 10.0, 6.2, 7.5, 5.4, color=C_ADAPTIVE, lw=1.4)

    # ══════════════════════════════════════════════════════════════════════
    # FINAL LOSS FORMULA BOX
    # ══════════════════════════════════════════════════════════════════════
    rbox(ax, 1.0, 2.2, 10.0, 1.5, fc=C_LOSS, tc='white', fs=9,
         text='')
    ax.text(6.0, 3.1,
            r'$\mathcal{L} = \frac{1}{B}\sum_{b=1}^{B}'
            r' w_{c_b}^{\mathrm{static}} \cdot f_{c_b}^{(t)}'
            r' \cdot (1-p_t)^{\gamma} \cdot (-\log\, p_t)$',
            ha='center', va='center', fontsize=12, color='white',
            fontweight='bold', zorder=6)
    ax.text(6.0, 2.5,
            'Adaptive Class-Balanced Focal Loss',
            ha='center', va='center', fontsize=9, color='#FADBD8',
            style='italic', zorder=6)

    # Arrows: focal + combined → loss
    arrow(ax, 2.0, 6.2, 3.0, 3.7, color=C_FOCAL, lw=1.4)
    arrow(ax, 6.0, 4.4, 6.0, 3.7, color='#5D6D7E', lw=1.4)

    # ══════════════════════════════════════════════════════════════════════
    # BACKPROP ARROW
    # ══════════════════════════════════════════════════════════════════════
    ax.text(6.0, 1.7, r'$\downarrow$ Backpropagation + Adam Update',
            ha='center', va='center', fontsize=9, color='#7B241C',
            fontweight='bold')

    # ══════════════════════════════════════════════════════════════════════
    # ADAPTIVE CALLBACK BOX (bottom-right, dashed)
    # ══════════════════════════════════════════════════════════════════════
    ax.add_patch(FancyBboxPatch(
        (7.8, -0.7), 3.8, 2.0,
        boxstyle='round,pad=0.06,rounding_size=0.15',
        fc='#E8F8F5', ec=C_CALLBACK, lw=1.5, linestyle='dashed',
        zorder=4))
    ax.text(9.7, 1.1, 'AdaptiveWeightCallback',
            ha='center', va='center', fontsize=8.5, color=C_CALLBACK,
            fontweight='bold', zorder=5)
    ax.text(9.7, 0.6,
            'After each epoch:\n'
            '1. predict(val_ds)\n'
            r'2. recall$_c$ per class' + '\n'
            r'3. EMA update $f_c^{(t+1)}$' + '\n'
            '4. normalize',
            ha='center', va='center', fontsize=7.2, color='#1E8449',
            linespacing=1.5, zorder=5)

    # Dashed curved arrow: Callback → Adaptive Factor (feedback loop)
    dashed_arrow(ax, 11.2, 1.3, 11.2, 6.2,
                 color=C_CALLBACK, lw=1.8, rad=0.0)
    ax.text(11.5, 3.7, r'update $f_c^{(t+1)}$',
            ha='left', va='center', fontsize=7.5, color=C_CALLBACK,
            style='italic', rotation=90)

    # Arrow from Loss → Callback (trigger)
    dashed_arrow(ax, 8.0, 2.2, 8.5, 1.3,
                 color='#7B241C', lw=1.2)
    ax.text(7.6, 1.6, 'end of\nepoch',
            ha='center', va='center', fontsize=7, color='#7B241C',
            style='italic')

    # ══════════════════════════════════════════════════════════════════════
    # LEGEND / ANNOTATION
    # ══════════════════════════════════════════════════════════════════════
    ax.text(0.3, -0.5,
            r'$\gamma=2.0$ (focal), $\beta=0.9999$ (CB), '
            r'$\tau=0.3$ (EMA), $\varepsilon=0.1$ (min factor)',
            ha='left', va='center', fontsize=8, color='#555', style='italic')

    plt.tight_layout(pad=0.3)
    for fmt in ('pdf', 'png'):
        plt.savefig(f'figure3_adaptive_cb_loss.{fmt}',
                    dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Saved: figure3_adaptive_cb_loss.pdf / .png')
    plt.show()
    plt.close()


if __name__ == '__main__':
    draw_stage5()
