"""
3D Isometric Architecture Diagrams — EfficientNetB4 + FSDA
Style : CNN paper style (inspired by DurianLSNet Fig.8)
        Prominent 3D blocks with grid texture, clean section boxes, legend.

Figure 1 : Overall pipeline  (21 x 9 in,  horizontal flow)
Figure 2 : FSDA block detail (14 x 18 in, dual-branch vertical)

Run:  python draw_paper_figures.py
Output: figure1_overall_architecture.pdf / .png
        figure2_fsda_block.pdf / .png
"""

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

matplotlib.rcParams.update({
    'font.family':  'DejaVu Sans',
    'font.size':    9,
    'pdf.fonttype': 42,
    'ps.fonttype':  42,
})

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def _h2r(h):
    h = h.lstrip('#')
    return [int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)]

def _r2h(rgb):
    return '#{:02x}{:02x}{:02x}'.format(
        *(max(0, min(255, int(v * 255))) for v in rgb))

def lighten(h, f=0.25):
    return _r2h([min(1.0, c + f) for c in _h2r(h)])

def darken(h, f=0.22):
    return _r2h([max(0.0, c - f) for c in _h2r(h)])

# ─────────────────────────────────────────────────────────────────────────────
# PALETTE
# ─────────────────────────────────────────────────────────────────────────────
C = dict(
    frozen   = '#9DC3E6',
    backbone = '#2E75B6',
    feat     = '#2E75B6',
    fsda     = '#C55A11',
    freq     = '#ED7D31',
    spatial  = '#548235',
    fusion   = '#7030A0',
    gap      = '#1F6B8E',
    bn       = '#2E8B57',
    fc       = '#375623',
    drop     = '#767171',
    softmax  = '#C00000',
    text     = '#1A1A1A',
    white    = '#FFFFFF',
    in_r     = '#FF6B6B',
    in_g     = '#4CAF50',
    in_b     = '#2196F3',
)

# ─────────────────────────────────────────────────────────────────────────────
# CORE 3D BLOCK WITH GRID TEXTURE
# ─────────────────────────────────────────────────────────────────────────────
def block3d(ax, x, y, w, h, dx=0.22, dy=0.14,
            fc=None, lw=0.9, zorder=3,
            nh=6, nv=3, alpha=1.0):
    """
    3D rectangular prism with grid texture on all three visible faces.
    (x,y) = bottom-left of front face.  dx,dy = depth offset right+up.
    """
    if fc is None:
        fc = C['feat']
    tc = lighten(fc, 0.30)
    sc = darken(fc, 0.24)
    ec = '#1C1C1C'
    gkw = dict(color='white', lw=0.35, alpha=0.40, zorder=zorder + 1)

    # Front face
    ax.add_patch(plt.Polygon(
        [[x, y], [x+w, y], [x+w, y+h], [x, y+h]],
        fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=zorder))
    for i in range(1, nh):
        gy = y + i * h / nh
        ax.plot([x, x+w], [gy, gy], **gkw)
    for j in range(1, nv):
        gx = x + j * w / nv
        ax.plot([gx, gx], [y, y+h], **gkw)

    # Top face
    ax.add_patch(plt.Polygon(
        [[x, y+h], [x+w, y+h], [x+w+dx, y+h+dy], [x+dx, y+h+dy]],
        fc=tc, ec=ec, lw=lw, alpha=alpha, zorder=zorder))
    for j in range(1, nv):
        t = j / nv
        ax.plot([x+t*w, x+t*w+dx], [y+h, y+h+dy], **gkw)
    for i in range(1, nh):
        t = i / nh
        y0 = y + h + t * dy
        ax.plot([x + t*dx, x+w + t*dx], [y0, y0], **gkw)

    # Right side face
    ax.add_patch(plt.Polygon(
        [[x+w, y], [x+w+dx, y+dy], [x+w+dx, y+h+dy], [x+w, y+h]],
        fc=sc, ec=ec, lw=lw, alpha=alpha, zorder=zorder))
    for i in range(1, nh):
        t = i / nh
        y0 = y + t * h
        ax.plot([x+w, x+w+dx], [y0, y0 + dy], **gkw)
    for j in range(1, 3):
        t = j / 3
        x0 = x + w + t * dx
        ax.plot([x0, x0], [y + t*dy, y+h + t*dy], **gkw)


def dim_label(ax, cx, cy_top, text, fs=6.8, zorder=15):
    """Italic dimension above a block."""
    ax.text(cx, cy_top + 0.08, text,
            ha='center', va='bottom', fontsize=fs,
            color=C['text'], style='italic', zorder=zorder)


def name_label(ax, cx, cy_bot, text, sub='', fs=8.5, zorder=15):
    """Bold name below a block."""
    ax.text(cx, cy_bot - 0.12, text,
            ha='center', va='top', fontsize=fs,
            fontweight='bold', color=C['text'], zorder=zorder,
            linespacing=1.3)
    if sub:
        ax.text(cx, cy_bot - 0.40, sub,
                ha='center', va='top', fontsize=fs - 1.5,
                color='#555', style='italic', zorder=zorder)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def arrow(ax, x1, y1, x2, y2, color='#333', lw=1.6, zorder=14):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                shrinkA=3, shrinkB=3,
                                connectionstyle='arc3,rad=0.0'),
                zorder=zorder)


def curved_arrow(ax, x1, y1, x2, y2, rad=0.3, color='#555', lw=1.4, zorder=14):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                shrinkA=4, shrinkB=4,
                                connectionstyle=f'arc3,rad={rad}'),
                zorder=zorder)


def rbox(ax, x, y, w, h, text='', sub='',
         fc=None, lw=1.0, tc='white', fs=8.0,
         radius=0.10, zorder=5):
    if fc is None:
        fc = C['backbone']
    ec = darken(fc, 0.18)
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f'round,pad=0.04,rounding_size={radius}',
        fc=fc, ec=ec, lw=lw, zorder=zorder))
    if text:
        ty = y + h / 2 + (0.055 if sub else 0)
        ax.text(x + w/2, ty, text,
                ha='center', va='center', fontsize=fs, color=tc,
                fontweight='bold', linespacing=1.25, zorder=zorder + 1)
    if sub:
        ax.text(x + w/2, y + h/2 - 0.11, sub,
                ha='center', va='center', fontsize=fs - 1.5,
                color=tc, alpha=0.88, style='italic', zorder=zorder + 1)


def section(ax, x, y, w, h, title='', tfc='#F0F4FF', ec='#4472C4',
            lw=1.6, ls='-', tfs=8.5, tcolor='#1F3864', radius=0.18, zorder=0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f'round,pad=0.05,rounding_size={radius}',
        fc=tfc, ec=ec, lw=lw, linestyle=ls, alpha=0.42, zorder=zorder))
    if title:
        ax.text(x + w/2, y + h + 0.10, title,
                ha='center', va='bottom', fontsize=tfs,
                fontweight='bold', color=tcolor, zorder=zorder + 2)


def op_circle(ax, cx, cy, r=0.22, symbol='+', fc=None, fs=13, zorder=12):
    if fc is None:
        fc = C['fusion']
    ax.add_patch(plt.Circle((cx, cy), r,
                             fc=fc, ec=darken(fc, 0.2), lw=1.2, zorder=zorder))
    ax.text(cx, cy, symbol,
            ha='center', va='center', fontsize=fs,
            color='white', fontweight='bold', zorder=zorder + 1)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — OVERALL ARCHITECTURE  (21 x 9 in)
# ─────────────────────────────────────────────────────────────────────────────
def draw_figure1():
    fig, ax = plt.subplots(figsize=(21, 9))
    ax.set_xlim(0, 21)
    ax.set_ylim(0, 9)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    CY = 4.55  # centre-y of main flow

    # ── TITLE ─────────────────────────────────────────────────────────────
    ax.text(10.5, 8.72,
            'EfficientNetB4 + FSDA Architecture for Garlic Disease Classification',
            ha='center', va='center', fontsize=13, fontweight='bold',
            color=C['text'],
            bbox=dict(boxstyle='round,pad=0.35', fc='#F5F5F5',
                      ec='#BBBBBB', lw=1.0))

    # ══════════════════════════════════════════════════════════════════════
    # 1. INPUT IMAGE — overlapping R/G/B channel blocks
    # ══════════════════════════════════════════════════════════════════════
    for i, col in enumerate([C['in_r'], C['in_g'], C['in_b']]):
        bx = 0.12 + i * 0.22
        by = CY - 1.60 - i * 0.12
        block3d(ax, bx, by, w=0.55, h=3.20, dx=0.16, dy=0.10,
                fc=col, lw=0.7, nh=5, nv=3, zorder=3 + i)

    dim_label(ax, 0.80, CY + 1.72, '380 x 380 x 3', fs=7.0)
    name_label(ax, 0.62, CY - 1.62, 'Input Image', fs=8.5)

    arrow(ax, 1.10, CY, 1.44, CY)

    # ══════════════════════════════════════════════════════════════════════
    # 2. EFFICIENTNETB4 BACKBONE
    # ══════════════════════════════════════════════════════════════════════
    section(ax, 1.40, 0.44, 6.10, 7.52,
            title='EfficientNetB4 Backbone  (ImageNet pretrained)',
            tfc='#EAF2FB', ec=C['backbone'], lw=2.0,
            tcolor='#1F4E79', tfs=9.0)

    blk_specs = [
        (0.58, 5.20, C['frozen'],   'B1'),
        (0.58, 4.70, C['frozen'],   'B2'),
        (0.58, 4.10, C['backbone'], 'B3'),
        (0.58, 3.60, C['backbone'], 'B4'),
        (0.58, 3.10, C['backbone'], 'B5'),
        (0.58, 2.65, C['backbone'], 'B6'),
        (0.58, 2.25, C['backbone'], 'B7'),
    ]
    bx = 1.56
    for bw, bh, bc, bl in blk_specs:
        by = CY - bh / 2
        block3d(ax, bx, by, bw, bh, dx=0.20, dy=0.13,
                fc=bc, lw=0.85, nh=8, nv=3, zorder=4)
        ax.text(bx + bw/2 + 0.10, CY, bl,
                ha='center', va='center', fontsize=7.5,
                color='white', fontweight='bold', rotation=90, zorder=9)
        bx += bw + 0.18

    rbox(ax, 1.54, 0.60, 1.18, 0.40, text='B1-B2  Frozen',
         sub='weights fixed', fc=C['frozen'], tc='#1A1A1A', fs=7.5, radius=0.08)
    rbox(ax, 2.84, 0.60, 1.40, 0.40, text='B3-B7  Fine-tuned',
         sub='BN always frozen', fc=C['backbone'], tc='white', fs=7.5, radius=0.08)

    arrow(ax, 7.52, CY, 7.90, CY)

    # ══════════════════════════════════════════════════════════════════════
    # 3. FEATURE MAP  12 x 12 x 1792
    # ══════════════════════════════════════════════════════════════════════
    for i in range(5):
        block3d(ax, 7.90 + i*0.16, CY - 1.35 - i*0.10,
                w=0.50, h=2.70, dx=0.16, dy=0.10,
                fc=C['feat'], lw=0.75, nh=7, nv=3, zorder=4 + i)

    dim_label(ax, 8.44, CY + 1.48, '12 x 12 x 1792', fs=7.0)
    name_label(ax, 8.22, CY - 1.48, 'Feature Map', fs=8.5)

    arrow(ax, 8.90, CY, 9.28, CY)

    # ══════════════════════════════════════════════════════════════════════
    # 4. FSDA BLOCK
    # ══════════════════════════════════════════════════════════════════════
    section(ax, 9.24, 0.44, 4.60, 7.52,
            title='FSDA Block  (Proposed)',
            tfc='#FEF3E8', ec=C['fsda'], lw=2.2,
            tcolor='#7B3200', tfs=9.5)

    # Large central FSDA 3D block
    block3d(ax, 9.44, CY - 1.90, w=1.60, h=3.80,
            dx=0.30, dy=0.20, fc=C['fsda'], lw=1.0, nh=8, nv=4, zorder=5)
    dim_label(ax, 9.79, CY + 2.10, '12 x 12 x 1792', fs=6.8)

    # Freq branch badge
    rbox(ax, 9.42, CY + 0.28, 1.62, 0.82,
         text='Freq Channel\nAttention',
         sub='FFT2D > log1p > FC(C/16)\n> FC(C) > x channel attn',
         fc=C['freq'], tc='white', fs=7.5, radius=0.10, zorder=8)

    # Spatial branch badge
    rbox(ax, 9.42, CY - 1.22, 1.62, 0.82,
         text='Spatial Attention',
         sub='AvgPool+MaxPool > Concat\n> Conv7x7 > Sigmoid > x spatial',
         fc=C['spatial'], tc='white', fs=7.5, radius=0.10, zorder=8)

    # Fusion badge
    rbox(ax, 9.42, CY - 0.22, 1.62, 0.44,
         text='Element-wise Add + BN (float32)',
         fc=C['fusion'], tc='white', fs=7.2, radius=0.08, zorder=8)

    # Right detail column
    detail = [
        ('Transpose + FFT2D',     '(B,H,W,C)->(B,C,H,W)->complex64',  C['freq']),
        ('log1p(|FFT|)',           '-> (B,C,H,W) float32',              C['freq']),
        ('Global Avg Pool',        'mean(H,W) -> (B,1792)',              C['freq']),
        ('FC1+ReLU  1792->112',   'no bias, float32',                   C['freq']),
        ('FC2+Sigmoid 112->1792', 'no bias, float32',                   C['freq']),
        ('Channel Reweight',       'x * attn -> (B,12,12,1792)',         lighten(C['freq'], 0.10)),
        ('AvgPool + MaxPool',      'reduce mean/max(C) -> (B,12,12,1)', C['spatial']),
        ('Concat -> Conv7x7',      'same pad, no bias -> (B,12,12,1)',   C['spatial']),
        ('Sigmoid Spatial Map',    '-> (B,12,12,1) float32',             C['spatial']),
        ('Spatial Reweight',       'x * sp_attn -> (B,12,12,1792)',      lighten(C['spatial'], 0.10)),
    ]
    dy_step = 0.70
    iy = 7.34
    for txt, sub, col in detail:
        rbox(ax, 11.22, iy - 0.28, 2.48, 0.52,
             text=txt, sub=sub, fc=col, tc='white', fs=6.0, radius=0.07, zorder=6)
        iy -= dy_step

    arrow(ax, 13.86, CY, 14.22, CY)

    # ══════════════════════════════════════════════════════════════════════
    # 5. ENHANCED FEATURE MAP
    # ══════════════════════════════════════════════════════════════════════
    for i in range(5):
        block3d(ax, 14.22 + i*0.16, CY - 1.35 - i*0.10,
                w=0.50, h=2.70, dx=0.16, dy=0.10,
                fc=C['fsda'], lw=0.75, nh=7, nv=3, zorder=4 + i)

    dim_label(ax, 14.76, CY + 1.48, '12 x 12 x 1792', fs=7.0)
    name_label(ax, 14.54, CY - 1.48, 'Enhanced\nFeature Map', fs=8.5)

    arrow(ax, 15.22, CY, 15.58, CY)

    # ══════════════════════════════════════════════════════════════════════
    # 6. CLASSIFICATION HEAD
    # ══════════════════════════════════════════════════════════════════════
    section(ax, 15.54, 0.44, 3.50, 7.52,
            title='Classification Head',
            tfc='#F0FFF0', ec=C['fc'], lw=2.0,
            tcolor='#1B5E20', tfs=9.0)

    head_items = [
        ('Global Avg Pool', '(B, 1792)',   C['gap'],     '->(B,1792)'),
        ('Batch Norm',       'float32',     C['bn'],      '->(B,1792)'),
        ('FC 256 + ReLU',   '+ L2(1e-5)',  C['fc'],      '->(B, 256)'),
        ('Dropout  0.5',    '',             C['drop'],    '->(B, 256)'),
        ('Softmax',          '3 classes',   C['softmax'], '->(B,   3)'),
    ]
    hy = 7.14
    hx, hw = 15.70, 3.14
    for name, sub, hc, dim in head_items:
        rbox(ax, hx, hy - 0.30, hw, 0.58,
             text=name, sub=sub if sub else None,
             fc=hc, tc='white', fs=8.0, radius=0.10, zorder=5)
        ax.text(hx + hw + 0.08, hy, dim,
                ha='left', va='center', fontsize=6.5,
                color='#444', style='italic')
        if hy > 1.72:
            arrow(ax, hx + hw/2, hy - 0.30, hx + hw/2, hy - 0.30 - 0.28,
                  color='#555', lw=1.2)
        hy -= 1.18

    rbox(ax, hx + 0.10, 0.56, hw - 0.20, 0.62,
         text='[Novel] Adaptive CB Focal Loss',
         sub='gamma=2.0  beta=0.9999  tau=0.3 (EMA)',
         fc='#BF3B0A', tc='white', fs=7.5, radius=0.10, zorder=6)

    arrow(ax, 19.08, CY, 19.44, CY)

    # ══════════════════════════════════════════════════════════════════════
    # 7. OUTPUT — 3 class bars
    # ══════════════════════════════════════════════════════════════════════
    section(ax, 19.40, 0.44, 1.50, 7.52,
            title='Output',
            tfc='#FFF0F0', ec=C['softmax'], lw=2.0,
            tcolor='#7B0000', tfs=9.0)

    cls_data = [
        ('Fully\nPeeled',     0.05, '#AAAAAA'),
        ('Partially\nPeeled', 0.07, '#AAAAAA'),
        ('Spoiled',           0.88, C['softmax']),
    ]
    bar_x0 = 19.52
    bar_max = 1.24
    for i, (cname, prob, bc) in enumerate(cls_data):
        by = 1.55 + i * 1.72
        bw = prob * bar_max
        ax.add_patch(plt.Rectangle((bar_x0, by), bw, 0.80,
                                   fc=bc, ec='#666', lw=0.8, zorder=5))
        ax.text(bar_x0 - 0.07, by + 0.40, cname,
                ha='right', va='center', fontsize=7.0,
                color=C['text'], linespacing=1.2)
        ax.text(bar_x0 + bw + 0.06, by + 0.40, f'{prob:.0%}',
                ha='left', va='center', fontsize=7.5,
                color='#CC0000' if prob > 0.5 else '#555',
                fontweight='bold' if prob > 0.5 else 'normal')

    # ── FORMULA ───────────────────────────────────────────────────────────
    ax.text(10.5, 0.20,
            r'$\mathrm{FSDA}(x)=\mathrm{BN}'
            r'\left(\mathrm{FreqAttn}(x)+\mathrm{SpatAttn}(x)\right)$',
            ha='center', va='center', fontsize=9.0, color='#3A2060',
            style='italic',
            bbox=dict(boxstyle='round,pad=0.30', fc='#F8F0FF',
                      ec='#B090E0', lw=0.9))

    # ── LEGEND ────────────────────────────────────────────────────────────
    legend_items = [
        (C['frozen'],   'Frozen Blocks (B1-B2)'),
        (C['backbone'], 'Fine-tuned Blocks (B3-B7)'),
        (C['freq'],     'Freq Channel Attention'),
        (C['spatial'],  'Spatial Attention'),
        (C['fusion'],   'Fusion (Add+BN)'),
        (C['softmax'],  'Softmax / Output'),
    ]
    lx = 0.20
    for fc_l, lbl in legend_items:
        ax.add_patch(plt.Rectangle((lx, -0.04), 0.36, 0.22,
                                   fc=fc_l, ec='#555', lw=0.7, zorder=5))
        ax.text(lx + 0.44, 0.07, lbl,
                ha='left', va='center', fontsize=7.5, color=C['text'])
        lx += len(lbl) * 0.108 + 0.58

    plt.tight_layout(pad=0.3)
    for fmt in ('pdf', 'png'):
        plt.savefig(f'figure1_overall_architecture.{fmt}',
                    dpi=300, bbox_inches='tight', facecolor='white')
    print('Saved: figure1_overall_architecture.pdf / .png')
    plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — FSDA BLOCK INTERNAL  (14 x 18 in)
# ─────────────────────────────────────────────────────────────────────────────
def draw_figure2():
    fig, ax = plt.subplots(figsize=(14, 18))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 18)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # ── TITLE ─────────────────────────────────────────────────────────────
    ax.text(7.0, 17.60,
            'FSDA Block: Frequency-Spatial Dual Attention  (Proposed)',
            ha='center', va='center', fontsize=13, fontweight='bold',
            color=C['text'],
            bbox=dict(boxstyle='round,pad=0.35', fc='#FFF3E0',
                      ec='#E07B00', lw=1.2))

    # ══════════════════════════════════════════════════════════════════════
    # INPUT FEATURE MAP
    # ══════════════════════════════════════════════════════════════════════
    for i in range(7):
        block3d(ax, 3.80 + i*0.24, 15.50 - i*0.15,
                w=1.20, h=1.00, dx=0.26, dy=0.17,
                fc=C['feat'], lw=0.85, nh=4, nv=3, zorder=3 + i)

    dim_label(ax, 5.40, 16.72, '12 x 12 x 1792  (from EfficientNetB4)', fs=7.8)
    ax.text(7.0, 17.04, 'Input Feature Map',
            ha='center', va='top', fontsize=11, fontweight='bold', color=C['text'])
    ax.text(7.0, 16.76,
            '(B, 12, 12, 1792)  —  float16 under mixed precision',
            ha='center', va='top', fontsize=7.8, color='#555', style='italic')

    # Split dot + arrow
    arrow(ax, 7.0, 16.24, 7.0, 15.28, color='#333', lw=2.0)
    ax.plot(7.0, 15.22, 'o', color='#333', ms=10, zorder=20)

    # Branch arrows
    ax.annotate('', xy=(3.10, 14.30), xytext=(7.0, 15.22),
                arrowprops=dict(arrowstyle='->', color=C['freq'], lw=1.8,
                                connectionstyle='arc3,rad=0.0'))
    ax.text(4.55, 14.94, 'Branch 1', ha='center', va='center',
            fontsize=9, color=C['freq'], fontweight='bold')

    ax.annotate('', xy=(10.90, 14.30), xytext=(7.0, 15.22),
                arrowprops=dict(arrowstyle='->', color=C['spatial'], lw=1.8,
                                connectionstyle='arc3,rad=0.0'))
    ax.text(9.45, 14.94, 'Branch 2', ha='center', va='center',
            fontsize=9, color=C['spatial'], fontweight='bold')

    # ══════════════════════════════════════════════════════════════════════
    # LEFT BRANCH: FREQUENCY CHANNEL ATTENTION
    # ══════════════════════════════════════════════════════════════════════
    section(ax, 0.30, 2.00, 6.00, 12.50,
            title='Branch 1 — Frequency Channel Attention',
            tfc='#FEF3E8', ec=C['freq'], lw=1.8, ls='-',
            tcolor='#7B3200', tfs=9.5)

    freq_steps = [
        ('Transpose',
         '(B,H,W,C) -> (B,C,H,W)',
         '#FADADB', '(B, 1792, 12, 12)'),
        ('2D FFT  (tf.signal.fft2d)',
         'real -> complex64 spectrum',
         '#FAC8A0', 'complex64'),
        ('Log-Magnitude',
         'log1p( |FFT| )  ->  float32',
         '#F5A870', '(B, 1792, 12, 12)'),
        ('Global Avg Pool',
         'mean over (H, W) axes',
         '#F08840', '(B, 1792)'),
        ('FC1 + ReLU',
         '1792 -> 112   no bias, float32',
         '#E86818', '(B, 112)'),
        ('FC2 + Sigmoid',
         '112 -> 1792   no bias, float32',
         '#C85000', '(B, 1792)'),
        ('Reshape',
         '(B, 1792) -> (B, 1, 1, 1792)',
         '#A04000', '(B, 1, 1, 1792)'),
        ('Channel Reweighting',
         'x_freq  =  x  *  attn   (broadcast)',
         '#7B3000', '(B, 12, 12, 1792)'),
    ]

    fy = 13.56
    for i, (ttl, sub, face, dim_out) in enumerate(freq_steps):
        tc_txt = C['text'] if i < 3 else C['white']
        rbox(ax, 0.52, fy - 0.42, 5.56, 0.84,
             text=ttl, sub=sub, fc=face, tc=tc_txt, fs=8.5, radius=0.10, zorder=5)
        ax.text(6.14, fy, dim_out,
                ha='left', va='center', fontsize=7.0,
                color='#444', style='italic')
        if i < len(freq_steps) - 1:
            arrow(ax, 0.52 + 5.56/2, fy - 0.42,
                  0.52 + 5.56/2, fy - 0.42 - 0.30,
                  color=C['freq'], lw=1.3)
        fy -= 1.42

    # ══════════════════════════════════════════════════════════════════════
    # RIGHT BRANCH: SPATIAL ATTENTION
    # ══════════════════════════════════════════════════════════════════════
    section(ax, 7.70, 2.00, 6.00, 12.50,
            title='Branch 2 — Spatial Attention  (CBAM-style)',
            tfc='#F0FFF0', ec=C['spatial'], lw=1.8, ls='-',
            tcolor='#1B5E20', tfs=9.5)

    sp_steps = [
        ('AvgPool  (channel dim)',
         'reduce_mean(C) -> (B,12,12,1)  float32',
         '#C8EACC', '(B, 12, 12, 1)'),
        ('MaxPool  (channel dim)',
         'reduce_max(C)  -> (B,12,12,1)  float32',
         '#A8D8AC', '(B, 12, 12, 1)'),
        ('Concatenate',
         '[AvgPool || MaxPool]  channel concat',
         '#80C488', '(B, 12, 12, 2)'),
        ('Conv 7x7  (1 filter)',
         'padding=same, no bias  ->  float32',
         '#58B062', '(B, 12, 12, 1)'),
        ('Sigmoid',
         'spatial attention map',
         '#3A9848', '(B, 12, 12, 1)'),
        ('Spatial Attention Map',
         'shape: 12 x 12 x 1  (float32)',
         '#267034', 'attn map'),
        ('Spatial Reweighting',
         'x_spat  =  x  *  sp_attn  (broadcast)',
         '#155020', '(B, 12, 12, 1792)'),
    ]

    sy = 13.56
    for i, (ttl, sub, face, dim_out) in enumerate(sp_steps):
        tc_txt = C['text'] if i < 4 else C['white']
        if i == 0:
            # AvgPool: left half
            rbox(ax, 7.92, sy - 0.42, 2.62, 0.84,
                 text=ttl, sub=sub[:28], fc=face, tc=tc_txt, fs=8.0, radius=0.10, zorder=5)
            ax.text(13.70, sy, dim_out,
                    ha='left', va='center', fontsize=7.0,
                    color='#444', style='italic')
            sy -= 1.42
            continue
        if i == 1:
            # MaxPool: right half (same row)
            rbox(ax, 10.76, sy - 0.42 + 1.42, 2.62, 0.84,
                 text=ttl, sub=sub[:28], fc=face, tc=tc_txt, fs=8.0, radius=0.10, zorder=5)
            # Converge arrows -> Concatenate
            arrow(ax, 7.92 + 1.31, sy - 0.42 + 1.42,
                  7.92 + 5.56/2, sy + 0.42,
                  color=C['spatial'], lw=1.1)
            arrow(ax, 10.76 + 1.31, sy - 0.42 + 1.42,
                  7.92 + 5.56/2, sy + 0.42,
                  color=C['spatial'], lw=1.1)
            continue

        rbox(ax, 7.92, sy - 0.42, 5.56, 0.84,
             text=ttl, sub=sub, fc=face, tc=tc_txt, fs=8.5, radius=0.10, zorder=5)
        ax.text(13.54, sy, dim_out,
                ha='left', va='center', fontsize=7.0,
                color='#444', style='italic')
        if i < len(sp_steps) - 1:
            arrow(ax, 7.92 + 5.56/2, sy - 0.42,
                  7.92 + 5.56/2, sy - 0.42 - 0.30,
                  color=C['spatial'], lw=1.3)
        sy -= 1.42

    # ══════════════════════════════════════════════════════════════════════
    # FUSION — Element-wise Add + Batch Normalisation
    # ══════════════════════════════════════════════════════════════════════
    section(ax, 2.60, 0.22, 8.80, 1.76,
            tfc='#EDE7F6', ec=C['fusion'], lw=1.6, ls='-', radius=0.16)

    # Arrows from bottom of each branch to fusion area
    arrow(ax, 0.52 + 5.56/2, fy + 1.42 - 0.42,
          4.80, 1.98, color=C['freq'], lw=1.6)
    arrow(ax, 7.92 + 5.56/2, sy + 1.42 - 0.42,
          9.20, 1.98, color=C['spatial'], lw=1.6)

    # Add circle
    op_circle(ax, 5.60, 1.10, r=0.34, symbol='+', fc=C['fusion'], fs=16, zorder=10)
    ax.text(5.60, 0.60, 'Element-wise Add', ha='center', va='top',
            fontsize=8.0, color=C['fusion'], fontweight='bold')

    arrow(ax, 5.96, 1.10, 6.80, 1.10, color=C['fusion'], lw=1.5)

    rbox(ax, 6.84, 0.70, 2.10, 0.82,
         text='Batch Norm', sub='float32 output',
         fc=C['fusion'], tc='white', fs=9.0, radius=0.12, zorder=8)

    arrow(ax, 8.96, 1.10, 9.68, 1.10, color=C['fusion'], lw=1.5)
    ax.text(9.72, 1.30, 'cast back\nto input dtype', ha='left', va='center',
            fontsize=7.2, color='#555', style='italic')

    # Arrows into the Add circle
    ax.annotate('', xy=(5.26, 1.10), xytext=(3.30, 1.10),
                arrowprops=dict(arrowstyle='->', color=C['freq'], lw=1.4,
                                connectionstyle='arc3,rad=0.0'))
    ax.annotate('', xy=(8.92, 1.10), xytext=(10.56, 1.10),
                arrowprops=dict(arrowstyle='<-', color=C['spatial'], lw=1.4,
                                connectionstyle='arc3,rad=0.0'))

    # ── FORMULA ───────────────────────────────────────────────────────────
    ax.text(7.0, 0.10,
            r'$\mathrm{FSDA}(x)=\mathrm{BN}'
            r'\left(\mathrm{FreqAttn}(x)+\mathrm{SpatAttn}(x)\right)$',
            ha='center', va='bottom', fontsize=10.5, color='#3A2060',
            style='italic',
            bbox=dict(boxstyle='round,pad=0.32', fc='#F8F0FF',
                      ec='#C0A0F0', lw=1.0))

    # ── OUTPUT pipeline note ──────────────────────────────────────────────
    ax.text(7.0, -0.06,
            'Output (B,12,12,1792) -> GAP->(B,1792) -> BN -> '
            'FC256+ReLU->(B,256) -> Dropout(0.5) -> Softmax->(B,3)',
            ha='center', va='top', fontsize=7.2, color='#555', style='italic')

    plt.tight_layout(pad=0.4)
    for fmt in ('pdf', 'png'):
        plt.savefig(f'figure2_fsda_block.{fmt}',
                    dpi=300, bbox_inches='tight', facecolor='white')
    print('Saved: figure2_fsda_block.pdf / .png')
    plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    draw_figure1()
    draw_figure2()
    print('\nDone. Four files saved:')
    print('  figure1_overall_architecture.pdf / .png')
    print('  figure2_fsda_block.pdf / .png')
