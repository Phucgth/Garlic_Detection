"""
EfficientNetB4 + FSDA Architecture Diagram
Style: 3D-perspective blocks (CNN paper style)
Output: architecture_EfficientNetB4_FSDA.png (high-res, suitable for paper)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as pe
import numpy as np

# ─────────────────────────────────────────────
# Helper: draw a 3D-style "feature map" block
# ─────────────────────────────────────────────
def draw_3d_block(ax, x, y, w, h, d,
                  face_color, edge_color='#333333',
                  alpha=1.0, lw=1.2, top_color=None, side_color=None):
    """
    Draw a 3D rectangular block at (x, y).
    w=width, h=height, d=depth offset (for 3D illusion).
    """
    if top_color is None:
        top_color = _lighten(face_color, 0.35)
    if side_color is None:
        side_color = _darken(face_color, 0.25)

    # Front face
    front = plt.Polygon(
        [[x, y], [x+w, y], [x+w, y+h], [x, y+h]],
        closed=True, facecolor=face_color, edgecolor=edge_color,
        linewidth=lw, alpha=alpha, zorder=3)
    ax.add_patch(front)

    # Top face
    top = plt.Polygon(
        [[x, y+h], [x+w, y+h],
         [x+w+d, y+h+d*0.6], [x+d, y+h+d*0.6]],
        closed=True, facecolor=top_color, edgecolor=edge_color,
        linewidth=lw, alpha=alpha, zorder=3)
    ax.add_patch(top)

    # Right face (side)
    side = plt.Polygon(
        [[x+w, y], [x+w+d, y+d*0.6],
         [x+w+d, y+h+d*0.6], [x+w, y+h]],
        closed=True, facecolor=side_color, edgecolor=edge_color,
        linewidth=lw, alpha=alpha, zorder=3)
    ax.add_patch(side)


def _lighten(hex_color, factor=0.3):
    rgb = _hex2rgb(hex_color)
    lightened = [min(1.0, c + factor) for c in rgb]
    return _rgb2hex(lightened)

def _darken(hex_color, factor=0.2):
    rgb = _hex2rgb(hex_color)
    darkened = [max(0.0, c - factor) for c in rgb]
    return _rgb2hex(darkened)

def _hex2rgb(h):
    h = h.lstrip('#')
    return [int(h[i:i+2], 16)/255.0 for i in (0, 2, 4)]

def _rgb2hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(
        int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))


def draw_arrow(ax, x1, y1, x2, y2, color='#444444', lw=1.8, style='->'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle='arc3,rad=0.0'),
                zorder=10)


def draw_label(ax, x, y, text, fontsize=8, color='#222222',
               ha='center', va='top', bold=False, rotation=0):
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, fontsize=fontsize, color=color,
            ha=ha, va=va, fontweight=weight,
            rotation=rotation, zorder=15)


def draw_dim_label(ax, x, y, text, fontsize=6.5, color='#555555'):
    ax.text(x, y, text, fontsize=fontsize, color=color,
            ha='center', va='center', style='italic', zorder=15)


def draw_rounded_box(ax, x, y, w, h, text, facecolor, edgecolor,
                     fontsize=8, text_color='white', bold=True, radius=0.15):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0.05,rounding_size={radius}",
                         facecolor=facecolor, edgecolor=edgecolor,
                         linewidth=1.5, zorder=5)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text,
            ha='center', va='center', fontsize=fontsize,
            color=text_color, fontweight=weight, zorder=6)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DRAWING
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(22, 10))
ax.set_xlim(0, 22)
ax.set_ylim(0, 10)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── TITLE ─────────────────────────────────────────────────────────────────
ax.text(11, 9.65, 'EfficientNetB4 + FSDA Architecture',
        ha='center', va='center', fontsize=17, fontweight='bold', color='#1a1a2e')
ax.text(11, 9.25, 'Frequency-Spatial Dual Attention for Garlic Disease Classification',
        ha='center', va='center', fontsize=10, color='#555555', style='italic')

# ═══════════════════════════════════════════════════════════════════
# SECTION BACKGROUNDS
# ═══════════════════════════════════════════════════════════════════
# Backbone section
ax.add_patch(FancyBboxPatch((0.2, 0.5), 7.2, 8.1,
    boxstyle="round,pad=0.1,rounding_size=0.2",
    facecolor='#f0f4ff', edgecolor='#aab4e0', linewidth=1.5, alpha=0.5, zorder=0))
ax.text(3.8, 8.65, 'EfficientNetB4 Backbone', ha='center', va='center',
        fontsize=9, color='#3d5a99', fontweight='bold')

# FSDA section
ax.add_patch(FancyBboxPatch((7.7, 0.5), 6.6, 8.1,
    boxstyle="round,pad=0.1,rounding_size=0.2",
    facecolor='#fff4e6', edgecolor='#e0aa60', linewidth=1.5, alpha=0.5, zorder=0))
ax.text(11.0, 8.65, 'FSDA Block (Novel)', ha='center', va='center',
        fontsize=9, color='#b85c00', fontweight='bold')

# Head section
ax.add_patch(FancyBboxPatch((14.6, 0.5), 4.3, 8.1,
    boxstyle="round,pad=0.1,rounding_size=0.2",
    facecolor='#f0fff4', edgecolor='#80c980', linewidth=1.5, alpha=0.5, zorder=0))
ax.text(16.75, 8.65, 'Classification Head', ha='center', va='center',
        fontsize=9, color='#1a6b1a', fontweight='bold')

# Output section
ax.add_patch(FancyBboxPatch((19.2, 0.5), 2.5, 8.1,
    boxstyle="round,pad=0.1,rounding_size=0.2",
    facecolor='#fff0f5', edgecolor='#e080a0', linewidth=1.5, alpha=0.5, zorder=0))
ax.text(20.45, 8.65, 'Output', ha='center', va='center',
        fontsize=9, color='#8b0040', fontweight='bold')


# ═══════════════════════════════════════════════════════════════════
# 1. INPUT IMAGE
# ═══════════════════════════════════════════════════════════════════
# Draw a small simulated "image" block
draw_3d_block(ax, x=0.3, y=3.8, w=0.6, h=2.0, d=0.18,
              face_color='#7ecba1', edge_color='#2a7a4b', lw=1.5)
draw_3d_block(ax, x=0.55, y=3.55, w=0.6, h=2.0, d=0.18,
              face_color='#5bb88a', edge_color='#2a7a4b', lw=1.5)
draw_3d_block(ax, x=0.8, y=3.3, w=0.6, h=2.0, d=0.18,
              face_color='#3da575', edge_color='#2a7a4b', lw=1.5)

draw_label(ax, 1.0, 3.1, 'Input Image\n380×380×3',
           fontsize=8, bold=True, color='#1a6b3a')

# ═══════════════════════════════════════════════════════════════════
# 2. EFFICIENTNETB4 BLOCKS — show as stacked blocks (blocks 1-7)
# ═══════════════════════════════════════════════════════════════════
colors_blocks = [
    ('#b0c4de', '#6a8caf'),  # block 1 - frozen
    ('#b0c4de', '#6a8caf'),  # block 2 - frozen
    ('#5b9bd5', '#2e6fa3'),  # block 3 - unfrozen
    ('#4a8bc5', '#1e5f93'),  # block 4 - unfrozen
    ('#3a7bb5', '#0e4f83'),  # block 5 - unfrozen
    ('#2a6ba5', '#003f73'),  # block 6 - unfrozen
    ('#1a5b95', '#002f63'),  # block 7 - unfrozen
]

block_xs = [1.7, 2.35, 3.0, 3.65, 4.3, 4.95, 5.6]
block_hs = [2.8, 2.5, 2.2, 2.0, 1.8, 1.6, 1.5]
block_ws = [0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45]

for i, (bx, bh, bw, (fc, ec)) in enumerate(
        zip(block_xs, block_hs, block_ws, colors_blocks)):
    by = 4.8 - bh/2
    draw_3d_block(ax, x=bx, y=by, w=bw, h=bh, d=0.2,
                  face_color=fc, edge_color=ec, lw=1.3)
    frozen = i < 2
    label_color = '#888888' if frozen else '#ffffff'
    style = 'italic' if frozen else 'normal'
    ax.text(bx + bw/2 + 0.1, by + bh/2 + 0.1,
            f'B{i+1}', fontsize=6.5, ha='center', va='center',
            color=label_color, fontweight='bold', style=style, zorder=10)

# Frozen / Unfrozen legend inside backbone
draw_rounded_box(ax, 1.65, 1.1, 1.1, 0.38, '[X] Frozen',
                 '#b0c4de', '#6a8caf', fontsize=7, text_color='#2a2a4a')
draw_rounded_box(ax, 2.85, 1.1, 1.3, 0.38, '[*] Fine-tuned',
                 '#2a6ba5', '#003f73', fontsize=7, text_color='white')

draw_label(ax, 4.1, 8.3, 'Blocks 1-2: Frozen\nBlocks 3-7: Fine-tuned\n(BN always frozen)',
           fontsize=7, color='#3d5a99')

# Arrow: input → backbone
draw_arrow(ax, 1.45, 4.8, 1.65, 4.8, color='#2a7a4b', lw=2.0)

# ═══════════════════════════════════════════════════════════════════
# 3. FEATURE MAP OUTPUT (12×12×1792)
# ═══════════════════════════════════════════════════════════════════
fm_x = 6.3
for i in range(4):
    draw_3d_block(ax, x=fm_x + i*0.18, y=3.7 - i*0.12,
                  w=0.55, h=2.2, d=0.15,
                  face_color='#1a5b95', edge_color='#001f5e', lw=1.2)

draw_label(ax, 6.9, 3.3, 'Feature Map\n12×12×1792',
           fontsize=7.5, bold=True, color='#001f5e')

# Arrow: backbone → feature map
draw_arrow(ax, 6.2, 4.8, 6.3, 4.8, color='#2e6fa3', lw=2.0)

# ═══════════════════════════════════════════════════════════════════
# 4. FSDA BLOCK
# ═══════════════════════════════════════════════════════════════════

# ── Arrow: feature map → FSDA split ──────────────────────────────
draw_arrow(ax, 7.0, 4.8, 7.8, 4.8, color='#b85c00', lw=2.0)

# Split point
ax.plot(8.1, 4.8, 'o', color='#b85c00', ms=7, zorder=15)

# Branch line to top (Freq)
ax.annotate('', xy=(8.5, 7.0), xytext=(8.1, 4.8),
            arrowprops=dict(arrowstyle='->', color='#b85c00', lw=1.8,
                            connectionstyle='arc3,rad=-0.2'), zorder=10)
# Branch line to bottom (Spatial)
ax.annotate('', xy=(8.5, 2.6), xytext=(8.1, 4.8),
            arrowprops=dict(arrowstyle='->', color='#b85c00', lw=1.8,
                            connectionstyle='arc3,rad=0.2'), zorder=10)

# ─── FREQUENCY BRANCH (top) ─────────────────────────────────────
freq_y_top = 6.2
draw_rounded_box(ax, 8.5, freq_y_top + 0.5, 1.55, 0.5,
                 'FFT2D\n(complex)', '#e07b39', '#a04000',
                 fontsize=7.5, text_color='white')
draw_arrow(ax, 9.27, freq_y_top + 0.5, 9.27, freq_y_top + 0.2, color='#a04000')
draw_rounded_box(ax, 8.5, freq_y_top - 0.25, 1.55, 0.5,
                 'log1p(|FFT|)\nmean(H,W)→(B,C)', '#e07b39', '#a04000',
                 fontsize=7, text_color='white')
draw_arrow(ax, 9.27, freq_y_top - 0.25, 9.27, freq_y_top - 0.55, color='#a04000')
draw_rounded_box(ax, 8.5, freq_y_top - 1.1, 1.55, 0.5,
                 'FC(C→C/16, ReLU)\nFC(C/16→C, Sigmoid)', '#e07b39', '#a04000',
                 fontsize=6.5, text_color='white')
draw_arrow(ax, 9.27, freq_y_top - 1.1, 9.27, freq_y_top - 1.4, color='#a04000')
draw_rounded_box(ax, 8.5, freq_y_top - 1.9, 1.55, 0.42,
                 'Channel Reweighting\nx ← x × attn(1,1,C)', '#c0551a', '#802000',
                 fontsize=6.5, text_color='white')

ax.text(9.27, 7.85, '① Frequency Channel\nAttention',
        ha='center', va='center', fontsize=7.5, color='#a04000',
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff0e0',
                  edgecolor='#e07b39', linewidth=1))

# ─── SPATIAL BRANCH (bottom) ────────────────────────────────────
sp_y_bot = 3.0
draw_rounded_box(ax, 8.5, sp_y_bot - 0.05, 1.55, 0.5,
                 'AvgPool + MaxPool\n→ (B,H,W,2)', '#5b9de0', '#2060a0',
                 fontsize=7, text_color='white')
draw_arrow(ax, 9.27, sp_y_bot - 0.05, 9.27, sp_y_bot - 0.35, color='#2060a0')
draw_rounded_box(ax, 8.5, sp_y_bot - 0.9, 1.55, 0.5,
                 'Conv2D(1, 7×7, same)\nSigmoid', '#5b9de0', '#2060a0',
                 fontsize=7, text_color='white')
draw_arrow(ax, 9.27, sp_y_bot - 0.9, 9.27, sp_y_bot - 1.2, color='#2060a0')
draw_rounded_box(ax, 8.5, sp_y_bot - 1.75, 1.55, 0.42,
                 'Spatial Reweighting\nx ← x × attn(H,W,1)', '#3070b0', '#103080',
                 fontsize=6.5, text_color='white')

ax.text(9.27, 1.6, '② Spatial Attention\n(CBAM-style)',
        ha='center', va='center', fontsize=7.5, color='#2060a0',
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f0ff',
                  edgecolor='#5b9de0', linewidth=1))

# ─── FUSION ─────────────────────────────────────────────────────
# Arrows from both branches to fusion point
draw_arrow(ax, 10.07, freq_y_top - 1.9, 10.7, 4.9, color='#a04000', lw=1.8)
draw_arrow(ax, 10.07, sp_y_bot - 1.33, 10.7, 4.7, color='#2060a0', lw=1.8)

# Fusion box
draw_rounded_box(ax, 10.7, 4.3, 1.8, 0.9,
                 '⊕  Element-wise Add\n+ BatchNorm (float32)',
                 '#8b5cf6', '#5b2cc0', fontsize=8, text_color='white')

ax.text(11.6, 5.5, 'FSDA Fusion\nFreqAttn(x) + SpatialAttn(x)',
        ha='center', va='center', fontsize=7.5, color='#5b2cc0',
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0e8ff',
                  edgecolor='#8b5cf6', linewidth=1))

# Formula
ax.text(11.6, 3.85, r'$\mathbf{FSDA}(x) = \mathrm{BN}(\mathrm{FreqAttn}(x) + \mathrm{SpatialAttn}(x))$',
        ha='center', va='center', fontsize=7.5, color='#3a006f',
        style='italic',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#f8f0ff',
                  edgecolor='#c0a0f0', linewidth=1))

# Arrow: fusion → head
draw_arrow(ax, 12.5, 4.75, 14.7, 4.75, color='#1a6b1a', lw=2.0)

# ═══════════════════════════════════════════════════════════════════
# 5. CLASSIFICATION HEAD
# ═══════════════════════════════════════════════════════════════════
head_x = 14.7
head_items = [
    (4.65, 'GlobalAveragePooling2D\n→ (B, 1792)', '#3aaa5f', '#1a7a3f'),
    (3.85, 'BatchNormalization', '#3aaa5f', '#1a7a3f'),
    (3.05, 'Dense(256, ReLU)\n+ L2(1e-5)', '#2a9a4f', '#0a6a2f'),
    (2.25, 'Dropout(0.5)', '#2a9a4f', '#0a6a2f'),
    (1.45, 'Dense(N_classes)\nSoftmax [float32]', '#1a8a3f', '#005a1f'),
]

prev_y = 4.75
for (hy, label, fc, ec) in head_items:
    draw_rounded_box(ax, head_x, hy - 0.28, 2.2, 0.48,
                     label, fc, ec, fontsize=7, text_color='white')
    draw_arrow(ax, head_x + 1.1, prev_y if prev_y != 4.75 else hy + 0.2,
               head_x + 1.1, hy + 0.2, color='#1a6b1a', lw=1.6)
    prev_y = hy - 0.28

# ═══════════════════════════════════════════════════════════════════
# 6. OUTPUT PROBABILITIES
# ═══════════════════════════════════════════════════════════════════
draw_arrow(ax, 16.9, 1.17, 19.3, 4.0, color='#8b0040', lw=2.0)

# Sample output bar chart style
classes = ['Healthy', 'Rust', 'Blight', 'Mold', 'Rot']
probs   = [0.04, 0.87, 0.05, 0.02, 0.02]
colors_bar = ['#cccccc', '#e05050', '#cccccc', '#cccccc', '#cccccc']

bar_x_start = 19.3
bar_y_start = 1.8
bar_h = 0.55
bar_max_w = 1.9

for i, (cls, p, bc) in enumerate(zip(classes, probs, colors_bar)):
    bw = p * bar_max_w
    by = bar_y_start + i * (bar_h + 0.15)
    rect = plt.Rectangle((bar_x_start, by), bw, bar_h,
                          facecolor=bc, edgecolor='#888888',
                          linewidth=0.8, zorder=5)
    ax.add_patch(rect)
    ax.text(bar_x_start - 0.05, by + bar_h/2, cls,
            ha='right', va='center', fontsize=6.5, color='#222222')
    ax.text(bar_x_start + bw + 0.05, by + bar_h/2, f'{p:.0%}',
            ha='left', va='center', fontsize=6.5,
            color='#c00000' if p > 0.5 else '#555555',
            fontweight='bold' if p > 0.5 else 'normal')

ax.text(20.45, 7.3, 'RUST\n(87%)',
        ha='center', va='center', fontsize=10, color='#c00000',
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffe8e8',
                  edgecolor='#e05050', linewidth=2))

draw_label(ax, 20.45, 1.5, 'Class Probabilities\n(Softmax Output)',
           fontsize=7.5, bold=True, color='#8b0040')

# ═══════════════════════════════════════════════════════════════════
# 7. BOTTOM SECTION LABELS
# ═══════════════════════════════════════════════════════════════════
sections = [
    (3.8,  'Feature Extraction\n(Pretrained + Fine-tuning)'),
    (11.0, 'Dual Attention Mechanism\n(Frequency + Spatial)'),
    (16.75,'Deep Classification Head'),
    (20.45,'Probabilistic Output'),
]
for sx, slabel in sections:
    ax.text(sx, 0.65, slabel, ha='center', va='center',
            fontsize=7.5, color='#444444', style='italic',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#f5f5f5',
                      edgecolor='#cccccc', linewidth=0.8))

# ─── Mixed precision note ───────────────────────────────────────
ax.text(11.0, 0.18,
        '[Mixed Precision]  backbone & FSDA compute in float16  |  '
        'BN fusion + head output in float32',
        ha='center', va='center', fontsize=7, color='#666666', style='italic')

# ─── FSDA formula box ────────────────────────────────────────────
ax.add_patch(FancyBboxPatch((7.75, 8.75), 6.5, 0.45,
    boxstyle="round,pad=0.05,rounding_size=0.1",
    facecolor='#fff8ee', edgecolor='#e09030', linewidth=1.2, zorder=4, alpha=0.9))
ax.text(11.0, 8.97,
        r'FreqChannelAttn: FFT2D → log|·| → FC(C/16) → Sigmoid     |     '
        r'SpatialAttn: [AvgPool‖MaxPool] → Conv(7×7) → Sigmoid',
        ha='center', va='center', fontsize=7, color='#7a3a00', zorder=5)

plt.tight_layout(rect=[0, 0.02, 1, 1])
out_path = 'architecture_EfficientNetB4_FSDA.png'
plt.savefig(out_path, dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.show()
print(f"✅ Saved: {out_path}")
