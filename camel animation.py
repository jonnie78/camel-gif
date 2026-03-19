# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 12:59:00 2026

@author: Jonathan
"""

"""
"The Straw That Broke the Camel's Back"
PHY30302 Coursework - Fibre Bundle Model
 
Clean minimal animation:
  - A curved beam representing the camel's back
  - Straws fall one by one and stack on the beam
  - Beam bends progressively downward as load increases
  - At critical load: beam snaps, straws scatter, beam falls
  - Live damage curve on the right
 
Saves to current working directory as: straw_camel.gif
"""
 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from matplotlib.collections import PatchCollection
 
rng = np.random.default_rng(42)
 
# ── FBM Simulation ────────────────────────────────────────────────────────────
N        = 60
N_STEPS  = 200
LOAD_MAX = 2.5
K        = 3.0
 
thresholds  = np.sort(rng.weibull(K, N) * 1.0)
loads       = np.linspace(0, LOAD_MAX, N_STEPS)
alive_hist  = []
stress_hist = []
alive_state = np.ones(N, dtype=bool)
 
for F in loads:
    n = alive_state.sum()
    if n == 0:
        alive_hist.append(alive_state.copy())
        stress_hist.append(0.0)
        continue
    sigma = (F * N) / n
    nf = alive_state & (thresholds < sigma)
    while nf.any():
        alive_state[nf] = False
        n = alive_state.sum()
        sigma = (F * N) / n if n > 0 else 0.0
        nf = alive_state & (thresholds < sigma) if n > 0 else np.zeros(N, bool)
    alive_hist.append(alive_state.copy())
    stress_hist.append(sigma)
 
alive_hist  = np.array(alive_hist)
stress_hist = np.array(stress_hist)
fraction    = alive_hist.sum(axis=1) / N
 
collapse_step = next(
    (i for i in range(1, N_STEPS) if alive_hist[i].sum() == 0), N_STEPS - 1)
print(f"Collapse at step {collapse_step}, F={loads[collapse_step]:.3f}")
 
# ── Layout constants ──────────────────────────────────────────────────────────
BG   = '#0e0e18'
FG   = '#ced6e8'
GREY = '#3a4060'
 
BEAM_X    = np.linspace(0.10, 0.90, 300)
BEAM_MID  = 0.5
BEAM_Y0   = 0.52     # resting beam centre height
BEAM_SAG0 = 0.04     # natural arch upward (camel hump shape)
STRAW_W   = 0.010
STRAW_H   = 0.090
STRAW_EVERY = 4      # frames per straw drop
MAX_STRAWS  = collapse_step // STRAW_EVERY + 3
 
# Pre-assign random x-jitter and angle for each straw
straw_dx  = rng.uniform(-0.12, 0.12, MAX_STRAWS + 4)
straw_ang = rng.uniform(-14, 14, MAX_STRAWS + 4)
# Colour: wheat/straw tones
straw_cols = [plt.cm.YlOrBr(0.25 + 0.55 * i / max(MAX_STRAWS, 1))
              for i in range(MAX_STRAWS + 4)]
 
def beam_y(x, sag, snap=0.0, snap_dir=0.0):
    """
    Beam y profile.
    sag  : 0 = natural hump, 1 = flat, >1 = bowing down
    snap : 0..1 post-collapse droop
    """
    # Normalised position along beam [-1, 1]
    u = (x - BEAM_MID) / (BEAM_MID - BEAM_X[0])
    arch  = BEAM_SAG0 * (1 - u**2)                # natural arch
    bow   = sag * 0.18 * (1 - u**2)               # progressive sag
    droop = snap * 0.30 * np.abs(np.sin(np.pi * (x - BEAM_X[0]) /
                                         (BEAM_X[-1] - BEAM_X[0])))
    return BEAM_Y0 + arch - bow - droop
 
def straw_beam_y(x_centre, sag, snap=0.0):
    """Y position of beam at a given x (for stacking straws)."""
    idx = np.argmin(np.abs(BEAM_X - x_centre))
    return beam_y(BEAM_X, sag, snap)[idx]
 
# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 5.5), facecolor=BG)
gs  = gridspec.GridSpec(1, 2, figure=fig,
                        left=0.04, right=0.97,
                        top=0.90, bottom=0.10,
                        wspace=0.28, width_ratios=[1.7, 1])
 
ax_s = fig.add_subplot(gs[0])   # scene
ax_d = fig.add_subplot(gs[1])   # damage curve
 
# Scene axes
ax_s.set_facecolor(BG)
ax_s.set_xlim(0, 1)
ax_s.set_ylim(0.05, 1.0)
ax_s.set_aspect('equal')
ax_s.axis('off')
 
# Ground line
ax_s.axhline(0.12, color=GREY, lw=1.2, zorder=1)
ax_s.fill_between([0, 1], [0.05, 0.05], [0.12, 0.12],
                  color='#181820', zorder=1)
 
# Support legs (two vertical posts)
for lx in [BEAM_X[0], BEAM_X[-1]]:
    ax_s.plot([lx, lx], [0.12, BEAM_Y0 - 0.06],
              color=GREY, lw=3.5, solid_capstyle='round',
              zorder=3)
 
# Beam (dynamic — redrawn each frame)
beam_line, = ax_s.plot([], [], color='#d4a84b', lw=5,
                        solid_capstyle='round', zorder=5)
 
# Crack/break marker (shown at collapse)
crack_left,  = ax_s.plot([], [], color='#ff4422', lw=3,
                          solid_capstyle='round', zorder=7)
crack_right, = ax_s.plot([], [], color='#ff4422', lw=3,
                          solid_capstyle='round', zorder=7)
 
# Straw containers
resting_straws  = []   # list of Rectangle patches on beam
falling_straw   = [None]
 
# Damage plot
for sp in ax_d.spines.values():
    sp.set_color(GREY)
ax_d.set_facecolor('#0e0e18')
ax_d.tick_params(colors='#6070a0', labelsize=8)
ax_d.xaxis.label.set_color('#8090b0')
ax_d.yaxis.label.set_color('#8090b0')
ax_d.set_xlim(0, LOAD_MAX)
ax_d.set_ylim(-0.05, 1.05)
ax_d.set_xlabel('Applied Load  F', fontsize=9)
ax_d.set_ylabel('Fraction intact', fontsize=9)
ax_d.set_title('Damage curve', color=FG, fontsize=10, pad=6)
ax_d.axvline(loads[collapse_step], color='#ff4422', lw=1.2,
             ls='--', alpha=0.55)
ax_d.text(loads[collapse_step] + 0.04, 0.55, 'collapse',
          color='#ff6644', fontsize=8, va='center', rotation=90)
ax_d.grid(color=GREY, alpha=0.25, lw=0.6)
dmg_line, = ax_d.plot([], [], color='#5aaaf0', lw=2.2, zorder=5)
 
# HUD text
load_txt = ax_s.text(0.5, 0.97, '', ha='center', va='top',
                      color=FG, fontsize=10,
                      transform=ax_s.transAxes)
status_txt = ax_s.text(0.5, 0.90, 'STABLE', ha='center', va='top',
                        color='#44ee88', fontsize=9,
                        transform=ax_s.transAxes)
 
fig.suptitle('"The Straw That Broke the Camel\'s Back"  —  Fibre Bundle Model',
             color=FG, fontsize=11, y=0.98)
 
# ── Helpers ───────────────────────────────────────────────────────────────────
def make_straw_patch(x_centre, y_bottom, angle_deg, color):
    """Return a Rectangle patch for a straw."""
    t = (plt.matplotlib.transforms.Affine2D()
         .rotate_deg(angle_deg)
         .translate(x_centre, y_bottom + STRAW_H / 2)
         + ax_s.transData)
    patch = mpatches.Rectangle(
        (-STRAW_W / 2, -STRAW_H / 2), STRAW_W, STRAW_H,
        facecolor=color, edgecolor='#7a5020',
        lw=0.6, zorder=8,
        transform=t)
    return patch
 
SCATTER_POS = None   # set once at collapse for scattered straws
 
# ── Update function ───────────────────────────────────────────────────────────
def update(frame):
    global resting_straws, SCATTER_POS
 
    step  = min(frame, N_STEPS - 1)
    F     = loads[step]
    n_alv = alive_hist[step].sum()
    past  = max(0, step - collapse_step)
 
    # Sag: increases linearly up to collapse, then beam drops
    sag  = min(1.0, step / max(collapse_step, 1))
    snap = min(1.0, past * 0.07)
 
    # ── Beam ──
    by = beam_y(BEAM_X, sag, snap)
 
    if past == 0:
        # Intact beam
        beam_line.set_data(BEAM_X, by)
        beam_line.set_color('#d4a84b')
        beam_line.set_linewidth(5)
        crack_left.set_data([], [])
        crack_right.set_data([], [])
    elif past <= 2:
        # Crack flash
        mid = len(BEAM_X) // 2
        beam_line.set_data([], [])
        crack_left.set_data(BEAM_X[:mid], by[:mid])
        crack_right.set_data(BEAM_X[mid:], by[mid:])
        crack_left.set_linewidth(4)
        crack_right.set_linewidth(4)
    else:
        # Two halves drooping after snap
        mid = len(BEAM_X) // 2
        beam_line.set_data([], [])
        left_droop  = by[:mid]  - snap * 0.20 * np.linspace(0, 1, mid)**2
        right_droop = by[mid:]  - snap * 0.20 * np.linspace(1, 0, len(BEAM_X) - mid)**2
        crack_left.set_data(BEAM_X[:mid], left_droop)
        crack_right.set_data(BEAM_X[mid:], right_droop)
        crack_left.set_color('#a07030')
        crack_right.set_color('#a07030')
        crack_left.set_linewidth(5)
        crack_right.set_linewidth(5)
 
    # ── Resting straws ──
    for p in resting_straws:
        p.remove()
    resting_straws = []
 
    n_pile = min(step // STRAW_EVERY, MAX_STRAWS)
 
    if past < 3:
        # Straws stacked on beam
        for i in range(n_pile):
            px   = BEAM_MID + straw_dx[i] * 0.65
            py   = straw_beam_y(px, sag) + i * 0.028
            col  = straw_cols[i]
            ang  = straw_ang[i] if i < len(straw_ang) else 0
            patch = make_straw_patch(px, py, straw_ang[i], col)
            ax_s.add_patch(patch)
            resting_straws.append(patch)
    else:
        # Scatter straws on ground after snap
        if SCATTER_POS is None:
            xs = np.linspace(0.12, 0.82, n_pile) + rng.uniform(-0.04, 0.04, n_pile)
            ys = 0.12 + rng.uniform(0, 0.04, n_pile)
            as_ = rng.uniform(-70, 70, n_pile)
            SCATTER_POS = list(zip(xs, ys, as_))
        for i, (px, py, ang) in enumerate(SCATTER_POS[:n_pile]):
            col   = straw_cols[i]
            alpha = max(0.5, 1.0 - snap * 0.5)
            patch = make_straw_patch(px, py, ang, col)
            patch.set_alpha(alpha)
            ax_s.add_patch(patch)
            resting_straws.append(patch)
 
    # ── Falling straw ──
    if falling_straw[0] is not None:
        try:
            falling_straw[0].remove()
        except Exception:
            pass
        falling_straw[0] = None
 
    phase = frame % STRAW_EVERY
    if (phase > 0 and n_pile < MAX_STRAWS
            and past == 0 and n_pile < collapse_step // STRAW_EVERY):
        t      = phase / STRAW_EVERY
        px     = BEAM_MID + straw_dx[n_pile] * 0.65
        dest_y = straw_beam_y(px, sag) + n_pile * 0.028
        fall_y = 0.95 - t * (0.95 - dest_y - STRAW_H / 2)
        ang    = straw_ang[n_pile] * t
        p      = make_straw_patch(px, fall_y, ang, straw_cols[n_pile])
        ax_s.add_patch(p)
        falling_straw[0] = p
 
    # ── HUD ──
    load_txt.set_text(f'F = {F:.2f}   |   {n_alv}/{N} fibres intact')
    if past > 3:
        status_txt.set_text('COLLAPSED')
        status_txt.set_color('#ff3311')
    elif past > 0:
        status_txt.set_text('THE STRAW — CASCADE')
        status_txt.set_color('#ff6622')
    elif sag > 0.75:
        status_txt.set_text('CRITICAL — NEAR FAILURE')
        status_txt.set_color('#ff9933')
    elif sag > 0.45:
        status_txt.set_text('WARNING — BENDING')
        status_txt.set_color('#ffcc44')
    else:
        status_txt.set_text('STABLE')
        status_txt.set_color('#44ee88')
 
    # ── Damage curve ──
    dmg_line.set_data(loads[:step + 1], fraction[:step + 1])
    for c in ax_d.collections:
        c.remove()
    ax_d.fill_between(loads[:step + 1], fraction[:step + 1],
                      alpha=0.15, color='#5aaaf0')
 
    return []
 
# ── Render ────────────────────────────────────────────────────────────────────
total = collapse_step + 40
print(f"Rendering {total} frames at 22fps...")
anim = FuncAnimation(fig, update, frames=total, interval=45, blit=False)
writer = PillowWriter(fps=22)
anim.save('straw_camel.gif', writer=writer, dpi=130)
plt.close()
print("Done — saved straw_camel.gif")