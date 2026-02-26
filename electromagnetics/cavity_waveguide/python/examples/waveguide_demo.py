"""
waveguide_demo.py — Visualise EM wave propagation in PEC waveguides.

生成ファイル:
  waveguide_dispersion.png   — 分散関係 ω vs β (直方体・円形 複数モード)
  waveguide_rect_modes.png   — 直方体導波管: 複数モードの横断面場パターン
  waveguide_rect_prop.png    — 直方体 TE₁₀: z 軸方向の伝搬を 4 断面で可視化
  waveguide_circ_modes.png   — 円形導波管: 複数モードの横断面場パターン

物理的な背景:
  - 導波管: 導体管内を電磁波が +z 方向へ伝搬
  - カットオフ周波数 fc: fc 以下では波は指数減衰 (エバネッセント)
  - 伝搬定数: β = √((ω/c)² − kc²) [実数: 伝搬, 虚数: 減衰]
  - 分散関係: ω² = (βc)² + (kc·c)² → 双曲線; 光の分散 ω=βc とは異なる
  - 群速度 vg = dω/dβ < c, 位相速度 vp = ω/β > c (情報は vg で伝搬)

使用法:
    python waveguide_demo.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os

# パッケージパスを追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), *(['..'] * 5), 'src'))

from mathphys.waveguide import (
    RectangularWaveguide,
    RectangularWaveguideMode,
    CircularWaveguide,
    CircularWaveguideMode,
)

# ── カラーマップ設定 ──────────────────────────────────────────────────────────
CMAP_MAG = 'hot_r'    # 場の強度
CMAP_DIV = 'seismic'  # 正負対称


def _normalise(arr: np.ndarray) -> np.ndarray:
    """配列を最大絶対値で正規化 [-1, 1]。ゼロ配列は 0 を返す。"""
    m = np.max(np.abs(arr))
    return arr / m if m > 0 else arr


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: 分散関係 ω vs β (直方体 + 円形, 複数モード)
# ─────────────────────────────────────────────────────────────────────────────

def fig_dispersion() -> plt.Figure:
    """
    直方体導波管と円形導波管の代表的な分散曲線を描画する。

    縦軸 ω = 2πf を規格化周波数 (ω/ωc) で表示。
    横軸 β も規格化 (β·a) で表示 (a = 広辺長)。

    光の分散 ω = c·β (真空中) と比較することで:
      - fc 付近では β ≈ 0 → ω ≈ ωc (グループ速度 vg = 0)
      - 高周波では β → ω/c (光の分散に漸近, vg → c)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Waveguide Dispersion Relations  ω vs β', fontsize=14, fontweight='bold')

    # ── (A) 直方体導波管 a=22.86 mm (WR-90 標準) ──
    a = 0.02286   # WR-90 広辺 (22.86 mm)
    b = 0.01016   # WR-90 狭辺 (10.16 mm)
    wg_rect = RectangularWaveguide(a, b)

    ax = axes[0]
    ax.set_title(f'Rectangular Waveguide (WR-90)\na = {a*1000:.2f} mm, b = {b*1000:.2f} mm',
                 fontsize=11)

    # 表示する代表モード
    rect_modes_show = [
        wg_rect.mode(1, 0, 'TE'),  # TE₁₀: 基本モード (最低次)
        wg_rect.mode(2, 0, 'TE'),  # TE₂₀
        wg_rect.mode(0, 1, 'TE'),  # TE₀₁
        wg_rect.mode(1, 1, 'TE'),  # TE₁₁
        wg_rect.mode(1, 1, 'TM'),  # TM₁₁
    ]
    colors = ['royalblue', 'tomato', 'limegreen', 'darkorchid', 'saddlebrown']

    f_max = 30e9   # 30 GHz まで表示
    freqs = np.linspace(0, f_max, 500)

    for mode, color in zip(rect_modes_show, colors):
        betas = np.array([float(np.real(mode.propagation_constant(f))) for f in freqs])
        fc_GHz = mode.cutoff_frequency * 1e-9
        # β を rad/m → ラベルに fc を表示
        ax.plot(betas, freqs * 1e-9, color=color, linewidth=2,
                label=f'{mode.label()}  (fc={fc_GHz:.2f} GHz)')
        # カットオフ周波数を水平破線でマーク
        ax.axhline(fc_GHz, color=color, linewidth=0.8, linestyle='--', alpha=0.5)

    # 光の分散 ω = c·β を参照線として表示
    beta_ref = np.linspace(0, 2 * np.pi * f_max / 3e8, 500)
    ax.plot(beta_ref, beta_ref * 3e8 / (2 * np.pi) * 1e-9,
            'k--', linewidth=1.5, label='Light line ω = c·β')

    ax.set_xlabel('β (rad/m)', fontsize=11)
    ax.set_ylabel('Frequency f (GHz)', fontsize=11)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(0, f_max * 1e-9)

    # ── (B) 円形導波管 R=10 mm ──
    R_circ = 0.010  # 半径 10 mm
    wg_circ = CircularWaveguide(R_circ)

    ax2 = axes[1]
    ax2.set_title(f'Circular Waveguide\nR = {R_circ*1000:.1f} mm', fontsize=11)

    circ_modes_show = [
        wg_circ.mode(1, 1, 'TE'),  # TE₁₁: 最低次モード (χ'₁₁ ≈ 1.841)
        wg_circ.mode(0, 1, 'TM'),  # TM₀₁ (χ₀₁ ≈ 2.405)
        wg_circ.mode(2, 1, 'TE'),  # TE₂₁
        wg_circ.mode(0, 1, 'TE'),  # TE₀₁
        wg_circ.mode(1, 1, 'TM'),  # TM₁₁
    ]

    for mode, color in zip(circ_modes_show, colors):
        betas = np.array([float(np.real(mode.propagation_constant(f))) for f in freqs])
        fc_GHz = mode.cutoff_frequency * 1e-9
        ax2.plot(betas, freqs * 1e-9, color=color, linewidth=2,
                 label=f'{mode.label()}  (fc={fc_GHz:.2f} GHz)')
        ax2.axhline(fc_GHz, color=color, linewidth=0.8, linestyle='--', alpha=0.5)

    ax2.plot(beta_ref, beta_ref * 3e8 / (2 * np.pi) * 1e-9,
             'k--', linewidth=1.5, label='Light line')
    ax2.set_xlabel('β (rad/m)', fontsize=11)
    ax2.set_ylabel('Frequency f (GHz)', fontsize=11)
    ax2.legend(fontsize=8, loc='upper left')
    ax2.grid(alpha=0.3)
    ax2.set_xlim(left=0)
    ax2.set_ylim(0, f_max * 1e-9)

    # 分散に関する物理的な注釈
    fig.text(0.5, 0.01,
             'Dashed lines mark cutoff frequencies fc.  Below fc: β=0 (evanescent).  '
             'Group velocity vg = dω/dβ < c,  Phase velocity vp = ω/β > c.',
             ha='center', fontsize=9, style='italic', color='gray')

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: 直方体導波管 — モードアトラス (複数モードの横断面)
# ─────────────────────────────────────────────────────────────────────────────

def fig_rect_mode_atlas() -> plt.Figure:
    """
    直方体導波管の代表的な 6 モードの横断面電場分布 (|E| + 流線) を並べて表示。

    各モードで fc が異なるため、同一周波数で単一モード動作できる
    「単一モード帯域」 (fc(TE₁₀) < f < fc(TE₂₀ or TE₀₁)) が重要。
    断面: z = 0 固定、fc の 1.5 倍の周波数で計算。
    """
    a = 0.02286   # WR-90 広辺
    b = 0.01016   # WR-90 狭辺
    wg = RectangularWaveguide(a, b)

    # 表示モードとその表示周波数 (各モードのカットオフ周波数の 1.5 倍)
    mode_specs = [
        wg.mode(1, 0, 'TE'),  # TE₁₀
        wg.mode(2, 0, 'TE'),  # TE₂₀
        wg.mode(0, 1, 'TE'),  # TE₀₁
        wg.mode(1, 1, 'TE'),  # TE₁₁
        wg.mode(1, 1, 'TM'),  # TM₁₁
        wg.mode(2, 1, 'TE'),  # TE₂₁
    ]

    # 横断面グリッド
    Nx, Ny = 60, 30
    x = np.linspace(0, a, Nx)
    y = np.linspace(0, b, Ny)
    XX, YY = np.meshgrid(x, y)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle(
        'Rectangular Waveguide (WR-90) — Mode Atlas\n'
        '|E| magnitude (colour) + E-field streamlines in xy cross-section at z=0',
        fontsize=12, fontweight='bold',
    )

    for ax, mode in zip(axes.flat, mode_specs):
        # 動作周波数: カットオフの 1.5 倍 (確実に伝搬する)
        freq_op = mode.cutoff_frequency * 1.5

        fld  = mode.fields(XX, YY, z=0.0, frequency=freq_op, phase=0.0)
        emag = np.sqrt(fld.Ex**2 + fld.Ey**2 + fld.Ez**2)

        im = ax.pcolormesh(XX * 1000, YY * 1000, emag,
                           cmap=CMAP_MAG, vmin=0, shading='gouraud')
        plt.colorbar(im, ax=ax, label='|E| (a.u.)', fraction=0.046, pad=0.04)

        # 横断面 (xy) の電場流線
        U, V = fld.Ex, fld.Ey
        strength = np.sqrt(U**2 + V**2)
        thr = 0.02 * strength.max() if strength.max() > 0 else 1e-30
        U = np.where(strength > thr, U, 0.0)
        V = np.where(strength > thr, V, 0.0)
        if np.any(np.abs(U) > 0) or np.any(np.abs(V) > 0):
            ax.streamplot(XX * 1000, YY * 1000, U, V,
                          color='cyan', linewidth=0.7, density=1.2, arrowsize=0.8)

        fc_GHz  = mode.cutoff_frequency * 1e-9
        fop_GHz = freq_op * 1e-9
        ax.set_title(
            f'{mode.label()}   fc = {fc_GHz:.2f} GHz\n(at f = {fop_GHz:.2f} GHz)',
            fontsize=9, fontweight='bold',
        )
        ax.set_xlabel('x (mm)', fontsize=8)
        ax.set_ylabel('y (mm)', fontsize=8)
        ax.set_aspect('equal')

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: 直方体 TE₁₀ — z 方向への伝搬 (4 断面)
# ─────────────────────────────────────────────────────────────────────────────

def fig_rect_te10_propagation() -> plt.Figure:
    """
    直方体導波管 TE₁₀ の電場が z 方向に伝搬する様子を 4 断面で可視化。

    同じ ωt=0 に固定して、z = 0, λg/4, λg/2, 3λg/4 での横断面を表示。
    (λg: 導波管波長 = 2π/β)

    これにより進行波の「空間的な位相変化」が直感的に理解できる。
    """
    a = 0.02286  # WR-90
    b = 0.01016
    mode = RectangularWaveguideMode(a, b, m=1, n=0, mode_type='TE')
    fc   = mode.cutoff_frequency  # ≈ 6.56 GHz

    # 動作周波数: fc の 1.4 倍 (十分伝搬するが低めに設定して λg を大きく)
    f_op = fc * 1.4
    beta = float(np.real(mode.propagation_constant(f_op)))
    lambda_g = 2 * np.pi / beta  # 導波管波長

    fc_GHz  = fc * 1e-9
    fop_GHz = f_op * 1e-9
    lg_mm   = lambda_g * 1000

    # 表示する z 位置: 0, λg/4, λg/2, 3λg/4
    z_positions = [0.0, lambda_g / 4, lambda_g / 2, 3 * lambda_g / 4]
    z_labels    = ['z = 0', r'z = λg/4', r'z = λg/2', r'z = 3λg/4']

    # 横断面グリッド
    Nx, Ny = 60, 30
    x = np.linspace(0, a, Nx)
    y = np.linspace(0, b, Ny)
    XX, YY = np.meshgrid(x, y)

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    fig.suptitle(
        f'Rectangular Waveguide TE₁₀  —  Propagation along z-axis\n'
        f'f = {fop_GHz:.2f} GHz (= 1.4 fc),  fc = {fc_GHz:.2f} GHz,  '
        f'λg = {lg_mm:.1f} mm,  β = {beta:.1f} rad/m',
        fontsize=12, fontweight='bold',
    )

    for col, (z_pos, z_lbl) in enumerate(zip(z_positions, z_labels)):
        fld  = mode.fields(XX, YY, z=z_pos, frequency=f_op, phase=0.0)

        ax_e = axes[0, col]   # 上段: 電場 (Ey は TE₁₀ の主成分)
        ax_h = axes[1, col]   # 下段: 磁場 (Hx, Hz)

        # TE₁₀ の電場は主に Ey
        Ey_norm = _normalise(fld.Ey)
        ax_e.pcolormesh(XX * 1000, YY * 1000, Ey_norm,
                        cmap=CMAP_DIV, vmin=-1, vmax=1, shading='gouraud')
        ax_e.set_title(z_lbl, fontsize=10)
        ax_e.set_xlabel('x (mm)', fontsize=9)
        if col == 0:
            ax_e.set_ylabel('y (mm)', fontsize=9)
            ax_e.text(-0.35, 0.5, 'Ey (colour)', transform=ax_e.transAxes,
                      rotation=90, va='center', fontsize=9, color='darkred', fontweight='bold')
        ax_e.set_aspect('equal')

        # 磁場 Hx (TE₁₀ は Hx と Hz を持つ)
        Hx_norm = _normalise(fld.Hx)
        im_h = ax_h.pcolormesh(XX * 1000, YY * 1000, Hx_norm,
                                cmap=CMAP_DIV, vmin=-1, vmax=1, shading='gouraud')
        ax_h.set_xlabel('x (mm)', fontsize=9)
        if col == 0:
            ax_h.set_ylabel('y (mm)', fontsize=9)
            ax_h.text(-0.35, 0.5, 'Hx (colour)', transform=ax_h.transAxes,
                      rotation=90, va='center', fontsize=9, color='purple', fontweight='bold')
        ax_h.set_aspect('equal')

    # 右側にカラーバーを追加
    cbar_ax = fig.add_axes([0.92, 0.1, 0.015, 0.8])
    sm = plt.cm.ScalarMappable(cmap=CMAP_DIV, norm=plt.Normalize(-1, 1))
    fig.colorbar(sm, cax=cbar_ax, label='Field amplitude (norm.)')

    fig.text(0.5, 0.01,
             'Phase of the travelling wave changes by 90° every λg/4 along z.  '
             'TE₁₀ dominant fields: Ey (electric), Hx and Hz (magnetic).',
             ha='center', fontsize=8, style='italic', color='gray')

    plt.tight_layout(rect=[0, 0.05, 0.91, 1])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: 円形導波管 — モードアトラス
# ─────────────────────────────────────────────────────────────────────────────

def fig_circ_mode_atlas() -> plt.Figure:
    """
    円形導波管の代表的な 6 モードの横断面電場分布を表示。

    円形導波管の特徴:
      - TE₁₁ が最低次モード (χ'₁₁ ≈ 1.841)
      - TM₀₁ は χ₀₁ ≈ 2.405 で次に低い
      - TE₀₁ は低損失モードとして長距離通信に有用 (Hφ のみで Eρ=0)
      - m ≥ 1 のモードは 2 重縮退 (cos/sin の 2 極性)

    断面: z = 0, 動作周波数 = fc × 1.5
    """
    R = 0.010  # 半径 10 mm
    wg = CircularWaveguide(R)

    mode_specs = [
        wg.mode(1, 1, 'TE'),  # TE₁₁: 最低次 (χ'₁₁ ≈ 1.841)
        wg.mode(0, 1, 'TM'),  # TM₀₁ (χ₀₁ ≈ 2.405)
        wg.mode(2, 1, 'TE'),  # TE₂₁
        wg.mode(0, 1, 'TE'),  # TE₀₁ (低損失モード; χ'₀₁ ≈ 3.832)
        wg.mode(1, 1, 'TM'),  # TM₁₁
        wg.mode(1, 2, 'TE'),  # TE₁₂
    ]

    # 極座標グリッドを Cartesian に変換
    Nr, Nphi = 40, 60
    r_1d   = np.linspace(0, R, Nr)
    phi_1d = np.linspace(0, 2 * np.pi, Nphi)
    RR, PHI = np.meshgrid(r_1d, phi_1d)
    # Cartesian 変換 (描画用)
    XX_cart = RR * np.cos(PHI)
    YY_cart = RR * np.sin(PHI)

    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    fig.suptitle(
        'Circular Waveguide (R = 10 mm) — Mode Atlas\n'
        '|E| magnitude in cross-section at z = 0  (colour + transverse arrows)',
        fontsize=12, fontweight='bold',
    )

    for ax, mode in zip(axes.flat, mode_specs):
        freq_op = mode.cutoff_frequency * 1.5  # カットオフの 1.5 倍で動作

        # 極座標で場を計算
        fld = mode.fields_polar(RR, PHI, z=0.0, frequency=freq_op, phase=0.0)
        # fields_polar の戻り値: Ex=Eρ, Ey=Eφ, Ez, Hx=Hρ, Hy=Hφ, Hz

        # Eρ, Eφ → Cartesian の Ex, Ey に変換
        Erho = fld.Ex  # 動径方向電場
        Ephi = fld.Ey  # 方位角電場
        Ex_c = Erho * np.cos(PHI) - Ephi * np.sin(PHI)  # Cartesian Ex
        Ey_c = Erho * np.sin(PHI) + Ephi * np.cos(PHI)  # Cartesian Ey

        emag = np.sqrt(Ex_c**2 + Ey_c**2 + fld.Ez**2)

        im = ax.pcolormesh(XX_cart * 1000, YY_cart * 1000, emag,
                           cmap=CMAP_MAG, vmin=0, shading='gouraud')
        plt.colorbar(im, ax=ax, label='|E| (a.u.)', fraction=0.046, pad=0.04)

        # 横断面電場の矢印
        step = 4
        ax.quiver(XX_cart[::step, ::step] * 1000, YY_cart[::step, ::step] * 1000,
                  Ex_c[::step, ::step], Ey_c[::step, ::step],
                  color='cyan', scale=None, scale_units='xy', alpha=0.8)

        # 円形壁を描画
        theta_c = np.linspace(0, 2 * np.pi, 200)
        ax.plot(R * np.cos(theta_c) * 1000, R * np.sin(theta_c) * 1000,
                'w-', linewidth=1.5)

        fc_GHz  = mode.cutoff_frequency * 1e-9
        fop_GHz = freq_op * 1e-9
        ax.set_title(
            f'{mode.label()}   fc = {fc_GHz:.2f} GHz\n(at f = {fop_GHz:.2f} GHz)',
            fontsize=9, fontweight='bold',
        )
        ax.set_xlabel('x (mm)', fontsize=8)
        ax.set_ylabel('y (mm)', fontsize=8)
        ax.set_aspect('equal')

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# メインルーティン
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    out_dir = os.path.dirname(os.path.abspath(__file__))

    tasks = [
        ('分散関係 ω vs β',          fig_dispersion,             'waveguide_dispersion.png'),
        ('直方体導波管 モードアトラス', fig_rect_mode_atlas,         'waveguide_rect_modes.png'),
        ('直方体 TE₁₀ z 方向伝搬',   fig_rect_te10_propagation,  'waveguide_rect_prop.png'),
        ('円形導波管 モードアトラス',  fig_circ_mode_atlas,         'waveguide_circ_modes.png'),
    ]

    for description, func, filename in tasks:
        print(f'生成中: {description} ...')
        fig = func()
        path = os.path.join(out_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  → {path}')

    print('\n完了。生成ファイル:')
    for _, _, fname in tasks:
        print(f'  {os.path.join(out_dir, fname)}')
