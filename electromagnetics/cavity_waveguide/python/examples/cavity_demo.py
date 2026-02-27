"""
cavity_demo.py — Visualise EM resonance modes in three types of PEC cavities.

生成ファイル:
  cavity_freq_chart.png  — 3種類のキャビティの共振周波数比較
  cavity_rect_te101.png  — 直方体 TE₁₀₁ モード: 4位相でのエネルギー交換
  cavity_rect_atlas.png  — 直方体キャビティ: 6つのモードの場パターン
  cavity_cyl_tm010.png   — 円筒型 TM₀₁₀ モード: ρ-z 断面 + 位相変化
  cavity_sph_tm11.png    — 球形 TM₁₁ モード: r-θ 断面

物理的な背景:
  - 共振キャビティ: 導体壁に囲まれた空洞で EM 波が定常波を形成
  - 境界条件: 完全導体 (PEC) 壁面で接線 E = 0
  - 時間依存性: E(r,t) = E(r)·cos(ωt), H(r,t) = H(r)·sin(ωt)
                電場と磁場は 90° 位相ずれ → エネルギーが E↔H 間で振動

使用法:
    python cavity_demo.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # ディスプレイなし環境でのバックエンド
import matplotlib.pyplot as plt
import sys
import os

# パッケージのパスを追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), *(['..'] * 5), 'src'))

from mathphys.cavity import (
    RectangularCavityMode,
    CylindricalCavityMode,
    SphericalCavityMode,
    rectangular_cavity_modes,
    cylindrical_cavity_modes,
    spherical_cavity_modes,
)

# ── カラーマップ設定 ──────────────────────────────────────────────────────────
CMAP_E   = 'RdBu_r'   # 電場分布 (赤=正, 青=負)
CMAP_H   = 'PuOr'     # 磁場分布
CMAP_MAG = 'hot_r'    # 場の強度 (明るいほど強い)
CMAP_DIV = 'seismic'  # 正負対称な発散型


def _normalise(arr: np.ndarray) -> np.ndarray:
    """配列を最大絶対値で正規化して [-1, 1] にスケール。"""
    m = np.max(np.abs(arr))
    return arr / m if m > 0 else arr


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: 共振周波数チャート
# ─────────────────────────────────────────────────────────────────────────────

def fig_frequency_chart() -> plt.Figure:
    """
    3種類のキャビティ (直方体・円筒・球形) の最低次共振周波数を棒グラフで比較する。

    異なるキャビティ形状では Bessel 零点や三角関数の零点が異なるため、
    同じ体積でも共振周波数のスペクトルが大きく異なる。
    """
    # キャビティ寸法 (SI単位 m)
    a, b, d    = 0.04, 0.02, 0.03   # 直方体: 4×2×3 cm
    R_cyl, L_cyl = 0.015, 0.03      # 円筒: 半径 1.5 cm, 高さ 3 cm
    R_sph      = 0.02               # 球形: 半径 2 cm

    # 各キャビティの最低次 8 モードを周波数昇順で取得
    rect_modes = rectangular_cavity_modes(a, b, d, n_modes=8)
    cyl_modes  = cylindrical_cavity_modes(R_cyl, L_cyl, n_modes=8)
    sph_modes  = spherical_cavity_modes(R_sph, n_modes=8)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    fig.suptitle('Electromagnetic Cavity — Resonant Frequencies', fontsize=14, fontweight='bold')

    cavity_specs = [
        (rect_modes, 'Rectangular\n(a=4, b=2, d=3 cm)', 'steelblue'),
        (cyl_modes,  'Cylindrical\n(R=1.5 cm, L=3 cm)', 'darkorange'),
        (sph_modes,  'Spherical\n(R=2 cm)',             'seagreen'),
    ]

    for ax, (modes, title, color) in zip(axes, cavity_specs):
        freqs_GHz = [mo.resonant_frequency * 1e-9 for mo in modes]
        labels    = [mo.label() for mo in modes]
        bars = ax.barh(range(len(freqs_GHz)), freqs_GHz,
                       color=color, alpha=0.8, edgecolor='k', linewidth=0.5)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Frequency (GHz)', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        # 各バーの右に周波数数値を表示
        for bar, f in zip(bars, freqs_GHz):
            ax.text(f + max(freqs_GHz) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f'{f:.2f} GHz', va='center', fontsize=8)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: 直方体 TE₁₀₁ — 4 位相でのエネルギー交換
# ─────────────────────────────────────────────────────────────────────────────

def fig_rect_te101_phases() -> plt.Figure:
    """
    直方体キャビティ TE₁₀₁ モードの電場・磁場を 4 位相で表示する。

    TE₁₀₁ はマイクロ波炉などで最も一般的に使われる基本モード。
    物理的なポイント:
      - E と H の位相は 90° ずれている (ωt=0 で E 最大・H=0, ωt=π/2 で H 最大・E=0)
      - エネルギーは電場エネルギー (ε₀|E|²/2) と磁場エネルギー (|H|²/2μ₀) の間を振動

    断面: xz 平面 (y = b/2) を可視化
    """
    a, b, d = 0.04, 0.02, 0.03   # キャビティ寸法 (m)
    mode = RectangularCavityMode(a, b, d, m=1, n=0, p=1, mode_type='TE')
    freq = mode.resonant_frequency

    # xz 断面グリッド (y = b/2 固定)
    Nx, Nz = 60, 80
    x  = np.linspace(0, a, Nx)
    z  = np.linspace(0, d, Nz)
    XX, ZZ = np.meshgrid(x, z)
    YY = np.ones_like(XX) * b / 2  # y = b/2 (中央断面)

    # 4 つの位相: ωt = 0, π/4, π/2, 3π/4
    phases = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    labels = [r'$\omega t = 0$    (E max, H=0)',
              r'$\omega t = \pi/4$',
              r'$\omega t = \pi/2$  (H max, E=0)',
              r'$\omega t = 3\pi/4$']

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    fig.suptitle(
        f'Rectangular Cavity TE₁₀₁  —  f = {freq * 1e-9:.3f} GHz\n'
        f'(a={a*100:.0f} cm, b={b*100:.0f} cm, d={d*100:.0f} cm)  ·  '
        'xz cross-section at y = b/2',
        fontsize=12, fontweight='bold',
    )

    for col, (ph, lbl) in enumerate(zip(phases, labels)):
        fld = mode.fields(XX, YY, ZZ, phase=ph)

        ax_e = axes[0, col]   # 上段: 電場
        ax_h = axes[1, col]   # 下段: 磁場

        # ── 電場 (Ex, Ez 成分) ──
        E_norm = _normalise(fld.E_magnitude)
        ax_e.pcolormesh(XX * 100, ZZ * 100, E_norm,
                        cmap=CMAP_MAG, vmin=0, vmax=1, shading='gouraud')
        # 矢印: Ex と Ez の方向を可視化
        step = 5
        ax_e.quiver(XX[::step, ::step] * 100, ZZ[::step, ::step] * 100,
                    fld.Ex[::step, ::step], fld.Ez[::step, ::step],
                    color='cyan', scale=8, width=0.003, alpha=0.9)
        ax_e.set_title(lbl, fontsize=9)
        ax_e.set_aspect('equal')
        ax_e.set_ylabel('z (cm)', fontsize=9) if col == 0 else None
        ax_e.set_xlabel('x (cm)', fontsize=9)
        # 左端ラベル
        if col == 0:
            ax_e.text(-0.35, 0.5, 'Electric\nField |E|', transform=ax_e.transAxes,
                      rotation=90, va='center', fontsize=9, color='darkred', fontweight='bold')

        # ── 磁場 (Hy 成分: xz 平面内で紙面に垂直) ──
        Hy_norm = _normalise(fld.Hy)
        ax_h.pcolormesh(XX * 100, ZZ * 100, Hy_norm,
                        cmap=CMAP_DIV, vmin=-1, vmax=1, shading='gouraud')
        ax_h.set_aspect('equal')
        ax_h.set_xlabel('x (cm)', fontsize=9)
        ax_h.set_ylabel('z (cm)', fontsize=9) if col == 0 else None
        if col == 0:
            ax_h.text(-0.35, 0.5, 'Magnetic\nField Hy', transform=ax_h.transAxes,
                      rotation=90, va='center', fontsize=9, color='purple', fontweight='bold')

    # 下部に物理的な説明文を追加
    fig.text(0.5, 0.01,
             'Energy oscillates between E (electric) and H (magnetic) fields at frequency f.\n'
             'At ωt=0: E is maximum, H=0.   At ωt=π/2: H is maximum, E=0.   '
             'Total U = Ue + Um = const.',
             ha='center', fontsize=8, style='italic', color='gray')

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: 直方体キャビティ — モードアトラス (6 モード)
# ─────────────────────────────────────────────────────────────────────────────

def fig_rect_mode_atlas() -> plt.Figure:
    """
    直方体キャビティの代表的な 6 モードの電場分布 (xy 断面) を比較する。

    TE モード: Ez = 0 (E は xy 面内のみ)
    TM モード: Hz = 0 (H は xy 面内のみ)

    モードの次数 (m, n, p) が高いほど節面が多くなり、共振周波数が高くなる。
    断面: z = d/2 (キャビティ中央)
    """
    a, b, d = 0.04, 0.02, 0.03  # キャビティ寸法 (m)

    # 代表的な 6 モードを選択
    selected_modes = [
        RectangularCavityMode(a, b, d, 1, 0, 1, 'TE'),  # 最低次 TE モード
        RectangularCavityMode(a, b, d, 1, 1, 1, 'TE'),
        RectangularCavityMode(a, b, d, 2, 0, 1, 'TE'),
        RectangularCavityMode(a, b, d, 1, 1, 0, 'TM'),  # 最低次 TM モード
        RectangularCavityMode(a, b, d, 2, 1, 0, 'TM'),
        RectangularCavityMode(a, b, d, 1, 1, 1, 'TM'),
    ]

    # xy 断面グリッド (z = d/2)
    Nx, Ny = 60, 40
    x = np.linspace(0, a, Nx)
    y = np.linspace(0, b, Ny)
    XX, YY = np.meshgrid(x, y)
    ZZ = np.ones_like(XX) * d / 2  # z = d/2 固定

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle(
        'Rectangular Cavity — Mode Atlas\n'
        '|E| magnitude (colour) + E-field streamlines in xy cross-section at z = d/2',
        fontsize=12, fontweight='bold',
    )

    for ax, mode in zip(axes.flat, selected_modes):
        fld  = mode.fields(XX, YY, ZZ, phase=0.0)  # ωt = 0 (E が最大)
        emag = fld.E_magnitude

        # 電場強度をカラーマップで表示
        im = ax.pcolormesh(XX * 100, YY * 100, emag,
                           cmap=CMAP_MAG, vmin=0, shading='gouraud')
        plt.colorbar(im, ax=ax, label='|E| (a.u.)', fraction=0.046, pad=0.04)

        # 電場の xy 成分を流線で表示 (向きと流れを直感的に理解)
        U, V = fld.Ex, fld.Ey
        # 極めて弱い領域ではマスクしてノイズを除去
        strength = np.sqrt(U**2 + V**2)
        threshold = 0.02 * strength.max()
        U = np.where(strength > threshold, U, 0.0)
        V = np.where(strength > threshold, V, 0.0)
        ax.streamplot(XX * 100, YY * 100, U, V,
                      color='cyan', linewidth=0.7, density=1.0, arrowsize=0.8)

        ax.set_title(
            f'{mode.label()}   f = {mode.resonant_frequency * 1e-9:.3f} GHz',
            fontsize=9, fontweight='bold',
        )
        ax.set_xlabel('x (cm)', fontsize=8)
        ax.set_ylabel('y (cm)', fontsize=8)
        ax.set_aspect('equal')

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: 円筒型キャビティ TM₀₁₀ — ρ-z 断面 + 位相変化
# ─────────────────────────────────────────────────────────────────────────────

def fig_cyl_tm010_phases() -> plt.Figure:
    """
    円筒型キャビティ TM₀₁₀ モードを ρ-z 断面で 4 位相に亘って可視化する。

    TM₀₁₀ は円筒型キャビティの最低次 TM モード:
      - p=0 なので z 方向の変化なし (軸対称)
      - Ez は ρ に対し J₀(χ₀₁ ρ/R) で変化 (χ₀₁ ≈ 2.405)
      - Hφ だけが存在し、Ez と 90° 位相ずれ

    加速器空洞 (粒子加速用) としても広く使われる基本モード。
    """
    R, L = 0.015, 0.03  # 円筒キャビティ: 半径 1.5 cm, 高さ 3 cm
    mode = CylindricalCavityMode(R, L, m=0, n=1, p=0, mode_type='TM')
    freq = mode.resonant_frequency

    # ρ-z グリッド
    Nr, Nz = 40, 60
    rho_1d = np.linspace(0, R, Nr)
    z_1d   = np.linspace(0, L, Nz)
    RHO, ZZ = np.meshgrid(rho_1d, z_1d)

    phases = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    labels = [r'$\omega t=0$  (E max)',
              r'$\omega t=\pi/4$',
              r'$\omega t=\pi/2$  (H max)',
              r'$\omega t=3\pi/4$']

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    fig.suptitle(
        f'Cylindrical Cavity TM₀₁₀  —  f = {freq * 1e-9:.3f} GHz\n'
        f'(R = {R*100:.1f} cm, L = {L*100:.1f} cm)  ·  ρ-z cross-section at φ = 0',
        fontsize=12, fontweight='bold',
    )

    for col, (ph, lbl) in enumerate(zip(phases, labels)):
        fld = mode.fields_rz(RHO, ZZ, phi=0.0, phase=ph)
        # fields_rz の戻り値: Ex=Eρ, Ey=Eφ, Ez, Hx=Hρ, Hy=Hφ, Hz

        ax_e = axes[0, col]   # 上段: 電場
        ax_h = axes[1, col]   # 下段: 磁場

        # ── Ez (軸方向電場) をカラー + Eρ を矢印で表示 ──
        Ez_norm = _normalise(fld.Ez)
        ax_e.pcolormesh(RHO * 100, ZZ * 100, Ez_norm,
                        cmap=CMAP_DIV, vmin=-1, vmax=1, shading='gouraud')
        step = 4
        ax_e.quiver(RHO[::step, ::step] * 100, ZZ[::step, ::step] * 100,
                    fld.Ex[::step, ::step],   # Eρ
                    fld.Ez[::step, ::step],   # Ez
                    color='k', scale=6, width=0.004, alpha=0.8)
        ax_e.set_title(lbl, fontsize=9)
        ax_e.set_xlabel('ρ (cm)', fontsize=9)
        if col == 0:
            ax_e.set_ylabel('z (cm)', fontsize=9)
            ax_e.text(-0.35, 0.5, 'Ez (colour)\nEρ (arrows)',
                      transform=ax_e.transAxes, rotation=90,
                      va='center', fontsize=8, color='darkred', fontweight='bold')

        # ── Hφ (方位角磁場) をカラーで表示 ──
        Hphi_norm = _normalise(fld.Hy)
        ax_h.pcolormesh(RHO * 100, ZZ * 100, Hphi_norm,
                        cmap=CMAP_DIV, vmin=-1, vmax=1, shading='gouraud')
        ax_h.set_xlabel('ρ (cm)', fontsize=9)
        if col == 0:
            ax_h.set_ylabel('z (cm)', fontsize=9)
            ax_h.text(-0.35, 0.5, 'Hφ (colour)',
                      transform=ax_h.transAxes, rotation=90,
                      va='center', fontsize=8, color='purple', fontweight='bold')

    # 物理説明
    fig.text(0.5, 0.01,
             'TM₀₁₀: Ez = J₀(χ₀₁ρ/R) cos(ωt),  Hφ = −(ωε₀/kc) J₀′(kc ρ) sin(ωt).  '
             'χ₀₁ ≈ 2.405 (first zero of J₀).',
             ha='center', fontsize=8, style='italic', color='gray')

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: 球形キャビティ TM₁₁ — r-θ 断面
# ─────────────────────────────────────────────────────────────────────────────

def fig_spherical_tm11() -> plt.Figure:
    """
    球形キャビティ TM₁₁ モードを r-θ 断面で 4 位相に亘って可視化する。

    球形キャビティ TM₁,₁ モード:
      - 球面 Bessel 関数 j₁(kr) で変化
      - 境界条件: r=R で j₁(kR) = 0 → χ₁₁ ≈ 4.493
      - 球座標 (Er, Eθ, Hφ) が非零

    Cartesian 座標 (x=r sinθ, z=r cosθ) に変換して描画することで
    直感的な球形キャビティの様子を表現する。
    """
    R    = 0.02  # 球半径 2 cm
    mode = SphericalCavityMode(R, l=1, n=1, mode_type='TM')
    freq = mode.resonant_frequency

    # r-θ グリッド
    Nr, Nt  = 40, 60
    r_1d    = np.linspace(1e-6, R, Nr)   # r=0 の特異点を避けるため微小値から開始
    theta_1d = np.linspace(0, np.pi, Nt)
    RR, TT  = np.meshgrid(r_1d, theta_1d)

    phases = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    labels = [r'$\omega t=0$  (E max)',
              r'$\omega t=\pi/4$',
              r'$\omega t=\pi/2$  (H max)',
              r'$\omega t=3\pi/4$']

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle(
        f'Spherical Cavity TM₁₁  —  f = {freq * 1e-9:.3f} GHz  (R = {R*100:.1f} cm)\n'
        'r-θ cross-section at φ=0  ·  colour = |E|,  arrows = (Er, Eθ) in Cartesian',
        fontsize=12, fontweight='bold',
    )

    for ax, ph, lbl in zip(axes, phases, labels):
        fld  = mode.fields_rtheta(RR, TT, phi=0.0, m_az=0, phase=ph)
        # fields_rtheta の戻り値: Ex=Er, Ey=Eθ, Ez=Eφ, Hx=Hr, Hy=Hθ, Hz=Hφ
        emag = np.sqrt(fld.Ex**2 + fld.Ey**2)

        # r-θ を Cartesian に変換 (x = r sinθ, z = r cosθ)
        X_cart = RR * np.sin(TT) * 100  # [cm]
        Z_cart = RR * np.cos(TT) * 100  # [cm]

        im = ax.pcolormesh(X_cart, Z_cart, _normalise(emag),
                           cmap=CMAP_MAG, vmin=0, vmax=1, shading='gouraud')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='|E| (norm.)')

        # 矢印: (Er, Eθ) を Cartesian に変換して表示
        step = 5
        # ベクトル変換: Cartesian での (x,z) 方向
        Er_r = fld.Ex[::step, ::step]          # 動径方向 Er
        Et_r = fld.Ey[::step, ::step]          # 極方向 Eθ
        sin_t = np.sin(TT[::step, ::step])
        cos_t = np.cos(TT[::step, ::step])
        # Er → (sinθ, cosθ),  Eθ → (cosθ, -sinθ) (Cartesian で)
        Ux = Er_r * sin_t + Et_r * cos_t   # x 方向
        Uz = Er_r * cos_t - Et_r * sin_t   # z 方向
        ax.quiver(X_cart[::step, ::step], Z_cart[::step, ::step],
                  Ux, Uz, color='cyan', scale=10, width=0.003, alpha=0.8)

        # 球の輪郭を描画
        theta_c = np.linspace(0, np.pi, 100)
        ax.plot(R * np.sin(theta_c) * 100, R * np.cos(theta_c) * 100,
                'w-', linewidth=1.5, label='PEC wall')

        ax.set_title(lbl, fontsize=10)
        ax.set_xlabel('r sinθ (cm)', fontsize=8)
        if ax == axes[0]:
            ax.set_ylabel('r cosθ (cm)', fontsize=8)
        ax.set_aspect('equal')

    # 物理説明
    fig.text(0.5, 0.01,
             'TM₁₁: Er, Eθ non-zero.  Boundary: j₁(kR)=0 → χ₁₁ ≈ 4.493.  '
             'f = c χ₁₁ / (2π R).',
             ha='center', fontsize=8, style='italic', color='gray')

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# メインルーティン
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    out_dir = os.path.dirname(os.path.abspath(__file__))

    tasks = [
        ('共振周波数チャート',            fig_frequency_chart,       'cavity_freq_chart.png'),
        ('直方体 TE₁₀₁ 位相パネル',      fig_rect_te101_phases,     'cavity_rect_te101.png'),
        ('直方体キャビティ モードアトラス', fig_rect_mode_atlas,       'cavity_rect_atlas.png'),
        ('円筒型 TM₀₁₀ 位相パネル',      fig_cyl_tm010_phases,      'cavity_cyl_tm010.png'),
        ('球形 TM₁₁ 断面',               fig_spherical_tm11,         'cavity_sph_tm11.png'),
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
