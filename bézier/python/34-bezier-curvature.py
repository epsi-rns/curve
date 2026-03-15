import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import BPoly


# ============================================================
# -- Switch  (only change this one line) ---------------------
# ============================================================

CASE = 2   # 1 = Quadratic 2D,  2 = Quartic 3D


# ============================================================
# -- CASE Data  (do not change) ------------------------------
# ============================================================

CASE_DATA = {
    1: dict(
        points      = np.array([
            [1, 1],   # P0
            [3, 3],   # P1
            [5, 1],   # P2
        ]),
        u_highlight = [0.00, 0.25, 0.50, 0.75, 1.00],
        u_dc        = 0.50,
        output      = '34-bezier_curvature_quadratic_2d.png',
    ),
    2: dict(
        points      = np.array([
            [0, 0, 1],   # P0
            [0, 4, 1],   # P1
            [4, 0, 1],   # P2
            [4, 4, 1],   # P3
            [5, 4, 1],   # P4
        ]),
        u_highlight = [0.00, 0.25, 0.50, 0.75, 1.00],
        u_dc        = 0.50,
        output      = '34-bezier_curvature_quartic_3d.png',
    ),
}


# ============================================================
# -- BezierCurve  (generic) ----------------------------------
# ============================================================

class BezierCurve:
    """
    Generic Bezier curve, any degree, 2D or 3D.

    Curve evaluation uses scipy.interpolate.BPoly.

    Curvature uses derivative Bezier curves:
      P'(u)  — BPoly with control points Q_i = n*(P_{i+1} - P_i)
      P''(u) — BPoly with control points R_i = (n-1)*(Q_{i+1} - Q_i)

    kappa = |P' x P''| / |P'|^3
      2D: cross product is scalar  x'y'' - y'x''
      3D: cross product is a vector, take its norm
    """

    def __init__(self, points: np.ndarray):
        self.points = points
        self.n      = len(points)
        self.degree = self.n - 1
        self.dim    = points.shape[1]

        c = points[:, np.newaxis, :]
        self._bpoly = BPoly(c, [0, 1])

        Q = self.degree * np.diff(points, axis=0)
        self._bpoly_d1 = BPoly(Q[:, np.newaxis, :], [0, 1])

        if len(Q) > 1:
            R = (self.degree - 1) * np.diff(Q, axis=0)
            self._bpoly_d2 = BPoly(R[:, np.newaxis, :], [0, 1])
        else:
            self._bpoly_d2 = None

        print(c)
        print(Q)
        print(R)

    def point(self, u):
        return self._bpoly(u)

    def curve(self, n_pts=300):
        return self._bpoly(np.linspace(0, 1, n_pts))

    def highlight_points(self, u_list):
        return self._bpoly(np.array(u_list))

    def de_casteljau(self, u):
        levels  = [self.points.copy()]
        current = self.points.copy()
        while len(current) > 1:
            current = np.array([
                (1-u)*current[j] + u*current[j+1]
                for j in range(len(current) - 1)
            ])
            levels.append(current)
        return levels

    def d1(self, u):
        return self._bpoly_d1(u)

    def d2(self, u):
        if self._bpoly_d2 is None:
            return np.zeros((len(np.atleast_1d(u)), self.dim))
        return self._bpoly_d2(u)

    def curvature(self, u):
        u_arr = np.atleast_1d(u)
        d1    = self.d1(u_arr)
        d2    = self.d2(u_arr)
        if self.dim == 2:
            cross = np.abs(d1[:, 0]*d2[:, 1] - d1[:, 1]*d2[:, 0])
        else:
            cross = np.linalg.norm(np.cross(d1, d2), axis=1)
        speed = np.linalg.norm(d1, axis=1)
        denom = np.where(speed > 1e-10, speed**3, np.inf)
        return cross / denom

    @property
    def degree_name(self):
        names = {2:'Quadratic', 3:'Cubic', 4:'Quartic', 5:'Quintic'}
        return names.get(self.degree, f'Degree-{self.degree}')

    @property
    def dim_name(self):
        return '3D' if self.dim == 3 else '2D'


# ============================================================
# -- BezierPlot  (full plot + curvature comb) ----------------
# ============================================================

class BezierPlot:
    """
    Full plot matching 32-bezier-scipy.py output:
      curve, control polygon, highlight points, DC levels,
      equation box — plus curvature comb added on top.
    """

    COLOR_CURVE   = '#1565C0'
    COLOR_POLYGON = '#E53935'
    COLOR_CTRL    = '#E53935'
    COLOR_POINT   = '#1B5E20'
    COLOR_LABEL   = '#4A148C'
    COLOR_COMB    = '#90A4AE'
    COLOR_DC = [
        '#F57F17',   # level 1 — amber
        '#EC407A',   # level 2 — pink
        '#7B1FA2',   # level 3 — violet
        '#1B5E20',   # level 4 — green (final point)
    ]

    def __init__(self, bezier, u_highlight, u_dc=0.5,
                 n_pts=300, comb_scale=None):
        self.bezier      = bezier
        self.u_highlight = u_highlight
        self.u_dc        = u_dc
        self.n_pts       = n_pts
        self.is_3d       = (bezier.dim == 3)

        self.u_vals    = np.linspace(0, 1, n_pts)
        self.curve_pts = bezier.curve(n_pts)
        self.kappa     = bezier.curvature(self.u_vals)

        if comb_scale is None:
            ext     = np.ptp(self.curve_pts[:, :2], axis=0).max()
            kap_max = np.percentile(self.kappa, 95)
            self.comb_scale = (0.15 * ext / kap_max
                               if kap_max > 1e-10 else 0.1)
        else:
            self.comb_scale = comb_scale

        sns.set_theme(style='whitegrid')
        if self.is_3d:
            self.fig = plt.figure(figsize=(10, 7))
            self.ax  = self.fig.add_subplot(111, projection='3d')
        else:
            self.fig, self.ax = plt.subplots(figsize=(9, 7))

    # -- Coordinate helpers ----------------------------------

    def _coords(self, pts):
        if self.is_3d:
            return pts[:, 0], pts[:, 1], pts[:, 2]
        return pts[:, 0], pts[:, 1]

    def _plot_line(self, pts, **kwargs):
        self.ax.plot(*self._coords(pts), **kwargs)

    def _scatter_pts(self, pts, **kwargs):
        if self.is_3d:
            xs, ys, zs = self._coords(pts)
            self.ax.scatter(xs, ys, zs, **kwargs)
        else:
            xs, ys = self._coords(pts)
            self.ax.scatter(xs, ys, **kwargs)

    def _annotate(self, pt, text, dx, dy, dz, color):
        if self.is_3d:
            self.ax.text(pt[0]+dx, pt[1]+dy, pt[2]+dz,
                         text, fontsize=8, color=color,
                         fontweight='bold')
        else:
            self.ax.annotate(text,
                             xy=(pt[0], pt[1]),
                             xytext=(pt[0]+dx, pt[1]+dy),
                             fontsize=8.5, color=color,
                             fontweight='bold')

    # -- Original 32-bezier-scipy drawing methods ------------

    def _draw_curve(self):
        pts = self.bezier.curve()
        self._plot_line(pts, color=self.COLOR_CURVE, linewidth=2.5,
                        label='Bezier Curve', zorder=3)

    def _draw_control_polygon(self):
        self._plot_line(self.bezier.points,
                        color=self.COLOR_POLYGON, linestyle='--',
                        linewidth=1.5, marker='o', markersize=8,
                        label='Control Polygon', zorder=4)

    def _label_control_points(self):
        for i, pt in enumerate(self.bezier.points):
            coords = ', '.join(f'{v:.4g}' for v in pt)
            name   = f'P{i}({coords})'
            dy     = 0.15 if i % 2 == 0 else -0.28
            self._annotate(pt, name, dx=0.10, dy=dy, dz=0.05,
                           color=self.COLOR_CTRL)

    def _draw_highlight_points(self):
        pts = self.bezier.highlight_points(self.u_highlight)
        self._scatter_pts(pts, color=self.COLOR_POINT,
                          s=60, zorder=5, label='Points at u')
        for u, pt in zip(self.u_highlight, pts):
            coords = ', '.join(f'{v:.3g}' for v in pt)
            dx = -0.55 if u == 1.0 else 0.08
            dy = -0.28 if u in (0.0, 1.0) else 0.10
            if self.is_3d:
                self.ax.text(pt[0]+0.08, pt[1]+0.08, pt[2]+0.05,
                             f'u={u:.2f}', fontsize=7,
                             color=self.COLOR_LABEL)
            else:
                self.ax.annotate(
                    f'u={u:.2f}\n({coords})',
                    xy=(pt[0], pt[1]),
                    xytext=(pt[0]+dx, pt[1]+dy),
                    fontsize=7.5, color=self.COLOR_LABEL,
                    bbox=dict(boxstyle='round,pad=0.2',
                              fc='white', alpha=0.65, ec='none'),
                )

    def _draw_de_casteljau(self):
        levels = self.bezier.de_casteljau(self.u_dc)
        for lv in range(1, len(levels) - 1):
            pts = levels[lv]
            col = self.COLOR_DC[min(lv-1, len(self.COLOR_DC)-1)]
            self._plot_line(pts, color=col, linewidth=1.4,
                            marker='o', markersize=6, zorder=6,
                            label=f'DC level {lv}')
        final = levels[-1]
        col   = self.COLOR_DC[min(len(levels)-2, len(self.COLOR_DC)-1)]
        self._scatter_pts(final, color=col, marker='D',
                          s=90, zorder=7,
                          label=f'DC final  u={self.u_dc:.2f}')

    def _add_equation_box(self):
        n     = self.bezier.degree
        terms = ' + '.join(
            f'B({n},{i})\u00b7P{i}' for i in range(self.bezier.n)
        )
        text = (f'P(u) = {terms}\n'
                f'B(n,i,u) = C(n,i)\u00b7u\u1d35\u00b7(1-u)^(n-i)\n'
                f'[evaluated via scipy.interpolate.BPoly]')
        if self.is_3d:
            self.fig.text(0.13, 0.92, text, fontsize=8,
                          fontfamily='monospace', va='top',
                          bbox=dict(boxstyle='round,pad=0.4',
                                    facecolor='lightyellow', alpha=0.85))
        else:
            self.ax.text(0.02, 0.97, text,
                         transform=self.ax.transAxes,
                         fontsize=8, fontfamily='monospace', va='top',
                         bbox=dict(boxstyle='round,pad=0.4',
                                   facecolor='lightyellow', alpha=0.85))

    # -- Curvature comb (new) --------------------------------

    def _draw_comb(self):
        """
        Perpendicular spikes along the curve,
        length proportional to kappa at each point.

        2D: normal = unit tangent rotated 90 degrees.
        3D: principal normal via N = (P' x P'') x P',
            normalized — points toward centre of curvature
            in full 3D space. Spikes drawn as 3D lines.
        """
        step  = max(1, self.n_pts // 60)
        u_sub = self.u_vals[::step]

        d1v   = self.bezier.d1(u_sub)   # (m, dim)
        d2v   = self.bezier.d2(u_sub)   # (m, dim)
        base  = self.bezier._bpoly(u_sub)  # (m, dim)
        kap_s = self.bezier.curvature(u_sub)

        if not self.is_3d:
            # 2D: rotate tangent 90 degrees
            speed = np.linalg.norm(d1v, axis=1, keepdims=True)
            speed = np.where(speed > 1e-10, speed, 1e-10)
            T = d1v / speed
            N = np.stack([-T[:, 1], T[:, 0]], axis=1)

            tip = base + self.comb_scale * kap_s[:, np.newaxis] * N

            for b, t in zip(base, tip):
                self.ax.plot([b[0], t[0]], [b[1], t[1]],
                             color=self.COLOR_COMB, alpha=0.65,
                             linewidth=1.2, zorder=2)
            self.ax.plot(tip[:, 0], tip[:, 1],
                         color=self.COLOR_COMB, alpha=0.5,
                         linewidth=1.0, linestyle='--', zorder=2,
                         label='Curvature comb')
        else:
            # 3D: principal normal N = (P' x P'') x P'
            cross_d1_d2 = np.cross(d1v, d2v)          # binormal direction
            N_raw       = np.cross(cross_d1_d2, d1v)  # principal normal

            norm = np.linalg.norm(N_raw, axis=1, keepdims=True)
            norm = np.where(norm > 1e-10, norm, 1e-10)
            N    = N_raw / norm

            tip = base + self.comb_scale * kap_s[:, np.newaxis] * N

            for b, t in zip(base, tip):
                self.ax.plot([b[0], t[0]], [b[1], t[1]], [b[2], t[2]],
                             color=self.COLOR_COMB, alpha=0.65,
                             linewidth=1.2)
            self.ax.plot(tip[:, 0], tip[:, 1], tip[:, 2],
                         color=self.COLOR_COMB, alpha=0.5,
                         linewidth=1.0, linestyle='--',
                         label='Curvature comb')

    # -- Decorations -----------------------------------------

    def _set_decorations(self):
        title = (f'Bezier Curve + Curvature Comb: '
                 f'{self.bezier.degree_name} '
                 f'({self.bezier.dim_name}, '
                 f'{self.bezier.n} Control Points)')
        self.ax.set_title(title, fontsize=13,
                          fontweight='bold', pad=14)
        self.ax.set_xlabel('x', fontsize=11)
        self.ax.set_ylabel('y', fontsize=11)
        if self.is_3d:
            self.ax.set_zlabel('z', fontsize=11)
        self.ax.legend(fontsize=8, loc='lower right')

        if not self.is_3d:
            pts   = self.bezier.points
            curve = self.bezier.curve()
            all_x = np.concatenate([pts[:, 0], curve[:, 0]])
            all_y = np.concatenate([pts[:, 1], curve[:, 1]])
            m = 0.8
            self.ax.set_xlim(all_x.min()-m, all_x.max()+m)
            self.ax.set_ylim(all_y.min()-m, all_y.max()+m)

    # -- Main ------------------------------------------------

    def draw(self):
        self._draw_comb()             # draw first so curve sits on top
        self._draw_curve()
        self._draw_control_polygon()
        self._label_control_points()
        self._draw_highlight_points()
        self._draw_de_casteljau()
        self._add_equation_box()
        self._set_decorations()

    def save(self, filename):
        self.fig.tight_layout()
        self.fig.savefig(filename, dpi=150)
        print(f'Saved: {filename}')

    def show(self):
        self.fig.tight_layout()
        plt.show()


# ============================================================
# -- Entry Point ---------------------------------------------
# ============================================================

def main() -> int:
    data   = CASE_DATA[CASE]
    bezier = BezierCurve(data['points'])
    plot   = BezierPlot(
        bezier,
        u_highlight = data['u_highlight'],
        u_dc        = data['u_dc'],
    )
    plot.draw()
    plot.save(data['output'])
    plot.show()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
