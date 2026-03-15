import numpy as np
from math import comb
from fractions import Fraction
from tabulate import tabulate


# ============================================================
# -- Switch  (only change this one line) ---------------------
# ============================================================

CASE = 2   # 1 = Quadratic 2D,  2 = Quartic 3D


# ============================================================
# -- Case Data  (do not change) ------------------------------
# ============================================================

CASE_DATA = {
    1: dict(
        points   = np.array([
            [1, 1],   # P0
            [3, 3],   # P1
            [5, 1],   # P2
        ], dtype=float),
        u_values = [0.00, 0.25, 0.50, 0.75, 1.00],
        axes     = ['x', 'y'],
    ),
    2: dict(
        points   = np.array([
            [0, 0, 1],   # P0
            [0, 4, 1],   # P1
            [4, 0, 1],   # P2
            [4, 4, 1],   # P3
            [5, 4, 1],   # P4
        ], dtype=float),
        u_values = [0.00, 0.25, 0.50, 0.75, 1.00],
        axes     = ['x', 'y', 'z'],
    ),
}


# ============================================================
# -- Bezier Matrix  (generic — do not change) ----------------
# ============================================================

class BezierMatrix:
    """
    Evaluates a Bezier curve of any degree via P(u) = U · M · G.

    M is derived automatically from the Bernstein definition.
    All evaluation is pure matrix multiplication — no loops
    over control points or basis functions at query time.

    M[k, i] = C(n,i) · C(n-i, k-i) · (-1)^(k-i)   for k >= i
            = 0                                       for k <  i

    Row k corresponds to power u^(n-k), so row 0 is u^n
    and the last row is the constant term u^0.
    This matches U = [u^n, u^(n-1), ..., u, 1].
    """

    def __init__(self, points: np.ndarray, axes: list):
        self.G      = points
        self.axes   = axes
        self.n      = len(points)
        self.degree = self.n - 1
        self.dim    = len(axes)
        self.M      = self._derive_M()
        self.MG     = self.M @ self.G   # pre-computed once

    def _derive_M(self) -> np.ndarray:
        n = self.degree
        M = np.zeros((n + 1, n + 1))
        for i in range(n + 1):
            m = n - i
            for j in range(m + 1):
                k         = i + j
                M[n-k, i] = comb(n, i) * comb(m, j) * ((-1) ** j)
        return M

    def U(self, u: float) -> np.ndarray:
        """Parameter row vector [u^n, u^(n-1), ..., u, 1]."""
        n = self.degree
        return np.array([[u ** (n - k) for k in range(n + 1)]])

    def evaluate_all(self, u_values: list) -> np.ndarray:
        """
        Evaluate at all u values in one batched matrix multiply.
        U_all shape (m, n+1)  ·  MG shape (n+1, dim)  →  (m, dim)
        """
        n     = self.degree
        U_all = np.array([
            [u ** (n - k) for k in range(n + 1)]
            for u in u_values
        ])
        return U_all @ self.MG


# ============================================================
# -- Console Printer  (generic — do not change) --------------
# ============================================================

class MatrixPrinter:
    """
    Prints G, M, M·G, polynomials, and evaluation table
    using tabulate for all matrix and table formatting.

    What tabulate replaces vs the previous version:
      _print_matrix  → tabulate() with 'simple' table format
      _print_table   → tabulate() with 'outline' table format
      _fmt / _fmt_fraction → kept only for polynomial strings
                             and cell values inside tabulate
    """

    WIDTH = 66

    def __init__(
        self,
        bm:       BezierMatrix,
        u_values: list,
        case_num: int,
    ):
        self.bm       = bm
        self.u_values = u_values
        self.case_num = case_num

    # -- Formatting helpers ----------------------------------

    def _ruler(self, char: str = '=') -> str:
        return char * self.WIDTH

    def _section(self, title: str):
        print()
        print(f'  {title}')
        print('  ' + '-' * (self.WIDTH - 2))

    def _fmt(self, v: float) -> str:
        """Integer if whole, else 4 decimal places."""
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return f'{v:.4f}'

    def _fmt_frac(self, v: float) -> str:
        """Exact fraction when possible, else 4 decimal places."""
        f = Fraction(v).limit_denominator(1024)
        if abs(float(f) - v) < 1e-9:
            return str(f) if f.denominator != 1 else str(f.numerator)
        return f'{v:.4f}'

    def _fmt_poly(self, coeffs: np.ndarray) -> str:
        """
        Build a polynomial string from a M·G column.
        coeffs[k]  =  coefficient of  u^(degree - k).
        """
        n   = self.bm.degree
        sup = {0:'', 1:'', 2:'²', 3:'³', 4:'⁴', 5:'⁵'}
        terms = []
        for k, c in enumerate(coeffs):
            if abs(c) < 1e-9:
                continue
            power = n - k
            u_s   = ('' if power == 0 else
                     'u' if power == 1 else
                     f'u{sup.get(power, f"^{power}")}')
            c_s   = self._fmt(abs(c))
            if c_s == '1' and u_s:
                c_s = ''
            terms.append((c, c_s + u_s))
        if not terms:
            return '0'
        parts = []
        for idx, (c, t) in enumerate(terms):
            parts.append(
                (f'-{t}' if c < 0 else t) if idx == 0
                else (f'- {t}' if c < 0 else f'+ {t}')
            )
        return '  '.join(parts)

    def _mat_to_table(
        self,
        mat:        np.ndarray,
        row_labels: list,
        col_labels: list,
        fmt_fn     = None,
    ) -> list:
        """
        Convert a numpy matrix to a list-of-lists suitable
        for tabulate, prepending row labels as the first column.
        fmt_fn defaults to self._fmt if not provided.
        """
        if fmt_fn is None:
            fmt_fn = self._fmt
        rows = []
        for r, rl in enumerate(row_labels):
            rows.append(
                [rl] + [fmt_fn(mat[r, c])
                        for c in range(mat.shape[1])]
            )
        return rows

    # -- Print sections --------------------------------------

    def print_all(self):
        self._print_title()
        self._print_G()
        self._print_M()
        self._print_MG()
        self._print_polynomials()
        self._print_table()
        print()
        print(self._ruler())
        print()

    def _print_title(self):
        names = {2:'Quadratic', 3:'Cubic',
                 4:'Quartic',   5:'Quintic'}
        dname = names.get(self.bm.degree, f'Degree-{self.bm.degree}')
        print()
        print(self._ruler())
        print(f'  Bezier Curve — Matrix Method  [P(u) = U · M · G]')
        print(f'  Case {self.case_num}  |  {dname}  |  '
              f'{self.bm.dim}D  |  {self.bm.n} Control Points')
        print(self._ruler())

    def _print_G(self):
        self._section('G  — Geometry Matrix (Control Points)')
        n = self.bm.degree
        rows    = self._mat_to_table(
            self.bm.G,
            row_labels = [f'P{i}' for i in range(self.bm.n)],
            col_labels = self.bm.axes,
        )
        print()
        print(tabulate(rows,
                       headers   = [''] + self.bm.axes,
                       tablefmt  = 'simple',
                       colalign  = ('left',)
                                   + ('right',) * self.bm.dim))

    def _print_M(self):
        self._section('M  — Basis Matrix (derived from Bernstein)')
        n       = self.bm.degree
        r_lbls  = [f'u^{n-k}' for k in range(self.bm.n)]
        c_lbls  = [f'P{i}'    for i in range(self.bm.n)]
        rows    = self._mat_to_table(self.bm.M, r_lbls, c_lbls)
        print()
        print(tabulate(rows,
                       headers  = [''] + c_lbls,
                       tablefmt = 'simple',
                       colalign = ('left',)
                                  + ('right',) * self.bm.n))

    def _print_MG(self):
        self._section('M · G  — Polynomial Coefficient Matrix')
        n      = self.bm.degree
        r_lbls = [f'u^{n-k}' for k in range(self.bm.n)]
        rows   = self._mat_to_table(self.bm.MG, r_lbls, self.bm.axes)
        print()
        print(tabulate(rows,
                       headers  = [''] + self.bm.axes,
                       tablefmt = 'simple',
                       colalign = ('left',)
                                  + ('right',) * self.bm.dim))

    def _print_polynomials(self):
        self._section('Resulting Polynomials')
        print()
        for d, axis in enumerate(self.bm.axes):
            poly = self._fmt_poly(self.bm.MG[:, d])
            print(f'  {axis}(u)  =  {poly}')

    def _print_table(self):
        self._section('Evaluation Table  —  P(u) = U · (M · G)')

        pts  = self.bm.evaluate_all(self.u_values)
        rows = []
        for u, pt in zip(self.u_values, pts):
            row = [self._fmt(u)]
            for v in pt:
                # Show exact fraction + decimal in same cell
                row.append(f'{self._fmt_frac(v)}  ({v:.4f})')
            rows.append(row)

        headers = ['u'] + [f'{ax}(u)' for ax in self.bm.axes]
        print()
        print(tabulate(rows,
                       headers  = headers,
                       tablefmt = 'outline',
                       colalign = ('right',) * (self.bm.dim + 1)))


# ============================================================
# -- Entry Point ---------------------------------------------
# ============================================================

def main() -> int:
    data = CASE_DATA[CASE]

    bm = BezierMatrix(
        points = data['points'],
        axes   = data['axes'],
    )

    printer = MatrixPrinter(
        bm       = bm,
        u_values = data['u_values'],
        case_num = CASE,
    )

    printer.print_all()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
