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
# -- Matrix Derivation Engine  (generic — do not change) -----
# ============================================================

class BezierMatrix:
    """
    Derives and applies the Bezier basis matrix M for a
    curve of any degree, then evaluates P(u) = U · M · G.

    G  (Geometry matrix)  shape (n, dim)
       Control points stacked as rows.

    M  (Basis matrix)     shape (n, n)
       Entry M[k, i] is the coefficient of u^(n-k) in B(n-1, i, u).
       Derived automatically — no hardcoding for specific degrees.

    U  (Parameter vector) shape (1, n)
       Power basis row: [u^n, u^(n-1), ..., u, 1]

    Derivation of M[k, i]:
       B(n,i,u) = C(n,i) · u^i · (1-u)^(n-i)
       Expanding (1-u)^m via binomial theorem and collecting
       powers gives:
         M[n-k, i] = C(n,i) · C(n-i, k-i) · (-1)^(k-i)
                     for k >= i, else 0
    """

    def __init__(self, points: np.ndarray, axes: list):
        self.G      = points
        self.axes   = axes
        self.n      = len(points)
        self.degree = self.n - 1
        self.dim    = len(axes)
        self.M      = self._derive_M()
        self.MG     = self.M @ self.G

    def _derive_M(self) -> np.ndarray:
        n = self.degree
        M = np.zeros((n + 1, n + 1), dtype=float)
        for i in range(n + 1):
            m = n - i
            for j in range(m + 1):
                k         = i + j
                M[n-k, i] = comb(n, i) * comb(m, j) * ((-1) ** j)
        return M

    def _U(self, u: float) -> np.ndarray:
        """Parameter row vector [u^n, u^(n-1), ..., u, 1]."""
        n = self.degree
        return np.array([[u ** (n - k) for k in range(n + 1)]])

    def evaluate(self, u: float) -> np.ndarray:
        """P(u) = U(u) · M · G.  Returns shape (dim,)."""
        return (self._U(u) @ self.MG)[0]

    def evaluate_steps(self, u: float) -> dict:
        """Return U and final point for step-by-step display."""
        U  = self._U(u)
        pt = (U @ self.MG)[0]
        return dict(U=U, point=pt)

    def evaluate_all(self, u_values: list) -> np.ndarray:
        """Evaluate at all u values in one batched multiply."""
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
    Prints the full pedagogical matrix-method derivation
    and evaluation of a Bezier curve to the console.

    Steps shown:
      1. Define matrices G, M, U  — with actual values
      2. Derive M  — column-by-column from Bernstein
      3. Compute M · G  — polynomial coefficient matrix
      4. Evaluate U · (M · G) at each u — dot product shown
      5. Summary evaluation table

    tabulate handles all matrix and table layout.
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

    def _step(self, n: int, title: str):
        print()
        print(f'  STEP {n} — {title}')
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
        Build polynomial string from a M·G column.
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

    def _mat_rows(
        self,
        mat:        np.ndarray,
        row_labels: list,
        fmt_fn     = None,
    ) -> list:
        """
        Convert numpy matrix to list-of-lists for tabulate,
        with row labels prepended as the first column.
        """
        if fmt_fn is None:
            fmt_fn = self._fmt
        return [
            [rl] + [fmt_fn(mat[r, c]) for c in range(mat.shape[1])]
            for r, rl in enumerate(row_labels)
        ]

    def _print_mat(
        self,
        mat:        np.ndarray,
        row_labels: list,
        col_labels: list,
        fmt_fn     = None,
    ):
        """Print a labelled matrix using tabulate 'simple' style."""
        rows = self._mat_rows(mat, row_labels, fmt_fn)
        print(tabulate(
            rows,
            headers  = [''] + col_labels,
            tablefmt = 'simple',
            colalign = ('left',) + ('right',) * mat.shape[1],
        ))

    # -- Print all -------------------------------------------

    def print_all(self):
        self._print_title()
        self._print_control_points()
        self._print_step1_define()
        self._print_step2_derive_M()
        self._print_step3_MG()
        self._print_step4_evaluate()
        self._print_step5_table()
        print()
        print(self._ruler())
        print()

    # -- Title and control points ----------------------------

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

    def _print_control_points(self):
        print()
        print('  Control Points:')
        for i, pt in enumerate(self.bm.G):
            coords = '  '.join(
                f'{ax} = {self._fmt(pt[j])}'
                for j, ax in enumerate(self.bm.axes)
            )
            print(f'    P{i} :  {coords}')

    # -- Step 1: define matrices -----------------------------

    def _print_step1_define(self):
        self._step(1, 'Matrix Formulation:  P(u) = U · M · G')
        n   = self.bm.degree
        dim = self.bm.dim

        print()
        print('  P(u)  =  U(u)  ·  M  ·  G')
        print()
        print(f'  U(u)  shape (1×{n+1})  '
              f'parameter vector  [u^{n}, ..., u, 1]')
        print(f'  M     shape ({n+1}×{n+1})  '
              f'basis matrix  (derived from Bernstein)')
        print(f'  G     shape ({n+1}×{dim})  '
              f'geometry matrix  (control points)')

        # G via tabulate
        print()
        self._print_mat(
            self.bm.G,
            row_labels = [f'P{i}' for i in range(self.bm.n)],
            col_labels = self.bm.axes,
        )

    # -- Step 2: derive M ------------------------------------

    def _print_step2_derive_M(self):
        self._step(2, 'Derive Basis Matrix M from Bernstein Polynomials')
        n   = self.bm.degree
        sup = {0:'', 1:'', 2:'²', 3:'³', 4:'⁴', 5:'⁵'}

        print()
        print('  Each column i of M holds the monomial coefficients')
        print(f'  of the Bernstein basis polynomial B({n}, i, u):')
        print()
        print('    B(n,i,u)  =  C(n,i) · u^i · (1-u)^(n-i)')
        print()
        print('  Expanding (1-u)^(n-i) via the binomial theorem:')
        print('    (1-u)^m  =  Σ  C(m,j) · (-1)^j · u^j')
        print()
        print('  Coefficient of u^k in B(n,i,u):')
        print('    M[n-k, i]  =  C(n,i) · C(n-i, k-i) · (-1)^(k-i)'
              '   for k >= i, else 0')
        print()
        print(f'  Row ordering: row 0 = u^{n} (highest),'
              f'  row {n} = u^0 (constant)')
        print()

        # Column-by-column derivation
        print('  Column-by-column derivation:')
        print()
        for i in range(n + 1):
            m   = n - i
            b   = '' if comb(n, i) == 1 else str(comb(n, i))
            u_p = ('' if i == 0 else 'u' if i == 1
                   else f'u{sup.get(i, f"^{i}")}')
            v_p = ('' if m == 0 else '(1-u)' if m == 1
                   else f'(1-u){sup.get(m, f"^{m}")}')
            print(f'  Column {i}  →  B({n},{i},u)  =  '
                  f'{b}{u_p}{v_p}')

            # Coefficients as a small tabulate table
            for k in range(n + 1):
                power = n - k
                c     = self.bm.M[k, i]
                if abs(c) < 1e-9:
                    continue
                u_str = ('1' if power == 0 else 'u' if power == 1
                         else f'u^{power}')
                print(f'    M[{k},{i}]  =  {int(c):>4}'
                      f'   (coeff of {u_str})')
            print()

        # Full M matrix
        print('  Full M matrix:')
        print()
        self._print_mat(
            self.bm.M,
            row_labels = [f'u^{n-k}' for k in range(self.bm.n)],
            col_labels = [f'P{i}' for i in range(self.bm.n)],
        )

    # -- Step 3: M · G ---------------------------------------

    def _print_step3_MG(self):
        self._step(3, 'Compute  M · G  — Polynomial Coefficient Matrix')
        n = self.bm.degree

        print()
        print('  M · G  gives the monomial coefficients for each axis.')
        print('  Row k of (M·G) is the coefficient of u^(n-k).')
        print()

        # M·G as tabulate table
        self._print_mat(
            self.bm.MG,
            row_labels = [f'u^{n-k}' for k in range(self.bm.n)],
            col_labels = self.bm.axes,
        )

        # Resulting polynomials
        print()
        print('  Resulting polynomials:')
        for d, axis in enumerate(self.bm.axes):
            print(f'    {axis}(u)  =  {self._fmt_poly(self.bm.MG[:, d])}')

    # -- Step 4: evaluate at each u --------------------------

    def _print_step4_evaluate(self):
        self._step(4, 'Evaluate  U · (M · G)  at Each u Value')
        n = self.bm.degree

        for u in self.u_values:
            steps = self.bm.evaluate_steps(u)
            U     = steps['U']
            pt    = steps['point']

            print()
            print(f'  ── u = {u} ' + '─' * (self.WIDTH - 10 - len(str(u))))

            # U vector as a single-row tabulate table
            u_headers = [f'u^{n-k}' for k in range(n + 1)]
            u_vals    = [[self._fmt_frac(U[0, k])
                          for k in range(n + 1)]]
            print()
            print('  U  =')
            print(tabulate(
                u_vals,
                headers  = u_headers,
                tablefmt = 'simple',
                colalign = ('right',) * (n + 1),
            ))

            # Dot product U · (M·G) per axis
            print()
            print('  U · (M·G)  =  P(u):')
            for d, axis in enumerate(self.bm.axes):
                mg_col    = self.bm.MG[:, d]
                dot_terms = '  +  '.join(
                    f'{self._fmt_frac(U[0,k])}·({self._fmt(mg_col[k])})'
                    for k in range(n + 1)
                    if abs(mg_col[k]) > 1e-9 or abs(U[0, k]) > 1e-9
                )
                print(f'    {axis}(u)  =  {dot_terms}')
                print(f'           =  {self._fmt_frac(pt[d])}')

    # -- Step 5: summary table -------------------------------

    def _print_step5_table(self):
        self._step(5, 'Summary — Evaluation Table')

        pts  = self.bm.evaluate_all(self.u_values)
        rows = []
        for u, pt in zip(self.u_values, pts):
            row = [self._fmt(u)]
            for v in pt:
                row.append(f'{self._fmt_frac(v)}  ({v:.4f})')
            rows.append(row)

        headers = ['u'] + [f'{ax}(u)' for ax in self.bm.axes]
        print()
        print(tabulate(
            rows,
            headers  = headers,
            tablefmt = 'outline',
            colalign = ('right',) * (self.bm.dim + 1),
        ))


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
