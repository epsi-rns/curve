import sympy as sp
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
        points   = [
            [1, 1],   # P0
            [3, 3],   # P1
            [5, 1],   # P2
        ],
        u_values = [0, sp.Rational(1,4), sp.Rational(1,2),
                       sp.Rational(3,4), 1],
        axes     = ['x', 'y'],
    ),
    2: dict(
        points   = [
            [0, 0, 1],   # P0
            [0, 4, 1],   # P1
            [4, 0, 1],   # P2
            [4, 4, 1],   # P3
            [5, 4, 1],   # P4
        ],
        u_values = [0, sp.Rational(1,4), sp.Rational(1,2),
                       sp.Rational(3,4), 1],
        axes     = ['x', 'y', 'z'],
    ),
}


# ============================================================
# -- Bezier Matrix  (generic — do not change) ----------------
# ============================================================

class BezierMatrix:
    """
    Evaluates a Bezier curve of any degree via P(u) = U · M · G
    using SymPy matrices throughout — exact rational arithmetic,
    no floating-point error.

    SymPy replaces numpy here:

      _derive_M()
        numpy zeros + manual comb()  →  sp.Matrix built directly
        from sp.binomial(), with sp.Rational entries.
        All coefficients stay as exact integers.

      G
        numpy array of floats  →  sp.Matrix of sp.Rational,
        so matrix products stay exact end-to-end.

      MG  =  M · G
        numpy @  →  sp.Matrix.__mul__  (exact rational result)

      evaluate_all()
        numpy Vandermonde loop  →  sp.Matrix of u^k powers
        using sp.Rational u_values, giving exact results.

    M[n-k, i]  =  C(n,i) · C(n-i, k-i) · (-1)^(k-i)
                  for k >= i,  else 0
    Row 0 = highest power u^n,  last row = constant u^0.
    """

    u = sp.Symbol('u')   # shared SymPy symbol

    def __init__(self, points: list, axes: list):
        self.axes   = axes
        self.n      = len(points)
        self.degree = self.n - 1
        self.dim    = len(axes)

        # G — geometry matrix, exact rational entries
        self.G  = sp.Matrix([
            [sp.Rational(c) for c in row]
            for row in points
        ])

        # M — basis matrix derived from Bernstein
        self.M  = self._derive_M()

        # M·G — polynomial coefficient matrix, computed once
        self.MG = self.M * self.G

    def _derive_M(self) -> sp.Matrix:
        """
        Derive the Bezier basis matrix M of shape (n+1, n+1).

        SymPy's sp.binomial() computes C(n,i) exactly as an
        integer — no manual comb() needed.  The result is a
        sp.Matrix with exact integer entries.
        """
        n    = self.degree
        size = n + 1
        M    = sp.zeros(size, size)

        for i in range(size):
            m = n - i
            for j in range(m + 1):
                k        = i + j
                M[n-k, i] = (sp.binomial(n, i)
                              * sp.binomial(m, j)
                              * (-1)**j)
        return M

    def U(self, u_val) -> sp.Matrix:
        """
        Parameter row vector  [u^n,  u^(n-1),  ...,  u,  1]
        as a (1 × n+1) SymPy matrix.
        u_val may be a sp.Rational or plain integer.
        """
        n = self.degree
        return sp.Matrix([[u_val ** (n - k)
                           for k in range(n + 1)]])

    def evaluate_all(self, u_values: list) -> sp.Matrix:
        """
        Evaluate at all u values in one batched matrix multiply.

        Builds U_all of shape (m, n+1) where each row is the
        power vector for one u value, then:
          P  =  U_all · M · G  =  U_all · MG

        All arithmetic stays in SymPy — results are exact
        rationals, not floats.
        """
        U_all = sp.Matrix([
            [u_val ** (self.degree - k)
             for k in range(self.degree + 1)]
            for u_val in u_values
        ])
        return U_all * self.MG   # (m, n+1) · (n+1, dim) → (m, dim)


# ============================================================
# -- Console Printer  (generic — do not change) --------------
# ============================================================

class MatrixPrinter:
    """
    Prints G, M, M·G, polynomials, and evaluation table
    using tabulate for all layout and SymPy for all math.

    What SymPy provides vs the numpy+tabulate version:
      _derive_M      → sp.Matrix with sp.binomial (exact int)
      MG             → sp.Matrix product  (exact rational)
      evaluate_all   → sp.Matrix product  (exact rational)
      _fmt_poly      → sp.Poly.as_expr()  (auto-formatted)
      cell values    → str(sp.Rational)   (exact fractions)
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

    def _fmt(self, v) -> str:
        """
        Format a SymPy number.
        Integers print without denominator, rationals as p/q.
        str() on a SymPy Rational already does the right thing.
        """
        return str(v)

    def _fmt_poly(self, col: int) -> str:
        """
        Build the polynomial string for one axis column of MG.

        MG[:, col] gives the monomial coefficients ordered
        from highest power (row 0 = u^n) to constant (last row).

        sp.Poly needs them in the same order SymPy expects —
        highest power first — which is exactly what MG provides.

        sp.Poly(coeffs, u).as_expr() returns a clean SymPy
        expression; str() renders it in standard notation.
        SymPy handles sign, zero suppression, and coefficient-
        of-1 automatically — no manual formatting needed.
        """
        n      = self.bm.degree
        coeffs = [self.bm.MG[k, col] for k in range(n + 1)]
        expr   = sp.Poly(coeffs, self.bm.u).as_expr()
        return str(expr)

    def _mat_rows(self, mat: sp.Matrix, row_labels: list) -> list:
        """
        Convert a SymPy matrix to list-of-lists for tabulate,
        with row labels prepended as the first column.
        str() on each SymPy entry gives exact fractions.
        """
        return [
            [rl] + [self._fmt(mat[r, c])
                    for c in range(mat.shape[1])]
            for r, rl in enumerate(row_labels)
        ]

    def _print_mat(
        self,
        mat:        sp.Matrix,
        row_labels: list,
        col_labels: list,
    ):
        """Print a labelled SymPy matrix using tabulate."""
        rows = self._mat_rows(mat, row_labels)
        print(tabulate(
            rows,
            headers  = [''] + col_labels,
            tablefmt = 'simple',
            colalign = ('left',) + ('right',) * mat.shape[1],
        ))

    # -- Print -----------------------------------------------

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
        print(f'  Bezier Curve — Matrix Method  '
              f'[P(u) = U · M · G]  (SymPy + tabulate)')
        print(f'  Case {self.case_num}  |  {dname}  |  '
              f'{self.bm.dim}D  |  {self.bm.n} Control Points')
        print(self._ruler())

    def _print_G(self):
        self._section('G  — Geometry Matrix (Control Points)')
        print()
        self._print_mat(
            self.bm.G,
            row_labels = [f'P{i}' for i in range(self.bm.n)],
            col_labels = self.bm.axes,
        )

    def _print_M(self):
        self._section('M  — Basis Matrix (derived via sp.binomial)')
        n = self.bm.degree
        print()
        self._print_mat(
            self.bm.M,
            row_labels = [f'u^{n-k}' for k in range(self.bm.n)],
            col_labels = [f'P{i}'    for i in range(self.bm.n)],
        )

    def _print_MG(self):
        self._section('M · G  — Polynomial Coefficient Matrix')
        n = self.bm.degree
        print()
        self._print_mat(
            self.bm.MG,
            row_labels = [f'u^{n-k}' for k in range(self.bm.n)],
            col_labels = self.bm.axes,
        )

    def _print_polynomials(self):
        self._section('Resulting Polynomials  (via sp.Poly)')
        print()
        for d, axis in enumerate(self.bm.axes):
            print(f'  {axis}(u)  =  {self._fmt_poly(d)}')

    def _print_table(self):
        self._section('Evaluation Table  —  P(u) = U · (M · G)')

        # One batched SymPy matrix multiply — exact rationals
        pts  = self.bm.evaluate_all(self.u_values)
        rows = []
        for r, u_val in enumerate(self.u_values):
            row = [self._fmt(u_val)]
            for c in range(self.bm.dim):
                exact = pts[r, c]
                dec   = f'({float(exact):.4f})'
                row.append(f'{self._fmt(exact)}  {dec}')
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
