from fractions import Fraction
import sympy as sp


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
# -- Bezier Expansion Engine  (generic — do not change) ------
# ============================================================

class BezierExpansion:
    """
    Symbolic algebraic expansion of a Bezier curve of any
    degree and any spatial dimension, powered by SymPy.

    SymPy replaces the manual Poly and BernsteinExpander
    classes from the pure-math version:

      BernsteinExpander.expand(n, i)
        → sp.expand(sp.binomial(n,i) * u**i * (1-u)**(n-i))
          SymPy handles binomial theorem, sign expansion,
          and collection of like powers automatically.

      Poly weighted sum + _build_polys()
        → sum of SymPy expressions, collected with sp.Poly
          SymPy tracks every coefficient exactly as a
          rational number — no floating-point rounding.

      Poly.__call__(u_val)  and  lambdify
        → sp.Poly.eval(u_val)  or  expr.subs(u, val)
          SymPy substitutes exact rational values and
          simplifies, returning exact results.

    De Casteljau stays as pure math — SymPy has no
    built-in De Casteljau algorithm, and the step-by-step
    reduction is the pedagogically essential part.
    """

    # SymPy symbol for the parameter
    u = sp.Symbol('u')

    def __init__(self, points: list, axes: list):
        # Store as SymPy rational numbers — exact arithmetic
        self.points  = [
            [sp.Rational(c) for c in row]
            for row in points
        ]
        self.axes    = axes
        self.n       = len(points)      # number of control pts
        self.degree  = self.n - 1      # polynomial degree
        self.dim     = len(axes)        # spatial dimension

        # ── Bernstein basis expressions ──────────────────────
        # Each basis[i] is a SymPy expression in u,
        # fully expanded into monomial form.
        #
        # B(n, i, u) = C(n,i) · u^i · (1-u)^(n-i)
        #
        # sp.expand() applies the binomial theorem to
        # (1-u)^(n-i), distributes C(n,i) and u^i, and
        # collects all powers — in one call, no manual steps.
        self.basis: list = [
            sp.expand(
                sp.binomial(self.degree, i)
                * self.u**i
                * (1 - self.u)**(self.degree - i)
            )
            for i in range(self.n)
        ]

        # ── Per-axis collected polynomials ───────────────────
        # For each axis, compute the Bernstein weighted sum:
        #   P_axis(u) = sum_{i=0}^{n}  basis[i] * coord_i
        #
        # sp.Poly(..., u) collects all terms by power of u,
        # giving a clean coefficient list in descending order.
        # .as_expr() converts back to a printable expression.
        self.polys: dict = {}
        self.sympy_exprs: dict = {}   # raw SymPy expr per axis

        for dim_idx, axis in enumerate(axes):
            expr = sum(
                self.basis[i] * self.points[i][dim_idx]
                for i in range(self.n)
            )
            # sp.simplify + sp.collect ensures all like powers
            # of u are combined into a single clean expression
            collected        = sp.collect(sp.expand(expr), self.u)
            self.sympy_exprs[axis] = collected
            self.polys[axis] = sp.Poly(collected, self.u)

    def evaluate(self, u_val) -> list:
        """
        Evaluate all axis expressions at u_val.
        u_val can be a SymPy Rational or plain int/float.
        Returns exact SymPy values (Rational or Integer).
        """
        return [
            self.sympy_exprs[ax].subs(self.u, u_val)
            for ax in self.axes
        ]


# ============================================================
# -- Console Printer  (generic — do not change) --------------
# ============================================================

class AlgebraicPrinter:
    """
    Prints the full pedagogical step-by-step algebraic
    expansion of a Bezier curve to the console.

    Uses SymPy's expression rendering for all math output,
    so intermediate and final forms are always exact.

    Steps shown:
      1. Bernstein form       — symbolic B(n,i,u) definitions
      2. Substitution         — coordinates plugged in
      3. Expanded terms       — each B(n,i,u) in monomial form
      4. Collected polynomial — final x(u), y(u)[, z(u)]
      5. Evaluation table     — P(u) at each u value (exact)
    """

    WIDTH = 66

    def __init__(
        self,
        expansion: BezierExpansion,
        u_values:  list,
        case_num:  int,
    ):
        self.exp      = expansion
        self.u_values = u_values
        self.case_num = case_num
        self.u        = BezierExpansion.u   # SymPy symbol

    # -- Formatting helpers ----------------------------------

    def _ruler(self, char: str = '=') -> str:
        return char * self.WIDTH

    def _step(self, n: int, title: str):
        print()
        print(f'  STEP {n} — {title}')
        print('  ' + '-' * (self.WIDTH - 2))

    def _fmt_sympy(self, expr) -> str:
        """
        Render a SymPy expression as a clean string.
        Uses sp.srepr for internal form, sp.pretty for display.
        We use str(expr) which gives standard infix notation,
        then lightly post-process for readability.
        """
        return str(expr)

    def _fmt_bernstein_compact(self, n: int, i: int) -> str:
        """
        Compact human-readable form of B(n,i,u).
        e.g.  B(4,2,u)  →  6u²(1-u)²
        Keeps it short for the substitution line.
        """
        sup    = {0:'', 1:'', 2:'²', 3:'³', 4:'⁴', 5:'⁵'}
        binom  = int(sp.binomial(n, i))
        m      = n - i
        u_part = (
            '' if i == 0 else
            'u' if i == 1 else
            f'u{sup.get(i, f"^{i}")}'
        )
        v_part = (
            '' if m == 0 else
            '(1-u)' if m == 1 else
            f'(1-u){sup.get(m, f"^{m}")}'
        )
        b = '' if binom == 1 else str(binom)
        return f'{b}{u_part}{v_part}' or '1'

    def _fmt_coeff(self, c) -> str:
        """Format a SymPy number as integer or fraction."""
        return str(c)

    # -- Print steps -----------------------------------------

    def print_all(self):
        self._print_title()
        self._print_control_points()
        self._print_step1_bernstein_form()
        self._print_step2_substitution()
        self._print_step3_expanded_terms()
        self._print_step4_collected()
        self._print_step5_evaluation()
        print()
        print(self._ruler())
        print()

    # -- Title and control points ----------------------------

    def _print_title(self):
        n     = self.exp.n
        deg   = self.exp.degree
        dim   = self.exp.dim
        names = {2:'Quadratic', 3:'Cubic',
                 4:'Quartic',   5:'Quintic'}
        dname = names.get(deg, f'Degree-{deg}')

        print()
        print(self._ruler())
        print(f'  Bezier Curve — Algebraic Expansion  (via SymPy)')
        print(f'  Case {self.case_num}  |  {dname}  |  '
              f'{dim}D  |  {n} Control Points')
        print(self._ruler())

    def _print_control_points(self):
        print()
        print('  Control Points:')
        for i, pt in enumerate(self.exp.points):
            coords = '  '.join(
                f'{ax} = {self._fmt_coeff(pt[j])}'
                for j, ax in enumerate(self.exp.axes)
            )
            print(f'    P{i} :  {coords}')

    # -- Step 1: Bernstein form ------------------------------

    def _print_step1_bernstein_form(self):
        self._step(1, 'Bernstein Form')
        n   = self.exp.degree
        dim = '  '.join(f'{ax}(u)' for ax in self.exp.axes)

        print(f'  P(u)  =  [{dim}]')
        print()

        # General sum notation
        terms = '  +  '.join(
            f'B({n},{i})·P{i}' for i in range(self.exp.n)
        )
        print(f'  P(u)  =  {terms}')
        print()

        # Explicit symbolic definition of each basis term
        print('  where:')
        for i in range(self.exp.n):
            compact = self._fmt_bernstein_compact(n, i)
            print(f'    B({n},{i},u)  =  {compact}')

    # -- Step 2: substitution --------------------------------

    def _print_step2_substitution(self):
        self._step(2, 'Substitution of Control Point Coordinates')
        n = self.exp.degree

        for dim_idx, axis in enumerate(self.exp.axes):
            parts = []
            for i in range(self.exp.n):
                coord   = self.exp.points[i][dim_idx]
                compact = self._fmt_bernstein_compact(n, i)
                c_str   = self._fmt_coeff(coord)
                parts.append(f'{compact}·{c_str}')
            line = '  +  '.join(parts)
            print(f'  {axis}(u)  =  {line}')

        print()
        print('  (Each basis term is now expanded by SymPy)')

    # -- Step 3: expanded individual terms -------------------

    def _print_step3_expanded_terms(self):
        self._step(3, 'Expand Each Bernstein Basis Term')
        n = self.exp.degree

        for i in range(self.exp.n):
            compact  = self._fmt_bernstein_compact(n, i)
            expanded = self._fmt_sympy(self.exp.basis[i])

            print(f'  B({n},{i},u)  =  {compact}')
            print(f'           =  {expanded}')
            print()

    # -- Step 4: collected polynomial ------------------------

    def _print_step4_collected(self):
        self._step(4, 'Collect Terms — Final Monomial Polynomial')
        n = self.exp.degree

        # Weighted contributions per axis
        for dim_idx, axis in enumerate(self.exp.axes):
            print(f'  {axis}(u) — weighted contributions:')
            for i in range(self.exp.n):
                coord    = self.exp.points[i][dim_idx]
                # SymPy multiplies and expands the weighted term
                term_expr = sp.expand(
                    self.exp.basis[i] * coord
                )
                c_str    = self._fmt_coeff(coord)
                exp_str  = self._fmt_sympy(term_expr)
                print(f'    B({n},{i})·{c_str:>4}  =  {exp_str}')
            print()

        # Final collected result per axis
        print('  Collected:')
        for axis in self.exp.axes:
            expr = self.exp.sympy_exprs[axis]
            print(f'    {axis}(u)  =  {self._fmt_sympy(expr)}')

    # -- Step 5: evaluation table ----------------------------

    def _print_step5_evaluation(self):
        self._step(5, 'Evaluation at Given u Values')

        axes = self.exp.axes
        dim  = len(axes)

        # ── Evaluation table ─────────────────────────────────
        # SymPy evaluates with exact rational arithmetic.
        # We show both the exact fraction and the decimal.
        col_u     = 6
        col_exact = 10
        col_dec   = 12

        # Header
        header = f'  {"u":>{col_u}}'
        for ax in axes:
            header += f'  {ax+"(u) exact":>{col_exact}}  '
            header += f'{"decimal":>{col_dec}}'
        print(header)
        print('  ' + '-' * (col_u + (col_exact + col_dec + 4)
                             * dim + 2))

        for u_val in self.u_values:
            results = self.exp.evaluate(u_val)
            row = f'  {str(u_val):>{col_u}}'
            for val in results:
                exact = self._fmt_coeff(val)
                dec   = f'{float(val):.6f}'
                row  += f'  {exact:>{col_exact}}  {dec:>{col_dec}}'
            print(row)

        # ── Verification: full substitution shown ────────────
        u_check = self.u_values[len(self.u_values) // 2]
        print()
        print(f'  Verification at u = {u_check}:')

        for axis in axes:
            expr     = self.exp.sympy_exprs[axis]
            # Show the polynomial with u substituted literally
            subst    = expr.subs(self.u, u_check)
            # sp.nsimplify keeps it exact; float() for decimal
            exact    = sp.nsimplify(subst)
            decimal  = float(exact)

            print(f'    {axis}({u_check})  =  '
                  f'{self._fmt_sympy(expr)}')
            print(f'             =  '
                  f'{self._fmt_sympy(expr.subs(self.u, u_check))}')
            print(f'             =  {exact}  '
                  f'≈  {decimal:.6f}')
            print()


# ============================================================
# -- Entry Point ---------------------------------------------
# ============================================================

def main() -> int:
    data = CASE_DATA[CASE]

    expansion = BezierExpansion(
        points = data['points'],
        axes   = data['axes'],
    )

    printer = AlgebraicPrinter(
        expansion = expansion,
        u_values  = data['u_values'],
        case_num  = CASE,
    )

    printer.print_all()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
