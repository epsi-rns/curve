import numpy as np
from math import comb


# ============================================================
# -- Switch  (only change this one line) ---------------------
# ============================================================

CASE = 2   # 1 = Quadratic 2D,  2 = Quartic 3D


# ============================================================
# -- CASE Data  (do not change) --------------------------
# ============================================================

CASE_DATA = {
    1: dict(
        points = np.array([
            [1, 1],   # P0
            [3, 3],   # P1
            [5, 1],   # P2
        ]),
        u_values = [0.00, 0.25, 0.50, 0.75, 1.00],
        axes     = ['x', 'y'],
    ),
    2: dict(
        points = np.array([
            [0, 0, 1],   # P0
            [0, 4, 1],   # P1
            [4, 0, 1],   # P2
            [4, 4, 1],   # P3
            [5, 4, 1],   # P4
        ]),
        u_values = [0.00, 0.25, 0.50, 0.75, 1.00],
        axes     = ['x', 'y', 'z'],
    ),
}


# ============================================================
# -- Algebraic Utilities  (generic — do not change) ----------
# ============================================================

class Poly:
    """
    Represents a univariate polynomial in u as a coefficient
    list, where coeffs[k] is the coefficient of u^k.

      poly = c0 + c1*u + c2*u^2 + ... + cn*u^n
      coeffs = [c0, c1, c2, ..., cn]

    Supports scalar multiplication, addition, and evaluation.
    Used to track symbolic expansion of Bernstein basis terms.
    """

    def __init__(self, coeffs: list):
        self.coeffs = list(coeffs)   # index k → coeff of u^k

    def __mul__(self, scalar: float) -> 'Poly':
        return Poly([c * scalar for c in self.coeffs])

    def __rmul__(self, scalar: float) -> 'Poly':
        return self.__mul__(scalar)

    def __add__(self, other: 'Poly') -> 'Poly':
        n = max(len(self.coeffs), len(other.coeffs))
        a = self.coeffs  + [0] * (n - len(self.coeffs))
        b = other.coeffs + [0] * (n - len(other.coeffs))
        return Poly([x + y for x, y in zip(a, b)])

    def __call__(self, u: float) -> float:
        """Evaluate polynomial at u using Horner's method."""
        result = 0.0
        for c in reversed(self.coeffs):
            result = result * u + c
        return result

    def degree(self) -> int:
        return len(self.coeffs) - 1

    def __repr__(self):
        """Standard representation for debugging."""
        return f"Poly({self.coeffs})"
    
    def __str__(self):
        """User-friendly string representation."""
        return " + ".join([f"{c}u^{k}" for k, c in enumerate(self.coeffs)])

class BernsteinExpander:
    """
    Expands the Bernstein basis polynomial B(n, i, u) into
    its monomial (standard polynomial) form.

    B(n, i, u) = C(n,i) * u^i * (1-u)^(n-i)

    Strategy:
      1. Expand (1-u)^(n-i) via binomial theorem into a Poly.
      2. Multiply by u^i  by shifting coefficients.
      3. Multiply by C(n,i) scalar.

    Returns a Poly object whose coeffs[k] = coefficient of u^k.
    """

    def expand(self, n: int, i: int) -> 'Poly':
        """Return B(n, i, u) as a Poly."""

        # Step A: expand (1-u)^m via binomial theorem
        #   (1-u)^m = sum_{j=0}^{m} C(m,j) * (-u)^j
        #           = sum_{j=0}^{m} C(m,j) * (-1)^j * u^j
        m = n - i
        one_minus_u_poly = Poly([
            comb(m, j) * ((-1) ** j)
            for j in range(m + 1)
        ])

        # Step B: multiply by u^i
        #   shifting all coefficients up by i positions
        shifted = Poly(
            [0] * i + one_minus_u_poly.coeffs
        )

        # Super Clear Debugging:
        # print(f"DEBUG B({n},{i}):")
        # print(f"  Shifted Coeffs: {shifted.coeffs}")
        # print(f"  Binomial multiplier: {comb(n, i)}")

        # Step C: multiply by binomial coefficient C(n, i)        
        final_poly = shifted * comb(n, i)
        # print(f"  Result: {final_poly.coeffs}\n")
        
        return final_poly

# ============================================================
# -- Algebraic Expansion Engine  (generic — do not change) ---
# ============================================================

class BezierExpansion:
    """
    Performs and stores the full algebraic expansion of a
    Bezier curve of any degree and any spatial dimension.

    For each axis (x, y[, z]):
      - Computes the Bernstein basis polynomial for each P_i
      - Weights it by the control point coordinate
      - Sums all terms to get the final monomial polynomial

    The result per axis is a Poly object ready for evaluation.
    """

    def __init__(self, points: np.ndarray, axes: list):
        self.points  = points           # shape (n, dim)
        self.axes    = axes             # ['x','y'] or ['x','y','z']
        self.n       = len(points)      # number of control points
        self.degree  = self.n - 1      # polynomial degree
        self.dim     = len(axes)        # spatial dimension

        self._expander = BernsteinExpander()

        # Expand B(n, i, u) once per i — shared across axes
        self._basis: list[Poly] = [
            self._expander.expand(self.degree, i)
            for i in range(self.n)
        ]

        # Final monomial polynomial per axis
        self.polys: dict[str, Poly] = self._build_polys()

    def _build_polys(self) -> dict:
        """
        For each axis, compute the weighted sum:
          P_axis(u) = sum_{i=0}^{n} B(n,i,u) * coord_i
        Returns dict mapping axis name → Poly.
        """
        result = {}
        for dim_idx, axis in enumerate(self.axes):
            poly = Poly([0])
            for i in range(self.n):
                coord = self.points[i, dim_idx]
                poly  = poly + self._basis[i] * coord
            result[axis] = poly
        return result

    def evaluate(self, u: float) -> np.ndarray:
        """Evaluate all axis polynomials at u."""
        return np.array([
            self.polys[ax](u) for ax in self.axes
        ])


# ============================================================
# -- Console Printer  (generic — do not change) --------------
# ============================================================

class AlgebraicPrinter:
    """
    Prints the full pedagogical step-by-step algebraic
    expansion of a Bezier curve to the console.

    Steps shown:
      1. Header — degree, dimensions, control points
      2. Bernstein basis — symbolic B(n,i,u) definitions
      3. Substitution — each basis weighted by coordinates
      4. Expanded terms — each B(n,i,u) in monomial form
      5. Collected polynomial — final x(u), y(u)[, z(u)]
      6. Evaluation table — P(u) at each requested u value
    """

    # Column width for the console ruler line
    WIDTH = 64

    def __init__(
        self,
        expansion:  BezierExpansion,
        u_values:   list,
        hw_number:  int,
    ):
        self.exp       = expansion
        self.u_values  = u_values
        self.hw        = hw_number

    # -- Formatting helpers ----------------------------------

    def _ruler(self, char: str = '=') -> str:
        return char * self.WIDTH

    def _header_line(self, text: str) -> str:
        return f'  {text}'

    def _fmt_coeff(self, c: float) -> str:
        """Format a coefficient as integer if whole, else 4dp."""
        if c == int(c):
            return str(int(c))
        return f'{c:.4f}'

    def _fmt_poly(self, poly: Poly, var: str = 'u') -> str:
        """
        Format a Poly as a readable string.
        Example: '4u² - 4u + 1'  or  '-4u⁴ + 12u³ ...'
        Skips zero terms. Handles sign spacing.
        """
        superscripts = {
            0: '',  1: '',  2: '²', 3: '³',
            4: '⁴', 5: '⁵', 6: '⁶',
        }
        terms = []
        for k, c in enumerate(poly.coeffs):
            if abs(c) < 1e-9:
                continue
            exp  = superscripts.get(k, f'^{k}')
            uvar = '' if k == 0 else (var if k == 1 else f'{var}{exp}')
            coef = self._fmt_coeff(abs(c))
            # Omit '1' coefficient when multiplied by u term
            if coef == '1' and uvar:
                coef = ''
            terms.append((k, c, coef + uvar))

        if not terms:
            return '0'

        # Build string with proper sign handling
        parts = []
        for idx, (k, c, term) in enumerate(terms):
            if idx == 0:
                parts.append(f'-{term}' if c < 0 else term)
            else:
                parts.append(f'- {term}' if c < 0 else f'+ {term}')

        return '  '.join(parts)

    def _fmt_bernstein_symbolic(self, n: int, i: int) -> str:
        """
        Return the symbolic Bernstein term as a string.
        E.g. B(2,1,u) = 2u(1-u)  written as formula.
        Uses compact notation for readability.
        """
        sup = {0:'',1:'',2:'²',3:'³',4:'⁴',5:'⁵'}
        binom  = comb(n, i)
        u_part = (
            ''       if i == 0 else
            'u'      if i == 1 else
            f'u{sup.get(i, f"^{i}")}'
        )
        m = n - i
        v_part = (
            ''           if m == 0 else
            '(1-u)'      if m == 1 else
            f'(1-u){sup.get(m, f"^{m}")}'
        )
        b = '' if binom == 1 else str(binom)
        return f'{b}{u_part}{v_part}'

    # -- Print steps -----------------------------------------

    def _print_step_header(self, step: int, title: str):
        print()
        print(f'  STEP {step} — {title}')
        print('  ' + '-' * (self.WIDTH - 2))

    def print_all(self):
        """Run and print all steps in order."""
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

    # -- Step 0: title and control points --------------------

    def _print_title(self):
        n     = self.exp.n
        deg   = self.exp.degree
        dim   = self.exp.dim
        names = {2:'Quadratic', 3:'Cubic', 4:'Quartic', 5:'Quintic'}
        dname = names.get(deg, f'Degree-{deg}')
        dstr  = f'{dim}D'

        print()
        print(self._ruler())
        print(f'  Bezier Curve — Algebraic Expansion')
        print(f'  CASE {self.hw}  |  {dname}  |  '
              f'{dstr}  |  {n} Control Points')
        print(self._ruler())

    def _print_control_points(self):
        print()
        print('  Control Points:')
        for i, pt in enumerate(self.exp.points):
            coords = '  '.join(
                f'{ax}={self._fmt_coeff(pt[j])}'
                for j, ax in enumerate(self.exp.axes)
            )
            print(f'    P{i} :  {coords}')

    # -- Step 1: Bernstein form ------------------------------

    def _print_step1_bernstein_form(self):
        self._print_step_header(1, 'Bernstein Form')
        n   = self.exp.degree
        dim = '  '.join(f'{ax}(u)' for ax in self.exp.axes)
        print(f'  P(u) =  [{dim}]')
        print()
        print('  P(u) = ', end='')
        terms = []
        for i in range(self.exp.n):
            sym = self._fmt_bernstein_symbolic(n, i)
            terms.append(f'B({n},{i})·P{i}')
        print('  +  '.join(terms))
        print()
        print('  where:')
        for i in range(self.exp.n):
            sym = self._fmt_bernstein_symbolic(n, i)
            print(f'    B({n},{i},u)  =  {sym}')

    # -- Step 2: substitution --------------------------------

    def _print_step2_substitution(self):
        self._print_step_header(2, 'Substitution of Control Point Coordinates')
        n = self.exp.degree

        for dim_idx, axis in enumerate(self.exp.axes):
            print(f'  {axis}(u) =', end='')
            parts = []
            for i in range(self.exp.n):
                coord = self.exp.points[i, dim_idx]
                sym   = self._fmt_bernstein_symbolic(n, i)
                c     = self._fmt_coeff(coord)
                parts.append(f'  {sym}·{c}')
            print('  +  '.join(parts))
        print()
        print('  (Each basis term will now be expanded individually)')

    # -- Step 3: expanded individual terms -------------------

    def _print_step3_expanded_terms(self):
        self._print_step_header(3, 'Expand Each Bernstein Basis Term')
        n = self.exp.degree

        for i in range(self.exp.n):
            basis = self.exp._basis[i]
            sym   = self._fmt_bernstein_symbolic(n, i)
            exp   = self._fmt_poly(basis)
            print(f'  B({n},{i},u)  =  {sym}')
            print(f'           =  {exp}')
            print()

    # -- Step 4: collected polynomial ------------------------

    def _print_step4_collected(self):
        self._print_step_header(4, 'Collect Terms — Final Monomial Polynomial')
        n = self.exp.degree

        # Show the weighted sum being assembled per axis
        for dim_idx, axis in enumerate(self.exp.axes):
            print(f'  {axis}(u) — weighted contributions:')
            for i in range(self.exp.n):
                coord   = self.exp.points[i, dim_idx]
                term    = self.exp._basis[i] * coord
                exp_str = self._fmt_poly(term)
                c_str   = self._fmt_coeff(coord)
                print(f'    B({n},{i})·{c_str:>4}  =  {exp_str}')
            print()

        # Final collected polynomial per axis
        print('  Collected:')
        for axis in self.exp.axes:
            poly = self.exp.polys[axis]
            print(f'    {axis}(u)  =  {self._fmt_poly(poly)}')

    # -- Step 5: evaluation table ----------------------------

    def _print_step5_evaluation(self):
        self._print_step_header(5, 'Evaluation at Given u Values')

        axes = self.exp.axes
        dim  = len(axes)

        # Header row
        col_u   = 6
        col_v   = 10
        header  = f'  {"u":>{col_u}}'
        for ax in axes:
            header += f'  {ax+"(u)":>{col_v}}'
        print(header)
        print('  ' + '-' * (col_u + (col_v + 2) * dim + 2))

        # Data rows
        for u in self.u_values:
            pt  = self.exp.evaluate(u)
            row = f'  {u:>{col_u}.2f}'
            for v in pt:
                row += f'  {v:>{col_v}.6f}'
            print(row)

        print()
        # Spot-check one u value with full substitution shown
        u_check = self.u_values[len(self.u_values) // 2]
        print(f'  Verification at u = {u_check:.2f}:')
        for ax in axes:
            poly = self.exp.polys[ax]
            val  = poly(u_check)
            expr = self._fmt_poly(poly)
            print(f'    {ax}({u_check:.2f})  =  {expr}')
            # Show substituted form
            sub_parts = []
            for k, c in enumerate(poly.coeffs):
                if abs(c) < 1e-9:
                    continue
                sub_parts.append(
                    f'{self._fmt_coeff(c)}·{u_check:.2f}^{k}'
                    if k > 0
                    else self._fmt_coeff(c)
                )
            print(f'           =  {"  +  ".join(sub_parts)}')
            print(f'           =  {val:.6f}')
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
        expansion  = expansion,
        u_values   = data['u_values'],
        hw_number  = CASE,
    )

    printer.print_all()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
