# ============================================================
# Bezier Curve — Algebraic Expansion  (Symbolic)
# Julia port of 02-bezier-sympy.py
# Uses Symbolics.jl  (replaces SymPy)
# ============================================================
#
# SymPy  →  Julia mapping:
#   sp.Symbol('u')              @variables u  (Symbolics.jl)
#   sp.Rational(1,4)            1//4          (built-in Rational{Int})
#   sp.binomial(n,i)            binomial(n,i) (built-in)
#   sp.expand(expr)             expand(expr)  (Symbolics.jl)
#   sp.collect(expr, u)         expand(expr)  (Symbolics collects on expand)
#   sp.Poly(expr, u)            no equivalent needed — expr already collected
#   expr.subs(u, val)           substitute(expr, Dict(u => val))
#   float(exact)                Float64(Symbolics.value(simplified))
#   str(expr)                   string(expr)
#
# ============================================================

using Symbolics
using Printf

@variables u   # equivalent to sp.Symbol('u')


# ============================================================
# -- Switch  (only change this one line) ---------------------
# ============================================================

const CASE = 2   # 1 = Quadratic 2D,  2 = Quartic 3D


# ============================================================
# -- Case Data  (do not change) ------------------------------
# ============================================================

const CASE_DATA = Dict(
    1 => (
        points   = Rational{Int}[1 1;
                                  3 3;
                                  5 1],
        u_values = Rational{Int}[0, 1//4, 1//2, 3//4, 1],
        axes     = ["x", "y"],
    ),
    2 => (
        points   = Rational{Int}[0 0 1;
                                  0 4 1;
                                  4 0 1;
                                  4 4 1;
                                  5 4 1],
        u_values = Rational{Int}[0, 1//4, 1//2, 3//4, 1],
        axes     = ["x", "y", "z"],
    ),
)


# ============================================================
# -- BezierExpansion -----------------------------------------
# ============================================================

struct BezierExpansion
    points  ::Matrix{Rational{Int}}
    axes    ::Vector{String}
    n       ::Int
    degree  ::Int
    dim     ::Int
    basis   ::Vector   # symbolic Symbolics.jl expressions per B(n,i,u)
    exprs   ::Dict{String,Any}  # collected symbolic expr per axis
end

"""
    BezierExpansion(points, axes)

Symbolic expansion of a Bezier curve of any degree / dimension.

  B(n, i, u) = C(n,i) * u^i * (1-u)^(n-i)

`expand()` from Symbolics.jl distributes powers and collects
like terms — replacing sp.expand + sp.collect in one step.
`binomial(n,i)` is Julia built-in (exact integer arithmetic).
Coefficients stay as Rational{Int} throughout via the control
point matrix — no floating-point rounding anywhere.
"""
function BezierExpansion(points::Matrix{Rational{Int}}, axes::Vector{String})
    n   = size(points, 1)
    deg = n - 1
    dim = length(axes)

    # Bernstein basis — one symbolic expression per i
    # expand() applies binomial theorem and collects powers of u
    basis = [
        expand(binomial(deg, i) * u^i * (1 - u)^(deg - i))
        for i in 0:n-1
    ]

    # Per-axis polynomial: weighted sum of basis terms, then expand
    exprs = Dict{String,Any}()
    for (dim_idx, axis) in enumerate(axes)
        expr = sum(basis[i+1] * points[i+1, dim_idx] for i in 0:n-1)
        exprs[axis] = expand(expr)
    end

    BezierExpansion(points, axes, n, deg, dim, basis, exprs)
end

"""
    evaluate(exp, u_val) -> Vector

Substitute exact Rational u_val into each axis expression.
Equivalent to SymPy's expr.subs(u, val) with sp.Rational values.
Returns Symbolics simplified expressions (exact rational values).
"""
function evaluate(exp::BezierExpansion, u_val::Rational{Int})
    [substitute(exp.exprs[ax], Dict(u => u_val)) for ax in exp.axes]
end


# ============================================================
# -- Formatting helpers --------------------------------------
# ============================================================

const WIDTH = 66

ruler(char::Char='=') = string(char)^WIDTH

# Format Rational: show as fraction or integer
fmt_rational(r::Rational{Int}) = isone(denominator(r)) ? string(numerator(r)) : string(r)
fmt_rational(n::Integer)       = string(n)

# Format a Symbolics expression to string (mirrors str(expr) in SymPy)
fmt_sym(expr) = string(expr)

# Format the numeric result of a substitution back to exact Rational
function to_exact_rational(val)
    # substitute returns a Symbolics expression; extract numeric value
    v = Symbolics.value(val)
    # v may be a SymbolicUtils numeric — convert to Rational if possible
    try
        r = rationalize(Int, Float64(v); tol=1e-12)
        return r
    catch
        return v
    end
end

# Compact human-readable B(n,i,u): e.g. 6u²(1-u)²
const SUPERSCRIPTS = Dict(0=>"", 1=>"", 2=>"²", 3=>"³", 4=>"⁴", 5=>"⁵")

function fmt_bernstein_compact(n::Int, i::Int)
    b      = binomial(n, i)
    m      = n - i
    u_part = i == 0 ? "" : (i == 1 ? "u" : "u$(get(SUPERSCRIPTS, i, "^$i"))")
    v_part = m == 0 ? "" : (m == 1 ? "(1-u)" : "(1-u)$(get(SUPERSCRIPTS, m, "^$m"))")
    b_str  = b == 1 ? "" : string(b)
    result = "$(b_str)$(u_part)$(v_part)"
    isempty(result) ? "1" : result
end


# ============================================================
# -- Printer functions ---------------------------------------
# ============================================================

print_step_header(step::Int, title::String) = (
    println();
    println("  STEP $step — $title");
    println("  " * "-"^(WIDTH - 2))
)

function print_title(exp::BezierExpansion, case_num::Int)
    names = Dict(2=>"Quadratic", 3=>"Cubic", 4=>"Quartic", 5=>"Quintic")
    dname = get(names, exp.degree, "Degree-$(exp.degree)")
    println()
    println(ruler())
    println("  Bezier Curve — Algebraic Expansion  (via Symbolics.jl)")
    println("  Case $case_num  |  $dname  |  $(exp.dim)D  |  $(exp.n) Control Points")
    println(ruler())
end

function print_control_points(exp::BezierExpansion)
    println()
    println("  Control Points:")
    for i in 1:exp.n
        coords = join(
            ["$(ax) = $(fmt_rational(exp.points[i,j]))" for (j,ax) in enumerate(exp.axes)],
            "  "
        )
        println("    P$(i-1) :  $coords")
    end
end

function print_step1_bernstein_form(exp::BezierExpansion)
    print_step_header(1, "Bernstein Form")
    n   = exp.degree
    dim = join(["$(ax)(u)" for ax in exp.axes], "  ")
    println("  P(u)  =  [$dim]")
    println()
    println("  P(u)  =  " * join(["B($n,$i)·P$i" for i in 0:exp.n-1], "  +  "))
    println()
    println("  where:")
    for i in 0:exp.n-1
        println("    B($n,$i,u)  =  $(fmt_bernstein_compact(n, i))")
    end
end

function print_step2_substitution(exp::BezierExpansion)
    print_step_header(2, "Substitution of Control Point Coordinates")
    n = exp.degree
    for (dim_idx, axis) in enumerate(exp.axes)
        parts = [
            "$(fmt_bernstein_compact(n, i-1))·$(fmt_rational(exp.points[i, dim_idx]))"
            for i in 1:exp.n
        ]
        println("  $(axis)(u)  =  $(join(parts, "  +  "))")
    end
    println()
    println("  (Each basis term is now expanded by Symbolics.jl)")
end

function print_step3_expanded_terms(exp::BezierExpansion)
    print_step_header(3, "Expand Each Bernstein Basis Term")
    n = exp.degree
    for i in 0:exp.n-1
        println("  B($n,$i,u)  =  $(fmt_bernstein_compact(n, i))")
        println("           =  $(fmt_sym(exp.basis[i+1]))")
        println()
    end
end

function print_step4_collected(exp::BezierExpansion)
    print_step_header(4, "Collect Terms — Final Monomial Polynomial")
    n = exp.degree
    for (dim_idx, axis) in enumerate(exp.axes)
        println("  $(axis)(u) — weighted contributions:")
        for i in 1:exp.n
            coord   = exp.points[i, dim_idx]
            term    = expand(exp.basis[i] * coord)
            println("    B($n,$(i-1))·$(lpad(fmt_rational(coord), 4))  =  $(fmt_sym(term))")
        end
        println()
    end
    println("  Collected:")
    for axis in exp.axes
        println("    $(axis)(u)  =  $(fmt_sym(exp.exprs[axis]))")
    end
end

function print_step5_evaluation(exp::BezierExpansion, u_values::Vector{Rational{Int}})
    print_step_header(5, "Evaluation at Given u Values")

    axes      = exp.axes
    col_u     = 6
    col_exact = 10
    col_dec   = 12

    # Header
    header = "  " * lpad("u", col_u)
    for ax in axes
        header *= "  " * lpad("$(ax)(u) exact", col_exact) * "  " * lpad("decimal", col_dec)
    end
    println(header)
    println("  " * "-"^(col_u + (col_exact + col_dec + 4) * length(axes) + 2))

    # Data rows — exact rational + decimal
    for u_val in u_values
        results = evaluate(exp, u_val)
        row = "  " * lpad(fmt_rational(u_val), col_u)
        for val in results
            r     = to_exact_rational(val)
            exact = fmt_rational(r)
            dec   = @sprintf("%.6f", Float64(r))
            row  *= "  " * lpad(exact, col_exact) * "  " * lpad(dec, col_dec)
        end
        println(row)
    end

    # Verification at middle u — full substitution shown
    u_check = u_values[div(length(u_values), 2) + 1]
    println()
    println("  Verification at u = $(fmt_rational(u_check)):")
    for axis in axes
        expr    = exp.exprs[axis]
        subst   = substitute(expr, Dict(u => u_check))
        exact   = to_exact_rational(subst)
        decimal = Float64(exact)
        println("    $(axis)($(fmt_rational(u_check)))  =  $(fmt_sym(expr))")
        println("             =  $(fmt_sym(subst))")
        println("             =  $(fmt_rational(exact))  ≈  $(@sprintf("%.6f", decimal))")
        println()
    end
end

function print_all(exp::BezierExpansion, u_values::Vector{Rational{Int}}, case_num::Int)
    print_title(exp, case_num)
    print_control_points(exp)
    print_step1_bernstein_form(exp)
    print_step2_substitution(exp)
    print_step3_expanded_terms(exp)
    print_step4_collected(exp)
    print_step5_evaluation(exp, u_values)
    println()
    println(ruler())
    println()
end


# ============================================================
# -- Entry Point ---------------------------------------------
# ============================================================

function main()
    data = CASE_DATA[CASE]
    exp  = BezierExpansion(data.points, data.axes)
    print_all(exp, data.u_values, CASE)
end

main()
