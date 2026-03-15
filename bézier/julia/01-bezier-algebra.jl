
# ============================================================
# Bezier Curve — Algebraic Expansion
# Julia port of 01-bezier-algebra.py
# ============================================================

using Printf

# ============================================================
# -- Switch  (only change this one line) ---------------------
# ============================================================

const CASE = 2   # 1 = Quadratic 2D,  2 = Quartic 3D


# ============================================================
# -- CASE Data  (do not change) ------------------------------
# ============================================================

const CASE_DATA = Dict(
    1 => (
        points   = [1 1;
                    3 3;
                    5 1],         # P0, P1, P2
        u_values = [0.00, 0.25, 0.50, 0.75, 1.00],
        axes     = ["x", "y"],
    ),
    2 => (
        points   = [0 0 1;
                    0 4 1;
                    4 0 1;
                    4 4 1;
                    5 4 1],       # P0 .. P4
        u_values = [0.00, 0.25, 0.50, 0.75, 1.00],
        axes     = ["x", "y", "z"],
    ),
)


# ============================================================
# -- Poly  (univariate polynomial in u) ----------------------
# ============================================================

# coeffs[k] is the coefficient of u^(k-1)  (1-indexed in Julia)
# So coeffs = [c0, c1, c2, ...] means c0 + c1*u + c2*u^2 + ...

struct Poly
    coeffs::Vector{Float64}
end

Poly(c::Vector{<:Real}) = Poly(Float64.(c))

# Scalar multiplication
Base.:*(p::Poly, s::Real) = Poly(p.coeffs .* s)
Base.:*(s::Real, p::Poly) = p * s

# Polynomial addition — zero-pad shorter one
function Base.:+(a::Poly, b::Poly)
    n = max(length(a.coeffs), length(b.coeffs))
    ca = vcat(a.coeffs, zeros(n - length(a.coeffs)))
    cb = vcat(b.coeffs, zeros(n - length(b.coeffs)))
    Poly(ca .+ cb)
end

# Evaluation via Horner's method
function (p::Poly)(u::Real)
    result = 0.0
    for c in reverse(p.coeffs)
        result = result * u + c
    end
    result
end

degree(p::Poly) = length(p.coeffs) - 1


# ============================================================
# -- Bernstein Expander --------------------------------------
# ============================================================

"""
    expand_bernstein(n, i) -> Poly

Expand B(n, i, u) = C(n,i) * u^i * (1-u)^(n-i) into monomial form.

Steps:
  A. Expand (1-u)^m via binomial theorem:
       (1-u)^m = Σ C(m,j)*(-1)^j * u^j
  B. Multiply by u^i  (shift coefficients by i)
  C. Multiply by C(n,i)
"""
function expand_bernstein(n::Int, i::Int)::Poly
    m = n - i

    # Step A: (1-u)^m
    one_minus_u = Poly([binomial(m, j) * (-1)^j for j in 0:m])

    # Step B: multiply by u^i  (prepend i zeros)
    shifted = Poly(vcat(zeros(i), one_minus_u.coeffs))

    # Step C: multiply by C(n,i)
    shifted * binomial(n, i)
end


# ============================================================
# -- BezierExpansion -----------------------------------------
# ============================================================

struct BezierExpansion
    points ::Matrix{Float64}   # (n_pts × dim)
    axes   ::Vector{String}
    n      ::Int               # number of control points
    degree ::Int
    dim    ::Int
    basis  ::Vector{Poly}      # B(degree, i, u) for i = 0..n-1
    polys  ::Dict{String,Poly} # final monomial per axis
end

function BezierExpansion(points::Matrix{<:Real}, axes::Vector{String})
    points = Float64.(points)
    n      = size(points, 1)
    deg    = n - 1
    dim    = length(axes)

    basis = [expand_bernstein(deg, i) for i in 0:n-1]

    polys = Dict{String,Poly}()
    for (dim_idx, axis) in enumerate(axes)
        poly = Poly([0.0])
        for i in 1:n
            coord = points[i, dim_idx]
            poly  = poly + basis[i] * coord
        end
        polys[axis] = poly
    end

    BezierExpansion(points, axes, n, deg, dim, basis, polys)
end

function evaluate(exp::BezierExpansion, u::Real)
    [exp.polys[ax](u) for ax in exp.axes]
end


# ============================================================
# -- Formatting helpers --------------------------------------
# ============================================================

const WIDTH = 64

ruler(char::Char='=') = string(char)^WIDTH

function fmt_coeff(c::Float64)
    c == round(c) ? string(Int(c)) : @sprintf("%.4f", c)
end

const SUPERSCRIPTS = Dict(
    0 => "", 1 => "", 2 => "²", 3 => "³",
    4 => "⁴", 5 => "⁵", 6 => "⁶",
)

function fmt_poly(p::Poly, var::String="u")
    terms = Tuple{Int,Float64,String}[]
    for (k, c) in enumerate(p.coeffs)
        k -= 1   # 0-indexed exponent
        abs(c) < 1e-9 && continue
        exp_str = get(SUPERSCRIPTS, k, "^$k")
        uvar    = k == 0 ? "" : (k == 1 ? var : "$var$exp_str")
        coef    = fmt_coeff(abs(c))
        coef    = (coef == "1" && !isempty(uvar)) ? "" : coef
        push!(terms, (k, c, coef * uvar))
    end
    isempty(terms) && return "0"

    parts = String[]
    for (idx, (k, c, term)) in enumerate(terms)
        if idx == 1
            push!(parts, c < 0 ? "-$term" : term)
        else
            push!(parts, c < 0 ? "- $term" : "+ $term")
        end
    end
    join(parts, "  ")
end

function fmt_bernstein_symbolic(n::Int, i::Int)
    sup = SUPERSCRIPTS
    b   = binomial(n, i)
    u_part = i == 0 ? "" : (i == 1 ? "u" : "u$(get(sup, i, "^$i"))")
    m      = n - i
    v_part = m == 0 ? "" : (m == 1 ? "(1-u)" : "(1-u)$(get(sup, m, "^$m"))")
    b_str  = b == 1 ? "" : string(b)
    "$(b_str)$(u_part)$(v_part)"
end


# ============================================================
# -- AlgebraicPrinter ----------------------------------------
# ============================================================

function print_step_header(step::Int, title::String)
    println()
    println("  STEP $step — $title")
    println("  " * "-"^(WIDTH - 2))
end

function print_title(exp::BezierExpansion, case_num::Int)
    names = Dict(2=>"Quadratic", 3=>"Cubic", 4=>"Quartic", 5=>"Quintic")
    dname = get(names, exp.degree, "Degree-$(exp.degree)")
    dstr  = "$(exp.dim)D"
    println()
    println(ruler())
    println("  Bezier Curve — Algebraic Expansion")
    println("  CASE $case_num  |  $dname  |  $dstr  |  $(exp.n) Control Points")
    println(ruler())
end

function print_control_points(exp::BezierExpansion)
    println()
    println("  Control Points:")
    for i in 1:exp.n
        coords = join(
            ["$(ax)=$(fmt_coeff(exp.points[i,j]))"
             for (j, ax) in enumerate(exp.axes)],
            "  "
        )
        println("    P$(i-1) :  $coords")
    end
end

function print_step1_bernstein_form(exp::BezierExpansion)
    print_step_header(1, "Bernstein Form")
    n   = exp.degree
    dim = join(["$(ax)(u)" for ax in exp.axes], "  ")
    println("  P(u) =  [$dim]")
    println()
    terms = ["B($n,$i)·P$i" for i in 0:exp.n-1]
    println("  P(u) =  " * join(terms, "  +  "))
    println()
    println("  where:")
    for i in 0:exp.n-1
        sym = fmt_bernstein_symbolic(n, i)
        println("    B($n,$i,u)  =  $sym")
    end
end

function print_step2_substitution(exp::BezierExpansion)
    print_step_header(2, "Substitution of Control Point Coordinates")
    n = exp.degree
    for (dim_idx, axis) in enumerate(exp.axes)
        parts = String[]
        for i in 1:exp.n
            coord = exp.points[i, dim_idx]
            sym   = fmt_bernstein_symbolic(n, i-1)
            c     = fmt_coeff(coord)
            push!(parts, "  $(sym)·$(c)")
        end
        println("  $(axis)(u) =" * join(parts, "  +  "))
    end
    println()
    println("  (Each basis term will now be expanded individually)")
end

function print_step3_expanded_terms(exp::BezierExpansion)
    print_step_header(3, "Expand Each Bernstein Basis Term")
    n = exp.degree
    for i in 0:exp.n-1
        basis = exp.basis[i+1]
        sym   = fmt_bernstein_symbolic(n, i)
        ex    = fmt_poly(basis)
        println("  B($n,$i,u)  =  $sym")
        println("           =  $ex")
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
            term    = exp.basis[i] * coord
            exp_str = fmt_poly(term)
            c_str   = fmt_coeff(coord)
            println("    B($n,$(i-1))·$(lpad(c_str,4))  =  $exp_str")
        end
        println()
    end
    println("  Collected:")
    for axis in exp.axes
        poly = exp.polys[axis]
        println("    $(axis)(u)  =  $(fmt_poly(poly))")
    end
end

function print_step5_evaluation(exp::BezierExpansion, u_values::Vector{Float64})
    print_step_header(5, "Evaluation at Given u Values")
    axes  = exp.axes
    dim   = length(axes)
    col_u = 6
    col_v = 10

    # Header
    header = "  " * lpad("u", col_u)
    for ax in axes
        header *= "  " * lpad("$(ax)(u)", col_v)
    end
    println(header)
    println("  " * "-"^(col_u + (col_v + 2) * dim + 2))

    # Data rows
    for u in u_values
        pt  = evaluate(exp, u)
        row = "  " * lpad(@sprintf("%.2f", u), col_u)
        for v in pt
            row *= "  " * lpad(@sprintf("%.6f", v), col_v)
        end
        println(row)
    end

    println()
    # Spot-check middle u
    u_check = u_values[div(length(u_values), 2) + 1]
    println("  Verification at u = $(@sprintf("%.2f", u_check)):")
    for axis in axes
        poly = exp.polys[axis]
        val  = poly(u_check)
        expr = fmt_poly(poly)
        println("    $(axis)($(@sprintf("%.2f", u_check)))  =  $expr")

        sub_parts = String[]
        for (k, c) in enumerate(poly.coeffs)
            k -= 1
            abs(c) < 1e-9 && continue
            push!(sub_parts,
                k > 0 ? "$(fmt_coeff(c))·$(@sprintf("%.2f", u_check))^$k"
                       : fmt_coeff(c)
            )
        end
        println("           =  " * join(sub_parts, "  +  "))
        println("           =  $(@sprintf("%.6f", val))")
        println()
    end
end

function print_all(exp::BezierExpansion, u_values::Vector{Float64}, case_num::Int)
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
    exp  = BezierExpansion(Float64.(data.points), data.axes)
    print_all(exp, data.u_values, CASE)
end

main()
