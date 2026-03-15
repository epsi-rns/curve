# ============================================================
# Bezier Curve — Matrix Method  [P(u) = U · M · G]
# Julia port of 05-bezier-matrix-sympy-text.py
# Exact rational arithmetic via built-in Rational{Int}
# Polynomial formatting via Symbolics.jl
# Table formatting via plain strings (no external deps for tables)
# ============================================================

using Symbolics
using Printf

@variables u


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
# -- BezierMatrix --------------------------------------------
# ============================================================

struct BezierMatrix
    G      ::Matrix{Rational{Int}}
    axes   ::Vector{String}
    n      ::Int
    degree ::Int
    dim    ::Int
    M      ::Matrix{Rational{Int}}
    MG     ::Matrix{Rational{Int}}
end

"""
    BezierMatrix(points, axes)

Exact rational Bezier matrix.  M derived from Bernstein via:
  M[n-k, i] = C(n,i) · C(n-i, k-i) · (-1)^(k-i)   for k >= i
All matrix products stay in Rational{Int} — no floats.
"""
function BezierMatrix(points::Matrix{Rational{Int}}, axes::Vector{String})
    n   = size(points, 1)
    deg = n - 1
    M   = derive_M(deg)
    BezierMatrix(points, axes, n, deg, length(axes), M, M * points)
end

function derive_M(deg::Int)::Matrix{Rational{Int}}
    n = deg
    M = zeros(Rational{Int}, n+1, n+1)
    for i in 0:n
        m = n - i
        for j in 0:m
            k = i + j
            M[n-k+1, i+1] = binomial(n,i) * binomial(m,j) * (-1)^j
        end
    end
    return M
end

function evaluate_all(bm::BezierMatrix, u_values::Vector{Rational{Int}})::Matrix{Rational{Int}}
    n    = bm.degree
    Uall = [u_val^(n-k) for u_val in u_values, k in 0:n]
    Uall * bm.MG
end


# ============================================================
# -- Formatting helpers --------------------------------------
# ============================================================

const WIDTH = 66

ruler(char::Char='=') = string(char)^WIDTH

# Format a Rational — integer if denominator is 1, else p//q notation
fmt_rat(r::Rational{Int}) = isone(denominator(r)) ? string(numerator(r)) : string(r)
fmt_rat(n::Integer)       = string(n)

section(title::String) = (
    println();
    println("  $title");
    println("  " * "-"^(WIDTH-2))
)

# ── Polynomial formatting via Symbolics.jl ───────────────────────────────────
# MG column k holds coefficient of u^(degree-k), highest power first.
# We build a symbolic expression by summing coeff * u^power, then expand().

function fmt_poly(mg_col::Vector{Rational{Int}}, deg::Int)::String
    expr = sum(
        mg_col[k+1] * u^(deg-k)
        for k in 0:deg
        if mg_col[k+1] != 0
    ; init = 0*u^0)
    string(expand(expr))
end

# ── Plain-text table printer ─────────────────────────────────────────────────
# style :simple  → header + underline
# style :outline → box border

function print_table(
        rows       ::Vector{Vector{String}},
        headers    ::Vector{String};
        row_labels ::Union{Vector{String}, Nothing} = nothing,
        style      ::Symbol = :simple)

    has_rl = row_labels !== nothing
    widths = [length(h) for h in headers]
    for row in rows
        for (c, cell) in enumerate(row)
            widths[c] = max(widths[c], length(cell))
        end
    end
    rl_w = has_rl ? max(maximum(length.(row_labels)), 2) : 0

    function fmt_row(cells, label=nothing)
        parts = String[]
        has_rl && push!(parts, lpad(something(label, ""), rl_w))
        for (c, cell) in enumerate(cells)
            push!(parts, lpad(cell, widths[c]))
        end
        "  " * join(parts, "  ")
    end

    hdr     = fmt_row(headers, has_rl ? "" : nothing)
    ubar    = "  " * (has_rl ? " "^rl_w * "  " : "") * join(["-"^w for w in widths], "  ")
    inner_w = sum(widths) + 2*(length(headers)-1) + (has_rl ? rl_w + 2 : 0)

    if style == :outline
        println("  ╒" * "═"^(inner_w+2) * "╕")
        println("  │ " * lstrip(hdr) * " │")
        println("  ╞" * "═"^(inner_w+2) * "╡")
        for (r, row) in enumerate(rows)
            println("  │ " * lstrip(fmt_row(row, has_rl ? row_labels[r] : nothing)) * " │")
        end
        println("  ╘" * "═"^(inner_w+2) * "╛")
    else
        println(hdr)
        println(ubar)
        for (r, row) in enumerate(rows)
            println(fmt_row(row, has_rl ? row_labels[r] : nothing))
        end
    end
end

function print_labeled_matrix(
        mat        ::Matrix{Rational{Int}},
        row_labels ::Vector{String},
        col_labels ::Vector{String})
    rows = [[fmt_rat(mat[r,c]) for c in 1:size(mat,2)] for r in 1:size(mat,1)]
    print_table(rows, col_labels; row_labels=row_labels)
end


# ============================================================
# -- Printer functions ---------------------------------------
# ============================================================

function print_title(bm::BezierMatrix, case_num::Int)
    names = Dict(2=>"Quadratic", 3=>"Cubic", 4=>"Quartic", 5=>"Quintic")
    dname = get(names, bm.degree, "Degree-$(bm.degree)")
    println()
    println(ruler())
    println("  Bezier Curve — Matrix Method  [P(u) = U · M · G]  (Symbolics.jl)")
    println("  Case $case_num  |  $dname  |  $(bm.dim)D  |  $(bm.n) Control Points")
    println(ruler())
end

function print_G(bm::BezierMatrix)
    section("G  — Geometry Matrix (Control Points)")
    println()
    print_labeled_matrix(bm.G, ["P$(i-1)" for i in 1:bm.n], bm.axes)
end

function print_M(bm::BezierMatrix)
    section("M  — Basis Matrix (derived via binomial)")
    n = bm.degree
    println()
    print_labeled_matrix(bm.M, ["u^$(n-k)" for k in 0:n], ["P$(i-1)" for i in 1:bm.n])
end

function print_MG(bm::BezierMatrix)
    section("M · G  — Polynomial Coefficient Matrix")
    n = bm.degree
    println()
    print_labeled_matrix(bm.MG, ["u^$(n-k)" for k in 0:n], bm.axes)
end

function print_polynomials(bm::BezierMatrix)
    section("Resulting Polynomials  (via Symbolics.jl)")
    println()
    for (d, axis) in enumerate(bm.axes)
        println("  $(axis)(u)  =  $(fmt_poly(bm.MG[:,d], bm.degree))")
    end
end

function print_eval_table(bm::BezierMatrix, u_values::Vector{Rational{Int}})
    section("Evaluation Table  —  P(u) = U · (M · G)")
    pts  = evaluate_all(bm, u_values)
    rows = Vector{Vector{String}}()
    for (r, u_val) in enumerate(u_values)
        row = [fmt_rat(u_val)]
        for c in 1:bm.dim
            exact = pts[r, c]
            dec   = @sprintf("%.4f", Float64(exact))
            push!(row, "$(fmt_rat(exact))  ($dec)")
        end
        push!(rows, row)
    end
    println()
    print_table(rows, vcat(["u"], ["$(ax)(u)" for ax in bm.axes]); style=:outline)
end

function print_all(bm::BezierMatrix, u_values::Vector{Rational{Int}}, case_num::Int)
    print_title(bm, case_num)
    print_G(bm)
    print_M(bm)
    print_MG(bm)
    print_polynomials(bm)
    print_eval_table(bm, u_values)
    println()
    println(ruler())
    println()
end


# ============================================================
# -- Entry Point ---------------------------------------------
# ============================================================

function main()
    data = CASE_DATA[CASE]
    bm   = BezierMatrix(data.points, data.axes)
    print_all(bm, data.u_values, CASE)
end

main()
