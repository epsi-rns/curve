# ============================================================
# Bezier Curve — Matrix Method  [P(u) = U · M · G]
# Julia port of 04-bezier-matrix-straight.py
# No external dependencies — plain string table formatting
# ============================================================

using Printf


# ============================================================
# -- Switch  (only change this one line) ---------------------
# ============================================================

const CASE = 2   # 1 = Quadratic 2D,  2 = Quartic 3D


# ============================================================
# -- Case Data  (do not change) ------------------------------
# ============================================================

const CASE_DATA = Dict(
    1 => (
        points   = Float64[1 1;
                           3 3;
                           5 1],
        u_values = [0.00, 0.25, 0.50, 0.75, 1.00],
        axes     = ["x", "y"],
    ),
    2 => (
        points   = Float64[0 0 1;
                           0 4 1;
                           4 0 1;
                           4 4 1;
                           5 4 1],
        u_values = [0.00, 0.25, 0.50, 0.75, 1.00],
        axes     = ["x", "y", "z"],
    ),
)


# ============================================================
# -- BezierMatrix --------------------------------------------
# ============================================================

struct BezierMatrix
    G      ::Matrix{Float64}
    axes   ::Vector{String}
    n      ::Int
    degree ::Int
    dim    ::Int
    M      ::Matrix{Float64}
    MG     ::Matrix{Float64}
end

function BezierMatrix(points::Matrix{Float64}, axes::Vector{String})
    n   = size(points, 1)
    deg = n - 1
    M   = derive_M(deg)
    BezierMatrix(points, axes, n, deg, length(axes), M, M * points)
end

function derive_M(deg::Int)::Matrix{Float64}
    n = deg
    M = zeros(Float64, n+1, n+1)
    for i in 0:n
        m = n - i
        for j in 0:m
            k = i + j
            M[n-k+1, i+1] = binomial(n,i) * binomial(m,j) * (-1)^j
        end
    end
    return M
end

function evaluate_all(bm::BezierMatrix, u_values::Vector{Float64})::Matrix{Float64}
    n    = bm.degree
    Uall = [u^(n-k) for u in u_values, k in 0:n]
    Uall * bm.MG
end


# ============================================================
# -- Formatting helpers --------------------------------------
# ============================================================

const WIDTH = 66
const SUPERSCRIPTS = Dict(0=>"", 1=>"", 2=>"²", 3=>"³", 4=>"⁴", 5=>"⁵")

ruler(char::Char='=') = string(char)^WIDTH

fmt(v::Float64) = abs(v - round(v)) < 1e-9 ? string(Int(round(v))) : @sprintf("%.4f", v)

function fmt_frac(v::Float64)
    r = rationalize(Int, v; tol=1e-9)
    abs(Float64(r) - v) < 1e-9 || return @sprintf("%.4f", v)
    isone(denominator(r)) ? string(numerator(r)) : string(r)
end

function fmt_poly(coeffs::Vector{Float64}, deg::Int)
    terms = Tuple{Float64,String}[]
    for (k, c) in enumerate(coeffs)
        k -= 1
        abs(c) < 1e-9 && continue
        power = deg - k
        u_s   = power == 0 ? "" : (power == 1 ? "u" : "u$(get(SUPERSCRIPTS, power, "^$power"))")
        c_s   = fmt(abs(c))
        c_s   = (c_s == "1" && !isempty(u_s)) ? "" : c_s
        push!(terms, (c, c_s * u_s))
    end
    isempty(terms) && return "0"
    parts = String[]
    for (idx, (c, t)) in enumerate(terms)
        push!(parts, idx == 1 ? (c < 0 ? "-$t" : t) : (c < 0 ? "- $t" : "+ $t"))
    end
    join(parts, "  ")
end

section(title::String) = (
    println();
    println("  $title");
    println("  " * "-"^(WIDTH-2))
)

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

    hdr      = fmt_row(headers, has_rl ? "" : nothing)
    ubar     = "  " * (has_rl ? " "^rl_w * "  " : "") * join(["-"^w for w in widths], "  ")
    inner_w  = sum(widths) + 2*(length(headers)-1) + (has_rl ? rl_w + 2 : 0)

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
        mat        ::Matrix{Float64},
        row_labels ::Vector{String},
        col_labels ::Vector{String};
        fmt_fn     = fmt)
    rows = [[fmt_fn(mat[r,c]) for c in 1:size(mat,2)] for r in 1:size(mat,1)]
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
    println("  Bezier Curve — Matrix Method  [P(u) = U · M · G]")
    println("  Case $case_num  |  $dname  |  $(bm.dim)D  |  $(bm.n) Control Points")
    println(ruler())
end

function print_G(bm::BezierMatrix)
    section("G  — Geometry Matrix (Control Points)")
    println()
    print_labeled_matrix(bm.G, ["P$(i-1)" for i in 1:bm.n], bm.axes)
end

function print_M(bm::BezierMatrix)
    section("M  — Basis Matrix (derived from Bernstein)")
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
    section("Resulting Polynomials")
    println()
    for (d, axis) in enumerate(bm.axes)
        println("  $(axis)(u)  =  $(fmt_poly(bm.MG[:,d], bm.degree))")
    end
end

function print_eval_table(bm::BezierMatrix, u_values::Vector{Float64})
    section("Evaluation Table  —  P(u) = U · (M · G)")
    pts  = evaluate_all(bm, u_values)
    rows = Vector{Vector{String}}()
    for (i, u) in enumerate(u_values)
        row = [fmt(u)]
        for d in 1:bm.dim
            v = pts[i, d]
            push!(row, "$(fmt_frac(v))  ($(@sprintf("%.4f", v)))")
        end
        push!(rows, row)
    end
    println()
    print_table(rows, vcat(["u"], ["$(ax)(u)" for ax in bm.axes]); style=:outline)
end

function print_all(bm::BezierMatrix, u_values::Vector{Float64}, case_num::Int)
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
