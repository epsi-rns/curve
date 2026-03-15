# ============================================================
# Bezier Curve — Matrix Method  [P(u) = U · M · G]
# Julia port of 03-bezier-matrix-pedagogical.py
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

function U_vec(deg::Int, u::Float64)::Matrix{Float64}
    reshape([u^(deg-k) for k in 0:deg], 1, deg+1)
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

print_step_header(step::Int, title::String) = (
    println();
    println("  STEP $step — $title");
    println("  " * "-"^(WIDTH-2))
)

# ── Plain-text table printer (no external deps) ──────────────────────────────
# style :simple  → header + underline (like tabulate 'simple')
# style :outline → box border         (like tabulate 'outline')

function print_table(
        rows       ::Vector{Vector{String}},
        headers    ::Vector{String};
        row_labels ::Union{Vector{String}, Nothing} = nothing,
        style      ::Symbol = :simple)

    has_rl  = row_labels !== nothing
    n_cols  = length(headers)
    widths  = [length(h) for h in headers]
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

    hdr  = fmt_row(headers, has_rl ? "" : nothing)
    ubar = "  " * (has_rl ? " "^rl_w * "  " : "") * join(["-"^w for w in widths], "  ")
    inner_w = sum(widths) + 2*(n_cols-1) + (has_rl ? rl_w + 2 : 0)

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
        fmt_fn     = fmt,
        style      ::Symbol = :simple)
    rows = [[fmt_fn(mat[r,c]) for c in 1:size(mat,2)] for r in 1:size(mat,1)]
    print_table(rows, col_labels; row_labels=row_labels, style=style)
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

function print_control_points(bm::BezierMatrix)
    println()
    println("  Control Points:")
    for i in 1:bm.n
        coords = join(
            ["$(ax) = $(fmt(bm.G[i,j]))" for (j,ax) in enumerate(bm.axes)],
            "  "
        )
        println("    P$(i-1) :  $coords")
    end
end

function print_step1_define(bm::BezierMatrix)
    print_step_header(1, "Matrix Formulation:  P(u) = U · M · G")
    n = bm.degree
    println()
    println("  P(u)  =  U(u)  ·  M  ·  G")
    println()
    println("  U(u)  shape (1×$(n+1))  parameter vector  [u^$n, ..., u, 1]")
    println("  M     shape ($(n+1)×$(n+1))  basis matrix  (derived from Bernstein)")
    println("  G     shape ($(n+1)×$(bm.dim))  geometry matrix  (control points)")
    println()
    print_labeled_matrix(bm.G, ["P$(i-1)" for i in 1:bm.n], bm.axes)
end

function print_step2_derive_M(bm::BezierMatrix)
    print_step_header(2, "Derive Basis Matrix M from Bernstein Polynomials")
    n   = bm.degree
    sup = SUPERSCRIPTS
    println()
    println("  Each column i of M holds the monomial coefficients")
    println("  of the Bernstein basis polynomial B($n, i, u):")
    println()
    println("    B(n,i,u)  =  C(n,i) · u^i · (1-u)^(n-i)")
    println()
    println("  Expanding (1-u)^(n-i) via the binomial theorem:")
    println("    (1-u)^m  =  Σ  C(m,j) · (-1)^j · u^j")
    println()
    println("  Coefficient of u^k in B(n,i,u):")
    println("    M[n-k, i]  =  C(n,i) · C(n-i, k-i) · (-1)^(k-i)   for k >= i, else 0")
    println()
    println("  Row ordering: row 0 = u^$n (highest),  row $n = u^0 (constant)")
    println()
    println("  Column-by-column derivation:")
    println()
    for i in 0:n
        m   = n - i
        b   = binomial(n, i) == 1 ? "" : string(binomial(n, i))
        u_p = i == 0 ? "" : (i == 1 ? "u" : "u$(get(sup, i, "^$i"))")
        v_p = m == 0 ? "" : (m == 1 ? "(1-u)" : "(1-u)$(get(sup, m, "^$m"))")
        println("  Column $i  →  B($n,$i,u)  =  $(b)$(u_p)$(v_p)")
        for k in 0:n
            c = bm.M[k+1, i+1]
            abs(c) < 1e-9 && continue
            power = n - k
            u_str = power == 0 ? "1" : (power == 1 ? "u" : "u^$power")
            println("    M[$k,$i]  =  $(lpad(Int(round(c)), 4))   (coeff of $u_str)")
        end
        println()
    end
    println("  Full M matrix:")
    println()
    print_labeled_matrix(bm.M, ["u^$(n-k)" for k in 0:n], ["P$(i-1)" for i in 1:bm.n])
end

function print_step3_MG(bm::BezierMatrix)
    print_step_header(3, "Compute  M · G  — Polynomial Coefficient Matrix")
    n = bm.degree
    println()
    println("  M · G  gives the monomial coefficients for each axis.")
    println("  Row k of (M·G) is the coefficient of u^(n-k).")
    println()
    print_labeled_matrix(bm.MG, ["u^$(n-k)" for k in 0:n], bm.axes)
    println()
    println("  Resulting polynomials:")
    for (d, axis) in enumerate(bm.axes)
        println("    $(axis)(u)  =  $(fmt_poly(bm.MG[:,d], n))")
    end
end

function print_step4_evaluate(bm::BezierMatrix, u_values::Vector{Float64})
    print_step_header(4, "Evaluate  U · (M · G)  at Each u Value")
    n = bm.degree
    for u in u_values
        Uv = U_vec(n, u)
        pt = vec(Uv * bm.MG)
        println()
        println("  ── u = $u " * "─"^(WIDTH - 10 - length(string(u))))
        u_headers = ["u^$(n-k)" for k in 0:n]
        u_row     = [[fmt_frac(Uv[1, k+1]) for k in 0:n]]
        println()
        println("  U  =")
        print_table(u_row, u_headers)
        println()
        println("  U · (M·G)  =  P(u):")
        for (d, axis) in enumerate(bm.axes)
            mg_col    = bm.MG[:, d]
            dot_parts = [
                "$(fmt_frac(Uv[1,k+1]))·($(fmt(mg_col[k+1])))"
                for k in 0:n
                if abs(mg_col[k+1]) > 1e-9 || abs(Uv[1,k+1]) > 1e-9
            ]
            println("    $(axis)(u)  =  $(join(dot_parts, "  +  "))")
            println("           =  $(fmt_frac(pt[d]))")
        end
    end
end

function print_step5_table(bm::BezierMatrix, u_values::Vector{Float64})
    print_step_header(5, "Summary — Evaluation Table")
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
    print_control_points(bm)
    print_step1_define(bm)
    print_step2_derive_M(bm)
    print_step3_MG(bm)
    print_step4_evaluate(bm, u_values)
    print_step5_table(bm, u_values)
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
