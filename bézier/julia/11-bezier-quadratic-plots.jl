# ============================================================
# Bezier Quadratic — 2D Plot
# Julia port of 11-bezier-quadratic.py
# Uses Plots.jl (replaces matplotlib + seaborn)
# ============================================================

using Plots
using Printf


# -- Data ----------------------------------------------------

const CONTROL_POINTS = [
    1.0  1.0;   # P0
    3.0  3.0;   # P1
    5.0  1.0;   # P2
]

const U_HIGHLIGHT = [0.00, 0.25, 0.50, 0.75, 1.00]


# -- Bezier Computation --------------------------------------

struct BezierQuadratic
    p0::Vector{Float64}
    p1::Vector{Float64}
    p2::Vector{Float64}
end

BezierQuadratic(pts::Matrix{Float64}) =
    BezierQuadratic(pts[1,:], pts[2,:], pts[3,:])

function basis(u::Float64)
    b0 = (1 - u)^2
    b1 = 2 * u * (1 - u)
    b2 = u^2
    return b0, b1, b2
end

function point(bz::BezierQuadratic, u::Float64)
    b0, b1, b2 = basis(u)
    b0 .* bz.p0 .+ b1 .* bz.p1 .+ b2 .* bz.p2
end

function curve(bz::BezierQuadratic, n::Int=200)
    u_vals = LinRange(0.0, 1.0, n)
    pts    = hcat([point(bz, u) for u in u_vals]...)'  # (n × 2)
    return pts
end

function highlight_points(bz::BezierQuadratic, u_list::Vector{Float64})
    hcat([point(bz, u) for u in u_list]...)'           # (m × 2)
end


# -- Plot ----------------------------------------------------

const COLOR_CURVE   = :steelblue
const COLOR_POLYGON = :crimson
const COLOR_POINT   = :darkgreen
const COLOR_LABEL   = :purple4
const COLOR_CTRL    = :crimson

# Per-u annotation offsets to avoid overlaps
const U_OFFSETS = Dict(
    0.00 => ( 0.08, -0.22),
    0.25 => ( 0.08,  0.08),
    0.50 => ( 0.08,  0.08),
    0.75 => ( 0.08,  0.08),
    1.00 => (-0.55, -0.22),
)

function draw_and_save(bz::BezierQuadratic, u_highlight::Vector{Float64}, filename::String)

    plt = plot(
        size        = (800, 600),
        background  = :white,
        grid        = true,
        gridalpha   = 0.3,
        gridcolor   = :lightgray,
        framestyle  = :box,
        xlims       = (0.5, 6.0),
        ylims       = (0.5, 3.5),
        aspect_ratio = :equal,
        xlabel      = "x",
        ylabel      = "y",
        title       = "Bezier Curve: Quadratic (3 Control Points)",
        titlefontsize = 13,
        labelfontsize = 11,
        legend      = :topright,
        legendfontsize = 9,
    )

    # ── Bezier curve ─────────────────────────────────────────
    cpts = curve(bz)
    plot!(plt,
        cpts[:,1], cpts[:,2],
        color     = COLOR_CURVE,
        linewidth = 2.5,
        label     = "Bezier Curve",
    )

    # ── Control polygon ──────────────────────────────────────
    cp = vcat(bz.p0', bz.p1', bz.p2')
    plot!(plt,
        cp[:,1], cp[:,2],
        color      = COLOR_POLYGON,
        linestyle  = :dash,
        linewidth  = 1.5,
        marker     = :circle,
        markersize = 6,
        label      = "Control Polygon",
    )

    # ── Control point labels ──────────────────────────────────
    offsets_ctrl = [(-0.25, -0.20), (0.08, 0.08), (0.08, -0.20)]
    names_ctrl   = ["P0(1,1)", "P1(3,3)", "P2(5,1)"]
    for (pt, name, (dx, dy)) in zip(eachrow(cp), names_ctrl, offsets_ctrl)
        annotate!(plt,
            pt[1] + dx, pt[2] + dy,
            text(name, COLOR_CTRL, :left, 9, :bold)
        )
    end

    # ── Highlighted points ────────────────────────────────────
    hpts = highlight_points(bz, u_highlight)
    scatter!(plt,
        hpts[:,1], hpts[:,2],
        color      = COLOR_POINT,
        markersize = 8,
        markerstrokewidth = 0,
        label      = "Points at u",
    )

    # ── Highlighted point labels ──────────────────────────────
    for u in u_highlight
        pt       = point(bz, u)
        x, y     = pt[1], pt[2]
        dx, dy   = get(U_OFFSETS, u, (0.08, 0.08))
        lbl      = @sprintf("u=%.2f\n(%.2f, %.2f)", u, x, y)
        annotate!(plt,
            x + dx, y + dy,
            text(lbl, COLOR_LABEL, :left, 7)
        )
    end

    # ── Parametric equations ──────────────────────────────────
    annotate!(plt,
        1.6, 0.75,
        text("x(u) = 1 + 4u\ny(u) = -4u^2 + 4u + 1",
             :black, :left, 8)
    )

    savefig(plt, filename)
    println("Saved: $filename")
    display(plt)
    println("Press Enter to close...")
    readline()
end


# -- Entry Point ---------------------------------------------

function main()
    bz = BezierQuadratic(CONTROL_POINTS)
    draw_and_save(bz, U_HIGHLIGHT, "11-bezier-quadratic-plots.png")
end

main()
