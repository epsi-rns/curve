# ============================================================
# Bezier Quadratic — 2D Plot
# Julia port of 11-bezier-quadratic.py
# Uses CairoMakie (proper layout engine)
# ============================================================

using CairoMakie
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

function point(bz::BezierQuadratic, u::Float64)
    b0 = (1 - u)^2
    b1 = 2 * u * (1 - u)
    b2 = u^2
    b0 .* bz.p0 .+ b1 .* bz.p1 .+ b2 .* bz.p2
end

function curve(bz::BezierQuadratic, n::Int=200)
    u_vals = LinRange(0.0, 1.0, n)
    hcat([point(bz, u) for u in u_vals]...)'   # (n × 2)
end

function highlight_points(bz::BezierQuadratic, u_list::Vector{Float64})
    hcat([point(bz, u) for u in u_list]...)'   # (m × 2)
end


# -- Plot ----------------------------------------------------

const COLOR_CURVE   = :steelblue
const COLOR_POLYGON = :crimson
const COLOR_POINT   = :darkgreen
const COLOR_LABEL   = :purple
const COLOR_CTRL    = :crimson

const U_OFFSETS = Dict(
    0.00 => ( 0.08, -0.22),
    0.25 => ( 0.08,  0.08),
    0.50 => ( 0.08,  0.08),
    0.75 => ( 0.08,  0.08),
    1.00 => (-0.55, -0.22),
)

function draw_and_save(bz::BezierQuadratic, u_highlight::Vector{Float64}, filename::String)

    fig = Figure(size = (800, 600), backgroundcolor = :white)

    ax = Axis(fig[1, 1],
        title           = "Bezier Curve: Quadratic (3 Control Points)",
        titlesize       = 14,
        titlegap        = 12,
        xlabel          = "x",
        ylabel          = "y",
        xlabelsize      = 13,
        ylabelsize      = 13,
        limits          = (0.5, 6.0, 0.5, 3.5),
        aspect          = DataAspect(),
        xgridcolor      = (:lightgray, 0.6),
        ygridcolor      = (:lightgray, 0.6),
        backgroundcolor = :white,
    )

    # ── Bezier curve ─────────────────────────────────────────
    cpts = curve(bz)
    lines!(ax, cpts[:,1], cpts[:,2],
        color     = COLOR_CURVE,
        linewidth = 2.5,
        label     = "Bezier Curve",
    )

    # ── Control polygon ──────────────────────────────────────
    cp = vcat(bz.p0', bz.p1', bz.p2')
    lines!(ax, cp[:,1], cp[:,2],
        color     = COLOR_POLYGON,
        linestyle = :dash,
        linewidth = 1.5,
        label     = "Control Polygon",
    )
    scatter!(ax, cp[:,1], cp[:,2],
        color       = COLOR_POLYGON,
        markersize  = 10,
        strokewidth = 0,
    )

    # ── Control point labels ──────────────────────────────────
    offsets_ctrl = [(-0.25, -0.20), (0.08, 0.08), (0.08, -0.20)]
    names_ctrl   = ["P0(1,1)", "P1(3,3)", "P2(5,1)"]
    for (i, (name, (dx, dy))) in enumerate(zip(names_ctrl, offsets_ctrl))
        text!(ax, cp[i,1]+dx, cp[i,2]+dy,
            text     = name,
            color    = COLOR_CTRL,
            fontsize = 10,
            font     = :bold,
        )
    end

    # ── Highlighted points ────────────────────────────────────
    hpts = highlight_points(bz, u_highlight)
    scatter!(ax, hpts[:,1], hpts[:,2],
        color       = COLOR_POINT,
        markersize  = 11,
        strokewidth = 0,
        label       = "Points at u",
    )

    # ── Highlighted point labels ──────────────────────────────
    for u in u_highlight
        pt     = point(bz, u)
        x, y   = pt[1], pt[2]
        dx, dy = get(U_OFFSETS, u, (0.08, 0.08))
        text!(ax, x+dx, y+dy,
            text     = @sprintf("u=%.2f\n(%.2f, %.2f)", u, x, y),
            color    = COLOR_LABEL,
            fontsize = 9,
        )
    end

    # ── Parametric equations ──────────────────────────────────
    text!(ax, 1.6, 0.75,
        text     = "x(u) = 1 + 4u\ny(u) = -4u^2 + 4u + 1",
        color    = :black,
        fontsize = 9,
    )

    # ── Legend ───────────────────────────────────────────────
    axislegend(ax,
        position        = :rt,
        labelsize       = 10,
        framecolor      = (:black, 0.3),
        backgroundcolor = (:white, 0.8),
    )

    save(filename, fig, px_per_unit = 1.5)
    println("Saved: $filename")

    display(fig)
    println("Press Enter to close...")
    readline()
end


# -- Entry Point ---------------------------------------------

function main()
    bz = BezierQuadratic(CONTROL_POINTS)
    draw_and_save(bz, U_HIGHLIGHT, "11-bezier-quadratic-makie.png")
end

main()
