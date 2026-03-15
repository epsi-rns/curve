# ============================================================
# Bezier Quartic — 3D Plot
# Julia port of 21-bezier-quartic.py
# Uses CairoMakie (replaces matplotlib + seaborn)
# ============================================================

using CairoMakie
using Printf


# -- Data ----------------------------------------------------

const CONTROL_POINTS = [
    0.0  0.0  1.0;   # P0
    0.0  4.0  1.0;   # P1
    4.0  0.0  1.0;   # P2
    4.0  4.0  1.0;   # P3
    5.0  4.0  1.0;   # P4
]

const U_HIGHLIGHT = round.(collect(0.0:0.1:1.0), digits=2)


# -- Bezier Computation --------------------------------------

struct BezierQuartic
    points::Matrix{Float64}
end

function bq_point(bz::BezierQuartic, u::Float64)
    v  = 1 - u
    b0 = v^4
    b1 = 4 * u * v^3
    b2 = 6 * u^2 * v^2
    b3 = 4 * u^3 * v
    b4 = u^4
    b0 .* bz.points[1,:] .+
    b1 .* bz.points[2,:] .+
    b2 .* bz.points[3,:] .+
    b3 .* bz.points[4,:] .+
    b4 .* bz.points[5,:]
end

function bq_curve(bz::BezierQuartic, n::Int=300)
    u_vals = LinRange(0.0, 1.0, n)
    hcat([bq_point(bz, u) for u in u_vals]...)'   # (n × 3)
end

function bq_highlight(bz::BezierQuartic, u_list::Vector{Float64})
    hcat([bq_point(bz, u) for u in u_list]...)'   # (m × 3)
end


# -- Plot ----------------------------------------------------

const COLOR_CURVE   = :steelblue
const COLOR_POLYGON = :crimson
const COLOR_CTRL    = :darkred
const COLOR_POINT   = :darkgreen
const COLOR_LABEL   = :purple

const CTRL_NAMES = [
    "P0(0,0,1)", "P1(0,4,1)", "P2(4,0,1)", "P3(4,4,1)", "P4(5,4,1)"
]

const CTRL_OFFSETS = [
    (-0.3, -0.4,  0.05),
    (-0.3,  0.3,  0.05),
    ( 0.15, -0.4, 0.05),
    ( 0.15,  0.3, 0.05),
    ( 0.15,  0.15, 0.05),
]

function draw_and_save(bz::BezierQuartic, u_highlight::Vector{Float64}, filename::String)

    fig = Figure(size = (900, 650), backgroundcolor = :white)

    ax = Axis3(fig[1, 1],
        title           = "Bezier Curve: Quartic (5 Control Points)",
        titlesize       = 14,
        titlegap        = 12,
        xlabel          = "x",
        ylabel          = "y",
        zlabel          = "z",
        xlabelsize      = 11,
        ylabelsize      = 11,
        zlabelsize      = 11,
        limits          = (0, 6, 0, 5, 0, 2),
        elevation       = deg2rad(22),
        azimuth         = deg2rad(360 - 50),
        backgroundcolor = :white,
        xgridcolor      = (:lightgray, 0.5),
        ygridcolor      = (:lightgray, 0.5),
        zgridcolor      = (:lightgray, 0.5),
    )

    # ── Bezier curve ─────────────────────────────────────────
    cpts = bq_curve(bz)
    lines!(ax, cpts[:,1], cpts[:,2], cpts[:,3],
        color     = COLOR_CURVE,
        linewidth = 2.5,
        label     = "Bezier Curve",
    )

    # ── Control polygon ──────────────────────────────────────
    cp = bz.points
    lines!(ax, cp[:,1], cp[:,2], cp[:,3],
        color     = COLOR_POLYGON,
        linestyle = :dash,
        linewidth = 1.5,
        label     = "Control Polygon",
    )
    scatter!(ax, cp[:,1], cp[:,2], cp[:,3],
        color       = COLOR_POLYGON,
        markersize  = 10,
        strokewidth = 0,
    )

    # ── Control point labels ──────────────────────────────────
    for (i, (name, (dx, dy, dz))) in enumerate(zip(CTRL_NAMES, CTRL_OFFSETS))
        text!(ax,
            cp[i,1]+dx, cp[i,2]+dy, cp[i,3]+dz,
            text     = name,
            color    = COLOR_CTRL,
            fontsize = 9,
            font     = :bold,
        )
    end

    # ── Highlighted points ────────────────────────────────────
    hpts = bq_highlight(bz, u_highlight)
    scatter!(ax, hpts[:,1], hpts[:,2], hpts[:,3],
        color       = COLOR_POINT,
        markersize  = 7,
        strokewidth = 0,
        label       = "Points at u",
    )

    # ── Highlighted point labels ──────────────────────────────
    for (i, u) in enumerate(u_highlight)
        pt = hpts[i, :]
        text!(ax,
            pt[1]+0.08, pt[2]+0.08, pt[3]+0.03,
            text     = @sprintf("u=%.1f", u),
            color    = COLOR_LABEL,
            fontsize = 7,
        )
    end

    # ── Parametric equations (2D overlay) ────────────────────
    eq = "x(u) = 13u^4 - 32u^3 + 24u^2\n" *
         "y(u) = -28u^4 + 64u^3 - 48u^2 + 16u\n" *
         "z(u) = 1"
    Label(fig[0, 1],
        eq,
        fontsize  = 9,
        font      = "Courier New",
        halign    = :left,
        padding   = (8, 8, 4, 4),
        tellwidth = false,
    )

    # ── Legend ───────────────────────────────────────────────
    axislegend(ax,
        position        = :lt,
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
    bz = BezierQuartic(CONTROL_POINTS)
    draw_and_save(bz, U_HIGHLIGHT, "21-bezier-quartic-makie.png")
end

main()
