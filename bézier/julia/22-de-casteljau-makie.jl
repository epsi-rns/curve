# ============================================================
# De Casteljau Algorithm — Quartic Bezier (3D)
# Julia port of 22-de-casteljau.py
# Uses CairoMakie
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

const U_SNAPSHOTS     = [0.25, 0.50, 0.75]
const SNAPSHOT_COLORS = [:deeppink, :darkorange, :purple]


# -- De Casteljau Computation --------------------------------

struct DeCasteljauQuartic
    points::Matrix{Float64}
end

lerp3(a, b, u) = (1 - u) .* a .+ u .* b

function reduce_level(pts::Matrix{Float64}, u::Float64)
    n = size(pts, 1)
    vcat([lerp3(pts[i,:], pts[i+1,:], u)' for i in 1:n-1]...)
end

function dc_steps(dc::DeCasteljauQuartic, u::Float64)
    l0 = dc.points
    l1 = reduce_level(l0, u)
    l2 = reduce_level(l1, u)
    l3 = reduce_level(l2, u)
    l4 = reduce_level(l3, u)
    return (level0=l0, level1=l1, level2=l2, level3=l3, level4=l4)
end

function dc_point(dc::DeCasteljauQuartic, u::Float64)
    dc_steps(dc, u).level4[1, :]
end

function dc_curve(dc::DeCasteljauQuartic, n::Int=300)
    u_vals = LinRange(0.0, 1.0, n)
    hcat([dc_point(dc, u) for u in u_vals]...)'   # (n × 3)
end


# -- Plot ----------------------------------------------------

const COLOR_CURVE   = :steelblue
const COLOR_POLYGON = :gray30

const LEVEL_STYLE = Dict(
    :level1 => (linewidth=1.4, linestyle=:solid),
    :level2 => (linewidth=1.2, linestyle=:dash),
    :level3 => (linewidth=1.0, linestyle=:dot),
)
const LEVEL_MSIZE = Dict(
    :level1 => 6, :level2 => 6, :level3 => 6, :level4 => 10
)

const CTRL_NAMES = [
    "P0(0,0,1)", "P1(0,4,1)", "P2(4,0,1)", "P3(4,4,1)", "P4(5,4,1)"
]
const CTRL_OFFSET = [
    (-0.5, -0.4,  0.06),
    (-0.5,  0.3,  0.06),
    ( 0.2, -0.4,  0.06),
    ( 0.2,  0.3,  0.06),
    ( 0.2,  0.15, 0.06),
]

function draw_and_save(
        dc         ::DeCasteljauQuartic,
        u_snapshots::Vector{Float64},
        colors     ::Vector{Symbol},
        filename   ::String)

    fig = Figure(size = (900, 650), backgroundcolor = :white)

    ax = Axis3(fig[1, 1],
        title           = "De Casteljau Algorithm — Quartic Bezier",
        titlesize       = 14,
        titlegap        = 12,
        xlabel          = "x",
        ylabel          = "y",
        zlabel          = "z",
        xlabelsize      = 11,
        ylabelsize      = 11,
        zlabelsize      = 11,
        limits          = (-0.5, 6.0, -0.5, 5.0, 0.0, 2.0),
        elevation       = deg2rad(25),
        azimuth         = deg2rad(360 - 55),
        backgroundcolor = :white,
        xgridcolor      = (:lightgray, 0.5),
        ygridcolor      = (:lightgray, 0.5),
        zgridcolor      = (:lightgray, 0.5),
    )

    # ── Full Bezier curve ────────────────────────────────────
    cpts = dc_curve(dc)
    lines!(ax, cpts[:,1], cpts[:,2], cpts[:,3],
        color     = COLOR_CURVE,
        linewidth = 2.5,
        label     = "Bezier Curve",
    )

    # ── Control polygon ──────────────────────────────────────
    cp = dc.points
    lines!(ax, cp[:,1], cp[:,2], cp[:,3],
        color     = COLOR_POLYGON,
        linestyle = :dash,
        linewidth = 1.4,
        label     = "Control Polygon (L0)",
    )
    scatter!(ax, cp[:,1], cp[:,2], cp[:,3],
        color       = COLOR_POLYGON,
        markersize  = 9,
        strokewidth = 0,
    )

    # ── Control point labels ──────────────────────────────────
    for (i, (name, (dx, dy, dz))) in enumerate(zip(CTRL_NAMES, CTRL_OFFSET))
        text!(ax, cp[i,1]+dx, cp[i,2]+dy, cp[i,3]+dz,
            text     = name,
            color    = COLOR_POLYGON,
            fontsize = 8,
            font     = :bold,
        )
    end

    # ── Level style legend entries (gray, once) ───────────────
    for (lbl, ls) in [("L1: solid", :solid), ("L2: dashed", :dash), ("L3: dotted", :dot)]
        lines!(ax, [NaN], [NaN], [NaN],
            color     = :gray60,
            linestyle = ls,
            linewidth = 1.2,
            label     = lbl,
        )
    end

    # ── Per-snapshot: reduction levels + final point ──────────
    for (u, color) in zip(u_snapshots, colors)
        s = dc_steps(dc, u)

        for (key, lkey) in [(:level1, :level1), (:level2, :level2), (:level3, :level3)]
            pts = getfield(s, key)
            st  = LEVEL_STYLE[lkey]
            lines!(ax, pts[:,1], pts[:,2], pts[:,3],
                color     = (color, 0.75),
                linewidth = st.linewidth,
                linestyle = st.linestyle,
            )
            scatter!(ax, pts[:,1], pts[:,2], pts[:,3],
                color       = (color, 0.85),
                markersize  = LEVEL_MSIZE[lkey],
                strokewidth = 0,
            )
        end

        # Final point T0 — diamond
        t0 = s.level4[1, :]
        scatter!(ax, [t0[1]], [t0[2]], [t0[3]],
            color       = color,
            marker      = :diamond,
            markersize  = 12,
            strokewidth = 0,
            label       = @sprintf("u=%.2f", u),
        )
        text!(ax, t0[1]+0.15, t0[2]+0.15, t0[3]+0.05,
            text     = @sprintf("T0(u=%.2f)\n(%.2f,%.2f,%.2f)", u, t0[1], t0[2], t0[3]),
            color    = color,
            fontsize = 7,
            font     = :bold,
        )
    end

    # ── Legend ───────────────────────────────────────────────
    axislegend(ax,
        position        = :lt,
        labelsize       = 9,
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
    dc = DeCasteljauQuartic(CONTROL_POINTS)
    draw_and_save(dc, U_SNAPSHOTS, SNAPSHOT_COLORS, "22-bezier-decasteljau-quartic-makie.png")
end

main()
