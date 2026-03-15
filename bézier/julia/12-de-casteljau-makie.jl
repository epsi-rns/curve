# ============================================================
# De Casteljau Algorithm — Quadratic Bezier
# Julia port of 12-de-casteljau.py
# Uses CairoMakie (proper layout engine, replaces Plots/GR)
# ============================================================

using CairoMakie
using Printf


# -- Data ----------------------------------------------------

const CONTROL_POINTS = [
    1.0  1.0;   # P0
    3.0  3.0;   # P1
    5.0  1.0;   # P2
]

const U_SNAPSHOTS     = [0.25, 0.50, 0.75]
const SNAPSHOT_COLORS = [:deeppink, :darkorange, :purple]


# -- De Casteljau Computation --------------------------------

struct DeCasteljauQuadratic
    p0::Vector{Float64}
    p1::Vector{Float64}
    p2::Vector{Float64}
end

DeCasteljauQuadratic(pts::Matrix{Float64}) =
    DeCasteljauQuadratic(pts[1,:], pts[2,:], pts[3,:])

lerp(a, b, u) = (1 - u) .* a .+ u .* b

function steps(dc::DeCasteljauQuadratic, u::Float64)
    q0 = lerp(dc.p0, dc.p1, u)
    q1 = lerp(dc.p1, dc.p2, u)
    r0 = lerp(q0, q1, u)
    return (
        level0 = vcat(dc.p0', dc.p1', dc.p2'),   # (3×2)
        level1 = vcat(q0',    q1'),                # (2×2)
        level2 = r0,                               # (2,)
    )
end

function dc_curve(dc::DeCasteljauQuadratic, n::Int=200)
    u_vals = LinRange(0.0, 1.0, n)
    hcat([steps(dc, u).level2 for u in u_vals]...)'  # (n×2)
end


# -- Plot ----------------------------------------------------

const COLOR_CURVE   = :steelblue
const COLOR_POLYGON = :gray40

const CTRL_NAMES  = ["P0(1,1)", "P1(3,3)", "P2(5,1)"]
const CTRL_OFFSET = [(-0.30, -0.20), (0.07, 0.08), (0.10, -0.20)]
const Q_OFFSETS   = [(-0.35, 0.05), (0.08, 0.05)]

function draw_and_save(
        dc         ::DeCasteljauQuadratic,
        u_snapshots::Vector{Float64},
        colors     ::Vector{Symbol},
        filename   ::String)

    fig = Figure(size = (800, 600), backgroundcolor = :white)

    ax = Axis(fig[1, 1],
        title          = "De Casteljau Algorithm — Quadratic Bezier",
        titlesize      = 16,
        titlegap       = 12,       # gap between title and plot area
        xlabel         = "x",
        ylabel         = "y",
        xlabelsize     = 13,
        ylabelsize     = 13,
        limits         = (0.4, 5.8, 0.4, 3.6),
        aspect         = DataAspect(),
        xgridcolor     = (:lightgray, 0.6),
        ygridcolor     = (:lightgray, 0.6),
        backgroundcolor = :white,
    )

    # Outer padding so title has space from canvas edge

    # ── Full Bezier curve ────────────────────────────────────
    cpts = dc_curve(dc)
    lines!(ax, cpts[:,1], cpts[:,2],
        color     = COLOR_CURVE,
        linewidth = 2.5,
        label     = "Bezier Curve",
    )

    # ── Control polygon ──────────────────────────────────────
    l0 = steps(dc, 0.0).level0
    lines!(ax, l0[:,1], l0[:,2],
        color       = COLOR_POLYGON,
        linestyle   = :dash,
        linewidth   = 1.4,
        label       = "Control Polygon",
    )
    scatter!(ax, l0[:,1], l0[:,2],
        color        = COLOR_POLYGON,
        markersize   = 10,
        strokewidth  = 0,
    )

    # ── Control point labels ──────────────────────────────────
    for (i, (name, (dx, dy))) in enumerate(zip(CTRL_NAMES, CTRL_OFFSET))
        text!(ax, l0[i,1]+dx, l0[i,2]+dy,
            text      = name,
            color     = COLOR_POLYGON,
            fontsize  = 10,
            font      = :bold,
        )
    end

    # ── Per-snapshot elements ────────────────────────────────
    for (u, color) in zip(u_snapshots, colors)
        s  = steps(dc, u)
        l1 = s.level1
        r0 = s.level2

        # Level-1 Q0–Q1 segment
        lines!(ax, l1[:,1], l1[:,2],
            color     = color,
            linewidth = 1.6,
            label     = @sprintf("De Casteljau u=%.2f", u),
        )

        # Level-1 Q0, Q1 dots
        scatter!(ax, l1[:,1], l1[:,2],
            color       = color,
            markersize  = 8,
            strokewidth = 0,
        )

        # Level-2 R0 diamond
        scatter!(ax, [r0[1]], [r0[2]],
            color       = color,
            marker      = :diamond,
            markersize  = 12,
            strokewidth = 0,
        )

        # Q0, Q1 labels
        for (j, (qname, (dx, dy))) in enumerate(zip(["Q0","Q1"], Q_OFFSETS))
            text!(ax, l1[j,1]+dx, l1[j,2]+dy,
                text     = @sprintf("%s(%.2f)", qname, u),
                color    = color,
                fontsize = 9,
            )
        end

        # R0 label
        text!(ax, r0[1]+0.08, r0[2]-0.22,
            text     = @sprintf("R0(%.2f)\n(%.2f,%.2f)", u, r0[1], r0[2]),
            color    = color,
            fontsize = 9,
            font     = :bold,
        )
    end

    # ── Legend ───────────────────────────────────────────────
    axislegend(ax,
        position   = :rt,
        labelsize  = 10,
        framecolor = (:black, 0.3),
        backgroundcolor = (:white, 0.8),
    )

    # ── Save and display ─────────────────────────────────────
    save(filename, fig, px_per_unit = 1.5)
    println("Saved: $filename")

    display(fig)
    println("Press Enter to close...")
    readline()
end


# -- Entry Point ---------------------------------------------

function main()
    dc = DeCasteljauQuadratic(CONTROL_POINTS)
    draw_and_save(dc, U_SNAPSHOTS, SNAPSHOT_COLORS, "12-bezier_decasteljau-makie.png")
end

main()
