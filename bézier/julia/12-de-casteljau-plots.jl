# ============================================================
# De Casteljau Algorithm — Quadratic Bezier
# Julia port of 12-de-casteljau.py
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

const U_SNAPSHOTS      = [0.25, 0.50, 0.75]
const SNAPSHOT_COLORS  = [:deeppink, :darkorange, :purple]


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
const COLOR_POLYGON = :gray30

const CTRL_NAMES  = ["P0(1,1)", "P1(3,3)", "P2(5,1)"]
const CTRL_OFFSET = [(-0.30, -0.20), (0.07, 0.08), (0.10, -0.20)]

const Q_OFFSETS = [(-0.35, 0.05), (0.08, 0.05)]

function draw_and_save(
        dc         ::DeCasteljauQuadratic,
        u_snapshots::Vector{Float64},
        colors     ::Vector{Symbol},
        filename   ::String)

    plt = plot(
        size          = (800, 600),
        background    = :white,
        grid          = true,
        gridalpha     = 0.3,
        gridcolor     = :lightgray,
        framestyle    = :box,
        xlims         = (0.4, 5.8),
        ylims         = (0.4, 3.6),
        aspect_ratio  = :equal,
        xlabel        = "x",
        ylabel        = "y",
        title         = "\nDe Casteljau Algorithm — Quadratic Bezier",
        titlefontsize = 13,

        labelfontsize = 11,
        legend        = :topright,
        legendfontsize = 8,
    )

    # ── Full Bezier curve ────────────────────────────────────
    cpts = dc_curve(dc)
    plot!(plt,
        cpts[:,1], cpts[:,2],
        color     = COLOR_CURVE,
        linewidth = 2.5,
        label     = "Bezier Curve",
    )

    # ── Control polygon ──────────────────────────────────────
    l0 = steps(dc, 0.0).level0
    plot!(plt,
        l0[:,1], l0[:,2],
        color      = COLOR_POLYGON,
        linestyle  = :dash,
        linewidth  = 1.4,
        marker     = :circle,
        markersize = 6,
        label      = "Control Polygon",
    )

    # ── Control point labels ──────────────────────────────────
    for (i, (name, (dx, dy))) in enumerate(zip(CTRL_NAMES, CTRL_OFFSET))
        annotate!(plt, l0[i,1]+dx, l0[i,2]+dy,
            text(name, COLOR_POLYGON, :left, 8, :bold))
    end

    # ── Per-snapshot: level-1 segment, dots, level-2 diamond ─
    for (u, color) in zip(u_snapshots, colors)
        s  = steps(dc, u)
        l1 = s.level1
        r0 = s.level2

        # Level-1 Q0–Q1 segment
        plot!(plt,
            l1[:,1], l1[:,2],
            color     = color,
            linewidth = 1.6,
            label     = @sprintf("De Casteljau u=%.2f", u),
        )

        # Level-1 Q0, Q1 dots
        scatter!(plt,
            l1[:,1], l1[:,2],
            color             = color,
            markersize        = 5,
            markerstrokewidth = 0,
            label             = false,
        )

        # Level-2 R0 diamond
        scatter!(plt,
            [r0[1]], [r0[2]],
            color             = color,
            marker            = :diamond,
            markersize        = 7,
            markerstrokewidth = 0,
            label             = false,
        )

        # Q0, Q1 labels
        for (j, (qname, (dx, dy))) in enumerate(zip(["Q0","Q1"], Q_OFFSETS))
            annotate!(plt, l1[j,1]+dx, l1[j,2]+dy,
                text(@sprintf("%s(%.2f)", qname, u), color, :left, 7))
        end

        # R0 label
        annotate!(plt, r0[1]+0.08, r0[2]-0.22,
            text(@sprintf("R0(%.2f)\n(%.2f,%.2f)", u, r0[1], r0[2]),
                 color, :left, 7, :bold))
    end

    savefig(plt, filename)
    println("Saved: $filename")
    display(plt)
    println("Press Enter to close...")
    readline()
end


# -- Entry Point ---------------------------------------------

function main()
    dc = DeCasteljauQuadratic(CONTROL_POINTS)
    draw_and_save(dc, U_SNAPSHOTS, SNAPSHOT_COLORS, "12-bezier-decasteljau-plots.png")
end

main()
