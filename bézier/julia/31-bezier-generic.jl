# ============================================================
# Bezier Curve — Generic (2D or 3D, any degree)
# Julia port of 31-bezier-generic.py
# Uses CairoMakie
# ============================================================

using GLMakie
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
        points      = Float64[1 1;
                               3 3;
                               5 1],
        u_highlight = [0.00, 0.25, 0.50, 0.75, 1.00],
        u_dc        = 0.50,
        output      = "31-bezier-quadratic-2d.png",
    ),
    2 => (
        points      = Float64[0 0 1;
                               0 4 1;
                               4 0 1;
                               4 4 1;
                               5 4 1],
        u_highlight = [0.00, 0.25, 0.50, 0.75, 1.00],
        u_dc        = 0.50,
        output      = "31-bezier-quartic-3d.png",
    ),
)


# ============================================================
# -- Bezier Computation --------------------------------------
# ============================================================

struct BezierCurve
    points    ::Matrix{Float64}   # (n × dim)
    n         ::Int               # number of control points
    degree    ::Int
    dim       ::Int               # 2 or 3
end

function BezierCurve(points::Matrix{Float64})
    n = size(points, 1)
    BezierCurve(points, n, n-1, size(points, 2))
end

function degree_name(bz::BezierCurve)
    names = Dict(2=>"Quadratic", 3=>"Cubic", 4=>"Quartic", 5=>"Quintic")
    get(names, bz.degree, "Degree-$(bz.degree)")
end

dim_name(bz::BezierCurve) = bz.dim == 3 ? "3D" : "2D"

function bernstein(n::Int, i::Int, u::Float64)
    binomial(n, i) * u^i * (1 - u)^(n - i)
end

function bz_point(bz::BezierCurve, u::Float64)
    sum(bernstein(bz.degree, i-1, u) .* bz.points[i,:] for i in 1:bz.n)
end

function bz_curve(bz::BezierCurve, n_pts::Int=300)
    u_vals = LinRange(0.0, 1.0, n_pts)
    hcat([bz_point(bz, u) for u in u_vals]...)'   # (n_pts × dim)
end

function bz_highlight(bz::BezierCurve, u_list::Vector{Float64})
    hcat([bz_point(bz, u) for u in u_list]...)'   # (m × dim)
end

function de_casteljau(bz::BezierCurve, u::Float64)
    levels  = Matrix{Float64}[copy(bz.points)]
    current = copy(bz.points)
    while size(current, 1) > 1
        n = size(current, 1)
        current = vcat([(
            (1-u) .* current[j,:] .+ u .* current[j+1,:]
        )' for j in 1:n-1]...)
        push!(levels, current)
    end
    return levels
end


# ============================================================
# -- Colors --------------------------------------------------
# ============================================================

const COLOR_CURVE   = :steelblue
const COLOR_POLYGON = :crimson
const COLOR_CTRL    = :crimson
const COLOR_POINT   = :darkgreen
const COLOR_LABEL   = :purple
const COLOR_DC      = [:darkorange, :deeppink, :darkviolet, :darkgreen]


# ============================================================
# -- Plot ----------------------------------------------------
# ============================================================

function draw_and_save(
        bz         ::BezierCurve,
        u_highlight::Vector{Float64},
        u_dc       ::Float64,
        filename   ::String)

    is_3d = bz.dim == 3
    title = "Bezier Curve: $(degree_name(bz)) ($(dim_name(bz)), $(bz.n) Control Points)"

    fig = Figure(size = (is_3d ? 900 : 800, is_3d ? 650 : 620),
                 backgroundcolor = :white)

    if is_3d
        ax = Axis3(fig[1, 1],
            title           = title,
            titlesize       = 14,
            titlegap        = 12,
            xlabel          = "x",
            ylabel          = "y",
            zlabel          = "z",
            xlabelsize      = 11,
            ylabelsize      = 11,
            zlabelsize      = 11,
            elevation       = deg2rad(35),
            azimuth         = deg2rad(295),
            limits          = (-0.5, 6.0, -0.5, 5.0, 0.0, 2.0),
            backgroundcolor = :white,
            xgridcolor      = (:lightgray, 0.5),
            ygridcolor      = (:lightgray, 0.5),
            zgridcolor      = (:lightgray, 0.5),
        )
    else
        ax = Axis(fig[1, 1],
            title           = title,
            titlesize       = 14,
            titlegap        = 12,
            xlabel          = "x",
            ylabel          = "y",
            xlabelsize      = 11,
            ylabelsize      = 11,
            backgroundcolor = :white,
            xgridcolor      = (:lightgray, 0.6),
            ygridcolor      = (:lightgray, 0.6),
        )
    end

    # ── Bezier curve ─────────────────────────────────────────
    cpts = bz_curve(bz)
    if is_3d
        lines!(ax, cpts[:,1], cpts[:,2], cpts[:,3],
            color=COLOR_CURVE, linewidth=2.5, label="Bezier Curve")
    else
        lines!(ax, cpts[:,1], cpts[:,2],
            color=COLOR_CURVE, linewidth=2.5, label="Bezier Curve")
    end

    # ── Control polygon ──────────────────────────────────────
    cp = bz.points
    if is_3d
        lines!(ax, cp[:,1], cp[:,2], cp[:,3],
            color=COLOR_POLYGON, linestyle=:dash, linewidth=1.5,
            label="Control Polygon")
        scatter!(ax, cp[:,1], cp[:,2], cp[:,3],
            color=COLOR_POLYGON, markersize=9, strokewidth=0)
    else
        lines!(ax, cp[:,1], cp[:,2],
            color=COLOR_POLYGON, linestyle=:dash, linewidth=1.5,
            label="Control Polygon")
        scatter!(ax, cp[:,1], cp[:,2],
            color=COLOR_POLYGON, markersize=9, strokewidth=0)
    end

    # ── Control point labels ──────────────────────────────────
    for i in 1:bz.n
        pt     = cp[i,:]
        coords = join([@sprintf("%.4g", v) for v in pt], ", ")
        name   = "P$(i-1)($coords)"
        dy     = i % 2 == 1 ? 0.15 : -0.28
        if is_3d
            text!(ax, pt[1]+0.10, pt[2]+dy, pt[3]+0.05,
                text=name, color=COLOR_CTRL, fontsize=8, font=:bold)
        else
            text!(ax, pt[1]+0.10, pt[2]+dy,
                text=name, color=COLOR_CTRL, fontsize=9, font=:bold)
        end
    end

    # ── Highlighted points ────────────────────────────────────
    hpts = bz_highlight(bz, u_highlight)
    if is_3d
        scatter!(ax, hpts[:,1], hpts[:,2], hpts[:,3],
            color=COLOR_POINT, markersize=7, strokewidth=0,
            label="Points at u")
        for (i, u) in enumerate(u_highlight)
            pt = hpts[i,:]
            text!(ax, pt[1]+0.08, pt[2]+0.08, pt[3]+0.05,
                text=@sprintf("u=%.2f", u),
                color=COLOR_LABEL, fontsize=7)
        end
    else
        scatter!(ax, hpts[:,1], hpts[:,2],
            color=COLOR_POINT, markersize=9, strokewidth=0,
            label="Points at u")
        for (i, u) in enumerate(u_highlight)
            pt     = hpts[i,:]
            coords = join([@sprintf("%.3g", v) for v in pt], ", ")
            lbl    = @sprintf("u=%.2f\n(%s)", u, coords)
            dx     = u == 1.0 ? -0.55 : 0.08
            dy     = u in (0.0, 1.0) ? -0.28 : 0.10
            text!(ax, pt[1]+dx, pt[2]+dy,
                text=lbl, color=COLOR_LABEL, fontsize=8)
        end
    end

    # ── De Casteljau levels ───────────────────────────────────
    # Draw all levels 1..n-1: horizontal segments within each level
    # plus vertical connectors between consecutive levels (the triangle)
    levels = de_casteljau(bz, u_dc)
    for lv in 2:length(levels)
        pts = levels[lv]
        col = COLOR_DC[min(lv-1, length(COLOR_DC))]
        is_final = (lv == length(levels))
        lbl = is_final ? @sprintf("DC final u=%.2f", u_dc) : "DC level $(lv-1)"

        if !is_final
            # Horizontal segment connecting points within this level
            if is_3d
                lines!(ax, pts[:,1], pts[:,2], pts[:,3],
                    color=col, linewidth=1.4, label=lbl)
            else
                lines!(ax, pts[:,1], pts[:,2],
                    color=col, linewidth=1.4, label=lbl)
            end
        end

        # Dots at each point in this level
        if is_3d
            scatter!(ax, pts[:,1], pts[:,2], pts[:,3],
                color=col, markersize=is_final ? 12 : 7,
                marker=is_final ? :diamond : :circle, strokewidth=0,
                label=is_final ? lbl : "")
        else
            scatter!(ax, pts[:,1], pts[:,2],
                color=col, markersize=is_final ? 12 : 7,
                marker=is_final ? :diamond : :circle, strokewidth=0,
                label=is_final ? lbl : "")
        end

        # Vertical connectors: lines from each point in level lv-1
        # to the corresponding point in level lv (the interpolation edge)
        prev = levels[lv-1]
        for j in 1:size(pts, 1)
            # Each point pts[j] was interpolated between prev[j] and prev[j+1]
            # Draw both edges of the interpolation triangle
            for k in [j, j+1]
                seg_x = [prev[k,1], pts[j,1]]
                seg_y = [prev[k,2], pts[j,2]]
                if is_3d
                    seg_z = [prev[k,3], pts[j,3]]
                    lines!(ax, seg_x, seg_y, seg_z,
                        color=(col, 0.35), linewidth=0.8, linestyle=:dot)
                else
                    lines!(ax, seg_x, seg_y,
                        color=(col, 0.35), linewidth=0.8, linestyle=:dot)
                end
            end
        end
    end

    # ── Equation box ─────────────────────────────────────────
    n     = bz.degree
    terms = join(["B($n,$(i-1))*P$(i-1)" for i in 1:bz.n], " + ")
    eq    = "P(u) = $terms\nB(n,i,u) = C(n,i)*u^i*(1-u)^(n-i)"
    Label(fig[0, 1],
        eq,
        fontsize  = 8,
        font      = "Courier New",
        halign    = :left,
        padding   = (8, 8, 4, 4),
        tellwidth = false,
    )

    # ── Legend ───────────────────────────────────────────────
    axislegend(ax,
        position        = is_3d ? :lt : :rb,
        labelsize       = 8,
        framecolor      = (:black, 0.3),
        backgroundcolor = (:white, 0.8),
    )

    # ── 2D auto limits ───────────────────────────────────────
    if !is_3d
        all_x = vcat(cp[:,1], cpts[:,1])
        all_y = vcat(cp[:,2], cpts[:,2])
        m = 0.8
        xlims!(ax, minimum(all_x)-m, maximum(all_x)+m)
        ylims!(ax, minimum(all_y)-m, maximum(all_y)+m)
    end

    save(filename, fig, px_per_unit=1.5)
    println("Saved: $filename")

    display(fig)
    println("Press Enter to close...")
    readline()
end


# ============================================================
# -- Entry Point ---------------------------------------------
# ============================================================

function main()
    data = CASE_DATA[CASE]
    bz   = BezierCurve(data.points)
    draw_and_save(bz, data.u_highlight, data.u_dc, data.output)
end

main()
