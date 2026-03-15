# ============================================================
# Bezier Curve — Interactive (2D or 3D, any degree)
# Julia port of 35-bezier-curvature-vue3-bulma.html
# Uses GLMakie — sliders, toggles, editable control points
# ============================================================

using GLMakie
using LinearAlgebra
using Statistics
using Printf


# ============================================================
# -- Case Data -----------------------------------------------
# ============================================================

const CASES = Dict(
    1 => (
        label    = "Case 1 — Quadratic 2D",
        points   = Float64[1 1; 5 7; 9 1],
        pt_names = ["P0","P1","P2"],
        pt_colors= [:crimson, :darkorange, :steelblue],
        dc_colors= [:darkorange, :darkgreen],
        dim      = 2,
    ),
    2 => (
        label    = "Case 2 — Quartic 3D",
        points   = Float64[0 0 1; 0 4 1; 4 0 1; 4 4 1; 5 4 1],
        pt_names = ["P0","P1","P2","P3","P4"],
        pt_colors= [:crimson, :purple, :darkorange, :teal, :darkgreen],
        dc_colors= [:goldenrod, :deeppink, :mediumpurple, :darkgreen],
        dim      = 3,
    ),
)


# ============================================================
# -- Bezier Math ---------------------------------------------
# ============================================================

function bernstein(n::Int, i::Int, u::Float64)
    binomial(n, i) * u^i * (1-u)^(n-i)
end

function bz_eval(pts::Matrix{Float64}, u::Float64)
    n = size(pts,1) - 1
    sum(bernstein(n, i-1, u) .* pts[i,:] for i in 1:size(pts,1))
end

function bz_curve(pts::Matrix{Float64}, n_pts::Int=300)
    hcat([bz_eval(pts, u) for u in LinRange(0,1,n_pts)]...)'
end

function bz_diff(pts::Matrix{Float64})
    Float64(size(pts,1) - 1) .* diff(pts, dims=1)
end

function de_casteljau(pts::Matrix{Float64}, u::Float64)
    levels = Matrix{Float64}[copy(pts)]
    cur    = copy(pts)
    while size(cur,1) > 1
        m   = size(cur,1)
        cur = vcat([((1-u).*cur[j,:] .+ u.*cur[j+1,:])' for j in 1:m-1]...)
        push!(levels, cur)
    end
    levels
end

function curvature(pts::Matrix{Float64}, u::Float64)
    d1pts = bz_diff(pts)
    d2pts = size(d1pts,1) > 1 ? bz_diff(d1pts) : nothing
    d1    = bz_eval(d1pts, u)
    d2    = isnothing(d2pts) ? zeros(size(pts,2)) : bz_eval(d2pts, u)
    spd   = norm(d1)
    spd < 1e-10 && return 0.0
    size(pts,2) == 2 ?
        abs(d1[1]*d2[2] - d1[2]*d2[1]) / spd^3 :
        norm(cross(d1, d2)) / spd^3
end

function monomial_coeffs(pts::Matrix{Float64}, d::Int)
    n = size(pts,1) - 1; coords = pts[:, d]; c = zeros(n+1)
    for k in 0:n
        s = 0.0
        for i in 0:k; s += binomial(n,i)*binomial(k,i)*(-1)^(k-i)*coords[i+1]; end
        c[n-k+1] = s
    end
    c
end

function poly_string(coeffs::Vector{Float64})
    n = length(coeffs)-1
    sups = ["u^$i" for i in n:-1:1]; push!(sups, "")
    parts = String[]
    for (i,c) in enumerate(coeffs)
        abs(c) < 1e-9 && continue
        push!(parts, @sprintf("%.3g%s", c, sups[i]))
    end
    isempty(parts) ? "0" : join(parts, " + ")
end

function make_eq_string(pts::Matrix{Float64}, u_val::Float64, dim::Int)
    axnames = dim == 3 ? ["x","y","z"] : ["x","y"]
    pt_u = bz_eval(pts, u_val)
    ls   = String[]
    for (ai,ax) in enumerate(axnames)
        push!(ls, "$(ax)(u) = $(poly_string(monomial_coeffs(pts, ai)))")
    end
    push!(ls, ""); push!(ls, @sprintf("At u = %.2f:", u_val))
    for (ai,ax) in enumerate(axnames)
        push!(ls, @sprintf("  %s = %.4f", ax, pt_u[ai]))
    end
    push!(ls, @sprintf("  κ   = %.4f", curvature(pts, u_val)))
    join(ls, "\n")
end


# ============================================================
# -- Comb computation ----------------------------------------
# ============================================================

function compute_comb(pts::Matrix{Float64}, n_pts::Int=120)
    dim   = size(pts,2)
    d1pts = bz_diff(pts)
    d2pts = size(d1pts,1) > 1 ? bz_diff(d1pts) : nothing
    u_vals = LinRange(0,1,n_pts)
    curve  = bz_curve(pts, n_pts)
    kappas = [curvature(pts, Float64(u)) for u in u_vals]
    ext    = maximum(maximum(curve[:,1:2],dims=1) .- minimum(curve[:,1:2],dims=1))
    kap95  = quantile(kappas, 0.95)
    scale  = kap95 > 1e-10 ? 0.15*ext/kap95 : 0.1
    step   = max(1, n_pts÷60)
    bases  = Vector{Float64}[]; tips = Vector{Float64}[]
    for i in 1:step:n_pts
        u = Float64(u_vals[i]); base = curve[i,:]; kap = kappas[i]
        d1 = bz_eval(d1pts, u)
        if dim == 2
            spd = norm(d1); spd < 1e-10 && continue
            N = [-d1[2], d1[1]] / spd
            tip = base .+ scale*kap.*N
        else
            d2    = isnothing(d2pts) ? zeros(3) : bz_eval(d2pts, u)
            N_raw = cross(cross(d1,d2), d1)
            nm    = norm(N_raw); nm < 1e-10 && continue
            tip   = base .+ scale*kap.*(N_raw/nm)
        end
        push!(bases, base); push!(tips, tip)
    end
    bases, tips
end


# ============================================================
# -- Draw onto axis ------------------------------------------
# ============================================================

function draw_all!(ax, pts, u_val, show_poly, show_dc, show_comb, case_idx)
    cs  = CASES[case_idx]
    dim = cs.dim
    crv = bz_curve(pts)

    # ── Bezier curve ─────────────────────────────────────────
    if dim == 2
        lines!(ax, crv[:,1], crv[:,2],
            color=:steelblue, linewidth=2.5, label="Bezier Curve")
    else
        lines!(ax, crv[:,1], crv[:,2], crv[:,3],
            color=:steelblue, linewidth=2.5, label="Bezier Curve")
    end

    # ── Control polygon ──────────────────────────────────────
    if show_poly
        if dim == 2
            lines!(ax, pts[:,1], pts[:,2],
                color=:crimson, linestyle=:dash, linewidth=1.5,
                label="Control Polygon")
        else
            lines!(ax, pts[:,1], pts[:,2], pts[:,3],
                color=:crimson, linestyle=:dash, linewidth=1.5,
                label="Control Polygon")
        end
        # Draw each control point individually (avoids color-vector issues in 3D)
        for i in 1:size(pts,1)
            pt  = pts[i,:]
            col = cs.pt_colors[i]
            if dim == 2
                scatter!(ax, [pt[1]], [pt[2]],
                    color=col, markersize=10, strokewidth=0)
            else
                scatter!(ax, [pt[1]], [pt[2]], [pt[3]],
                    color=col, markersize=10, strokewidth=0)
            end
        end
    end

    # ── Control point labels ─────────────────────────────────
    for i in 1:size(pts,1)
        pt     = pts[i,:]
        coords = join([@sprintf("%.4g",v) for v in pt], ",")
        lbl    = "$(cs.pt_names[i])($coords)"
        # Offset: alternate above/below in y, slightly offset in x
        dx = 0.15
        dy = i % 2 == 1 ? 0.15 : -0.30
        if dim == 2
            text!(ax, pt[1]+dx, pt[2]+dy,
                text=lbl, color=cs.pt_colors[i], fontsize=10, font=:bold)
        else
            text!(ax, pt[1]+dx, pt[2]+dy, pt[3]+0.05,
                text=lbl, color=cs.pt_colors[i], fontsize=9, font=:bold)
        end
    end

    # ── De Casteljau ─────────────────────────────────────────
    if show_dc
        levels = de_casteljau(pts, u_val)
        n_lv   = length(levels)
        for lv in 2:n_lv
            lv_pts   = levels[lv]
            col      = cs.dc_colors[min(lv-1, length(cs.dc_colors))]
            is_final = (lv == n_lv)
            lbl      = is_final ? @sprintf("DC final u=%.2f", u_val) : "DC L$(lv-1)"
            if size(lv_pts, 1) > 1
                if dim == 2
                    lines!(ax, lv_pts[:,1], lv_pts[:,2],
                        color=col, linewidth=1.8, label=lbl)
                else
                    lines!(ax, lv_pts[:,1], lv_pts[:,2], lv_pts[:,3],
                        color=col, linewidth=1.8, label=lbl)
                end
            end
            if dim == 2
                scatter!(ax, lv_pts[:,1], lv_pts[:,2],
                    color=col, markersize=is_final ? 14 : 8,
                    marker=is_final ? :diamond : :circle, strokewidth=0,
                    label=is_final ? lbl : " ")
            else
                scatter!(ax, lv_pts[:,1], lv_pts[:,2], lv_pts[:,3],
                    color=col, markersize=is_final ? 14 : 8,
                    marker=is_final ? :diamond : :circle, strokewidth=0,
                    label=is_final ? lbl : " ")
            end
        end
    end
    # ── Point at u ───────────────────────────────────────────
    pt_u = bz_eval(pts, u_val)
    if dim == 2
        scatter!(ax, [pt_u[1]], [pt_u[2]],
            color=:black, markersize=10, strokewidth=2,
            strokecolor=:white, label="Point at u")
    else
        scatter!(ax, [pt_u[1]], [pt_u[2]], [pt_u[3]],
            color=:black, markersize=10, strokewidth=2,
            strokecolor=:white, label="Point at u")
    end

    # ── Curvature comb ───────────────────────────────────────
    if show_comb
        bases, tips = compute_comb(pts)
        for (b,t) in zip(bases, tips)
            if dim == 2
                lines!(ax, [b[1],t[1]], [b[2],t[2]],
                    color=(:slategray,0.5), linewidth=1.0)
            else
                lines!(ax, [b[1],t[1]], [b[2],t[2]], [b[3],t[3]],
                    color=(:slategray,0.5), linewidth=1.0)
            end
        end
        if !isempty(tips)
            tm = hcat(tips...)'
            if dim == 2
                lines!(ax, tm[:,1], tm[:,2],
                    color=(:slategray,0.4), linewidth=1.0,
                    linestyle=:dash, label="Curvature Comb")
            else
                lines!(ax, tm[:,1], tm[:,2], tm[:,3],
                    color=(:slategray,0.4), linewidth=1.0,
                    linestyle=:dash, label="Curvature Comb")
            end
        end
    end
end


# ============================================================
# -- Main UI -------------------------------------------------
# ============================================================

function main()

    active_case = Ref(1)
    pts_obs     = Observable(copy(CASES[1].points))
    switching   = Ref(false)

    fig        = Figure(size=(1200, 720), backgroundcolor=:white)
    plot_layout = fig[1,1] = GridLayout()
    ax_ref      = Ref{Any}(nothing)
    legend_ref  = Ref{Any}(nothing)

    function make_ax!(dim)
        if !isnothing(legend_ref[]); delete!(legend_ref[]); legend_ref[]=nothing; end
        if !isnothing(ax_ref[]);    delete!(ax_ref[]);     ax_ref[]=nothing;     end
        if dim == 2
            ax_ref[] = Axis(plot_layout[1,1],
                title="Bezier Curve — Interactive",
                titlesize=14, titlegap=10,
                xlabel="x", ylabel="y",
                backgroundcolor=:white,
                xgridcolor=(:lightgray,0.5),
                ygridcolor=(:lightgray,0.5))
        else
            ax_ref[] = Axis3(plot_layout[1,1],
                title="Bezier Curve — Interactive",
                titlesize=14, titlegap=10,
                xlabel="x", ylabel="y", zlabel="z",
                backgroundcolor=:white,
                elevation=deg2rad(30), azimuth=deg2rad(295),
                xgridcolor=(:lightgray,0.5),
                ygridcolor=(:lightgray,0.5),
                zgridcolor=(:lightgray,0.5))
        end
    end

    # ── Controls ─────────────────────────────────────────────
    controls = fig[1,2] = GridLayout()
    colsize!(fig.layout, 1, Relative(0.68))
    colsize!(fig.layout, 2, Relative(0.32))
    r = Ref(1); nr!() = (r[]+=1; r[])

    Label(controls[r[],1],  "Case", fontsize=11, font=:bold, halign=:left)
    cg = controls[nr!(),1] = GridLayout()
    btn_c1 = Button(cg[1,1], label="Quadratic 2D",
        buttoncolor=:steelblue, labelcolor=:white)
    btn_c2 = Button(cg[1,2], label="Quartic 3D",
        buttoncolor=:white, labelcolor=:black)

    Label(controls[nr!(),1], "u — Parameter", fontsize=11, font=:bold, halign=:left)
    sg    = SliderGrid(controls[nr!(),1],
        (label="u", range=0.0:0.01:1.0, startvalue=0.5, format="{:.2f}"))
    u_obs = sg.sliders[1].value

    Label(controls[nr!(),1], "Display", fontsize=11, font=:bold, halign=:left)
    tg = controls[nr!(),1] = GridLayout()
    Label(tg[1,1], "Control Polygon", fontsize=10, halign=:left)
    Label(tg[2,1], "De Casteljau",    fontsize=10, halign=:left)
    Label(tg[3,1], "Curvature Comb",  fontsize=10, halign=:left)
    tog_poly = Toggle(tg[1,2], active=true)
    tog_dc   = Toggle(tg[2,2], active=true)
    tog_comb = Toggle(tg[3,2], active=true)

    Label(controls[nr!(),1], "Control Points", fontsize=11, font=:bold, halign=:left)
    pt_grid   = controls[nr!(),1] = GridLayout()
    btn_reset = Button(controls[nr!(),1],
        label="↺ Reset Points", buttoncolor=:white, labelcolor=:black)
    Label(controls[nr!(),1], "Equation", fontsize=11, font=:bold, halign=:left)
    eq_label  = Label(controls[nr!(),1], " ",
        fontsize=9, font="Courier New",
        halign=:left, justification=:left, tellwidth=false)

    # ── Redraw ───────────────────────────────────────────────
    function redraw()
        cs_  = CASES[active_case[]]
        pts_ = pts_obs[]
        u_   = u_obs[]
        dim_ = cs_.dim

        if !isnothing(legend_ref[])
            delete!(legend_ref[]); legend_ref[]=nothing
        end

        empty!(ax_ref[])
        ax = ax_ref[]

        draw_all!(ax, pts_, u_,
            tog_poly.active[], tog_dc.active[], tog_comb.active[],
            active_case[])

        crv = bz_curve(pts_)
        if dim_ == 2
            ax_x = vcat(pts_[:,1], crv[:,1])
            ax_y = vcat(pts_[:,2], crv[:,2])
            xlims!(ax, minimum(ax_x)-0.8, maximum(ax_x)+0.8)
            ylims!(ax, minimum(ax_y)-0.8, maximum(ax_y)+0.8)
        else
            ax_x = vcat(pts_[:,1], crv[:,1])
            ax_y = vcat(pts_[:,2], crv[:,2])
            ax_z = vcat(pts_[:,3], crv[:,3])
            xlims!(ax, minimum(ax_x)-0.5, maximum(ax_x)+0.5)
            ylims!(ax, minimum(ax_y)-0.5, maximum(ax_y)+0.5)
            zlims!(ax, minimum(ax_z)-0.5, maximum(ax_z)+0.5)
        end

        legend_ref[] = axislegend(ax,
            position=dim_==2 ? :rb : :lt,
            labelsize=9, framecolor=(:black,0.3),
            backgroundcolor=(:white,0.85))

        eq_label.text[] = make_eq_string(pts_, u_, dim_)
    end

    # ── Textbox fields ───────────────────────────────────────
    function rebuild_pt_fields!()
        foreach(delete!, contents(pt_grid))
        cs_  = CASES[active_case[]]
        pts_ = pts_obs[]
        for i in 1:size(pts_,1)
            Label(pt_grid[i,1], cs_.pt_names[i],
                fontsize=9, color=cs_.pt_colors[i], font=:bold, halign=:left)
            for d in 1:cs_.dim
                let row=i, col=d
                    tb = Textbox(pt_grid[row, col+1],
                        stored_string    = @sprintf("%.1f", pts_[row,col]),
                        displayed_string = @sprintf("%.1f", pts_[row,col]),
                        width=58, fontsize=10,
                        reset_on_defocus = false,
                        defocus_on_submit = true)
                    on(tb.stored_string) do val
                        v = tryparse(Float64, val)
                        isnothing(v) && return
                        new_pts = copy(pts_obs[])
                        new_pts[row, col] = v
                        pts_obs[] = new_pts
                    end
                end
            end
        end
    end

    # ── Wire observables ─────────────────────────────────────
    on(u_obs)           do _; redraw(); end
    on(pts_obs)         do _; switching[] || redraw(); end
    on(tog_poly.active) do _; redraw(); end
    on(tog_dc.active)   do _; redraw(); end
    on(tog_comb.active) do _; redraw(); end

    on(btn_c1.clicks) do _
        switching[]=true; active_case[]=1
        make_ax!(CASES[1].dim)
        pts_obs[]=copy(CASES[1].points); switching[]=false
        btn_c1.buttoncolor[]=:steelblue; btn_c1.labelcolor[]=:white
        btn_c2.buttoncolor[]=:white;     btn_c2.labelcolor[]=:black
        rebuild_pt_fields!(); redraw()
        tog_poly.active[] = false
        tog_poly.active[] = true
    end

    on(btn_c2.clicks) do _
        switching[]=true; active_case[]=2
        make_ax!(CASES[2].dim)
        pts_obs[]=copy(CASES[2].points); switching[]=false
        btn_c2.buttoncolor[]=:steelblue; btn_c2.labelcolor[]=:white
        btn_c1.buttoncolor[]=:white;     btn_c1.labelcolor[]=:black
        rebuild_pt_fields!(); redraw()
        tog_poly.active[] = false
        tog_poly.active[] = true
    end

    on(btn_reset.clicks) do _
        pts_obs[]=copy(CASES[active_case[]].points)
        rebuild_pt_fields!()
    end

    # ── Init ─────────────────────────────────────────────────
    display(fig)
    make_ax!(CASES[1].dim)
    rebuild_pt_fields!()
    redraw()
    # GLMakie needs one rendered frame before scatter! appears on a fresh axis.
    # Toggling an observable forces a second render pass reliably.
    tog_poly.active[] = false
    tog_poly.active[] = true

    println("Interactive window open. Press Enter to close...")
    readline()
end

main()
