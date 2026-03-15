// ============================================================
// Bezier Quartic 3D -- Interactive UI
// Homework 2: P0(0,0,1) P1(0,4,1) P2(4,0,1) P3(4,4,1) P4(5,4,1)
// ============================================================
funcprot(0);

// -- 1. Global Data ------------------------------------------
global P u h_slider h_pts h_axes h_drawn;

P = [0, 0, 1;
     0, 4, 1;
     4, 0, 1;
     4, 4, 1;
     5, 4, 1];
u       = 0.5;
h_pts   = list();
h_drawn = list();   // tracks every drawn graphics handle

// -- 2. Helpers ----------------------------------------------

function pt = lerp(a, b, v)
    pt = (1 - v) .* a + v .* b;
endfunction

// Delete only the previously drawn objects — axes untouched
function clear_drawn()
    global h_drawn;
    for k = 1:size(h_drawn)
        if is_handle_valid(h_drawn(k)) then
            delete(h_drawn(k));
        end
    end
    h_drawn = list();
endfunction

function h = draw_line3(xs, ys, zs, col, thick, lstyle)
    global h_drawn;
    param3d(xs, ys, zs);
    h            = gce();
    h.foreground = col;
    h.thickness  = thick;
    h.line_style = lstyle;
    h_drawn($+1) = h;
endfunction

function h = draw_dot3(px, py, pz, col, msz, mstyle)
    global h_drawn;
    param3d(px, py, pz);
    h                 = gce();
    h.mark_mode       = "on";
    h.mark_style      = mstyle;
    h.mark_foreground = col;
    h.mark_background = col;
    h.mark_size       = msz;
    h.line_mode       = "off";
    h_drawn($+1) = h;
endfunction

function draw_label(x, y, txt, col)
    global h_drawn;
    xstring(x, y, txt);
    tx            = gce();
    tx.font_size  = 2;
    tx.foreground = col;
    h_drawn($+1)  = tx;
endfunction

// -- 3. Plot -------------------------------------------------

function update_plot()
    global P u h_axes;

    sca(h_axes);

    // Delete only drawn content — axes/view/rotation untouched
    clear_drawn();

    // Update title only
    h_axes.title.text = "Quartic Bezier 3D   u = " + string(u);

    // ── Control polygon ──────────────────────────────────
    draw_line3(P(:,1), P(:,2), P(:,3), color("red"), 1, 2);

    // Control point markers + labels
    pt_labels = ["P0","P1","P2","P3","P4"];
    for i = 1:5
        draw_dot3(P(i,1), P(i,2), P(i,3), color("red"), 6, 9);
        draw_label(P(i,1)+0.1, P(i,2)+0.1, pt_labels(i), color("red"));
    end

    // ── Bezier curve via matrix form ─────────────────────
    M = [1  -4   6  -4  1; ..
        -4  12 -12   4  0; ..
         6 -12   6   0  0; ..
        -4   4   0   0  0; ..
         1   0   0   0  0];
    n    = 200;
    tv   = linspace(0, 1, n)';
    Umat = [tv.^4, tv.^3, tv.^2, tv, ones(n,1)];
    cpts = Umat * M * P;
    draw_line3(cpts(:,1), cpts(:,2), cpts(:,3), color("blue"), 3, 1);

    // ── De Casteljau ─────────────────────────────────────
    Q = zeros(4, 3);
    for j = 1:4; Q(j,:) = lerp(P(j,:), P(j+1,:), u); end

    R = zeros(3, 3);
    for j = 1:3; R(j,:) = lerp(Q(j,:), Q(j+1,:), u); end

    S = zeros(2, 3);
    for j = 1:2; S(j,:) = lerp(R(j,:), R(j+1,:), u); end

    T = lerp(S(1,:), S(2,:), u);

    // Level 1 Q — amber
    draw_line3(Q(:,1), Q(:,2), Q(:,3), color("darkorange"), 2, 1);
    for j = 1:4
        draw_dot3(Q(j,1), Q(j,2), Q(j,3), color("darkorange"), 5, 9);
    end

    // Level 2 R — pink
    draw_line3(R(:,1), R(:,2), R(:,3), color("hotpink"), 2, 1);
    for j = 1:3
        draw_dot3(R(j,1), R(j,2), R(j,3), color("hotpink"), 5, 9);
    end

    // Level 3 S — violet
    draw_line3(S(:,1), S(:,2), S(:,3), color("darkviolet"), 2, 1);
    for j = 1:2
        draw_dot3(S(j,1), S(j,2), S(j,3), color("darkviolet"), 5, 9);
    end

    // Level 4 T — green diamond
    draw_dot3(T(1), T(2), T(3), color("darkgreen"), 10, 6);
    t0_lbl = "T0(" + string(round(T(1)*1000)/1000) + "," ..
                   + string(round(T(2)*1000)/1000) + "," ..
                   + string(round(T(3)*1000)/1000) + ")";
    draw_label(T(1)+0.12, T(2)+0.12, t0_lbl, color("darkgreen"));

endfunction

// -- 4. Callbacks --------------------------------------------

function sync_all()
    global P u h_slider h_pts;
    for row = 1:5
        for col = 1:3
            P(row,col) = evstr(get(h_pts((row-1)*3+col), "string"));
        end
    end
    u = get(h_slider, "value");
    update_plot();
endfunction

function reset_data()
    global P h_slider h_pts;
    P = [0, 0, 1; 0, 4, 1; 4, 0, 1; 4, 4, 1; 5, 4, 1];
    for row = 1:5
        for col = 1:3
            set(h_pts((row-1)*3+col), "string", string(P(row,col)));
        end
    end
    set(h_slider, "value", 0.5);
    sync_all();
endfunction

// -- 5. Figure and UI ----------------------------------------

f = figure( ..
    "figure_name", "Quartic 3D Bezier", ..
    "position",    [50, 50, 1020, 700] ..
);

// Plot axes — right portion, leaves left panel free
h_axes = newaxes();
h_axes.axes_bounds = [0.22, 0.06, 0.76, 0.90];

// ── One-time axes setup ────────────────────────────────────
sca(h_axes);
h_axes.view            = "3d";
h_axes.rotation_angles = [65, 45];
h_axes.data_bounds     = [-0.5, -0.5, 0; 6, 5, 2];
h_axes.axes_visible    = "on";
h_axes.grid            = [1, 1];
h_axes.title.font_size   = 3;
h_axes.x_label.text      = "X";
h_axes.x_label.font_size = 2;
h_axes.y_label.text      = "Y";
h_axes.y_label.font_size = 2;
h_axes.z_label.text      = "Z";
h_axes.z_label.font_size = 2;

// ── Column headers ─────────────────────────────────────────
uicontrol(f, "style","text","string","Pt", ..
    "position",[10,640,28,20],"fontweight","bold");
uicontrol(f, "style","text","string","X", ..
    "position",[42,640,44,20],"fontweight","bold");
uicontrol(f, "style","text","string","Y", ..
    "position",[90,640,44,20],"fontweight","bold");
uicontrol(f, "style","text","string","Z", ..
    "position",[138,640,44,20],"fontweight","bold");

// ── P0 .. P4 input rows ────────────────────────────────────
pt_names = ["P0","P1","P2","P3","P4"];
for i = 1:5
    yp = 640 - (i * 38);
    uicontrol(f, "style","text","string",pt_names(i), ..
        "position",[10,yp,28,22]);
    for j = 1:3
        h = uicontrol(f, "style","edit", ..
            "string",   string(P(i,j)), ..
            "position", [42+(j-1)*48, yp, 44, 22], ..
            "callback", "sync_all()");
        h_pts($+1) = h;
    end
end

// ── u slider ──────────────────────────────────────────────
uicontrol(f, "style","text","string","Parameter u", ..
    "position",[10,90,140,20]);
h_slider = uicontrol(f, "style","slider", ..
    "min",0,"max",1,"value",0.5, ..
    "position",[10,65,185,22], ..
    "callback","sync_all()");

// ── Reset button ──────────────────────────────────────────
uicontrol(f, "style","pushbutton","string","Reset", ..
    "position",[10,25,185,30], ..
    "callback","reset_data()");

// ── Initial draw ──────────────────────────────────────────
update_plot();
