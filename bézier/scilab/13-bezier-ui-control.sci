// ============================================================
// Bezier Quadratic -- Interactive UI
// Scilab script
// Run with: exec('03-ui-control.sci', -1)
// ============================================================
funcprot(0);

// -- 1. Global Data ------------------------------------------
global P u;
global h_p0x h_p0y h_p1x h_p1y h_p2x h_p2y;
global h_slider h_axes;

P = [1, 1; 3, 3; 5, 1];
u = 0.5;

// -- 2. Helpers ----------------------------------------------

function pt = lerp(a, b, t)
    pt = (1 - t) .* a + t .* b;
endfunction

function clear_axes(ax)
    // Safely delete all children of an axes handle
    while size(ax.children, "*") > 0
        delete(ax.children(1));
    end
endfunction

// -- 3. Plot -------------------------------------------------

// Helper: draw a styled polyline, returns the Polyline handle.
// Uses plot2d which reliably returns a Compound; we unwrap it.
function h = draw_line(xs, ys, col, thick, lstyle)
    plot2d(xs, ys, style=col);
    c = gce();                  // Compound
    h = c.children(1);         // Polyline inside
    h.line_mode  = "on";
    h.foreground = col;
    h.thickness  = thick;
    h.line_style = lstyle;      // 1=solid 2=dashed 3=dotted
endfunction

// Helper: draw a single marker point, returns the Polyline handle.
// Scatter plot of one point via plot2d with negative style.
// Unwraps Compound -> Polyline for reliable property access.
function h = draw_dot(px, py, col, msz, mstyle)
    plot2d(px, py, style=-mstyle);
    c = gce();                  // Compound
    h = c.children(1);         // Polyline inside
    h.mark_foreground = col;
    h.mark_background = col;
    h.mark_size       = msz;
endfunction

function update_plot()
    global P u h_axes;

    sca(h_axes);
    clear_axes(h_axes);

    // Axes appearance
    h_axes.data_bounds  = [0, 0; 6, 4];
    h_axes.axes_visible = "on";
    h_axes.x_label.text = "x";
    h_axes.y_label.text = "y";
    h_axes.title.text   = "Bezier Quadratic   u = " + string(u);
    h_axes.title.font_size   = 3;
    h_axes.x_label.font_size = 2;
    h_axes.y_label.font_size = 2;
    h_axes.grid = [1, 1];

    // ── Control polygon ──────────────────────────────────
    draw_line(P(:,1), P(:,2), color("red"), 1, 2);

    // ── Control point markers + labels ───────────────────
    for i = 1:3
        draw_dot(P(i,1), P(i,2), color("red"), 8, 9);
        lbl = "P" + string(i-1) ..
            + "(" + string(P(i,1)) ..
            + "," + string(P(i,2)) + ")";
        xstring(P(i,1) + 0.1, P(i,2) + 0.1, lbl);
        tx = gce();
        tx.font_size  = 2;
        tx.foreground = color("red");
    end

    // ── Bezier curve via matrix form ─────────────────────
    M     = [1, -2, 1; -2, 2, 0; 1, 0, 0];
    n     = 200;
    tv    = linspace(0, 1, n)';
    cpts  = [tv.^2, tv, ones(n,1)] * M * P;
    draw_line(cpts(:,1), cpts(:,2), color("blue"), 3, 1);

    // ── De Casteljau ─────────────────────────────────────
    Q0 = lerp(P(1,:), P(2,:), u);
    Q1 = lerp(P(2,:), P(3,:), u);
    R0 = lerp(Q0, Q1, u);

    // Level-1 segment Q0-Q1
    draw_line([Q0(1); Q1(1)], [Q0(2); Q1(2)], ..
              color("darkorange"), 2, 1);

    // Q0 marker + label
    draw_dot(Q0(1), Q0(2), color("darkorange"), 8, 9);
    xstring(Q0(1) - 0.35, Q0(2) + 0.1, "Q0");
    tx = gce(); tx.font_size = 2;
    tx.foreground = color("darkorange");

    // Q1 marker + label
    draw_dot(Q1(1), Q1(2), color("darkorange"), 8, 9);
    xstring(Q1(1) + 0.1, Q1(2) + 0.1, "Q1");
    tx = gce(); tx.font_size = 2;
    tx.foreground = color("darkorange");

    // R0 diamond marker + label
    draw_dot(R0(1), R0(2), color("darkgreen"), 12, 6);
    r0x    = round(R0(1) * 1000) / 1000;
    r0y    = round(R0(2) * 1000) / 1000;
    r0_lbl = "R0(" + string(r0x) + "," + string(r0y) + ")";
    xstring(R0(1) + 0.12, R0(2) + 0.12, r0_lbl);
    tx = gce(); tx.font_size = 2;
    tx.foreground = color("darkgreen");

endfunction

// -- 4. Callbacks --------------------------------------------

function sync_data()
    global P u;
    global h_p0x h_p0y h_p1x h_p1y h_p2x h_p2y h_slider;

    P(1,1) = evstr(get(h_p0x, "string"));
    P(1,2) = evstr(get(h_p0y, "string"));
    P(2,1) = evstr(get(h_p1x, "string"));
    P(2,2) = evstr(get(h_p1y, "string"));
    P(3,1) = evstr(get(h_p2x, "string"));
    P(3,2) = evstr(get(h_p2y, "string"));
    u      = get(h_slider, "value");

    update_plot();
endfunction

function reset_data()
    global h_p0x h_p0y h_p1x h_p1y h_p2x h_p2y h_slider;

    set(h_p0x,   "string", "1");
    set(h_p0y,   "string", "1");
    set(h_p1x,   "string", "3");
    set(h_p1y,   "string", "3");
    set(h_p2x,   "string", "5");
    set(h_p2y,   "string", "1");
    set(h_slider, "value", 0.5);

    sync_data();
endfunction

// -- 5. Figure and UI ----------------------------------------

f = figure( ..
    "figure_name", "Bezier Quadratic", ..
    "position",    [80, 80, 920, 640] ..
);

// Plot axes: right portion of the figure
// [left, bottom, width, height] as fractions 0..1
h_axes = newaxes();
h_axes.axes_bounds = [0.22, 0.06, 0.76, 0.90];

// ── u slider ──────────────────────────────────────────────
uicontrol(f, ..
    "style",    "text", ..
    "string",   "Parameter u", ..
    "position", [10, 90, 120, 20]);

h_slider = uicontrol(f, ..
    "style",    "slider", ..
    "min",      0, ..
    "max",      1, ..
    "value",    0.5, ..
    "position", [10, 65, 175, 22], ..
    "callback", "sync_data()");

// ── P0 ────────────────────────────────────────────────────
uicontrol(f, ..
    "style",    "text", ..
    "string",   "P0", ..
    "position", [10, 560, 30, 20]);
uicontrol(f, ..
    "style",    "text", ..
    "string",   "x:", ..
    "position", [44, 560, 18, 20]);
h_p0x = uicontrol(f, ..
    "style",    "edit", ..
    "string",   "1", ..
    "position", [63, 560, 48, 22], ..
    "callback", "sync_data()");
uicontrol(f, ..
    "style",    "text", ..
    "string",   "y:", ..
    "position", [115, 560, 18, 20]);
h_p0y = uicontrol(f, ..
    "style",    "edit", ..
    "string",   "1", ..
    "position", [134, 560, 48, 22], ..
    "callback", "sync_data()");

// ── P1 ────────────────────────────────────────────────────
uicontrol(f, ..
    "style",    "text", ..
    "string",   "P1", ..
    "position", [10, 525, 30, 20]);
uicontrol(f, ..
    "style",    "text", ..
    "string",   "x:", ..
    "position", [44, 525, 18, 20]);
h_p1x = uicontrol(f, ..
    "style",    "edit", ..
    "string",   "3", ..
    "position", [63, 525, 48, 22], ..
    "callback", "sync_data()");
uicontrol(f, ..
    "style",    "text", ..
    "string",   "y:", ..
    "position", [115, 525, 18, 20]);
h_p1y = uicontrol(f, ..
    "style",    "edit", ..
    "string",   "3", ..
    "position", [134, 525, 48, 22], ..
    "callback", "sync_data()");

// ── P2 ────────────────────────────────────────────────────
uicontrol(f, ..
    "style",    "text", ..
    "string",   "P2", ..
    "position", [10, 490, 30, 20]);
uicontrol(f, ..
    "style",    "text", ..
    "string",   "x:", ..
    "position", [44, 490, 18, 20]);
h_p2x = uicontrol(f, ..
    "style",    "edit", ..
    "string",   "5", ..
    "position", [63, 490, 48, 22], ..
    "callback", "sync_data()");
uicontrol(f, ..
    "style",    "text", ..
    "string",   "y:", ..
    "position", [115, 490, 18, 20]);
h_p2y = uicontrol(f, ..
    "style",    "edit", ..
    "string",   "1", ..
    "position", [134, 490, 48, 22], ..
    "callback", "sync_data()");

// ── Reset button ──────────────────────────────────────────
uicontrol(f, ..
    "style",    "pushbutton", ..
    "string",   "Reset", ..
    "position", [10, 440, 175, 30], ..
    "callback", "reset_data()");

// ── Initial draw ──────────────────────────────────────────
update_plot();
