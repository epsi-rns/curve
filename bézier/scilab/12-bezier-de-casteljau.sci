// -- 1. Data Setup -----------------------------------------------------------
P0 = [1, 1]; P1 = [3, 3]; P2 = [5, 1];
u_snapshots = [0.25, 0.50, 0.75];
colors = ["#E91E63", "#FF6F00", "#6A1B9A"]; // Hex strings work in modern Scilab

// -- 2. Functions ------------------------------------------------------------
function pt = lerp(a, b, u)
    pt = (1 - u) * a + u * b;
endfunction

// -- 3. Computation & Plotting -----------------------------------------------
clf();
drawlater(); // Start buffer to prevent flickering
ax = gca();
ax.grid_style = [1 1]; // Set whitegrid style
ax.data_bounds = [0.4, 0.4; 5.8, 3.6]; // Set axis limits

// Draw the smooth Bezier Curve
u_fine = linspace(0, 1, 200);
M = [1, -2, 1; -2, 2, 0; 1, 0, 0];
P_mat = [P0; P1; P2];
curve_pts = [];
for u = u_fine
    curve_pts = [curve_pts; [u^2, u, 1] * M * P_mat];
end
plot(curve_pts(:,1), curve_pts(:,2), "color", "#1565C0", "thickness", 3);

// Draw Control Polygon
plot([P0(1) P1(1) P2(1)], [P0(2) P1(2) P2(2)], "k--o", "markerSize", 2);

// Labels for Control Points
xstring(P0(1)-0.4, P0(2)-0.2, "P0(1,1)");
xstring(P1(1)+0.1, P1(2)+0.1, "P1(3,3)");
xstring(P2(1)+0.1, P2(2)-0.2, "P2(5,1)");

// -- 4. De Casteljau Snapshots & Annotations ---------------------------------
for i = 1:length(u_snapshots)
    u = u_snapshots(i);
    c = colors(i);
    
    Q0 = lerp(P0, P1, u);
    Q1 = lerp(P1, P2, u);
    R0 = lerp(Q0, Q1, u);
    
    // Draw Level 1 Segment (Q0-Q1)
    plot([Q0(1) Q1(1)], [Q0(2) Q1(2)], "color", c, "thickness", 2);
    
    // Draw Q0, Q1 Dots
    plot([Q0(1) Q1(1)], [Q0(2) Q1(2)], "color", c, "marker", "o", "markerSize", 5);
    
    // Draw R0 Diamond (The Curve Point)
    plot(R0(1), R0(2), "color", c, "marker", "d", "markerSize", 10, "markerBackground", c);
    
    // Coordinate Annotations (Simplified)
    xstring(R0(1)+0.1, R0(2)-0.2, "R0(" + string(u) + "): (" + string(R0(1)) + "," + string(R0(2)) + ")");
end

// -- 5. Decoration -----------------------------------------------------------
xtitle("De Casteljau Algorithm — Quadratic Bezier", "x", "y");
hl = legend(["Bezier Curve", "Control Polygon", "u=0.25", "u=0.50", "u=0.75"]);
drawnow(); // Render the complete plot
