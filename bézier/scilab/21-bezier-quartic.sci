funcprot(0); // Suppress function redefinition warnings

// -- 1. Data Setup -----------------------------------------------------------
// Control points from the second homework
P = [0, 0, 1;   // P0
     0, 4, 1;   // P1
     4, 0, 1;   // P2
     4, 4, 1;   // P3
     5, 4, 1];  // P4

// -- 2. Quartic Bezier Matrix (M) -------------------------------------------
// Derived from the binomial expansion of (1-u)^4
M = [ 1, -4,  6, -4,  1;
     -4, 12, -12,  4,  0;
      6, -12,  6,  0,  0;
     -4,  4,  0,  0,  0;
      1,  0,  0,  0,  0];

// -- 3. Computation ---------------------------------------------------------
u_vals = linspace(0, 1, 100);
curve_pts = zeros(length(u_vals), 3);

for i = 1:length(u_vals)
    u = u_vals(i);
    // Power basis vector for degree 4
    U = [u^4, u^3, u^2, u, 1];
    // Matrix Formula: P(u) = U * M * P
    curve_pts(i, :) = U * M * P;
end

// -- 4. Plotting ------------------------------------------------------------
clf();
// Plot Control Polygon (dashed black line with circles)
param3d(P(:,1), P(:,2), P(:,3));
h_poly = gce(); 
h_poly.line_style = 3; // Dashed
h_poly.mark_style = 9; // Circle markers
h_poly.foreground = 1; // Black

// Plot Bezier Curve (thick blue line)
param3d(curve_pts(:,1), curve_pts(:,2), curve_pts(:,3));
h_curve = gce();
h_curve.thickness = 3;
h_curve.foreground = 2; // Blue

// -- 5. Decorations & Interactivity -----------------------------------------
xtitle("Quartic Bezier Curve 3D", "X axis", "Y axis", "Z axis");
show_window(); // Bring window to front

// Set 3D view limits to match your Python version
ax = gca();
ax.data_bounds = [0, 0, 0; 6, 5, 2]; 
ax.view_type = "free"; // Enables rotation
