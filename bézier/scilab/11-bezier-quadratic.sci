// 1. Define Control Points (P0, P1, P2)
// Each row is a point [x, y]
P = [1, 1; 
     3, 3; 
     5, 1];

// 2. Define the Quadratic Bézier Basis Matrix (M)
// Derived from: (1-u)^2, 2u(1-u), u^2
M = [ 1, -2,  1;
     -2,  2,  0;
      1,  0,  0];

// 3. Generate u values from 0 to 1
u_vals = 0:0.01:1;
curve_pts = [];

// 4. Calculate curve points using Matrix Multiplication
for i = 1:length(u_vals)
    u = u_vals(i);
    U = [u^2, u, 1];
    // Matrix Formula: P(u) = U * M * P
    point = U * M * P;
    curve_pts = [curve_pts; point];
end

// 5. Plotting
clf(); // Clear current figure
plot(P(:,1), P(:,2), 'r--o'); // Plot Control Polygon (red dashed with circles)
plot(curve_pts(:,1), curve_pts(:,2), 'b-', 'LineWidth', 2); // Plot Bézier Curve (blue)

// 6. Aesthetics
xtitle("Quadratic Bezier Curve (Scilab)", "X-axis", "Y-axis");
legend(["Control Polygon", "Bezier Curve"]);
xgrid();
