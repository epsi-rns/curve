funcprot(0); // Suppress redefinition warnings

// -- 1. Data Setup -----------------------------------------------------------
P = [0, 0, 1;   // P0
     0, 4, 1;   // P1
     4, 0, 1;   // P2
     4, 4, 1;   // P3
     5, 4, 1];  // P4

u_snapshots = [0.25, 0.50, 0.75];
// Using standard Scilab color indices: 5=Red/Pink, 25=Orange, 13=Purple
colors = [5, 25, 13]; 

function pt = lerp(a, b, v)
    pt = (1 - v) * a + v * b;
endfunction

// -- 2. Main Plotting --------------------------------------------------------
clf();
ax = gca();
ax.view = "3d";  // Fixed: Use "3d" instead of "free"
ax.data_bounds = [-0.5, -0.5, 0; 6, 5, 2];
xtitle("Quartic De Casteljau 3D", "X", "Y", "Z");

// Draw the static Bezier Curve (using your derived Matrix)
M = [1 -4 6 -4 1; -4 12 -12 4 0; 6 -12 6 0 0; -4 4 0 0 0; 1 0 0 0 0];
u_fine = linspace(0, 1, 100);
curve_pts = [];
for t = u_fine
    curve_pts = [curve_pts; [t^4, t^3, t^2, t, 1] * M * P];
end
param3d(curve_pts(:,1), curve_pts(:,2), curve_pts(:,3));
e = gce(); e.foreground = 2; e.thickness = 3; // Blue Curve

// Draw original Control Polygon
param3d(P(:,1), P(:,2), P(:,3));
e = gce(); e.line_style = 3; e.mark_style = 9; // Dashed black with circles

// -- 3. De Casteljau Levels -------------------------------------------------
for i = 1:length(u_snapshots)
    u = u_snapshots(i);
    col = colors(i);
    
    // Level 1: reduction to 4 points (Q)
    Q = zeros(4, 3);
    for j = 1:4; Q(j,:) = lerp(P(j,:), P(j+1,:), u); end
    param3d(Q(:,1), Q(:,2), Q(:,3)); 
    e = gce(); e.foreground = col; e.line_style = 2; // Dashed snapshot color
    
    // Level 2: reduction to 3 points (R)
    R = zeros(3, 3);
    for j = 1:3; R(j,:) = lerp(Q(j,:), Q(j+1,:), u); end
    param3d(R(:,1), R(:,2), R(:,3)); 
    e = gce(); e.foreground = col; e.thickness = 1;
    
    // Level 3: reduction to 2 points (S)
    S = zeros(2, 3);
    for j = 1:2; S(j,:) = lerp(R(j,:), R(j+1,:), u); end
    param3d(S(:,1), S(:,2), S(:,3)); 
    e = gce(); e.foreground = col; e.thickness = 2;
    
    // Level 4: Final Curve Point (T)
    T = lerp(S(1,:), S(2,:), u);
    param3d(T(1), T(2), T(3)); 
    e = gce(); 
    e.mark_style = 10;          // Diamond marker
    e.mark_foreground = col; 
    e.mark_size = 5;
end

xgrid();
