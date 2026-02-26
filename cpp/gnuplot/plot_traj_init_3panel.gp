# 3-panel trajectory plot (x2,y2) for nearlinear/intermediate/chaotic

set datafile separator comma

# Output
set terminal pngcairo size 1800,600 font "Sans,12"
set output "trajectories_3cases.png"

# Layout: 1 row, 3 columns
set multiplot layout 1,3 title "Double pendulum tip trajectory (x2,y2)" font ",14"

# Common styling
set grid
set key off
set tics out
set size ratio -1

# Filenames
f1 = "case_nearlinear.csv"
f2 = "case_intermediate.csv"
f3 = "case_chaotic.csv"

# --- Panel 1 ---
set title "Near-linear"
set xlabel "x2"
set ylabel "y2"
set xrange [-2:2]
set yrange [-2:2]
plot f1 every ::1 using 6:7 with lines lw 1 lc rgb "#1f77b4"

# --- Panel 2 ---
set title "Intermediate"
set xlabel "x2"
unset ylabel
set xrange [-2:2]
set yrange [-2:2]
plot f2 every ::1 using 6:7 with lines lw 1 lc rgb "#ff7f0e"

# --- Panel 3 ---
set title "Chaotic"
set xlabel "x2"
set xrange [-2:2]
set yrange [-2:2]
plot f3 every ::1 using 6:7 with lines lw 1 lc rgb "#d62728"

unset multiplot
set output
